"""
eo_brain/agent/brain.py - Agentic interface (LLM function calling).

Manual function calling loop instead of SDK-side automatic dispatch:
the google-genai SDK's automatic function calling silently fails on
certain function signatures, giving us zero visibility into errors.
Manual mode gives full control over dispatch, logging, and error handling.

The loop talks to the model only through ``eo_brain.agent.llm.LLMProvider``;
the Gemini SDK lives behind ``eo_brain.agent.providers.gemini``. Everything
with side effects (provider, geocoder, job repository, session store,
clock, sleep) is injected into ``Agent``, so the loop runs in tests and
evals with no network, database or API key.

Division of labour:
  * the model decides WHICH tool to call and with WHICH raw arguments,
    including the period: days_back, or start_date and end_date;
  * code decides everything else - alias resolution, range and date checks,
    the window (from one clock reading per request, the same reading that
    tells the model today's date), geocoding, bbox/resolution rules, job
    creation, session context, and a reply that states the period that ran.

Rejections come in two kinds (validators.ValidationError / NeedsUserError).
After a job-creating call is rejected for a reason only the user can resolve
(window too long, area outside the limits, no previous area, ...), no further
job-creating call executes in the same request: it gets a deterministic
refusal instead, so a model cannot turn "2 years" into a 365-day job by
retrying. Model-fixable rejections (unknown alias, number format, a place the
geocoder did not find) still allow a corrected retry.

``query_database`` hands a question to the data agent (data_agent.py), a
separate text-to-SQL sub-agent with its own prompt, context, database role
and a deterministic SQL gate. This loop only sees its bounded result.
"""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence
from uuid import UUID

from eo_brain.agent.geocoder import (
    GeocodingResult,
    PlaceCandidate,
    PlaceChoiceNeeded,
    geocode_or_fail,
    normalize as normalize_place,
)
from eo_brain.agent.llm import (
    LLMProvider,
    Message,
    RateLimitError,
    ToolResult,
    ToolResults,
    ToolSpec,
    UserMessage,
)
from eo_brain.agent.session import SessionStore, session_store
from eo_brain.agent.validators import (
    Period,
    ValidationError,
    as_utc,
    date_given,
    estimate_area_km2,
    resolve_algorithm,
    resolve_period,
    resolve_resolution,
    validate_algorithm_compatibility,
    validate_bbox,
    validate_location,
)
from eo_brain.algorithms.base import ALGORITHMS as ALGO_REGISTRY

logger = logging.getLogger(__name__)

_MAX_INPUT_CHARS = 2000
_MAX_RETRIES = 3
_BACKOFF_BASE_SEC = 5
_MAX_TOOL_ROUNDS = 5  # safety cap on tool call loops
# Place lookups per chat request: each geocode_or_fail call makes at most
# geocoder.MAX_LOOKUPS_PER_QUERY (4) Nominatim searches, so at most 8 per request.
_MAX_GEOCODES_PER_REQUEST = 2
_DEFAULT_DAYS_BACK = 30
_PROVIDER_NAME = "element84-s2"

_MAX_ROUNDS_TEXT = "Analysis request is being processed."

# Tools that create a job. Only these are blocked after a needs-user rejection.
JOB_TOOLS = frozenset({"launch_analysis", "rerun_with_different_algorithm"})
_NEEDS_USER_ACTION = (
    "Only the user can resolve this. Tell the user this limit and ask how they want to "
    "proceed. Do not call launch_analysis or rerun_with_different_algorithm again in this "
    "turn with other values, and do not start a different analysis instead."
)
_BLOCKED_ACTION = (
    "NOT EXECUTED: an earlier call in this request was rejected for a reason only the user "
    "can resolve (the error above), so no further analysis can be started in this turn. "
    + _NEEDS_USER_ACTION
)
_NO_RESPONSE_TEXT = "No response from AI model."
_UNAVAILABLE_TEXT = "AI service temporarily unavailable. Please retry in a minute."
_NOT_CONFIGURED_TEXT = "AI agent is not configured. Set GEMINI_API_KEY."


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sanitize_input(text: str) -> str:
    """Drop control characters (keeps tab/newline/CR) and cap the length."""
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return cleaned[:_MAX_INPUT_CHARS]


def _rejection(message: str, *, needs_user: bool) -> Dict[str, Any]:
    """Tool result for a rejected call. ``needs_user`` marks the kind only the user can resolve."""
    if not needs_user:
        return {"error": message}
    return {"error": message, "needs_user": True, "action_required": _NEEDS_USER_ACTION}


def _choice_rejection(e: PlaceChoiceNeeded) -> Dict[str, Any]:
    """Tool result when only the user can choose the place: the candidates, and how to pass the answer."""
    return {
        "error": str(e),
        "needs_user": True,
        "candidates": [{"location": c.label, "kind": c.kind, "in": c.where} for c in e.candidates],
        "action_required": (
            "Only the user can choose. List every candidate with its kind and where it is, and ask "
            "which one is meant. Do not pick one yourself, and do not call launch_analysis or "
            "rerun_with_different_algorithm again in this turn. When the user answers, call "
            "launch_analysis with location set to the chosen candidate's 'location' text exactly."
        ),
    }


# -- Tool declarations (provider-neutral JSON Schema) --

TOOL_SPECS: List[ToolSpec] = [
    ToolSpec(
        name="launch_analysis",
        description=(
            "Geocode a location and create a satellite analysis job. "
            "Call this whenever the user asks to analyze an area with one of the "
            "supported analyses (see algorithm). Pass what the user asked for; the "
            "tool checks the limits and rejects what is outside them."
        ),
        parameters={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "The place as the user named it, with any qualifiers separated by commas "
                        "(place, province or region, country), e.g. 'Casentino, Arezzo, Italia', "
                        "'Florence, Italy', 'vallata del Casentino'. Keep words such as 'valle' or "
                        "'provincia': they say what kind of place is meant. If the user is choosing "
                        "among candidates an earlier reply listed, pass the chosen candidate's "
                        "location text exactly."
                    ),
                },
                "algorithm": {
                    "type": "string",
                    "description": (
                        "Analysis type: ndvi, ndwi, mndwi, flood, lisa. "
                        "Also accepts: vegetation, water, drought, anomaly. "
                        "These are the only analyses: if the user asks for something "
                        "else (traffic, weather, air quality, ...), do not call this tool."
                    ),
                },
                "days_back": {
                    "type": "integer",
                    "description": (
                        "A look-back window in days ending today, for 'the last N days, weeks, "
                        "months or years', exactly as the user asked ('last week' = 7, "
                        "'last 3 months' = 90, 'last year' = 365, 'last 2 years' = 730). "
                        "Never turn a calendar period (named dates, months, seasons or years) "
                        "into days_back: a look-back window ends today, so it would analyse a "
                        "different period; pass start_date and end_date instead. Pass neither "
                        "days_back nor dates when the user gives no period (default: the last "
                        "30 days). Never shorten or shift a period to fit a limit: the tool "
                        "checks the limits."
                    ),
                },
                "start_date": {
                    "type": "string",
                    "description": (
                        "First day of the period, YYYY-MM-DD, inclusive (UTC), when the user "
                        "names calendar dates, months, seasons or years: 'August to September "
                        "2026' = 2026-08-01 to 2026-09-30; 'on 3 November 2026' = 2026-11-03 "
                        "to 2026-11-03. Always together with end_date, never with days_back."
                    ),
                },
                "end_date": {
                    "type": "string",
                    "description": (
                        "Last day of the period, YYYY-MM-DD, inclusive (UTC). Always together "
                        "with start_date. For a period that runs up to now ('since 1 October', "
                        "'from X to today'), end_date is today's date."
                    ),
                },
            },
            "required": ["location", "algorithm"],
        },
    ),
    ToolSpec(
        name="get_job_status",
        description=(
            "Status and result facts of one analysis job: progress, its message and, once "
            "completed, its coverage (share of the requested area with cloud-free data, why "
            "the rest has none, scenes and acquisition dates used). Use it for 'is it done' "
            "and for 'why does the map look like this' about an analysis of this conversation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "job_id": {
                    "type": "string",
                    "description": "The job UUID. Omit it for the latest analysis launched in this conversation.",
                },
            },
        },
    ),
    ToolSpec(
        name="list_capabilities",
        description="List all available satellite analysis algorithms.",
        parameters={"type": "object", "properties": {}},
    ),
    ToolSpec(
        name="rerun_with_different_algorithm",
        description=(
            "Re-run analysis on the same area with a different algorithm. "
            "Use when user says 'same area', 'now do water', 'repeat with flood'."
        ),
        parameters={
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "description": "New algorithm. Leave empty to reuse previous.",
                },
            },
        },
    ),
    ToolSpec(
        name="query_database",
        description=(
            "Answer a question about analyses that ALREADY exist (past jobs, their status, "
            "alert areas, hectares by severity, overlaps, comparisons) with a read-only "
            "database query. Pass the user's question, self-contained, in the user's "
            "language. Never use it to start a new analysis."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The question to answer from stored results.",
                },
            },
            "required": ["question"],
        },
    ),
]

SYSTEM_PROMPT = """\
You are EO Brain, a satellite intelligence assistant. You help users analyze \
geographical areas using real satellite data (Sentinel-2, Landsat).

CAPABILITIES:
- NDVI: Vegetation health index (green = healthy, brown = stressed)
- NDWI: Water detection using near-infrared
- MNDWI: Enhanced water detection using shortwave infrared
- LISA: Spatial anomaly detection (outlier clusters in vegetation)
- Flood: Flood extent mapping with severity classification
- Stored results: answer questions about analyses already run (status, alert areas, \
  hectares per severity, overlaps, comparisons) with query_database

RULES:
1. When the user asks to analyze an area with one of the analyses above, ALWAYS call \
   launch_analysis immediately. Never describe what you would do. Never ask for confirmation first.
2. If the user references the previous area ("same zone", "now water", "repeat"), \
   call rerun_with_different_algorithm.
3. If a tool returns an error, relay the exact error message to the user as a helpful suggestion. \
   If the result has needs_user, only the user can resolve it: relay the limit and ask the user \
   how to proceed. Do not retry with other values (a shorter window, another place, another analysis). \
   If it lists candidate places, show every candidate with its kind and where it is, and ask which \
   one is meant; never choose one yourself.
4. NEVER fabricate job IDs, URLs, or analysis results. Only report data returned by tools.
5. NEVER run analyses the user did not request. If the user asks for something the analyses above \
   do not measure (traffic, weather, air quality, population, ...), say it is not supported and list \
   what is supported; do not run a supported analysis in its place.
6. After a successful launch, briefly confirm what was launched, the period as the start and end \
   dates from the tool result (date_range), never only as a number of days, and the estimated time.
7. You understand Italian and English. Respond in the same language the user writes in.
8. Default to Italian when the language is ambiguous (short commands, place names only).
9. When parsing compact input like "ndvi firenze", treat the first token as the algorithm \
   and the rest as the location. Call launch_analysis immediately.
10. The period of an analysis is given in one of two ways. Always interpret temporal references. \
   For "the last N days/weeks/months/years" pass days_back, a look-back window ending today: \
   "last week" = 7, "last 3 months" = 90, "last year" = 365, "last 2 years" = 730, \
   "last 5 years" = 1825. When the user names calendar dates, months, seasons or years \
   ("August to September 2026", "in March", "since 1 October", "on 3 November 2026"), pass \
   start_date and end_date (YYYY-MM-DD, both inclusive) and no days_back. Never convert a \
   calendar period into days_back: a look-back window ends today, so it would analyse a \
   different period. A month or date given without a year means its most recent occurrence \
   that is not in the future (see TODAY below); for a month still in progress, or a period \
   "to today", end_date is today's date. With no period at all, pass neither (default: the \
   last 30 days). Pass the period the user asked for even when it looks long; never shorten or \
   shift it yourself. If the tool rejects it with needs_user, tell the user the limit and ask \
   which period they want.
11. Route by intent: a request to analyze an area now -> launch_analysis. A question about \
   analyses that already exist (how many, how much area, which ones, compare, status history) \
   -> query_database with the question. Never launch an analysis to answer such a question.
12. Only state numbers that appear in tool results (for query_database: in the returned rows). \
   If query_database returns status "refused" or "failed", say so plainly; do not guess.
13. Questions about what an analysis shows or why it looks the way it does (only part of the \
   area covered, a strip, gaps, odd values) are answered only from tool results: call \
   get_job_status (no job_id = the latest analysis of this conversation) or, for stored \
   analyses, query_database, and explain with what they return (coverage, its reason, scenes, \
   dates). If the results do not explain it, say the data does not tell; never offer causes \
   they do not state (satellite orbits, terrain, sensor faults, processing errors).
"""


# The coverage facts get_job_status passes to the model (core.coverage writes them).
_COVERAGE_FACTS = (
    "coverage_pct", "low_coverage", "low_coverage_below_pct",
    "uncovered_masked_pct", "uncovered_no_data_pct",
    "scenes_found", "scenes_too_cloudy", "scenes_checked", "scenes_with_clear_pixels", "scenes_used",
    "acquisition_days_with_data", "first_date_with_data", "last_date_with_data",
)


def _coverage_facts(coverage: Any) -> Dict[str, Any]:
    if not isinstance(coverage, dict):
        return {"note": "coverage was not recorded for this analysis (it ran before coverage reporting)"}
    return {k: coverage.get(k) for k in _COVERAGE_FACTS if k in coverage}


def _window(start: Any, end: Any) -> Optional[str]:
    if not start or not end:
        return None
    return f"{str(start)[:10]} to {str(end)[:10]}"


def today_line(now: datetime) -> str:
    """Appended to the system prompt of each request: the model has no clock of its own."""
    return (
        f"TODAY: {as_utc(now).date()} (UTC). Use it to resolve 'today', 'yesterday', "
        "'this month' and dates given without a year."
    )


def _period_args(period: Period) -> Dict[str, Any]:
    """The period in the form it was requested, as launch_analysis arguments."""
    if period.days_back is not None:
        return {"days_back": period.days_back, "start_date": None, "end_date": None}
    return {"days_back": None, "start_date": str(period.start_date), "end_date": str(period.end_date)}


def _with_period(text: str, launch: Dict[str, Any]) -> str:
    """
    The reply states the period that ran. A first call with days_back=60 for
    "August to September" looks like a request for the last 60 days, so code
    cannot catch that substitution; it can make sure it is not silent.
    """
    start, end = launch.get("start_date"), launch.get("end_date")
    if not start or not end or (start in text and end in text):
        return text
    sentence = f"Period analysed: {start} to {end}."
    return f"{text}\n\n{sentence}" if text else sentence


# -- Per-request bookkeeping --

@dataclass
class ToolInvocation:
    """One executed tool call, as the model requested it and as code answered."""

    name: str
    args: Dict[str, Any]
    result: Dict[str, Any]


@dataclass
class _RequestContext:
    """
    State of one chat() call. Tool handlers append to it; replaces the old
    thread-local side channel for created job ids.
    """

    session_id: str
    # The one clock reading of this request: today's date for the model,
    # date validation and every window (set by run(), else on first use).
    now: Optional[datetime] = None
    created_jobs: List[str] = field(default_factory=list)
    last_launch: Optional[Dict[str, Any]] = None
    last_query: Optional[Any] = None  # DataAgentResult of the last successful query
    trace: List[ToolInvocation] = field(default_factory=list)
    # Error text of the first job-creating call rejected for a reason only the
    # user can resolve. Once set, job-creating calls are refused for the rest
    # of this request (all rounds, and any retry or fallback model).
    needs_user: Optional[str] = None
    # Place lookups made in this request (see _MAX_GEOCODES_PER_REQUEST).
    geocodes: int = 0


@dataclass
class AgentResult:
    text: str
    job_id: Optional[str] = None
    bbox: Optional[List[float]] = None
    tool_calls: List[ToolInvocation] = field(default_factory=list)
    model: Optional[str] = None
    error: Optional[str] = None
    query: Optional[Dict[str, Any]] = None  # executed SQL + rows, for the client

    def to_api_dict(self) -> Dict[str, Any]:
        """Shape consumed by POST /api/chat (see eo_brain.api.main)."""
        return {"text": self.text, "job_id": self.job_id, "bbox": self.bbox, "query": self.query}


class _AllModelsFailed(RuntimeError):
    pass


class Agent:
    """Function-calling loop over the analysis tools, with injected dependencies."""

    def __init__(
        self,
        provider: LLMProvider,
        *,
        models: Sequence[str],
        job_repo: Any,
        geocode: Callable[[str], GeocodingResult] = geocode_or_fail,
        sessions: Optional[SessionStore] = None,
        clock: Callable[[], datetime] = utc_now,
        sleep: Callable[[float], None] = time.sleep,
        max_tool_rounds: int = _MAX_TOOL_ROUNDS,
        max_retries: int = _MAX_RETRIES,
        backoff_base_sec: float = _BACKOFF_BASE_SEC,
        system_prompt: str = SYSTEM_PROMPT,
        tools: Sequence[ToolSpec] = tuple(TOOL_SPECS),
        data_agent: Any = None,
    ):
        if not models:
            raise ValueError("At least one model id is required")
        self.provider = provider
        self.models = list(models)
        self.job_repo = job_repo
        self.geocode = geocode
        self.sessions = sessions if sessions is not None else session_store
        self.clock = clock
        self.sleep = sleep
        self.max_tool_rounds = max_tool_rounds
        self.max_retries = max_retries
        self.backoff_base_sec = backoff_base_sec
        self.system_prompt = system_prompt
        self.tools = list(tools)
        self.data_agent = data_agent
        self._dispatch: Dict[str, Callable[..., Dict[str, Any]]] = {
            "launch_analysis": self._launch_analysis,
            "get_job_status": self._get_job_status,
            "list_capabilities": self._list_capabilities,
            "rerun_with_different_algorithm": self._rerun_with_different_algorithm,
            "query_database": self._query_database,
        }

    # -- Tool implementations --

    def _now(self, ctx: _RequestContext) -> datetime:
        if ctx.now is None:
            ctx.now = self.clock()
        return ctx.now

    def _launch_analysis(
        self,
        ctx: _RequestContext,
        location: str,
        algorithm: str,
        days_back: Any = None,
        start_date: Any = None,
        end_date: Any = None,
    ) -> Dict[str, Any]:
        """Geocode + create analysis job. Returns dict with job_id or error."""
        return self._launch(ctx, location, algorithm, days_back=days_back,
                            start_date=start_date, end_date=end_date)

    def _launch(
        self,
        ctx: _RequestContext,
        location: str,
        algorithm: str,
        days_back: Any = None,
        start_date: Any = None,
        end_date: Any = None,
        place: Optional[GeocodingResult] = None,
    ) -> Dict[str, Any]:
        """
        ``place``: an area already resolved (rerun on the session's area); no lookup then.
        No period argument at all means the default window, or, for the answer to a
        place question, the period of the request that asked it.
        """
        logger.info(
            f"launch_analysis: location='{location}', algorithm='{algorithm}', "
            f"days_back={days_back}, start_date={start_date}, end_date={end_date}"
        )
        try:
            location = validate_location(location)
            algo = resolve_algorithm(algorithm)
        except ValidationError as e:
            logger.warning(f"Validation failed: {e}")
            return _rejection(str(e), needs_user=e.needs_user)

        geo = place if place is not None else self._pending_choice(ctx, location)
        if geo is not None and place is None:
            logger.info(f"'{location}' is a candidate the user was asked to choose: {geo.name}")
            if days_back is None and not date_given(start_date) and not date_given(end_date):
                # The answer to a place question often repeats no period: the
                # user already gave one, with the request that was asked about.
                remembered = self._pending_period(ctx)
                days_back, start_date, end_date = (
                    remembered["days_back"], remembered["start_date"], remembered["end_date"]
                )

        try:
            period = resolve_period(days_back, start_date, end_date, self._now(ctx),
                                    default_days_back=_DEFAULT_DAYS_BACK)
        except ValidationError as e:
            logger.warning(f"Period rejected: {e}")
            return _rejection(str(e), needs_user=e.needs_user)

        if geo is None:
            if ctx.geocodes >= _MAX_GEOCODES_PER_REQUEST:
                return _rejection(
                    f"The place could not be resolved in {_MAX_GEOCODES_PER_REQUEST} attempts in this "
                    "request. Ask the user for a more precise place name (town, province, region or "
                    "country).",
                    needs_user=True,
                )
            ctx.geocodes += 1
            try:
                geo = self.geocode(location)
                logger.info(f"Geocoded '{location}' -> {geo.name} bbox={geo.bbox}")
            except PlaceChoiceNeeded as e:
                # Only the user can choose: list the candidates and remember them,
                # so that the answer resolves to exactly the one chosen.
                logger.info(f"Place choice needed for '{location}': {[c.label for c in e.candidates]}")
                self._remember_choice(ctx, e, algo, period)
                return _choice_rejection(e)
            except ValueError as e:
                # Model-fixable: the name may be misspelt or too vague.
                logger.warning(f"Geocoding failed: {e}")
                return _rejection(str(e), needs_user=False)

        try:
            bbox = validate_bbox(geo.to_list())
            algo = validate_algorithm_compatibility(algo)
        except ValidationError as e:
            return _rejection(str(e), needs_user=e.needs_user)

        try:
            resolution = resolve_resolution(bbox, requested=10)

            job = self.job_repo.create(
                algorithm=algo,
                provider=_PROVIDER_NAME,
                bbox=bbox,
                start_date=period.start,
                end_date=period.end,
                params={
                    "resolution_m": resolution,
                    "geocoded_name": geo.name,
                    "apply_cloud_mask": True,
                },
            )
            logger.info(f"Job created: {job.id} ({algo} @ {geo.name})")
        except Exception as e:
            logger.error(f"Job creation failed: {e}", exc_info=True)
            return {"error": f"Failed to create job: {e}"}

        ctx.created_jobs.append(str(job.id))

        if ctx.session_id:
            try:
                session = self.sessions.get_or_create(ctx.session_id)
                session.set_context(
                    last_bbox=bbox, last_location=geo.name,
                    last_algorithm=algo, last_job_id=str(job.id),
                    pending_place=None,
                )
            except Exception:
                logger.warning("Could not store session context", exc_info=True)

        area = estimate_area_km2(bbox)
        if area < 50:
            time_est = "20-40 seconds"
        elif area < 200:
            time_est = "30-60 seconds"
        elif area < 600:
            time_est = "1-2 minutes"
        else:
            time_est = "2-4 minutes"

        first, last = str(period.start_date), str(period.end_date)
        result = {
            "job_id": str(job.id), "status": "pending", "algorithm": algo,
            "location": geo.name, "bbox": bbox, "resolution_m": resolution,
            "date_range": f"{first} to {last}",
            "start_date": first, "end_date": last,
            "estimated_time": time_est,
            "message": (
                f"Analysis '{algo.upper()}' queued for {geo.name} "
                f"({first} to {last}, {resolution}m). "
                f"Estimated time: {time_est}."
            ),
        }
        if period.days_back is not None:
            # Only for the look-back form: a calendar period must not be
            # described back to the user as "N days".
            result["days_back"] = period.days_back
        kind = f" ({geo.kind})" if getattr(geo, "kind", None) else ""
        notes, said = [], []
        fallback_from = getattr(geo, "fallback_from", None)
        if fallback_from:
            # Say which place was actually used: the retried text may match something
            # narrower or different from what the user meant (a town for its province).
            notes.append(
                f"'{fallback_from}' was not found as written; searched for "
                f"'{geo.matched_query}' instead and used {geo.name}{kind}."
            )
            said.append(f"'{fallback_from}' not found as written, used {geo.name}{kind}.")
        if not getattr(geo, "area_like", True):
            notes.append(
                f"{geo.name} is a {geo.kind}, not a town, region or natural area: nothing else with "
                "that name was found, and the analysis covers its bounding box."
            )
            said.append(f"{geo.name} is a {geo.kind}; the analysis covers its bounding box.")
        if notes:
            result["geocoding_note"] = " ".join(notes) + " Tell the user."
            result["message"] += " Location: " + " ".join(said)
        ctx.last_launch = result
        return result

    # -- Choosing among places --

    def _remember_choice(self, ctx: _RequestContext, e: PlaceChoiceNeeded, algo: str, period: Period) -> None:
        if not ctx.session_id:
            return
        self.sessions.get_or_create(ctx.session_id).set_context(pending_place={
            "query": e.query, "reason": e.reason, "algorithm": algo, **_period_args(period),
            "candidates": list(e.candidates), "note_shown": False,
        })

    def _pending_period(self, ctx: _RequestContext) -> Dict[str, Any]:
        """The period of the request that asked the place question, as it was requested."""
        pending = self.sessions.get_or_create(ctx.session_id).get_context("pending_place") or {}
        return {k: pending.get(k) for k in ("days_back", "start_date", "end_date")}

    def _pending_choice(self, ctx: _RequestContext, location: str) -> Optional[GeocodingResult]:
        """
        The candidate ``location`` picks among those the user was last asked to
        choose from, or None. It picks one when it is that candidate's label,
        or when it names the candidate and all its words appear in that
        candidate's label and kind, and in no other candidate's
        ("Casentino, Arezzo" picks the region, "Casentino" picks none).
        """
        if not ctx.session_id:
            return None
        pending = self.sessions.get_or_create(ctx.session_id).get_context("pending_place")
        if not pending:
            return None
        candidates: List[PlaceCandidate] = pending["candidates"]
        wanted = normalize_place(location)
        for c in candidates:
            if wanted == normalize_place(c.label):
                return c.result
        tokens = set(wanted.split())
        hits = []
        for c in candidates:
            own = set(normalize_place(c.label.split(",")[0]).split())
            if tokens and own <= tokens and tokens <= set(normalize_place(f"{c.label} {c.kind}").split()):
                hits.append(c)
        return hits[0].result if len(hits) == 1 else None

    def _pending_note(self, session_id: str) -> Optional[str]:
        """Context for the turn after a place question (the model sees no history), shown once."""
        if not session_id:
            return None
        session = self.sessions.get_or_create(session_id)
        pending = session.get_context("pending_place")
        if not pending or pending.get("note_shown"):
            return None
        pending["note_shown"] = True
        listed = "; ".join(f"{i}) {c.label} ({c.kind})" for i, c in enumerate(pending["candidates"], 1))
        algo = pending["algorithm"]
        if pending.get("days_back") is not None:
            period = f"{pending['days_back']} days"
            pass_back = f"algorithm '{algo}' and days_back {pending['days_back']}"
        else:
            period = f"{pending['start_date']} to {pending['end_date']}"
            pass_back = (f"algorithm '{algo}', start_date {pending['start_date']} and "
                         f"end_date {pending['end_date']}")
        return (
            "[Context from the app, not written by the user: the previous reply asked the user to "
            f"choose the place for '{pending['query']}' ({algo}, {period}). Candidates: {listed}. "
            "If the message below chooses one of them (by number, name, province or region), call "
            f"launch_analysis with location set to that candidate's text exactly as listed, "
            f"{pass_back}. Otherwise ignore this context.]"
        )

    def _get_job_status(self, ctx: _RequestContext, job_id: Optional[str] = None) -> Dict[str, Any]:
        if not job_id:
            if ctx.session_id:
                job_id = self.sessions.get_or_create(ctx.session_id).get_context("last_job_id")
            if not job_id:
                return {"error": (
                    "No analysis was launched in this conversation. Ask for the job ID, or "
                    "use query_database for analyses stored earlier."
                )}
        try:
            uid = UUID(str(job_id))
        except ValueError:
            return {"error": f"Invalid job ID: {job_id}"}

        try:
            job = self.job_repo.get(uid)
        except Exception as e:
            return {"error": f"Database error: {e}"}

        if not job:
            return {"error": f"Job {job_id} not found."}

        result_summary = None
        if job.status == "completed" and job.result:
            result_summary = {
                "tile_url": job.result.get("tile_url"),
                "summary": job.result.get("summary"),
                "coverage": _coverage_facts(job.result.get("coverage")),
            }
        params = job.params or {}
        return {
            "job_id": str(job.id), "status": job.status,
            "algorithm": job.algorithm, "progress": job.progress,
            "location": params.get("geocoded_name"),
            "time_window": _window(job.start_date, job.end_date),
            "message": job.message, "result": result_summary,
        }

    def _list_capabilities(self, ctx: _RequestContext) -> Dict[str, Any]:
        return {
            "algorithms": [
                {"name": k, "description": v.description, "required_bands": v.required_bands}
                for k, v in ALGO_REGISTRY.items()
            ]
        }

    def _rerun_with_different_algorithm(
        self, ctx: _RequestContext, algorithm: str = ""
    ) -> Dict[str, Any]:
        # No previous area: needs the user, because any place the model chose for
        # a launch_analysis retry would be invented. (If the user did name a
        # place in the same message, this also blocks that legitimate retry.)
        if not ctx.session_id:
            return _rejection("No session context. Please specify a location.", needs_user=True)

        session = self.sessions.get_or_create(ctx.session_id)
        last_location = session.get_context("last_location")
        if not last_location:
            return _rejection(
                "No previous analysis in this session. Specify a location.", needs_user=True
            )

        algo = algorithm if algorithm else session.get_context("last_algorithm", "ndvi")
        last_bbox = session.get_context("last_bbox")
        place = None
        if last_bbox and len(last_bbox) == 4:
            # The same area as the last analysis, as stored: geocoding its name
            # again could resolve to another place.
            w, s, e, n = (float(v) for v in last_bbox)
            place = GeocodingResult(name=last_location, lat=(s + n) / 2, lon=(w + e) / 2,
                                    bbox=(w, s, e, n), source="session")
        return self._launch(ctx, location=last_location, algorithm=algo, place=place)

    def _query_database(self, ctx: _RequestContext, question: str) -> Dict[str, Any]:
        """Delegate to the isolated data agent; it sees the question, not this conversation."""
        if self.data_agent is None:
            return {"error": "Querying stored results is not configured on this server."}
        if not isinstance(question, str) or not question.strip():
            return {"error": "Question is empty."}

        area_hint = None
        if ctx.session_id:
            session = self.sessions.get_or_create(ctx.session_id)
            loc, bbox = session.get_context("last_location"), session.get_context("last_bbox")
            if loc and bbox:
                w, s, e, n = (round(float(v), 5) for v in bbox)
                area_hint = (
                    f"the user's current area is '{loc}', bbox west={w} south={s} "
                    f"east={e} north={n} (EPSG:4326)"
                )

        result = self.data_agent.answer(question, area_hint=area_hint)
        if result.status == "ok":
            ctx.last_query = result
        return result.to_tool_response()

    # -- Loop --

    def _execute_tool_call(
        self, ctx: _RequestContext, fn_name: str, fn_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Dispatch a tool call and return the result dict. Never raises."""
        handler = self._dispatch.get(fn_name)
        if fn_name in JOB_TOOLS and ctx.needs_user is not None:
            # Deterministic refusal: the request already hit a limit only the
            # user can lift, so a second launch would be a substituted request.
            logger.warning(f"Blocked {fn_name}({fn_args}) after a needs-user rejection")
            blocked = {
                "error": ctx.needs_user, "needs_user": True, "blocked": True,
                "action_required": _BLOCKED_ACTION,
            }
            ctx.trace.append(ToolInvocation(name=fn_name, args=dict(fn_args), result=blocked))
            return blocked
        if not handler:
            logger.error(f"Unknown tool function: {fn_name}")
            result: Dict[str, Any] = {"error": f"Unknown function: {fn_name}"}
        else:
            logger.info(f"Executing tool: {fn_name}({fn_args})")
            try:
                result = handler(ctx, **fn_args)
                logger.info(
                    f"Tool {fn_name} returned: job_id={result.get('job_id', 'N/A')}, "
                    f"error={result.get('error', 'none')}"
                )
            except Exception as e:
                logger.error(f"Tool {fn_name} threw: {e}", exc_info=True)
                result = {"error": f"Internal error in {fn_name}: {e}"}
        if fn_name in JOB_TOOLS and result.get("needs_user"):
            ctx.needs_user = str(result.get("error", ""))
        ctx.trace.append(ToolInvocation(name=fn_name, args=dict(fn_args), result=result))
        return result

    def _run_tool_loop(self, model: str, message: str, ctx: _RequestContext) -> str:
        """
        Ask the model, execute any tool calls, feed results back; repeat until
        the model answers with text or the round cap is hit.
        """
        messages: List[Message] = [UserMessage(text=message)]
        system_prompt = f"{self.system_prompt}\n{today_line(self._now(ctx))}"

        for _ in range(self.max_tool_rounds):
            turn = self.provider.generate(
                model=model,
                system_prompt=system_prompt,
                tools=self.tools,
                messages=messages,
            )
            if turn is None:
                return _NO_RESPONSE_TEXT

            if not turn.tool_calls:
                return turn.text or ""

            results = [
                ToolResult(name=call.name, response=self._execute_tool_call(ctx, call.name, call.args))
                for call in turn.tool_calls
            ]
            messages.append(turn)
            messages.append(ToolResults(results=results))

        logger.warning("Exceeded max tool call rounds")
        if ctx.needs_user is not None and not ctx.created_jobs:
            # The model kept calling instead of answering; the limit it hit is
            # the honest answer, not "being processed" (nothing is).
            return ctx.needs_user
        return _MAX_ROUNDS_TEXT

    def _call_with_fallback(self, message: str, ctx: _RequestContext) -> tuple[str, str]:
        """Try each model in order, retrying on rate limits. Returns (text, model)."""
        last_error: Optional[BaseException] = None

        for model in self.models:
            for attempt in range(self.max_retries):
                try:
                    return self._run_tool_loop(model, message, ctx), model
                except RateLimitError as e:
                    last_error = e
                    if self._has_side_effects(ctx):
                        break
                    if e.daily_quota:
                        logger.warning(f"{model} daily quota exhausted, trying next model")
                        break
                    if attempt + 1 < self.max_retries:
                        wait = self.backoff_base_sec * (2 ** attempt)
                        logger.warning(f"{model} rate limited, retry in {wait}s")
                        self.sleep(wait)
                except Exception as e:
                    last_error = e
                    logger.error(f"{model} error: {e}", exc_info=True)
                    break

            if self._has_side_effects(ctx):
                # A job already exists for this request: replaying the request
                # on another attempt/model could create a duplicate. Report what
                # was done, deterministically, instead.
                logger.warning(
                    f"Model call failed after a job was created ({last_error}); "
                    "not retrying to avoid duplicate jobs"
                )
                return str(ctx.last_launch.get("message", "")), model

        raise _AllModelsFailed(f"All models exhausted. Last error: {last_error}")

    @staticmethod
    def _has_side_effects(ctx: _RequestContext) -> bool:
        return bool(ctx.created_jobs) and ctx.last_launch is not None

    def run(self, message: str, session_id: str) -> AgentResult:
        """
        Process a user message. Each request is stateless towards the model
        (no conversation history is sent) to avoid poisoning from partial
        tool-call artifacts in history. Session context (last_location,
        last_algorithm) is kept server-side for "same area" follow-ups via
        rerun_with_different_algorithm.
        """
        message = sanitize_input(message)
        if not message.strip():
            return AgentResult(text="Please enter a message.")

        self.sessions.get_or_create(session_id)
        ctx = _RequestContext(session_id=session_id, now=self.clock())
        note = self._pending_note(session_id)
        model_message = f"{note}\n\nUser message: {message}" if note else message

        try:
            reply_text, model = self._call_with_fallback(model_message, ctx)
        except _AllModelsFailed as e:
            logger.error("All models exhausted", exc_info=True)
            return AgentResult(text=_UNAVAILABLE_TEXT, tool_calls=ctx.trace, error=str(e))

        job_id = ctx.created_jobs[-1] if ctx.created_jobs else None
        if job_id:
            logger.info(f"Job created via tool call: {job_id}")

        reply_text = re.sub(r"\s*\[JOB_CREATED:[^\]]*\]", "", reply_text).strip()
        if job_id and ctx.last_launch is not None:
            reply_text = _with_period(reply_text, ctx.last_launch)

        # Pass bbox back so frontend can fly to the analysis area
        bbox = None
        if job_id:
            try:
                job = self.job_repo.get(UUID(job_id))
                if job:
                    bbox = job.bbox
            except Exception:
                logger.warning("Could not read back job bbox", exc_info=True)

        query = ctx.last_query.to_api_payload() if ctx.last_query is not None else None
        return AgentResult(
            text=reply_text, job_id=job_id, bbox=bbox, tool_calls=ctx.trace, model=model,
            query=query,
        )

    def chat(self, message: str, session_id: str) -> Dict[str, Any]:
        return self.run(message, session_id).to_api_dict()


# -- Default (production) agent, built lazily on first use --

_default_agent: Optional[Agent] = None
_default_lock = threading.Lock()


def get_default_agent() -> Optional[Agent]:
    """Gemini-backed agent from environment, or None if no API key is set."""
    global _default_agent
    with _default_lock:
        if _default_agent is None:
            if not os.environ.get("GEMINI_API_KEY"):
                logger.warning("GEMINI_API_KEY not set. Agent chat will be unavailable.")
                return None

            from eo_brain.agent.providers.gemini import GeminiProvider, model_chain_from_env
            from eo_brain.db.database import job_repo

            models = model_chain_from_env()
            provider = GeminiProvider()
            data_agent = None
            agent_db_url = os.environ.get("AGENT_DATABASE_URL", "").strip()
            if agent_db_url:
                from eo_brain.agent.data_agent import DataAgent
                from eo_brain.agent.sql_exec import ReadOnlyExecutor

                data_agent = DataAgent(provider, models=models, executor=ReadOnlyExecutor(agent_db_url))
            else:
                logger.warning("AGENT_DATABASE_URL not set. query_database will be unavailable.")
            _default_agent = Agent(provider, models=models, job_repo=job_repo, data_agent=data_agent)
            logger.info(
                f"Agent initialised (provider=gemini, models={models}, "
                f"data_agent={'on' if data_agent else 'off'})"
            )
        return _default_agent


def chat(message: str, session_id: str) -> dict:
    """Entry point used by POST /api/chat. Returns text + optional job_id + bbox."""
    agent = get_default_agent()
    if agent is None:
        return {"text": _NOT_CONFIGURED_TEXT, "job_id": None, "bbox": None, "query": None}
    return agent.chat(message, session_id)
