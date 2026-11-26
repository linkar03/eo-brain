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
  * the model decides WHICH tool to call and with WHICH raw arguments;
  * code decides everything else - alias resolution, range checks,
    geocoding, bbox/resolution rules, job creation, session context.

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
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence
from uuid import UUID

from eo_brain.agent.geocoder import GeocodingResult, geocode_or_fail
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
    ValidationError,
    estimate_area_km2,
    resolve_algorithm,
    resolve_resolution,
    validate_algorithm_compatibility,
    validate_bbox,
    validate_location,
    validate_temporal_range,
)
from eo_brain.algorithms.base import ALGORITHMS as ALGO_REGISTRY

logger = logging.getLogger(__name__)

_MAX_INPUT_CHARS = 2000
_MAX_RETRIES = 3
_BACKOFF_BASE_SEC = 5
_MAX_TOOL_ROUNDS = 5  # safety cap on tool call loops
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
                    "description": "Place name, e.g. 'Florence, Italy', 'Po River', 'Milan'",
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
                        "Temporal window in days, exactly as the user asked "
                        "('last 3 months' = 90, 'last 2 years' = 730). Omit it when the "
                        "user gives no window (default 30). Never shorten it to fit a "
                        "limit: the tool checks the limit."
                    ),
                },
            },
            "required": ["location", "algorithm"],
        },
    ),
    ToolSpec(
        name="get_job_status",
        description="Check the current status of an analysis job by its ID.",
        parameters={
            "type": "object",
            "properties": {
                "job_id": {"type": "string", "description": "The job UUID"},
            },
            "required": ["job_id"],
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
   how to proceed. Do not retry with other values (a shorter window, another place, another analysis).
4. NEVER fabricate job IDs, URLs, or analysis results. Only report data returned by tools.
5. NEVER run analyses the user did not request. If the user asks for something the analyses above \
   do not measure (traffic, weather, air quality, population, ...), say it is not supported and list \
   what is supported; do not run a supported analysis in its place.
6. After a successful tool call, briefly confirm what was launched and the estimated time.
7. You understand Italian and English. Respond in the same language the user writes in.
8. Default to Italian when the language is ambiguous (short commands, place names only).
9. When parsing compact input like "ndvi firenze", treat the first token as the algorithm \
   and the rest as the location. Call launch_analysis immediately.
10. The days_back parameter controls the temporal window. Default is 30 days. \
   If the user says "last 3 months" use days_back=90, "last year" use days_back=365, \
   "last week" use days_back=7, etc. Always interpret temporal references. \
   Pass the window the user asked for even when it looks long ("last 2 years" = 730, \
   "last 5 years" = 1825); never shorten it yourself. If it is over the limit the tool \
   rejects it: tell the user the limit and ask which window they want.
11. Route by intent: a request to analyze an area now -> launch_analysis. A question about \
   analyses that already exist (how many, how much area, which ones, compare, status history) \
   -> query_database with the question. Never launch an analysis to answer such a question.
12. Only state numbers that appear in tool results (for query_database: in the returned rows). \
   If query_database returns status "refused" or "failed", say so plainly; do not guess.
"""


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
    created_jobs: List[str] = field(default_factory=list)
    last_launch: Optional[Dict[str, Any]] = None
    last_query: Optional[Any] = None  # DataAgentResult of the last successful query
    trace: List[ToolInvocation] = field(default_factory=list)
    # Error text of the first job-creating call rejected for a reason only the
    # user can resolve. Once set, job-creating calls are refused for the rest
    # of this request (all rounds, and any retry or fallback model).
    needs_user: Optional[str] = None


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

    def _launch_analysis(
        self,
        ctx: _RequestContext,
        location: str,
        algorithm: str,
        days_back: Optional[int] = _DEFAULT_DAYS_BACK,
    ) -> Dict[str, Any]:
        """Geocode + create analysis job. Returns dict with job_id or error."""
        logger.info(
            f"launch_analysis: location='{location}', algorithm='{algorithm}', days_back={days_back}"
        )
        if days_back is None:
            days_back = _DEFAULT_DAYS_BACK

        try:
            location = validate_location(location)
            algo = resolve_algorithm(algorithm)
            days_back = validate_temporal_range(days_back)
        except ValidationError as e:
            logger.warning(f"Validation failed: {e}")
            return _rejection(str(e), needs_user=e.needs_user)

        try:
            geo = self.geocode(location)
            logger.info(f"Geocoded '{location}' -> {geo.name} bbox={geo.bbox}")
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
            end_date = self.clock()
            start_date = end_date - timedelta(days=days_back)

            job = self.job_repo.create(
                algorithm=algo,
                provider=_PROVIDER_NAME,
                bbox=bbox,
                start_date=start_date,
                end_date=end_date,
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

        result = {
            "job_id": str(job.id), "status": "pending", "algorithm": algo,
            "location": geo.name, "bbox": bbox, "resolution_m": resolution,
            "days_back": days_back,
            "date_range": f"{start_date.date()} to {end_date.date()}",
            "estimated_time": time_est,
            "message": (
                f"Analysis '{algo.upper()}' queued for {geo.name} "
                f"({start_date.date()} to {end_date.date()}, {resolution}m). "
                f"Estimated time: {time_est}."
            ),
        }
        ctx.last_launch = result
        return result

    def _get_job_status(self, ctx: _RequestContext, job_id: str) -> Dict[str, Any]:
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
            }
        return {
            "job_id": str(job.id), "status": job.status,
            "algorithm": job.algorithm, "progress": job.progress,
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
        return self._launch_analysis(ctx, location=last_location, algorithm=algo)

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

        for _ in range(self.max_tool_rounds):
            turn = self.provider.generate(
                model=model,
                system_prompt=self.system_prompt,
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
        ctx = _RequestContext(session_id=session_id)

        try:
            reply_text, model = self._call_with_fallback(message, ctx)
        except _AllModelsFailed as e:
            logger.error("All models exhausted", exc_info=True)
            return AgentResult(text=_UNAVAILABLE_TEXT, tool_calls=ctx.trace, error=str(e))

        job_id = ctx.created_jobs[-1] if ctx.created_jobs else None
        if job_id:
            logger.info(f"Job created via tool call: {job_id}")

        reply_text = re.sub(r"\s*\[JOB_CREATED:[^\]]*\]", "", reply_text).strip()

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
