"""
evals/provider_guard.py - Keep live eval runs honest about provider failures.

A case whose model call failed has no answer to score. ``ProviderGuard``
wraps the live provider for the whole run and records, per case, every
error the provider raised. It also remembers which models reported an
exhausted DAILY quota and refuses to send them further requests (it raises
the same ``RateLimitError(daily_quota=True)`` locally), so a fallback chain
moves on immediately and the runner can stop once every model is exhausted.

``run_cases`` is the loop both suites share: it stops at that point and
marks the remaining cases "not run", and re-runs a case once (after a pause)
when it errored only because of a transient failure (HTTP 5xx / timeout /
per-minute 429). Neither agent retries a 5xx itself, and the data agent does
not retry a per-minute 429 either, so without this a moment of "high demand"
turns straight into errored cases.

Case outcomes (both suites):
    pass / fail   the model answered and the answer was scored
    error         no answer to score: the model call failed (error_kind
                  "provider") or the harness itself broke ("harness")
    not_run       never attempted because the run stopped early
Only pass / fail cases enter any accuracy, refusal, routing, blocked or
execution-accuracy denominator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from eo_brain.agent.llm import ProviderError, RateLimitError

OUTCOMES = ("pass", "fail", "error", "not_run")
TRANSIENT_RETRY_PAUSE_S = 20.0
DEFAULT_CASE_RETRIES = 1


@dataclass(frozen=True)
class ProviderErrorRecord:
    model: str
    kind: str              # daily_quota | rate_limit | unavailable | provider | other
    status: Optional[int]
    transient: bool
    sent: bool             # False when the guard answered locally (model already exhausted)
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _record(model: str, exc: BaseException, *, sent: bool) -> ProviderErrorRecord:
    if isinstance(exc, RateLimitError):
        kind = "daily_quota" if exc.daily_quota else "rate_limit"
    elif isinstance(exc, ProviderError):
        kind = "unavailable" if exc.transient else "provider"
    else:
        kind = "other"
    return ProviderErrorRecord(
        model=model,
        kind=kind,
        status=getattr(exc, "status", None),
        transient=bool(getattr(exc, "transient", False)),
        sent=sent,
        message=str(exc),
    )


class ProviderGuard:
    """Wraps a live LLMProvider for one eval run (see module docstring)."""

    def __init__(self, inner: Any, models: Sequence[str]):
        self.inner = inner
        self.models = list(models)
        self.name = f"guarded:{getattr(inner, 'name', type(inner).__name__)}"
        self.exhausted: List[str] = []    # models with an exhausted daily quota, in order seen
        self.requests_sent = 0
        self._case_errors: List[ProviderErrorRecord] = []

    # -- per-case bookkeeping
    def begin_case(self) -> None:
        self._case_errors = []

    @property
    def case_errors(self) -> List[ProviderErrorRecord]:
        return list(self._case_errors)

    @property
    def all_exhausted(self) -> bool:
        return bool(self.models) and all(m in self.exhausted for m in self.models)

    def stop_reason(self) -> Optional[str]:
        if not self.all_exhausted:
            return None
        return f"daily quota exhausted for every model in the chain ({', '.join(self.models)})"

    # -- LLMProvider
    def generate(self, *, model, system_prompt, tools, messages):
        if model in self.exhausted:
            exc = RateLimitError(
                f"daily quota of {model} already exhausted earlier in this run; request not sent",
                daily_quota=True,
            )
            self._case_errors.append(_record(model, exc, sent=False))
            raise exc
        self.requests_sent += 1
        try:
            return self.inner.generate(model=model, system_prompt=system_prompt, tools=tools, messages=messages)
        except Exception as exc:
            self._case_errors.append(_record(model, exc, sent=True))
            if isinstance(exc, RateLimitError) and exc.daily_quota and model not in self.exhausted:
                self.exhausted.append(model)
            raise


def run_cases(
    cases: Sequence[Dict[str, Any]],
    run_one: Callable[[Dict[str, Any]], Any],
    *,
    not_run: Callable[[Dict[str, Any], str], Any],
    guard: Optional[ProviderGuard] = None,
    case_retries: int = 0,
    retry_pause_s: float = TRANSIENT_RETRY_PAUSE_S,
    pause: Callable[[float], None] = time.sleep,
):
    """
    Run ``run_one`` per case and return ``(results, stop_reason)``.

    Each result must have ``outcome``, ``error_kind``, ``error_reason``,
    ``runs``, ``elapsed_s`` and ``provider_errors`` attributes. Without a
    guard (scripted mode) this is a plain loop.
    """
    results: List[Any] = []
    stop_reason: Optional[str] = None
    for i, case in enumerate(cases):
        if guard is not None and guard.all_exhausted:
            stop_reason = guard.stop_reason()
            results.extend(not_run(c, stop_reason) for c in cases[i:])
            break

        trail: List[ProviderErrorRecord] = []
        runs = 0
        started = time.monotonic()
        while True:
            runs += 1
            if guard is not None:
                guard.begin_case()
            result = run_one(case)
            errors = guard.case_errors if guard is not None else []
            trail.extend(errors)
            if result.outcome == "error" and result.error_kind == "provider" and errors:
                result.error_reason = errors[-1].kind
            retry = (
                guard is not None
                and result.outcome == "error"
                and result.error_kind == "provider"
                and runs <= case_retries
                and bool(errors)
                and errors[-1].transient
                and not guard.all_exhausted
            )
            if not retry:
                break
            pause(retry_pause_s)

        result.runs = runs
        result.elapsed_s = round(time.monotonic() - started, 3)
        result.provider_errors = [e.to_dict() for e in trail]
        results.append(result)

    if stop_reason is None and guard is not None and guard.all_exhausted and len(results) == len(cases):
        # The quota ran out during the last case: nothing left to skip, but say so.
        stop_reason = guard.stop_reason()
    return results, stop_reason


def completeness(results: Sequence[Any]) -> Dict[str, Any]:
    """Counts every report carries, so an incomplete run cannot pass for a result."""
    counts = {o: sum(1 for r in results if r.outcome == o) for o in OUTCOMES}
    answered = counts["pass"] + counts["fail"]
    return {
        "cases": len(results),
        "answered": answered,
        "errored": counts["error"],
        "not_run": counts["not_run"],
        "complete": answered == len(results),
        "errored_by_kind": {
            k: sum(1 for r in results if r.outcome == "error" and r.error_kind == k) for k in ("provider", "harness")
        },
        "incomplete_case_ids": [r.id for r in results if r.outcome in ("error", "not_run")],
    }


def one_line(text: Any, limit: int = 160) -> str:
    """Single-line, truncated rendering for the console table (the JSON keeps the full text)."""
    s = " ".join(str(text or "").split())
    return s if len(s) <= limit else s[: limit - 3] + "..."


def print_completeness(summary: Dict[str, Any], stop_reason: Optional[str]) -> None:
    print(f"answered (scored)       {summary['answered']}/{summary['cases']}  "
          f"(errored {summary['errored']}, not run {summary['not_run']})")
    if stop_reason:
        print(f"stopped early           {stop_reason}")
    if not summary["complete"]:
        print()
        print("*" * 100)
        print(f"RUN INCOMPLETE: {summary['cases'] - summary['answered']} of {summary['cases']} cases have no scored "
              f"answer. The metrics below cover only the {summary['answered']} answered case(s);")
        print("they are NOT a result for the dataset. Errored / not-run case ids are listed in the JSON report.")
        print("*" * 100)
        print()
