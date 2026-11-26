"""
evals/scoring.py - Turn one agent run into a scored case result.

Scoring rules (kept deliberately simple and documented in the README):

* The scored call is the FIRST tool call of the final turn.
* ``algorithm`` is compared after the deterministic alias resolution
  (``validators.resolve_algorithm``), because aliases are accepted by design.
  If a job was created, the job's algorithm is used (this also covers
  ``rerun_with_different_algorithm`` with an empty argument).
* ``days_back`` falls back to the tool default (30) when the model omits it.
* ``location`` is compared on normalised text (casefold, no accents, no
  punctuation) by whole-word phrase containment.
* A refusal case passes when the final turn created no job, whether the
  model declined in text or a deterministic validator rejected the call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from eo_brain.agent.validators import ValidationError, resolve_algorithm
from evals.stubs import contains_phrase

DEFAULT_DAYS_BACK = 30
JOB_TOOLS = {"launch_analysis", "rerun_with_different_algorithm"}


@dataclass
class CaseResult:
    id: str
    lang: str
    kind: str                     # "tool_call" | "refusal"
    expected_tool: Optional[str]
    called_tools: List[str]
    effective_args: Dict[str, Any]
    arg_checks: Dict[str, bool]
    job_created: bool
    tool_ok: Optional[bool]       # None for refusal cases
    args_ok: Optional[bool]       # None for refusal cases
    refused_ok: Optional[bool]    # None for tool_call cases
    passed: bool
    reply_text: str = ""
    tool_errors: List[str] = field(default_factory=list)
    note: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _resolved_algorithm(raw: Any) -> Optional[str]:
    if raw is None or raw == "":
        return None
    try:
        return resolve_algorithm(str(raw))
    except ValidationError:
        return str(raw).strip().lower()


def _days(raw: Any) -> Any:
    if raw is None:
        return DEFAULT_DAYS_BACK
    if isinstance(raw, float) and raw.is_integer():
        return int(raw)
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return raw


def effective_args(call) -> Dict[str, Any]:
    """Values the scorer compares, derived from one ToolInvocation."""
    args, result = call.args, call.result or {}
    job = "job_id" in result and "error" not in result and call.name in JOB_TOOLS
    eff: Dict[str, Any] = {}
    if call.name in JOB_TOOLS:
        eff["algorithm"] = result["algorithm"] if job else _resolved_algorithm(args.get("algorithm"))
    if call.name == "launch_analysis":
        eff["days_back"] = _days(args.get("days_back"))
        eff["location"] = args.get("location")
    if call.name == "rerun_with_different_algorithm":
        eff["location"] = result.get("location") if job else None
    if call.name == "get_job_status":
        eff["job_id"] = str(args.get("job_id", "")).strip().lower()
    return eff


def check_constraint(value: Any, constraint: Dict[str, Any]) -> bool:
    if "equals" in constraint:
        expected = constraint["equals"]
        if isinstance(expected, str) and isinstance(value, str):
            return value.strip().lower() == expected.strip().lower()
        return value == expected
    if "any_of" in constraint:
        return value in constraint["any_of"]
    if "range" in constraint:
        lo, hi = constraint["range"]
        return isinstance(value, (int, float)) and not isinstance(value, bool) and lo <= value <= hi
    if "matches" in constraint:
        return isinstance(value, str) and any(contains_phrase(value, p) for p in constraint["matches"])
    raise ValueError(f"Unknown constraint: {constraint!r}")


def score_case(case: Dict[str, Any], agent_result, note: str = "") -> CaseResult:
    expect = case["expect"]
    kind = expect["type"]
    calls = list(agent_result.tool_calls)
    called = [c.name for c in calls]
    job_created = agent_result.job_id is not None
    errors = [str(c.result.get("error")) for c in calls if c.result and c.result.get("error")]

    base = dict(
        id=case["id"],
        lang=case.get("lang", "?"),
        kind=kind,
        called_tools=called,
        job_created=job_created,
        reply_text=agent_result.text,
        tool_errors=errors,
        note=note or (agent_result.error or ""),
        tags=list(case.get("tags", [])),
    )

    if kind == "refusal":
        ok = not job_created
        return CaseResult(
            expected_tool=None, effective_args={}, arg_checks={},
            tool_ok=None, args_ok=None, refused_ok=ok, passed=ok and not note, **base,
        )

    if kind != "tool_call":
        raise ValueError(f"Case {case['id']}: unknown expect.type {kind!r}")

    expected_tool = expect["tool"]
    first = calls[0] if calls else None
    tool_ok = first is not None and first.name == expected_tool
    eff = effective_args(first) if first else {}
    checks = {
        name: tool_ok and check_constraint(eff.get(name), constraint)
        for name, constraint in (expect.get("args") or {}).items()
    }
    args_ok = tool_ok and all(checks.values())
    needs_job = expect.get("job_created", expected_tool in JOB_TOOLS)
    job_ok = job_created if needs_job else True
    return CaseResult(
        expected_tool=expected_tool, effective_args=eff, arg_checks=checks,
        tool_ok=tool_ok, args_ok=args_ok, refused_ok=None,
        passed=bool(tool_ok and args_ok and job_ok and not note), **base,
    )


def _ratio(num: int, den: int) -> Dict[str, Any]:
    return {"correct": num, "total": den, "rate": (num / den) if den else None}


def summarize(results: List[CaseResult]) -> Dict[str, Any]:
    tool_cases = [r for r in results if r.kind == "tool_call"]
    refusal_cases = [r for r in results if r.kind == "refusal"]
    routing_cases = [r for r in tool_cases if "routing" in r.tags]
    return {
        "cases": len(results),
        "passed": sum(r.passed for r in results),
        "tool_selection_accuracy": _ratio(sum(bool(r.tool_ok) for r in tool_cases), len(tool_cases)),
        "argument_accuracy": _ratio(sum(bool(r.args_ok) for r in tool_cases), len(tool_cases)),
        "correct_refusal_rate": _ratio(sum(bool(r.refused_ok) for r in refusal_cases), len(refusal_cases)),
        "routing_accuracy": _ratio(sum(bool(r.tool_ok) for r in routing_cases), len(routing_cases)),
        "false_refusals": sum(
            1 for r in tool_cases if r.expected_tool in JOB_TOOLS and not r.job_created
        ),
        "errored": sum(1 for r in results if r.note),
    }
