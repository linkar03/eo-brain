"""
evals/run.py - Eval runner for both agents.

    python -m evals.run --provider scripted                      # tool suite, offline
    python -m evals.run --provider scripted --suite sql \
        --admin-url postgresql://postgres:postgres@127.0.0.1:5432/postgres
    python -m evals.run --provider gemini [--suite all]          # live, needs GEMINI_API_KEY

Suites:
* ``tools`` (default) - text -> tool call for the orchestrator, including the
  routing cases (query_database vs launch_analysis). Runs the real ``Agent``
  loop, dispatch and validators; geocoding, job storage, the clock and the
  data agent are stubbed (evals/stubs.py), so no network and no database.
* ``sql`` - question -> SQL -> rows for the data agent, on a throwaway
  database with the SYNTHETIC seed (evals/sql_suite.py). Needs a PostGIS
  superuser URL (``--admin-url`` or ``TEST_DATABASE_URL``); the runner
  creates and drops its own database.

``scripted`` replays the canned model output stored in each case
(``scripted:``). It checks the harness, the loops, the validators, the SQL
gate and the read-only role. It is NOT a measure of model quality: by
construction it only shows whether the plumbing turns ideal model output
into the expected outcome.

Live runs: a case whose model call failed is outcome "error" and is left out
of every rate; once every model in the chain reports an exhausted daily
quota the run stops and the remaining cases are "not run" (see
evals/provider_guard.py). Each report says how many cases were answered and
whether the run is ``complete``.

Exit codes:
  0  scripted: every case passed / live: run complete (every case answered)
  1  scripted regression gate failed (some case is not a pass)
  2  usage or configuration error (missing API key, missing database URL)
  3  live run INCOMPLETE (some case errored or was not run): not a result
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from eo_brain.agent.brain import Agent
from eo_brain.agent.llm import ModelTurn, ToolCall
from eo_brain.agent.providers.scripted import ScriptedProvider, relay_tool_results
from eo_brain.agent.session import SessionStore
from evals.provider_guard import (
    DEFAULT_CASE_RETRIES,
    ProviderGuard,
    one_line,
    print_completeness,
    run_cases,
)
from evals.scoring import CaseResult, errored_case, not_run_case, score_case, summarize
from evals.stubs import FIXED_NOW, InMemoryJobRepository, StubDataAgent, StubGeocoder

HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases.v1.yaml"
DEFAULT_SQL_CASES = HERE / "sql_cases.v1.yaml"
DEFAULT_REPORT_DIR = HERE / "reports"
SCRIPTED_MODEL = "scripted-replay"

EXIT_OK = 0
EXIT_REGRESSION = 1     # scripted: some case is not a pass
EXIT_USAGE = 2
EXIT_INCOMPLETE = 3     # live: some case errored or was not run

SCRIPTED_BANNER = (
    "PROVIDER: scripted - replays canned tool calls stored in the case file.\n"
    "This checks the eval harness, the dispatch loop, the stubbed geocoder and the\n"
    "deterministic validators. It is NOT a measure of model quality."
)
LIVE_BANNER = (
    "PROVIDER: {provider} (live) - models: {models}\n"
    "Geocoding, job storage and the clock are stubbed; the model calls are real."
)


def load_dataset(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if data.get("version") != 1:
        raise SystemExit(f"{path}: unsupported dataset version {data.get('version')!r}")
    ids = [c["id"] for c in data["cases"]]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise SystemExit(f"{path}: duplicate case ids {sorted(dupes)}")
    return data


def _model_turn(step: Dict[str, Any]) -> ModelTurn:
    if "calls" in step:
        return ModelTurn(tool_calls=[
            ToolCall(name=c["tool"], args=dict(c.get("args") or {})) for c in step["calls"]
        ])
    return ModelTurn(text=step.get("text", ""))


def scripted_steps(turn: Dict[str, Any]) -> List[Any]:
    """
    Canned model output for one user turn:
      {calls: [...]}            one round of tool calls, then relay the results as text
      {text: "..."}             a text answer, no tool call
      {turns: [{calls}, ...]}   several model rounds in the same user turn (e.g. a retry
                                after a rejected call); ends with relay unless the last
                                round is {text}
    """
    script = turn.get("scripted")
    if script is None:
        raise ValueError(f"turn {turn.get('input')!r} has no 'scripted' reply")
    if "turns" in script:
        rounds = list(script["turns"])
        if not rounds or any("calls" not in r for r in rounds[:-1]):
            raise ValueError(f"turn {turn.get('input')!r}: 'turns' needs {{calls}} rounds, "
                             f"optionally ending with one {{text}}")
        steps: List[Any] = [_model_turn(r) for r in rounds]
        if "calls" in rounds[-1]:
            steps.append(relay_tool_results)
        return steps
    if "calls" in script:
        return [_model_turn(script), relay_tool_results]
    return [_model_turn(script)]


def run_case(case: Dict[str, Any], *, provider, models: List[str], gazetteer, sleep) -> CaseResult:
    turns = [*case.get("setup", []), case]
    if getattr(provider, "name", "") == "scripted":
        provider = ScriptedProvider([step for t in turns for step in scripted_steps(t)])

    agent = Agent(
        provider,
        models=models,
        job_repo=InMemoryJobRepository(),
        geocode=StubGeocoder(gazetteer),
        sessions=SessionStore(),
        clock=lambda: FIXED_NOW,
        sleep=sleep,
        data_agent=StubDataAgent(),
    )
    session_id = f"eval-{case['id']}"

    for setup in case.get("setup", []):
        res = agent.run(setup["input"], session_id)
        if res.error:  # the model never answered the setup turn: nothing to score
            return errored_case(case, res, during=f"setup turn {setup['input']!r}")
        if setup.get("expect_job", True) and res.job_id is None:
            return score_case(case, res, note=f"setup turn {setup['input']!r} created no job")

    result = agent.run(case["input"], session_id)
    return score_case(case, result)


def _fmt_rate(metric: Dict[str, Any]) -> str:
    if metric["rate"] is None:
        return "n/a"
    return f"{metric['correct']}/{metric['total']} ({metric['rate']:.1%})"


def print_table(results: List[CaseResult]) -> None:
    header = f"{'ID':<36} {'LANG':<4} {'EXPECTED':<31} {'CALLED':<31} {'ARGS':<5} {'JOB':<4} RESULT"
    print(header)
    print("-" * len(header))
    for r in results:
        expected = r.expected_tool or "(refuse)"
        called = ",".join(r.called_tools) or "(none)"
        args = "-" if r.args_ok is None else ("ok" if r.args_ok else "FAIL")
        job = "yes" if r.job_created else "no"
        verdict = r.outcome.upper().replace("_", " ")
        if r.outcome in ("error", "not_run"):
            called, args, job = "-", "-", "-"
        print(f"{r.id[:36]:<36} {r.lang:<4} {expected[:31]:<31} {called[:31]:<31} {args:<5} {job:<4} {verdict}")
        if r.outcome == "error":
            where = f", {r.note}" if r.note else ""
            print(f"    -> no model answer ({r.error_reason or r.error_kind}{where}): {one_line(r.error)}")
        elif r.outcome == "fail":
            bad = [k for k, ok in r.arg_checks.items() if not ok]
            detail = f"args failed: {bad} effective={r.effective_args}" if bad else ""
            print(f"    -> {one_line(r.note or detail or r.tool_errors or r.reply_text)}")
        if r.guard_blocked:
            print(f"    -> needs-user guard refused {r.guard_blocked} job-creating call(s)")


def print_summary(summary: Dict[str, Any], stop_reason: Optional[str] = None) -> None:
    print()
    print_completeness(summary, stop_reason)
    print(f"cases passed            {summary['passed']}/{summary['answered']} answered")
    print(f"tool-selection accuracy {_fmt_rate(summary['tool_selection_accuracy'])}")
    print(f"argument accuracy       {_fmt_rate(summary['argument_accuracy'])}")
    print(f"correct-refusal rate    {_fmt_rate(summary['correct_refusal_rate'])}")
    print(f"routing accuracy        {_fmt_rate(summary['routing_accuracy'])}  (query_database vs launch_analysis)")
    print(f"false refusals          {summary['false_refusals']}")
    print(f"setup failures          {summary['setup_failures']}  (a setup turn was answered but created no job)")
    print(f"needs-user guard fired  {summary['guard_blocked_cases']} case(s)  "
          f"(a job-creating call was refused after a limit only the user can lift)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nreport: {path}")


def _run_meta(args, models, banner, guard: Optional[ProviderGuard], summary, stop_reason, started) -> Dict[str, Any]:
    live = args.provider != "scripted"
    warning = None
    if not summary["complete"]:
        warning = (
            f"INCOMPLETE RUN: only {summary['answered']} of {summary['cases']} cases have a scored answer "
            f"({summary['errored']} errored, {summary['not_run']} not run). The rates cover the answered "
            f"cases only and are NOT a result for the dataset."
        )
    return {
        "provider": args.provider,
        "models": models,
        "measures_model_quality": live,
        "complete": summary["complete"],
        "counts": {k: summary[k] for k in ("cases", "answered", "errored", "not_run")},
        "warning": warning,
        "stopped_early": stop_reason,
        "model_requests_sent": guard.requests_sent if guard is not None else None,
        "daily_quota_exhausted": list(guard.exhausted) if guard is not None else [],
        "note": banner,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def run_tools_suite(args, provider, models, sleep, banner, guard: Optional[ProviderGuard] = None) -> Dict[str, Any]:
    data = load_dataset(args.cases)
    cases = [c for c in data["cases"] if not args.only or c["id"] in args.only]
    print(banner)
    print(f"suite: tools - dataset: {args.cases.name} (version {data['version']}, {len(cases)} cases)\n")

    started = datetime.now(timezone.utc)
    results, stop_reason = run_cases(
        cases,
        lambda c: run_case(c, provider=provider, models=models, gazetteer=data["gazetteer"], sleep=sleep),
        not_run=not_run_case,
        guard=guard,
        case_retries=args.case_retries if guard is not None else 0,
        pause=sleep,
    )
    summary = summarize(results)
    print_table(results)
    print_summary(summary, stop_reason)

    _write_report(args.report or DEFAULT_REPORT_DIR / f"{args.provider}.json", {
        "schema": "agent-tool-call-eval-report/v2",
        **_run_meta(args, models, banner, guard, summary, stop_reason, started),
        "dataset": {"path": args.cases.name, "version": data["version"], "sha256": _sha256(args.cases)},
        "summary": summary,
        "cases": [r.to_dict() for r in results],
    })
    return summary


def run_sql_suite_cli(args, provider, models, banner, guard: Optional[ProviderGuard] = None,
                      sleep=time.sleep) -> Optional[Dict[str, Any]]:
    from evals import sql_suite
    from evals.db import create_scratch_db, drop_scratch_db, fingerprint

    admin_url = args.admin_url or os.environ.get("TEST_DATABASE_URL")
    if not admin_url:
        print("--suite sql needs --admin-url or TEST_DATABASE_URL (a PostGIS superuser URL).", file=sys.stderr)
        return None
    data = sql_suite.load_sql_dataset(args.sql_cases)
    cases = [c for c in data["cases"] if not args.only or c["id"] in args.only]
    if args.provider == "scripted":
        banner = sql_suite.SCRIPTED_SQL_BANNER
    print(banner)
    print(f"suite: sql - dataset: {args.sql_cases.name} (version {data['version']}, {len(cases)} cases), "
          f"SYNTHETIC seed\n")

    started = datetime.now(timezone.utc)
    if guard is not None and guard.all_exhausted:
        # Nothing can be answered: do not even build the scratch database.
        stop_reason = guard.stop_reason()
        results = [sql_suite.not_run_sql_case(c, stop_reason) for c in cases]
    else:
        db = create_scratch_db(admin_url, prefix="agent_eval")
        try:
            results, stop_reason = sql_suite.run_sql_suite(
                cases, agent_dsn=db.agent_dsn, admin_dsn=db.admin_dsn,
                provider=provider, models=models, fingerprint=fingerprint,
                guard=guard, case_retries=args.case_retries if guard is not None else 0, pause=sleep,
            )
        finally:
            drop_scratch_db(admin_url, db)
    summary = sql_suite.summarize_sql(results)
    sql_suite.print_sql_table(results)
    sql_suite.print_sql_summary(summary, stop_reason)

    _write_report(args.sql_report or DEFAULT_REPORT_DIR / f"{args.provider}-sql.json", {
        "schema": "data-agent-sql-eval-report/v2",
        **_run_meta(args, models, banner, guard, summary, stop_reason, started),
        "dataset": {"path": args.sql_cases.name, "version": data["version"], "sha256": _sha256(args.sql_cases)},
        "seed": {"path": "fixtures/synthetic_seed.sql", "sha256": _sha256(HERE / "fixtures" / "synthetic_seed.sql")},
        "summary": summary,
        "cases": [r.to_dict() for r in results],
    })
    return summary


def make_live_provider(args):
    """(provider, models) for --provider gemini. Separate so tests can swap in a fake."""
    from eo_brain.agent.providers.gemini import GeminiProvider, model_chain_from_env

    return GeminiProvider(), (args.model or model_chain_from_env())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["scripted", "gemini"], required=True)
    ap.add_argument("--suite", choices=["tools", "sql", "all"], default="tools")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="tool-suite dataset")
    ap.add_argument("--sql-cases", type=Path, default=DEFAULT_SQL_CASES, help="sql-suite dataset")
    ap.add_argument("--admin-url", help="PostGIS superuser URL for the sql suite (default: $TEST_DATABASE_URL)")
    ap.add_argument("--model", action="append", help="model id (repeat for a fallback chain); gemini only")
    ap.add_argument("--only", action="append", default=[], help="run only these case ids")
    ap.add_argument("--case-retries", type=int, default=DEFAULT_CASE_RETRIES,
                    help="live only: re-runs of a case that errored on a transient provider failure "
                         f"(5xx, timeout, per-minute 429), after a pause (default {DEFAULT_CASE_RETRIES}; 0 = off)")
    ap.add_argument("--report", type=Path, help="tool-suite JSON report (default: evals/reports/<provider>.json)")
    ap.add_argument("--sql-report", type=Path,
                    help="sql-suite JSON report (default: evals/reports/<provider>-sql.json)")
    ap.add_argument("--verbose", action="store_true", help="show agent logs")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.CRITICAL)

    guard: Optional[ProviderGuard] = None
    if args.provider == "scripted":
        provider: Any = ScriptedProvider([])
        models = [SCRIPTED_MODEL]
        sleep = lambda _s: None  # noqa: E731 - no real waiting offline
        banner = SCRIPTED_BANNER
    else:
        if not os.environ.get("GEMINI_API_KEY"):
            print("--provider gemini needs GEMINI_API_KEY in the environment.", file=sys.stderr)
            return EXIT_USAGE
        live_provider, models = make_live_provider(args)
        guard = provider = ProviderGuard(live_provider, models)
        sleep = time.sleep
        banner = LIVE_BANNER.format(provider=args.provider, models=", ".join(models))

    summaries: List[Dict[str, Any]] = []
    if args.suite in ("tools", "all"):
        summaries.append(run_tools_suite(args, provider, models, sleep, banner, guard))
    if args.suite in ("sql", "all"):
        if args.suite == "all":
            print("\n" + "=" * 100 + "\n")
        summary = run_sql_suite_cli(args, provider, models, banner, guard, sleep)
        if summary is None:
            return EXIT_USAGE
        summaries.append(summary)

    if args.provider == "scripted":
        # Offline mode is a regression gate: every case must pass.
        return EXIT_OK if all(s["passed"] == s["cases"] for s in summaries) else EXIT_REGRESSION
    if not all(s["complete"] for s in summaries):
        print(f"\nexit {EXIT_INCOMPLETE}: the live run is INCOMPLETE - see the warning in the report(s).")
        return EXIT_INCOMPLETE
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
