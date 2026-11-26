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
from evals.scoring import CaseResult, score_case, summarize
from evals.stubs import FIXED_NOW, InMemoryJobRepository, StubDataAgent, StubGeocoder

HERE = Path(__file__).resolve().parent
DEFAULT_CASES = HERE / "cases.v1.yaml"
DEFAULT_SQL_CASES = HERE / "sql_cases.v1.yaml"
DEFAULT_REPORT_DIR = HERE / "reports"
SCRIPTED_MODEL = "scripted-replay"

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


def scripted_steps(turn: Dict[str, Any]) -> List[Any]:
    """Canned model output for one user turn."""
    script = turn.get("scripted")
    if script is None:
        raise ValueError(f"turn {turn.get('input')!r} has no 'scripted' reply")
    if "calls" in script:
        calls = [ToolCall(name=c["tool"], args=dict(c.get("args") or {})) for c in script["calls"]]
        return [ModelTurn(tool_calls=calls), relay_tool_results]
    return [ModelTurn(text=script.get("text", ""))]


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
        verdict = "PASS" if r.passed else "FAIL"
        print(f"{r.id[:36]:<36} {r.lang:<4} {expected[:31]:<31} {called[:31]:<31} {args:<5} {job:<4} {verdict}")
        if not r.passed:
            bad = [k for k, ok in r.arg_checks.items() if not ok]
            detail = f"args failed: {bad} effective={r.effective_args}" if bad else ""
            print(f"    -> {r.note or detail or r.tool_errors or r.reply_text[:120]}")


def print_summary(summary: Dict[str, Any]) -> None:
    print()
    print(f"cases passed            {summary['passed']}/{summary['cases']}")
    print(f"tool-selection accuracy {_fmt_rate(summary['tool_selection_accuracy'])}")
    print(f"argument accuracy       {_fmt_rate(summary['argument_accuracy'])}")
    print(f"correct-refusal rate    {_fmt_rate(summary['correct_refusal_rate'])}")
    print(f"routing accuracy        {_fmt_rate(summary['routing_accuracy'])}  (query_database vs launch_analysis)")
    print(f"false refusals          {summary['false_refusals']}")
    print(f"errored cases           {summary['errored']}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_report(path: Path, report: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\nreport: {path}")


def run_tools_suite(args, provider, models, sleep, banner) -> bool:
    data = load_dataset(args.cases)
    cases = [c for c in data["cases"] if not args.only or c["id"] in args.only]
    print(banner)
    print(f"suite: tools - dataset: {args.cases.name} (version {data['version']}, {len(cases)} cases)\n")

    started = datetime.now(timezone.utc)
    results = [
        run_case(c, provider=provider, models=models, gazetteer=data["gazetteer"], sleep=sleep)
        for c in cases
    ]
    summary = summarize(results)
    print_table(results)
    print_summary(summary)

    _write_report(args.report or DEFAULT_REPORT_DIR / f"{args.provider}.json", {
        "schema": "agent-tool-call-eval-report/v1",
        "provider": args.provider,
        "models": models,
        "measures_model_quality": args.provider != "scripted",
        "note": banner,
        "dataset": {"path": args.cases.name, "version": data["version"], "sha256": _sha256(args.cases)},
        "started_at": started.isoformat(),
        "summary": summary,
        "cases": [r.to_dict() for r in results],
    })
    return summary["passed"] == summary["cases"]


def run_sql_suite_cli(args, provider, models, banner) -> Optional[bool]:
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
    db = create_scratch_db(admin_url, prefix="agent_eval")
    try:
        results = sql_suite.run_sql_suite(
            cases, agent_dsn=db.agent_dsn, admin_dsn=db.admin_dsn,
            provider=provider, models=models, fingerprint=fingerprint,
        )
    finally:
        drop_scratch_db(admin_url, db)
    summary = sql_suite.summarize_sql(results)
    sql_suite.print_sql_table(results)
    sql_suite.print_sql_summary(summary)

    _write_report(args.sql_report or DEFAULT_REPORT_DIR / f"{args.provider}-sql.json", {
        "schema": "data-agent-sql-eval-report/v1",
        "provider": args.provider,
        "models": models,
        "measures_model_quality": args.provider != "scripted",
        "note": banner,
        "dataset": {"path": args.sql_cases.name, "version": data["version"], "sha256": _sha256(args.sql_cases)},
        "seed": {"path": "fixtures/synthetic_seed.sql", "sha256": _sha256(HERE / "fixtures" / "synthetic_seed.sql")},
        "started_at": started.isoformat(),
        "summary": summary,
        "cases": [r.to_dict() for r in results],
    })
    return summary["passed"] == summary["cases"]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=["scripted", "gemini"], required=True)
    ap.add_argument("--suite", choices=["tools", "sql", "all"], default="tools")
    ap.add_argument("--cases", type=Path, default=DEFAULT_CASES, help="tool-suite dataset")
    ap.add_argument("--sql-cases", type=Path, default=DEFAULT_SQL_CASES, help="sql-suite dataset")
    ap.add_argument("--admin-url", help="PostGIS superuser URL for the sql suite (default: $TEST_DATABASE_URL)")
    ap.add_argument("--model", action="append", help="model id (repeat for a fallback chain); gemini only")
    ap.add_argument("--only", action="append", default=[], help="run only these case ids")
    ap.add_argument("--report", type=Path, help="tool-suite JSON report (default: evals/reports/<provider>.json)")
    ap.add_argument("--sql-report", type=Path,
                    help="sql-suite JSON report (default: evals/reports/<provider>-sql.json)")
    ap.add_argument("--verbose", action="store_true", help="show agent logs")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.CRITICAL)

    if args.provider == "scripted":
        provider: Any = ScriptedProvider([])
        models = [SCRIPTED_MODEL]
        sleep = lambda _s: None  # noqa: E731 - no real waiting offline
        banner = SCRIPTED_BANNER
    else:
        if not os.environ.get("GEMINI_API_KEY"):
            print("--provider gemini needs GEMINI_API_KEY in the environment.", file=sys.stderr)
            return 2
        from eo_brain.agent.providers.gemini import GeminiProvider, model_chain_from_env

        provider = GeminiProvider()
        models = args.model or model_chain_from_env()
        sleep = time.sleep
        banner = LIVE_BANNER.format(provider=args.provider, models=", ".join(models))

    all_passed = True
    if args.suite in ("tools", "all"):
        all_passed &= run_tools_suite(args, provider, models, sleep, banner)
    if args.suite in ("sql", "all"):
        if args.suite == "all":
            print("\n" + "=" * 100 + "\n")
        ok = run_sql_suite_cli(args, provider, models, banner)
        if ok is None:
            return 2
        all_passed &= ok

    # Offline mode is a regression gate: every case must pass.
    if args.provider == "scripted" and not all_passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
