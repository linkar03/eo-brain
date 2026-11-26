"""
Tests for the data-agent SQL eval suite (evals/sql_suite.py).

The offline tests check the dataset and the scoring rules. The tests marked
``db`` run the suite against a throwaway database and are skipped unless
TEST_DATABASE_URL is set (see tests/test_db_agent.py).
"""

import copy
import os

import pytest

from eo_brain.agent.data_agent import DataAgentResult, SQLAttempt
from eo_brain.agent.sql_gate import SQLGate
from evals import sql_suite
from evals.sql_suite import (
    geometry_leaks,
    load_sql_dataset,
    normalize_rows,
    score_sql_case,
    scripted_sql_steps,
    summarize_sql,
)

ADMIN_URL = os.environ.get("TEST_DATABASE_URL", "").strip()
VIEWS = ["analyses", "alert_areas", "analysis_severity_summary"]


@pytest.fixture(scope="module")
def dataset():
    return load_sql_dataset(sql_suite.__file__.replace("sql_suite.py", "sql_cases.v1.yaml"))


# ----------------------------------------------------------------- dataset


def test_dataset_covers_both_languages_and_both_kinds(dataset):
    cases = dataset["cases"]
    assert {c["lang"] for c in cases} == {"it", "en"}
    kinds = [c["expect"]["type"] for c in cases]
    assert kinds.count("sql_result") >= 10
    assert kinds.count("blocked") >= 8


def test_every_gold_sql_passes_the_gate(dataset):
    gate = SQLGate(VIEWS, max_limit=100)
    for case in dataset["cases"]:
        if case["expect"]["type"] == "sql_result":
            decision = gate.check(case["expect"]["gold_sql"])
            assert decision.ok, (case["id"], decision.reason)


def test_adversarial_scripts_are_rejected_by_the_gate_except_marked_safe_answers(dataset):
    """Each blocked case's scripted SQL must be stopped by the gate, unless the
    case allows a safe answer (then only the final attempt may pass)."""
    gate = SQLGate(VIEWS, max_limit=100)
    for case in dataset["cases"]:
        if case["expect"]["type"] != "blocked":
            continue
        sqls = [t["sql"] for t in case["scripted"] if "sql" in t]
        verdicts = [gate.check(s).ok for s in sqls]
        if case["expect"].get("allow_safe_answer"):
            assert not any(verdicts[:-1]), case["id"]
        else:
            assert not any(verdicts), (case["id"], verdicts)


def test_scripted_steps_map_to_tool_calls(dataset):
    case = next(c for c in dataset["cases"] if c["id"] == "it-block-public-jobs")
    steps = scripted_sql_steps(case)
    assert [s.tool_calls[0].name for s in steps] == ["run_sql", "run_sql", "refuse"]


def test_dataset_loader_rejects_unknown_kind(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\ncases:\n  - {id: x, question: q, expect: {type: maybe}}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_sql_dataset(bad)


def test_dataset_loader_requires_gold_sql(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\ncases:\n  - {id: x, question: q, expect: {type: sql_result}}\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_sql_dataset(bad)


# ----------------------------------------------------------------- scoring


def test_normalize_rows_ignores_order_and_numeric_type():
    a = [["x", 12], ["y", 6.0000001]]
    b = [["y", 6.0], ["x", 12.0]]
    assert normalize_rows(a, ordered=False) == normalize_rows(b, ordered=False)
    assert normalize_rows(a, ordered=True) != normalize_rows(b, ordered=True)


def test_normalize_rows_keeps_real_differences():
    assert normalize_rows([[12.0]], ordered=False) != normalize_rows([[12.5]], ordered=False)
    assert normalize_rows([[1, 2]], ordered=False) != normalize_rows([[1]], ordered=False)
    assert normalize_rows([["1"]], ordered=False) != normalize_rows([[1]], ordered=False)


@pytest.mark.parametrize("value", [
    "0101000020E6100000000000000000F03F000000000000F03F",
    "0103000020E61000000100000005000000",
    "POLYGON((0 0,1 0,1 1,0 0))",
    "SRID=4326;MULTIPOLYGON(((0 0,1 0,1 1,0 0)))",
    '{"type": "Point", "coordinates": [1, 2]}',
])
def test_geometry_leak_detector_flags_geometry(value):
    assert geometry_leaks([[value]])


@pytest.mark.parametrize("value", [
    "00000000-0000-4000-a000-000000000001", "<geometry omitted>", "Testville North (synthetic)",
    "No satellite data found.", 12.0, None,
])
def test_geometry_leak_detector_ignores_normal_values(value):
    assert not geometry_leaks([[value]])


def _case(kind, **expect):
    return {"id": "c", "lang": "en", "expect": {"type": kind, **expect}}


def _result(status, rows=None, attempts=None):
    return DataAgentResult(status=status, question="q", rows=rows or [], sql="SELECT 1" if status == "ok" else None,
                           attempts=attempts or [])


def test_sql_result_passes_only_on_matching_rows():
    case = _case("sql_result", gold_sql="x")
    assert score_sql_case(case, _result("ok", [[12.0]]), gold_rows=[[12]], db_unchanged=True).passed
    assert not score_sql_case(case, _result("ok", [[8.0]]), gold_rows=[[12]], db_unchanged=True).passed
    assert not score_sql_case(case, _result("failed"), gold_rows=[[12]], db_unchanged=True).passed
    assert not score_sql_case(case, _result("refused"), gold_rows=[[12]], db_unchanged=True).passed


def test_sql_result_fails_when_gold_is_broken_or_db_changed():
    case = _case("sql_result", gold_sql="x")
    assert not score_sql_case(case, _result("ok", [[1]]), gold_rows=None, db_unchanged=True).passed
    assert not score_sql_case(case, _result("ok", [[1]]), gold_rows=[[1]], db_unchanged=False).passed
    assert not score_sql_case(case, _result("ok", [[1]]), gold_rows=[[1]], db_unchanged=True, note="x").passed


def test_blocked_case_scoring():
    strict = _case("blocked")
    lenient = _case("blocked", allow_safe_answer=True)
    assert score_sql_case(strict, _result("refused"), gold_rows=None, db_unchanged=True).blocked_by == "refusal"
    assert score_sql_case(strict, _result("failed"), gold_rows=None, db_unchanged=True).blocked_by == "gave_up"
    assert not score_sql_case(strict, _result("ok", [[1]]), gold_rows=None, db_unchanged=True).passed
    safe = score_sql_case(lenient, _result("ok", [[1]]), gold_rows=None, db_unchanged=True)
    assert safe.passed and safe.blocked_by == "safe_answer"


def test_geometry_leak_fails_even_an_allowed_safe_answer():
    lenient = _case("blocked", allow_safe_answer=True)
    leaked = _result("ok", [["0101000020E6100000000000000000F03F000000000000F03F"]])
    assert not score_sql_case(lenient, leaked, gold_rows=None, db_unchanged=True).passed


def test_summary_counts():
    results = [
        score_sql_case(_case("sql_result", gold_sql="x"),
                       _result("ok", [[1]], [SQLAttempt("a", "e", "gate"), SQLAttempt("b")]),
                       gold_rows=[[1]], db_unchanged=True),
        score_sql_case(_case("sql_result", gold_sql="x"), _result("ok", [[2]]), gold_rows=[[1]], db_unchanged=True),
        score_sql_case(_case("blocked"), _result("failed", attempts=[SQLAttempt("d", "e", "gate")] * 3),
                       gold_rows=None, db_unchanged=True),
    ]
    s = summarize_sql(results)
    assert s["passed"] == 2
    assert s["execution_accuracy"] == {"correct": 1, "total": 2, "rate": 0.5}
    assert s["blocked_rate"]["correct"] == 1
    assert s["recovered_after_retry"] == {"correct": 1, "total": 1, "rate": 1.0}
    assert s["gate_rejections"] == 4
    assert s["db_unchanged"] is True


# ----------------------------------------------------------------- against a database


@pytest.mark.db
@pytest.mark.skipif(not ADMIN_URL, reason="TEST_DATABASE_URL not set (PostGIS superuser URL)")
def test_scripted_sql_suite_passes_and_a_wrong_answer_is_caught(dataset):
    from eo_brain.agent.providers.scripted import ScriptedProvider
    from evals.db import create_scratch_db, drop_scratch_db, fingerprint

    cases = copy.deepcopy(dataset["cases"])
    wrong = copy.deepcopy(next(c for c in cases if c["id"] == "en-sql-alert-ha-latest-flood"))
    wrong["id"] = "wrong-answer-control"
    # Plausible but wrong: oldest flood instead of latest (8 ha, not 12 ha).
    wrong["scripted"] = [{"sql": wrong["scripted"][0]["sql"].replace("DESC", "ASC")}]
    cases.append(wrong)

    db = create_scratch_db(ADMIN_URL, prefix="agent_test_eval")
    try:
        results = sql_suite.run_sql_suite(
            cases, agent_dsn=db.agent_dsn, admin_dsn=db.admin_dsn,
            provider=ScriptedProvider([]), models=["scripted-replay"], fingerprint=fingerprint,
        )
    finally:
        drop_scratch_db(ADMIN_URL, db)

    by_id = {r.id: r for r in results}
    control = by_id.pop("wrong-answer-control")
    assert control.status == "ok" and control.rows == [[8.0]] and control.gold_rows == [[12.0]]
    assert not control.passed
    failed = [r.id for r in by_id.values() if not r.passed]
    assert failed == []
    assert all(r.db_unchanged for r in results)
    assert sum(len(r.leaks) for r in results) == 0


def test_cli_sql_suite_without_a_database_exits_2(monkeypatch, capsys):
    from evals import run

    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    assert run.main(["--provider", "scripted", "--suite", "sql"]) == 2
    assert "TEST_DATABASE_URL" in capsys.readouterr().err
