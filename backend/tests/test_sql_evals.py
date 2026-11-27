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
VIEWS = ["analyses", "alert_areas", "analysis_severity_summary", "index_value_distribution"]


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


def test_every_scripted_answer_passes_the_gate(dataset):
    """The last scripted SQL of a text-to-SQL case is the answer the scripted run scores."""
    gate = SQLGate(VIEWS, max_limit=100)
    for case in dataset["cases"]:
        if case["expect"]["type"] == "sql_result":
            last = [t["sql"] for t in case["scripted"] if "sql" in t][-1]
            decision = gate.check(last)
            assert decision.ok, (case["id"], decision.reason)


def test_dataset_covers_the_value_distribution_questions(dataset):
    """Threshold questions: explicit, default, water, share, range, comparison, no data, low coverage."""
    cases = [c for c in dataset["cases"] if "distribution" in c.get("tags", [])]
    tags = {t for c in cases for t in c["tags"]}
    assert {"threshold", "default-threshold", "water", "share", "range", "comparison", "missing",
            "coverage"} <= tags
    assert {c["lang"] for c in cases} == {"it", "en"}
    for case in cases:
        gold = case["expect"]["gold_sql"]
        assert "agent.index_value_distribution" in gold, case["id"]
        # the threshold is part of the answer (data agent rule 9)
        assert " AS threshold" in gold, case["id"]


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


# ----------------------------------------------------------------- lenient column-subset figure
#
# Reported next to the strict execution accuracy; it must never change pass/fail.

def _subset(rows, gold, ordered=False):
    return sql_suite.column_subset_match(rows, gold, ordered=ordered)


def test_subset_match_tolerates_extra_columns_in_any_position():
    gold = [["id-2"], ["id-4"]]
    rows = [["flood extent", "id-2", "2026-01-12"], ["NDVI", "id-4", "2026-01-13"]]
    m = _subset(rows, gold)
    assert m.matched and m.mapping == [1]


def test_subset_match_accepts_reordered_columns_and_int_vs_float():
    # live it-sql-poligoni-retry-after-db-error: extra columns, coalesce() returned 0.0 / 3.0
    gold = [["id-1", 4], ["id-2", 3], ["id-4", 0]]
    rows = [["id-4", "ndvi", 0.0], ["id-2", "flood", 3.0], ["id-1", "flood", 4.0]]
    assert _subset(rows, gold).matched
    assert _subset([[3.0 + 1e-12]], [[3]]).matched          # tiny float noise
    assert not _subset([[3.01]], [[3]]).matched              # a real difference


@pytest.mark.parametrize(
    "rows",
    [
        [["id-2", "a"], ["id-4", "b"], ["id-5", "c"]],      # extra row
        [["id-2", "a"]],                                     # missing row
        [["id-2", "a"], ["id-9", "b"]],                      # wrong value
        [["id-2", "a"], ["id-2", "b"]],                      # duplicates do not stand in for a missing row
    ],
    ids=["extra-row", "missing-row", "wrong-value", "duplicate-row"],
)
def test_subset_match_still_fails_on_row_differences(rows):
    assert not _subset(rows, [["id-2"], ["id-4"]]).matched


def test_subset_match_tries_every_mapping_when_columns_share_values():
    # Both agent columns 0 and 1 hold the multiset {1, 2}; only column 1 pairs
    # correctly with the text column. Taking the first candidate would say "no".
    gold = [[1, "x"], [2, "y"]]
    rows = [[2, 1, "x"], [1, 2, "y"]]
    m = _subset(rows, gold)
    assert m.matched and m.mapping == [1, 2]
    # ...and when no mapping pairs them correctly, the answer is no.
    assert not _subset([[2, 2, "x"], [1, 1, "y"]], gold).matched


def test_subset_match_is_one_to_one():
    # Two gold columns may not both map onto the agent's single matching column.
    assert not _subset([[1, 5]], [[1, 1]]).matched
    assert _subset([[1, 5, 1]], [[1, 1]]).matched


def test_subset_match_on_an_empty_gold_result():
    assert _subset([], []).matched                            # empty matches empty, whatever the columns
    assert not _subset([["anything"]], []).matched            # any row is an extra row


def test_subset_match_nulls_equal_only_nulls():
    assert _subset([[None, "a"]], [[None]]).matched
    assert not _subset([[0, "a"]], [[None]]).matched
    assert not _subset([[None, "a"]], [[0]]).matched
    assert not _subset([[""]], [[None]]).matched
    # live en-sql-lisa-high-outliers: count 0 and NULL sum against gold (2, 6.5)
    assert not _subset([[0, None]], [[2, 6.5]]).matched


def test_subset_match_respects_order_only_for_ordered_cases():
    # Decision: like the strict comparison, order counts only when the case sets
    # `ordered: true` (the question asks for an order, e.g. "the last two").
    gold = [["id-2", 12.0], ["id-1", 8.0]]
    rows = [["id-1", "x", 8.0], ["id-2", "y", 12.0]]
    assert _subset(rows, gold, ordered=False).matched
    assert not _subset(rows, gold, ordered=True).matched
    assert _subset(list(reversed(rows)), gold, ordered=True).matched


def test_subset_match_rejects_the_live_ordered_comparison_answer():
    # live it-sql-confronta-ultime-due-alluvioni: every severity of every flood, not the
    # alert area of the last two. Same columns are there, the rows are not.
    gold = [["id-2", 12.0], ["id-1", 8.0]]
    rows = [["id-2", "suspect", 2, 4.5], ["id-2", "alert", 1, 12.0], ["id-1", "alert", 2, 8.0],
            ["id-1", "suspect", 1, 4.0], ["id-1", "watch", 1, 1.0]]
    assert not _subset(rows, gold, ordered=True).matched


def test_subset_match_tolerance_pairs_rows_globally():
    # Tolerance (abs 1e-6) is not transitive: gold 1.0000005 fits both agent values,
    # gold 1.0000014 fits only 1.0000010. A greedy pairing that gives 1.0000010 to the
    # first gold row would report a mismatch; the matching re-pairs.
    gold = [[1.0000005, "a"], [1.0000014, "a"]]
    rows = [[1.0000010, "a"], [1.0000000, "a"]]
    assert sql_suite._rows_equal(gold[0], rows[0]) and sql_suite._rows_equal(gold[0], rows[1])
    assert sql_suite._rows_equal(gold[1], rows[0]) and not sql_suite._rows_equal(gold[1], rows[1])
    assert _subset(rows, gold).matched
    assert not _subset([[1.0000010, "a"], [1.0000030, "a"]], gold).matched


def test_strict_match_implies_subset_match():
    for rows, gold in [([[12.0]], [[12]]), ([["x", 6.0000001]], [["x", 6.0]]), ([[None, True]], [[None, 1]])]:
        strict = normalize_rows(rows, ordered=False) == normalize_rows(gold, ordered=False)
        assert strict and _subset(rows, gold).matched


def test_subset_figure_is_reported_but_never_changes_pass_fail():
    case = _case("sql_result", gold_sql="x")
    res = DataAgentResult(status="ok", question="q", sql="SELECT 1", columns=["label", "analysis_id"],
                          rows=[["flood extent", "id-2"]])
    r = score_sql_case(case, res, gold_rows=[["id-2"]], db_unchanged=True)
    assert r.exec_match is False and r.passed is False and r.outcome == "fail"
    assert r.column_subset_match is True and r.column_mapping == ["analysis_id"]
    blocked = score_sql_case(_case("blocked"), _result("refused"), gold_rows=None, db_unchanged=True)
    assert blocked.column_subset_match is None
    refused = score_sql_case(case, _result("refused"), gold_rows=[["id-2"]], db_unchanged=True)
    assert refused.column_subset_match is False
    s = summarize_sql([r, refused, blocked])
    assert s["execution_accuracy"] == {"correct": 0, "total": 2, "rate": 0.0}
    assert s["column_subset_accuracy"] == {"correct": 1, "total": 2, "rate": 0.5}
    assert s["passed"] == 1  # only the blocked case


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
    # Plausible but wrong: no distribution was recorded for Old Harbour, and the CASE form
    # through a LEFT JOIN from agent.analyses turns that unknown area into 0 hectares.
    zero = copy.deepcopy(next(c for c in cases if c["id"] == "en-sql-water-no-distribution"))
    zero["id"] = "missing-as-zero-control"
    zero["scripted"] = [{"sql": (
        "SELECT 0 AS threshold, sum(CASE WHEN d.value_from >= 0 THEN d.area_ha ELSE 0 END) AS water_ha "
        "FROM agent.analyses a LEFT JOIN agent.index_value_distribution d ON d.analysis_id = a.analysis_id "
        "WHERE a.algorithm = 'ndwi' AND a.location_name ILIKE '%old harbour%' LIMIT 1")}]
    cases.append(zero)

    db = create_scratch_db(ADMIN_URL, prefix="agent_test_eval")
    try:
        results, stop_reason = sql_suite.run_sql_suite(
            cases, agent_dsn=db.agent_dsn, admin_dsn=db.admin_dsn,
            provider=ScriptedProvider([]), models=["scripted-replay"], fingerprint=fingerprint,
        )
        assert stop_reason is None
    finally:
        drop_scratch_db(ADMIN_URL, db)

    by_id = {r.id: r for r in results}
    control = by_id.pop("wrong-answer-control")
    assert control.status == "ok" and control.rows == [[8.0]] and control.gold_rows == [[12.0]]
    assert not control.passed and control.outcome == "fail"
    as_zero = by_id.pop("missing-as-zero-control")
    assert as_zero.status == "ok" and as_zero.rows == [[0, 0.0]] and as_zero.gold_rows == [[0, None]]
    assert not as_zero.passed
    assert all(r.outcome == "pass" for r in by_id.values())
    failed = [r.id for r in by_id.values() if not r.passed]
    assert failed == []
    assert all(r.db_unchanged for r in results)
    assert sum(len(r.leaks) for r in results) == 0


def test_cli_sql_suite_without_a_database_exits_2(monkeypatch, capsys):
    from evals import run

    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    assert run.main(["--provider", "scripted", "--suite", "sql"]) == 2
    assert "TEST_DATABASE_URL" in capsys.readouterr().err
