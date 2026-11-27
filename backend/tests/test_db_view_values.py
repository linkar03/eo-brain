"""
Database-backed check of the closed-set columns of the agent views.

The data agent learns the views from their catalog COMMENTs (sql_exec.load_schema_doc).
A column with a fixed set of values must list them as "Values: ..." (see the header of
eo_brain/db/agent_views.sql), exactly as the view returns them, so the model filters
with quadrant_label = 'HL' instead of guessing from an explanation.

The expected values are NOT a second hand-written list. They come from what decides
what the view can produce:

  * the view's own CASE mappings, read back with pg_get_viewdef (labels);
  * the job_status enum in the database (status);
  * the Python constants the writers use: the algorithm registry (jobs are created only
    with its keys, by the agent and by POST /analyze), Severity and MoranQuadrant
    (eo_brain.algorithms.base), and DISCRETE_OUTPUT_ALGORITHMS (eo_brain/api/main.py,
    the algorithms whose results are vectorised into alert polygons).

The view's CASE mappings are themselves a copy of Severity / MoranQuadrant / the
registry, so the first test pins those copies to the Python constants.

Needs TEST_DATABASE_URL (PostGIS superuser URL); skipped when unset.
"""

import ast
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import psycopg2
import pytest
import sqlglot
from sqlglot import exp

import eo_brain
from eo_brain.algorithms.base import ALGORITHMS, MoranQuadrant, Severity
from eo_brain.core.coverage import LOW_COVERAGE_PCT
from eo_brain.core.scenes import DEFAULT_SELECTION
from evals.db import create_scratch_db, drop_scratch_db

ADMIN_URL = os.environ.get("TEST_DATABASE_URL", "").strip()

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not ADMIN_URL, reason="TEST_DATABASE_URL not set (PostGIS superuser URL)"),
]

VIEWS = ("analyses", "alert_areas", "analysis_severity_summary")
API_MAIN = Path(eo_brain.__file__).resolve().parent / "api" / "main.py"

_TOKEN = r"(?:'[^']*'|-?\d+)"
_VALUES_RE = re.compile(rf"\bValues: ({_TOKEN}(?:, {_TOKEN})*)")


# ---------------------------------------------------------------- helpers

def documented_values(comment: str) -> Optional[Set[Any]]:
    """The set after "Values:" (quoted -> str, bare -> int), or None if there is no list."""
    found = _VALUES_RE.findall(comment or "")
    if len(found) != 1:
        return None
    return {t[1:-1] if t.startswith("'") else int(t) for t in re.findall(_TOKEN, found[0])}


def problems(comment: str, expected: Set[Any]) -> List[str]:
    documented = documented_values(comment)
    if documented is None:
        return [f"no single 'Values: ...' list in {comment!r}"]
    out = []
    if expected - documented:
        out.append(f"missing {sorted(expected - documented, key=repr)}")
    if documented - expected:
        out.append(f"listed but never produced {sorted(documented - expected, key=repr)}")
    return out


def discrete_output_algorithms() -> Set[str]:
    """
    DISCRETE_OUTPUT_ALGORITHMS from eo_brain/api/main.py, read with ast instead of an
    import: the API module needs the full raster stack, which CI does not install.
    Fails loudly if the constant stops being a literal assignment there.
    """
    tree = ast.parse(API_MAIN.read_text(encoding="utf-8-sig"))  # main.py starts with a BOM
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "DISCRETE_OUTPUT_ALGORITHMS" for t in node.targets
        ):
            return set(ast.literal_eval(node.value))
    raise AssertionError(f"DISCRETE_OUTPUT_ALGORITHMS is no longer a literal in {API_MAIN}")


def _literal(node) -> Any:
    while isinstance(node, (exp.Cast, exp.Paren)):
        node = node.this
    if isinstance(node, exp.Literal):
        return node.this if node.is_string else int(node.this)
    return _NOT_LITERAL


_NOT_LITERAL = object()


def case_mappings(viewdef: str) -> Dict[str, Dict[Any, Any]]:
    """
    {column: {WHEN key: THEN value}} for every projected CASE whose results are all
    literals, i.e. every column whose values the view itself fixes.
    """
    select = sqlglot.parse_one(viewdef, read="postgres")
    assert isinstance(select, exp.Select), viewdef
    maps = {}
    for proj in select.expressions:
        node = proj.this if isinstance(proj, exp.Alias) else proj
        if not isinstance(node, exp.Case):
            continue
        pairs = {_literal(i.this): _literal(i.args["true"]) for i in node.args.get("ifs") or []}
        if pairs and all(v is not _NOT_LITERAL for v in pairs.values()):
            maps[proj.alias_or_name] = pairs
    return maps


# ---------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def conn():
    db = create_scratch_db(ADMIN_URL, prefix="agent_values")
    c = psycopg2.connect(db.admin_dsn, connect_timeout=5)
    yield c
    c.rollback()
    c.close()
    drop_scratch_db(ADMIN_URL, db)


def _fetch(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    conn.rollback()
    return rows


@pytest.fixture(scope="module")
def comments(conn) -> Dict[tuple, str]:
    rows = _fetch(conn, """
        SELECT c.relname, a.attname, coalesce(col_description(c.oid, a.attnum), '')
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
        WHERE n.nspname = 'agent' AND c.relkind = 'v'
    """)
    return {(view, col): comment for view, col, comment in rows}


@pytest.fixture(scope="module")
def mappings(conn) -> Dict[str, Dict[str, Dict[Any, Any]]]:
    out = {}
    for view in VIEWS:
        [(viewdef,)] = _fetch(conn, "SELECT pg_get_viewdef(%s::regclass, true)", (f"agent.{view}",))
        out[view] = case_mappings(viewdef)
    return out


@pytest.fixture(scope="module")
def expected(conn, mappings) -> Dict[tuple, Set[Any]]:
    status = {r[0] for r in _fetch(conn, """
        SELECT e.enumlabel FROM pg_attribute a JOIN pg_enum e ON e.enumtypid = a.atttypid
        WHERE a.attrelid = 'public.jobs'::regclass AND a.attname = 'status'
    """)}
    severity = {m.value: m.name.lower() for m in Severity if m is not Severity.NONE}
    quadrant = {m.value: m.name for m in MoranQuadrant if m is not MoranQuadrant.NONE}
    alert_algorithms = discrete_output_algorithms()
    return {
        ("analyses", "algorithm"): set(ALGORITHMS),
        ("analyses", "algorithm_label"): set(mappings["analyses"]["algorithm_label"].values()),
        ("analyses", "status"): status,
        ("alert_areas", "algorithm"): alert_algorithms,
        ("alert_areas", "severity"): set(severity),
        ("alert_areas", "severity_label"): set(severity.values()),
        ("alert_areas", "quadrant"): set(quadrant),
        ("alert_areas", "quadrant_label"): set(quadrant.values()),
        ("analysis_severity_summary", "algorithm"): alert_algorithms,
        ("analysis_severity_summary", "severity"): set(severity),
        ("analysis_severity_summary", "severity_label"): set(severity.values()),
    }


# ---------------------------------------------------------------- tests

def test_view_case_mappings_match_the_python_constants(mappings):
    severity = {m.value: m.name.lower() for m in Severity if m is not Severity.NONE}
    quadrant = {m.value: m.name for m in MoranQuadrant if m is not MoranQuadrant.NONE}
    # Every algorithm a job can have gets a label, so the CASE's ELSE never fires.
    assert set(mappings["analyses"]["algorithm_label"]) == set(ALGORITHMS)
    assert mappings["alert_areas"]["severity_label"] == severity
    assert mappings["alert_areas"]["quadrant_label"] == quadrant
    assert mappings["analysis_severity_summary"]["severity_label"] == severity
    assert discrete_output_algorithms() <= set(ALGORITHMS)


def test_every_closed_set_column_lists_exactly_its_values(comments, expected):
    report = {
        f"agent.{view}.{col}": problems(comments.get((view, col), ""), values)
        for (view, col), values in expected.items()
    }
    assert {k: v for k, v in report.items() if v} == {}


def test_no_closed_set_column_is_left_out(comments, mappings, expected):
    """A new CASE-label column, or a new 'Values:' comment, must be added to `expected`."""
    case_columns = {(view, col) for view, cols in mappings.items() for col in cols}
    listed = {key for key, comment in comments.items() if "Values:" in comment}
    assert case_columns - set(expected) == set()
    assert listed - set(expected) == set()


def test_values_present_in_the_seed_are_listed(conn, comments, expected):
    for view, col in expected:
        present = {r[0] for r in _fetch(conn, f"SELECT DISTINCT {col} FROM agent.{view} WHERE {col} IS NOT NULL")}
        assert present, f"seed has no value for agent.{view}.{col}"
        assert present <= documented_values(comments[(view, col)]), (view, col, present)


def test_coverage_comments_state_the_python_thresholds(comments):
    """The coverage columns quote two thresholds; they must be the ones the pipeline uses."""
    low = comments[("analyses", "low_coverage")]
    assert f"coverage_pct is below {LOW_COVERAGE_PCT:g}:" in low, low
    cloudy = comments[("analyses", "scenes_too_cloudy")]
    assert f"was {DEFAULT_SELECTION.loose_cloud_bound:g} percent or more" in cloudy, cloudy


def test_coverage_columns_read_the_seeded_results(conn):
    rows = _fetch(conn, """
        SELECT right(analysis_id::text, 3), coverage_pct, low_coverage, uncovered_masked_pct,
               scenes_found, scenes_used, acquisition_days, first_acquisition::text
        FROM agent.analyses ORDER BY analysis_id
    """)
    assert rows == [
        ("001", 97.8, False, 2.2, 14, 4, 4, "2025-12-14"),
        ("002", 100.0, False, 0.0, 41, 5, 5, "2025-10-20"),
        ("003", 91.2, False, 8.8, 120, 6, 6, "2025-02-17"),
        ("004", 41.5, True, 58.5, 9, 2, 1, "2026-01-09"),
        ("005", None, None, None, None, None, None, None),
        ("006", None, None, None, None, None, None, None),
    ]


@pytest.mark.parametrize(
    "comment,expected,problem",
    [
        # the comment shipped before stored values were listed: the live model then
        # filtered quadrant_label ILIKE '%high outlier%' and got no rows
        ("LISA only: HH high cluster, LL low cluster, HL high outlier, LH low outlier.",
         {"HH", "LL", "HL", "LH"}, "no single 'Values: ...' list"),
        ("Values: 'HH', 'LL', 'HL'. Meaning: ...", {"HH", "LL", "HL", "LH"}, "missing ['LH']"),
        ("Values: 1, 2, 3, 4.", {1, 2, 3}, "listed but never produced [4]"),
        ("Values: '1', '2', '3'.", {1, 2, 3}, "missing"),  # text '1' is not the integer code 1
    ],
)
def test_the_check_rejects_bad_comments(comment, expected, problem):
    assert any(problem in p for p in problems(comment, expected))
