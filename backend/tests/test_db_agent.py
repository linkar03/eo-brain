"""
Database-backed tests for the data agent's isolation model.

Needs a PostGIS superuser URL to a maintenance database, e.g.

    TEST_DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/postgres

Each module run creates a throwaway database (evals/db.py), applies the
same SQL files docker compose applies, loads the synthetic seed, and drops
it at the end. Skipped when TEST_DATABASE_URL is unset; if it is set but
unreachable the tests error instead of silently skipping.

Part (a) talks to PostgreSQL as agent_reader directly, BYPASSING the SQL
gate: it proves the role alone blocks writes and reads outside the views.
"""

import os
import time

import psycopg2
import pytest
from psycopg2 import errors as pg_errors

from eo_brain.agent.data_agent import DataAgent
from eo_brain.agent.llm import ModelTurn, ToolCall
from eo_brain.agent.providers.scripted import ScriptedProvider
from eo_brain.agent.sql_exec import QueryError, ReadOnlyExecutor
from eo_brain.agent.sql_gate import SQLGate
from evals.db import create_scratch_db, drop_scratch_db, fingerprint

ADMIN_URL = os.environ.get("TEST_DATABASE_URL", "").strip()

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not ADMIN_URL, reason="TEST_DATABASE_URL not set (PostGIS superuser URL)"),
]

VIEWS = {"analyses", "alert_areas", "analysis_severity_summary"}


@pytest.fixture(scope="module")
def scratch():
    db = create_scratch_db(ADMIN_URL, prefix="agent_test")
    before = fingerprint(db.admin_dsn)
    yield db
    try:
        assert fingerprint(db.admin_dsn) == before, "database content changed during the tests"
    finally:
        drop_scratch_db(ADMIN_URL, db)


@pytest.fixture
def agent_conn(scratch):
    conn = psycopg2.connect(scratch.agent_dsn, connect_timeout=5)
    yield conn
    conn.rollback()
    conn.close()


def run(conn, sql, *, read_write=False):
    with conn.cursor() as cur:
        try:
            if read_write:
                cur.execute("SET TRANSACTION READ WRITE")
            cur.execute(sql)
            return cur.fetchall() if cur.description else None
        finally:
            conn.rollback()


# ---------------------------------------------------------------- (a) role, gate bypassed

def test_role_attributes(scratch):
    conn = psycopg2.connect(scratch.admin_dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT rolsuper, rolcreatedb, rolcreaterole, rolinherit, rolreplication, rolbypassrls, rolconfig "
                "FROM pg_roles WHERE rolname = 'agent_reader'"
            )
            sup, createdb, createrole, inherit, repl, bypass, config = cur.fetchone()
    finally:
        conn.close()
    assert not any([sup, createdb, createrole, inherit, repl, bypass])
    assert "default_transaction_read_only=on" in config
    assert "statement_timeout=3s" in config


def test_session_defaults(agent_conn):
    assert run(agent_conn, "SHOW default_transaction_read_only") == [("on",)]
    assert run(agent_conn, "SHOW statement_timeout") == [("3s",)]
    assert run(agent_conn, "SELECT current_user") == [("agent_reader",)]


def test_can_read_the_views(agent_conn):
    assert run(agent_conn, "SELECT count(*) FROM agent.analyses") == [(6,)]
    assert run(agent_conn, "SELECT count(*) FROM agent.alert_areas") == [(10,)]
    # geography maths works (needs the spatial_ref_sys grant)
    assert run(agent_conn, "SELECT count(*) FROM agent.analyses WHERE footprint_area_ha > 0") == [(6,)]


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM public.jobs LIMIT 1",
        "SELECT * FROM public.alert_features LIMIT 1",
        "SELECT * FROM public.alert_feature_stats LIMIT 1",
        "SELECT * FROM public.geometry_columns LIMIT 1",
        "SELECT * FROM public.geography_columns LIMIT 1",
        "SELECT pg_read_file('/etc/hostname')",
        "SELECT pg_ls_dir('.')",
        "SELECT lo_import('/etc/hostname')",
    ],
)
def test_reads_outside_the_views_are_denied(agent_conn, sql):
    with pytest.raises(pg_errors.InsufficientPrivilege):
        run(agent_conn, sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM public.jobs",
        "UPDATE agent.analyses SET algorithm = 'x'",
        "INSERT INTO agent.analyses (algorithm) VALUES ('x')",
        "CREATE TABLE public.t (a int)",
        "CREATE TEMP TABLE t (a int)",
    ],
)
def test_writes_fail_in_the_default_read_only_transaction(agent_conn, sql):
    with pytest.raises(pg_errors.ReadOnlySqlTransaction):
        run(agent_conn, sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM public.jobs",
        "DELETE FROM agent.analyses",
        "UPDATE agent.analyses SET algorithm = 'x'",
        "INSERT INTO agent.analyses (algorithm) VALUES ('x')",
        "TRUNCATE public.jobs",
        "CREATE TABLE public.t (a int)",
        "CREATE TABLE agent.t (a int)",
        "CREATE VIEW agent.v AS SELECT 1",
        "CREATE TEMP TABLE t (a int)",
        "CREATE SCHEMA s2",
        "COPY public.jobs TO STDOUT",
    ],
)
def test_writes_fail_even_if_read_only_is_switched_off(agent_conn, sql):
    """default_transaction_read_only is user-settable; privileges are not."""
    with pytest.raises(pg_errors.InsufficientPrivilege):
        run(agent_conn, sql, read_write=True)


def test_statement_timeout_fires(agent_conn):
    start = time.monotonic()
    with pytest.raises(pg_errors.QueryCanceled):
        run(agent_conn, "SELECT pg_sleep(10)")
    assert time.monotonic() - start < 8


def test_residual_system_catalogs_are_readable_so_the_gate_must_block_them(agent_conn):
    """
    PostgreSQL grants catalog access to PUBLIC; the role cannot remove it.
    This is the one place where the SQL gate is the only layer.
    """
    sql = "SELECT count(*) FROM pg_catalog.pg_class LIMIT 1"
    assert run(agent_conn, sql)[0][0] > 0
    assert not SQLGate(VIEWS).check(sql).ok


# ---------------------------------------------------------------- (b) executor and schema doc

@pytest.fixture
def executor(scratch):
    return ReadOnlyExecutor(scratch.agent_dsn)


def test_schema_doc_comes_from_the_catalog(executor):
    doc = executor.load_schema_doc()
    assert set(doc.relation_names()) == VIEWS
    text = doc.render()
    assert "agent.analyses: One row per satellite analysis job" in text
    assert "severity_label (text): Severity name. Values: 'watch', 'suspect', 'alert'" in text
    assert "footprint (geometry(Polygon,4326))" in text


def test_executor_never_returns_geometry(executor):
    r = executor.execute(
        "SELECT analysis_id, footprint, footprint::geography AS g FROM agent.analyses ORDER BY created_at LIMIT 2"
    )
    assert r.columns == ["analysis_id", "footprint", "g"]
    assert r.rows[0][0] == "00000000-0000-4000-a000-000000000001"
    assert {row[1] for row in r.rows} == {"<geometry omitted>"}
    assert {row[2] for row in r.rows} == {"<geography omitted>"}
    assert r.omitted_columns == ["footprint (geometry)", "g (geography)"]


def test_executor_row_cap(scratch):
    r = ReadOnlyExecutor(scratch.agent_dsn, max_rows=2).execute("SELECT feature_id FROM agent.alert_areas LIMIT 10")
    assert r.row_count == 2 and r.truncated


def test_executor_timeout(scratch):
    with pytest.raises(QueryError, match="statement timeout"):
        ReadOnlyExecutor(scratch.agent_dsn, statement_timeout_ms=300).execute("SELECT pg_sleep(2)")


def test_executor_permission_error(executor):
    with pytest.raises(QueryError, match="permission denied"):
        executor.execute("SELECT * FROM public.jobs LIMIT 1")


@pytest.mark.parametrize(
    "sql,expected",
    [
        ("SELECT total_area_ha FROM agent.analysis_severity_summary "
         "WHERE analysis_id = '00000000-0000-4000-a000-000000000002' AND severity_label = 'alert' LIMIT 1", [[12.0]]),
        ("SELECT count(*) FROM agent.analyses WHERE status = 'completed' LIMIT 1", [[4]]),
        ("SELECT quadrant_label, area_ha FROM agent.alert_areas WHERE algorithm = 'lisa' AND severity = 3 LIMIT 5",
         [["HL", 6.0]]),
        ("SELECT window_days, processing_seconds FROM agent.analyses "
         "WHERE analysis_id = '00000000-0000-4000-a000-000000000001' LIMIT 1", [[30, 95.0]]),
    ],
)
def test_views_match_the_seed(executor, sql, expected):
    assert executor.execute(sql).rows == expected


# ---------------------------------------------------------------- (c) data agent end to end

def test_data_agent_against_the_database(executor):
    steps = [
        ModelTurn(tool_calls=[ToolCall("run_sql", {"sql": "SELECT * FROM jobs LIMIT 1"})]),
        ModelTurn(tool_calls=[ToolCall("run_sql", {"sql": "SELECT missing FROM agent.analyses LIMIT 1"})]),
        ModelTurn(tool_calls=[ToolCall("run_sql", {"sql":
            "SELECT total_area_ha FROM agent.analysis_severity_summary WHERE severity_label = 'alert' "
            "AND analysis_id = (SELECT analysis_id FROM agent.analyses WHERE algorithm = 'flood' "
            "AND status = 'completed' ORDER BY created_at DESC LIMIT 1) LIMIT 1"})]),
    ]
    agent = DataAgent(ScriptedProvider(steps), models=["m"], executor=executor)
    res = agent.answer("How many hectares at alert severity in the latest completed flood analysis?")
    assert res.status == "ok" and res.rows == [[12.0]]
    assert [a.stage for a in res.attempts] == ["gate", "database", ""]
