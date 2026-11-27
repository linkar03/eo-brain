"""
Database-backed tests of the flood and lisa class areas: the severity_class_areas
table, the agent.severity_areas view the data agent reads, the zone-size columns
of agent.analyses, the writer (JobRepository.complete), and the upgrade of a
database built with the SQL files of a27bebf (before any of this existed).

Expected numbers are the known answers listed at the top of
evals/fixtures/synthetic_seed.sql.

Needs TEST_DATABASE_URL (PostGIS superuser URL); skipped when unset.
"""

import os
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import pytest
from psycopg2 import errors as pg_errors

from eo_brain.core.severity_areas import SeverityClass, severity_classes
from evals.db import DB_DIR, create_scratch_db, drop_scratch_db

ADMIN_URL = os.environ.get("TEST_DATABASE_URL", "").strip()

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not ADMIN_URL, reason="TEST_DATABASE_URL not set (PostGIS superuser URL)"),
]

# The SQL files as they were at a27bebf (schema.sql has not changed since). CI checks
# out one commit only, so the old versions live here rather than in git history.
OLD_SQL = Path(__file__).resolve().parent / "fixtures" / "db_a27bebf"


def job(n: int) -> str:
    return f"00000000-0000-4000-a000-{n:012d}"


FLOOD, LISA, OLD_FLOOD = job(2), job(3), job(1)


@pytest.fixture(scope="module")
def scratch():
    db = create_scratch_db(ADMIN_URL, prefix="agent_classes")
    yield db
    drop_scratch_db(ADMIN_URL, db)


@pytest.fixture
def admin(scratch):
    conn = psycopg2.connect(scratch.admin_dsn, connect_timeout=5)
    yield conn
    conn.rollback()
    conn.close()


@pytest.fixture
def agent(scratch):
    conn = psycopg2.connect(scratch.agent_dsn, connect_timeout=5)
    yield conn
    conn.rollback()
    conn.close()


def q(conn, sql, params=()):
    with conn.cursor() as cur:
        try:
            cur.execute(sql, params)
            return cur.fetchall() if cur.description else None
        finally:
            conn.rollback()


def one(conn, sql, params=()):
    [(value,)] = q(conn, sql, params)
    return value


def total(conn, analysis, where, column="area_ha"):
    return one(conn, f"SELECT sum({column}) FROM agent.severity_areas WHERE analysis_id = %s AND {where}",
               (analysis,))


# ---------------------------------------------------------------- the view, as agent_reader

def test_rows_per_analysis(agent):
    rows = q(agent, """
        SELECT right(analysis_id::text, 3), algorithm, severity, severity_label, quadrant, quadrant_label, vegetated
        FROM agent.severity_areas ORDER BY 1, severity, quadrant, vegetated
    """)
    assert rows == [
        ("002", "flood", 0, "none", None, None, None),
        ("002", "flood", 1, "watch", None, None, None),
        ("002", "flood", 2, "suspect", None, None, None),
        ("002", "flood", 3, "alert", None, None, None),
        ("003", "lisa", 0, "none", None, None, False),
        ("003", "lisa", 0, "none", None, None, True),
        ("003", "lisa", 1, "watch", 3, "HL", True),
        ("003", "lisa", 1, "watch", 4, "LH", True),
        ("003", "lisa", 2, "suspect", 3, "HL", True),
        ("003", "lisa", 2, "suspect", 4, "LH", True),
        ("003", "lisa", 3, "alert", 3, "HL", True),
        ("003", "lisa", 3, "alert", 4, "LH", True),
    ]


def test_areas_and_shares_are_the_seed_answers(agent):
    assert total(agent, FLOOD, "severity >= 1") == pytest.approx(18.02)
    assert total(agent, FLOOD, "severity >= 1", "area_km2") == pytest.approx(0.1802)
    assert total(agent, FLOOD, "severity >= 1", "pct_of_covered_area") == pytest.approx(100 * 1802 / 888120)
    assert total(agent, FLOOD, "severity_label = 'alert'") == pytest.approx(12.64)
    assert total(agent, LISA, "severity >= 1") == pytest.approx(10.62)
    assert total(agent, LISA, "severity >= 1", "pct_of_covered_area") == pytest.approx(100 * 1062 / 797400)
    assert total(agent, LISA, "severity >= 1", "pct_of_vegetated_area") == pytest.approx(100 * 1062 / 561062)
    assert total(agent, LISA, "quadrant_label = 'HL'") == pytest.approx(8.10)
    assert total(agent, LISA, "quadrant_label = 'LH'") == pytest.approx(2.52)
    assert total(agent, LISA, "severity = 3") == pytest.approx(6.52)
    assert total(agent, LISA, "vegetated") == pytest.approx(5610.62)


def test_shares_add_up_to_100_and_follow_the_table(agent, admin):
    sums = q(agent, """
        SELECT right(analysis_id::text, 3), sum(pct_of_covered_area), sum(pct_of_vegetated_area),
               sum(area_ha), count(pct_of_vegetated_area)
        FROM agent.severity_areas GROUP BY 1 ORDER BY 1
    """)
    assert [(a, round(c, 9), None if v is None else round(v, 9), n) for a, c, v, _, n in sums] == [
        ("002", 100.0, None, 0), ("003", 100.0, 100.0, 7)]
    table = dict(q(admin, "SELECT right(job_id::text, 3), sum(area_m2) / 1e4 FROM severity_class_areas GROUP BY 1"))
    assert {a: pytest.approx(ha) for a, _, _, ha, _ in sums} == table
    # every row's share is its own pixels over the analysis's
    rows = q(agent, "SELECT area_ha, pct_of_covered_area FROM agent.severity_areas WHERE analysis_id = %s", (LISA,))
    assert all(p == pytest.approx(100 * ha / 7974.0) for ha, p in rows)


def test_a_missing_analysis_is_null_and_an_empty_class_is_zero(agent):
    """NULL: no areas were recorded (001 ran before; ndvi never has any). 0: recorded, empty."""
    assert total(agent, OLD_FLOOD, "severity >= 1") is None
    assert total(agent, OLD_FLOOD, "severity >= 1", "pct_of_covered_area") is None
    assert total(agent, job(4), "severity >= 1") is None
    assert total(agent, LISA, "severity = 2 AND quadrant_label = 'HL'") == 0.0
    assert total(agent, LISA, "severity = 2 AND quadrant_label = 'HL'", "pct_of_covered_area") == 0.0
    assert total(agent, LISA, "severity = 2 AND quadrant_label = 'HL'", "pct_of_vegetated_area") == 0.0


def test_zone_size_columns_of_agent_analyses(agent):
    rows = q(agent, """
        SELECT right(analysis_id::text, 3), min_zone_pixels, min_zone_area_m2, small_patches_left_out,
               small_patches_area_ha, zone_limit_reached
        FROM agent.analyses WHERE min_zone_pixels IS NOT NULL OR algorithm IN ('flood', 'lisa') ORDER BY 1
    """)
    assert rows == [
        ("001", None, None, None, None, None),     # old-style vector_stats: not available
        ("002", 4, 400.0, 76, 1.52, False),
        ("003", 4, 400.0, 131, 2.62, False),
        ("006", None, None, None, None, None),     # pending
    ]
    assert one(agent, "SELECT count(*) FROM agent.analyses WHERE algorithm NOT IN ('flood', 'lisa') "
                      "AND (min_zone_pixels IS NOT NULL OR small_patches_left_out IS NOT NULL)") == 0


def test_zones_and_small_patches_add_up_to_the_class_area(agent):
    """The seed is consistent: mapped zones + patches left out = area of severity 1 to 3."""
    rows = q(agent, """
        SELECT a.analysis_id,
               (SELECT sum(f.area_ha) FROM agent.alert_areas f WHERE f.analysis_id = a.analysis_id),
               a.small_patches_area_ha,
               (SELECT sum(s.area_ha) FROM agent.severity_areas s
                WHERE s.analysis_id = a.analysis_id AND s.severity >= 1)
        FROM agent.analyses a WHERE a.min_zone_pixels IS NOT NULL
    """)
    assert len(rows) == 2
    for _, zones, small, exact in rows:
        assert zones + small == pytest.approx(exact)
        assert zones < exact


def test_lisa_zones_have_an_outlier_quadrant(agent):
    assert q(agent, "SELECT DISTINCT quadrant, quadrant_label FROM agent.alert_areas "
                    "WHERE algorithm = 'lisa' ORDER BY 1") == [(3, "HL"), (4, "LH")]
    assert q(agent, "SELECT DISTINCT quadrant_label FROM agent.alert_areas WHERE algorithm = 'flood'") == [(None,)]


# ---------------------------------------------------------------- permissions

def test_agent_reader_reads_the_view_not_the_table(agent):
    with pytest.raises(pg_errors.InsufficientPrivilege):
        q(agent, "SELECT * FROM public.severity_class_areas LIMIT 1")
    privileges = q(agent, """
        SELECT has_table_privilege('agent_reader', 'agent.severity_areas', 'SELECT'),
               has_table_privilege('agent_reader', 'agent.severity_areas', 'INSERT, UPDATE, DELETE, TRUNCATE'),
               has_table_privilege('agent_reader', 'public.severity_class_areas', 'SELECT, INSERT, UPDATE, DELETE')
    """)
    assert privileges == [(True, False, False)]


@pytest.mark.parametrize("sql", [
    "DELETE FROM public.severity_class_areas",
    "INSERT INTO public.severity_class_areas VALUES (%s, 'flood', 1, 0, 1, 100.0)",
    "UPDATE public.severity_class_areas SET pixel_count = 0",
])
def test_agent_reader_cannot_write_the_table(agent, sql):
    with agent.cursor() as cur:
        cur.execute("SET TRANSACTION READ WRITE")
        with pytest.raises(pg_errors.InsufficientPrivilege):
            cur.execute(sql, (FLOOD,) if "%s" in sql else None)
    agent.rollback()


# ---------------------------------------------------------------- the table

def test_class_rows_are_deleted_with_their_job(admin):
    with admin.cursor() as cur:
        cur.execute("DELETE FROM jobs WHERE id = %s", (LISA,))
        cur.execute("SELECT count(*) FROM severity_class_areas WHERE job_id = %s", (LISA,))
        assert cur.fetchone()[0] == 0
    admin.rollback()
    assert one(admin, "SELECT count(*) FROM severity_class_areas WHERE job_id = %s", (LISA,)) == 11


@pytest.mark.parametrize(
    "row, error",
    [
        ((FLOOD, "flood", 4, 0, 1, 100.0), pg_errors.CheckViolation),      # no severity 4
        ((LISA, "lisa", 0, 5, 1, 100.0), pg_errors.CheckViolation),        # no quadrant 5
        ((FLOOD, "flood", 1, 3, 1, 100.0), pg_errors.CheckViolation),      # flood has no quadrant
        ((LISA, "lisa", 2, 1, 1, 100.0), pg_errors.CheckViolation),        # a lisa severity is an outlier
        ((job(4), "ndvi", 0, 0, 1, 100.0), pg_errors.CheckViolation),      # ndvi has no classes
        ((FLOOD, "flood", 1, 0, -1, 0.0), pg_errors.CheckViolation),       # negative count
        ((FLOOD, "flood", 1, 0, 0, 100.0), pg_errors.CheckViolation),      # area without pixels
        ((FLOOD, "flood", 1, 0, 3, 0.0), pg_errors.CheckViolation),        # pixels without area
        ((FLOOD, "flood", 1, 0, 52, 5200.0), pg_errors.UniqueViolation),   # the class already has its row
        ((FLOOD, "lisa", 1, 3, 1, 100.0), pg_errors.ForeignKeyViolation),  # 002 is a flood job
        ((job(99), "flood", 1, 0, 1, 100.0), pg_errors.ForeignKeyViolation),
    ],
)
def test_the_table_refuses_impossible_rows(admin, row, error):
    with pytest.raises(error):
        q(admin, "INSERT INTO severity_class_areas (job_id, algorithm, severity, quadrant, pixel_count, area_m2) "
                 "VALUES (%s, %s, %s, %s, %s, %s)", row)


def test_every_class_of_each_algorithm_fits_the_table(admin):
    """The Python class list and the table's checks agree (no class the writer emits is refused)."""
    with admin.cursor() as cur:
        cur.execute("DELETE FROM severity_class_areas WHERE job_id IN (%s, %s)", (FLOOD, LISA))
        for jid, algorithm in ((FLOOD, "flood"), (LISA, "lisa")):
            for s, quad in severity_classes(algorithm):
                cur.execute("INSERT INTO severity_class_areas VALUES (%s, %s, %s, %s, 1, 1.0)",
                            (jid, algorithm, s, quad))
    admin.rollback()


# ---------------------------------------------------------------- the writer

@pytest.fixture
def repo(scratch, monkeypatch):
    from eo_brain.db import database

    monkeypatch.setattr(database, "DATABASE_URL", scratch.admin_dsn)
    monkeypatch.setattr(database, "_pool", None)
    created = []
    try:
        yield database.JobRepository(), created
    finally:
        for jid in created:
            database.JobRepository().delete(jid)
        database.close_pool()


def _new_job(repo, created, algorithm):
    record = repo.create(algorithm, "element84-s2", [10.0, 40.0, 10.01, 40.01],
                         datetime(2025, 6, 1, tzinfo=timezone.utc), datetime(2025, 6, 30, tzinfo=timezone.utc),
                         {"geocoded_name": "Writer Test (synthetic)", "resolution_m": 20})
    created.append(record.id)
    return record.id


def _classes(algorithm, counts):
    return [SeverityClass(algorithm, s, quad, counts.get((s, quad), 0), counts.get((s, quad), 0) * 400.0)
            for s, quad in severity_classes(algorithm)]


def test_complete_writes_every_class_with_the_job(repo, admin, agent):
    jobs, created = repo
    jid = _new_job(jobs, created, "lisa")
    jobs.complete(jid, result={"coverage": {"covered_px": 100}}, cog_path="/x.tif", output_crs="EPSG:32632",
                  message="done", severity_classes=_classes("lisa", {(0, 0): 40, (0, 1): 50, (3, 4): 10}))
    rows = q(admin, "SELECT severity, quadrant, pixel_count, area_m2 FROM severity_class_areas "
                    "WHERE job_id = %s ORDER BY severity, quadrant", (str(jid),))
    assert len(rows) == 11 and (3, 4, 10, 4000.0) in rows and (2, 3, 0, 0.0) in rows
    assert one(admin, "SELECT status::text FROM jobs WHERE id = %s", (str(jid),)) == "completed"
    assert total(agent, str(jid), "severity >= 1") == pytest.approx(0.4)
    assert total(agent, str(jid), "severity >= 1", "pct_of_vegetated_area") == pytest.approx(100 * 10 / 60)

    # completing again replaces the rows; no severity_classes leaves them alone
    jobs.complete(jid, result={}, cog_path="/x.tif", output_crs="EPSG:32632",
                  severity_classes=_classes("lisa", {(0, 0): 7}))
    assert one(admin, "SELECT sum(pixel_count) FROM severity_class_areas WHERE job_id = %s", (str(jid),)) == 7
    jobs.complete(jid, result={}, cog_path="/x.tif", output_crs="EPSG:32632")
    assert one(admin, "SELECT count(*) FROM severity_class_areas WHERE job_id = %s", (str(jid),)) == 11


def test_a_failed_class_write_does_not_fail_the_completion(repo, admin):
    jobs, created = repo
    jid = _new_job(jobs, created, "flood")
    wrong = _classes("lisa", {(0, 0): 5})      # lisa classes for a flood job: refused by the foreign key
    jobs.complete(jid, result={"coverage": {}}, cog_path="/x.tif", output_crs="EPSG:32632",
                  message="done", severity_classes=wrong)
    assert one(admin, "SELECT count(*) FROM severity_class_areas WHERE job_id = %s", (str(jid),)) == 0
    assert q(admin, "SELECT status::text, message FROM jobs WHERE id = %s", (str(jid),)) == [("completed", "done")]


def test_seed_unchanged_by_the_writer_tests(admin):
    assert one(admin, "SELECT count(*) FROM jobs") == 11
    assert one(admin, "SELECT count(*) FROM severity_class_areas") == 15


# ---------------------------------------------------------------- upgrade from a27bebf

_OLD_ROWS = """
INSERT INTO jobs (id, status, algorithm, provider, bbox, start_date, end_date, params, progress,
                  created_at, completed_at, result)
VALUES
  ('00000000-0000-4000-b000-000000000001', 'completed', 'lisa', 'element84-s2',
   ST_MakeEnvelope(12.0, 45.0, 12.1, 45.1, 4326), '2025-01-01', '2025-02-01',
   '{"geocoded_name": "Upgrade Plain (synthetic)", "resolution_m": 10}', 100,
   '2026-01-01 10:00:00+00', '2026-01-01 10:01:00+00',
   '{"coverage": {"coverage_pct": 95.0, "area_px": 1000, "covered_px": 950},
     "vector_stats": {"feature_count": 2, "total_area_m2": 9000.0, "by_severity": {"1": 1, "2": 0, "3": 1}}}'),
  ('00000000-0000-4000-b000-000000000002', 'completed', 'ndvi', 'element84-s2',
   ST_MakeEnvelope(12.0, 45.0, 12.1, 45.1, 4326), '2025-01-01', '2025-02-01',
   '{"geocoded_name": "Upgrade Plain (synthetic)", "resolution_m": 20}', 100,
   '2026-01-01 11:00:00+00', '2026-01-01 11:01:00+00',
   '{"coverage": {"coverage_pct": 99.0, "area_px": 100, "covered_px": 99},
     "value_distribution": {"index": "ndvi", "pixels": 99, "min": 0.1, "max": 0.7, "mean": 0.4}}');
INSERT INTO alert_features (job_id, algorithm, severity, quadrant, geometry, centroid, area_m2, perimeter_m)
VALUES
  ('00000000-0000-4000-b000-000000000001', 'lisa', 3, NULL,
   ST_MakeEnvelope(12.01, 45.01, 12.02, 45.02, 4326), ST_MakePoint(12.015, 45.015), 6000.0, 400.0),
  ('00000000-0000-4000-b000-000000000001', 'lisa', 1, NULL,
   ST_MakeEnvelope(12.03, 45.03, 12.04, 45.04, 4326), ST_MakePoint(12.035, 45.035), 3000.0, 300.0);
INSERT INTO index_value_bins VALUES ('00000000-0000-4000-b000-000000000002', 165, 0.65, 0.66, 99, 39600.0);
"""

_CONTENT = [
    "SELECT md5(string_agg(t::text, '|' ORDER BY id)) FROM jobs t",
    "SELECT md5(string_agg(t::text, '|' ORDER BY id)) FROM alert_features t",
    "SELECT md5(string_agg(t::text, '|' ORDER BY job_id, bin_index)) FROM index_value_bins t",
]
_CURRENT = [DB_DIR / "schema_vectors.sql", DB_DIR / "agent_views.sql", DB_DIR / "agent_role.sql"]


@pytest.fixture(scope="module")
def upgraded():
    """A database built from the a27bebf files, with rows, then brought up to date with the current ones."""
    old_files = [DB_DIR / "schema.sql"] + [OLD_SQL / p.name for p in _CURRENT]
    assert all(p.exists() for p in old_files)
    db = create_scratch_db(ADMIN_URL, seed=False, prefix="agent_upgrade", files=old_files)
    try:
        conn = psycopg2.connect(db.admin_dsn, connect_timeout=5)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(_OLD_ROWS)
            assert _relation_exists(cur, "severity_class_areas") is False
            before = [_fetch_one(cur, sql) for sql in _CONTENT]
            for path in _CURRENT:
                cur.execute(path.read_text(encoding="utf-8"))
            after_once = [_fetch_one(cur, sql) for sql in _CONTENT]
            for path in _CURRENT:                       # and again: the files are idempotent
                cur.execute(path.read_text(encoding="utf-8"))
            after_twice = [_fetch_one(cur, sql) for sql in _CONTENT]
        conn.close()
        yield db, before, after_once, after_twice
    finally:
        drop_scratch_db(ADMIN_URL, db)


def _fetch_one(cur, sql):
    cur.execute(sql)
    return cur.fetchone()[0]


def _relation_exists(cur, name):
    cur.execute("SELECT to_regclass(%s) IS NOT NULL", (f"public.{name}",))
    return cur.fetchone()[0]


def test_the_upgrade_keeps_every_row(upgraded):
    _, before, after_once, after_twice = upgraded
    assert None not in before
    assert before == after_once == after_twice


def test_the_upgrade_adds_the_table_the_key_and_the_view(upgraded):
    db, *_ = upgraded
    conn = psycopg2.connect(db.admin_dsn, connect_timeout=5)
    try:
        assert one(conn, "SELECT count(*) FROM pg_constraint WHERE conname = 'jobs_id_algorithm_key'") == 1
        assert one(conn, "SELECT count(*) FROM severity_class_areas") == 0
        # a job completed after the upgrade can store its classes
        with conn.cursor() as cur:
            cur.execute("INSERT INTO severity_class_areas VALUES "
                        "('00000000-0000-4000-b000-000000000001', 'lisa', 3, 4, 2, 200.0)")
            cur.execute("SELECT area_ha, quadrant_label FROM agent.severity_areas")
            assert cur.fetchall() == [(0.02, "LH")]
        conn.rollback()
    finally:
        conn.close()


def test_old_analyses_read_as_not_available_after_the_upgrade(upgraded):
    db, *_ = upgraded
    conn = psycopg2.connect(db.agent_dsn, connect_timeout=5)
    try:
        assert one(conn, "SELECT count(*) FROM agent.severity_areas") == 0
        assert one(conn, "SELECT sum(area_ha) FROM agent.severity_areas WHERE severity >= 1") is None
        assert q(conn, """
            SELECT algorithm, min_zone_pixels, min_zone_area_m2, small_patches_left_out, small_patches_area_ha,
                   zone_limit_reached, value_mean
            FROM agent.analyses ORDER BY created_at
        """) == [("lisa", None, None, None, None, None, None), ("ndvi", None, None, None, None, None, 0.4)]
        # the zones stored before the quadrant was recorded keep a NULL quadrant
        assert q(conn, "SELECT severity, quadrant, quadrant_label, area_ha FROM agent.alert_areas "
                       "ORDER BY severity") == [(1, None, None, 0.3), (3, None, None, 0.6)]
        assert q(conn, "SELECT severity, feature_count, total_area_ha FROM agent.analysis_severity_summary "
                       "ORDER BY severity") == [(1, 1, 0.3), (3, 1, 0.6)]
        assert one(conn, "SELECT sum(area_ha) FROM agent.index_value_distribution") == pytest.approx(3.96)
        with pytest.raises(pg_errors.InsufficientPrivilege):
            q(conn, "SELECT * FROM public.severity_class_areas LIMIT 1")
    finally:
        conn.close()


def test_the_upgraded_views_match_a_fresh_database(upgraded, scratch):
    """Same agent views, columns and comments as a database built from the current files."""
    sql = """
        SELECT c.relname, a.attname, format_type(a.atttypid, a.atttypmod),
               coalesce(col_description(c.oid, a.attnum), ''), coalesce(obj_description(c.oid, 'pg_class'), ''),
               has_table_privilege('agent_reader', c.oid, 'SELECT')
        FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
        WHERE n.nspname = 'agent' ORDER BY c.relname, a.attnum
    """
    db, *_ = upgraded
    old = psycopg2.connect(db.admin_dsn, connect_timeout=5)
    fresh = psycopg2.connect(scratch.admin_dsn, connect_timeout=5)
    try:
        assert q(old, sql) == q(fresh, sql)
    finally:
        old.close()
        fresh.close()
