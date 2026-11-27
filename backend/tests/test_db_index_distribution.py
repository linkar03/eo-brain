"""
Database-backed tests of the index value distribution: the index_value_bins
table, the agent.index_value_distribution view the data agent reads, the
value_* columns of agent.analyses, and the writer (JobRepository.complete).

Expected numbers are the known answers listed at the top of
evals/fixtures/synthetic_seed.sql.

Needs TEST_DATABASE_URL (PostGIS superuser URL); skipped when unset.
"""

import os
from datetime import datetime, timezone

import psycopg2
import pytest
from psycopg2 import errors as pg_errors

from eo_brain.core.distribution import ValueBin, bin_edge
from evals.db import create_scratch_db, drop_scratch_db

ADMIN_URL = os.environ.get("TEST_DATABASE_URL", "").strip()

pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not ADMIN_URL, reason="TEST_DATABASE_URL not set (PostGIS superuser URL)"),
]


def job(n: int) -> str:
    return f"00000000-0000-4000-a000-{n:012d}"


WITH_BINS = {job(4), job(7), job(8), job(9), job(10)}


@pytest.fixture(scope="module")
def scratch():
    db = create_scratch_db(ADMIN_URL, prefix="agent_dist")
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


def area_ha(conn, analysis, where):
    return one(conn, f"SELECT sum(area_ha) FROM agent.index_value_distribution "
                     f"WHERE analysis_id = %s AND {where}", (analysis,))


# ---------------------------------------------------------------- the view, as agent_reader

def test_threshold_filters_give_the_seeded_areas(agent):
    assert area_ha(agent, job(7), "value_from >= 0.65") == pytest.approx(4786.44)
    # the 0.64 interval (0.64 <= value < 0.65) is below 0.65: the lower edge decides
    assert area_ha(agent, job(7), "value_from >= 0.64") == pytest.approx(4786.44 + 435.08)
    assert area_ha(agent, job(7), "value_from >= 0.4") == pytest.approx(6787.88)
    assert area_ha(agent, job(4), "value_from >= 0.65") == pytest.approx(554.32)
    assert area_ha(agent, job(8), "value_from >= 0.65") == pytest.approx(1517.64)
    # from 0.2 to 0.5: the 0.50 interval is out, the 0.49 one would be in
    assert area_ha(agent, job(8), "value_from >= 0.2 AND value_to <= 0.5") == pytest.approx(3962.52)
    # water: the 0.00 interval is in, the -0.01 one is out
    assert area_ha(agent, job(9), "value_from >= 0") == pytest.approx(2506.0)
    assert area_ha(agent, job(9), "value_to <= 0") == pytest.approx(8353.04 - 2506.0)
    assert area_ha(agent, job(10), "value_from >= 0") == pytest.approx(2579.76)
    # nothing above the threshold is NULL with the filter in WHERE, 0 with the filter in the sum
    assert area_ha(agent, job(9), "value_from >= 0.65") is None
    assert one(agent, "SELECT sum(CASE WHEN value_from >= 0.65 THEN area_ha ELSE 0 END) "
                      "FROM agent.index_value_distribution WHERE analysis_id = %s", (job(9),)) == 0


def test_shares_are_of_the_whole_covered_area_whatever_the_filter(agent):
    rows = q(agent, "SELECT analysis_id::text, sum(pct_of_covered_area) FROM agent.index_value_distribution "
                    "GROUP BY analysis_id")
    assert {a for a, _ in rows} == WITH_BINS
    assert all(total == pytest.approx(100.0, abs=1e-9) for _, total in rows)
    # the window runs before the caller's filter: the share stays relative to the whole analysis
    share = one(agent, "SELECT sum(pct_of_covered_area) FROM agent.index_value_distribution "
                       "WHERE analysis_id = %s AND value_from >= 0", (job(10),))
    assert share == pytest.approx(34.00058, abs=1e-4)
    share = one(agent, "SELECT sum(pct_of_covered_area) FROM agent.index_value_distribution "
                       "WHERE value_from >= 0.65 AND location_name ILIKE '%%sample plain%%' "
                       "AND analysis_id = %s", (job(7),))
    assert share == pytest.approx(55.0019, abs=1e-4)


def test_interval_edges_are_two_decimal_numbers(agent):
    rows = q(agent, "SELECT DISTINCT pg_typeof(value_from)::text, pg_typeof(value_to)::text, "
                    "pg_typeof(area_ha)::text, pg_typeof(pct_of_covered_area)::text "
                    "FROM agent.index_value_distribution")
    assert rows == [("numeric", "numeric", "double precision", "double precision")]
    edges = q(agent, "SELECT min(value_from)::float8, max(value_to)::float8, "
                     "bool_and(value_to - value_from = 0.01), bool_and(scale(value_from) = 2) "
                     "FROM agent.index_value_distribution")
    assert edges == [(-0.62, 0.84, True, True)]


def test_value_columns_come_from_the_distribution_block_not_the_summary(agent):
    rows = q(agent, "SELECT right(analysis_id::text, 3), value_mean, value_min, value_max "
                    "FROM agent.analyses ORDER BY analysis_id")
    by_id = {r[0]: r[1:] for r in rows}
    assert by_id["004"] == (0.4085, 0.083, 0.717)
    # 007 also carries an older-style result.summary (mean 0.5874, over the whole grid)
    assert by_id["007"] == (0.6107, 0.123, 0.837)
    assert by_id["009"] == (-0.1952, -0.617, 0.377)
    # no distribution recorded (011 has a summary), flood / lisa, failed, pending
    for n in ("001", "002", "003", "005", "006", "011"):
        assert by_id[n] == (None, None, None), n


def test_mean_min_max_describe_the_same_pixels_as_the_intervals(agent):
    rows = q(agent, """
        SELECT a.analysis_id::text, a.value_min, a.value_max, a.value_mean,
               min(d.value_from)::float8, max(d.value_to)::float8,
               sum((d.value_from + 0.005)::float8 * d.pct_of_covered_area) / 100
        FROM agent.analyses a JOIN agent.index_value_distribution d ON d.analysis_id = a.analysis_id
        GROUP BY a.analysis_id, a.value_min, a.value_max, a.value_mean
    """)
    assert {r[0] for r in rows} == WITH_BINS
    for analysis, vmin, vmax, mean, lo, hi, mid_mean in rows:
        assert lo <= vmin and vmax <= hi, analysis
        # the mean of values inside 0.01-wide intervals is within half a width of the midpoint mean
        assert abs(mean - mid_mean) <= 0.005 + 1e-9, analysis


def test_no_rows_for_analyses_without_a_distribution(agent):
    rows = q(agent, """
        SELECT a.algorithm, a.status, count(d.analysis_id)
        FROM agent.analyses a LEFT JOIN agent.index_value_distribution d ON d.analysis_id = a.analysis_id
        WHERE a.analysis_id NOT IN %s
        GROUP BY a.analysis_id, a.algorithm, a.status
    """, (tuple(WITH_BINS),))
    assert len(rows) == 6 and all(n == 0 for _, _, n in rows)
    # 011 is a completed ndwi analysis: no rows means no recorded distribution, not no water
    assert one(agent, "SELECT count(*) FROM agent.index_value_distribution "
                      "WHERE location_name ILIKE '%%old harbour%%'") == 0


# ---------------------------------------------------------------- the table

def test_intervals_count_exactly_the_covered_pixels(admin):
    rows = q(admin, """
        SELECT b.job_id::text, j.algorithm, j.status::text, sum(b.pixel_count), sum(b.area_m2),
               (j.result -> 'coverage' ->> 'covered_px')::bigint,
               (j.result -> 'value_distribution' ->> 'pixels')::bigint,
               (j.result -> 'value_distribution' ->> 'area_m2')::float8,
               (j.params ->> 'resolution_m')::int,
               bool_and(b.area_m2 = b.pixel_count * power((j.params ->> 'resolution_m')::int, 2))
        FROM index_value_bins b JOIN jobs j ON j.id = b.job_id
        GROUP BY b.job_id, j.id
    """)
    assert {r[0] for r in rows} == WITH_BINS
    for jid, algorithm, status, pixels, area, covered, block_pixels, block_area, res, area_from_grid in rows:
        assert algorithm in {"ndvi", "ndwi", "mndwi"} and status == "completed", jid
        assert pixels == covered == block_pixels, jid
        assert area == pytest.approx(block_area) and area_from_grid, jid


def test_agent_reader_reads_the_view_but_not_the_table(admin, agent):
    assert one(agent, "SELECT count(*) FROM agent.index_value_distribution") == 39
    with pytest.raises(pg_errors.InsufficientPrivilege):
        q(agent, "SELECT * FROM public.index_value_bins LIMIT 1")
    privileges = q(admin, """
        SELECT has_table_privilege('agent_reader', 'agent.index_value_distribution', 'SELECT'),
               has_table_privilege('agent_reader', 'public.index_value_bins', 'SELECT'),
               has_table_privilege('agent_reader', 'agent.index_value_distribution', 'INSERT, UPDATE, DELETE')
    """)
    assert privileges == [(True, False, False)]


def test_intervals_are_deleted_with_their_job(admin):
    with admin.cursor() as cur:
        cur.execute("SELECT count(*) FROM index_value_bins WHERE job_id = %s", (job(7),))
        assert cur.fetchone()[0] == 9
        cur.execute("DELETE FROM jobs WHERE id = %s", (job(7),))
        cur.execute("SELECT count(*) FROM index_value_bins WHERE job_id = %s", (job(7),))
        assert cur.fetchone()[0] == 0
    admin.rollback()
    assert one(admin, "SELECT count(*) FROM index_value_bins WHERE job_id = %s", (job(7),)) == 9


@pytest.mark.parametrize(
    "values",
    [
        "(%s, 165, 0.64, 0.65, 10, 1000.0)",     # edge does not match the interval number
        "(%s, 165, 0.65, 0.67, 10, 1000.0)",     # not 0.01 wide
        "(%s, 200, 1.00, 1.01, 10, 1000.0)",     # past the last interval
        "(%s, 165, 0.65, 0.66, 0, 0.0)",         # an empty interval is not stored
        "(%s, 165, 0.65, 0.66, 10, -1.0)",       # negative area
    ],
)
def test_the_table_refuses_malformed_intervals(admin, values):
    with pytest.raises(pg_errors.CheckViolation):
        q(admin, "INSERT INTO index_value_bins (job_id, bin_index, value_from, value_to, pixel_count, area_m2) "
                 f"VALUES {values}", (job(11),))


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


def _new_job(repo, created, algorithm="ndvi"):
    record = repo.create(algorithm, "element84-s2", [10.0, 40.0, 10.01, 40.01],
                         datetime(2025, 6, 1, tzinfo=timezone.utc), datetime(2025, 6, 30, tzinfo=timezone.utc),
                         {"geocoded_name": "Writer Test (synthetic)", "resolution_m": 20})
    created.append(record.id)
    return record.id


def _bins(*spec):
    return [ValueBin(k, bin_edge(k), bin_edge(k + 1), n, n * 400.0) for k, n in spec]


def test_complete_writes_the_intervals_with_the_job(repo, admin):
    jobs, created = repo
    jid = _new_job(jobs, created)
    jobs.complete(jid, result={"coverage": {"covered_px": 30}}, cog_path="/x.tif", output_crs="EPSG:32632",
                  message="done", value_bins=_bins((165, 10), (199, 20)))
    rows = q(admin, "SELECT bin_index, value_from::float8, value_to::float8, pixel_count, area_m2 "
                    "FROM index_value_bins WHERE job_id = %s ORDER BY bin_index", (str(jid),))
    assert rows == [(165, 0.65, 0.66, 10, 4000.0), (199, 0.99, 1.0, 20, 8000.0)]
    assert one(admin, "SELECT status::text FROM jobs WHERE id = %s", (str(jid),)) == "completed"

    # completing again replaces the intervals; no value_bins leaves them alone
    jobs.complete(jid, result={}, cog_path="/x.tif", output_crs="EPSG:32632", value_bins=_bins((100, 5)))
    assert q(admin, "SELECT bin_index, pixel_count FROM index_value_bins WHERE job_id = %s",
             (str(jid),)) == [(100, 5)]
    jobs.complete(jid, result={}, cog_path="/x.tif", output_crs="EPSG:32632")
    assert one(admin, "SELECT count(*) FROM index_value_bins WHERE job_id = %s", (str(jid),)) == 1


def test_a_failed_interval_write_does_not_fail_the_completion(repo, admin):
    jobs, created = repo
    jid = _new_job(jobs, created)
    bad = [ValueBin(165, 0.64, 0.65, 10, 4000.0)]     # violates the edge check
    jobs.complete(jid, result={"coverage": {}}, cog_path="/x.tif", output_crs="EPSG:32632",
                  message="done", value_bins=bad)
    assert one(admin, "SELECT count(*) FROM index_value_bins WHERE job_id = %s", (str(jid),)) == 0
    assert q(admin, "SELECT status::text, message FROM jobs WHERE id = %s", (str(jid),)) == [("completed", "done")]


def test_seed_unchanged_by_the_writer_tests(admin):
    """The writer tests clean up after themselves (the counts the other tests read)."""
    assert one(admin, "SELECT count(*) FROM jobs") == 11
    assert one(admin, "SELECT count(*) FROM index_value_bins") == 39
