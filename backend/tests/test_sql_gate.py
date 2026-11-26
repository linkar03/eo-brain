"""
Tests for eo_brain.agent.sql_gate - the security-critical layer between
model-written SQL and the database. Offline: parsing only.
"""

import pytest

from eo_brain.agent.sql_gate import ALLOWED_FUNCTIONS, MAX_SQL_CHARS, SQLGate

VIEWS = ["analyses", "alert_areas", "analysis_severity_summary"]


@pytest.fixture(scope="module")
def gate():
    return SQLGate(VIEWS, max_limit=200)


def rejected(gate, sql, fragment):
    d = gate.check(sql)
    assert not d.ok, f"accepted: {sql!r}"
    assert fragment.lower() in d.reason.lower(), f"{sql!r}: reason {d.reason!r} lacks {fragment!r}"


# ---------------------------------------------------------------- accepted

ACCEPTED = [
    "SELECT count(*) FROM agent.analyses LIMIT 1",
    "select * from agent.analyses limit 200",
    "SELECT * FROM AGENT.ANALYSES LIMIT 5",  # unquoted identifiers fold to lower case
    'SELECT * FROM "agent"."analyses" LIMIT 5',
    "SELECT * FROM agent.analyses LIMIT 5;",
    "SELECT * FROM agent.analyses LIMIT 5 ;  \n",
    "SELECT * FROM agent.analyses LIMIT 5 OFFSET 10",
    "SELECT a.location_name, s.total_area_ha FROM agent.analyses a "
    "JOIN agent.analysis_severity_summary s ON s.analysis_id = a.analysis_id "
    "WHERE s.severity_label = 'alert' ORDER BY a.created_at DESC NULLS LAST LIMIT 2",
    "WITH last_flood AS (SELECT analysis_id FROM agent.analyses WHERE algorithm = 'flood' "
    "ORDER BY created_at DESC LIMIT 1) SELECT sum(area_ha) FROM agent.alert_areas "
    "WHERE analysis_id IN (SELECT analysis_id FROM last_flood) AND severity = 3 LIMIT 1",
    "WITH a AS (SELECT * FROM agent.analyses), b AS (SELECT * FROM a) SELECT * FROM b LIMIT 3",
    "WITH a AS (SELECT * FROM agent.analyses) SELECT algorithm FROM a UNION SELECT algorithm FROM a LIMIT 5",
    "SELECT algorithm FROM agent.analyses EXCEPT SELECT algorithm FROM agent.alert_areas LIMIT 5",
    "SELECT round(sum(area_ha)::numeric, 2), avg(area_m2), max(area_km2), min(area_km2), stddev(area_m2) "
    "FROM agent.alert_areas LIMIT 1",
    "SELECT count(*) FILTER (WHERE severity = 3) FROM agent.alert_areas LIMIT 1",
    "SELECT analysis_id, rank() OVER (PARTITION BY algorithm ORDER BY created_at DESC), "
    "row_number() OVER (ORDER BY created_at), lag(created_at) OVER (ORDER BY created_at) FROM agent.analyses LIMIT 5",
    "SELECT * FROM agent.analyses WHERE created_at > now() - interval '30 days' LIMIT 5",
    "SELECT date_trunc('month', created_at), extract(epoch FROM completed_at - created_at), age(created_at), "
    "current_date, current_timestamp FROM agent.analyses LIMIT 5",
    "SELECT CASE WHEN severity = 3 THEN 'alert' ELSE 'other' END, coalesce(quadrant_label, '-'), "
    "nullif(severity, 0), greatest(area_m2, 1), least(area_m2, 1), abs(-1), ceil(1.2), floor(1.2) "
    "FROM agent.alert_areas LIMIT 5",
    "SELECT * FROM agent.analyses a WHERE EXISTS (SELECT 1 FROM agent.alert_areas f "
    "WHERE f.analysis_id = a.analysis_id) AND a.algorithm IN ('flood', 'lisa') "
    "AND a.window_days BETWEEN 7 AND 365 AND a.location_name ILIKE '%testville%' "
    "AND a.completed_at IS NOT NULL AND a.status IS DISTINCT FROM 'failed' LIMIT 5",
    "SELECT ST_Area(footprint::geography) / 10000 AS ha FROM agent.analyses "
    "WHERE ST_Intersects(footprint, ST_MakeEnvelope(11.0, 44.0, 11.2, 44.2, 4326)) LIMIT 10",
    "SELECT a.analysis_id, b.analysis_id, ST_Area(ST_Intersection(a.footprint, b.footprint)::geography) "
    "FROM agent.analyses a JOIN agent.analyses b ON ST_Overlaps(a.footprint, b.footprint) "
    "AND a.analysis_id <> b.analysis_id LIMIT 20",
    "SELECT ST_X(ST_Centroid(geometry)), ST_Y(ST_Centroid(geometry)), ST_Distance(geometry::geography, "
    "ST_SetSRID(ST_MakePoint(11.0, 44.0), 4326)::geography) FROM agent.alert_areas LIMIT 3",
    "SELECT DISTINCT algorithm FROM agent.analyses LIMIT 10",
    "SELECT algorithm, count(*) FROM agent.analyses GROUP BY algorithm HAVING count(*) > 1 ORDER BY 2 DESC LIMIT 10",
    "SELECT (SELECT count(*) FROM agent.alert_areas) AS n FROM agent.analyses LIMIT 1",
    "WITH pg_class AS (SELECT * FROM agent.analyses) SELECT * FROM pg_class LIMIT 1",  # CTE shadows by name
]


@pytest.mark.parametrize("sql", ACCEPTED)
def test_accepted(gate, sql):
    d = gate.check(sql)
    assert d.ok, d.reason
    assert not d.sql.endswith(";")


# ---------------------------------------------------------------- stacked statements

@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM agent.analyses LIMIT 1; DROP TABLE jobs",
        "SELECT * FROM agent.analyses LIMIT 1;;",
        "SELECT 1 FROM agent.analyses LIMIT 1; SELECT 2 FROM agent.analyses LIMIT 1",
        "SELECT * FROM agent.analyses WHERE location_name = 'a;b' LIMIT 1",
        "; SELECT * FROM agent.analyses LIMIT 1",
    ],
)
def test_semicolons_anywhere_but_the_end(gate, sql):
    rejected(gate, sql, "multiple statements")


# ---------------------------------------------------------------- writes, DDL, other statements

@pytest.mark.parametrize(
    "sql,fragment",
    [
        ("INSERT INTO agent.analyses (algorithm) VALUES ('x')", "INSERT"),
        ("INSERT INTO jobs SELECT * FROM jobs", "INSERT"),
        ("UPDATE jobs SET status = 'failed'", "UPDATE"),
        ("DELETE FROM jobs", "DELETE"),
        ("DELETE FROM agent.analyses WHERE true", "DELETE"),
        ("MERGE INTO jobs j USING jobs s ON j.id = s.id WHEN MATCHED THEN DELETE", "only SELECT"),
        ("TRUNCATE jobs", "TRUNCATE"),
        ("CREATE TABLE t (a int)", "CREATE"),
        ("CREATE TEMP TABLE t AS SELECT * FROM agent.analyses", "CREATE"),
        ("CREATE VIEW agent.x AS SELECT 1", "CREATE"),
        ("DROP TABLE jobs", "DROP"),
        ("DROP SCHEMA agent CASCADE", "DROP"),
        ("ALTER TABLE jobs ADD COLUMN x int", "ALTER"),
        ("ALTER ROLE agent_reader SET default_transaction_read_only = off", "only SELECT"),
        ("GRANT ALL ON jobs TO PUBLIC", "only SELECT"),
        ("REVOKE ALL ON agent.analyses FROM PUBLIC", "only SELECT"),
        ("COPY jobs TO '/tmp/jobs.csv'", "COPY"),
        ("COPY (SELECT * FROM agent.analyses) TO PROGRAM 'sh'", "COPY"),
        ("COPY jobs FROM '/etc/passwd'", "COPY"),
        ("SET statement_timeout = 0", "SET"),
        ("SET SESSION AUTHORIZATION postgres", "only SELECT"),
        ("RESET ALL", "only SELECT"),
        ("SHOW ALL", "only SELECT"),
        ("BEGIN", "only SELECT"),
        ("COMMIT", "only SELECT"),
        ("VACUUM jobs", "only SELECT"),
        ("ANALYZE jobs", "only SELECT"),
        ("EXPLAIN ANALYZE SELECT * FROM agent.analyses LIMIT 1", "only SELECT"),
        ("CALL some_proc()", "only SELECT"),
        ("LISTEN chan", "only SELECT"),
        ("NOTIFY chan", "only SELECT"),
        ("PREPARE p AS SELECT 1", "only SELECT"),
        ("EXECUTE p", "only SELECT"),
        ("DECLARE c CURSOR FOR SELECT * FROM agent.analyses", "only SELECT"),
        ("LOCK TABLE jobs", "only SELECT"),
        ("REFRESH MATERIALIZED VIEW agent.x", "only SELECT"),
        ("COMMENT ON TABLE jobs IS 'x'", "only SELECT"),
        ("CHECKPOINT", "only SELECT"),
        ("VALUES (1)", "only SELECT"),
    ],
)
def test_non_select_statements(gate, sql, fragment):
    rejected(gate, sql, fragment)


@pytest.mark.parametrize(
    "sql,fragment",
    [
        ("SELECT * INTO new_table FROM agent.analyses LIMIT 1", "INTO"),
        ("SELECT * INTO TEMP t FROM agent.analyses LIMIT 1", "INTO"),
        ("SELECT * FROM agent.analyses LIMIT 1 FOR UPDATE", "locking"),
        ("SELECT * FROM agent.analyses LIMIT 1 FOR SHARE", "locking"),
    ],
)
def test_select_forms_with_side_effects(gate, sql, fragment):
    rejected(gate, sql, fragment)


# ---------------------------------------------------------------- CTEs hiding writes

@pytest.mark.parametrize(
    "sql",
    [
        "WITH d AS (DELETE FROM jobs RETURNING *) SELECT * FROM d LIMIT 1",
        "WITH u AS (UPDATE jobs SET status = 'failed' RETURNING id) SELECT * FROM u LIMIT 1",
        "WITH i AS (INSERT INTO jobs (algorithm) VALUES ('x') RETURNING id) SELECT * FROM i LIMIT 1",
        "WITH a AS (SELECT * FROM agent.analyses), d AS (DELETE FROM jobs RETURNING id) "
        "SELECT * FROM a LIMIT 1",
        "SELECT * FROM agent.analyses WHERE analysis_id IN "
        "(WITH d AS (DELETE FROM jobs RETURNING id) SELECT id FROM d) LIMIT 1",
    ],
)
def test_writes_hidden_in_ctes(gate, sql):
    d = gate.check(sql)
    assert not d.ok
    assert any(w in d.reason for w in ("DELETE", "UPDATE", "INSERT", "RETURNING", "VALUES", "write")), d.reason


def test_recursive_cte(gate):
    rejected(
        gate,
        "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT n FROM r LIMIT 5",
        "RECURSIVE",
    )


# ---------------------------------------------------------------- comments and quoting tricks

@pytest.mark.parametrize(
    "sql,fragment",
    [
        ("SELECT * FROM agent.analyses LIMIT 1 -- trailing", "comments"),
        ("SELECT * FROM agent.analyses /* c */ LIMIT 1", "comments"),
        ("SELECT * FROM agent.analyses /* /* nested */ */ LIMIT 1", "comments"),
        ("SELECT * FROM agent.analyses LIMIT 1 */", "comments"),
        ("SELECT * FROM agent.analyses WHERE location_name = '--' LIMIT 1", "comments"),
        ("SELECT 'a\\'; DELETE FROM jobs; --' FROM agent.analyses LIMIT 1", "multiple statements"),
        ("SELECT E'a\\'' FROM agent.analyses LIMIT 1", "backslash"),
        ("SELECT $$x$$ FROM agent.analyses LIMIT 1", "dollar"),
        ("SELECT $tag$x$tag$ FROM agent.analyses LIMIT 1", "dollar"),
        ("SELECT * FROM agent.analyses WHERE algorithm = $1 LIMIT 1", "dollar"),
        ("SELECT U&'d!0061t!+000061' UESCAPE '!' FROM agent.analyses LIMIT 1", "Unicode"),
        ("SELECT u&'\\0041' FROM agent.analyses LIMIT 1", "backslash"),
        ("SELECT * FROM agent.analyses\x00 LIMIT 1", "control"),
        ("SELECT * FROM agent.analyses\x0b LIMIT 1", "control"),
    ],
)
def test_lexical_tricks(gate, sql, fragment):
    rejected(gate, sql, fragment)


@pytest.mark.parametrize(
    "sql,fragment",
    [
        ('SELECT * FROM "Agent".analyses LIMIT 1', "not allowed"),          # quoted: case-sensitive
        ('SELECT * FROM agent."Analyses" LIMIT 1', "not allowed"),
        ('SELECT * FROM agent."analyses ""x" LIMIT 1', "not allowed"),
        ('SELECT * FROM "public"."jobs" LIMIT 1', "public.jobs"),
        ('SELECT * FROM "pg_catalog"."pg_roles" LIMIT 1', "pg_catalog.pg_roles"),
        ("SELECT * FROM agent.analyses_backup LIMIT 1", "not allowed"),
    ],
)
def test_identifier_tricks(gate, sql, fragment):
    rejected(gate, sql, fragment)


# ---------------------------------------------------------------- relations outside agent.*

@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM public.jobs LIMIT 1",
        "SELECT * FROM public.alert_features LIMIT 1",
        "SELECT * FROM public.alert_feature_stats LIMIT 1",
        "SELECT * FROM public.spatial_ref_sys LIMIT 1",
        "SELECT * FROM jobs LIMIT 1",
        "SELECT * FROM alert_features LIMIT 1",
        "TABLE jobs",
        "SELECT * FROM pg_catalog.pg_roles LIMIT 1",
        "SELECT * FROM pg_roles LIMIT 1",
        "SELECT * FROM pg_authid LIMIT 1",
        "SELECT * FROM pg_shadow LIMIT 1",
        "SELECT * FROM pg_stat_activity LIMIT 1",
        "SELECT * FROM information_schema.tables LIMIT 1",
        "SELECT * FROM information_schema.role_table_grants LIMIT 1",
        "SELECT * FROM tiger.geocode_settings LIMIT 1",
        "SELECT * FROM topology.topology LIMIT 1",
        "SELECT * FROM mydb.agent.analyses LIMIT 1",
        "SELECT * FROM agent.analyses a JOIN pg_class c ON true LIMIT 1",
        "SELECT * FROM agent.analyses, pg_user LIMIT 1",
        "SELECT * FROM (SELECT * FROM pg_class) s LIMIT 1",
        "SELECT (SELECT rolpassword FROM pg_authid LIMIT 1) FROM agent.analyses LIMIT 1",
        "SELECT * FROM agent.analyses WHERE EXISTS (SELECT 1 FROM pg_user) LIMIT 1",
        "SELECT * FROM agent.analyses WHERE analysis_id IN (SELECT id FROM public.jobs) LIMIT 1",
        "SELECT algorithm FROM agent.analyses UNION SELECT rolname FROM pg_roles LIMIT 1",
        "WITH x AS (SELECT * FROM pg_roles) SELECT * FROM x LIMIT 1",
        # CTE defined later is not visible to an earlier CTE: pg_roles below is the catalog table
        "WITH x AS (SELECT * FROM pg_roles), pg_roles AS (SELECT 1) SELECT * FROM x LIMIT 1",
        # CTE defined inside a subquery is not visible outside it
        "SELECT * FROM (WITH pg_class AS (SELECT 1) SELECT * FROM pg_class) s, pg_class LIMIT 1",
        # a CTE cannot see itself without RECURSIVE
        "WITH pg_roles AS (SELECT * FROM pg_roles) SELECT * FROM pg_roles LIMIT 1",
    ],
)
def test_relations_outside_agent_schema(gate, sql):
    d = gate.check(sql)
    assert not d.ok, f"accepted: {sql!r}"
    assert "not allowed" in d.reason or "relation" in d.reason, d.reason


def test_unqualified_view_name_gets_a_helpful_reason(gate):
    rejected(gate, "SELECT * FROM analyses LIMIT 1", "agent.analyses")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT pg_catalog.pg_roles.rolname FROM agent.analyses LIMIT 1",
        "SELECT agent.analyses.algorithm FROM agent.analyses LIMIT 1",
    ],
)
def test_overqualified_columns(gate, sql):
    rejected(gate, sql, "too many qualifiers")


# ---------------------------------------------------------------- functions

DANGEROUS = [
    "pg_sleep(10)", "pg_catalog.pg_sleep(10)", "pg_read_file('/etc/passwd')",
    "pg_read_binary_file('/etc/passwd')", "pg_ls_dir('.')", "pg_stat_file('x')",
    "dblink('host=x', 'select 1')", "dblink_exec('host=x', 'drop table jobs')",
    "set_config('statement_timeout', '0', false)", "current_setting('data_directory')",
    "lo_import('/etc/passwd')", "lo_export(1, '/tmp/x')", "lo_get(1)", "lo_from_bytea(0, 'x')",
    "pg_terminate_backend(1)", "pg_cancel_backend(1)", "pg_reload_conf()",
    "txid_current()", "nextval('s')", "setval('s', 1)", "query_to_xml('select 1', true, true, '')",
    "version()", "inet_server_addr()", "pg_backend_pid()", "has_table_privilege('jobs', 'select')",
    "format('%s', 1)", "concat('a', 'b')", "string_agg(algorithm, ',')", "array_agg(algorithm)",
    "to_json(algorithm)", "json_agg(algorithm)", "row_to_json(a)", "md5('x')", "random()",
    "ST_AsText(footprint)", "ST_AsGeoJSON(footprint)", "ST_AsBinary(footprint)", "ST_AsEWKT(footprint)",
    "ST_AsKML(footprint)", "ST_AsSVG(footprint)", "ST_AsMVT(a)", "ST_GeomFromText('POINT(0 0)')",
    "upper(location_name)", "lower(location_name)", "length(location_name)",
]


@pytest.mark.parametrize("call", DANGEROUS)
def test_functions_outside_allow_list(gate, call):
    d = gate.check(f"SELECT {call} FROM agent.analyses a LIMIT 1")
    assert not d.ok, call
    assert "not allowed" in d.reason, d.reason


@pytest.mark.parametrize("call", ["pg_sleep(10)", "set_config('a', 'b', false)", "dblink('a', 'b')"])
def test_functions_rejected_anywhere(gate, call):
    for sql in [
        f"SELECT * FROM agent.analyses WHERE {call} IS NULL LIMIT 1",
        f"SELECT * FROM agent.analyses ORDER BY {call} LIMIT 1",
        f"WITH x AS (SELECT {call}) SELECT * FROM agent.analyses LIMIT 1",
        f"SELECT * FROM agent.analyses GROUP BY {call} LIMIT 1",
    ]:
        assert not gate.check(sql).ok, sql


@pytest.mark.parametrize("word", ["current_user", "session_user", "user", "current_role", "current_schema"])
def test_niladic_keyword_functions(gate, word):
    rejected(gate, f"SELECT {word} FROM agent.analyses LIMIT 1", "not allowed")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM generate_series(1, 10) LIMIT 1",
        "SELECT * FROM agent.analyses a, LATERAL (SELECT 1) b LIMIT 1",
        "SELECT * FROM (VALUES (1), (2)) v(x) LIMIT 1",
    ],
)
def test_from_clause_constructs(gate, sql):
    assert not gate.check(sql).ok


def test_allow_list_has_no_output_or_admin_functions():
    for name in ALLOWED_FUNCTIONS:
        assert not name.startswith(("pg_", "lo_", "st_as", "dblink", "set_", "current_setting"))


# ---------------------------------------------------------------- casts and operators

@pytest.mark.parametrize(
    "expr",
    [
        "footprint::text", "CAST(footprint AS varchar)", "footprint::bytea", "algorithm::regclass",
        "analysis_id::oid", "footprint::json", "CAST(footprint AS char(10))",
    ],
)
def test_casts_outside_allow_list(gate, expr):
    rejected(gate, f"SELECT {expr} FROM agent.analyses LIMIT 1", "cast")


def test_string_concatenation(gate):
    rejected(gate, "SELECT footprint || '' FROM agent.analyses LIMIT 1", "concatenation")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT location_name ~ 'x' FROM agent.analyses LIMIT 1",
        "SELECT ARRAY[1, 2] FROM agent.analyses LIMIT 1",
        "SELECT footprint[1] FROM agent.analyses LIMIT 1",
    ],
)
def test_other_operators(gate, sql):
    assert not gate.check(sql).ok


# ---------------------------------------------------------------- LIMIT

@pytest.mark.parametrize(
    "sql,fragment",
    [
        ("SELECT * FROM agent.analyses", "needs LIMIT"),
        ("SELECT count(*) FROM agent.analyses", "needs LIMIT"),
        ("SELECT * FROM agent.analyses LIMIT 0", "between 1 and 200"),
        ("SELECT * FROM agent.analyses LIMIT 201", "between 1 and 200"),
        ("SELECT * FROM agent.analyses LIMIT 1000000", "between 1 and 200"),
        ("SELECT * FROM agent.analyses LIMIT ALL", "literal integer"),
        ("SELECT * FROM agent.analyses LIMIT NULL", "literal integer"),
        ("SELECT * FROM agent.analyses LIMIT '5'", "literal integer"),
        ("SELECT * FROM agent.analyses LIMIT 5 + 1", "literal integer"),
        ("SELECT * FROM agent.analyses LIMIT -1", "literal integer"),
        ("SELECT * FROM agent.analyses LIMIT (SELECT 1)", "literal integer"),
        ("SELECT * FROM agent.analyses LIMIT 1.5", "literal integer"),
        ("SELECT * FROM (SELECT * FROM agent.analyses LIMIT 5) s", "needs LIMIT"),
        ("SELECT * FROM agent.analyses LIMIT 5 OFFSET 100000", "OFFSET"),
        ("SELECT * FROM agent.analyses FETCH FIRST 5 ROWS ONLY", "FETCH"),
        ("SELECT a FROM agent.analyses UNION SELECT b FROM agent.alert_areas", "needs LIMIT"),
    ],
)
def test_limit_rules(gate, sql, fragment):
    rejected(gate, sql, fragment)


def test_custom_limit_cap():
    g = SQLGate(VIEWS, max_limit=10)
    assert g.check("SELECT * FROM agent.analyses LIMIT 10").ok
    assert not g.check("SELECT * FROM agent.analyses LIMIT 11").ok


# ---------------------------------------------------------------- malformed input

@pytest.mark.parametrize(
    "sql,fragment",
    [
        ("", "empty"),
        ("   ", "empty"),
        (";", "empty"),
        (None, "string"),
        (123, "string"),
        (["SELECT 1"], "string"),
        ("SELEC * FROM agent.analyses LIMIT 1", "parse"),
        ("SELECT * FROM agent.analyses WHERE LIMIT 1", "parse"),
        ("SELECT (((( FROM agent.analyses LIMIT 1", "parse"),
        ("SELECT 'unterminated FROM agent.analyses LIMIT 1", "parse"),
        ("SELECT " + "1 + " * 2000 + "1 FROM agent.analyses LIMIT 1", "longer than"),
    ],
)
def test_malformed(gate, sql, fragment):
    rejected(gate, sql, fragment)


def test_max_length_boundary(gate):
    template = "SELECT * FROM agent.analyses WHERE algorithm = '{}' LIMIT 1"
    filler = "x" * (MAX_SQL_CHARS - len(template.format("")))
    assert gate.check(template.format(filler)).ok
    assert not gate.check(template.format(filler + "x")).ok


def test_requires_allowed_relations():
    with pytest.raises(ValueError):
        SQLGate([])
