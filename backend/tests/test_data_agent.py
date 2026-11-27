"""
Offline tests for the text-to-SQL sub-agent (eo_brain.agent.data_agent), the
result shaping in eo_brain.agent.sql_exec, and the query_database tool of the
orchestrator. Scripted model, fake executor: no network, no database.
"""

import datetime as dt
import decimal
import uuid

import pytest

from eo_brain.agent.brain import Agent
from eo_brain.agent.data_agent import MAX_SQL_ATTEMPTS, DataAgent
from eo_brain.agent.llm import ModelTurn, ProviderError, ToolCall, ToolResults, UserMessage
from eo_brain.agent.providers.scripted import ScriptedProvider, relay_tool_results
from eo_brain.agent.session import SessionStore
from eo_brain.agent.sql_exec import ColumnDoc, QueryError, QueryResult, SchemaDoc, ViewDoc, shape_result
from evals.stubs import FIXED_NOW, InMemoryJobRepository, StubGeocoder

DOC = SchemaDoc(
    schema="agent",
    views=[
        ViewDoc("analyses", "One row per analysis.", [
            ColumnDoc("analysis_id", "uuid", "Unique id."),
            ColumnDoc("algorithm", "text", "Algorithm code."),
            ColumnDoc("footprint", "geometry(Polygon,4326)", "Never returned."),
        ]),
        ViewDoc("analysis_severity_summary", "Per severity.", [
            ColumnDoc("severity_label", "text", ""),
            ColumnDoc("total_area_ha", "double precision", "Hectares."),
        ]),
    ],
)

GOOD_SQL = "SELECT severity_label, total_area_ha FROM agent.analysis_severity_summary LIMIT 5"
ROWS = QueryResult(columns=["severity_label", "total_area_ha"], rows=[["alert", 12.0]], row_count=1, truncated=False)


class FakeExecutor:
    max_rows = 100

    def __init__(self, *responses, doc=DOC):
        self.responses = list(responses)
        self.executed = []
        self.doc = doc

    def execute(self, sql):
        self.executed.append(sql)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def load_schema_doc(self):
        if isinstance(self.doc, Exception):
            raise self.doc
        return self.doc


def run_sql(sql):
    return ModelTurn(tool_calls=[ToolCall("run_sql", {"sql": sql})])


def refuse(reason):
    return ModelTurn(tool_calls=[ToolCall("refuse", {"reason": reason})])


def make(steps, *responses, models=("m",), **kw):
    provider = ScriptedProvider(steps)
    executor = FakeExecutor(*responses, **kw)
    return DataAgent(provider, models=list(models), executor=executor), provider, executor


# ---------------------------------------------------------------- data agent loop

def test_success():
    agent, provider, executor = make([run_sql(GOOD_SQL + ";")], ROWS)
    res = agent.answer("quanti ettari in allerta?")
    assert res.status == "ok"
    assert res.sql == GOOD_SQL and executor.executed == [GOOD_SQL]
    assert res.rows == [["alert", 12.0]] and res.row_count == 1 and not res.truncated
    assert len(res.attempts) == 1 and res.model == "m"


def test_isolated_context_and_tools():
    agent, provider, _ = make([run_sql(GOOD_SQL)], ROWS)
    agent.answer("how many hectares?")
    [call] = provider.calls
    assert call.messages == [UserMessage(text="how many hectares?")]
    assert call.tool_names == ["run_sql", "refuse"]
    assert "agent.analyses: One row per analysis." in call.system_prompt
    assert "footprint (geometry(Polygon,4326)): Never returned." in call.system_prompt
    assert "LIMIT n, 1 <= n <= 100" in call.system_prompt
    assert "EO Brain" not in call.system_prompt  # not the orchestrator's prompt


def test_prompt_asks_for_the_threshold_as_a_column():
    """A figure filtered on an index value reaches the orchestrator with its threshold in the rows."""
    agent, _, _ = make([])
    flat = " ".join(agent.system_prompt().split())
    rule = flat.split(" 9. ")[1].split(" VIEWS ")[0]
    assert "filter on an index value (value_from, value_to)" in rule
    assert "return the threshold you applied as a column" in rule
    assert "SELECT 0.65 AS threshold" in rule
    assert "both bounds" in rule
    assert "default documented on agent.index_value_distribution" in rule
    # rule 6 (only the columns needed) must not read as "drop the threshold"
    assert "a threshold you filtered on is needed" in flat.split(" 6. ")[1].split(" 7. ")[0]


def test_prompt_sends_areas_to_the_class_view_and_counts_with_the_zone_size():
    """Flood / lisa: hectares and shares from agent.severity_areas; a zone count with its minimum size."""
    agent, _, _ = make([])
    flat = " ".join(agent.system_prompt().split())
    rule = flat.split(" 10. ")[1].split(" VIEWS ")[0]
    assert "hectares and shares come from agent.severity_areas" in rule
    assert "only count and locate mapped zones" in rule
    assert "also return min_zone_area_m2 and small_patches_left_out of agent.analyses" in rule
    # and rule 6 keeps those columns
    assert "so is the minimum zone size of a zone count: rule 10" in flat.split(" 6. ")[1].split(" 7. ")[0]


def test_area_hint_is_appended_by_code():
    agent, provider, _ = make([run_sql(GOOD_SQL)], ROWS)
    agent.answer("which analyses overlap this area?", area_hint="bbox west=1 south=2 east=3 north=4")
    text = provider.calls[0].messages[0].text
    assert text.startswith("which analyses overlap this area?")
    assert "bbox west=1 south=2 east=3 north=4" in text


def test_refusal_never_touches_the_database():
    agent, _, executor = make([refuse("I can only read results.")])
    res = agent.answer("delete everything")
    assert res.status == "refused" and res.reason == "I can only read results."
    assert executor.executed == []


def test_gate_rejection_is_fed_back_then_success():
    agent, provider, executor = make([run_sql("SELECT * FROM analyses LIMIT 5"), run_sql(GOOD_SQL)], ROWS)
    res = agent.answer("q")
    assert res.status == "ok"
    assert [a.stage for a in res.attempts] == ["gate", ""]
    assert executor.executed == [GOOD_SQL]  # rejected SQL never executed
    feedback = provider.calls[1].messages[-1]
    assert isinstance(feedback, ToolResults)
    assert "rejected by SQL gate" in feedback.results[0].response["error"]
    assert "agent.analyses" in feedback.results[0].response["error"]


def test_database_error_is_fed_back_then_success():
    agent, provider, executor = make(
        [run_sql("SELECT nope FROM agent.analyses LIMIT 1"), run_sql(GOOD_SQL)],
        QueryError('database error 42703: column "nope" does not exist'), ROWS,
    )
    res = agent.answer("q")
    assert res.status == "ok"
    assert [a.stage for a in res.attempts] == ["database", ""]
    assert "column \"nope\"" in provider.calls[1].messages[-1].results[0].response["error"]


def test_retries_are_bounded_and_failure_is_honest():
    bad = "DELETE FROM jobs"
    agent, provider, executor = make([run_sql(bad)] * 10)
    res = agent.answer("q")
    assert res.status == "failed"
    assert len(provider.calls) == MAX_SQL_ATTEMPTS == 3
    assert executor.executed == []
    assert "gave up after 3 attempts" in res.error and "DELETE" in res.error
    assert res.gate_rejections == 3
    assert res.to_tool_response() == {"status": "failed", "error": res.error}


def test_database_errors_exhaust_retries():
    err = QueryError("query cancelled after 3000 ms (statement timeout)")
    agent, _, executor = make([run_sql(GOOD_SQL)] * 3, err, err, err)
    res = agent.answer("q")
    assert res.status == "failed" and "statement timeout" in res.error
    assert len(executor.executed) == 3


def test_text_instead_of_tool_call_counts_as_attempt():
    agent, provider, _ = make([ModelTurn(text="Here is the SQL: ..."), run_sql(GOOD_SQL)], ROWS)
    res = agent.answer("q")
    assert res.status == "ok" and [a.stage for a in res.attempts] == ["protocol", ""]
    assert provider.calls[1].messages[-1] == UserMessage(text="Answer by calling run_sql or refuse.")


def test_no_candidate_counts_as_attempt():
    agent, _, _ = make([None, None, None])
    res = agent.answer("q")
    assert res.status == "failed" and "no tool call" in res.error


def test_unknown_tool_counts_as_attempt():
    agent, _, _ = make([ModelTurn(tool_calls=[ToolCall("drop_table", {})]), run_sql(GOOD_SQL)], ROWS)
    res = agent.answer("q")
    assert res.status == "ok" and res.attempts[0].error == "unknown tool drop_table"


def test_only_first_call_of_a_turn_is_processed():
    turn = ModelTurn(tool_calls=[ToolCall("run_sql", {"sql": "SELECT * FROM jobs LIMIT 1"}),
                                 ToolCall("run_sql", {"sql": GOOD_SQL})])
    agent, provider, executor = make([turn, run_sql(GOOD_SQL)], ROWS)
    res = agent.answer("q")
    assert res.status == "ok" and executor.executed == [GOOD_SQL]
    results = provider.calls[1].messages[-1].results
    assert results[1].response == {"error": "only one tool call per turn is processed"}


def test_provider_failure_falls_back_then_fails():
    agent, provider, _ = make([ProviderError("down"), run_sql(GOOD_SQL)], ROWS, models=("a", "b"))
    assert agent.answer("q").status == "ok"
    assert [c.model for c in provider.calls] == ["a", "b"]

    agent, _, _ = make([ProviderError("down"), ProviderError("down")], models=("a", "b"))
    res = agent.answer("q")
    assert res.status == "failed" and "model unavailable" in res.error


def test_schema_unavailable():
    agent, provider, _ = make([], doc=QueryError("database unavailable: timeout"))
    res = agent.answer("q")
    assert res.status == "failed" and "database unavailable" in res.error
    assert provider.calls == []


@pytest.mark.parametrize("question", ["", "   ", None])
def test_empty_question(question):
    agent, provider, _ = make([])
    assert agent.answer(question).status == "failed"
    assert provider.calls == []


def test_gate_comes_from_the_schema_doc():
    # alert_areas exists in the real database but not in this doc -> rejected
    agent, _, executor = make([run_sql("SELECT * FROM agent.alert_areas LIMIT 1")] * 3)
    assert agent.answer("q").status == "failed"
    assert executor.executed == []


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM agent.analyses LIMIT 1; DELETE FROM jobs",
        "WITH d AS (DELETE FROM jobs RETURNING *) SELECT * FROM d LIMIT 1",
        "SELECT pg_sleep(30) FROM agent.analyses LIMIT 1",
        "SELECT * FROM pg_catalog.pg_authid LIMIT 1",
        "SELECT ST_AsText(footprint) FROM agent.analyses LIMIT 1",
        "COPY jobs TO '/tmp/x'",
    ],
)
def test_malicious_sql_never_reaches_the_executor(sql):
    agent, _, executor = make([run_sql(sql)] * 3)
    assert agent.answer("q").status == "failed"
    assert executor.executed == []


def test_payload_shapes():
    agent, _, _ = make([run_sql(GOOD_SQL)], ROWS)
    res = agent.answer("q")
    assert res.to_tool_response() == {
        "status": "ok", "sql": GOOD_SQL, "columns": ["severity_label", "total_area_ha"],
        "rows": [["alert", 12.0]], "row_count": 1, "truncated": False,
    }
    assert res.to_api_payload() == {
        "sql": GOOD_SQL, "columns": ["severity_label", "total_area_ha"],
        "rows": [["alert", 12.0]], "row_count": 1, "truncated": False,
    }


# ---------------------------------------------------------------- result shaping

GEOM_OID = 99999  # dynamic OID of the PostGIS type in a real database


def test_shape_result_redacts_non_scalar_types():
    desc = [("id", 2950), ("footprint", GEOM_OID), ("area", 1700), ("at", 1184), ("d", 1186)]
    fetched = [(uuid.UUID(int=1), "0103000020E6...", decimal.Decimal("12.5"),
                dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc), dt.timedelta(seconds=95))]
    r = shape_result(desc, fetched, max_rows=10, max_payload_chars=10_000, type_names={GEOM_OID: "geometry"})
    assert r.rows == [["00000000-0000-0000-0000-000000000001", "<geometry omitted>", 12.5,
                       "2026-01-01T00:00:00+00:00", "95 seconds"]]
    assert r.omitted_columns == ["footprint (geometry)"]


def test_shape_result_row_cap():
    r = shape_result([("n", 23)], [(i,) for i in range(11)], max_rows=10, max_payload_chars=10_000)
    assert r.row_count == 10 and r.truncated


def test_shape_result_payload_cap():
    rows = [("x" * 200,) for _ in range(50)]
    r = shape_result([("t", 25)], rows, max_rows=100, max_payload_chars=1_000)
    assert r.truncated and 0 < r.row_count < 50


def test_shape_result_long_text_is_clipped():
    r = shape_result([("t", 25)], [("y" * 1000,)], max_rows=1, max_payload_chars=10_000)
    assert len(r.rows[0][0]) == 303 and r.rows[0][0].endswith("...")


def test_schema_doc_render():
    text = DOC.render()
    assert text.splitlines()[0] == "agent.analyses: One row per analysis."
    assert "  - algorithm (text): Algorithm code." in text
    assert DOC.relation_names() == ["analyses", "analysis_severity_summary"]


# ---------------------------------------------------------------- orchestrator integration

GAZ = [{"name": "Firenze, Toscana, Italia", "aliases": ["firenze"], "bbox": [11.15, 43.72, 11.33, 43.83]}]


def orchestrator(steps, data_agent=None):
    provider = ScriptedProvider(steps)
    agent = Agent(
        provider, models=["m"], job_repo=InMemoryJobRepository(), geocode=StubGeocoder(GAZ),
        sessions=SessionStore(), clock=lambda: FIXED_NOW, sleep=lambda s: None, data_agent=data_agent,
    )
    return agent, provider


def query(question):
    return ModelTurn(tool_calls=[ToolCall("query_database", {"question": question})])


def test_query_database_not_configured():
    agent, provider = orchestrator([query("quante analisi?"), relay_tool_results])
    res = agent.run("quante analisi?", "s1")
    assert res.text == "Querying stored results is not configured on this server."
    assert res.query is None


def test_query_database_round_trip():
    data_agent, sub_provider, _ = make([run_sql(GOOD_SQL)], ROWS)
    agent, provider = orchestrator([query("ettari in allerta?"), ModelTurn(text="12 ettari.")], data_agent)
    res = agent.run("quanti ettari in allerta nell'ultima alluvione?", "s1")
    assert res.text == "12 ettari."
    assert res.query == {"sql": GOOD_SQL, "columns": ["severity_label", "total_area_ha"],
                         "rows": [["alert", 12.0]], "row_count": 1, "truncated": False}
    assert res.to_api_dict()["query"] == res.query
    assert res.job_id is None
    [tool_result] = [r for m in provider.calls[1].messages if isinstance(m, ToolResults) for r in m.results]
    assert tool_result.response["rows"] == [["alert", 12.0]]
    # the sub-agent saw the question only, not the orchestrator's conversation
    assert sub_provider.calls[0].messages == [UserMessage(text="ettari in allerta?")]


def test_query_database_refusal_has_no_payload():
    data_agent, _, _ = make([refuse("read-only")])
    agent, provider = orchestrator([query("cancella tutto"), relay_tool_results], data_agent)
    res = agent.run("cancella tutto", "s1")
    assert res.query is None
    [r] = [r for m in provider.calls[1].messages if isinstance(m, ToolResults) for r in m.results]
    assert r.response == {"status": "refused", "reason": "read-only"}


def test_query_database_gets_session_area_hint():
    data_agent, sub_provider, _ = make([run_sql(GOOD_SQL)], ROWS)
    agent, _ = orchestrator(
        [ModelTurn(tool_calls=[ToolCall("launch_analysis", {"location": "Firenze", "algorithm": "flood"})]),
         relay_tool_results, query("which analyses overlap this area?"), relay_tool_results],
        data_agent,
    )
    agent.run("flood firenze", "s1")
    agent.run("which analyses overlap this area?", "s1")
    text = sub_provider.calls[0].messages[0].text
    assert "Firenze, Toscana, Italia" in text and "west=11.15" in text and "north=43.83" in text
