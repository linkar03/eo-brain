"""
Unit tests for the function-calling loop in eo_brain.agent.brain.

The model is a ScriptedProvider, the geocoder and the job repository are
in-memory stubs: no network, no database, no API key.
"""

from datetime import timedelta
from uuid import UUID

import pytest

from eo_brain.agent import brain
from eo_brain.agent.brain import Agent, SYSTEM_PROMPT, TOOL_SPECS
from eo_brain.agent.llm import (
    ModelTurn,
    ProviderError,
    RateLimitError,
    ToolCall,
    ToolResults,
    UserMessage,
)
from eo_brain.agent.providers.scripted import ScriptedProvider, relay_tool_results
from eo_brain.agent.session import SessionStore
from evals.stubs import FIXED_NOW, InMemoryJobRepository, StubGeocoder

GAZETTEER = [
    {"name": "Firenze, Toscana, Italia", "aliases": ["firenze", "florence"], "bbox": [11.15, 43.72, 11.33, 43.83]},
    {"name": "Pisa, Toscana, Italia", "aliases": ["pisa"], "bbox": [10.33, 43.66, 10.48, 43.76]},
    {"name": "Bologna, Emilia-Romagna, Italia", "aliases": ["bologna"], "bbox": [11.23, 44.42, 11.43, 44.56]},
]


def call(name, **args):
    return ModelTurn(tool_calls=[ToolCall(name=name, args=args)])


def text(t):
    return ModelTurn(text=t)


class Harness:
    def __init__(self, steps=(), models=("model-a",), **kwargs):
        self.provider = ScriptedProvider(steps)
        self.repo = kwargs.pop("job_repo", None) or InMemoryJobRepository()
        self.geocoder = StubGeocoder(GAZETTEER)
        self.sessions = SessionStore()
        self.sleeps = []
        self.agent = Agent(
            self.provider,
            models=list(models),
            job_repo=self.repo,
            geocode=self.geocoder,
            sessions=self.sessions,
            clock=lambda: FIXED_NOW,
            sleep=self.sleeps.append,
            **kwargs,
        )

    def add(self, *steps):
        self.provider.steps.extend(steps)

    def tool_results_seen(self, call_index):
        """ToolResults the model received on its call_index-th generate() call."""
        msgs = self.provider.calls[call_index].messages
        return [r for m in msgs if isinstance(m, ToolResults) for r in m.results]


# ---------------------------------------------------------------- dispatch

def test_launch_analysis_creates_job_and_feeds_result_back():
    h = Harness([call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=90), relay_tool_results])
    res = h.agent.run("ndvi firenze ultimi 3 mesi", "s1")

    [job] = h.repo.all()
    assert res.job_id == str(job.id)
    assert res.bbox == pytest.approx([11.15, 43.72, 11.33, 43.83])
    assert job.algorithm == "ndvi" and job.provider == "element84-s2"
    assert job.end_date == FIXED_NOW
    assert job.end_date - job.start_date == timedelta(days=90)
    assert job.end_date.tzinfo is not None  # timezone-aware, no utcnow()
    assert job.params == {"resolution_m": 20, "geocoded_name": "Firenze, Toscana, Italia", "apply_cloud_mask": True}

    [tool_result] = h.tool_results_seen(1)
    assert tool_result.name == "launch_analysis"
    assert tool_result.response["job_id"] == res.job_id
    assert tool_result.response["days_back"] == 90
    assert res.text == tool_result.response["message"]
    assert [c.name for c in res.tool_calls] == ["launch_analysis"]
    assert res.model == "model-a"


def test_provider_receives_system_prompt_and_all_tools():
    h = Harness([text("ciao")])
    h.agent.run("ciao", "s1")
    rec = h.provider.calls[0]
    assert rec.system_prompt == SYSTEM_PROMPT
    assert rec.tool_names == [t.name for t in TOOL_SPECS]
    assert rec.messages == [UserMessage(text="ciao")]


def test_alias_and_default_window():
    h = Harness([call("launch_analysis", location="Pisa", algorithm="vegetazione"), relay_tool_results])
    h.agent.run("vegetazione pisa", "s1")
    [job] = h.repo.all()
    assert job.algorithm == "ndvi"
    assert job.end_date - job.start_date == timedelta(days=30)


@pytest.mark.parametrize("days", [90.0, "90"])
def test_days_back_number_types_are_accepted(days):
    h = Harness([call("launch_analysis", location="Pisa", algorithm="ndvi", days_back=days), relay_tool_results])
    assert h.agent.run("x", "s1").job_id is not None
    [job] = h.repo.all()
    assert job.end_date - job.start_date == timedelta(days=90)


def test_days_back_null_means_default():
    h = Harness([call("launch_analysis", location="Pisa", algorithm="ndvi", days_back=None), relay_tool_results])
    h.agent.run("x", "s1")
    [job] = h.repo.all()
    assert job.end_date - job.start_date == timedelta(days=30)


def test_list_capabilities():
    h = Harness([call("list_capabilities"), relay_tool_results])
    h.agent.run("cosa sai fare?", "s1")
    [r] = h.tool_results_seen(1)
    assert sorted(a["name"] for a in r.response["algorithms"]) == ["flood", "lisa", "mndwi", "ndvi", "ndwi"]


def test_multiple_calls_in_one_turn_all_dispatched():
    h = Harness([
        ModelTurn(tool_calls=[
            ToolCall("launch_analysis", {"location": "Firenze", "algorithm": "ndvi"}),
            ToolCall("launch_analysis", {"location": "Pisa", "algorithm": "flood"}),
        ]),
        relay_tool_results,
    ])
    res = h.agent.run("ndvi firenze e flood pisa", "s1")
    jobs = h.repo.all()
    assert len(jobs) == 2
    assert res.job_id == str(jobs[-1].id)  # last created job is reported
    assert len(h.tool_results_seen(1)) == 2


# ---------------------------------------------------------------- errors relayed

def test_unknown_tool_is_reported_to_model():
    h = Harness([call("delete_everything", confirm=True), relay_tool_results])
    res = h.agent.run("x", "s1")
    [r] = h.tool_results_seen(1)
    assert r.response == {"error": "Unknown function: delete_everything"}
    assert res.job_id is None
    assert res.text == "Unknown function: delete_everything"


@pytest.mark.parametrize(
    "args,fragment",
    [
        ({"location": "Firenze", "algorithm": "traffic"}, "Unknown algorithm 'traffic'"),
        ({"location": "Firenze", "algorithm": "ndvi", "days_back": 400}, "exceeds maximum"),
        ({"location": "Firenze", "algorithm": "ndvi", "days_back": 0}, "must be >= 1"),
        ({"location": "xqzvbnm", "algorithm": "ndvi"}, "Could not geocode"),
        ({"location": "  ", "algorithm": "ndvi"}, "Location is empty"),
    ],
)
def test_validation_errors_are_relayed_and_create_no_job(args, fragment):
    h = Harness([call("launch_analysis", **args), relay_tool_results])
    res = h.agent.run("x", "s1")
    [r] = h.tool_results_seen(1)
    assert fragment in r.response["error"]
    assert fragment in res.text
    assert res.job_id is None and h.repo.all() == []


def test_handler_exception_is_relayed():
    h = Harness([call("launch_analysis", location="Firenze", algorithm="ndvi", resolution=5), relay_tool_results])
    h.agent.run("x", "s1")
    [r] = h.tool_results_seen(1)
    assert r.response["error"].startswith("Internal error in launch_analysis:")


def test_repository_failure_is_relayed():
    class BrokenRepo(InMemoryJobRepository):
        def create(self, **kwargs):
            raise RuntimeError("db down")

    h = Harness([call("launch_analysis", location="Firenze", algorithm="ndvi"), relay_tool_results], job_repo=BrokenRepo())
    res = h.agent.run("x", "s1")
    assert res.text == "Failed to create job: db down"
    assert res.job_id is None


# ---------------------------------------------------------------- needs-user guard
#
# After a job-creating call is rejected for a reason only the user can resolve,
# no further job-creating call may execute in the same request.

LIMIT_730 = "days_back=730 exceeds maximum"


def test_window_rejected_at_730_then_retried_at_365_creates_no_job():
    # The exact live sequence of eval case it-refuse-window-2-anni.
    h = Harness([
        call("launch_analysis", location="Bologna", algorithm="flood", days_back=730),
        call("launch_analysis", location="Bologna", algorithm="flood", days_back=365),
        relay_tool_results,
    ])
    res = h.agent.run("alluvioni a Bologna negli ultimi 2 anni", "s1")

    assert h.repo.all() == [] and res.job_id is None
    first, second = res.tool_calls
    assert first.args["days_back"] == 730 and LIMIT_730 in first.result["error"]
    assert first.result["needs_user"] is True and "Only the user" in first.result["action_required"]
    assert second.args["days_back"] == 365
    assert second.result["blocked"] is True and second.result["error"] == first.result["error"]
    assert "NOT EXECUTED" in second.result["action_required"]
    assert h.geocoder.queries == []  # the blocked call never reached the geocoder
    [seen] = h.tool_results_seen(2)[1:]
    assert seen.response["blocked"] is True  # the model is told why
    assert LIMIT_730 in res.text


def test_second_call_in_the_same_round_is_blocked():
    h = Harness([
        ModelTurn(tool_calls=[
            ToolCall("launch_analysis", {"location": "Bologna", "algorithm": "flood", "days_back": 730}),
            ToolCall("launch_analysis", {"location": "Bologna", "algorithm": "flood", "days_back": 365}),
        ]),
        relay_tool_results,
    ])
    res = h.agent.run("alluvioni a Bologna negli ultimi 2 anni", "s1")
    assert h.repo.all() == [] and res.job_id is None
    assert [c.result.get("blocked", False) for c in res.tool_calls] == [False, True]


def test_rerun_is_blocked_too_after_a_needs_user_rejection():
    h = Harness([
        call("launch_analysis", location="Pisa", algorithm="flood"), relay_tool_results,
        call("launch_analysis", location="Pisa", algorithm="ndvi", days_back=1825),
        call("rerun_with_different_algorithm", algorithm="ndvi"),
        relay_tool_results,
    ])
    h.agent.run("flood pisa", "s1")
    res = h.agent.run("ndvi stessa zona ultimi 5 anni", "s1")
    assert len(h.repo.all()) == 1 and res.job_id is None
    assert res.tool_calls[1].name == "rerun_with_different_algorithm"
    assert res.tool_calls[1].result["blocked"] is True


def test_no_previous_area_blocks_a_launch_on_an_invented_place():
    h = Harness([
        call("rerun_with_different_algorithm", algorithm="flood"),
        call("launch_analysis", location="Firenze", algorithm="flood"),
        relay_tool_results,
    ])
    res = h.agent.run("stessa zona ma con alluvione", "fresh-session")
    assert h.repo.all() == [] and res.job_id is None
    assert res.tool_calls[0].result["needs_user"] is True
    assert res.tool_calls[1].result["blocked"] is True
    assert res.text == "No previous analysis in this session. Specify a location."


def test_empty_location_blocks_an_invented_place():
    h = Harness([
        call("launch_analysis", location="", algorithm="ndvi"),
        call("launch_analysis", location="Firenze", algorithm="ndvi"),
        relay_tool_results,
    ])
    res = h.agent.run("fammi un'analisi ndvi", "s1")
    assert h.repo.all() == [] and res.tool_calls[1].result["blocked"] is True


@pytest.mark.parametrize(
    "bad,fixed,fragment",
    [
        ({"location": "Firenze", "algorithm": "salute colture"}, {"location": "Firenze", "algorithm": "ndvi"},
         "Unknown algorithm"),
        ({"location": "Fiorenze", "algorithm": "ndvi"}, {"location": "Firenze", "algorithm": "ndvi"},
         "Could not geocode"),
        ({"location": "Firenze", "algorithm": "ndvi", "days_back": "three months"},
         {"location": "Firenze", "algorithm": "ndvi", "days_back": 90}, "whole number"),
        ({"location": "Firenze", "algorithm": "ndvi", "days_back": 0},
         {"location": "Firenze", "algorithm": "ndvi", "days_back": 1}, "must be >= 1"),
        ({"location": "Firenze " + "x" * 200, "algorithm": "ndvi"}, {"location": "Firenze", "algorithm": "ndvi"},
         "too long"),
    ],
    ids=["unknown-alias", "geocode-miss", "days-format", "days-zero", "location-too-long"],
)
def test_model_fixable_rejection_allows_the_corrected_retry(bad, fixed, fragment):
    h = Harness([call("launch_analysis", **bad), call("launch_analysis", **fixed), relay_tool_results])
    res = h.agent.run("x", "s1")
    first, second = res.tool_calls
    assert fragment in first.result["error"]
    assert "needs_user" not in first.result and "blocked" not in second.result
    [job] = h.repo.all()
    assert res.job_id == str(job.id) and job.algorithm == "ndvi"


def test_non_job_tools_still_run_after_a_block():
    h = Harness([
        call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=730),
        call("list_capabilities"),
        relay_tool_results,
    ])
    res = h.agent.run("x", "s1")
    assert "algorithms" in res.tool_calls[1].result


def test_block_lasts_for_one_request_only():
    h = Harness([
        call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=730), relay_tool_results,
        call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=365), relay_tool_results,
    ])
    assert h.agent.run("ndvi firenze ultimi 2 anni", "s1").job_id is None
    # The user now asks for one year themselves: a new request, so it runs.
    assert h.agent.run("ok, allora l'ultimo anno", "s1").job_id is not None


def test_block_survives_a_retry_of_the_same_request():
    h = Harness([
        call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=730),
        RateLimitError("429"),
        call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=365),
        relay_tool_results,
    ])
    res = h.agent.run("ndvi firenze ultimi 2 anni", "s1")
    assert h.repo.all() == [] and res.job_id is None
    assert res.tool_calls[-1].result["blocked"] is True


def test_round_cap_after_a_block_reports_the_limit():
    steps = [call("launch_analysis", location="Firenze", algorithm="ndvi", days_back=d)
             for d in (730, 365, 300, 200, 100)]
    h = Harness(steps, max_tool_rounds=5)
    res = h.agent.run("x", "s1")
    assert h.repo.all() == []
    assert res.text.startswith(LIMIT_730)


def test_job_created_before_a_limit_is_kept_and_reported():
    h = Harness([
        call("launch_analysis", location="Firenze", algorithm="ndvi"),
        call("launch_analysis", location="Pisa", algorithm="ndvi", days_back=730),
        call("launch_analysis", location="Pisa", algorithm="ndvi", days_back=365),
        relay_tool_results,
    ])
    res = h.agent.run("ndvi firenze, e pisa ultimi 2 anni", "s1")
    [job] = h.repo.all()
    assert res.job_id == str(job.id) and job.params["geocoded_name"].startswith("Firenze")


# ---------------------------------------------------------------- loop limits

def test_max_tool_rounds_cap():
    h = Harness([call("list_capabilities") for _ in range(10)], max_tool_rounds=5)
    res = h.agent.run("x", "s1")
    assert len(h.provider.calls) == 5
    assert len(res.tool_calls) == 5
    assert res.text == "Analysis request is being processed."


def test_no_candidate():
    h = Harness([None])
    assert h.agent.run("x", "s1").text == "No response from AI model."


def test_job_marker_is_stripped():
    h = Harness([text("Fatto [JOB_CREATED:abc]")])
    assert h.agent.run("x", "s1").text == "Fatto"


# ---------------------------------------------------------------- fallback & retry

def test_daily_quota_moves_to_next_model_without_waiting():
    h = Harness([RateLimitError("429 PerDay", daily_quota=True), text("ok")], models=["a", "b"])
    res = h.agent.run("x", "s1")
    assert [c.model for c in h.provider.calls] == ["a", "b"]
    assert h.sleeps == []
    assert res.text == "ok" and res.model == "b"


def test_rate_limit_retries_with_backoff_then_succeeds():
    h = Harness([RateLimitError("429"), RateLimitError("429"), text("ok")], models=["a", "b"])
    res = h.agent.run("x", "s1")
    assert [c.model for c in h.provider.calls] == ["a", "a", "a"]
    assert h.sleeps == [5, 10]
    assert res.text == "ok"


def test_rate_limit_exhausts_retries_then_falls_back():
    h = Harness([RateLimitError("429")] * 3 + [text("ok")], models=["a", "b"])
    res = h.agent.run("x", "s1")
    assert [c.model for c in h.provider.calls] == ["a", "a", "a", "b"]
    assert h.sleeps == [5, 10]  # no pointless sleep after the last attempt
    assert res.model == "b"


def test_other_provider_error_falls_back_immediately():
    h = Harness([ProviderError("500 boom"), text("ok")], models=["a", "b"])
    res = h.agent.run("x", "s1")
    assert [c.model for c in h.provider.calls] == ["a", "b"]
    assert h.sleeps == [] and res.text == "ok"


def test_all_models_exhausted():
    h = Harness([ProviderError("down"), ProviderError("down")], models=["a", "b"])
    res = h.agent.run("x", "s1")
    assert res.text == "AI service temporarily unavailable. Please retry in a minute."
    assert "down" in res.error
    assert res.job_id is None


def test_failure_after_job_creation_does_not_duplicate_the_job():
    h = Harness(
        [call("launch_analysis", location="Firenze", algorithm="ndvi"), RateLimitError("429"), text("never")],
        models=["a", "b"],
    )
    res = h.agent.run("ndvi firenze", "s1")
    assert len(h.repo.all()) == 1
    assert len(h.provider.calls) == 2  # no retry, no fallback
    assert res.job_id == str(h.repo.all()[0].id)
    assert res.text.startswith("Analysis 'NDVI' queued for Firenze")


def test_requires_a_model():
    with pytest.raises(ValueError):
        Agent(ScriptedProvider(), models=[], job_repo=InMemoryJobRepository())


# ---------------------------------------------------------------- input handling

def test_input_is_sanitised_before_reaching_the_model():
    h = Harness([text("ok")])
    raw = "ndvi\x00 fi\x07renze\n\tok" + "x" * 3000
    h.agent.run(raw, "s1")
    sent = h.provider.calls[0].messages[0].text
    assert "\x00" not in sent and "\x07" not in sent
    assert sent.startswith("ndvi firenze\n\tok")
    assert len(sent) == 2000


@pytest.mark.parametrize("raw", ["", "   ", "\x00\x01\x02"])
def test_empty_input_never_reaches_the_model(raw):
    h = Harness([])
    assert h.agent.run(raw, "s1").text == "Please enter a message."
    assert h.provider.calls == []


def test_each_request_is_stateless_towards_the_model():
    h = Harness([text("one"), text("two")])
    h.agent.run("first", "s1")
    h.agent.run("second", "s1")
    assert h.provider.calls[1].messages == [UserMessage(text="second")]


# ---------------------------------------------------------------- session / rerun

def test_rerun_uses_session_context():
    h = Harness([
        call("launch_analysis", location="Firenze", algorithm="ndvi"), relay_tool_results,
        call("rerun_with_different_algorithm", algorithm="water"), relay_tool_results,
    ])
    first = h.agent.run("ndvi firenze", "s1")
    second = h.agent.run("stessa zona, acqua", "s1")
    jobs = h.repo.all()
    assert [j.algorithm for j in jobs] == ["ndvi", "ndwi"]
    assert jobs[1].bbox == jobs[0].bbox
    assert second.job_id != first.job_id
    ctx = h.sessions.get_or_create("s1").context
    assert ctx["last_algorithm"] == "ndwi" and ctx["last_job_id"] == second.job_id


def test_rerun_without_algorithm_reuses_previous():
    h = Harness([
        call("launch_analysis", location="Pisa", algorithm="flood"), relay_tool_results,
        call("rerun_with_different_algorithm"), relay_tool_results,
    ])
    h.agent.run("flood pisa", "s1")
    h.agent.run("ripeti", "s1")
    assert [j.algorithm for j in h.repo.all()] == ["flood", "flood"]


def test_rerun_is_isolated_per_session():
    h = Harness([
        call("launch_analysis", location="Pisa", algorithm="flood"), relay_tool_results,
        call("rerun_with_different_algorithm", algorithm="ndvi"), relay_tool_results,
    ])
    h.agent.run("flood pisa", "s1")
    res = h.agent.run("same area, ndvi", "other-session")
    assert res.text == "No previous analysis in this session. Specify a location."
    assert len(h.repo.all()) == 1


def test_rerun_without_session_id():
    h = Harness([call("rerun_with_different_algorithm", algorithm="ndvi"), relay_tool_results])
    assert h.agent.run("same area", "").text == "No session context. Please specify a location."


# ---------------------------------------------------------------- job status

def test_get_job_status_variants():
    h = Harness()
    job = h.repo.create(algorithm="flood", provider="p", bbox=[0, 0, 1, 1],
                        start_date=FIXED_NOW, end_date=FIXED_NOW, params={})
    job.status, job.result = "completed", {"tile_url": "http://t", "summary": {"a": 1}, "other": 2}
    from eo_brain.agent.brain import _RequestContext

    ctx = _RequestContext(session_id="s1")
    assert h.agent._get_job_status(ctx, "not-a-uuid") == {"error": "Invalid job ID: not-a-uuid"}
    missing = str(UUID(int=999))
    assert h.agent._get_job_status(ctx, missing) == {"error": f"Job {missing} not found."}
    found = h.agent._get_job_status(ctx, str(job.id))
    assert found["status"] == "completed"
    assert found["result"] == {
        "tile_url": "http://t", "summary": {"a": 1},
        "coverage": {"note": "coverage was not recorded for this analysis (it ran before coverage reporting)"},
    }


def test_get_job_status_reports_coverage_facts_only():
    h = Harness()
    job = h.repo.create(algorithm="ndvi", provider="p", bbox=[0, 0, 1, 1],
                        start_date=FIXED_NOW - timedelta(days=30), end_date=FIXED_NOW,
                        params={"geocoded_name": "Casentino"})
    coverage = {"coverage_pct": 7.2, "low_coverage": True, "uncovered_masked_pct": 0.3,
                "uncovered_no_data_pct": 92.4, "scenes_found": 27, "scenes_used": 1,
                "area_px": 380172, "selection": {"scene_ids": ["x"]}}
    job.status, job.result = "completed", {"tile_url": "t", "summary": {}, "coverage": coverage}
    job.message = "Completed with partial coverage: ..."
    from eo_brain.agent.brain import _RequestContext

    found = h.agent._get_job_status(_RequestContext(session_id="s1"), str(job.id))
    assert found["location"] == "Casentino"
    assert found["time_window"] == f"{(FIXED_NOW - timedelta(days=30)).date()} to {FIXED_NOW.date()}"
    assert found["message"].startswith("Completed with partial coverage")
    assert found["result"]["coverage"] == {
        "coverage_pct": 7.2, "low_coverage": True, "uncovered_masked_pct": 0.3,
        "uncovered_no_data_pct": 92.4, "scenes_found": 27, "scenes_used": 1,
    }


def test_get_job_status_without_id_uses_the_sessions_latest_job():
    h = Harness([call("launch_analysis", location="Firenze", algorithm="ndvi"), text("ok"),
                 call("get_job_status"), relay_tool_results])
    launched = h.agent.run("ndvi firenze", "s1")
    status = h.agent.run("perché la mappa copre solo una striscia?", "s1")
    assert launched.job_id and status.job_id is None
    [inv] = status.tool_calls
    assert inv.name == "get_job_status" and inv.result["job_id"] == launched.job_id


def test_get_job_status_without_id_and_without_a_job_says_so():
    from eo_brain.agent.brain import _RequestContext

    h = Harness()
    out = h.agent._get_job_status(_RequestContext(session_id="fresh"))
    assert "No analysis was launched in this conversation" in out["error"]
    assert "query_database" in out["error"]


def test_prompt_forbids_invented_causes():
    rule = brain.SYSTEM_PROMPT.split("13.")[1]
    assert "only from tool results" in rule and "never offer causes" in rule
    assert "get_job_status" in rule and "query_database" in rule


# ---------------------------------------------------------------- module entry point

def test_chat_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(brain, "_default_agent", None)
    assert brain.chat("ndvi firenze", "s1") == {
        "text": "AI agent is not configured. Set GEMINI_API_KEY.", "job_id": None, "bbox": None, "query": None,
    }


def test_chat_uses_injected_default_agent(monkeypatch):
    h = Harness([text("ciao")])
    monkeypatch.setattr(brain, "_default_agent", h.agent)
    assert brain.chat("ciao", "s1") == {"text": "ciao", "job_id": None, "bbox": None, "query": None}


def test_launch_reports_a_geocoder_fallback():
    from eo_brain.agent.geocoder import GeocodingResult

    def geocode(q):
        return GeocodingResult(name="Casentino, Arezzo, Toscana, Italia", lat=43.7, lon=11.8,
                               bbox=(11.6, 43.55, 11.98, 43.88), source="nominatim_bbox",
                               matched_query="Casentino", fallback_from=q)

    h = Harness([call("launch_analysis", location="vallata del Casentino", algorithm="ndvi"), relay_tool_results])
    h.agent.geocode = geocode
    res = h.agent.run("ndvi vallata del Casentino", "s1")
    [inv] = res.tool_calls
    assert inv.result["location"] == "Casentino, Arezzo, Toscana, Italia"
    assert "'vallata del Casentino' was not found as written" in inv.result["geocoding_note"]
    assert "used Casentino, Arezzo, Toscana, Italia" in inv.result["message"]


# ---------------------------------------------------------------- which place (recorded Nominatim responses)

from pathlib import Path  # noqa: E402

from eo_brain.agent.geocoder import GeocodingResult  # noqa: E402
from evals.stubs import load_recorded_searches  # noqa: E402

RECORDED = load_recorded_searches(Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "nominatim_recorded.json")
REGION = "Casentino, Arezzo, Tuscany, Italy"
VILLAGE = "Casentino, Sant'Eusanio Forconese, L'Aquila, Abruzzo, 67040, Italy"


def place_harness(*steps):
    h = Harness(list(steps))
    h.geocoder = StubGeocoder(GAZETTEER, RECORDED)
    h.agent.geocode = h.geocoder
    return h


def ask_casentino(h, session="s1"):
    h.add(call("launch_analysis", location="Casentino", algorithm="vegetation", days_back=60), relay_tool_results)
    return h.agent.run("ndvi Casentino ultimi 60 giorni", session)


def test_ambiguous_place_is_a_question_with_candidates_and_no_job():
    h = place_harness()
    res = ask_casentino(h)
    [inv] = res.tool_calls
    assert res.job_id is None and h.repo.all() == []
    assert inv.result["needs_user"] is True
    assert inv.result["candidates"] == [
        {"location": VILLAGE, "kind": "village", "in": "L'Aquila, Abruzzo, Italy"},
        {"location": REGION, "kind": "region", "in": "Arezzo, Tuscany, Italy"},
    ]
    assert "Do not pick one yourself" in inv.result["action_required"]
    assert "location' text exactly" in inv.result["action_required"]
    pending = h.sessions.get_or_create("s1").get_context("pending_place")
    assert pending["algorithm"] == "ndvi" and pending["days_back"] == 60
    assert [c.label for c in pending["candidates"]] == [VILLAGE, REGION]


def test_model_cannot_pick_a_candidate_itself_in_the_same_turn():
    h = place_harness(
        call("launch_analysis", location="Casentino", algorithm="ndvi"),
        call("launch_analysis", location=REGION, algorithm="ndvi"),
        text("Ho scelto io."),
    )
    res = h.agent.run("ndvi Casentino", "s1")
    assert res.job_id is None and h.repo.all() == []
    assert res.tool_calls[1].result["blocked"] is True


@pytest.mark.parametrize("label", [REGION, VILLAGE])
def test_the_answer_resolves_to_exactly_the_listed_candidate(label):
    h = place_harness()
    asked = ask_casentino(h)
    listed = {c["location"]: c for c in asked.tool_calls[0].result["candidates"]}
    searches = list(h.geocoder.searches)
    h.add(call("launch_analysis", location=label, algorithm="ndvi", days_back=60), relay_tool_results)
    res = h.agent.run("quello giusto", "s1")
    [job] = h.repo.all()
    assert res.job_id == str(job.id) and job.params["geocoded_name"] == label
    assert h.geocoder.searches == searches                  # pinned candidate: no new lookup
    expected = h.geocoder._resolver.resolve(label).bbox     # what that label names on its own
    assert job.bbox == pytest.approx(list(expected))
    assert label in listed
    assert h.sessions.get_or_create("s1").get_context("pending_place") is None


def test_an_answer_naming_the_place_and_its_province_picks_that_one():
    h = place_harness()
    ask_casentino(h)
    searches = list(h.geocoder.searches)
    h.add(call("launch_analysis", location="Casentino, Arezzo", algorithm="ndvi"), relay_tool_results)
    res = h.agent.run("Casentino in provincia di Arezzo", "s1")
    assert res.job_id is not None and h.repo.all()[0].params["geocoded_name"] == REGION
    assert h.geocoder.searches == searches


def test_repeating_the_ambiguous_name_asks_again():
    h = place_harness()
    ask_casentino(h)
    h.add(call("launch_analysis", location="Casentino", algorithm="ndvi"), relay_tool_results)
    res = h.agent.run("Casentino", "s1")
    assert res.job_id is None and res.tool_calls[0].result["candidates"]


def test_context_note_after_a_question_is_shown_once():
    h = place_harness()
    ask_casentino(h)
    h.add(text("Quale?"), text("Ok."))
    h.agent.run("quello in Toscana", "s1")
    h.agent.run("grazie", "s1")
    first = h.provider.calls[-2].messages[0].text
    assert first.startswith("[Context from the app, not written by the user")
    assert f"1) {VILLAGE} (village); 2) {REGION} (region)" in first
    assert "algorithm 'ndvi' and days_back 60" in first
    assert first.endswith("User message: quello in Toscana")
    assert h.provider.calls[-1].messages == [UserMessage(text="grazie")]


def test_valley_phrasing_launches_on_the_region_and_says_how_it_was_found():
    h = place_harness(call("launch_analysis", location="vallata del Casentino", algorithm="ndvi"), relay_tool_results)
    res = h.agent.run("Per favore, trova NDVI nella vallata del Casentino.", "s1")
    [inv] = res.tool_calls
    assert inv.result["location"] == REGION
    assert "searched for 'Casentino' instead and used " + REGION + " (region)" in inv.result["geocoding_note"]
    assert "Location: 'vallata del Casentino' not found as written" in inv.result["message"]


def test_region_asked_and_only_a_city_found_is_a_question():
    h = place_harness(call("launch_analysis", location="piana di Pisa", algorithm="ndvi"), relay_tool_results)
    res = h.agent.run("NDVI sulla piana di Pisa", "s1")
    [inv] = res.tool_calls
    assert res.job_id is None and inv.result["needs_user"]
    assert inv.result["candidates"] == [{"location": "Pisa, Toscana, Italia", "kind": "city", "in": "Toscana, Italia"}]
    assert "No region named as in 'piana di Pisa'" in inv.result["error"]


def test_a_non_area_last_resort_is_reported():
    def geocode(q):
        return GeocodingResult(name="Via Roma, Pisa", lat=43.7, lon=10.4, bbox=(10.39, 43.69, 10.41, 43.71),
                               source="nominatim_bbox", kind="road", area_like=False)

    h = Harness([call("launch_analysis", location="Via Roma, Pisa", algorithm="ndvi"), relay_tool_results])
    h.agent.geocode = geocode
    [inv] = h.agent.run("ndvi via roma pisa", "s1").tool_calls
    assert "Via Roma, Pisa is a road, not a town, region or natural area" in inv.result["geocoding_note"]
    assert "the analysis covers its bounding box" in inv.result["message"]


def test_place_lookups_per_request_are_bounded():
    h = place_harness(
        call("launch_analysis", location="Xyzzy", algorithm="ndvi"),
        call("launch_analysis", location="Plugh", algorithm="ndvi"),
        call("launch_analysis", location="Firenze", algorithm="ndvi"),
        text("..."),
    )
    res = h.agent.run("ndvi xyzzy", "s1")
    assert h.geocoder.queries == ["Xyzzy", "Plugh"]
    third = res.tool_calls[2].result
    assert third["needs_user"] and "could not be resolved in 2 attempts" in third["error"]
    assert res.job_id is None


def test_rerun_reuses_the_stored_area_without_a_lookup():
    h = place_harness(
        call("launch_analysis", location="vallata del Casentino", algorithm="ndvi"), relay_tool_results,
        call("rerun_with_different_algorithm", algorithm="water"), relay_tool_results,
    )
    h.agent.run("ndvi vallata del Casentino", "s1")
    queries = list(h.geocoder.queries)
    res = h.agent.run("stessa zona, acqua", "s1")
    first, second = h.repo.all()
    assert h.geocoder.queries == queries == ["vallata del Casentino"]
    assert second.bbox == first.bbox and second.params["geocoded_name"] == REGION
    assert "geocoding_note" not in res.tool_calls[0].result


def test_location_parameter_asks_for_commas_and_exact_candidates():
    spec = next(t for t in TOOL_SPECS if t.name == "launch_analysis")
    desc = spec.parameters["properties"]["location"]["description"]
    assert "separated by commas" in desc and "'Casentino, Arezzo, Italia'" in desc
    assert "candidate's location text exactly" in desc


def test_prompt_says_to_list_candidates_and_not_choose():
    assert "If it lists candidate places, show every candidate" in SYSTEM_PROMPT
    assert "never choose one yourself" in SYSTEM_PROMPT
