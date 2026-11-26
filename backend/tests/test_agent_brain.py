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
    assert found["result"] == {"tile_url": "http://t", "summary": {"a": 1}}


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
