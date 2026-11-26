"""
Offline tests for the Gemini adapter: translation to/from google-genai types
and error mapping. A fake client stands in for the network; no API key.
"""

import pytest

from eo_brain.agent.brain import TOOL_SPECS
from eo_brain.agent.llm import (
    ModelTurn,
    ProviderError,
    RateLimitError,
    ToolCall,
    ToolResult,
    ToolResults,
    UserMessage,
)
from eo_brain.agent.providers.gemini import (
    DEFAULT_FALLBACK_MODELS,
    DEFAULT_MODEL,
    model_chain_from_env,
)


class TestModelChain:
    def test_defaults(self):
        assert model_chain_from_env({}) == [DEFAULT_MODEL, *DEFAULT_FALLBACK_MODELS]

    def test_empty_values_mean_defaults(self):
        assert model_chain_from_env({"GEMINI_MODEL": " ", "GEMINI_FALLBACK_MODELS": ""}) == model_chain_from_env({})

    def test_override_and_dedupe(self):
        env = {"GEMINI_MODEL": "m1", "GEMINI_FALLBACK_MODELS": "m2, m1 ,m3,,"}
        assert model_chain_from_env(env) == ["m1", "m2", "m3"]


genai = pytest.importorskip("google.genai")
from google.genai import errors, types  # noqa: E402

from eo_brain.agent.providers.gemini import GeminiProvider  # noqa: E402


class FakeModels:
    def __init__(self, outcome):
        self.outcome = outcome
        self.requests = []

    def generate_content(self, *, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


class FakeClient:
    def __init__(self, outcome):
        self.models = FakeModels(outcome)


def _response(*parts):
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=list(parts)))]
    )


def _generate(provider, messages=(UserMessage("ndvi firenze"),)):
    return provider.generate(model="m", system_prompt="SYS", tools=TOOL_SPECS, messages=list(messages))


def test_request_shape():
    client = FakeClient(_response(types.Part(text="ciao")))
    _generate(GeminiProvider(client=client))
    req = client.models.requests[0]
    assert req["model"] == "m"
    cfg = req["config"]
    assert cfg.system_instruction == "SYS"
    assert cfg.automatic_function_calling.disable is True
    decls = {d.name: d for d in cfg.tools[0].function_declarations}
    assert set(decls) == {t.name for t in TOOL_SPECS}
    launch = decls["launch_analysis"].parameters
    assert launch.type == types.Type.OBJECT
    assert launch.required == ["location", "algorithm"]
    assert launch.properties["days_back"].type == types.Type.INTEGER
    assert decls["list_capabilities"].parameters.properties == {}
    [content] = req["contents"]
    assert content.role == "user" and content.parts[0].text == "ndvi firenze"


def test_text_response():
    turn = _generate(GeminiProvider(client=FakeClient(_response(types.Part(text="a"), types.Part(text="b")))))
    assert turn.text == "a b" and turn.tool_calls == []


def test_function_call_response_keeps_raw_content():
    resp = _response(types.Part(function_call=types.FunctionCall(name="launch_analysis", args={"location": "Firenze", "algorithm": "ndvi"})))
    turn = _generate(GeminiProvider(client=FakeClient(resp)))
    assert turn.tool_calls == [ToolCall("launch_analysis", {"location": "Firenze", "algorithm": "ndvi"})]
    assert turn.raw is resp.candidates[0].content


@pytest.mark.parametrize(
    "resp",
    [
        types.GenerateContentResponse(candidates=[]),
        types.GenerateContentResponse(candidates=[types.Candidate(content=None)]),
    ],
)
def test_no_candidate_is_none(resp):
    assert _generate(GeminiProvider(client=FakeClient(resp))) is None


def test_history_translation():
    raw = types.Content(role="model", parts=[types.Part(function_call=types.FunctionCall(name="x", args={}))])
    client = FakeClient(_response(types.Part(text="ok")))
    _generate(
        GeminiProvider(client=client),
        [
            UserMessage("u"),
            ModelTurn(tool_calls=[ToolCall("x", {})], raw=raw),
            ToolResults([ToolResult("x", {"error": "nope"})]),
            ModelTurn(text="t", tool_calls=[ToolCall("y", {"a": 1})]),
        ],
    )
    contents = client.models.requests[0]["contents"]
    assert contents[1] is raw  # replayed verbatim
    assert contents[2].role == "user"
    assert contents[2].parts[0].function_response.name == "x"
    assert contents[2].parts[0].function_response.response == {"error": "nope"}
    assert contents[3].role == "model"
    assert contents[3].parts[0].text == "t"
    assert contents[3].parts[1].function_call.name == "y"


def _client_error(message):
    return errors.ClientError(429, {"error": {"code": 429, "message": message, "status": "RESOURCE_EXHAUSTED"}})


def test_rate_limit_mapping():
    with pytest.raises(RateLimitError) as exc:
        _generate(GeminiProvider(client=FakeClient(_client_error("Resource has been exhausted"))))
    assert exc.value.daily_quota is False


def test_daily_quota_mapping():
    with pytest.raises(RateLimitError) as exc:
        _generate(GeminiProvider(client=FakeClient(_client_error("Quota exceeded ... limit: 0, PerDay"))))
    assert exc.value.daily_quota is True


@pytest.mark.parametrize(
    "error",
    [
        errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}}),
        errors.ServerError(503, {"error": {"code": 503, "message": "x", "status": "UNAVAILABLE"}}),
        ConnectionError("offline"),
    ],
)
def test_other_errors_map_to_provider_error(error):
    with pytest.raises(ProviderError) as exc:
        _generate(GeminiProvider(client=FakeClient(error)))
    assert not isinstance(exc.value, RateLimitError)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="GEMINI_API_KEY"):
        GeminiProvider()


@pytest.mark.parametrize(
    "error,status,transient",
    [
        (errors.ClientError(400, {"error": {"code": 400, "message": "bad", "status": "INVALID_ARGUMENT"}}), 400, False),
        (errors.ClientError(404, {"error": {"code": 404, "message": "no model", "status": "NOT_FOUND"}}), 404, False),
        (errors.ServerError(503, {"error": {"code": 503, "message": "high demand", "status": "UNAVAILABLE"}}),
         503, True),
        (errors.ServerError(500, {"error": {"code": 500, "message": "x", "status": "INTERNAL"}}), 500, True),
        (ConnectionError("offline"), None, False),
    ],
)
def test_provider_error_carries_status_and_transient_flag(error, status, transient):
    with pytest.raises(ProviderError) as exc:
        _generate(GeminiProvider(client=FakeClient(error)))
    assert exc.value.status == status
    assert exc.value.transient is transient


def test_rate_limits_are_transient_unless_daily():
    with pytest.raises(RateLimitError) as per_minute:
        _generate(GeminiProvider(client=FakeClient(_client_error("Resource has been exhausted"))))
    assert per_minute.value.transient is True and per_minute.value.status == 429
    with pytest.raises(RateLimitError) as daily:
        _generate(GeminiProvider(client=FakeClient(_client_error("Quota exceeded ... limit: 0, PerDay"))))
    assert daily.value.transient is False


def test_free_tier_daily_quota_message_is_detected():
    """Shape of the 429 body seen on a real free-tier run (no secrets in it)."""
    body = {"error": {
        "code": 429,
        "message": "You exceeded your current quota. * Quota exceeded for metric: "
                   "generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 20, "
                   "model: gemini-3.6-flash, description: Generate content requests per day limit exceeded.",
        "status": "RESOURCE_EXHAUSTED",
        "details": [{"@type": "type.googleapis.com/google.rpc.QuotaFailure", "violations": [
            {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier", "quotaValue": "20"}]}],
    }}
    with pytest.raises(RateLimitError) as exc:
        _generate(GeminiProvider(client=FakeClient(errors.ClientError(429, body))))
    assert exc.value.daily_quota is True
