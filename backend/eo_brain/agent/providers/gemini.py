"""
eo_brain/agent/providers/gemini.py - Google Gemini adapter (google-genai SDK).

Translates the provider-neutral types in ``eo_brain.agent.llm`` to the
google-genai wire format and back. Automatic function calling (AFC) is
disabled: the agent loop in ``brain.py`` dispatches tool calls itself so
every call, argument and error is visible and logged.

The SDK is imported lazily, so the rest of the agent (and its tests) does
not need ``google-genai`` installed.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Sequence

from eo_brain.agent.llm import (
    Message,
    ModelTurn,
    ProviderError,
    RateLimitError,
    ToolCall,
    ToolResults,
    ToolSpec,
    UserMessage,
)

logger = logging.getLogger(__name__)

# Model ids are configuration, not code. These defaults were checked against
# Google's model and deprecation pages (https://aistudio.google.com/docs/models,
# https://aistudio.google.com/docs/deprecations) on 2026-11-26: both are listed
# as Stable with function calling supported. gemini-3.6-flash is the documented
# replacement for gemini-2.0-flash, which is shut down. Override with
# GEMINI_MODEL / GEMINI_FALLBACK_MODELS rather than editing this file.
DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_FALLBACK_MODELS = ("gemini-3.5-flash-lite",)


def model_chain_from_env(env: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Primary model from GEMINI_MODEL, then the comma-separated
    GEMINI_FALLBACK_MODELS. Unset or empty variables use the defaults.
    Duplicates are dropped, order is kept.
    """
    env = os.environ if env is None else env
    primary = (env.get("GEMINI_MODEL") or "").strip() or DEFAULT_MODEL
    raw_fallbacks = (env.get("GEMINI_FALLBACK_MODELS") or "").strip()
    fallbacks = (
        [m.strip() for m in raw_fallbacks.split(",") if m.strip()]
        if raw_fallbacks
        else list(DEFAULT_FALLBACK_MODELS)
    )
    chain: List[str] = []
    for model in [primary, *fallbacks]:
        if model not in chain:
            chain.append(model)
    return chain


def _is_rate_limit(exc: BaseException) -> bool:
    # google-genai raises errors.ClientError with code 429.
    if getattr(exc, "code", None) == 429:
        return True
    # Older stacks surfaced google.api_core ResourceExhausted instead.
    try:
        from google.api_core.exceptions import ResourceExhausted
    except ImportError:  # pragma: no cover - depends on installed extras
        return False
    return isinstance(exc, ResourceExhausted)


def _is_daily_quota(exc: BaseException) -> bool:
    text = str(exc)
    return "PerDay" in text or "limit: 0" in text


def _to_schema(types: Any, spec: Dict[str, Any]) -> Any:
    """JSON-Schema dict -> google.genai.types.Schema (the subset we use)."""
    kwargs: Dict[str, Any] = {"type": str(spec["type"]).upper()}
    if "description" in spec:
        kwargs["description"] = spec["description"]
    if "properties" in spec:
        kwargs["properties"] = {
            name: _to_schema(types, sub) for name, sub in spec["properties"].items()
        }
    if spec.get("required"):
        kwargs["required"] = list(spec["required"])
    if "enum" in spec:
        kwargs["enum"] = list(spec["enum"])
    return types.Schema(**kwargs)


class GeminiProvider:
    """LLMProvider backed by ``google.genai.Client.models.generate_content``."""

    name = "gemini"

    def __init__(self, api_key: Optional[str] = None, *, client: Any = None):
        from google.genai import types  # lazy: optional dependency

        self._types = types
        if client is None:
            from google import genai

            key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
            if not key:
                raise ProviderError("GEMINI_API_KEY is not set")
            client = genai.Client(api_key=key)
        self._client = client

    # -- translation ---------------------------------------------------------

    def _tool(self, tools: Sequence[ToolSpec]) -> Any:
        t = self._types
        return t.Tool(
            function_declarations=[
                t.FunctionDeclaration(
                    name=spec.name,
                    description=spec.description,
                    parameters=_to_schema(t, spec.parameters),
                )
                for spec in tools
            ]
        )

    def _contents(self, messages: Sequence[Message]) -> List[Any]:
        t = self._types
        contents: List[Any] = []
        for msg in messages:
            if isinstance(msg, UserMessage):
                contents.append(t.Content(role="user", parts=[t.Part(text=msg.text)]))
            elif isinstance(msg, ModelTurn):
                if isinstance(msg.raw, t.Content):
                    # Replay verbatim: keeps any metadata the API attached.
                    contents.append(msg.raw)
                    continue
                parts = [
                    t.Part(function_call=t.FunctionCall(name=c.name, args=dict(c.args)))
                    for c in msg.tool_calls
                ]
                if msg.text:
                    parts.insert(0, t.Part(text=msg.text))
                contents.append(t.Content(role="model", parts=parts))
            elif isinstance(msg, ToolResults):
                contents.append(
                    t.Content(
                        role="user",
                        parts=[
                            t.Part(
                                function_response=t.FunctionResponse(
                                    name=r.name, response=r.response
                                )
                            )
                            for r in msg.results
                        ],
                    )
                )
            else:  # pragma: no cover - guarded by the type system
                raise TypeError(f"Unsupported message type: {type(msg).__name__}")
        return contents

    @staticmethod
    def _turn(response: Any) -> Optional[ModelTurn]:
        candidates = getattr(response, "candidates", None)
        candidate = candidates[0] if candidates else None
        content = getattr(candidate, "content", None) if candidate else None
        if content is None:
            return None
        parts = content.parts or []
        calls = [
            ToolCall(name=p.function_call.name, args=dict(p.function_call.args or {}))
            for p in parts
            if p.function_call
        ]
        texts = [p.text for p in parts if p.text]
        return ModelTurn(text=" ".join(texts), tool_calls=calls, raw=content)

    # -- LLMProvider ---------------------------------------------------------

    def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        tools: Sequence[ToolSpec],
        messages: Sequence[Message],
    ) -> Optional[ModelTurn]:
        t = self._types
        config = t.GenerateContentConfig(
            system_instruction=system_prompt,
            tools=[self._tool(tools)],
            automatic_function_calling=t.AutomaticFunctionCallingConfig(disable=True),
        )
        try:
            response = self._client.models.generate_content(
                model=model,
                contents=self._contents(messages),
                config=config,
            )
        except Exception as exc:
            if _is_rate_limit(exc):
                raise RateLimitError(str(exc), daily_quota=_is_daily_quota(exc)) from exc
            status = getattr(exc, "code", None)
            raise ProviderError(
                f"{type(exc).__name__}: {exc}", status=status if isinstance(status, int) else None
            ) from exc
        return self._turn(response)
