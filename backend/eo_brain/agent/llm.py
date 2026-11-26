"""
eo_brain/agent/llm.py - Provider-neutral types for the function-calling loop.

The agent loop in ``brain.py`` only speaks these types. Each LLM SDK gets a
thin adapter in ``eo_brain/agent/providers/`` that translates them to and from
its own wire format. Nothing in this module imports a vendor SDK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Union


@dataclass(frozen=True)
class ToolSpec:
    """A tool the model may call. ``parameters`` is a JSON-Schema object."""

    name: str
    description: str
    parameters: Dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """One function call requested by the model."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelTurn:
    """
    One model response: either text, tool calls, or both.

    ``raw`` holds the provider-native payload (e.g. a Gemini ``Content``).
    When the turn is sent back as history, the same provider replays it
    verbatim, so provider-specific metadata on the original response is
    not lost in translation.
    """

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Any = None


@dataclass(frozen=True)
class UserMessage:
    text: str


@dataclass(frozen=True)
class ToolResult:
    name: str
    response: Dict[str, Any]


@dataclass(frozen=True)
class ToolResults:
    """Results of the tool calls of the previous model turn."""

    results: List[ToolResult]


Message = Union[UserMessage, ModelTurn, ToolResults]


class ProviderError(Exception):
    """Any failure talking to the model provider."""


class RateLimitError(ProviderError):
    """
    The provider rejected the request for quota reasons.

    ``daily_quota`` is True when retrying the same model today is pointless
    and the caller should move on to the next model in the chain.
    """

    def __init__(self, message: str, *, daily_quota: bool = False):
        super().__init__(message)
        self.daily_quota = daily_quota


class LLMProvider(Protocol):
    """The only surface the agent loop needs from a model provider."""

    name: str

    def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        tools: Sequence[ToolSpec],
        messages: Sequence[Message],
    ) -> Optional[ModelTurn]:
        """
        Return the model's next turn, or None if the provider returned no
        candidate at all. Raise RateLimitError / ProviderError on failure.
        """
        ...
