"""
eo_brain/agent/providers/scripted.py - Deterministic stand-in for a model.

``ScriptedProvider`` replays a fixed list of steps, one per ``generate``
call. It never looks at the user's text, so it measures nothing about
language understanding. It exists to drive the agent loop in tests and in
the offline eval mode, where it checks the plumbing (dispatch, validation,
session context, scoring) with no network and no API key.

A step is one of:
  * a ``ModelTurn``                      -> returned as-is
  * an exception instance                -> raised (e.g. RateLimitError)
  * ``None``                             -> "no candidate" response
  * a callable ``(model, messages)``     -> its return value is used as above
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

from eo_brain.agent.llm import Message, ModelTurn, ToolResults, ToolSpec


class ScriptExhausted(AssertionError):
    """The agent asked for more turns than the script provides."""


@dataclass
class RecordedCall:
    model: str
    system_prompt: str
    tool_names: List[str]
    messages: List[Message]


class ScriptedProvider:
    """LLMProvider that replays ``steps`` in order and records every call."""

    def __init__(self, steps: Iterable[Any] = (), name: str = "scripted"):
        self.steps: List[Any] = list(steps)
        self.name = name
        self.calls: List[RecordedCall] = []

    def generate(
        self,
        *,
        model: str,
        system_prompt: str,
        tools: Sequence[ToolSpec],
        messages: Sequence[Message],
    ) -> Optional[ModelTurn]:
        self.calls.append(
            RecordedCall(
                model=model,
                system_prompt=system_prompt,
                tool_names=[t.name for t in tools],
                messages=list(messages),
            )
        )
        if not self.steps:
            raise ScriptExhausted(
                f"Script exhausted after {len(self.calls) - 1} turn(s) (model={model})"
            )
        step = self.steps.pop(0)
        if callable(step) and not isinstance(step, (ModelTurn, BaseException)):
            step = step(model, list(messages))
        if isinstance(step, BaseException):
            raise step
        if step is None or isinstance(step, ModelTurn):
            return step
        raise TypeError(f"Unsupported script step: {step!r}")


def relay_tool_results(model: str, messages: List[Message]) -> ModelTurn:
    """
    Script step that answers like rule 3/6 of the system prompt asks a real
    model to: relay the tool's own message or error text, nothing else.
    """
    last = messages[-1] if messages else None
    if not isinstance(last, ToolResults):
        return ModelTurn(text="")
    lines = []
    for result in last.results:
        resp = result.response
        lines.append(str(resp.get("error") or resp.get("message") or f"{result.name}: ok"))
    return ModelTurn(text=" ".join(lines))
