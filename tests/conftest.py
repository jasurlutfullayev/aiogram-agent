from __future__ import annotations

import pytest

from aiogram_agent import LLMResponse, ToolCall
from aiogram_agent.tools import Tool
from aiogram_agent.types import Message


class ScriptedProvider:
    """Returns pre-baked responses and records what it was sent."""

    def __init__(self, *responses: LLMResponse) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, list[Message], list[Tool]]] = []

    async def complete(self, system: str, messages: list[Message], tools: list[Tool]) -> LLMResponse:
        self.calls.append((system, [dict(m) for m in messages], list(tools)))
        if not self.responses:
            return LLMResponse(text="(no more scripted responses)")
        return self.responses.pop(0)


def call(name: str, **arguments) -> ToolCall:
    call.n = getattr(call, "n", 0) + 1
    return ToolCall(id=f"call_{call.n}", name=name, arguments=arguments)


@pytest.fixture
def scripted():
    return ScriptedProvider
