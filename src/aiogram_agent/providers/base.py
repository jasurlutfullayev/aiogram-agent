from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol, runtime_checkable

from ..tools import Tool
from ..types import LLMResponse, Message

#: Receives the full text generated so far (not just the delta).
TextCallback = Callable[[str], Awaitable[None]]


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can answer a conversation, optionally calling tools.

    Providers may additionally implement streaming::

        async def stream(self, system, messages, tools, on_text) -> LLMResponse
    """

    async def complete(
        self,
        system: str,
        messages: list[Message],
        tools: list[Tool],
    ) -> LLMResponse: ...
