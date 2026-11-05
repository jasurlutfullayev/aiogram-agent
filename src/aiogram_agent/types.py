"""Provider-neutral message format.

Every provider converts to and from these plain dicts, so memory backends
can store history as JSON and you can switch OpenAI <-> Anthropic freely.

    {"role": "user", "content": "...", "images": [Image, ...]}   # images optional
    {"role": "assistant", "content": "...", "tool_calls": [ToolCall-as-dict, ...]}
    {"role": "tool", "tool_call_id": "...", "name": "...", "content": "..."}
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, TypedDict

Message = dict[str, Any]


class Image(TypedDict):
    """A base64-encoded image attached to a user message."""

    media_type: str  # e.g. "image/jpeg"
    data: str  # base64, no "data:" prefix


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class AgentReply:
    """What :meth:`Agent.run` returns."""

    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    handed_off: bool = False
    failed: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
