from __future__ import annotations

from typing import Any

from ..tools import Tool
from ..types import LLMResponse, Message, ToolCall
from .base import TextCallback


class AnthropicProvider:
    """Claude provider via the Messages API.

    Requires ``pip install aiogram-agent[anthropic]``.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5",
        *,
        api_key: str | None = None,
        max_tokens: int = 1024,
        temperature: float | None = 0.3,
        client: Any = None,
    ) -> None:
        if client is None:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(api_key=api_key)
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    @staticmethod
    def _user_blocks(m: Message) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {"type": "base64", "media_type": img["media_type"], "data": img["data"]},
            }
            for img in m.get("images") or []
        ]
        if m.get("content"):
            blocks.append({"type": "text", "text": m["content"]})
        return blocks

    @classmethod
    def _convert(cls, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        def add_user_blocks(blocks: list[dict[str, Any]]) -> None:
            # Anthropic requires alternating roles: merge consecutive user turns.
            if out and out[-1]["role"] == "user":
                prev = out[-1]["content"]
                if isinstance(prev, str):
                    prev = [{"type": "text", "text": prev}]
                out[-1]["content"] = prev + blocks
            else:
                out.append({"role": "user", "content": blocks})

        for m in messages:
            role = m["role"]
            if role == "assistant":
                blocks: list[dict[str, Any]] = []
                if m.get("content"):
                    blocks.append({"type": "text", "text": m["content"]})
                for tc in m.get("tool_calls") or []:
                    blocks.append(
                        {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["arguments"]}
                    )
                if not blocks:
                    continue
                if out and out[-1]["role"] == "assistant":
                    out[-1]["content"].extend(blocks)
                else:
                    out.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                add_user_blocks(
                    [{"type": "tool_result", "tool_use_id": m["tool_call_id"], "content": m["content"]}]
                )
            elif m.get("images") or (out and out[-1]["role"] == "user"):
                add_user_blocks(cls._user_blocks(m))
            else:
                out.append({"role": "user", "content": m["content"]})
        return out

    def _kwargs(self, system: str, messages: list[Message], tools: list[Tool]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._convert(messages),
        }
        if system:
            kwargs["system"] = system
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
            ]
        return kwargs

    @staticmethod
    def _parse(resp: Any) -> LLMResponse:
        texts: list[str] = []
        calls: list[ToolCall] = []
        for block in resp.content:
            if block.type == "text":
                texts.append(block.text)
            elif block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, arguments=dict(block.input or {})))
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text="".join(texts),
            tool_calls=calls,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )

    async def complete(self, system: str, messages: list[Message], tools: list[Tool]) -> LLMResponse:
        return self._parse(await self.client.messages.create(**self._kwargs(system, messages, tools)))

    async def stream(
        self, system: str, messages: list[Message], tools: list[Tool], on_text: TextCallback
    ) -> LLMResponse:
        text = ""
        async with self.client.messages.stream(**self._kwargs(system, messages, tools)) as stream:
            async for delta in stream.text_stream:
                text += delta
                await on_text(text)
            final = await stream.get_final_message()
        return self._parse(final)
