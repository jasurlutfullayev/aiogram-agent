from __future__ import annotations

import json
from typing import Any

from ..tools import Tool
from ..types import LLMResponse, Message, ToolCall
from .base import TextCallback


def _parse_args(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


class OpenAIProvider:
    """Chat Completions provider. Works with any OpenAI-compatible API
    (OpenAI, Groq, Together, OpenRouter, local vLLM/Ollama) via ``base_url``.

    Requires ``pip install aiogram-agent[openai]``.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float | None = 0.3,
        max_tokens: int | None = 1024,
        client: Any = None,
    ) -> None:
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _convert(system: str, messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        for m in messages:
            role = m["role"]
            if role == "assistant" and m.get("tool_calls"):
                out.append(
                    {
                        "role": "assistant",
                        "content": m.get("content") or None,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": json.dumps(tc["arguments"], ensure_ascii=False),
                                },
                            }
                            for tc in m["tool_calls"]
                        ],
                    }
                )
            elif role == "tool":
                out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": m["content"]})
            elif role == "user" and m.get("images"):
                parts: list[dict[str, Any]] = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img['media_type']};base64,{img['data']}"},
                    }
                    for img in m["images"]
                ]
                if m.get("content"):
                    parts.append({"type": "text", "text": m["content"]})
                out.append({"role": "user", "content": parts})
            else:
                out.append({"role": role, "content": m["content"]})
        return out

    def _kwargs(self, system: str, messages: list[Message], tools: list[Tool]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "messages": self._convert(system, messages)}
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_tokens is not None:
            kwargs["max_tokens"] = self.max_tokens
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
                }
                for t in tools
            ]
        return kwargs

    async def complete(self, system: str, messages: list[Message], tools: list[Tool]) -> LLMResponse:
        resp = await self.client.chat.completions.create(**self._kwargs(system, messages, tools))
        msg = resp.choices[0].message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=_parse_args(tc.function.arguments))
            for tc in (msg.tool_calls or [])
        ]
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text=msg.content or "",
            tool_calls=calls,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )

    async def stream(
        self, system: str, messages: list[Message], tools: list[Tool], on_text: TextCallback
    ) -> LLMResponse:
        kwargs = self._kwargs(system, messages, tools)
        kwargs["stream"] = True
        kwargs["stream_options"] = {"include_usage": True}
        stream = await self.client.chat.completions.create(**kwargs)

        text = ""
        pending: dict[int, dict[str, str]] = {}
        result = LLMResponse()
        async for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage:
                result.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
                result.output_tokens = getattr(usage, "completion_tokens", 0) or 0
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if getattr(delta, "content", None):
                text += delta.content
                await on_text(text)
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = pending.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if tc.id:
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if fn.name:
                        slot["name"] += fn.name
                    if fn.arguments:
                        slot["arguments"] += fn.arguments

        result.text = text
        result.tool_calls = [
            ToolCall(id=s["id"], name=s["name"], arguments=_parse_args(s["arguments"]))
            for _, s in sorted(pending.items())
        ]
        return result


class OpenAITranscriber:
    """Speech-to-text for voice messages via the OpenAI audio API.

    ``gpt-4o-mini-transcribe`` handles Uzbek and Russian well and is cheap;
    ``whisper-1`` also works.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini-transcribe",
        *,
        language: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any = None,
    ) -> None:
        if client is None:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self.client = client
        self.model = model
        self.language = language

    async def transcribe(self, audio: bytes, filename: str = "voice.ogg") -> str:
        kwargs: dict[str, Any] = {"model": self.model, "file": (filename, audio)}
        if self.language:
            kwargs["language"] = self.language
        resp = await self.client.audio.transcriptions.create(**kwargs)
        return (getattr(resp, "text", None) or "").strip()
