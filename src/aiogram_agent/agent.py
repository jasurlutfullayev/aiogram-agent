from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any

from .context import AgentContext
from .handoff import HandoffManager
from .memory import InMemoryMemory, Memory, trim_history
from .providers.base import LLMProvider, TextCallback
from .tools import Tool, tool
from .types import AgentReply, Image, Message

log = logging.getLogger("aiogram_agent")

SystemPrompt = str | Callable[[AgentContext], str | Awaitable[str]]
ReplyHook = Callable[[AgentContext, AgentReply], Any]


class Agent:
    """An LLM agent with tools and per-chat memory.

    The agent itself knows nothing about Telegram; plug it into aiogram with
    :func:`aiogram_agent.create_router`, or call :meth:`run` from anywhere.
    """

    def __init__(
        self,
        provider: LLMProvider,
        system_prompt: SystemPrompt = "You are a helpful assistant.",
        *,
        tools: Iterable[Tool] = (),
        memory: Memory | None = None,
        handoff: HandoffManager | None = None,
        max_steps: int = 6,
        max_history: int = 30,
        fallback_text: str = "Sorry, something went wrong. Please try again.",
        on_reply: ReplyHook | None = None,
    ) -> None:
        self.provider = provider
        self.system_prompt = system_prompt
        self.memory: Memory = memory or InMemoryMemory()
        self.handoff = handoff
        self.max_steps = max_steps
        self.max_history = max_history
        self.fallback_text = fallback_text
        self.on_reply = on_reply
        self._tools: dict[str, Tool] = {}
        self._locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)
        for t in tools:
            self.add_tool(t)
        if handoff is not None:
            self.add_tool(handoff.as_tool())

    # -- tools -------------------------------------------------------------
    @property
    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def add_tool(self, t: Tool) -> Tool:
        if not isinstance(t, Tool):
            raise TypeError("Expected a Tool; decorate the function with @tool")
        self._tools[t.name] = t
        return t

    def tool(self, func: Callable[..., Any] | None = None, **kwargs: Any) -> Any:
        """Register a tool on this agent: ``@agent.tool``."""

        def wrap(fn: Callable[..., Any]) -> Tool:
            return self.add_tool(tool(fn, **kwargs))

        return wrap(func) if func is not None else wrap

    @property
    def can_stream(self) -> bool:
        return callable(getattr(self.provider, "stream", None))

    # -- running -----------------------------------------------------------
    async def _system(self, ctx: AgentContext) -> str:
        if callable(self.system_prompt):
            value = self.system_prompt(ctx)
            if inspect.isawaitable(value):
                value = await value
            return str(value)
        return self.system_prompt

    async def reset(self, chat_id: int) -> None:
        await self.memory.clear(chat_id)
        if self.handoff:
            self.handoff.release(chat_id)

    async def run(
        self,
        text: str,
        ctx: AgentContext | int,
        *,
        images: Sequence[Image] = (),
        on_text: TextCallback | None = None,
    ) -> AgentReply:
        """Answer one user message.

        Args:
            text: The user's message.
            ctx: An :class:`AgentContext` or a bare chat id.
            images: Photos sent with the message (needs a vision model).
            on_text: Async callback receiving the reply as it is generated,
                ``await on_text(text_so_far)``. Used for streaming; ignored if
                the provider can't stream.
        """
        if isinstance(ctx, int):
            ctx = AgentContext(chat_id=ctx)
        async with self._locks[ctx.chat_id]:
            reply = await self._run(text, ctx, list(images), on_text)
        if self.on_reply is not None:
            try:
                result = self.on_reply(ctx, reply)
                if inspect.isawaitable(result):
                    await result
            except Exception:
                log.exception("on_reply hook failed")
        return reply

    async def _complete(self, system: str, messages: list[Message], on_text: TextCallback | None):
        if on_text is not None and self.can_stream:
            return await self.provider.stream(system, messages, self.tools, on_text)  # type: ignore[attr-defined]
        return await self.provider.complete(system, messages, self.tools)

    async def _run(
        self, text: str, ctx: AgentContext, images: list[Image], on_text: TextCallback | None
    ) -> AgentReply:
        history = trim_history(await self.memory.get(ctx.chat_id), self.max_history)
        system = await self._system(ctx)
        user: Message = {"role": "user", "content": text}
        if images:
            user["images"] = [dict(img) for img in images]
        new: list[Message] = [user]
        reply = AgentReply(text="")

        for _ in range(self.max_steps):
            try:
                resp = await self._complete(system, history + new, on_text)
            except Exception:
                log.exception("LLM provider failed for chat %s", ctx.chat_id)
                reply.text = self.fallback_text
                reply.failed = True
                return reply  # don't persist a half-finished turn
            reply.input_tokens += resp.input_tokens
            reply.output_tokens += resp.output_tokens

            assistant: Message = {"role": "assistant", "content": resp.text}
            if resp.tool_calls:
                assistant["tool_calls"] = [c.to_dict() for c in resp.tool_calls]
            new.append(assistant)

            if not resp.tool_calls:
                reply.text = resp.text
                break

            final_text: str | None = None
            for call in resp.tool_calls:
                reply.tool_calls.append(call)
                t = self._tools.get(call.name)
                if t is None:
                    result = f"Error: unknown tool '{call.name}'"
                else:
                    try:
                        result = await t(call.arguments, ctx)
                    except Exception as exc:  # report to the model, let it recover
                        log.exception("Tool %s failed", call.name)
                        result = f"Error: {type(exc).__name__}: {exc}"
                    else:
                        if t.final:
                            final_text = result
                            if call.name == "call_operator":
                                reply.handed_off = True
                new.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result})
            if final_text is not None:
                reply.text = final_text
                new.append({"role": "assistant", "content": final_text})
                break
        else:
            reply.text = reply.text or self.fallback_text

        await self.memory.append(ctx.chat_id, [_without_images(m) for m in new])
        return reply


def _without_images(message: Message) -> Message:
    """Images are sent to the model once; history keeps a short marker instead
    of megabytes of base64."""
    if "images" not in message:
        return message
    m = {k: v for k, v in message.items() if k != "images"}
    n = len(message["images"])
    marker = "[photo]" if n == 1 else f"[{n} photos]"
    m["content"] = f"{marker} {m.get('content') or ''}".strip()
    return m
