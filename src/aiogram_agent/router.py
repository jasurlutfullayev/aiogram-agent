"""Ready-made aiogram 3 router that connects an :class:`Agent` to a bot."""

from __future__ import annotations

import base64
import logging
import random
import time
from typing import Protocol

from aiogram import Bot, F, Router
from aiogram.enums import ChatAction, ChatType
from aiogram.exceptions import TelegramAPIError, TelegramBadRequest
from aiogram.filters import Command
from aiogram.types import Message
from aiogram.utils.chat_action import ChatActionSender

from .agent import Agent
from .context import AgentContext
from .ratelimit import RateLimiter
from .types import Image

log = logging.getLogger("aiogram_agent")

TELEGRAM_LIMIT = 4096


class Transcriber(Protocol):
    async def transcribe(self, audio: bytes, filename: str = ...) -> str: ...


def split_text(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Split long text on paragraph/line/word boundaries to fit Telegram."""
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    while len(text) > limit:
        cut = -1
        for sep in ("\n\n", "\n", " "):
            cut = text.rfind(sep, 0, limit)
            if cut > 0:
                break
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


def context_from_message(message: Message, bot: Bot) -> AgentContext:
    user = message.from_user
    return AgentContext(
        chat_id=message.chat.id,
        user_id=user.id if user else None,
        username=user.username if user else None,
        full_name=user.full_name if user else None,
        language_code=user.language_code if user else None,
        bot=bot,
        message=message,
    )


class StreamWriter:
    """Shows the reply while it is being generated.

    In private chats it uses Telegram's native ``sendMessageDraft`` (the reply
    "types itself" like in ChatGPT). Elsewhere, or if drafts fail, it sends one
    message and edits it.
    """

    def __init__(self, bot: Bot, chat_id: int, *, use_draft: bool, interval: float = 0.7) -> None:
        self.bot = bot
        self.chat_id = chat_id
        # Drafts need aiogram >= 3.23 / Bot API 9.3; older aiogram falls back to edits.
        self.use_draft = use_draft and hasattr(bot, "send_message_draft")
        self.interval = interval
        self.draft_id = random.randint(1, 2**31 - 1)
        self.message_id: int | None = None
        self._last_sent = ""
        self._last_time = 0.0

    async def __call__(self, text: str) -> None:
        now = time.monotonic()
        if now - self._last_time < self.interval or not text.strip():
            return
        self._last_time = now
        preview = text[:TELEGRAM_LIMIT]
        if preview == self._last_sent:
            return
        try:
            if self.use_draft:
                await self.bot.send_message_draft(chat_id=self.chat_id, draft_id=self.draft_id, text=preview)
            elif self.message_id is None:
                sent = await self.bot.send_message(self.chat_id, preview)
                self.message_id = sent.message_id
            else:
                await self.bot.edit_message_text(
                    text=preview, chat_id=self.chat_id, message_id=self.message_id
                )
            self._last_sent = preview
        except TelegramAPIError as exc:
            if self.use_draft:
                log.info("Drafts unavailable (%s); falling back to message edits", exc)
                self.use_draft = False
            # "message is not modified" and similar are harmless while streaming

    async def finish(self, text: str) -> None:
        chunks = split_text(text)
        if not chunks:
            return
        if self.message_id is not None:
            first, chunks = chunks[0], chunks[1:]
            if first != self._last_sent:
                try:
                    await self.bot.edit_message_text(
                        text=first, chat_id=self.chat_id, message_id=self.message_id
                    )
                except TelegramBadRequest:
                    chunks.insert(0, first)
        for chunk in chunks:
            await self.bot.send_message(self.chat_id, chunk)


def create_router(
    agent: Agent,
    *,
    name: str = "aiogram_agent",
    private_only: bool = True,
    reset_command: str | None = "reset",
    reset_text: str = "Conversation cleared.",
    show_typing: bool = True,
    stream: bool = False,
    photos: bool = True,
    transcriber: Transcriber | None = None,
    rate_limit: RateLimiter | None = None,
    rate_limit_text: str = "Too many messages, please wait a minute.",
) -> Router:
    """Build a router. Include it **last** so your own command handlers win::

    dp.include_router(my_router)
    dp.include_router(create_router(agent, stream=True))

    Args:
        stream: Show the answer while it is generated (needs a provider with
            ``stream``; OpenAI and Anthropic providers support it).
        photos: Pass photos to the model (use a vision-capable model).
        transcriber: e.g. ``OpenAITranscriber()`` to answer voice messages.
        rate_limit: e.g. ``RateLimiter(10, 60)`` to cap messages per user.
    """
    router = Router(name=name)
    handoff = agent.handoff
    scope = [F.chat.type == ChatType.PRIVATE] if private_only else []

    if handoff is not None:

        @router.message(F.chat.id == handoff.operator_chat_id, F.reply_to_message)
        async def operator_reply(message: Message, bot: Bot) -> None:
            chat_id = handoff.chat_for(message.reply_to_message.message_id)
            if chat_id is None:
                return
            if (message.text or "").strip().lower() in ("/done", "/release"):
                handoff.release(chat_id)
                await message.reply("✅ Returned to the bot.")
                if handoff.released_notice:
                    await bot.send_message(chat_id, handoff.released_notice)
                return
            await message.copy_to(chat_id)
            # Let the model see what the human said, so it stays consistent later.
            if message.text:
                await agent.memory.append(
                    chat_id, [{"role": "assistant", "content": f"[manager] {message.text}"}]
                )

    if reset_command:

        @router.message(Command(reset_command), *scope)
        async def reset(message: Message) -> None:
            await agent.reset(message.chat.id)
            await message.answer(reset_text)

    async def gate(message: Message, bot: Bot) -> AgentContext | None:
        """Common checks: rate limit and operator mode. Returns None if handled."""
        ctx = context_from_message(message, bot)
        if rate_limit is not None and not rate_limit.allow(ctx.user_id or ctx.chat_id):
            await message.answer(rate_limit_text)
            return None
        if handoff is not None and handoff.is_active(ctx.chat_id):
            if message.text:
                await handoff.forward_customer_message(bot, ctx, message.text)
            else:
                sent = await message.copy_to(handoff.operator_chat_id)
                handoff.link(sent.message_id, ctx.chat_id)
            await agent.memory.append(
                ctx.chat_id, [{"role": "user", "content": message.text or message.caption or "[media]"}]
            )
            return None
        return ctx

    async def respond(message: Message, bot: Bot, ctx: AgentContext, text: str, images: list[Image]) -> None:
        writer: StreamWriter | None = None
        if stream and agent.can_stream:
            writer = StreamWriter(bot, ctx.chat_id, use_draft=message.chat.type == ChatType.PRIVATE)
        if show_typing:
            async with ChatActionSender(bot=bot, chat_id=ctx.chat_id, action=ChatAction.TYPING):
                reply = await agent.run(text, ctx, images=images, on_text=writer)
        else:
            reply = await agent.run(text, ctx, images=images, on_text=writer)

        if writer is not None:
            await writer.finish(reply.text)
        else:
            for chunk in split_text(reply.text):
                await message.answer(chunk)

    @router.message(F.text, *scope)
    async def on_text(message: Message, bot: Bot) -> None:
        ctx = await gate(message, bot)
        if ctx is not None:
            await respond(message, bot, ctx, message.text or "", [])

    if photos:

        @router.message(F.photo, *scope)
        async def on_photo(message: Message, bot: Bot) -> None:
            ctx = await gate(message, bot)
            if ctx is None:
                return
            buf = await bot.download(message.photo[-1])
            image: Image = {"media_type": "image/jpeg", "data": base64.b64encode(buf.read()).decode()}
            await respond(message, bot, ctx, message.caption or "", [image])

    if transcriber is not None:

        @router.message(F.voice | F.audio | F.video_note, *scope)
        async def on_voice(message: Message, bot: Bot) -> None:
            ctx = await gate(message, bot)
            if ctx is None:
                return
            media = message.voice or message.audio or message.video_note
            async with ChatActionSender(bot=bot, chat_id=ctx.chat_id, action=ChatAction.TYPING):
                buf = await bot.download(media)
                filename = "voice.ogg" if message.voice else ("audio.mp3" if message.audio else "video.mp4")
                text = await transcriber.transcribe(buf.read(), filename)
            if not text:
                await message.answer("🎤 ?")
                return
            ctx.extra["transcribed"] = True
            await respond(message, bot, ctx, text, [])

    return router
