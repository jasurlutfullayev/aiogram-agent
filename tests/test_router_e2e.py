"""End-to-end: real aiogram Dispatcher + Router, fake Telegram HTTP session."""

import datetime as dt

import pytest
from aiogram import Bot, Dispatcher
from aiogram.client.session.base import BaseSession
from aiogram.methods import (
    CopyMessage,
    EditMessageText,
    GetFile,
    SendMessage,
    SendMessageDraft,
)
from aiogram.types import Chat, File, MessageId, Update, User
from aiogram.types import Message as TgMessage
from conftest import ScriptedProvider, call

from aiogram_agent import Agent, HandoffManager, LLMResponse, RateLimiter, create_router

USER = User(id=111, is_bot=False, first_name="Ali", username="ali", language_code="uz")
OPERATORS = -100500


class FakeSession(BaseSession):
    def __init__(self):
        super().__init__()
        self.requests = []
        self._id = 1000

    def sent(self, kind):
        return [r for r in self.requests if isinstance(r, kind)]

    async def make_request(self, bot, method, timeout=None):
        self.requests.append(method)
        self._id += 1
        if isinstance(method, SendMessage):
            return TgMessage(
                message_id=self._id,
                date=dt.datetime.now(),
                chat=Chat(id=method.chat_id, type="private"),
                text=method.text,
            )
        if isinstance(method, CopyMessage):
            return MessageId(message_id=self._id)
        if isinstance(method, GetFile):
            return File(file_id=method.file_id, file_unique_id="u", file_path="photos/p.jpg")
        if isinstance(method, EditMessageText):
            return True
        return True  # SendChatAction, SendMessageDraft, ...

    async def stream_content(self, url, headers=None, timeout=30, chunk_size=65536, raise_for_status=True):
        yield b"\xff\xd8jpeg-bytes"

    async def close(self):
        pass


def update(n, *, chat_id=USER.id, chat_type="private", text=None, reply_to=None, photo=False, voice=False):
    data = {
        "message_id": n,
        "date": dt.datetime.now(),
        "chat": Chat(id=chat_id, type=chat_type),
        "from_user": USER,
        "text": text,
    }
    if reply_to is not None:
        data["reply_to_message"] = TgMessage(
            message_id=reply_to, date=dt.datetime.now(), chat=Chat(id=chat_id, type=chat_type), text="x"
        )
    if photo:
        data["photo"] = [
            {"file_id": "small", "file_unique_id": "s", "width": 90, "height": 90},
            {"file_id": "big", "file_unique_id": "b", "width": 900, "height": 900},
        ]
        data["caption"] = "narxi?"
    if voice:
        data["voice"] = {"file_id": "v", "file_unique_id": "v", "duration": 2}
    return Update(update_id=n, message=TgMessage(**data))


@pytest.fixture
def env():
    session = FakeSession()
    bot = Bot("42:TEST", session=session)
    return bot, session


async def feed(bot, router, *updates):
    dp = Dispatcher()
    dp.include_router(router)
    for u in updates:
        await dp.feed_update(bot, u)


async def test_text_message_gets_agent_reply(env):
    bot, session = env
    agent = Agent(ScriptedProvider(LLMResponse(text="Assalomu alaykum!")))
    await feed(bot, create_router(agent), update(1, text="salom"))
    assert [m.text for m in session.sent(SendMessage)] == ["Assalomu alaykum!"]


async def test_long_reply_is_split(env):
    bot, session = env
    agent = Agent(ScriptedProvider(LLMResponse(text=("word " * 1000).strip())))
    await feed(bot, create_router(agent), update(1, text="tell me everything"))
    assert len(session.sent(SendMessage)) == 2


async def test_streaming_uses_drafts_then_sends_final(env):
    bot, session = env

    class Streaming(ScriptedProvider):
        async def stream(self, system, messages, tools, on_text):
            await on_text("Stul ")
            return await self.complete(system, messages, tools)

    agent = Agent(Streaming(LLMResponse(text="Stul narxi 450 000 so'm")))
    await feed(bot, create_router(agent, stream=True), update(1, text="narxi?"))
    assert session.sent(SendMessageDraft)[0].text == "Stul "
    assert [m.text for m in session.sent(SendMessage)] == ["Stul narxi 450 000 so'm"]


async def test_photo_reaches_model_as_image(env):
    bot, session = env
    provider = ScriptedProvider(LLMResponse(text="Bu ofis kreslosi"))
    await feed(bot, create_router(Agent(provider)), update(1, photo=True))
    user_msg = provider.calls[0][1][-1]
    assert user_msg["content"] == "narxi?"
    assert user_msg["images"][0]["media_type"] == "image/jpeg"
    assert session.sent(GetFile)[0].file_id == "big"


async def test_voice_is_transcribed(env):
    bot, session = env

    class FakeTranscriber:
        async def transcribe(self, audio, filename="voice.ogg"):
            assert filename == "voice.ogg"
            return "stul bormi"

    provider = ScriptedProvider(LLMResponse(text="Ha, bor"))
    await feed(bot, create_router(Agent(provider), transcriber=FakeTranscriber()), update(1, voice=True))
    assert provider.calls[0][1][-1]["content"] == "stul bormi"
    assert session.sent(SendMessage)[-1].text == "Ha, bor"


async def test_rate_limit(env):
    bot, session = env
    agent = Agent(ScriptedProvider(LLMResponse(text="a"), LLMResponse(text="b")))
    router = create_router(agent, rate_limit=RateLimiter(1, 60), rate_limit_text="slow down")
    await feed(bot, router, update(1, text="1"), update(2, text="2"))
    assert [m.text for m in session.sent(SendMessage)] == ["a", "slow down"]


async def test_full_handoff_cycle(env):
    bot, session = env
    handoff = HandoffManager(OPERATORS, customer_notice="Menejer ulanmoqda")
    provider = ScriptedProvider(
        LLMResponse(tool_calls=[call("call_operator", reason="chegirma so'rayapti")]),
        LLMResponse(text="Yana savol bormi?"),
    )
    agent = Agent(provider, handoff=handoff)
    router = create_router(agent)
    dp = Dispatcher()
    dp.include_router(router)

    await dp.feed_update(bot, update(1, text="chegirma bering"))
    to_ops, to_user = session.sent(SendMessage)
    assert to_ops.chat_id == OPERATORS and "chegirma so'rayapti" in to_ops.text
    assert to_user.text == "Menejer ulanmoqda"
    ops_msg_id = 1001  # first message sent by FakeSession

    await dp.feed_update(bot, update(2, text="10% bo'ladimi?"))  # forwarded, not answered by the model
    assert len(provider.calls) == 1
    assert session.sent(SendMessage)[-1].chat_id == OPERATORS

    await dp.feed_update(
        bot, update(3, chat_id=OPERATORS, chat_type="supergroup", text="5% beramiz", reply_to=ops_msg_id)
    )
    assert session.sent(CopyMessage)[-1].chat_id == USER.id

    await dp.feed_update(
        bot, update(4, chat_id=OPERATORS, chat_type="supergroup", text="/done", reply_to=ops_msg_id)
    )
    assert not handoff.is_active(USER.id)

    await dp.feed_update(bot, update(5, text="rahmat"))
    assert session.sent(SendMessage)[-1].text == "Yana savol bormi?"
    history = provider.calls[-1][1]
    assert {"role": "assistant", "content": "[manager] 5% beramiz"} in history
