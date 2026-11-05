from types import SimpleNamespace as NS

from conftest import ScriptedProvider, call

from aiogram_agent import Agent, LLMResponse, RateLimiter, tool
from aiogram_agent.providers.anthropic import AnthropicProvider
from aiogram_agent.providers.openai import OpenAIProvider, OpenAITranscriber

IMG = {"media_type": "image/png", "data": "aGVsbG8="}


@tool
def stock(sku: str) -> str:
    """Stock level."""
    return "12"


class AsyncIter:
    def __init__(self, items):
        self.items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.items:
            raise StopAsyncIteration
        return self.items.pop(0)


def chunk(content=None, tool_calls=None, usage=None):
    choices = (
        []
        if content is None and tool_calls is None
        else [NS(delta=NS(content=content, tool_calls=tool_calls))]
    )
    return NS(choices=choices, usage=usage)


async def test_openai_stream_text_and_tool_calls():
    chunks = [
        chunk("Sal"),
        chunk("om"),
        chunk(tool_calls=[NS(index=0, id="c1", function=NS(name="stock", arguments='{"sk'))]),
        chunk(tool_calls=[NS(index=0, id=None, function=NS(name=None, arguments='u": "A1"}'))]),
        chunk(usage=NS(prompt_tokens=7, completion_tokens=3)),
    ]
    seen = []

    async def create(**kwargs):
        assert kwargs["stream"] is True
        return AsyncIter(chunks)

    p = OpenAIProvider(client=NS(chat=NS(completions=NS(create=create))))

    async def on_text(t):
        seen.append(t)

    out = await p.stream("s", [{"role": "user", "content": "hi"}], [stock], on_text)
    assert seen == ["Sal", "Salom"]
    assert out.text == "Salom"
    assert out.tool_calls[0].id == "c1"
    assert out.tool_calls[0].arguments == {"sku": "A1"}
    assert (out.input_tokens, out.output_tokens) == (7, 3)


async def test_anthropic_stream():
    class Stream:
        text_stream = AsyncIter(["Ass", "alomu alaykum"])

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get_final_message(self):
            return NS(
                content=[NS(type="text", text="Assalomu alaykum")], usage=NS(input_tokens=1, output_tokens=2)
            )

    p = AnthropicProvider(client=NS(messages=NS(stream=lambda **kw: Stream())))
    seen = []

    async def on_text(t):
        seen.append(t)

    out = await p.stream("s", [{"role": "user", "content": "hi"}], [], on_text)
    assert seen[-1] == "Assalomu alaykum"
    assert out.text == "Assalomu alaykum"


def test_images_are_converted_for_both_providers():
    msgs = [{"role": "user", "content": "what is this?", "images": [IMG]}]
    oa = OpenAIProvider._convert("", msgs)[0]["content"]
    assert oa[0] == {"type": "image_url", "image_url": {"url": "data:image/png;base64,aGVsbG8="}}
    assert oa[1] == {"type": "text", "text": "what is this?"}
    an = AnthropicProvider._convert(msgs)[0]["content"]
    assert an[0]["source"] == {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="}
    assert an[1] == {"type": "text", "text": "what is this?"}


def test_anthropic_merges_consecutive_assistant_messages():
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"role": "assistant", "content": "[manager] I can give 5% off"},
        {"role": "user", "content": "ok"},
    ]
    out = AnthropicProvider._convert(msgs)
    assert [m["role"] for m in out] == ["user", "assistant", "user"]
    assert len(out[1]["content"]) == 2


async def test_agent_sends_images_once_and_stores_marker():
    provider = ScriptedProvider(LLMResponse(text="A red chair"), LLMResponse(text="ok"))
    agent = Agent(provider)
    await agent.run("what is it?", 1, images=[IMG])
    assert provider.calls[0][1][-1]["images"] == [IMG]
    stored = await agent.memory.get(1)
    assert stored[0] == {"role": "user", "content": "[photo] what is it?"}
    await agent.run("price?", 1)
    assert all("images" not in m for m in provider.calls[1][1])


async def test_agent_streams_when_provider_supports_it():
    class Streaming(ScriptedProvider):
        async def stream(self, system, messages, tools, on_text):
            resp = await self.complete(system, messages, tools)
            if resp.text:
                await on_text(resp.text[:3])
                await on_text(resp.text)
            return resp

    provider = Streaming(LLMResponse(tool_calls=[call("stock", sku="A")]), LLMResponse(text="12 in stock"))
    seen = []

    async def on_text(t):
        seen.append(t)

    reply = await Agent(provider, tools=[stock]).run("x", 1, on_text=on_text)
    assert reply.text == "12 in stock"
    assert seen == ["12 ", "12 in stock"]


async def test_on_reply_hook_receives_usage():
    got = []
    agent = Agent(
        ScriptedProvider(LLMResponse(text="hi", input_tokens=10, output_tokens=2)),
        on_reply=lambda ctx, reply: got.append((ctx.chat_id, reply.input_tokens)),
    )
    await agent.run("x", 9)
    assert got == [(9, 10)]


async def test_transcriber():
    seen = {}

    async def create(**kwargs):
        seen.update(kwargs)
        return NS(text="  salom  ")

    t = OpenAITranscriber(language="uz", client=NS(audio=NS(transcriptions=NS(create=create))))
    assert await t.transcribe(b"ogg", "v.ogg") == "salom"
    assert seen["file"] == ("v.ogg", b"ogg")
    assert seen["language"] == "uz"


def test_rate_limiter():
    rl = RateLimiter(2, 60)
    assert rl.allow(1, now=0) and rl.allow(1, now=1)
    assert not rl.allow(1, now=2)
    assert rl.allow(2, now=2)
    assert rl.allow(1, now=61)
