from conftest import ScriptedProvider, call

from aiogram_agent import Agent, AgentContext, HandoffManager, LLMResponse, SQLiteMemory, tool
from aiogram_agent.memory import trim_history


@tool
def price(item: str) -> str:
    """Get the price of an item."""
    return {"chair": "450 000 UZS"}.get(item, "not found")


async def test_plain_answer_is_saved_to_memory():
    agent = Agent(ScriptedProvider(LLMResponse(text="Salom!")), "be nice")
    reply = await agent.run("hi", 7)
    assert reply.text == "Salom!"
    assert await agent.memory.get(7) == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Salom!"},
    ]


async def test_tool_loop():
    provider = ScriptedProvider(
        LLMResponse(tool_calls=[call("price", item="chair")]),
        LLMResponse(text="Stul narxi 450 000 so'm."),
    )
    agent = Agent(provider, tools=[price])
    reply = await agent.run("chair price?", 1)
    assert reply.text == "Stul narxi 450 000 so'm."
    assert [c.name for c in reply.tool_calls] == ["price"]
    second_request = provider.calls[1][1]
    assert second_request[-1]["role"] == "tool"
    assert second_request[-1]["content"] == "450 000 UZS"


async def test_tool_error_is_reported_to_model():
    @tool
    def broken() -> str:
        """Always fails."""
        raise RuntimeError("db down")

    provider = ScriptedProvider(
        LLMResponse(tool_calls=[call("broken")]),
        LLMResponse(text="Sorry, try later."),
    )
    reply = await Agent(provider, tools=[broken]).run("x", 1)
    assert reply.text == "Sorry, try later."
    assert "RuntimeError: db down" in provider.calls[1][1][-1]["content"]


async def test_unknown_tool():
    provider = ScriptedProvider(LLMResponse(tool_calls=[call("nope")]), LLMResponse(text="ok"))
    await Agent(provider).run("x", 1)
    assert "unknown tool" in provider.calls[1][1][-1]["content"]


async def test_max_steps_falls_back():
    provider = ScriptedProvider(*[LLMResponse(tool_calls=[call("price", item="x")]) for _ in range(5)])
    agent = Agent(provider, tools=[price], max_steps=3, fallback_text="fallback")
    reply = await agent.run("x", 1)
    assert reply.text == "fallback"
    assert len(provider.calls) == 3


async def test_provider_crash_does_not_corrupt_memory():
    class Boom:
        async def complete(self, *a):
            raise ConnectionError

    agent = Agent(Boom(), fallback_text="fb")
    assert (await agent.run("x", 1)).text == "fb"
    assert await agent.memory.get(1) == []


async def test_dynamic_system_prompt_gets_context():
    provider = ScriptedProvider(LLMResponse(text="ok"))
    agent = Agent(provider, lambda ctx: f"lang={ctx.language_code}")
    await agent.run("x", AgentContext(chat_id=1, language_code="uz"))
    assert provider.calls[0][0] == "lang=uz"


async def test_handoff_tool_is_final_and_activates_operator_mode():
    handoff = HandoffManager(operator_chat_id=-100, customer_notice="Menejer ulanmoqda")
    provider = ScriptedProvider(LLMResponse(tool_calls=[call("call_operator", reason="wants discount")]))
    agent = Agent(provider, handoff=handoff)
    reply = await agent.run("chegirma bormi?", 5)
    assert reply.text == "Menejer ulanmoqda"
    assert reply.handed_off
    assert handoff.is_active(5)
    assert len(provider.calls) == 1  # model not asked again after a final tool
    await agent.reset(5)
    assert not handoff.is_active(5)


async def test_history_is_trimmed_without_orphan_tool_results():
    msgs = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "name": "t", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "1", "name": "t", "content": "r"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "d"},
    ]
    assert trim_history(msgs, 4) == msgs[4:]
    assert trim_history(msgs, 0) == msgs


async def test_sqlite_memory(tmp_path):
    mem = SQLiteMemory(str(tmp_path / "m.db"), max_stored=3)
    await mem.append(1, [{"role": "user", "content": str(i)} for i in range(5)])
    await mem.append(2, [{"role": "user", "content": "other"}])
    assert [m["content"] for m in await mem.get(1)] == ["2", "3", "4"]
    await mem.clear(1)
    assert await mem.get(1) == []
    assert len(await mem.get(2)) == 1
