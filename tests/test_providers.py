from types import SimpleNamespace as NS

from aiogram_agent import tool
from aiogram_agent.providers.anthropic import AnthropicProvider
from aiogram_agent.providers.openai import OpenAIProvider

HISTORY = [
    {"role": "user", "content": "price?"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "c1", "name": "price", "arguments": {"item": "x"}}],
    },
    {"role": "tool", "tool_call_id": "c1", "name": "price", "content": "10"},
    {"role": "user", "content": "and y?"},
]


@tool
def price(item: str) -> str:
    """Price."""
    return "1"


class Recorder:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


async def test_openai_roundtrip():
    rec = Recorder(
        NS(
            choices=[
                NS(
                    message=NS(
                        content=None,
                        tool_calls=[NS(id="c2", function=NS(name="price", arguments='{"item": "y"}'))],
                    )
                )
            ],
            usage=NS(prompt_tokens=10, completion_tokens=2),
        )
    )
    p = OpenAIProvider(client=NS(chat=NS(completions=rec)))
    out = await p.complete("sys", HISTORY, [price])
    msgs = rec.kwargs["messages"]
    assert msgs[0] == {"role": "system", "content": "sys"}
    assert msgs[2]["tool_calls"][0]["function"]["arguments"] == '{"item": "x"}'
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": "10"}
    assert rec.kwargs["tools"][0]["function"]["name"] == "price"
    assert out.tool_calls[0].arguments == {"item": "y"}
    assert out.input_tokens == 10


async def test_anthropic_merges_tool_result_and_user_text():
    rec = Recorder(
        NS(
            content=[NS(type="text", text="Y costs 20")],
            usage=NS(input_tokens=5, output_tokens=3),
        )
    )
    p = AnthropicProvider(client=NS(messages=rec))
    out = await p.complete("sys", HISTORY, [price])
    msgs = rec.kwargs["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert msgs[1]["content"][0] == {"type": "tool_use", "id": "c1", "name": "price", "input": {"item": "x"}}
    assert msgs[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "c1", "content": "10"},
        {"type": "text", "text": "and y?"},
    ]
    assert rec.kwargs["system"] == "sys"
    assert rec.kwargs["tools"][0]["input_schema"]["required"] == ["item"]
    assert out.text == "Y costs 20"
