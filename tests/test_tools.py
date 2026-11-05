from typing import Literal

from aiogram_agent import AgentContext, tool


def test_schema_from_signature_and_docstring():
    @tool
    async def find_product(
        query: str,
        max_price: int | None = None,
        sort: Literal["cheap", "new"] = "cheap",
        tags: list[str] | None = None,
        ctx: AgentContext = None,
    ) -> str:
        """Search the product catalog.

        Args:
            query: What the customer is looking for.
            max_price: Upper price limit in UZS.
        """
        return "x"

    assert find_product.name == "find_product"
    assert find_product.description == "Search the product catalog."
    props = find_product.parameters["properties"]
    assert props["query"] == {"type": "string", "description": "What the customer is looking for."}
    assert props["max_price"]["type"] == "integer"
    assert props["max_price"]["description"] == "Upper price limit in UZS."
    assert props["sort"] == {"enum": ["cheap", "new"], "type": "string", "default": "cheap"}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}}
    assert "ctx" not in props
    assert find_product.parameters["required"] == ["query"]
    assert find_product.takes_context == "ctx"


async def test_call_injects_context_and_serializes():
    @tool(name="whoami")
    def who(ctx: AgentContext) -> dict:
        """Return the chat id."""
        return {"chat": ctx.chat_id, "name": "Жасур"}

    out = await who({}, AgentContext(chat_id=42))
    assert out == '{"chat": 42, "name": "Жасур"}'


async def test_unknown_arguments_are_dropped():
    @tool
    def add(a: int, b: int) -> int:
        return a + b

    assert await add({"a": 2, "b": 3, "junk": 1}, AgentContext(chat_id=1)) == "5"
