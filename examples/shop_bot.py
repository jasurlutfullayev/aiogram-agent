"""A furniture-shop sales agent: catalog, orders, photos, voice, manager handoff.

pip install "aiogram-agent[openai]"
export BOT_TOKEN=...  OPENAI_API_KEY=...  OPERATOR_CHAT_ID=-100...
python examples/shop_bot.py
"""

import asyncio
import logging
import os

from aiogram import Bot, Dispatcher

from aiogram_agent import Agent, AgentContext, HandoffManager, RateLimiter, SQLiteMemory, create_router, tool
from aiogram_agent.providers import OpenAIProvider, OpenAITranscriber

CATALOG = [
    {"sku": "CH-01", "name": "Office chair Ergo", "price_uzs": 1_450_000, "stock": 12},
    {"sku": "CH-02", "name": "School chair Junior", "price_uzs": 390_000, "stock": 140},
    {"sku": "DS-01", "name": "Teacher desk 140x70", "price_uzs": 2_100_000, "stock": 0},
]
ORDERS: list[dict] = []


@tool
def search_catalog(query: str, max_price_uzs: int | None = None) -> list[dict]:
    """Find products by name.

    Args:
        query: Words from the product name, e.g. "chair".
        max_price_uzs: Optional price ceiling in Uzbek sum.
    """
    q = query.lower()
    return [
        p
        for p in CATALOG
        if any(w in p["name"].lower() for w in q.split())
        and (max_price_uzs is None or p["price_uzs"] <= max_price_uzs)
    ]


@tool
def create_order(sku: str, quantity: int, phone: str, ctx: AgentContext) -> str:
    """Place an order once the customer confirmed the item, quantity and phone.

    Args:
        sku: Product SKU from search_catalog.
        quantity: Number of units.
        phone: Customer phone, e.g. +998901234567.
    """
    ORDERS.append({"sku": sku, "qty": quantity, "phone": phone, "chat": ctx.chat_id})
    return f"Order #{len(ORDERS)} created"


SYSTEM = """You are a polite sales manager of a furniture company in Tashkent.
Answer in the customer's language (Uzbek Latin or Russian).
Always check prices and stock with search_catalog, never invent them.
Before create_order, confirm item, quantity and phone number.
If the customer wants a discount, a custom size or a person — call_operator."""


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    bot = Bot(os.environ["BOT_TOKEN"])
    handoff = HandoffManager(
        operator_chat_id=int(os.environ["OPERATOR_CHAT_ID"]),
        customer_notice="Menejerga ulayapman, bir daqiqa kuting 🙌",
    )
    agent = Agent(
        OpenAIProvider("gpt-4o-mini"),
        SYSTEM,
        tools=[search_catalog, create_order],
        memory=SQLiteMemory("shop.db"),
        handoff=handoff,
    )
    dp = Dispatcher()
    dp.include_router(
        create_router(
            agent,
            stream=True,  # reply "types itself" like ChatGPT
            transcriber=OpenAITranscriber(),  # answers voice messages
            rate_limit=RateLimiter(15, 60),  # protects your API budget
        )
    )
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
