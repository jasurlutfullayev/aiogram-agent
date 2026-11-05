<div align="center">

# 🤖 aiogram-agent

**Turn any [aiogram 3](https://github.com/aiogram/aiogram) bot into an AI sales & support agent — in 10 lines.**

Tools from plain Python functions · memory · streaming replies · photos · voice · hand-off to a human manager

[![CI](https://github.com/jasurlutfullayev/aiogram-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/jasurlutfullayev/aiogram-agent/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/aiogram-agent.svg)](https://pypi.org/project/aiogram-agent/)
[![Python](https://img.shields.io/pypi/pyversions/aiogram-agent.svg)](https://pypi.org/project/aiogram-agent/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

```python
from aiogram_agent import Agent, tool, create_router
from aiogram_agent.providers import OpenAIProvider


@tool
def price(item: str) -> str:
    """Get the price of a product.

    Args:
        item: Product name.
    """
    return db.get_price(item)


agent = Agent(OpenAIProvider("gpt-4o-mini"), "You are a polite shop assistant.", tools=[price])
dp.include_router(create_router(agent, stream=True))
```

That's a production-shaped bot: it answers customers, calls **your** functions when it needs real data, remembers the conversation, streams the answer as it's generated, and splits long replies.

## Why not just call the OpenAI API?

Because a bot that sells needs more than one API call. This library is the boring-but-hard part, built from the patterns of Telegram sales agents the author runs in production for furniture, phone and installment shops.

| | ChatGPT-wrapper bot | **aiogram-agent** |
|---|---|---|
| Looks up real prices / stock / orders | ❌ makes them up | ✅ `@tool` on any function |
| Remembers the conversation | in RAM, if at all | ✅ SQLite · Redis · PostgreSQL |
| Customer wants a human | 🤷 | ✅ built-in `call_operator` → your managers' group |
| Reply appears gradually | ❌ waits 10 s | ✅ native Telegram drafts (`stream=True`) |
| Customer sends a photo / voice | ignored | ✅ vision models · speech-to-text |
| Someone spams the bot | your bill 📈 | ✅ `RateLimiter` per user |
| Switch OpenAI ↔ Claude ↔ local model | rewrite | ✅ change one line |
| Unit-testable without Telegram | ❌ | ✅ `await agent.run("hi", chat_id=1)` |

## Install

```bash
pip install "aiogram-agent[openai]"     # or [anthropic], [redis], [postgres], [all]
```

Python 3.10+, aiogram 3.10+ (streaming drafts need aiogram 3.23+, older versions fall back to message edits).

## Features

### 🛠 Tools are just functions

Type hints + Google-style docstring → JSON schema. Sync or async. Ask for `AgentContext` and you get the chat, user, `bot` and `message`.

```python
from typing import Literal
from aiogram_agent import tool, AgentContext


@tool
async def create_order(
    sku: str, quantity: int, delivery: Literal["pickup", "courier"], ctx: AgentContext
) -> dict:
    """Create an order after the customer confirmed everything.

    Args:
        sku: Product SKU.
        quantity: Units.
        delivery: How the customer gets the goods.
    """
    order = await crm.create(sku, quantity, delivery, telegram_id=ctx.user_id)
    await ctx.bot.send_photo(ctx.chat_id, order.invoice_url)  # full aiogram access
    return {"order_id": order.id}
```

Exceptions inside a tool are sent back to the model so it can apologise or retry — a failing tool doesn't break the conversation.

### 🙋 Hand-off to a human

```python
handoff = HandoffManager(operator_chat_id=-1001234567890, customer_notice="Connecting you to a manager…")
agent = Agent(provider, SYSTEM_PROMPT, tools=[...], handoff=handoff)
```

When the customer asks for a discount, complains or wants a person, the model calls `call_operator`. Managers get a summary in their group, **reply** to talk to the customer through the bot, and reply `/done` to give the chat back to the AI — which then sees what the manager promised.

### ⚡ Streaming, 📷 photos, 🎤 voice, 🛡 rate limits

```python
from aiogram_agent import RateLimiter
from aiogram_agent.providers import OpenAITranscriber

dp.include_router(
    create_router(
        agent,
        stream=True,  # answer "types itself", like ChatGPT
        transcriber=OpenAITranscriber(),  # voice messages → text (great for Uzbek & Russian)
        rate_limit=RateLimiter(15, 60),  # max 15 messages per user per minute
    )
)
```

Photos are passed to vision models automatically (`photos=True` by default); history keeps a `[photo]` marker instead of megabytes of base64.

### 🧠 Memory backends

```python
SQLiteMemory("bot.db")  # zero dependencies
RedisMemory(Redis.from_url(REDIS_URL), ttl_seconds=30 * 86400)  # several replicas
memory = PostgresMemory(await asyncpg.create_pool(DB_URL))
await memory.setup()
```

Or implement three async methods — `get`, `append`, `clear` — for anything else. History is trimmed without ever splitting a tool call from its result.

### 📊 Cost tracking

```python
async def log_usage(ctx, reply):
    await stats.add(ctx.user_id, reply.input_tokens, reply.output_tokens, reply.handed_off)


agent = Agent(provider, prompt, on_reply=log_usage)
```

### 🌐 Providers

```python
OpenAIProvider("gpt-4o-mini")
OpenAIProvider("llama-3.3-70b", base_url="https://api.groq.com/openai/v1")  # any OpenAI-compatible API
AnthropicProvider("claude-sonnet-4-5")
```

Writing your own is one method: `async def complete(system, messages, tools) -> LLMResponse`.

## Full example

[`examples/shop_bot.py`](examples/shop_bot.py) — furniture shop: catalog search, orders, photos, voice, streaming, managers' group. ~90 lines.

```bash
pip install "aiogram-agent[openai]"
export BOT_TOKEN=... OPENAI_API_KEY=... OPERATOR_CHAT_ID=-100...
python examples/shop_bot.py
```

## Testing your agent

```python
from aiogram_agent import Agent, LLMResponse


class Scripted:
    async def complete(self, system, messages, tools):
        return LLMResponse(text="Salom!")


reply = await Agent(Scripted()).run("hi", chat_id=1)
assert reply.text == "Salom!"
```

The library's own suite runs real aiogram `Dispatcher` updates against a fake Telegram session — see [`tests/test_router_e2e.py`](tests/test_router_e2e.py).

## Roadmap

Contributions are very welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Ideas:

- [ ] Google Gemini provider
- [ ] Persist hand-off state (survive restarts)
- [ ] Inline keyboards from tools (product cards with "Buy" buttons)
- [ ] Summarise long histories instead of trimming
- [ ] MCP tools support
- [ ] Group chats: answer only when mentioned

## O'zbekcha

`aiogram-agent` — aiogram 3 botini sotuv va mijozlarga xizmat ko'rsatuvchi AI agentga aylantiradi: oddiy funksiyalardan tool'lar, suhbat xotirasi, javobni jonli ko'rsatish, rasm va ovozli xabarlar, menejerga ulash. Issue va PR'larni o'zbek tilida ham yozishingiz mumkin.

## По-русски

`aiogram-agent` превращает бота на aiogram 3 в AI-агента продаж и поддержки: инструменты из обычных функций, память диалога, стриминг ответа, фото и голосовые, передача живому менеджеру. Issues и PR можно писать на русском.

## License

MIT © Jasur Lutfullaev
