# Changelog

## 0.2.0

- Streaming replies: native Telegram drafts in private chats, message edits elsewhere (`create_router(stream=True)`)
- Photos are passed to vision models (OpenAI and Anthropic)
- Voice, audio and video notes via `OpenAITranscriber`
- `RedisMemory` and `PostgresMemory` backends
- `RateLimiter` per user
- `on_reply` hook for usage/cost tracking; `AgentReply.failed`
- Media sent during hand-off is forwarded to managers
- End-to-end tests with a real aiogram `Dispatcher`

## 0.1.0

- `Agent` with tool calling loop, `@tool` decorator with schema from type hints and docstrings
- `InMemoryMemory`, `SQLiteMemory`
- `HandoffManager` with `call_operator`
- OpenAI and Anthropic providers
- `create_router` for aiogram 3
