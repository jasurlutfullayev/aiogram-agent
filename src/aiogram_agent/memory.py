"""Conversation history storage, keyed by chat id."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections import defaultdict
from typing import Any, Protocol, runtime_checkable

from .types import Message


@runtime_checkable
class Memory(Protocol):
    async def get(self, chat_id: int) -> list[Message]: ...
    async def append(self, chat_id: int, messages: list[Message]) -> None: ...
    async def clear(self, chat_id: int) -> None: ...


def trim_history(messages: list[Message], max_messages: int) -> list[Message]:
    """Keep the tail of the history without breaking a tool-call exchange.

    The result always starts with a ``user`` message, so providers never see
    an orphaned tool result.
    """
    if max_messages <= 0 or len(messages) <= max_messages:
        tail = list(messages)
    else:
        tail = messages[-max_messages:]
    while tail and tail[0]["role"] != "user":
        tail.pop(0)
    return tail


class InMemoryMemory:
    """Process-local memory. Fine for development; lost on restart."""

    def __init__(self) -> None:
        self._data: dict[int, list[Message]] = defaultdict(list)

    async def get(self, chat_id: int) -> list[Message]:
        return list(self._data.get(chat_id, []))

    async def append(self, chat_id: int, messages: list[Message]) -> None:
        self._data[chat_id].extend(messages)

    async def clear(self, chat_id: int) -> None:
        self._data.pop(chat_id, None)


class SQLiteMemory:
    """Persistent memory in a single SQLite file. No extra dependencies.

    ``max_stored`` caps how many messages per chat are kept on disk.
    """

    def __init__(self, path: str = "agent_memory.db", max_stored: int = 200) -> None:
        self.path = path
        self.max_stored = max_stored
        self._lock = asyncio.Lock()
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS agent_messages ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT,"
                " chat_id INTEGER NOT NULL,"
                " payload TEXT NOT NULL)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS ix_agent_messages_chat ON agent_messages(chat_id, id)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _get(self, chat_id: int) -> list[Message]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT payload FROM agent_messages WHERE chat_id = ? ORDER BY id", (chat_id,)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def _append(self, chat_id: int, messages: list[Message]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO agent_messages(chat_id, payload) VALUES (?, ?)",
                [(chat_id, json.dumps(m, ensure_ascii=False)) for m in messages],
            )
            if self.max_stored > 0:
                conn.execute(
                    "DELETE FROM agent_messages WHERE chat_id = ? AND id NOT IN ("
                    " SELECT id FROM agent_messages WHERE chat_id = ? ORDER BY id DESC LIMIT ?)",
                    (chat_id, chat_id, self.max_stored),
                )

    def _clear(self, chat_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM agent_messages WHERE chat_id = ?", (chat_id,))

    async def get(self, chat_id: int) -> list[Message]:
        async with self._lock:
            return await asyncio.to_thread(self._get, chat_id)

    async def append(self, chat_id: int, messages: list[Message]) -> None:
        async with self._lock:
            await asyncio.to_thread(self._append, chat_id, messages)

    async def clear(self, chat_id: int) -> None:
        async with self._lock:
            await asyncio.to_thread(self._clear, chat_id)


class RedisMemory:
    """Memory in Redis — survives restarts and works with several bot replicas.

    Requires ``pip install aiogram-agent[redis]``. Pass a ``redis.asyncio``
    client (``Redis.from_url("redis://localhost")``).
    """

    def __init__(
        self,
        redis: Any,
        *,
        prefix: str = "aiogram_agent:history:",
        max_stored: int = 200,
        ttl_seconds: int | None = 60 * 60 * 24 * 30,
    ) -> None:
        self.redis = redis
        self.prefix = prefix
        self.max_stored = max_stored
        self.ttl = ttl_seconds

    def _key(self, chat_id: int) -> str:
        return f"{self.prefix}{chat_id}"

    async def get(self, chat_id: int) -> list[Message]:
        raw = await self.redis.lrange(self._key(chat_id), 0, -1)
        return [json.loads(item) for item in raw]

    async def append(self, chat_id: int, messages: list[Message]) -> None:
        if not messages:
            return
        key = self._key(chat_id)
        pipe = self.redis.pipeline()
        pipe.rpush(key, *[json.dumps(m, ensure_ascii=False) for m in messages])
        if self.max_stored > 0:
            pipe.ltrim(key, -self.max_stored, -1)
        if self.ttl:
            pipe.expire(key, self.ttl)
        await pipe.execute()

    async def clear(self, chat_id: int) -> None:
        await self.redis.delete(self._key(chat_id))


class PostgresMemory:
    """Memory in PostgreSQL (``jsonb``), via asyncpg.

    Requires ``pip install aiogram-agent[postgres]``::

        pool = await asyncpg.create_pool(DATABASE_URL)
        memory = PostgresMemory(pool)
        await memory.setup()   # creates the table once
    """

    def __init__(self, pool: Any, *, table: str = "aiogram_agent_messages", max_stored: int = 200) -> None:
        if not table.replace("_", "").isalnum():
            raise ValueError("table name must be alphanumeric/underscore")
        self.pool = pool
        self.table = table
        self.max_stored = max_stored

    async def setup(self) -> None:
        await self.pool.execute(
            f"CREATE TABLE IF NOT EXISTS {self.table} ("
            " id BIGSERIAL PRIMARY KEY,"
            " chat_id BIGINT NOT NULL,"
            " payload JSONB NOT NULL,"
            " created_at TIMESTAMPTZ NOT NULL DEFAULT now())"
        )
        await self.pool.execute(
            f"CREATE INDEX IF NOT EXISTS {self.table}_chat_idx ON {self.table}(chat_id, id)"
        )

    async def get(self, chat_id: int) -> list[Message]:
        rows = await self.pool.fetch(
            f"SELECT payload::text FROM {self.table} WHERE chat_id = $1 ORDER BY id", chat_id
        )
        return [json.loads(r[0]) for r in rows]

    async def append(self, chat_id: int, messages: list[Message]) -> None:
        if not messages:
            return
        async with self.pool.acquire() as conn, conn.transaction():
            await conn.executemany(
                f"INSERT INTO {self.table}(chat_id, payload) VALUES ($1, $2::jsonb)",
                [(chat_id, json.dumps(m, ensure_ascii=False)) for m in messages],
            )
            if self.max_stored > 0:
                await conn.execute(
                    f"DELETE FROM {self.table} WHERE chat_id = $1 AND id < ("
                    f" SELECT min(id) FROM (SELECT id FROM {self.table} WHERE chat_id = $1"
                    " ORDER BY id DESC LIMIT $2) t)",
                    chat_id,
                    self.max_stored,
                )

    async def clear(self, chat_id: int) -> None:
        await self.pool.execute(f"DELETE FROM {self.table} WHERE chat_id = $1", chat_id)
