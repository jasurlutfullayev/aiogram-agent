import os
import uuid

import pytest

from aiogram_agent import PostgresMemory, RedisMemory


async def test_redis_memory():
    fakeredis = pytest.importorskip("fakeredis")
    mem = RedisMemory(fakeredis.FakeAsyncRedis(), max_stored=3, ttl_seconds=100)
    await mem.append(1, [{"role": "user", "content": f"{i} салом"} for i in range(5)])
    await mem.append(1, [])
    assert [m["content"] for m in await mem.get(1)] == ["2 салом", "3 салом", "4 салом"]
    assert await mem.redis.ttl("aiogram_agent:history:1") > 0
    await mem.clear(1)
    assert await mem.get(1) == []


@pytest.fixture
async def pg_pool():
    asyncpg = pytest.importorskip("asyncpg")
    dsn = os.environ.get("AIOGRAM_AGENT_TEST_PG", "postgresql://postgres@localhost/postgres")
    try:
        pool = await asyncpg.create_pool(dsn, min_size=1, max_size=2, timeout=3)
    except Exception as exc:  # no server available
        pytest.skip(f"PostgreSQL not available: {exc}")
    yield pool
    await pool.close()


async def test_postgres_memory(pg_pool):
    table = f"t_{uuid.uuid4().hex[:8]}"
    mem = PostgresMemory(pg_pool, table=table, max_stored=3)
    try:
        await mem.setup()
        await mem.setup()  # idempotent
        await mem.append(5, [{"role": "user", "content": str(i)} for i in range(5)])
        await mem.append(6, [{"role": "user", "content": "other", "images": []}])
        assert [m["content"] for m in await mem.get(5)] == ["2", "3", "4"]
        assert (await mem.get(6))[0]["content"] == "other"
        await mem.clear(5)
        assert await mem.get(5) == []
    finally:
        await pg_pool.execute(f"DROP TABLE IF EXISTS {table}")


def test_postgres_rejects_bad_table_name():
    with pytest.raises(ValueError):
        PostgresMemory(None, table="x; drop table users")
