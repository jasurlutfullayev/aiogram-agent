"""LLM sales/support agents for aiogram 3 bots: tools, memory, human handoff."""

from .agent import Agent
from .context import AgentContext
from .handoff import HandoffManager
from .memory import InMemoryMemory, Memory, PostgresMemory, RedisMemory, SQLiteMemory
from .providers import LLMProvider
from .ratelimit import RateLimiter
from .tools import Tool, tool
from .types import AgentReply, Image, LLMResponse, ToolCall

__version__ = "0.2.0"

__all__ = [
    "Agent",
    "AgentContext",
    "AgentReply",
    "HandoffManager",
    "Image",
    "InMemoryMemory",
    "LLMProvider",
    "LLMResponse",
    "Memory",
    "PostgresMemory",
    "RateLimiter",
    "RedisMemory",
    "SQLiteMemory",
    "Tool",
    "ToolCall",
    "create_router",
    "tool",
]


def __getattr__(name: str):  # keep `import aiogram_agent` light for non-bot use
    if name == "create_router":
        from .router import create_router

        return create_router
    raise AttributeError(name)
