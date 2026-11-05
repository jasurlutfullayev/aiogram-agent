from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiogram import Bot
    from aiogram.types import Message


@dataclass
class AgentContext:
    """Per-request data passed to tools that declare a parameter of this type."""

    chat_id: int
    user_id: int | None = None
    username: str | None = None
    full_name: str | None = None
    language_code: str | None = None
    bot: Bot | None = None
    message: Message | None = None
    #: Free-form storage for your own data (e.g. CRM ids). Lives for one request.
    extra: dict[str, Any] = field(default_factory=dict)
