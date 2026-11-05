"""Pass a conversation to a human operator and take it back.

Flow:
1. The model calls the built-in ``call_operator`` tool (e.g. the customer
   wants a discount, complains, or asks something outside the catalog).
2. The chat switches to *operator mode*: new customer messages are forwarded
   to ``operator_chat_id`` (a group or a person) instead of the model.
3. An operator **replies** to a forwarded message; the bot delivers the reply
   to the customer.
4. The operator replies with ``/done`` and the agent takes over again.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from .context import AgentContext
from .tools import Tool, tool

if TYPE_CHECKING:
    from aiogram import Bot


class HandoffManager:
    def __init__(
        self,
        operator_chat_id: int,
        *,
        customer_notice: str = "Connecting you to a manager, please wait a moment.",
        operator_header: str = "🙋 Operator needed",
        released_notice: str | None = None,
        max_tracked_messages: int = 5000,
    ) -> None:
        self.operator_chat_id = operator_chat_id
        self.customer_notice = customer_notice
        self.operator_header = operator_header
        self.released_notice = released_notice
        self._active: set[int] = set()
        # operator-chat message id -> customer chat id
        self._links: OrderedDict[int, int] = OrderedDict()
        self._max = max_tracked_messages

    # -- state -------------------------------------------------------------
    def is_active(self, chat_id: int) -> bool:
        return chat_id in self._active

    def activate(self, chat_id: int) -> None:
        self._active.add(chat_id)

    def release(self, chat_id: int) -> None:
        self._active.discard(chat_id)

    def link(self, operator_message_id: int, chat_id: int) -> None:
        self._links[operator_message_id] = chat_id
        while len(self._links) > self._max:
            self._links.popitem(last=False)

    def chat_for(self, operator_message_id: int) -> int | None:
        return self._links.get(operator_message_id)

    # -- telegram side -----------------------------------------------------
    @staticmethod
    def _who(ctx: AgentContext) -> str:
        parts = [ctx.full_name or "—"]
        if ctx.username:
            parts.append(f"@{ctx.username}")
        parts.append(f"id {ctx.user_id or ctx.chat_id}")
        return " · ".join(parts)

    async def notify_operator(self, bot: Bot, ctx: AgentContext, text: str) -> None:
        sent = await bot.send_message(self.operator_chat_id, text)
        self.link(sent.message_id, ctx.chat_id)

    async def forward_customer_message(self, bot: Bot, ctx: AgentContext, text: str) -> None:
        await self.notify_operator(bot, ctx, f"💬 {self._who(ctx)}\n\n{text}")

    def as_tool(self) -> Tool:
        manager = self

        @tool(name="call_operator", final=True)
        async def call_operator(reason: str, ctx: AgentContext) -> str:
            """Hand the conversation to a human manager. Use when the customer
            asks for a person, wants a custom deal or discount, is unhappy, or
            asks something you cannot answer from your tools.

            Args:
                reason: Short summary for the manager: what the customer wants.
            """
            manager.activate(ctx.chat_id)
            if ctx.bot is not None:
                await manager.notify_operator(
                    ctx.bot,
                    ctx,
                    f"{manager.operator_header}\n{manager._who(ctx)}\n\n{reason}\n\n"
                    "↩️ Reply to this message to answer. Reply /done to return the chat to the bot.",
                )
            return manager.customer_notice

        return call_operator
