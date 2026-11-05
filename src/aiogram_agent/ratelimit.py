from __future__ import annotations

import time
from collections import defaultdict, deque


class RateLimiter:
    """Sliding-window limit per user, to protect your LLM budget from spam.

    ``RateLimiter(10, 60)`` allows 10 messages per user per minute.
    """

    def __init__(self, max_messages: int, per_seconds: float) -> None:
        if max_messages < 1 or per_seconds <= 0:
            raise ValueError("max_messages must be >= 1 and per_seconds > 0")
        self.max_messages = max_messages
        self.per_seconds = per_seconds
        self._hits: defaultdict[int, deque[float]] = defaultdict(deque)

    def allow(self, user_id: int, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        hits = self._hits[user_id]
        while hits and now - hits[0] >= self.per_seconds:
            hits.popleft()
        if len(hits) >= self.max_messages:
            return False
        hits.append(now)
        return True
