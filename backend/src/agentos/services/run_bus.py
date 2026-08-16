"""Redis-backed run bus shared by the Celery worker and the API.

A run is identified by a ``run_id``. The worker appends every workflow event
to a bounded Redis list (backlog for late SSE subscribers) and publishes the
same payload onto a pub/sub channel; the API streams that channel to the
browser. ``set_final`` marks the run terminal (and retires it from the
per-user active set used by the concurrency guard).
"""

from __future__ import annotations

import json
from typing import Any

CHANNEL_TEMPLATE = "agentos:runs:{run_id}"
EVENTS_KEY_TEMPLATE = "agentos:runs:{run_id}:events"
FINAL_KEY_TEMPLATE = "agentos:runs:{run_id}:final"
ACTIVE_KEY_TEMPLATE = "agentos:users:{user_id}:active_runs"
BACKLOG_LIMIT = 2000
RUN_TTL_SECONDS = 60 * 60 * 24


class RunStore:
    """All Redis operations for a fix-agent run, behind one small API.

    ``redis`` is duck-typed (``redis.asyncio.Redis`` in production, the
    in-memory fake in tests); redis-py's async stubs churn too much to
    contract them precisely.
    """

    def __init__(self, redis: Any) -> None:
        self.redis = redis

    def channel(self, run_id: str) -> str:
        return CHANNEL_TEMPLATE.format(run_id=run_id)

    def events_key(self, run_id: str) -> str:
        return EVENTS_KEY_TEMPLATE.format(run_id=run_id)

    def final_key(self, run_id: str) -> str:
        return FINAL_KEY_TEMPLATE.format(run_id=run_id)

    def active_key(self, user_id: str | int) -> str:
        return ACTIVE_KEY_TEMPLATE.format(user_id=user_id)

    async def append_event(self, run_id: str, payload: dict[str, Any]) -> None:
        """Publish a payload to the run channel and append it to the backlog."""
        data = json.dumps(payload, default=str)
        events_key = self.events_key(run_id)
        async with self.redis.pipeline() as pipe:
            pipe.publish(self.channel(run_id), data)
            pipe.rpush(events_key, data)
            pipe.ltrim(events_key, -BACKLOG_LIMIT, -1)
        await pipe.execute()

    async def backlog(self, run_id: str) -> list[str]:
        return [str(item) for item in await self.redis.lrange(self.events_key(run_id), 0, -1)]

    async def set_final(
        self, run_id: str, payload: dict[str, Any], user_id: str | int | None
    ) -> None:
        await self.redis.set(
            self.final_key(run_id),
            json.dumps(payload, default=str),
            ex=RUN_TTL_SECONDS,
        )
        if user_id is not None:
            await self.redis.srem(self.active_key(user_id), run_id)

    async def get_final(self, run_id: str) -> dict[str, Any] | None:
        raw = await self.redis.get(self.final_key(run_id))
        if raw is None:
            return None
        return json.loads(str(raw))

    async def count_active(self, user_id: str | int) -> int:
        return int(await self.redis.scard(self.active_key(user_id)))

    async def add_active(self, user_id: str | int, run_id: str) -> None:
        key = self.active_key(user_id)
        await self.redis.sadd(key, run_id)
        await self.redis.expire(key, RUN_TTL_SECONDS)
