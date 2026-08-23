"""In-memory stand-ins for the small slice of ``redis.asyncio`` we use.

Only the commands the ``RunStore`` and the runs API touch are implemented —
enough for unit tests without a Redis server.
"""

from __future__ import annotations

from typing import Any


class FakePipeline:
    """Buffered command batch that applies itself on ``execute``."""

    def __init__(self, store: FakeRedis) -> None:
        self._store = store
        self._ops: list[tuple[str, ...]] = []

    def publish(self, channel: str, data: str) -> FakePipeline:
        self._ops.append(("publish", channel, data))
        return self

    def rpush(self, key: str, data: str) -> FakePipeline:
        self._ops.append(("rpush", key, data))
        return self

    def ltrim(self, key: str, start: int, end: int) -> FakePipeline:
        self._ops.append(("ltrim", key, start, end))
        return self

    async def __aenter__(self) -> FakePipeline:
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        self._ops = []

    async def execute(self) -> None:
        ops, self._ops = self._ops, []
        for op in ops:
            command, *args = op
            await getattr(self._store, command)(*args)


class FakePubSub:
    """Preloaded message queue; ``get_message`` never blocks (used for SSE)."""

    def __init__(self, queues: dict[str, list[str]]) -> None:
        self._queues = {channel: list(items) for channel, items in queues.items()}

    async def subscribe(self, channel: str) -> None:
        self._queues.setdefault(channel, [])

    async def get_message(
        self, ignore_subscribe_messages: bool = True, timeout: float = 15
    ) -> dict[str, Any] | None:
        for _channel, items in self._queues.items():
            if items:
                return {"type": "message", "data": items.pop(0)}
        return None

    async def aclose(self) -> None:
        self._queues = {}


class FakeRedis:
    """Minimal async redis stand-in (strings, lists, sets, pub/sub)."""

    def __init__(self) -> None:
        self._strings: dict[str, str] = {}
        self._lists: dict[str, list[str]] = {}
        self._sets: dict[str, set[str]] = {}
        self.published: dict[str, list[str]] = {}
        self._pubsub_queues: dict[str, list[str]] = {}

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self._strings[key] = value

    async def get(self, key: str) -> str | None:
        return self._strings.get(key)

    async def rpush(self, key: str, value: str) -> None:
        self._lists.setdefault(key, []).append(value)

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        items = self._lists.get(key, [])
        length = len(items)
        start_index = max(start + length, 0) if start < 0 else min(start, length)
        end_index = end + length if end < 0 else min(end, length - 1)
        return items[start_index : end_index + 1]

    async def ltrim(self, key: str, start: int, end: int) -> None:
        items = self._lists.get(key, [])
        length = len(items)
        start_index = max(start + length, 0) if start < 0 else min(start, length)
        end_index = end + length if end < 0 else min(end, length - 1)
        self._lists[key] = items[start_index : end_index + 1]

    async def sadd(self, key: str, value: str) -> None:
        self._sets.setdefault(key, set()).add(value)

    async def srem(self, key: str, value: str) -> None:
        self._sets.get(key, set()).discard(value)

    async def scard(self, key: str) -> int:
        return len(self._sets.get(key, set()))

    async def expire(self, key: str, seconds: int) -> None:
        pass

    async def publish(self, channel: str, data: str) -> None:
        self.published.setdefault(channel, []).append(data)
        self._pubsub_queues.setdefault(channel, []).append(data)

    def pipeline(self) -> FakePipeline:
        return FakePipeline(self)

    def pubsub(self) -> FakePubSub:
        return FakePubSub(self._pubsub_queues)
