"""Unit tests for the run bus (Redis wrapper used by worker + API)."""

from __future__ import annotations

import json

from agentos.services.run_bus import RunStore
from tests.fake_redis import FakeRedis


async def test_append_event_publishes_and_bounds_backlog() -> None:
    fake = FakeRedis()
    store = RunStore(fake)
    run_id = "run-abc"
    for i in range(10):
        await store.append_event(run_id, {"type": "event", "index": i})
    assert fake.published[f"agentos:runs:{run_id}"] == [
        json.dumps({"type": "event", "index": i}) for i in range(10)
    ]
    assert len(await store.backlog(run_id)) == 10
    assert json.loads((await store.backlog(run_id))[0]) == {"type": "event", "index": 0}


async def test_backlog_is_bounded() -> None:
    fake = FakeRedis()
    store = RunStore(fake)
    for i in range(2100):
        await store.append_event("run-x", {"index": i})
    backlog = await store.backlog("run-x")
    assert len(backlog) == 2000
    assert json.loads(backlog[0]) == {"index": 100}


async def test_set_final_retires_active_run() -> None:
    fake = FakeRedis()
    store = RunStore(fake)
    await store.add_active(7, "run-a")
    await store.add_active(7, "run-b")
    assert await store.count_active(7) == 2

    await store.set_final("run-a", {"status": "completed"}, user_id=7)
    final_a = await store.get_final("run-a")
    assert final_a is not None
    assert final_a["status"] == "completed"
    assert await store.count_active(7) == 1

    await store.set_final("run-z", {"status": "failed"}, user_id=None)
    final_z = await store.get_final("run-z")
    assert final_z is not None
    assert final_z["status"] == "failed"


async def test_unknown_run_has_no_state() -> None:
    store = RunStore(FakeRedis())
    assert await store.get_final("nope") is None
    assert await store.backlog("nope") == []
    assert await store.count_active(1) == 0
