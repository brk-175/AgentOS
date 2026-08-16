"""Endpoint tests for the runs API: enqueue, status, SSE streaming."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentos.api.deps import get_db
from agentos.api.runs import get_run_store
from agentos.app import create_app
from agentos.core.config import get_settings
from agentos.models.user import User
from agentos.services.run_bus import RunStore
from tests.conftest import seed_authenticated_user
from tests.fake_redis import FakeRedis

API_PREFIX = get_settings().api_prefix


@pytest.fixture()
async def runs_env(
    db_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[tuple[httpx.AsyncClient, FakeRedis, list[tuple[Any, ...]]]]:
    fake = FakeRedis()
    app = create_app()
    app.state.redis = fake

    async def override_db() -> AsyncIterator[AsyncSession]:
        async with db_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    app.dependency_overrides[get_run_store] = lambda: RunStore(fake)

    enqueued: list[tuple[Any, ...]] = []

    def capture_enqueue(
        run_id: str, target: dict[str, Any], access_token: str, user_id: str
    ) -> None:
        enqueued.append((run_id, target, access_token, user_id))

    monkeypatch.setattr("agentos.api.runs.enqueue_run", capture_enqueue)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, fake, enqueued


async def _user_id(db_factory: async_sessionmaker[AsyncSession], github_id: int) -> uuid.UUID:
    async with db_factory() as session:
        user = await session.scalar(select(User).where(User.github_id == github_id))
    assert user is not None
    return user.id


async def test_start_run_requires_auth(
    runs_env: tuple[httpx.AsyncClient, FakeRedis, list[tuple[str, Any]]],
) -> None:
    client, _, _ = runs_env
    response = await client.post(
        f"{API_PREFIX}/runs",
        json={"repo_full_name": "octocat/Hello-World", "number": 1},
    )
    assert response.status_code == 401


async def test_start_run_enqueues_and_marks_active(
    runs_env: tuple[httpx.AsyncClient, FakeRedis, list[tuple[str, Any]]],
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, fake, enqueued = runs_env
    cookie = await seed_authenticated_user(db_factory, github_id=111, username="alice")
    client.cookies.set("agentos_session", cookie)

    response = await client.post(
        f"{API_PREFIX}/runs",
        json={
            "repo_full_name": "octocat/Hello-World",
            "kind": "pr",
            "number": 7,
            "base_branch": "trunk",
        },
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"
    assert len(body["run_id"]) == 32

    run_id, target, token, user_id = enqueued[0]
    assert run_id == body["run_id"]
    assert target == {
        "repo_full_name": "octocat/Hello-World",
        "kind": "pr",
        "number": 7,
        "title": "",
        "base_branch": "trunk",
    }
    assert token == "gho_test_access_token"
    assert isinstance(user_id, str)
    user_db_id = await _user_id(db_factory, github_id=111)
    assert user_id == str(user_db_id)
    assert await fake.scard(f"agentos:users:{user_id}:active_runs") == 1


async def test_start_run_rejects_above_concurrency(
    runs_env: tuple[httpx.AsyncClient, FakeRedis, list[tuple[str, Any]]],
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, fake, enqueued = runs_env
    cookie = await seed_authenticated_user(db_factory, github_id=222, username="bob")
    client.cookies.set("agentos_session", cookie)
    user_db_id = await _user_id(db_factory, github_id=222)
    limit = get_settings().run_concurrency_per_user
    for i in range(limit):
        await fake.sadd(f"agentos:users:{user_db_id}:active_runs", f"busy-{i}")

    response = await client.post(
        f"{API_PREFIX}/runs",
        json={"repo_full_name": "octocat/Hello-World", "number": 1},
    )
    assert response.status_code == 429
    assert enqueued == []


async def test_get_run_reports_running_and_completed(
    runs_env: tuple[httpx.AsyncClient, FakeRedis, list[tuple[str, Any]]],
) -> None:
    client, fake, _ = runs_env
    store = RunStore(fake)
    await store.append_event("run-1", {"type": "event", "stage": "investigate"})
    await store.append_event("run-1", {"type": "event", "stage": "design"})

    running = await client.get(f"{API_PREFIX}/runs/run-1")
    assert running.status_code == 200
    assert running.json()["status"] == "running"
    assert [e["stage"] for e in running.json()["events"]] == ["investigate", "design"]

    await store.set_final(
        "run-1",
        {"status": "completed", "state": {"pr_url": "https://example.com/pr"}},
        user_id=None,
    )
    done = await client.get(f"{API_PREFIX}/runs/run-1")
    assert done.json()["status"] == "completed"
    assert done.json()["state"]["pr_url"] == "https://example.com/pr"

    assert (await client.get(f"{API_PREFIX}/runs/unknown")).status_code == 404


async def test_run_events_streams_backlog_then_live_until_final(
    runs_env: tuple[httpx.AsyncClient, FakeRedis, list[tuple[str, Any]]],
) -> None:
    client, fake, _ = runs_env
    store = RunStore(fake)
    run_id = "run-stream"
    await store.append_event(run_id, {"type": "event", "stage": "investigate", "kind": "target"})
    await fake.publish(
        store.channel(run_id),
        json.dumps({"type": "final", "state": {"pr_url": "https://example.com/pr/9"}}),
    )

    async with client.stream("GET", f"{API_PREFIX}/runs/{run_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line async for line in response.aiter_lines()]

    data_lines = [line for line in lines if line.startswith("data: ")]
    assert json.loads(data_lines[0].removeprefix("data: "))["kind"] == "target"
    final_payload = json.loads(data_lines[-1].removeprefix("data: "))
    assert final_payload["type"] == "final"


async def test_run_events_404_for_unknown_run(
    runs_env: tuple[httpx.AsyncClient, FakeRedis, list[tuple[str, Any]]],
) -> None:
    client, _, _ = runs_env
    response = await client.get(f"{API_PREFIX}/runs/nope/events")
    assert response.status_code == 404
