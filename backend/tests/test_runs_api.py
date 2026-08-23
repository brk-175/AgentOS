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


async def test_newly_enqueued_run_is_visible_immediately(
    runs_env: tuple[httpx.AsyncClient, FakeRedis, list[tuple[str, Any]]],
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Regression: a just-enqueued run must read back as ``queued`` (not 404)."""
    client, _, _ = runs_env
    cookie = await seed_authenticated_user(db_factory, github_id=555, username="erin")
    client.cookies.set("agentos_session", cookie)

    created = await client.post(
        f"{API_PREFIX}/runs",
        json={"repo_full_name": "octocat/Hello-World", "number": 9},
    )
    assert created.status_code == 202
    run_id = created.json()["run_id"]

    status = await client.get(f"{API_PREFIX}/runs/{run_id}")
    assert status.status_code == 200
    body = status.json()
    assert body["status"] == "queued"
    assert body["state"] is None
    assert body["events"] == []


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


async def test_persist_run_roundtrip(db_factory: async_sessionmaker[AsyncSession]) -> None:
    from agentos.agent.state import RunTarget
    from agentos.services.run_records import get_run_record, list_run_records, persist_run

    target = RunTarget(
        repo_full_name="octocat/Hello-World", kind="issue", number=1, title="Crash"
    )
    async with db_factory() as session:
        await persist_run(
            session,
            run_id="abc123run",
            user_id="11111111-1111-1111-1111-111111111111",
            target=target,
            status="completed",
            applied_branch="fix/issue-1-1",
            pr_url="https://pr",
            investigation="crash",
            hypothesis="null guard",
            proposed_changes=[
                {"path": "x.py", "content": "ok", "edits": [], "delete": False, "explanation": ""}
            ],
            evaluation={
                "verdict": "approved",
                "scores": {"correctness": 5.0, "minimality": 5.0, "behavior_preservation": 5.0, "grounding": 5.0},
            },
        )
        row = await get_run_record(session, "abc123run")
        assert row is not None
        assert row["status"] == "completed"
        assert row["pr_url"] == "https://pr"
        assert row["evaluation"]["verdict"] == "approved"
        assert len(await list_run_records(session, "11111111-1111-1111-1111-111111111111")) == 1
        assert await get_run_record(session, "missing") is None


async def test_list_runs_includes_inflight_runs_first(
    runs_env: tuple[httpx.AsyncClient, FakeRedis, list[tuple[str, Any]]],
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    """Regression: a queued/running run must show in the history without a DB row."""
    client, fake, enqueued = runs_env
    cookie = await seed_authenticated_user(db_factory, github_id=666, username="frank")
    client.cookies.set("agentos_session", cookie)

    response = await client.post(
        f"{API_PREFIX}/runs",
        json={"repo_full_name": "octocat/Hello-World", "kind": "issue", "number": 4},
    )
    assert response.status_code == 202
    run_id = response.json()["run_id"]

    history = await client.get(f"{API_PREFIX}/runs")
    assert history.status_code == 200
    rows = history.json()
    assert len(rows) == 1
    assert rows[0]["run_id"] == run_id
    assert rows[0]["status"] == "queued"
    assert rows[0]["repo_full_name"] == "octocat/Hello-World"
    assert rows[0]["number"] == 4
    assert rows[0]["evaluation"] is None


async def test_list_runs_requires_auth(runs_env: tuple[httpx.AsyncClient, FakeRedis, list[tuple[str, Any]]]) -> None:
    client, _, _ = runs_env
    response = await client.get(f"{API_PREFIX}/runs")
    assert response.status_code == 401


async def test_get_run_falls_back_to_durable_record(
    runs_env: tuple[httpx.AsyncClient, FakeRedis, list[tuple[str, Any]]],
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, _, _ = runs_env
    cookie = await seed_authenticated_user(db_factory, github_id=333, username="carol")
    client.cookies.set("agentos_session", cookie)

    from agentos.agent.state import RunTarget
    from agentos.services.run_records import persist_run

    async with db_factory() as session:
        await persist_run(
            session,
            run_id="durable-run-1",
            user_id="33333333-3333-3333-3333-333333333333",
            target=RunTarget(repo_full_name="octocat/Hello-World", kind="issue", number=3),
            status="completed",
            applied_branch="fix/issue-3-99",
            pr_url="https://pr/9",
            investigation="i",
            hypothesis="h",
            proposed_changes=[],
            evaluation={"verdict": "changes_requested", "scores": {}},
        )

    missing = await client.get(f"{API_PREFIX}/runs/durable-run-1")
    assert missing.status_code == 200
    body = missing.json()
    assert body["status"] == "completed"
    assert body["state"]["pr_url"] == "https://pr/9"
    assert body["state"]["evaluation"]["verdict"] == "changes_requested"


async def test_list_runs_returns_history_for_user(
    runs_env: tuple[httpx.AsyncClient, FakeRedis, list[tuple[str, Any]]],
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, _, _ = runs_env
    cookie = await seed_authenticated_user(db_factory, github_id=444, username="dave")
    client.cookies.set("agentos_session", cookie)

    from agentos.agent.state import RunTarget
    from agentos.services.run_records import persist_run

    async with db_factory() as session:
        user = await session.scalar(select(User).where(User.github_id == 444))
        assert user is not None
        await persist_run(
            session,
            run_id="hist-run-1",
            user_id=user.id,
            target=RunTarget(repo_full_name="octocat/Hello-World", kind="issue", number=1),
            status="completed",
            applied_branch="fix/issue-1-1",
            pr_url=None,
            investigation="",
            hypothesis="",
            proposed_changes=[],
            evaluation=None,
        )
        await persist_run(
            session,
            run_id="hist-run-2",
            user_id=user.id,
            target=RunTarget(repo_full_name="octocat/Hello-World", kind="issue", number=2),
            status="failed",
            applied_branch=None,
            pr_url=None,
            investigation="",
            hypothesis="",
            proposed_changes=[],
            evaluation=None,
        )

    response = await client.get(f"{API_PREFIX}/runs")
    assert response.status_code == 200
    rows = response.json()
    assert [row["run_id"] for row in rows] == ["hist-run-2", "hist-run-1"]
    assert rows[0]["status"] == "failed"
