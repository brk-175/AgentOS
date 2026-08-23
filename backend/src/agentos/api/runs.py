"""Fix-run endpoints: start a run, stream its progress, read its state.

``POST /runs`` enqueues the Celery fix-workflow task for the authenticated
user's GitHub token. Progress is streamed via server-sent events from a Redis
pub/sub channel (``RunStore``); late subscribers get the bounded backlog
replayed first. A run ends with a ``final`` (or ``error``) event, after which
``GET /runs/{run_id}`` serves the completed state for 24h.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agentos.api.deps import AuthContext, DbSession
from agentos.core.config import get_settings
from agentos.core.logging import get_logger
from agentos.services.run_bus import RunStore
from agentos.services.run_records import get_run_record, list_run_records
from agentos.tasks import run_fix_workflow

logger = get_logger(__name__)

router = APIRouter(prefix="/runs", tags=["runs"])


def get_run_store(request: Request) -> RunStore:
    """Build a ``RunStore`` over the app's lifespan-managed Redis client."""
    return RunStore(request.app.state.redis)


RunStoreDep = Annotated[RunStore, Depends(get_run_store)]


class RunRequest(BaseModel):
    """What to fix: a repo + issue/PR number, plus branch options."""

    repo_full_name: str = Field(pattern=r"^[\w.-]+/[\w.-]+$")
    kind: Literal["issue", "pr"] = "issue"
    number: int = Field(gt=0)
    title: str = ""
    base_branch: str = "main"


class RunOut(BaseModel):
    run_id: str
    status: str


def enqueue_run(run_id: str, target: dict[str, Any], access_token: str, user_id: str) -> None:
    """Indirection over the Celery task so tests can capture enqueues."""
    run_fix_workflow.delay(run_id, target, access_token, user_id)


@router.post("", response_model=RunOut, status_code=202)
async def start_run(
    payload: RunRequest,
    auth: AuthContext,
    store: RunStoreDep,
) -> RunOut:
    """Enqueue a fix run (per-user concurrency limited)."""
    user_id = str(auth.user.id)
    if await store.count_active(user_id) >= get_settings().run_concurrency_per_user:
        raise HTTPException(status_code=429, detail="Concurrent run limit reached")
    run_id = uuid4().hex
    await store.add_active(user_id, run_id)
    try:
        enqueue_run(run_id, payload.model_dump(), auth.access_token, user_id)
        # Readers must see "queued" (not 404) before the worker publishes anything.
        await store.mark_queued(run_id)
    except Exception as exc:  # noqa: BLE001 - broker down; keep the client informed
        await store.set_final(
            run_id, {"status": "failed", "detail": f"enqueue failed: {exc}"}, user_id=None
        )
        logger.warning("run %s could not be enqueued", run_id, exc_info=True)
        raise HTTPException(status_code=503, detail="Run queue unavailable") from exc
    return RunOut(run_id=run_id, status="queued")


async def _load_run_state(store: RunStore, run_id: str) -> tuple[dict[str, Any] | None, list[str]]:
    """Live Redis state for a run (``(None, [])`` when the run is unknown)."""
    final = await store.get_final(run_id)
    backlog = await store.backlog(run_id)
    return final, backlog


@router.get("")
async def list_runs(
    auth: AuthContext,
    db: DbSession,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Durable run history for the authenticated user (newest first)."""
    if limit > 100:
        limit = 100
    return await list_run_records(db, auth.user.id, limit=limit)


@router.get("/{run_id}")
async def get_run(run_id: str, store: RunStoreDep, db: DbSession) -> dict[str, Any]:
    """Current run state: pending events + final result (once completed).

    Serves the live Redis view while the run is active/warm; falls back to
    the durable ``fix_runs`` record (+ judge evaluation) after the 24h TTL.
    """
    final, backlog = await _load_run_state(store, run_id)
    if final is not None or backlog:
        events = [json.loads(item) for item in backlog]
        return {
            "run_id": run_id,
            "status": (final or {}).get("status", "running"),
            "state": (final or {}).get("state"),
            "detail": (final or {}).get("detail"),
            "events": events,
        }
    record = await get_run_record(db, run_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run_id,
        "status": record["status"],
        "state": {
            "investigation": record["investigation"],
            "root_cause_hypothesis": record["root_cause_hypothesis"],
            "proposed_changes": record["proposed_changes"],
            "applied_branch": record["applied_branch"],
            "pr_url": record["pr_url"],
            "evaluation": record["evaluation"],
        },
        "detail": record["status"],
        "events": [],
    }
    if record is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "run_id": run_id,
        "status": record["status"],
        "state": {
            "investigation": record["investigation"],
            "root_cause_hypothesis": record["root_cause_hypothesis"],
            "proposed_changes": record["proposed_changes"],
            "applied_branch": record["applied_branch"],
            "pr_url": record["pr_url"],
            "evaluation": record["evaluation"],
        },
        "detail": record["status"],
        "events": [],
    }


@router.get("/{run_id}/events")
async def stream_run_events(run_id: str, store: RunStoreDep) -> StreamingResponse:
    """Server-sent events: backlog replay, then live events until the run ends."""
    final, backlog = await _load_run_state(store, run_id)
    if final is None and not backlog:
        raise HTTPException(status_code=404, detail="Run not found")
    pubsub = store.redis.pubsub()

    async def event_stream() -> AsyncIterator[str]:
        await pubsub.subscribe(store.channel(run_id))
        try:
            for item in await store.backlog(run_id):
                yield f"data: {item}\n\n"
            while True:
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=15)
                if message is None:
                    yield ": keepalive\n\n"
                    continue
                raw = message["data"]
                data = raw if isinstance(raw, str) else raw.decode("utf-8", "replace")
                yield f"data: {data}\n\n"
                if json.loads(data).get("type") in {"final", "error"}:
                    break
        finally:
            await pubsub.aclose()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
