"""Celery wiring: worker app + the fix-workflow task.

``execute_run`` is the async core — it streams the fix-agent graph and pushes
every fresh ``RunEvent`` through a ``publish`` callback, so unit tests can run
it without Redis or a real MCP server. The Celery task wraps it with a
Redis-backed publisher (``RunStore``) and always ends a run with a terminal
``final``/``error`` payload.

Worker:  poetry run celery -A agentos.tasks.celery_app worker --loglevel=info
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

import redis.asyncio as aioredis
from celery import Celery
from langchain_core.tools import BaseTool

from agentos.agent.graph import create_agent_graph, create_agent_llm
from agentos.agent.mcp_adapter import GitHubMCPTools
from agentos.agent.state import RunTarget
from agentos.core.config import get_settings
from agentos.services.run_bus import RunStore

logger = logging.getLogger(__name__)

celery_app = Celery("agentos")
celery_app.conf.update(
    broker_url=get_settings().celery_broker_url,
    result_backend=get_settings().celery_result_backend,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_default_queue="agentos",
    result_expires=60 * 60 * 24,
)


def _compact_state(state: dict[str, Any]) -> dict[str, Any]:
    """The view of a run the UI needs (no raw file contents)."""
    return {
        "investigation": state.get("investigation"),
        "root_cause_hypothesis": state.get("root_cause_hypothesis"),
        "proposed_changes": [change.model_dump() for change in state.get("proposed_changes") or []],
        "applied_branch": state.get("applied_branch"),
        "pr_url": state.get("pr_url"),
        "events": [event.model_dump(mode="json") for event in state.get("events") or []],
    }


async def execute_run(
    run_id: str,
    target: RunTarget,
    access_token: str | None,
    *,
    publish: Callable[[dict[str, Any]], Awaitable[None]],
    model: Any = None,
    tools: Sequence[BaseTool] | None = None,
) -> dict[str, Any]:
    """Run the fix-agent pipeline, publishing each fresh event as it happens.

    ``model``/``tools`` are injectable for tests; by default the task boots the
    real OpenRouter model and the GitHub MCP server with the user's token.
    """
    initial = {
        "target": target,
        "messages": [],
        "events": [],
        "context": [],
    }

    async def stream(stream_tools: Sequence[BaseTool]) -> dict[str, Any]:
        graph = create_agent_graph(
            model=model if model is not None else create_agent_llm(),
            tools=stream_tools,
            token=access_token,
        )
        final: dict[str, Any] = {}
        seen_events = 0
        first_snapshot = True
        async for snapshot in graph.astream(initial, stream_mode="values"):
            final = snapshot
            fresh = snapshot["events"][seen_events:]
            if fresh:
                for event in fresh:
                    await publish(
                        {
                            "run_id": run_id,
                            "type": "event",
                            "stage": event.stage,
                            "kind": event.kind,
                            "detail": event.detail,
                            "time": event.time.isoformat(),
                        }
                    )
                seen_events = len(snapshot["events"])
            elif first_snapshot:
                await publish({"run_id": run_id, "type": "start"})
            first_snapshot = False
        await publish({"run_id": run_id, "type": "final", "state": _compact_state(final)})
        return _compact_state(final)

    if tools is None:
        async with GitHubMCPTools(token=access_token) as adapter:
            return await stream(adapter.tools)
    return await stream(tools)


@celery_app.task(name="agentos.run_fix_workflow")
def run_fix_workflow(
    run_id: str,
    target: dict[str, Any],
    access_token: str,
    user_id: str,
) -> dict[str, Any]:
    """Celery entrypoint: run the fix for ``target`` and stream events to Redis."""

    async def _inner() -> dict[str, Any]:
        settings = get_settings()
        redis = aioredis.from_url(
            settings.redis_url,
            socket_connect_timeout=5,
            socket_timeout=5,
            decode_responses=True,
        )
        store = RunStore(redis)

        async def publish(payload: dict[str, Any]) -> None:
            await store.append_event(run_id, payload)

        try:
            result = await execute_run(
                run_id,
                RunTarget.model_validate(target),
                access_token,
                publish=publish,
            )
            await store.set_final(run_id, {"status": "completed", "state": result}, user_id=user_id)
            return result
        except Exception as exc:  # noqa: BLE001 - runs must always terminate visibly
            logger.exception("fix run %s failed", run_id)
            await publish(
                {
                    "run_id": run_id,
                    "type": "error",
                    "stage": "run",
                    "kind": "error",
                    "detail": f"run failed: {exc}",
                }
            )
            failed = {"status": "failed", "detail": str(exc)}
            await store.set_final(run_id, failed, user_id=user_id)
            return failed
        finally:
            await redis.aclose()

    return asyncio.run(_inner())
