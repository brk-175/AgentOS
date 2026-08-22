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
from agentos.agent.state import Retriever, RunTarget
from agentos.core.config import get_settings
from agentos.db.session import build_engine, build_session_factory
from agentos.services.judge import create_judge_llm, evaluate_run
from agentos.services.rag import build_retriever
from agentos.services.run_bus import RunStore
from agentos.services.run_records import persist_run

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
    retrieval: Retriever | None = None,
    judge: Any = None,
) -> dict[str, Any]:
    """Run the fix-agent pipeline, publishing each fresh event as it happens.

    ``model``/``tools``/``retrieval``/``judge`` are injectable for tests; by
    default the task boots the real opencode agent model and the GitHub MCP
    server with the user's token. When ``judge`` is provided (or a real model
    is active), the completed run is scored by the judge LLM and the verdict
    lands in the terminal state + an ``eval`` event (degradable).
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
            retrieval=retrieval,
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
        state = _compact_state(final)
        state["evaluation"] = await _evaluate(
            run_id, target, final, judge=judge, publish=publish
        )
        await publish({"run_id": run_id, "type": "final", "state": state})
        return state

    if tools is None:
        async with GitHubMCPTools(token=access_token) as adapter:
            return await stream(adapter.tools)
    return await stream(tools)


async def _evaluate(
    run_id: str,
    target: RunTarget,
    final: dict[str, Any],
    *,
    judge: Any,
    publish: Callable[[dict[str, Any]], Awaitable[None]],
) -> dict[str, Any] | None:
    """Score the finished run with the judge model; degrade on any failure.

    Skipped entirely when no judge is available (unit-test stub mode).
    Returns the serialized verdict dict (or ``None`` when evaluation was
    skipped/failed) and publishes an ``eval`` event for the SSE stream.
    """
    if judge is None:
        return None
    changes = final.get("proposed_changes") or []
    if not changes:
        await publish(
            {
                "run_id": run_id,
                "type": "event",
                "stage": "eval",
                "kind": "skip",
                "detail": "no changes to evaluate",
            }
        )
        return None
    try:
        verdict = await evaluate_run(
            target,
            investigation=final.get("investigation"),
            hypothesis=final.get("root_cause_hypothesis"),
            changes=changes,
            applied_branch=final.get("applied_branch"),
            pr_url=final.get("pr_url"),
            judge=judge,
        )
    except Exception as exc:  # noqa: BLE001 - evaluation must never kill the run
        logger.exception("judge evaluation for %s failed", target.repo_full_name)
        await publish(
            {
                "run_id": run_id,
                "type": "event",
                "stage": "eval",
                "kind": "error",
                "detail": f"evaluation failed: {exc}",
            }
        )
        return None
    await publish(
        {
            "run_id": run_id,
            "type": "event",
            "stage": "eval",
            "kind": "verdict",
            "detail": f"{verdict.verdict} (score {verdict.scores.model_dump()})",
        }
    )
    return verdict.model_dump()


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

        engine = None
        try:
            if access_token:
                engine = build_engine(settings)
                retrieval = build_retriever(engine)
            else:
                retrieval = None
            run_target = RunTarget.model_validate(target)
            result = await execute_run(
                run_id,
                run_target,
                access_token,
                publish=publish,
                retrieval=retrieval,
                judge=create_judge_llm(),
            )
            await store.set_final(run_id, {"status": "completed", "state": result}, user_id=user_id)
            try:
                await _persist_finished_run(engine, run_id, user_id, run_target, result)
            except Exception:  # noqa: BLE001 - persistence is best-effort
                logger.exception("persisting run %s failed (run itself succeeded)", run_id)
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
            try:
                await _persist_failed_run(engine, run_id, user_id, target)
            except Exception:  # noqa: BLE001 - persistence is best-effort
                logger.exception("persisting failed marker for run %s failed", run_id)
            return failed
        finally:
            if engine is not None:
                await engine.dispose()
            await redis.aclose()

    return asyncio.run(_inner())


async def _persist_finished_run(
    engine: Any, run_id: str, user_id: str, target: RunTarget, result: dict[str, Any]
) -> None:
    """Write the durable ``fix_runs`` row for a completed run (best-effort).

    ``result`` is the compacted state — ``proposed_changes`` are already plain
    dicts (``_compact_state`` dumps them), so pass them through as-is.
    """
    if engine is None:
        return
    factory = build_session_factory(engine)
    async with factory() as session:
        await persist_run(
            session,
            run_id=run_id,
            user_id=user_id,
            target=target,
            status="completed",
            applied_branch=result.get("applied_branch"),
            pr_url=result.get("pr_url"),
            investigation=result.get("investigation"),
            hypothesis=result.get("root_cause_hypothesis"),
            proposed_changes=result.get("proposed_changes") or [],
            evaluation=result.get("evaluation"),
        )


async def _persist_failed_run(
    engine: Any, run_id: str, user_id: str, target: dict[str, Any]
) -> None:
    """Write a failed marker row (best-effort; DB outage must not mask the real error)."""
    if engine is None:
        return
    factory = build_session_factory(engine)
    async with factory() as session:
        await persist_run(
            session,
            run_id=run_id,
            user_id=user_id,
            target=RunTarget.model_validate(target),
            status="failed",
            applied_branch=None,
            pr_url=None,
            investigation=None,
            hypothesis=None,
            proposed_changes=[],
            evaluation=None,
        )
