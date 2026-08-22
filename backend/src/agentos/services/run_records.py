"""Durable run persistence: write/read ``fix_runs`` rows from the API/Celery.

``persist_run`` upserts by ``run_id`` (a run completes at most once — the
Celery task writes it after the pipeline ends, or a failed marker if the run
crashed). ``get_run_record`` serves ``GET /runs/{run_id}`` history after the
Redis 24h TTL, and ``list_run_records`` backs the user's run history.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select

from agentos.agent.state import RunTarget
from agentos.models.run_record import FixRun


async def persist_run(
    session: Any,
    *,
    run_id: str,
    user_id: str | uuid.UUID,
    target: RunTarget,
    status: str,
    applied_branch: str | None,
    pr_url: str | None,
    investigation: str | None,
    hypothesis: str | None,
    proposed_changes: list[dict[str, Any]],
    evaluation: dict[str, Any] | None,
) -> None:
    """Create/replace the durable record for a terminalized run."""
    existing = await session.scalar(select(FixRun).where(FixRun.run_id == run_id))
    if existing is None:
        existing = FixRun(run_id=run_id, user_id=uuid.UUID(str(user_id)))
        session.add(existing)
    existing.repo_full_name = target.repo_full_name
    existing.kind = target.kind
    existing.number = target.number
    existing.title = target.title
    existing.base_branch = target.base_branch
    existing.status = status
    existing.applied_branch = applied_branch
    existing.pr_url = pr_url
    existing.investigation = investigation or ""
    existing.root_cause_hypothesis = hypothesis or ""
    existing.proposed_changes = proposed_changes
    existing.evaluation = evaluation
    existing.completed_at = datetime.now(UTC)
    await session.commit()


async def _row_to_dict(row: FixRun) -> dict[str, Any]:
    return {
        "run_id": row.run_id,
        "repo_full_name": row.repo_full_name,
        "kind": row.kind,
        "number": row.number,
        "title": row.title,
        "base_branch": row.base_branch,
        "status": row.status,
        "applied_branch": row.applied_branch,
        "pr_url": row.pr_url,
        "investigation": row.investigation,
        "root_cause_hypothesis": row.root_cause_hypothesis,
        "proposed_changes": row.proposed_changes or [],
        "evaluation": row.evaluation,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
    }


async def get_run_record(session: Any, run_id: str) -> dict[str, Any] | None:
    """Return the durable record (or ``None`` when the run was never persisted)."""
    row = await session.scalar(select(FixRun).where(FixRun.run_id == run_id))
    return await _row_to_dict(row) if row is not None else None


async def list_run_records(
    session: Any,
    user_id: str | uuid.UUID,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Recent durable runs for a user, newest first."""
    rows = (
        await session.scalars(
            select(FixRun)
            .where(FixRun.user_id == uuid.UUID(str(user_id)))
            .order_by(FixRun.created_at.desc(), FixRun.run_id.desc())
            .limit(limit)
        )
    ).all()
    return [await _row_to_dict(row) for row in rows]
