"""Durable fix-run records: one row per pipeline execution.

Redis (``RunStore``) holds the live run bus for 24h; ``fix_runs`` is the
permanent history — target, terminal status, the branch/PR the run produced,
plus the judge verdict (``evaluation``) from the Step 5.4 stage. JSONB-ish
columns keep the model portable (JSON in sqlite for tests, native on PG).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from agentos.models.base import Base, TimestampMixin


class FixRun(Base, TimestampMixin):
    """A terminalized fix-agent run, persisted for history + the API."""

    __tablename__ = "fix_runs"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    run_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    repo_full_name: Mapped[str] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(16))
    number: Mapped[int | None] = mapped_column()
    title: Mapped[str] = mapped_column(String(512), default="")
    base_branch: Mapped[str] = mapped_column(String(128), default="main")
    status: Mapped[str] = mapped_column(String(16))  # completed | failed
    applied_branch: Mapped[str | None] = mapped_column(String(255))
    pr_url: Mapped[str | None] = mapped_column(String(512))
    investigation: Mapped[str] = mapped_column(default="")
    root_cause_hypothesis: Mapped[str] = mapped_column(default="")
    proposed_changes: Mapped[list[dict]] = mapped_column(JSON, default=list)
    evaluation: Mapped[dict | None] = mapped_column(JSON)  # judge verdict (Step 5.4)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
