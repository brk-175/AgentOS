"""User-connected GitHub repositories."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from agentos.models.base import Base, TimestampMixin


class Repository(Base, TimestampMixin):
    """A repository connected to a user's account."""

    __tablename__ = "repositories"
    __table_args__ = (
        UniqueConstraint("user_id", "full_name", name="uq_repositories_user_full_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    full_name: Mapped[str] = mapped_column(String(255))  # "owner/repo"
    default_branch: Mapped[str] = mapped_column(String(128), default="main")
    private: Mapped[bool] = mapped_column(Boolean, default=False)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))  # RAG ingest
