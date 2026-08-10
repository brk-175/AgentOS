"""User accounts (one per GitHub identity)."""

from __future__ import annotations

import uuid

from sqlalchemy import BigInteger, String
from sqlalchemy.orm import Mapped, mapped_column

from agentos.models.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    """A registered AgentOS user, identified by their GitHub account."""

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    github_id: Mapped[int] = mapped_column(BigInteger, unique=True)
    username: Mapped[str] = mapped_column(String(64), index=True)
    name: Mapped[str | None] = mapped_column(String(255))
    email: Mapped[str | None] = mapped_column(String(255))
    avatar_url: Mapped[str | None] = mapped_column(String(512))
