"""Declarative base and shared column types for all ORM models.

``Base`` is the single metadata registry Alembic autogenerates from. Keep
model definitions out of gathering imports so ``env.py`` can import it
without side effects.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for AgentOS tables."""


class TimestampMixin:
    """Adds ``created_at`` / ``updated_at`` (DB-managed) to a model."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )
