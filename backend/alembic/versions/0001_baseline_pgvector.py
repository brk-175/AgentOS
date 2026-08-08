"""baseline: enable pgvector extension

Revision ID: 00000001
Revises:
Create Date: 2026-08-08

"""
from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "00000001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector")