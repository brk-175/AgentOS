"""rag: repository_documents (embedding chunks per indexed file)

Revision ID: c91f2b4a0d17
Revises: bb97a7ad9d2d
Create Date: 2026-08-15 22:10:00.000000

"""
from __future__ import annotations

from collections.abc import Sequence

import pgvector.sqlalchemy
import sqlalchemy as sa

from alembic import op

revision: str = "c91f2b4a0d17"
down_revision: str | None = "bb97a7ad9d2d"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "repository_documents",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("repo_full_name", sa.String(length=255), nullable=False),
        sa.Column("path", sa.String(length=1024), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(dim=1536), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "repo_full_name", "path", "chunk_index", name="uq_repository_documents_repo_path_chunk"
        ),
    )
    op.create_index(
        op.f("ix_repository_documents_repo_full_name"),
        "repository_documents",
        ["repo_full_name"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_repository_documents_repo_full_name"), table_name="repository_documents"
    )
    op.drop_table("repository_documents")