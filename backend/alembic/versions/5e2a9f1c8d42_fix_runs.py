"""fix_runs: durable run history + judge evaluation

Revision ID: 5e2a9f1c8d42
Revises: c91f2b4a0d17
Create Date: 2026-08-22 12:00:00.000000

"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "5e2a9f1c8d42"
down_revision: str | None = "c91f2b4a0d17"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "fix_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("run_id", sa.String(length=32), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("repo_full_name", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("number", sa.Integer(), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("base_branch", sa.String(length=128), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("applied_branch", sa.String(length=255), nullable=True),
        sa.Column("pr_url", sa.String(length=512), nullable=True),
        sa.Column("investigation", sa.Text(), nullable=False),
        sa.Column("root_cause_hypothesis", sa.Text(), nullable=False),
        sa.Column("proposed_changes", sa.JSON(), nullable=False),
        sa.Column("evaluation", sa.JSON(), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index(op.f("ix_fix_runs_run_id"), "fix_runs", ["run_id"], unique=True)
    op.create_index(op.f("ix_fix_runs_user_id"), "fix_runs", ["user_id"], unique=False)
    op.create_index(
        op.f("ix_fix_runs_repo_full_name"), "fix_runs", ["repo_full_name"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_fix_runs_repo_full_name"), table_name="fix_runs")
    op.drop_index(op.f("ix_fix_runs_user_id"), table_name="fix_runs")
    op.drop_index(op.f("ix_fix_runs_run_id"), table_name="fix_runs")
    op.drop_table("fix_runs")
