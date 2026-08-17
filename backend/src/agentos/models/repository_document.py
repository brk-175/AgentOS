"""Embedded repository files: one row per text chunk of an indexed file.

The ``embedding`` column is a pgvector ``vector`` on Postgres; the sqlite
variant keeps the JSON representation so unit tests (in-memory sqlite) can
exercise the same model without a pgvector database. Row storage is portable;
only the *similarity operator* in ``services/rag`` differs per dialect.
"""

from __future__ import annotations

import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from agentos.core.config import get_settings
from agentos.models.base import Base, TimestampMixin

_EMBEDDING_DIMENSIONS = get_settings().embeddings_dimensions
EmbeddingVector = Vector(_EMBEDDING_DIMENSIONS).with_variant(JSON(), "sqlite")


class RepositoryDocument(Base, TimestampMixin):
    """A chunk of a repo file, embedded for semantic retrieval."""

    __tablename__ = "repository_documents"
    __table_args__ = (
        UniqueConstraint(
            "repo_full_name", "path", "chunk_index", name="uq_repository_documents_repo_path_chunk"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid.uuid4)
    repo_full_name: Mapped[str] = mapped_column(String(255), index=True)
    path: Mapped[str] = mapped_column(String(1024))
    chunk_index: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(EmbeddingVector, nullable=False)
