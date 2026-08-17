"""Tests for the RAG service: indexing + semantic search (sqlite path)."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from agentos.models.repository_document import RepositoryDocument
from agentos.services.rag import (
    ContentFile,
    build_retriever,
    index_repository,
    search_repository,
)

DIMENSIONS = 1536

DOC_VECTORS: dict[str, list[float]] = {
    "crash on empty input null pointer bug": [1.0, 0.0, 0.0],
    "setup install configure build servers": [0.0, 1.0, 0.0],
    "hello world agent": [0.0, 0.0, 1.0],
    "zzzz qqqq xxxx": [1.0, 1.0, 1.0],
}
QUERY_VECTORS: dict[str, list[float]] = {
    "crash null": [1.0, 0.0, 0.0],
    "crash bug": [5.0, 0.0, 5.0],
}


class FakeEmbeddings:
    """Deterministic scripted embeddings (no network, no randomness)."""

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._pad(DOC_VECTORS.get(text, [0.0, 0.0, 0.0])) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return self._pad(QUERY_VECTORS.get(text, [0.0, 0.0, 0.0]))

    @staticmethod
    def _pad(vector: list[float]) -> list[float]:
        return vector + [0.0] * (DIMENSIONS - len(vector))


async def _count_documents(session: AsyncSession, repo: str) -> int:
    return int(
        await session.scalar(
            select(func.count())
            .select_from(RepositoryDocument)
            .where(RepositoryDocument.repo_full_name == repo)
        )
    )


async def test_index_stores_chunks_and_replaces_previous_index(
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    files = [
        ContentFile(path="README.md", content="AgentOS fixes GitHub issues fast."),
        ContentFile(
            path="src/entrypoint.py", content="def main():\n    pass\n" + "# " + "x" * 1600
        ),
    ]
    async with db_factory() as session:
        summary = await index_repository(
            session,
            "tok",
            "octocat/AgentOS",
            embeddings=FakeEmbeddings(),
            files=files,
        )
        assert summary.files_indexed == 2
        assert summary.chunks == 3  # README (1) + long entrypoint (2)
        assert summary.chars > 0

        rows = (await session.scalars(select(RepositoryDocument))).all()
        assert len(rows) == 3
        assert {row.path for row in rows} == {"README.md", "src/entrypoint.py"}
        assert all(len(row.embedding) == DIMENSIONS for row in rows)
        entry_chunks = {row.chunk_index for row in rows if row.path == "src/entrypoint.py"}
        assert entry_chunks == {0, 1}

        await index_repository(
            session,
            "tok",
            "octocat/AgentOS",
            embeddings=FakeEmbeddings(),
            files=[ContentFile(path="src/entrypoint.py", content="def main():\n    pass\n")],
        )
        assert await _count_documents(session, "octocat/AgentOS") == 1


async def test_search_ranks_most_similar_chunk_first(
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    files = [
        ContentFile(path="docs/install.md", content="setup install configure build servers"),
        ContentFile(path="src/crash.py", content="crash on empty input null pointer bug"),
        ContentFile(path="README.md", content="hello world agent"),
    ]
    async with db_factory() as session:
        await index_repository(
            session,
            "tok",
            "octocat/demo",
            embeddings=FakeEmbeddings(),
            files=files,
        )
        hits = await search_repository(
            session, "octocat/demo", "crash null", embeddings=FakeEmbeddings(), top_k=2
        )
    assert hits[0].path == "src/crash.py"
    assert hits[0].score == pytest.approx(1.0)
    rank = {hit.path: hit.score for hit in hits}
    assert rank["src/crash.py"] >= rank.get("docs/install.md", 0)


async def test_search_threshold_filters_unrelated_chunks(
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with db_factory() as session:
        await index_repository(
            session,
            "tok",
            "octocat/demo",
            embeddings=FakeEmbeddings(),
            files=[ContentFile(path="a.txt", content="zzzz qqqq xxxx")],
        )
        hits = await search_repository(
            session,
            "octocat/demo",
            "crash bug",
            embeddings=FakeEmbeddings(),
            top_k=5,
            threshold=0.99,
        )
    assert hits == []


async def test_search_on_empty_index_returns_nothing(
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with db_factory() as session:
        hits = await search_repository(
            session, "octocat/demo", "anything", embeddings=FakeEmbeddings()
        )
    assert hits == []


async def test_build_retriever_maps_hits_to_context_docs(
    db_engine: AsyncEngine,
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    async with db_factory() as session:
        await index_repository(
            session,
            "tok",
            "octocat/demo",
            embeddings=FakeEmbeddings(),
            files=[
                ContentFile(path="src/crash.py", content="crash on empty input null pointer bug")
            ],
        )
    retrieve = build_retriever(db_engine, embeddings=FakeEmbeddings())
    docs = await retrieve("octocat/demo", "crash null")
    assert len(docs) == 1
    assert docs[0].path == "src/crash.py"
    assert docs[0].content == "crash on empty input null pointer bug"
    assert docs[0].chunk_index == 0
    assert docs[0].score == pytest.approx(1.0)
    assert await retrieve("octocat/demo", "zzz nothing indexed") == []
