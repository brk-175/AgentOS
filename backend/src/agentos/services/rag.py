"""RAG: index repository files into pgvector and search them semantically.

``index_repository`` replaces the stored chunks for a repo (idempotent),
reading files either from an injected source (tests) or by walking the repo
through the GitHub MCP tools (bounded: depth 2, ≤ ``max_files`` files, each
≤ ``max_chars``). ``search_repository`` returns the top-k chunks by cosine
similarity — via the pgvector ``<=>`` operator on Postgres, via an in-Python
scan on other dialects (sqlite tests).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from agentos.agent.mcp_adapter import GitHubMCPTools
from agentos.core.logging import get_logger
from agentos.models.repository_document import RepositoryDocument
from agentos.services.embeddings import chunk_text, create_embeddings_client

logger = get_logger(__name__)

MAX_INDEX_FILES = 100
MAX_INDEX_FILE_CHARS = 10_000
MAX_DEPTH = 1  # depth 0 = repo root; each index adds one level


@dataclass(frozen=True)
class ContentFile:
    """A file to index: path + raw content."""

    path: str
    content: str


@dataclass(frozen=True)
class IndexSummary:
    """What ``index_repository`` stored."""

    repo_full_name: str
    files_indexed: int
    chunks: int
    chars: int


@dataclass(frozen=True)
class SearchHit:
    """A ranked chunk matching a query."""

    path: str
    chunk_index: int
    content: str
    score: float


def _find_tool(tools: Sequence[Any], name: str) -> Any | None:
    return next((tool for tool in tools if getattr(tool, "name", None) == name), None)


async def _fetch_repo_files(
    access_token: str,
    repo_full_name: str,
    *,
    max_files: int,
    max_chars: int,
) -> list[ContentFile]:
    """Walk the repo root via MCP and read the promising files (bounded)."""
    owner, _, repo = repo_full_name.partition("/")
    async with GitHubMCPTools(token=access_token) as adapter:
        listing_tool = _find_tool(adapter.tools, "list_repo_files")
        read_tool = _find_tool(adapter.tools, "read_file")
        if listing_tool is None or read_tool is None:
            raise RuntimeError("MCP tools for indexing are unavailable")

        async def walk(path: str, depth: int) -> list[str]:
            listing = json.loads(
                await listing_tool.ainvoke({"owner": owner, "name": repo, "path": path})
            )
            paths: list[str] = []
            for entry in listing:
                kind = entry.get("kind")
                if kind == "dir" and depth > 0:
                    paths.extend(await walk(entry["path"], depth - 1))
                elif kind == "file" and entry.get("size", 0) <= max_chars:
                    paths.append(entry["path"])
            return paths

        files: list[ContentFile] = []
        for path in (await walk("", MAX_DEPTH))[:max_files]:
            content = await read_tool.ainvoke({"owner": owner, "name": repo, "path": path})
            files.append(ContentFile(path=path, content=content))
        return files


async def index_repository(
    db: AsyncSession,
    access_token: str,
    repo_full_name: str,
    *,
    embeddings: Any | None = None,
    files: Sequence[ContentFile] | None = None,
    max_files: int = MAX_INDEX_FILES,
    max_chars: int = MAX_INDEX_FILE_CHARS,
) -> IndexSummary:
    """(Re)build the chunk index for a repository; replaces existing rows."""
    source = (
        files
        if files is not None
        else await _fetch_repo_files(
            access_token, repo_full_name, max_files=max_files, max_chars=max_chars
        )
    )
    if source:
        embedder = embeddings if embeddings is not None else create_embeddings_client()

    chunks: list[tuple[str, int, str]] = []
    chars = 0
    for file in source:
        for index, piece in enumerate(chunk_text(file.content)):
            chunks.append((file.path, index, piece))
            chars += len(piece)

    await db.execute(
        delete(RepositoryDocument).where(RepositoryDocument.repo_full_name == repo_full_name)
    )
    if chunks:
        vectors = await embedder.aembed_documents([text for _, _, text in chunks])
        db.add_all(
            [
                RepositoryDocument(
                    repo_full_name=repo_full_name,
                    path=path,
                    chunk_index=index,
                    content=text,
                    embedding=list(vector),
                )
                for (path, index, text), vector in zip(chunks, vectors, strict=True)
            ]
        )
    await db.commit()
    logger.info("indexed %s: %d files, %d chunks", repo_full_name, len(source), len(chunks))
    return IndexSummary(
        repo_full_name=repo_full_name,
        files_indexed=len(source),
        chunks=len(chunks),
        chars=chars,
    )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def search_repository(
    db: AsyncSession,
    repo_full_name: str,
    query: str,
    *,
    embeddings: Any | None = None,
    top_k: int = 5,
    threshold: float = 0.25,
) -> list[SearchHit]:
    """Return the top-``top_k`` chunks most similar to ``query`` (≥ score)."""
    embedder = embeddings if embeddings is not None else create_embeddings_client()
    query_vector = await embedder.aembed_query(query)
    stmt = select(RepositoryDocument).where(RepositoryDocument.repo_full_name == repo_full_name)
    dialect = getattr(db.bind, "dialect", None)
    if dialect is not None and dialect.name == "postgresql":
        rows = (
            await db.scalars(
                stmt.order_by(
                    RepositoryDocument.embedding.cosine_distance(list(query_vector))
                ).limit(top_k * 4)
            )
        ).all()
    else:
        rows = (await db.scalars(stmt)).all()
        rows = sorted(rows, key=lambda row: _cosine(row.embedding, query_vector), reverse=True)

    hits: list[SearchHit] = []
    for row in rows:
        score = _cosine(row.embedding, query_vector)
        if score >= threshold:
            hits.append(
                SearchHit(
                    path=row.path, chunk_index=row.chunk_index, content=row.content, score=score
                )
            )
        if len(hits) == top_k:
            break
    return hits
