"""RAG: index repository files into pgvector and search them semantically.

``index_repository`` incrementally syncs the stored chunks for a repo:
chunks whose content is byte-identical keep their existing embedding (no
embedding API call), only new/changed chunks are embedded, and rows for
files that disappeared are removed. Files are read either from an injected
source (tests) or by walking the repo through the GitHub MCP tools (bounded
by depth and per-file/whole-repo caps). ``search_repository`` returns the
top-k chunks by cosine similarity — via the pgvector ``<=>`` operator on
Postgres, via an in-Python scan on other dialects (sqlite tests).
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt
from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from agentos.agent.mcp_adapter import GitHubMCPTools
from agentos.agent.state import ContextDoc, Retriever
from agentos.core.logging import get_logger
from agentos.db.session import build_session_factory
from agentos.models.repository_document import RepositoryDocument
from agentos.services.embeddings import chunk_text, create_embeddings_client

logger = get_logger(__name__)

MAX_INDEX_FILES = 100
MAX_INDEX_FILE_CHARS = 40_000
MAX_DEPTH = 10  # depth 0 = repo root; each index adds one level


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
    chunks_embedded: int = 0


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
            entries = listing["items"] if isinstance(listing, dict) else listing
            paths: list[str] = []
            for entry in entries:
                kind = entry.get("kind")
                if kind == "dir" and depth > 0:
                    paths.extend(await walk(entry["path"], depth - 1))
                elif kind == "file" and entry.get("size", 0) <= max_chars:
                    paths.append(entry["path"])
            return paths

        files: list[ContentFile] = []
        for path in (await walk("", MAX_DEPTH))[:max_files]:
            content = await read_tool.ainvoke({"owner": owner, "name": repo, "path": path})
            if "\x00" in content[:4096]:
                continue  # binary file (NUL bytes) — not text, not indexable
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
    """(Re)build the chunk index for a repository — incrementally.

    Chunks whose ``content`` is byte-identical to an already-stored row keep
    their existing embedding (no embedding API call). Only new chunks and
    chunks whose text changed are embedded. Rows for files/chunks that are no
    longer present are removed. Re-indexing an unchanged repo therefore costs
    zero embedding tokens; the fetched-file/count columns report the full
    repo, ``chunks_embedded`` reports what actually hit the embedder.
    """
    source = (
        files
        if files is not None
        else await _fetch_repo_files(
            access_token, repo_full_name, max_files=max_files, max_chars=max_chars
        )
    )

    chunks: list[tuple[str, int, str]] = []
    chars = 0
    for file in source:
        content = file.content.replace("\x00", "")  # Postgres cannot store NUL bytes
        for index, piece in enumerate(chunk_text(content)):
            chunks.append((file.path, index, piece))
            chars += len(piece)

    target_keys = {(path, index) for path, index, _ in chunks}
    existing = {
        (row.path, row.chunk_index): row
        for row in (
            await db.scalars(
                select(RepositoryDocument).where(
                    RepositoryDocument.repo_full_name == repo_full_name
                )
            )
        ).all()
    }

    stale = [
        row for (path, index), row in existing.items() if (path, index) not in target_keys
    ]
    for row in stale:
        await db.delete(row)

    to_embed: list[tuple[str, int, str]] = []
    for path, index, text in chunks:
        existing_row = existing.get((path, index))
        if existing_row is not None and existing_row.content == text:
            continue  # unchanged — reuse the stored embedding
        to_embed.append((path, index, text))

    if to_embed:
        embedder = embeddings if embeddings is not None else create_embeddings_client()
        vectors = await embedder.aembed_documents([text for _, _, text in to_embed])
        new_rows: list[RepositoryDocument] = []
        for (path, index, text), vector in zip(to_embed, vectors, strict=True):
            existing_row = existing.get((path, index))
            if existing_row is not None:
                existing_row.content = text
                existing_row.embedding = list(vector)
            else:
                new_rows.append(
                    RepositoryDocument(
                        repo_full_name=repo_full_name,
                        path=path,
                        chunk_index=index,
                        content=text,
                        embedding=list(vector),
                    )
                )
        db.add_all(new_rows)

    await db.commit()
    logger.info(
        "indexed %s: %d files, %d chunks (%d new/changed embedded, %d reused, %d stale removed)",
        repo_full_name,
        len(source),
        len(chunks),
        len(to_embed),
        len(chunks) - len(to_embed),
        len(stale),
    )
    return IndexSummary(
        repo_full_name=repo_full_name,
        files_indexed=len(source),
        chunks=len(chunks),
        chars=chars,
        chunks_embedded=len(to_embed),
    )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


_KEYWORD_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by", "can", "could", "did", "do", "does", "done", "for", "from", "had", "has", "have", "he", "her", "here", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just", "like", "made", "make", "may", "me", "mine", "more", "most", "must", "my", "no", "nor", "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "upon", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "would", "you", "your", "yours", "itself", "myself", "ourselves", "themselves", "ourselves"]
)


def significant_keywords(text: str, *, max_words: int = 8) -> list[str]:
    """Extract the most content-bearing words from a query/issue text.

    Words are lowercased, stripped of non-alphanumerics and common stopwords,
    and capped at ``max_words``. Used as the literal-content fallback for
    retrieval, so files whose *content* mentions an issue's terms surface
    even when their embedding ranks below the similarity threshold.
    """
    tokens = [w for w in re.findall(r"[a-z0-9]{4,}", text.lower()) if w not in _KEYWORD_STOPWORDS]
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], -len(pair[0])))
    return [word for word, _ in ranked[:max_words]]


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


def build_retriever(
    engine: AsyncEngine,
    *,
    top_k: int = 5,  # matches the agent's MAX_RAG_CONTEXT
    threshold: float = 0.25,
    embeddings: Any | None = None,
) -> Retriever:
    """Bind the RAG store to the agent's retrieval interface.

    The returned callable performs a semantic search on ``repo_full_name``
    and maps the hits onto ``ContextDoc`` entries (path, content, chunk,
    score) that ``investigate`` merges into the agent context.
    """
    factory = build_session_factory(engine)

    async def retrieve(repo_full_name: str, query: str, limit: int = top_k) -> list[ContextDoc]:
        async with factory() as session:
            hits = await search_repository(
                session,
                repo_full_name,
                query,
                embeddings=embeddings,
                top_k=limit,
                threshold=threshold,
            )
            docs = [
                ContextDoc(
                    path=hit.path,
                    content=hit.content,
                    chunk_index=hit.chunk_index,
                    score=hit.score,
                )
                for hit in hits
            ]
            seen = {(doc.path, doc.chunk_index) for doc in docs}
            extendable = limit - len(docs)
            keywords = significant_keywords(query)
            if extendable > 0 and keywords:
                clauses = [
                    RepositoryDocument.content.ilike(f"%{keyword}%") for keyword in keywords
                ]
                rows = (
                    await session.scalars(
                        select(RepositoryDocument).where(
                            RepositoryDocument.repo_full_name == repo_full_name,
                            or_(*clauses),
                        )
                    )
                ).all()
                for row in rows:
                    if (row.path, row.chunk_index) in seen:
                        continue
                    seen.add((row.path, row.chunk_index))
                    # literal match — no cosine score; rendered without provenance
                    docs.append(
                        ContextDoc(
                            path=row.path,
                            content=row.content,
                            chunk_index=row.chunk_index,
                            score=None,
                        )
                    )
                    if len(docs) >= limit:
                        break
        return docs

    return retrieve
