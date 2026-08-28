"""Repository endpoints: live GitHub data for the authenticated user.

Listing is fetched on demand from GitHub using the user's decrypted token —
no local cache yet. The ``Repository`` row becomes relevant when RAG
indexing lands (Stage 4).
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from agentos.api.deps import AuthContext, DbSession
from agentos.core.logging import get_logger
from agentos.services import github, rag
from agentos.services.embeddings import create_embeddings_client

logger = get_logger(__name__)

router = APIRouter(prefix="/repos", tags=["repos"])


def get_embeddings() -> Any:
    """Dependency exposing the OpenRouter embeddings client (overridable)."""
    return create_embeddings_client()


EmbeddingsDep = Annotated[Any, Depends(get_embeddings)]


class RepositoryOut(BaseModel):
    """Public repository metadata (mirrors ``github.RepositorySummary``)."""

    full_name: str
    private: bool
    default_branch: str
    description: str | None
    updated_at: str
    html_url: str


def _map_error(exc: github.GitHubClientError) -> HTTPException:
    if exc.status == 401:
        return HTTPException(
            status_code=401,
            detail="GitHub token is invalid or revoked; reconnect your GitHub account",
        )
    if exc.status == 403:
        return HTTPException(
            status_code=429, detail="GitHub API rate limit exceeded; try again later"
        )
    if exc.status == 404:
        return HTTPException(status_code=404, detail="Repository not found or inaccessible")
    return HTTPException(status_code=502, detail=f"GitHub API error: {exc}")


@router.get("", response_model=list[RepositoryOut])
async def list_repos(auth: AuthContext) -> list[RepositoryOut]:
    """List repositories the user's GitHub token can access."""
    try:
        repos = await github.list_repositories(auth.access_token)
    except github.GitHubClientError as exc:
        raise _map_error(exc) from exc
    return [RepositoryOut.model_validate(asdict(repo)) for repo in repos]


@router.get("/fetch", response_model=RepositoryOut)
async def fetch_repo(
    auth: AuthContext,
    full_name: str = Query(min_length=3, max_length=255, description="Repository as owner/name"),
) -> RepositoryOut:
    """Fetch fresh metadata for a single repository."""
    try:
        repo = await github.get_repository(auth.access_token, full_name)
    except github.GitHubClientError as exc:
        raise _map_error(exc) from exc
    return RepositoryOut.model_validate(asdict(repo))


class TargetOut(BaseModel):
    """One issue or pull request for the new-run picker."""

    kind: Literal["issue", "pr"]
    number: int
    title: str
    state: str
    created_at: str
    updated_at: str
    merged_at: str | None = None


@router.get("/{full_name:path}/targets", response_model=list[TargetOut])
async def list_targets(
    auth: AuthContext,
    full_name: str,
    kind: Literal["issue", "pr"] = Query(default="issue", description="Target kind"),
) -> list[TargetOut]:
    """List the repo's issues or pull requests, newest first (paginated)."""
    try:
        targets = await github.list_issue_pulls(auth.access_token, full_name, kind)
    except github.GitHubClientError as exc:
        raise _map_error(exc) from exc
    return [TargetOut.model_validate(asdict(target)) for target in targets]


class IndexOut(BaseModel):
    """What the RAG index stored for a repository."""

    full_name: str
    files_indexed: int
    chunks: int
    chars: int


class SearchHitOut(BaseModel):
    path: str
    chunk_index: int
    content: str
    score: float


class SearchOut(BaseModel):
    query: str
    hits: list[SearchHitOut]


@router.post("/{full_name:path}/index", response_model=IndexOut)
async def index_repo(
    auth: AuthContext,
    db: DbSession,
    full_name: str,
    embeddings: EmbeddingsDep,
) -> IndexOut:
    """Index a repository's files into the vector store (idempotent)."""
    try:
        await github.get_repository(auth.access_token, full_name)
    except github.GitHubClientError as exc:
        raise _map_error(exc) from exc
    try:
        summary = await rag.index_repository(
            db, auth.access_token, full_name, embeddings=embeddings
        )
    except Exception as exc:  # noqa: BLE001 - MCP/GitHub failures map to 502
        logger.warning("indexing %s failed: %s", full_name, exc)
        raise HTTPException(status_code=502, detail=f"Indexing failed: {exc}") from exc
    return IndexOut(
        full_name=summary.repo_full_name,
        files_indexed=summary.files_indexed,
        chunks=summary.chunks,
        chars=summary.chars,
    )


@router.get("/{full_name:path}/search", response_model=SearchOut)
async def search_repo(
    auth: AuthContext,
    db: DbSession,
    full_name: str,
    embeddings: EmbeddingsDep,
    q: str = Query(min_length=2, max_length=512, description="Search query"),
    top_k: int = Query(default=5, ge=1, le=20),
) -> SearchOut:
    """Semantic search over an indexed repository (vector similarity)."""
    hits = await rag.search_repository(db, full_name, q, embeddings=embeddings, top_k=top_k)
    return SearchOut(
        query=q,
        hits=[
            SearchHitOut(
                path=hit.path,
                chunk_index=hit.chunk_index,
                content=hit.content,
                score=round(hit.score, 4),
            )
            for hit in hits
        ],
    )
