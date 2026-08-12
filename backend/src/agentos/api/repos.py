"""Repository endpoints: live GitHub data for the authenticated user.

Listing is fetched on demand from GitHub using the user's decrypted token —
no local cache yet. The ``Repository`` row becomes relevant when RAG
indexing lands (Stage 4).
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from agentos.api.deps import AuthContext
from agentos.core.logging import get_logger
from agentos.services import github

logger = get_logger(__name__)

router = APIRouter(prefix="/repos", tags=["repos"])


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
