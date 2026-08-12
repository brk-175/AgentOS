"""GitHub REST API client: repository listing and metadata.

Operates with a raw access token (decrypted from the user's
``OAuthConnection``). All failures surface as ``GitHubClientError`` with a
stable ``status`` attribute so callers can branch on the cause (401 → token
invalid/revoked, 403 → rate limited/forbidden, 404 → not found).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

from agentos.core.logging import get_logger

logger = get_logger(__name__)

GITHUB_API_URL = "https://api.github.com"
HTTP_TIMEOUT_SECONDS = 15.0
PER_PAGE = 100
MAX_PAGES = 10

_NEXT_LINK_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


class GitHubClientError(Exception):
    """GitHub API failure; ``status`` holds the HTTP status (may be None)."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class RepositorySummary:
    """Public-facing repository metadata used across API + agent surfaces."""

    full_name: str
    private: bool
    default_branch: str
    description: str | None
    updated_at: str
    html_url: str


def _headers(access_token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "AgentOS",
    }


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 401:
        raise GitHubClientError("GitHub token is invalid or revoked", status=401)
    if response.status_code == 403:
        raise GitHubClientError("GitHub API rate limit exceeded or access forbidden", status=403)
    if response.status_code == 404:
        raise GitHubClientError("Repository not found or inaccessible", status=404)
    if response.status_code >= 400:
        raise GitHubClientError(
            f"GitHub API error: HTTP {response.status_code}", status=response.status_code
        )


def _parse_repo(payload: dict) -> RepositorySummary:
    return RepositorySummary(
        full_name=payload["full_name"],
        private=payload["private"],
        default_branch=payload.get("default_branch") or "main",
        description=payload.get("description"),
        updated_at=payload.get("updated_at") or "",
        html_url=payload.get("html_url") or f"https://github.com/{payload['full_name']}",
    )


async def list_repositories(access_token: str) -> list[RepositorySummary]:
    """Return repos the token can access, newest-updated first (paginated)."""
    headers = _headers(access_token)
    repos: list[RepositorySummary] = []
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        url = f"{GITHUB_API_URL}/user/repos?sort=updated&per_page={PER_PAGE}"
        for _ in range(MAX_PAGES):
            response = await client.get(url, headers=headers)
            _raise_for_status(response)
            repos.extend(_parse_repo(item) for item in response.json())
            next_match = _NEXT_LINK_RE.search(response.headers.get("link", ""))
            if next_match is None:
                break
            url = next_match.group(1)
    return repos


async def get_repository(access_token: str, full_name: str) -> RepositorySummary:
    """Return metadata for a single repository (``owner/name``)."""
    headers = _headers(access_token)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{GITHUB_API_URL}/repos/{full_name}", headers=headers)
    _raise_for_status(response)
    return _parse_repo(response.json())
