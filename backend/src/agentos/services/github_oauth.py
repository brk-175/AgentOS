"""GitHub OAuth web-flow client: authorize URL, token exchange, profile fetch.

Speaks to github.com directly (no SDK); all secrets (client id/secret) are
passed in from ``Settings`` so this module stays side-effect free.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlencode

import httpx

from agentos.core.logging import get_logger

logger = get_logger(__name__)

GITHUB_AUTHORIZE_URL = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN_URL = "https://github.com/login/oauth/access_token"
GITHUB_API_URL = "https://api.github.com"
GITHUB_USER_PATH = "/user"

HTTP_TIMEOUT_SECONDS = 15.0


class GitHubOAuthError(Exception):
    """Raised when GitHub rejects an exchange or API call."""


@dataclass(frozen=True)
class OAuthToken:
    """Normalised GitHub OAuth token response."""

    access_token: str
    token_type: str = "Bearer"
    scope: str = ""
    expires_in: int | None = None


def build_authorize_url(*, client_id: str, redirect_uri: str, state: str, scope: str) -> str:
    """Build the GitHub login/oauth/authorize URL for the web flow."""
    params = urlencode(
        {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "scope": scope,
            "allow_signup": "true",
        }
    )
    return f"{GITHUB_AUTHORIZE_URL}?{params}"


async def exchange_code(
    *, client_id: str, client_secret: str, code: str, redirect_uri: str
) -> OAuthToken:
    """Exchange an authorization code for an access token."""
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.post(
            GITHUB_TOKEN_URL,
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": redirect_uri,
            },
            headers={"Accept": "application/json"},
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise GitHubOAuthError("GitHub returned a non-JSON token response") from exc
    if "access_token" not in payload:
        reason = payload.get("error_description") or payload.get("error") or "unknown error"
        raise GitHubOAuthError(f"token exchange failed: {reason}")
    return OAuthToken(
        access_token=payload["access_token"],
        token_type=payload.get("token_type", "Bearer"),
        scope=payload.get("scope", ""),
        expires_in=payload.get("expires_in"),
    )


async def fetch_user_profile(access_token: str) -> dict:
    """Fetch the authenticated user's profile from the GitHub API."""
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "AgentOS",
    }
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(f"{GITHUB_API_URL}{GITHUB_USER_PATH}", headers=headers)
    if response.status_code != 200:
        raise GitHubOAuthError(f"profile fetch failed: HTTP {response.status_code}")
    try:
        return response.json()
    except ValueError as exc:
        raise GitHubOAuthError("GitHub returned a non-JSON profile") from exc
