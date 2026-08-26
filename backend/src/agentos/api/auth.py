"""GitHub OAuth endpoints: login redirect, callback, session, me, logout.

The callback exchanges the GitHub code, upserts the user + encrypted token
connection, and issues the HMAC-signed session cookie (``SessionTokens``).
The OAuth ``state`` is delivered via a short-lived HttpOnly cookie and
compared with constant-time equality to close CSRF/login CSRF holes.
"""

from __future__ import annotations

import hmac
import secrets
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy import select

from agentos.api.deps import SESSION_COOKIE, AuthContext, DbSession
from agentos.core.config import get_settings
from agentos.core.logging import get_logger
from agentos.core.security import SESSION_TTL_SECONDS, get_session_tokens, get_token_cipher
from agentos.models.oauth_connection import OAuthConnection
from agentos.models.user import User
from agentos.services import github_oauth

logger = get_logger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

STATE_COOKIE = "agentos_oauth_state"
STATE_TTL_SECONDS = 600
_GITHUB_PROVIDER = "github"


def _redirect_with_error(error: str) -> RedirectResponse:
    response = RedirectResponse(
        f"{get_settings().frontend_url}/?auth_error={error}", status_code=302
    )
    response.delete_cookie(STATE_COOKIE, path="/")
    return response


@router.get("/github/login")
async def github_login() -> RedirectResponse:
    """Start the GitHub OAuth web flow (302 to github.com)."""
    settings = get_settings()
    state = secrets.token_urlsafe(32)
    url = github_oauth.build_authorize_url(
        client_id=settings.github_client_id,
        redirect_uri=settings.github_oauth_callback_url,
        state=state,
        scope=settings.github_scope,
    )
    response = RedirectResponse(url, status_code=302)
    response.set_cookie(
        STATE_COOKIE,
        state,
        max_age=STATE_TTL_SECONDS,
        httponly=True,
        samesite="none" if settings.cookie_secure else "lax",
        secure=settings.cookie_secure,
        path="/",
    )
    return response


@router.get("/github/callback")
async def github_callback(
    request: Request,
    db: DbSession,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> RedirectResponse:
    """Complete the web flow: verify state, exchange code, upsert user, log in."""
    settings = get_settings()
    stored_state = request.cookies.get(STATE_COOKIE)
    if not code or error:
        return _redirect_with_error(error or "oauth_failed")
    if not state or not stored_state or not hmac.compare_digest(state, stored_state):
        return _redirect_with_error("invalid_state")

    try:
        token = await github_oauth.exchange_code(
            client_id=settings.github_client_id,
            client_secret=settings.github_client_secret,
            code=code,
            redirect_uri=settings.github_oauth_callback_url,
        )
        profile = await github_oauth.fetch_user_profile(token.access_token)
    except github_oauth.GitHubOAuthError as exc:
        logger.warning("github oauth failure", extra={"extra_fields": {"reason": str(exc)}})
        return _redirect_with_error("github_error")

    user = await db.scalar(select(User).where(User.github_id == profile["id"]))
    if user is None:
        user = User(
            github_id=profile["id"],
            username=profile["login"],
            name=profile.get("name"),
            email=profile.get("email"),
            avatar_url=profile.get("avatar_url"),
        )
        db.add(user)
        await db.flush()

    connection = await db.scalar(
        select(OAuthConnection).where(
            OAuthConnection.user_id == user.id,
            OAuthConnection.provider == _GITHUB_PROVIDER,
        )
    )
    expires_at = (
        datetime.now(UTC) + timedelta(seconds=token.expires_in) if token.expires_in else None
    )
    if connection is None:
        connection = OAuthConnection(user_id=user.id, provider=_GITHUB_PROVIDER)
        db.add(connection)
    connection.access_token_encrypted = get_token_cipher().encrypt_token(token.access_token)
    connection.token_type = token.token_type
    connection.scope = token.scope
    connection.expires_at = expires_at
    await db.commit()

    session_token = get_session_tokens().create(user.id)
    response = RedirectResponse(f"{settings.frontend_url}/dashboard", status_code=302)
    response.delete_cookie(STATE_COOKIE, path="/")
    response.set_cookie(
        SESSION_COOKIE,
        session_token,
        max_age=SESSION_TTL_SECONDS,
        httponly=True,
        samesite="none" if settings.cookie_secure else "lax",
        secure=settings.cookie_secure,
        path="/",
    )
    return response


@router.get("/me")
async def me(auth: AuthContext) -> dict:
    """Return the current user from the session cookie, or 401."""
    user = auth.user
    return {
        "id": str(user.id),
        "github_id": user.github_id,
        "username": user.username,
        "name": user.name,
        "email": user.email,
        "avatar_url": user.avatar_url,
    }


@router.post("/logout")
async def logout(response: Response) -> Response:
    """Clear the session cookie."""
    response = Response(status_code=204)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response
