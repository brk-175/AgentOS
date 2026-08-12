"""Shared API dependencies: session-cookie authentication.

``get_authenticated_user`` resolves the signed session cookie into the user
row plus their active GitHub ``OAuthConnection``, decrypting the stored token
so downstream services (repos, agent runs) can call GitHub on the user's
behalf.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from agentos.core.security import TokenCipherError, get_session_tokens, get_token_cipher
from agentos.db.session import get_db
from agentos.models.oauth_connection import OAuthConnection
from agentos.models.user import User

SESSION_COOKIE = "agentos_session"

DbSession = Annotated[AsyncSession, Depends(get_db)]


@dataclass(frozen=True)
class AuthenticatedUser:
    """The logged-in user plus their decrypted GitHub access token."""

    user: User
    connection: OAuthConnection
    access_token: str


async def get_authenticated_user(request: Request, db: DbSession) -> AuthenticatedUser:
    """Resolve the session cookie to a user + decrypted GitHub token (or 401)."""
    token = request.cookies.get(SESSION_COOKIE)
    user_id = get_session_tokens().verify(token) if token else None
    if user_id is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=401, detail="Not authenticated")
    connection = await db.scalar(
        select(OAuthConnection).where(
            OAuthConnection.user_id == user.id,
            OAuthConnection.provider == "github",
        )
    )
    if connection is None:
        raise HTTPException(status_code=401, detail="GitHub account not connected")
    try:
        access_token = get_token_cipher().decrypt_token(connection.access_token_encrypted)
    except TokenCipherError as exc:
        raise HTTPException(status_code=401, detail="Stored token is invalid") from exc
    return AuthenticatedUser(user=user, connection=connection, access_token=access_token)


AuthContext = Annotated[AuthenticatedUser, Depends(get_authenticated_user)]
