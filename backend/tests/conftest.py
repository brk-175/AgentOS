"""Shared test fixtures: in-memory sqlite DB + ASGI test client.

The ``client`` fixture overrides ``get_db`` with an in-memory sqlite engine,
so endpoint tests need no Postgres or network.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from agentos.app import create_app
from agentos.core.config import get_settings
from agentos.core.security import TokenCipher, get_session_tokens
from agentos.db.session import get_db
from agentos.models.base import Base
from agentos.models.oauth_connection import OAuthConnection
from agentos.models.user import User

DbFactory = async_sessionmaker[AsyncSession]


@pytest.fixture()
async def db_engine() -> AsyncIterator[AsyncEngine]:
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest.fixture()
async def db_factory(db_engine: AsyncEngine) -> AsyncIterator[DbFactory]:
    yield async_sessionmaker(db_engine, expire_on_commit=False)


@pytest.fixture()
async def client(db_factory: DbFactory) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()

    async def override_db() -> AsyncIterator[AsyncSession]:
        async with db_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client


async def seed_authenticated_user(
    db_factory: DbFactory,
    *,
    github_id: int,
    username: str,
    access_token: str = "gho_test_access_token",
    encrypted_token: str | None = None,
) -> str:
    """Insert a user + GitHub connection and return a valid session cookie."""
    async with db_factory() as session:
        user = User(github_id=github_id, username=username)
        session.add(user)
        await session.flush()
        ciphertext = encrypted_token or TokenCipher(get_settings().fernet_key).encrypt_token(
            access_token
        )
        session.add(
            OAuthConnection(
                user_id=user.id,
                provider="github",
                access_token_encrypted=ciphertext,
                token_type="bearer",
                scope="repo",
            )
        )
        await session.commit()
        user_id = user.id
    return get_session_tokens().create(user_id)


def auth_cookie(client: httpx.AsyncClient, cookie: str) -> None:
    client.cookies.set("agentos_session", cookie)
