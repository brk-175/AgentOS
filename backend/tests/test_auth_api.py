"""GitHub OAuth endpoint tests: login redirect, callback, session, me, logout.

Uses an ASGI transport + in-memory sqlite (dependency override for
``get_db``) and monkeypatched GitHub service calls — no network, no Postgres.
"""

from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

from agentos.app import create_app
from agentos.core.config import get_settings
from agentos.core.security import TokenCipher
from agentos.db.session import get_db
from agentos.models.base import Base
from agentos.models.oauth_connection import OAuthConnection
from agentos.models.user import User
from agentos.services import github_oauth

TEST_PROFILE = {
    "id": 42424242,
    "login": "agentos-test",
    "name": "AgentOS Tester",
    "email": "tester@example.com",
    "avatar_url": "https://avatars.githubusercontent.com/u/42424242",
}

LOGIN_PATH = "/api/v1/auth/github/login"
CALLBACK_PATH = "/api/v1/auth/github/callback"
ME_PATH = "/api/v1/auth/me"
LOGOUT_PATH = "/api/v1/auth/logout"


@pytest.fixture()
async def _db() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(
        "sqlite+aiosqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    try:
        yield factory
    finally:
        await engine.dispose()


@pytest.fixture()
async def client(_db: async_sessionmaker[AsyncSession]) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app()

    async def override_db() -> AsyncIterator[AsyncSession]:
        async with _db() as session:
            yield session

    app.dependency_overrides[get_db] = override_db
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as test_client:
        yield test_client


async def _login_and_state(client: httpx.AsyncClient) -> str:
    response = await client.get(LOGIN_PATH, follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"].startswith("https://github.com/login/oauth/authorize")
    return client.cookies["agentos_oauth_state"]


@pytest.fixture()
def patch_github(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_exchange_code(**kwargs: object) -> github_oauth.OAuthToken:
        return github_oauth.OAuthToken(
            access_token="gho_test_access_token",
            token_type="Bearer",
            scope="repo user:email",
            expires_in=3600,
        )

    async def fake_fetch_user_profile(access_token: str) -> dict:
        return TEST_PROFILE

    monkeypatch.setattr(github_oauth, "exchange_code", fake_exchange_code)
    monkeypatch.setattr(github_oauth, "fetch_user_profile", fake_fetch_user_profile)


async def test_login_redirects_to_github(client: httpx.AsyncClient) -> None:
    state = await _login_and_state(client)
    assert len(state) == 43


async def test_callback_happy_path(
    client: httpx.AsyncClient,
    patch_github: None,
    _db: async_sessionmaker[AsyncSession],
) -> None:
    state = await _login_and_state(client)
    response = await client.get(
        f"{CALLBACK_PATH}?code=fake_code&state={state}", follow_redirects=False
    )
    assert response.status_code == 302
    assert response.headers["location"] == f"{get_settings().frontend_url}/dashboard"
    assert "agentos_oauth_state" not in client.cookies

    me = await client.get(ME_PATH)
    assert me.status_code == 200
    body = me.json()
    assert body["username"] == "agentos-test"
    assert body["github_id"] == TEST_PROFILE["id"]

    async with _db() as session:
        stored_user = (await session.execute(select(User))).scalar_one()
        stored_conn = (await session.execute(select(OAuthConnection))).scalar_one()
        cipher = TokenCipher(get_settings().fernet_key)
        assert stored_conn.access_token_encrypted != "gho_test_access_token"
        assert cipher.decrypt_token(stored_conn.access_token_encrypted) == "gho_test_access_token"
        assert stored_conn.provider == "github"
        assert stored_user.github_id == TEST_PROFILE["id"]


async def test_callback_rejects_wrong_state(client: httpx.AsyncClient, patch_github: None) -> None:
    await _login_and_state(client)
    response = await client.get(
        f"{CALLBACK_PATH}?code=fake_code&state=attacker-state", follow_redirects=False
    )
    assert response.status_code == 302
    assert "auth_error=invalid_state" in response.headers["location"]
    assert "agentos_session" not in client.cookies


async def test_callback_without_state(client: httpx.AsyncClient, patch_github: None) -> None:
    response = await client.get(f"{CALLBACK_PATH}?code=fake_code", follow_redirects=False)
    assert response.status_code == 302
    assert "auth_error=invalid_state" in response.headers["location"]


async def test_callback_github_error(client: httpx.AsyncClient, patch_github: None) -> None:
    state = await _login_and_state(client)
    response = await client.get(
        f"{CALLBACK_PATH}?error=access_denied&state={state}", follow_redirects=False
    )
    assert response.status_code == 302
    assert "auth_error=access_denied" in response.headers["location"]
    assert "agentos_session" not in client.cookies


async def test_me_requires_session(client: httpx.AsyncClient) -> None:
    response = await client.get(ME_PATH)
    assert response.status_code == 401


async def test_logout_clears_session(client: httpx.AsyncClient, patch_github: None) -> None:
    state = await _login_and_state(client)
    await client.get(f"{CALLBACK_PATH}?code=fake_code&state={state}", follow_redirects=False)
    assert (await client.get(ME_PATH)).status_code == 200
    assert (await client.post(LOGOUT_PATH)).status_code == 204
    assert (await client.get(ME_PATH)).status_code == 401
