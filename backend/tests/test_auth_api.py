"""GitHub OAuth endpoint tests: login redirect, callback, session, me, logout.

Uses the shared ASGI + in-memory sqlite fixtures (see conftest.py) and
monkeypatched GitHub service calls — no network, no Postgres.
"""

import httpx
import pytest
from sqlalchemy import select

from agentos.core.config import get_settings
from agentos.core.security import TokenCipher, get_session_tokens
from agentos.models.oauth_connection import OAuthConnection
from agentos.models.user import User
from agentos.services import github_oauth
from tests.conftest import DbFactory

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
    db_factory: DbFactory,
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

    async with db_factory() as session:
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


async def test_me_with_stored_connection(client: httpx.AsyncClient, db_factory: DbFactory) -> None:
    async with db_factory() as session:
        user = User(github_id=999999, username="seeded-user")
        session.add(user)
        await session.flush()
        session.add(
            OAuthConnection(
                user_id=user.id,
                provider="github",
                access_token_encrypted=TokenCipher(get_settings().fernet_key).encrypt_token(
                    "gho_seeded_token"
                ),
                token_type="bearer",
                scope="repo",
            )
        )
        await session.commit()
        user_id = user.id
    client.cookies.set("agentos_session", get_session_tokens().create(user_id))
    response = await client.get(ME_PATH)
    assert response.status_code == 200
    assert response.json()["username"] == "seeded-user"


async def test_me_requires_connection(client: httpx.AsyncClient, db_factory: DbFactory) -> None:
    async with db_factory() as session:
        user = User(github_id=999998, username="no-connection")
        session.add(user)
        await session.commit()
        user_id = user.id
    client.cookies.set("agentos_session", get_session_tokens().create(user_id))
    assert (await client.get(ME_PATH)).status_code == 401


async def test_me_rejects_unreadable_token(
    client: httpx.AsyncClient, db_factory: DbFactory
) -> None:
    async with db_factory() as session:
        user = User(github_id=999997, username="bad-token")
        session.add(user)
        await session.flush()
        session.add(
            OAuthConnection(
                user_id=user.id,
                provider="github",
                access_token_encrypted="not-valid-ciphertext",
            )
        )
        await session.commit()
        user_id = user.id
    client.cookies.set("agentos_session", get_session_tokens().create(user_id))
    assert (await client.get(ME_PATH)).status_code == 401


async def test_logout_clears_session(client: httpx.AsyncClient, patch_github: None) -> None:
    state = await _login_and_state(client)
    await client.get(f"{CALLBACK_PATH}?code=fake_code&state={state}", follow_redirects=False)
    assert (await client.get(ME_PATH)).status_code == 200
    assert (await client.post(LOGOUT_PATH)).status_code == 204
    assert (await client.get(ME_PATH)).status_code == 401
