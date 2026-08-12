"""Repository API tests: live GitHub data mapped through the user's token."""

import httpx
import pytest

from agentos.services import github
from tests.conftest import DbFactory, auth_cookie, seed_authenticated_user

REPOS_PATH = "/api/v1/repos"
FETCH_PATH = "/api/v1/repos/fetch"

FAKE_REPOS = [
    github.RepositorySummary(
        full_name="octo/repo-a",
        private=False,
        default_branch="main",
        description="First repo",
        updated_at="2026-01-02T00:00:00Z",
        html_url="https://github.com/octo/repo-a",
    ),
    github.RepositorySummary(
        full_name="octo/repo-b",
        private=True,
        default_branch="trunk",
        description=None,
        updated_at="2026-01-01T00:00:00Z",
        html_url="https://github.com/octo/repo-b",
    ),
]


async def _authed_client(client: httpx.AsyncClient, db_factory: DbFactory) -> None:
    auth_cookie(client, await seed_authenticated_user(db_factory, github_id=31337, username="octo"))


async def test_list_repos_requires_auth(client: httpx.AsyncClient) -> None:
    response = await client.get(REPOS_PATH)
    assert response.status_code == 401


async def test_list_repos_success(
    client: httpx.AsyncClient, db_factory: DbFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _authed_client(client, db_factory)

    async def fake_list(access_token: str) -> list[github.RepositorySummary]:
        assert access_token == "gho_test_access_token"
        return FAKE_REPOS

    monkeypatch.setattr(github, "list_repositories", fake_list)
    response = await client.get(REPOS_PATH)
    assert response.status_code == 200
    body = response.json()
    assert [repo["full_name"] for repo in body] == ["octo/repo-a", "octo/repo-b"]
    assert body[1]["private"] is True
    assert body[1]["description"] is None
    assert body[0]["default_branch"] == "main"


async def test_list_repos_token_revoked(
    client: httpx.AsyncClient, db_factory: DbFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _authed_client(client, db_factory)

    async def fake_list(access_token: str) -> list[github.RepositorySummary]:
        raise github.GitHubClientError("GitHub token is invalid or revoked", status=401)

    monkeypatch.setattr(github, "list_repositories", fake_list)
    response = await client.get(REPOS_PATH)
    assert response.status_code == 401
    assert "reconnect" in response.json()["detail"].lower()


async def test_list_repos_rate_limited(
    client: httpx.AsyncClient, db_factory: DbFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _authed_client(client, db_factory)

    async def fake_list(access_token: str) -> list[github.RepositorySummary]:
        raise github.GitHubClientError("rate limit", status=403)

    monkeypatch.setattr(github, "list_repositories", fake_list)
    response = await client.get(REPOS_PATH)
    assert response.status_code == 429


async def test_fetch_repo_success(
    client: httpx.AsyncClient, db_factory: DbFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _authed_client(client, db_factory)

    async def fake_get(access_token: str, full_name: str) -> github.RepositorySummary:
        assert full_name == "octo/repo-a"
        return FAKE_REPOS[0]

    monkeypatch.setattr(github, "get_repository", fake_get)
    response = await client.get(f"{FETCH_PATH}?full_name=octo/repo-a")
    assert response.status_code == 200
    assert response.json()["full_name"] == "octo/repo-a"


async def test_fetch_repo_not_found(
    client: httpx.AsyncClient, db_factory: DbFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _authed_client(client, db_factory)

    async def fake_get(access_token: str, full_name: str) -> github.RepositorySummary:
        raise github.GitHubClientError("Repository not found or inaccessible", status=404)

    monkeypatch.setattr(github, "get_repository", fake_get)
    response = await client.get(f"{FETCH_PATH}?full_name=octo/ghost")
    assert response.status_code == 404


async def test_fetch_repo_requires_full_name(
    client: httpx.AsyncClient, db_factory: DbFactory
) -> None:
    await _authed_client(client, db_factory)
    response = await client.get(FETCH_PATH)
    assert response.status_code == 422
