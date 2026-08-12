"""GitHub REST client tests via httpx.MockTransport (no network)."""

from collections.abc import Awaitable, Callable

import httpx
import pytest

from agentos.services import github

PAGE_1 = [
    {
        "full_name": "octo/repo-a",
        "private": False,
        "default_branch": "main",
        "description": "First repo",
        "updated_at": "2026-01-02T00:00:00Z",
        "html_url": "https://github.com/octo/repo-a",
    },
    {
        "full_name": "octo/repo-b",
        "private": True,
        "default_branch": "trunk",
        "description": None,
        "updated_at": "2026-01-01T00:00:00Z",
        "html_url": "https://github.com/octo/repo-b",
    },
]
PAGE_2 = [
    {
        "full_name": "octo/repo-c",
        "private": False,
        "default_branch": "main",
        "description": "Second page repo",
        "updated_at": "2025-12-01T00:00:00Z",
        "html_url": "https://github.com/octo/repo-c",
    }
]
NEXT_URL = f"{github.GITHUB_API_URL}/user/repos?sort=updated&per_page=100&page=2"

Handler = Callable[[httpx.Request], Awaitable[httpx.Response]]


def _mock_client(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> None:
    real_client = httpx.AsyncClient

    def factory(**kwargs: object) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(github.httpx, "AsyncClient", factory)


async def test_list_repositories_single_page(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/user/repos"
        assert request.headers["authorization"] == "Bearer gho_test"
        return httpx.Response(200, json=PAGE_1)

    _mock_client(monkeypatch, handler)
    result = await github.list_repositories("gho_test")
    assert [r.full_name for r in result] == ["octo/repo-a", "octo/repo-b"]
    assert result[0].private is False
    assert result[1].private is True
    assert result[1].default_branch == "trunk"
    assert result[1].description is None


async def test_list_repositories_ignores_missing_default_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[{"full_name": "octo/bare", "private": False, "url": "x"}])

    _mock_client(monkeypatch, handler)
    result = await github.list_repositories("gho_test")
    assert result[0].default_branch == "main"


async def test_list_repositories_follows_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        if request.url.params.get("page") == "2":
            return httpx.Response(200, json=PAGE_2)
        return httpx.Response(200, json=PAGE_1, headers={"link": f'<{NEXT_URL}>; rel="next"'})

    _mock_client(monkeypatch, handler)
    result = await github.list_repositories("gho_test")
    assert [r.full_name for r in result] == ["octo/repo-a", "octo/repo-b", "octo/repo-c"]
    assert len(seen) == 2
    assert seen[0].startswith(f"{github.GITHUB_API_URL}/user/repos")
    assert "page=2" in seen[1]


async def test_list_repositories_401(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    _mock_client(monkeypatch, handler)
    with pytest.raises(github.GitHubClientError) as excinfo:
        await github.list_repositories("gho_test")
    assert excinfo.value.status == 401


async def test_get_repository_success(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octo/repo-a"
        return httpx.Response(200, json=PAGE_1[0])

    _mock_client(monkeypatch, handler)
    result = await github.get_repository("gho_test", "octo/repo-a")
    assert result.full_name == "octo/repo-a"
    assert result.html_url == "https://github.com/octo/repo-a"


async def test_get_repository_404(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    _mock_client(monkeypatch, handler)
    with pytest.raises(github.GitHubClientError) as excinfo:
        await github.get_repository("gho_test", "octo/private-repo")
    assert excinfo.value.status == 404
