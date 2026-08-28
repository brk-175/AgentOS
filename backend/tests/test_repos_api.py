"""Endpoint tests for RAG: repo indexing + semantic search."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from agentos.api.deps import get_db
from agentos.api.repos import get_embeddings
from agentos.app import create_app
from agentos.core.config import get_settings
from agentos.services.github import RepositorySummary
from agentos.services.rag import IndexSummary, SearchHit
from tests.conftest import seed_authenticated_user

API_PREFIX = get_settings().api_prefix


class _FakeEmbeddings:
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    async def aembed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


@pytest.fixture()
async def repos_env(
    db_factory: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[httpx.AsyncClient, pytest.MonkeyPatch]]:
    app = create_app()
    app.dependency_overrides[get_embeddings] = lambda: _FakeEmbeddings()

    async def override_db() -> AsyncIterator[AsyncSession]:
        async with db_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_db

    async def fake_get_repository(access_token: str, full_name: str) -> RepositorySummary:
        if full_name == "missing/repo":
            from agentos.services.github import GitHubClientError

            raise GitHubClientError("Repository not found or inaccessible", status=404)
        return RepositorySummary(
            full_name=full_name,
            private=False,
            default_branch="main",
            description=None,
            updated_at="2026-01-01T00:00:00Z",
            html_url=f"https://github.com/{full_name}",
        )

    monkeypatch.setattr("agentos.api.repos.github.get_repository", fake_get_repository)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client, monkeypatch


async def _auth_cookie(
    client: httpx.AsyncClient, db_factory: async_sessionmaker[AsyncSession]
) -> None:
    cookie = await seed_authenticated_user(db_factory, github_id=333, username="carol")
    client.cookies.set("agentos_session", cookie)


async def test_index_requires_auth(
    repos_env: tuple[httpx.AsyncClient, pytest.MonkeyPatch],
) -> None:
    client, _ = repos_env
    response = await client.post(f"{API_PREFIX}/repos/octocat/AgentOS/index")
    assert response.status_code == 401


async def test_index_calls_rag_and_reports_summary(
    repos_env: tuple[httpx.AsyncClient, pytest.MonkeyPatch],
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, monkeypatch = repos_env
    await _auth_cookie(client, db_factory)
    captured: dict[str, Any] = {}

    async def fake_index(db: AsyncSession, token: str, repo: str, *, embeddings: Any = None) -> Any:
        captured["token"] = token
        captured["repo"] = repo
        return IndexSummary(repo_full_name=repo, files_indexed=4, chunks=12, chars=500)

    monkeypatch.setattr("agentos.api.repos.rag.index_repository", fake_index)

    response = await client.post(f"{API_PREFIX}/repos/octocat/AgentOS/index")
    assert response.status_code == 200
    assert response.json() == {
        "full_name": "octocat/AgentOS",
        "files_indexed": 4,
        "chunks": 12,
        "chars": 500,
    }
    assert captured["token"] == "gho_test_access_token"
    assert captured["repo"] == "octocat/AgentOS"


async def test_index_missing_repo_maps_to_404(
    repos_env: tuple[httpx.AsyncClient, pytest.MonkeyPatch],
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, _ = repos_env
    await _auth_cookie(client, db_factory)
    response = await client.post(f"{API_PREFIX}/repos/missing/repo/index")
    assert response.status_code == 404


async def test_index_rag_failure_maps_to_502(
    repos_env: tuple[httpx.AsyncClient, pytest.MonkeyPatch],
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, monkeypatch = repos_env
    await _auth_cookie(client, db_factory)

    async def broken_index(
        db: AsyncSession, token: str, repo: str, *, embeddings: Any = None
    ) -> Any:
        raise RuntimeError("MCP server died")

    monkeypatch.setattr("agentos.api.repos.rag.index_repository", broken_index)

    response = await client.post(f"{API_PREFIX}/repos/octocat/AgentOS/index")
    assert response.status_code == 502


async def test_search_returns_ranked_hits(
    repos_env: tuple[httpx.AsyncClient, pytest.MonkeyPatch],
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, monkeypatch = repos_env
    await _auth_cookie(client, db_factory)

    async def fake_search(
        db: AsyncSession, repo: str, query: str, *, embeddings: Any = None, **_: Any
    ) -> list[SearchHit]:
        return [SearchHit(path="src/crash.py", chunk_index=0, content="null bug", score=0.9215)]

    monkeypatch.setattr("agentos.api.repos.rag.search_repository", fake_search)

    response = await client.get(f"{API_PREFIX}/repos/octocat/AgentOS/search", params={"q": "crash"})
    assert response.status_code == 200
    assert response.json() == {
        "query": "crash",
        "hits": [
            {"path": "src/crash.py", "chunk_index": 0, "content": "null bug", "score": 0.9215}
        ],
    }


async def test_search_validates_query_length(
    repos_env: tuple[httpx.AsyncClient, pytest.MonkeyPatch],
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, _ = repos_env
    await _auth_cookie(client, db_factory)
    response = await client.get(f"{API_PREFIX}/repos/octocat/AgentOS/search", params={"q": "x"})
    assert response.status_code == 422


async def test_targets_lists_issues_for_repo(
    repos_env: tuple[httpx.AsyncClient, pytest.MonkeyPatch],
    db_factory: async_sessionmaker[AsyncSession],
) -> None:
    client, monkeypatch = repos_env
    await _auth_cookie(client, db_factory)
    captured: dict[str, Any] = {}

    async def fake_list(token: str, repo: str, kind: str) -> list[Any]:
        captured["token"] = token
        captured["repo"] = repo
        captured["kind"] = kind
        from agentos.services.github import IssuePull

        return [
            IssuePull(
                kind="issue",
                number=11,
                title="Remove futile tabs",
                state="open",
                created_at="2026-08-01T10:00:00Z",
                updated_at="2026-08-02T10:00:00Z",
            )
        ]

    monkeypatch.setattr("agentos.api.repos.github.list_issue_pulls", fake_list)

    response = await client.get(
        f"{API_PREFIX}/repos/octocat/AgentOS/targets", params={"kind": "issue"}
    )
    assert response.status_code == 200
    assert response.json() == [
        {
            "kind": "issue",
            "number": 11,
            "title": "Remove futile tabs",
            "state": "open",
            "created_at": "2026-08-01T10:00:00Z",
            "updated_at": "2026-08-02T10:00:00Z",
            "merged_at": None,
        }
    ]
    assert captured == {
        "token": "gho_test_access_token",
        "repo": "octocat/AgentOS",
        "kind": "issue",
    }


async def test_targets_requires_auth(
    repos_env: tuple[httpx.AsyncClient, pytest.MonkeyPatch],
) -> None:
    client, _ = repos_env
    response = await client.get(f"{API_PREFIX}/repos/octocat/AgentOS/targets")
    assert response.status_code == 401
