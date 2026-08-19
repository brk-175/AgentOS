"""Tool logic tests via httpx.MockTransport + direct calls (no MCP wire)."""

import json
from collections.abc import Callable, Coroutine

import httpx
import pytest

from github_mcp_server import server

Handler = (
    Callable[[httpx.Request], httpx.Response]
    | Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]
)


def _mock_client(monkeypatch: pytest.MonkeyPatch, handler: Handler) -> None:
    real_client = httpx.AsyncClient

    def factory(timeout: float) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handler), timeout=timeout)

    monkeypatch.setattr(server.httpx, "AsyncClient", factory)


async def test_list_repo_files_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octo/repo/contents/src"
        return httpx.Response(
            200,
            json=[
                {"type": "dir", "name": "agentos", "path": "src/agentos", "size": 0},
                {"type": "file", "name": "main.py", "path": "src/main.py", "size": 1200},
            ],
        )

    _mock_client(monkeypatch, handler)
    result = await server.list_repo_files("octo", "repo", "src")
    assert result == {
        "kind": "listing",
        "items": [
            {"kind": "dir", "name": "agentos", "path": "src/agentos", "size": 0},
            {"kind": "file", "name": "main.py", "path": "src/main.py", "size": 1200},
        ],
    }


async def test_list_repo_files_root(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octo/repo/contents"
        return httpx.Response(200, json=[])

    _mock_client(monkeypatch, handler)
    assert await server.list_repo_files("octo", "repo") == {"kind": "listing", "items": []}


async def test_list_repo_files_single_file_returns_one_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"type": "file", "name": "README.md", "path": "README.md", "size": 99},
        )

    _mock_client(monkeypatch, handler)
    entries = (await server.list_repo_files("octo", "repo", "README.md"))["items"]
    assert len(entries) == 1
    assert entries[0]["kind"] == "file"


async def test_list_repo_files_404(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    _mock_client(monkeypatch, handler)
    with pytest.raises(ValueError):
        await server.list_repo_files("octo", "repo", "ghost")


async def test_list_repo_files_401(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"message": "Bad credentials"})

    _mock_client(monkeypatch, handler)
    with pytest.raises(ValueError):
        await server.list_repo_files("octo", "repo")


async def test_read_file_returns_raw_text(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"] == "application/vnd.github.raw+json"
        return httpx.Response(200, text="print('hello world')\n")

    _mock_client(monkeypatch, handler)
    content = await server.read_file("octo", "repo", "main.py")
    assert content == "print('hello world')\n"


async def test_read_file_rejects_huge_content(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="x" * (server.MAX_READ_CHARS + 1))

    _mock_client(monkeypatch, handler)
    with pytest.raises(ValueError):
        await server.read_file("octo", "repo", "big.txt")


async def test_get_issue_fetches_details_and_comments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/repos/octo/repo/issues/1":
            return httpx.Response(
                200,
                json={
                    "number": 1,
                    "title": "Crash on empty input",
                    "state": "open",
                    "user": {"login": "octo"},
                    "labels": [{"name": "bug"}, {"name": "priority"}],
                    "body": "App crashes when input is empty.",
                },
            )
        assert request.url.path == "/repos/octo/repo/issues/1/comments"
        return httpx.Response(
            200,
            json=[
                {"user": {"login": "helper"}, "body": "Reproduced locally."},
                {"user": {"login": "octo"}, "body": "Thanks!"},
            ],
        )

    _mock_client(monkeypatch, handler)
    result = await server.get_issue("octo", "repo", 1)
    assert result == {
        "number": 1,
        "title": "Crash on empty input",
        "state": "open",
        "reporter": "octo",
        "labels": ["bug", "priority"],
        "body": "App crashes when input is empty.",
        "comments": [
            {"user": "helper", "body": "Reproduced locally."},
            {"user": "octo", "body": "Thanks!"},
        ],
    }


async def test_get_issue_404(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"message": "Not Found"})

    _mock_client(monkeypatch, handler)
    with pytest.raises(ValueError):
        await server.get_issue("octo", "repo", 999)


async def test_get_pr_fetches_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octo/repo/pulls/5"
        return httpx.Response(
            200,
            json={
                "number": 5,
                "title": "Fix empty input crash",
                "state": "open",
                "user": {"login": "octo"},
                "head": {"repo": {"full_name": "octo/repo"}, "ref": "fix/empty-input"},
                "base": {"ref": "main"},
                "merged": False,
                "additions": 12,
                "deletions": 2,
                "changed_files": 1,
                "body": "Adds a null check.",
            },
        )

    _mock_client(monkeypatch, handler)
    result = await server.get_pr("octo", "repo", 5)
    assert result == {
        "number": 5,
        "title": "Fix empty input crash",
        "state": "open",
        "author": "octo",
        "head": "octo/repo:fix/empty-input",
        "base": "main",
        "merged": False,
        "additions": 12,
        "deletions": 2,
        "changed_files": 1,
        "body": "Adds a null check.",
    }


async def test_get_pr_diff_returns_plain_text(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octo/repo/pulls/5"
        assert request.headers["accept"] == "application/vnd.github.v3.diff"
        return httpx.Response(200, text="diff --git a/main.py b/main.py\n+fixed\n")

    _mock_client(monkeypatch, handler)
    assert await server.get_pr_diff("octo", "repo", 5) == "diff --git a/main.py b/main.py\n+fixed\n"


async def test_get_pr_diff_rejects_huge_diff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="x" * (server.MAX_DIFF_CHARS + 1))

    _mock_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="diff exceeds"):
        await server.get_pr_diff("octo", "repo", 5)


async def test_create_branch_creates_ref(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "GITHUB_TOKEN", "test-token")

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            assert request.url.path == "/repos/octo/repo/git/ref/heads/main"
            return httpx.Response(200, json={"object": {"sha": "abc123"}})
        assert request.url.path == "/repos/octo/repo/git/refs"
        body = json.loads(request.read())
        assert body == {"ref": "refs/heads/feature/x", "sha": "abc123"}
        return httpx.Response(
            201, json={"ref": "refs/heads/feature/x", "object": {"sha": "abc123"}}
        )

    _mock_client(monkeypatch, handler)
    result = await server.create_branch("octo", "repo", "main", "feature/x")
    assert result == {"ref": "refs/heads/feature/x", "sha": "abc123", "base_sha": "abc123"}


async def test_create_branch_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "GITHUB_TOKEN", "")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"object": {"sha": "abc123"}})

    _mock_client(monkeypatch, handler)
    with pytest.raises(ValueError, match="GITHUB_TOKEN"):
        await server.create_branch("octo", "repo", "main", "feature/x")


async def test_create_commit_upserts_and_deletes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "GITHUB_TOKEN", "test-token")
    seen: list[tuple[str, str]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.method, request.url.path))
        if request.url.path == "/repos/octo/repo/git/ref/heads/main":
            return httpx.Response(200, json={"object": {"sha": "base_commit"}})
        if request.url.path == "/repos/octo/repo/git/commits/base_commit":
            return httpx.Response(200, json={"tree": {"sha": "base_tree"}})
        if request.url.path.endswith("/git/blobs"):
            return httpx.Response(201, json={"sha": "blob_sha"})
        if request.url.path.endswith("/git/trees"):
            body = json.loads(request.read())
            assert body["base_tree"] == "base_tree"
            assert body["tree"] == [
                {"path": "app.py", "mode": "100644", "type": "blob", "sha": "blob_sha"},
                {"path": "old.py", "sha": None},
            ]
            return httpx.Response(201, json={"sha": "new_tree"})
        if request.url.path.endswith("/git/commits"):
            body = json.loads(request.read())
            assert body == {
                "message": "fix bug",
                "tree": "new_tree",
                "parents": ["base_commit"],
            }
            return httpx.Response(201, json={"sha": "commit_sha"})
        if request.method == "PATCH":
            assert json.loads(request.read()) == {"sha": "commit_sha", "force": False}
            return httpx.Response(200, json={"ref": "refs/heads/main"})
        return httpx.Response(500)

    _mock_client(monkeypatch, handler)
    result = await server.create_commit(
        "octo",
        "repo",
        "main",
        "fix bug",
        [
            {"path": "app.py", "content": "print('fixed')\n"},
            {"path": "old.py", "delete": True},
        ],
    )
    assert result == {"branch": "main", "commit_sha": "commit_sha"}
    assert seen == [
        ("GET", "/repos/octo/repo/git/ref/heads/main"),
        ("GET", "/repos/octo/repo/git/commits/base_commit"),
        ("POST", "/repos/octo/repo/git/blobs"),
        ("POST", "/repos/octo/repo/git/trees"),
        ("POST", "/repos/octo/repo/git/commits"),
        ("PATCH", "/repos/octo/repo/git/refs/heads/main"),
    ]


async def test_create_commit_rejects_empty_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "GITHUB_TOKEN", "test-token")
    _mock_client(monkeypatch, lambda request: httpx.Response(500))
    with pytest.raises(ValueError, match="changes must contain"):
        await server.create_commit("octo", "repo", "main", "noop", [])


async def test_create_commit_rejects_change_without_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "GITHUB_TOKEN", "test-token")
    _mock_client(monkeypatch, lambda request: httpx.Response(500))
    with pytest.raises(ValueError, match="non-empty path"):
        await server.create_commit("octo", "repo", "main", "noop", [{"content": "x"}])


async def test_create_pull_request_opens_pr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "GITHUB_TOKEN", "test-token")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octo/repo/pulls"
        body = json.loads(request.read())
        assert body == {
            "title": "Fix bug",
            "head": "feature/x",
            "base": "main",
            "body": "Closes #1",
        }
        return httpx.Response(
            201,
            json={"number": 42, "html_url": "https://github.com/octo/repo/pull/42"},
        )

    _mock_client(monkeypatch, handler)
    result = await server.create_pull_request(
        "octo", "repo", "Fix bug", "feature/x", "main", "Closes #1"
    )
    assert result == {"number": 42, "url": "https://github.com/octo/repo/pull/42"}


async def test_create_pull_request_requires_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(server, "GITHUB_TOKEN", "")
    _mock_client(monkeypatch, lambda request: httpx.Response(500))
    with pytest.raises(ValueError, match="GITHUB_TOKEN"):
        await server.create_pull_request("octo", "repo", "t", "h", "b")
