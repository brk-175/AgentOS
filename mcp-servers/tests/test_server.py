"""Tool logic tests via httpx.MockTransport + direct calls (no MCP wire)."""

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
    entries = await server.list_repo_files("octo", "repo", "src")
    assert entries == [
        {"kind": "dir", "name": "agentos", "path": "src/agentos", "size": 0},
        {"kind": "file", "name": "main.py", "path": "src/main.py", "size": 1200},
    ]


async def test_list_repo_files_root(monkeypatch: pytest.MonkeyPatch) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/repos/octo/repo/contents"
        return httpx.Response(200, json=[])

    _mock_client(monkeypatch, handler)
    assert await server.list_repo_files("octo", "repo") == []


async def test_list_repo_files_single_file_returns_one_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"type": "file", "name": "README.md", "path": "README.md", "size": 99},
        )

    _mock_client(monkeypatch, handler)
    entries = await server.list_repo_files("octo", "repo", "README.md")
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
