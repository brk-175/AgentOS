"""GitHub MCP server: repo-exploration tools for the AgentOS agent.

Talked to by the LangGraph agent over the stdio transport. Reads
``GITHUB_TOKEN`` from the environment (the agent runner passes the user's
decrypted OAuth token); making it optional keeps public-repo smoke tests
token-free. Every tool failure surfaces as a ``ValueError`` so the MCP
layer converts it into a clean tool error the model can act on.
"""

from __future__ import annotations

import os
from typing import Any
from urllib.parse import quote

import httpx
from mcp.server.mcpserver import MCPServer

mcp = MCPServer(name="github-mcp-server")

GITHUB_API_URL = "https://api.github.com"
HTTP_TIMEOUT_SECONDS = 15.0
MAX_READ_CHARS = 1_000_000

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def _github_headers(raw: bool = False) -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github.raw+json" if raw else "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "AgentOS-MCP",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    return headers


def _check_status(response: httpx.Response) -> None:
    if response.status_code == 401:
        raise ValueError("GitHub token is invalid or missing — grant the runner a valid token")
    if response.status_code == 404:
        raise ValueError("Path or repository not found (or you lack access)")
    if response.status_code == 403:
        raise ValueError("GitHub API rate limit exceeded or access forbidden")
    if response.status_code >= 400:
        raise ValueError(f"GitHub API error: HTTP {response.status_code}")


def _json_or_raise(response: httpx.Response) -> Any:
    _check_status(response)
    return response.json()


@mcp.tool()
async def list_repo_files(owner: str, name: str, path: str = "") -> list[dict]:
    """List the files and directories at ``path`` in the ``owner/name`` repository.

    Returns one entry per item with ``kind`` (file/dir), ``name``, ``path``
    and ``size`` (bytes; 0 for directories). Pass a subdirectory in ``path``
    to navigate deeper, or omit it to start at the repository root.
    """
    url = f"{GITHUB_API_URL}/repos/{owner}/{name}/contents/{quote(path, safe='/')}".rstrip("/")
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(url, headers=_github_headers())
    payload = _json_or_raise(response)
    if isinstance(payload, dict):
        payload = [payload]
    return [
        {
            "kind": entry["type"],
            "name": entry["name"],
            "path": entry["path"],
            "size": entry.get("size", 0),
        }
        for entry in payload
    ]


@mcp.tool()
async def read_file(owner: str, name: str, path: str) -> str:
    """Read the raw contents of a single file in the ``owner/name`` repository.

    Returns the file text as-is (max 1,000,000 chars) for the agent's
    analysis. For non-text or overly large files, use ``list_repo_files``
    and target specific files instead.
    """
    url = f"{GITHUB_API_URL}/repos/{owner}/{name}/contents/{quote(path, safe='/')}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(url, headers=_github_headers(raw=True))
    _check_status(response)
    content = response.text
    if len(content) > MAX_READ_CHARS:
        raise ValueError(f"file exceeds {MAX_READ_CHARS} characters — target a smaller file")
    return content


def main() -> None:
    """Run the MCP server over stdio (the agent's runner connects via stdio)."""
    mcp.run()


if __name__ == "__main__":
    main()
