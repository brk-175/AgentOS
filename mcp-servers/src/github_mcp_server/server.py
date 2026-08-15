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
MAX_DIFF_CHARS = 200_000

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")


def _github_headers(raw: bool = False, diff: bool = False) -> dict[str, str]:
    if diff:
        accept = "application/vnd.github.v3.diff"
    elif raw:
        accept = "application/vnd.github.raw+json"
    else:
        accept = "application/vnd.github+json"
    headers = {
        "Accept": accept,
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
    if response.status_code == 422:
        raise ValueError(
            "GitHub rejected the request (HTTP 422) — e.g. branch or PR already exists"
        )
    if response.status_code >= 400:
        raise ValueError(f"GitHub API error: HTTP {response.status_code}")


def _json_or_raise(response: httpx.Response) -> Any:
    _check_status(response)
    return response.json()


def _require_token() -> None:
    """Fail fast for write operations that GitHub rejects without auth."""
    if not GITHUB_TOKEN:
        raise ValueError(
            "GITHUB_TOKEN is required for write operations — set it in the server environment"
        )


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


@mcp.tool()
async def get_issue(owner: str, name: str, number: int) -> dict[str, Any]:
    """Fetch the details of issue ``number`` in ``owner/name``.

    Returns the title, state, reporter, labels, body, and comment bodies —
    the context the agent needs to understand the problem. Note that pull
    requests are issues on GitHub, so this also works for them.
    """
    url = f"{GITHUB_API_URL}/repos/{owner}/{name}/issues/{number}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(url, headers=_github_headers())
        payload = _json_or_raise(response)
        comments_response = await client.get(
            f"{GITHUB_API_URL}/repos/{owner}/{name}/issues/{number}/comments",
            headers=_github_headers(),
        )
        comments = _json_or_raise(comments_response)
    return {
        "number": payload["number"],
        "title": payload["title"],
        "state": payload["state"],
        "reporter": payload["user"]["login"],
        "labels": [label["name"] for label in payload.get("labels", [])],
        "body": payload.get("body") or "",
        "comments": [
            {"user": comment["user"]["login"], "body": comment.get("body") or ""}
            for comment in comments
        ],
    }


@mcp.tool()
async def get_pr(owner: str, name: str, number: int) -> dict[str, Any]:
    """Fetch the details of pull request ``number`` in ``owner/name``.

    Returns the title, state, author, head/base branches, body, and diff
    statistics (files changed, additions, deletions) so the agent can
    evaluate the change without downloading the full diff.
    """
    url = f"{GITHUB_API_URL}/repos/{owner}/{name}/pulls/{number}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(url, headers=_github_headers())
        payload = _json_or_raise(response)
    return {
        "number": payload["number"],
        "title": payload["title"],
        "state": payload["state"],
        "author": payload["user"]["login"],
        "head": f"{payload['head']['repo']['full_name']}:{payload['head']['ref']}",
        "base": payload["base"]["ref"],
        "merged": payload.get("merged", False),
        "additions": payload.get("additions", 0),
        "deletions": payload.get("deletions", 0),
        "changed_files": payload.get("changed_files", 0),
        "body": payload.get("body") or "",
    }


@mcp.tool()
async def get_pr_diff(owner: str, name: str, number: int) -> str:
    """Return the unified diff of pull request ``number`` in ``owner/name``.

    The raw patch text (files, hunks, changed lines) — the primary input
    for reviewing or fixing a PR. Capped at 200,000 characters; use
    ``get_pr`` for the summary instead if the diff is too large.
    """
    url = f"{GITHUB_API_URL}/repos/{owner}/{name}/pulls/{number}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(url, headers=_github_headers(diff=True))
        _check_status(response)
        content = response.text
    if len(content) > MAX_DIFF_CHARS:
        raise ValueError(f"diff exceeds {MAX_DIFF_CHARS} characters — use get_pr for a summary")
    return content


@mcp.tool()
async def create_branch(owner: str, name: str, base_branch: str, new_branch: str) -> dict[str, Any]:
    """Create a new branch ``new_branch`` starting from ``base_branch`` in ``owner/name``.

    Returns the new branch ref and the commit SHA it points at. Fails cleanly
    if the base branch is unknown or the new branch already exists.
    """
    _require_token()
    base_ref = quote(base_branch, safe="")
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.get(
            f"{GITHUB_API_URL}/repos/{owner}/{name}/git/ref/heads/{base_ref}",
            headers=_github_headers(),
        )
        base_sha = _json_or_raise(response)["object"]["sha"]
        response = await client.post(
            f"{GITHUB_API_URL}/repos/{owner}/{name}/git/refs",
            headers=_github_headers(),
            json={"ref": f"refs/heads/{new_branch}", "sha": base_sha},
        )
        payload = _json_or_raise(response)
    return {"ref": payload["ref"], "sha": payload["object"]["sha"], "base_sha": base_sha}


@mcp.tool()
async def create_commit(
    owner: str, name: str, branch: str, message: str, changes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Commit file ``changes`` to ``branch`` in ``owner/name`` as a single commit.

    Each entry in ``changes`` is ``{"path": "...", "content": "..."}`` to
    create/update a file, or ``{"path": "...", "delete": true}`` to remove it.
    Commits through the git database API (blobs → tree → commit) and
    fast-forwards the branch ref. Returns the branch and the new commit SHA.
    """
    _require_token()
    if not changes:
        raise ValueError("changes must contain at least one file change")
    if any(not change.get("path") for change in changes):
        raise ValueError("every change must include a non-empty path")

    branch_ref = quote(branch, safe="")
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        base_url = f"{GITHUB_API_URL}/repos/{owner}/{name}"
        response = await client.get(
            f"{base_url}/git/ref/heads/{branch_ref}", headers=_github_headers()
        )
        base_sha = _json_or_raise(response)["object"]["sha"]
        response = await client.get(f"{base_url}/git/commits/{base_sha}", headers=_github_headers())
        base_tree_sha = _json_or_raise(response)["tree"]["sha"]

        tree_items: list[dict[str, Any]] = []
        for change in changes:
            if change.get("delete"):
                tree_items.append({"path": change["path"], "sha": None})
                continue
            response = await client.post(
                f"{base_url}/git/blobs",
                headers=_github_headers(),
                json={"content": change["content"], "encoding": "utf-8"},
            )
            blob_sha = _json_or_raise(response)["sha"]
            tree_items.append(
                {"path": change["path"], "mode": "100644", "type": "blob", "sha": blob_sha}
            )

        response = await client.post(
            f"{base_url}/git/trees",
            headers=_github_headers(),
            json={"base_tree": base_tree_sha, "tree": tree_items},
        )
        new_tree_sha = _json_or_raise(response)["sha"]
        response = await client.post(
            f"{base_url}/git/commits",
            headers=_github_headers(),
            json={"message": message, "tree": new_tree_sha, "parents": [base_sha]},
        )
        commit_sha = _json_or_raise(response)["sha"]
        response = await client.patch(
            f"{base_url}/git/refs/heads/{branch_ref}",
            headers=_github_headers(),
            json={"sha": commit_sha, "force": False},
        )
        _check_status(response)
    return {"branch": branch, "commit_sha": commit_sha}


@mcp.tool()
async def create_pull_request(
    owner: str, name: str, title: str, head: str, base: str, body: str = ""
) -> dict[str, Any]:
    """Open a pull request from ``head`` into ``base`` in ``owner/name``.

    Returns the PR number and its HTML URL for human review. The ``head``
    branch must exist and differ from ``base``.
    """
    _require_token()
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT_SECONDS) as client:
        response = await client.post(
            f"{GITHUB_API_URL}/repos/{owner}/{name}/pulls",
            headers=_github_headers(),
            json={"title": title, "head": head, "base": base, "body": body},
        )
        payload = _json_or_raise(response)
    return {"number": payload["number"], "url": payload["html_url"]}


def main() -> None:
    """Run the MCP server over stdio (the agent's runner connects via stdio)."""
    mcp.run()


if __name__ == "__main__":
    main()
