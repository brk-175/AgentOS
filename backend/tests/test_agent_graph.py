"""Tests for the fix-agent pipeline and the MCP tool adapter."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import StructuredTool, ToolException
from langchain_openai import ChatOpenAI

from agentos.agent.graph import MAX_RAG_CONTEXT, create_agent_graph, create_agent_llm
from agentos.agent.mcp_adapter import (
    GitHubMCPTools,
    default_github_mcp_command,
    json_schema_to_model,
)
from agentos.agent.state import ContextDoc, FileChange, RunTarget
from agentos.services.rag import IndexSummary

ISSUE_INPUT: dict[str, Any] = {
    "target": RunTarget(repo_full_name="octocat/Hello-World", kind="issue", number=1),
    "messages": [],
    "events": [],
    "context": [],
}


def test_create_agent_llm_is_openrouter_bound_chat_model() -> None:
    llm = create_agent_llm()
    assert isinstance(llm, BaseChatModel)
    assert isinstance(llm, ChatOpenAI)


def test_json_schema_to_model_required_optional_and_defaults() -> None:
    schema = {
        "type": "object",
        "required": ["owner", "name"],
        "properties": {
            "owner": {"type": "string"},
            "name": {"type": "string"},
            "path": {"type": "string", "default": ""},
        },
    }
    model = json_schema_to_model("list_repo_files", schema)
    assert model.model_fields["owner"].is_required()
    assert model.model_fields["name"].is_required()
    assert not model.model_fields["path"].is_required()
    instance = model(owner="octocat", name="Hello-World")
    assert instance.path == ""
    assert instance.model_dump() == {"owner": "octocat", "name": "Hello-World", "path": ""}


async def test_graph_streams_through_all_four_nodes_in_order() -> None:
    graph = create_agent_graph()
    visited: list[str] = []
    async for update in graph.astream(dict(ISSUE_INPUT), stream_mode="updates"):
        visited.extend(update.keys())
    assert visited == ["investigate", "design", "apply", "pr"]


async def test_graph_invoke_produces_full_state_and_trace() -> None:
    graph = create_agent_graph()
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["investigation"] == "issue #1 in octocat/Hello-World"
    assert final["root_cause_hypothesis"] == ""
    assert final["proposed_changes"] == []
    assert final["applied_branch"] is None
    assert final["pr_url"] is None
    stages = [event.stage for event in final["events"]]
    assert stages == ["investigate", "design", "apply", "pr"]


async def test_live_github_mcp_tools_exposes_all_eight_tools() -> None:
    """End-to-end: boot the real MCP server over stdio and list its tools."""
    try:
        adapter = GitHubMCPTools()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    async with adapter:
        names = {tool.name for tool in adapter.tools}
    assert {
        "list_repo_files",
        "read_file",
        "get_issue",
        "get_pr",
        "get_pr_diff",
        "create_branch",
        "create_commit",
        "create_pull_request",
    } <= names


async def test_live_tool_call_against_public_repo() -> None:
    """Call ``list_repo_files`` on a public repo — no token required."""
    try:
        adapter = GitHubMCPTools()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))

    async with adapter as bound:
        listing_tool = next(t for t in bound.tools if t.name == "list_repo_files")
        raw = await listing_tool.ainvoke({"owner": "octocat", "name": "Hello-World"})
    assert "README" in raw


class FakeModel:
    """Minimal stand-in for ``BaseChatModel`` serving canned responses in order
    (the graph calls it once per stage; the last scripted response repeats)."""

    def __init__(self, *contents: str) -> None:
        self._contents = contents or ("{}",)
        self._index = 0
        self.calls: list[list[Any]] = []

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.calls.append(messages)
        content = self._contents[min(self._index, len(self._contents) - 1)]
        self._index += 1
        return AIMessage(content=content)


def make_tool(
    name: str, body: str | Exception, calls: list[dict[str, Any]] | None = None
) -> StructuredTool:
    """A LangChain tool returning canned text or raising a given error."""

    async def invoke(**kwargs: Any) -> str:
        if calls is not None:
            calls.append({"tool": name, **kwargs})
        if isinstance(body, Exception):
            raise body
        return body

    return StructuredTool.from_function(
        coroutine=invoke, name=name, description=name, infer_schema=False
    )


ISSUE_JSON = json.dumps(
    {
        "number": 1,
        "title": "App crashes on empty input",
        "state": "open",
        "reporter": "octo",
        "labels": ["bug"],
        "body": "Repro: run the app with no arguments.",
        "comments": [],
    }
)
LISTING_JSON = json.dumps([{"kind": "file", "name": "README.md", "path": "README.md", "size": 120}])
PLAIN_TOOLS = [
    make_tool("get_issue", ISSUE_JSON),
    make_tool("list_repo_files", LISTING_JSON),
    make_tool("read_file", "# AgentOS\ndemo repo\n"),
]
MODEL_JSON = (
    '{"investigation": "App crashes when run without arguments.",'
    ' "root_cause_hypothesis": "missing null check in the entrypoint"}'
)
DESIGN_JSON = json.dumps(
    [
        {
            "path": "entrypoint.py",
            "content": 'def main():\n    args = sys.argv[1:] or ["default"]\n',
            "explanation": "default arguments so the app no longer crashes",
        },
        {
            "path": "legacy.py",
            "delete": True,
            "content": "",
            "explanation": "dead module removed",
        },
    ]
)


async def test_investigate_uses_tools_and_model() -> None:
    graph = create_agent_graph(model=FakeModel(MODEL_JSON, "garbage"), tools=PLAIN_TOOLS)
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["investigation"] == "App crashes when run without arguments."
    assert final["root_cause_hypothesis"] == "missing null check in the entrypoint"
    assert [event.kind for event in final["events"]] == [
        "target",
        "context",
        "hypothesis",
        "error",
        "error",
    ]
    assert final["events"][0].detail.startswith("issue #1 loaded")
    assert final["events"][1].detail == "read 1 file(s)"


def test_default_github_mcp_command_prefers_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_MCP_COMMAND", "/usr/bin/gh-mcp")
    command, args = default_github_mcp_command()
    assert command == "/usr/bin/gh-mcp"
    assert args == []


def test_default_github_mcp_command_falls_back_to_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_MCP_COMMAND", raising=False)

    def fake_which(name: str) -> str | None:
        return "/opt/bin/github-mcp-server" if name == "github-mcp-server" else None

    monkeypatch.setattr("agentos.agent.mcp_adapter.shutil.which", fake_which)
    command, args = default_github_mcp_command()
    assert command == "/opt/bin/github-mcp-server"
    assert args == []


async def test_investigate_uses_rag_retrieval_context() -> None:
    calls: list[tuple[str, str, int]] = []

    async def fake_retrieve(repo: str, query: str, top_k: int) -> list[ContextDoc]:
        calls.append((repo, query, top_k))
        return [
            ContextDoc(
                path="src/app.py",
                content="def run():\n    return None  # null crash",
                chunk_index=3,
                score=0.91,
            ),
            ContextDoc(
                path="src/app.py", content="args = sys.argv or []", chunk_index=4, score=0.87
            ),
        ]

    model = FakeModel(MODEL_JSON, "garbage")
    graph = create_agent_graph(model=model, tools=PLAIN_TOOLS, retrieval=fake_retrieve)
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert calls == [
        (
            "octocat/Hello-World",
            "App crashes on empty input Repro: run the app with no arguments.",
            MAX_RAG_CONTEXT,
        )
    ]
    rag_event = next(e for e in final["events"] if e.kind == "rag")
    assert rag_event.detail == "retrieved 2 relevant chunk(s)"
    prompt = next(m for m in model.calls[0] if isinstance(m, HumanMessage)).content
    assert "def run():" in prompt
    assert "[chunk 3, relevance 0.91]" in prompt
    assert len(final["context"]) == 3
    assert final["context"][0].path == "src/app.py"
    assert final["context"][0].score == 0.91
    assert final["context"][1].chunk_index == 4
    assert [event.kind for event in final["events"]] == [
        "target",
        "rag",
        "context",
        "hypothesis",
        "error",
        "error",
    ]


async def test_investigate_uses_issue_title_and_body_as_retrieval_query() -> None:
    queries: list[str] = []

    async def fake_retrieve(repo: str, query: str, top_k: int) -> list[ContextDoc]:
        queries.append(query)
        return []

    target = RunTarget(
        repo_full_name="octocat/Hello-World",
        kind="issue",
        number=7,
        title="App crashes on empty input",
    )
    graph = create_agent_graph(
        model=FakeModel(MODEL_JSON, "garbage"), tools=PLAIN_TOOLS, retrieval=fake_retrieve
    )
    final = await graph.ainvoke(dict(ISSUE_INPUT, target=target))
    assert queries == ["App crashes on empty input Repro: run the app with no arguments."]
    assert final["investigation"] == "App crashes when run without arguments."
    rag_event = next(e for e in final["events"] if e.kind == "rag")
    assert rag_event.detail == "no relevant chunks found"


async def test_investigate_degrades_when_retrieval_fails() -> None:
    async def broken_retrieve(repo: str, query: str, top_k: int) -> list[ContextDoc]:
        raise RuntimeError("db down")

    model = FakeModel(MODEL_JSON, "garbage")
    graph = create_agent_graph(model=model, tools=PLAIN_TOOLS, retrieval=broken_retrieve)
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["investigation"] == "App crashes when run without arguments."
    assert final["root_cause_hypothesis"] == "missing null check in the entrypoint"
    rag_event = next(e for e in final["events"] if e.kind == "rag")
    assert rag_event.detail == "retrieval failed: db down"
    # repo context is RAG-only now — nothing landed, so no repo content
    assert final["context"] == []
    prompt = next(m for m in model.calls[0] if isinstance(m, HumanMessage)).content
    assert "def run():" not in prompt
    assert "README.md" not in prompt


async def test_investigate_indexes_repo_when_retrieval_misses() -> None:
    calls: list[tuple[str, str, int]] = []
    indexed: list[str] = []
    indexed_flag: list[bool] = [False]

    async def fake_retrieve(repo: str, query: str, top_k: int) -> list[ContextDoc]:
        calls.append((repo, query, top_k))
        if indexed_flag[0]:
            return [
                ContextDoc(
                    path="src/app.py",
                    content="args = sys.argv or []  # null-safe entry",
                    chunk_index=4,
                    score=0.87,
                )
            ]
        return []

    async def fake_indexer(repo: str) -> IndexSummary:
        indexed.append(repo)
        indexed_flag[0] = True
        return IndexSummary(
            repo_full_name=repo, files_indexed=3, chunks=12, chars=1200, chunks_embedded=12
        )

    model = FakeModel(MODEL_JSON, "garbage")
    graph = create_agent_graph(
        model=model, tools=PLAIN_TOOLS, retrieval=fake_retrieve, indexer=fake_indexer
    )
    final = await graph.ainvoke(dict(ISSUE_INPUT))

    assert indexed == ["octocat/Hello-World"]  # indexed once, before the re-query
    assert len(calls) == 2  # first miss, then a hit after indexing
    assert any(e.kind == "indexed" and e.detail.startswith("indexed 3 file(s)") for e in final["events"])
    assert {doc.path for doc in final["context"]} >= {"src/app.py"}
    prompt = next(m for m in model.calls[0] if isinstance(m, HumanMessage)).content
    assert "args = sys.argv or []" in prompt


async def test_investigate_degrades_when_auto_indexing_fails() -> None:
    async def fake_retrieve(repo: str, query: str, top_k: int) -> list[ContextDoc]:
        return []

    async def broken_indexer(repo: str) -> IndexSummary:
        raise RuntimeError("embedding api down")

    graph = create_agent_graph(
        model=FakeModel(MODEL_JSON, "garbage"),
        tools=PLAIN_TOOLS,
        retrieval=fake_retrieve,
        indexer=broken_indexer,
    )
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    rag_detail = next(e.detail for e in final["events"] if e.kind == "rag")
    assert rag_detail == "auto-indexing failed: embedding api down"
    # no repo context to fall back to — the run still produces an investigation
    assert final["context"] == []
    assert final["investigation"] == "App crashes when run without arguments."


async def test_design_produces_structured_changes() -> None:
    graph = create_agent_graph(model=FakeModel(MODEL_JSON, DESIGN_JSON), tools=PLAIN_TOOLS)
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["proposed_changes"] == [
        FileChange(
            path="entrypoint.py",
            content='def main():\n    args = sys.argv[1:] or ["default"]\n',
            delete=False,
            explanation="default arguments so the app no longer crashes",
        ),
        FileChange(path="legacy.py", content="", delete=True, explanation="dead module removed"),
    ]
    design = next(e for e in final["events"] if e.stage == "design")
    assert design.kind == "design"
    assert design.detail == "2 change(s): entrypoint.py, legacy.py"


def test_reject_corrupt_changes_detects_placeholders() -> None:
    from agentos.agent.graph import _reject_corrupt_changes

    corrupt = [
        FileChange(
            path="src/app.tsx",
            content="import x from 'y'\n// ... rest of modes\n<div className=\"...\">",
        )
    ]
    assert _reject_corrupt_changes(corrupt) == ["src/app.tsx"]
    healthy = [FileChange(path="src/app.tsx", content="export const a = 1;\n")]
    assert _reject_corrupt_changes(healthy) == []


async def test_design_invalid_patch_falls_back() -> None:
    graph = create_agent_graph(
        model=FakeModel(MODEL_JSON, "certainly not an array"),
        tools=PLAIN_TOOLS,
    )
    visited: list[str] = []
    async for update in graph.astream(dict(ISSUE_INPUT), stream_mode="updates"):
        visited.extend(update.keys())
    assert visited == ["investigate", "design", "apply"]
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["proposed_changes"] == []
    assert final["applied_branch"] is None
    assert "error" in [event.kind for event in final["events"]]


async def test_design_retries_with_reduced_context_after_json_failure() -> None:
    model = FakeModel(MODEL_JSON, "no JSON here", DESIGN_JSON)
    graph = create_agent_graph(model=model, tools=PLAIN_TOOLS)
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["proposed_changes"] == [
        FileChange(
            path="entrypoint.py",
            content='def main():\n    args = sys.argv[1:] or ["default"]\n',
            delete=False,
            explanation="default arguments so the app no longer crashes",
        ),
        FileChange(
            path="legacy.py", content="", delete=True, explanation="dead module removed"
        ),
    ]
    design_errors = [e for e in final["events"] if e.stage == "design" and e.kind == "error"]
    assert len(design_errors) == 1
    assert "model output contained no JSON object or array" in design_errors[0].detail
    assert any(
        e.stage == "design" and e.detail == "retry succeeded with reduced context"
        for e in final["events"]
    )


PR_JSON = json.dumps(
    {
        "number": 42,
        "url": "https://github.com/octocat/Hello-World/pull/42",
    }
)
PR_TOOL = make_tool(
    "create_pull_request",
    PR_JSON,
)

# What the MCP server returns today: flattened string refs + author login
# (not the raw GitHub API dict shape with base/head objects).
RICH_PR_JSON = json.dumps(
    {
        "number": 42,
        "url": "https://github.com/octocat/Hello-World/pull/42",
        "title": "fix: default args so the app stops crashing",
        "body": "## What\nAdds default CLI args.\n\n## Files\n- entrypoint.py",
        "state": "open",
        "draft": False,
        "author": "octocat",
        "created_at": "2026-08-28T00:36:10Z",
        "base": "main",
        "head": "fix/issue-1-1787877360",
        "changed_files": 1,
        "additions": 3,
        "deletions": 1,
    }
)

PR_SUMMARY_JSON = json.dumps(
    {
        "title": "fix: default args so the app stops crashing",
        "body": "## What\nAdds default CLI args.\n\n## Files\n- entrypoint.py",
    }
)


async def test_full_run_creates_branch_and_commit() -> None:
    branch_calls: list[dict[str, Any]] = []
    commit_calls: list[dict[str, Any]] = []
    pr_calls: list[dict[str, Any]] = []
    tools = [
        *PLAIN_TOOLS,
        make_tool(
            "create_branch",
            json.dumps({"ref": "fix/issue-1", "sha": "ABC123", "base_sha": "XYZ"}),
            branch_calls,
        ),
        make_tool(
            "create_commit",
            json.dumps({"branch": "fix/issue-1", "commit_sha": "c0ffee123456"}),
            commit_calls,
        ),
        make_tool("create_pull_request", PR_JSON, pr_calls),
    ]
    graph = create_agent_graph(
        model=FakeModel(MODEL_JSON, DESIGN_JSON, PR_SUMMARY_JSON), tools=tools, token="ght_test"
    )
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["applied_branch"].startswith("fix/issue-1-")
    assert final["pr_url"] == "https://github.com/octocat/Hello-World/pull/42"
    assert [event.kind for event in final["events"]] == [
        "target",
        "context",
        "hypothesis",
        "design",
        "branch",
        "commit",
        "summary",
        "pr",
    ]
    assert final["events"][-1].detail == (
        "opened PR #42: https://github.com/octocat/Hello-World/pull/42"
    )
    assert branch_calls[0]["tool"] == "create_branch"
    assert branch_calls[0]["owner"] == "octocat"
    assert branch_calls[0]["name"] == "Hello-World"
    assert branch_calls[0]["base_branch"] == "main"
    assert branch_calls[0]["new_branch"].startswith("fix/issue-1-")
    assert commit_calls[0]["branch"] == branch_calls[0]["new_branch"]
    assert commit_calls[0]["message"].startswith("AgentOS: fix issue #1")
    assert commit_calls[0]["changes"] == [
        {
            "path": "entrypoint.py",
            "content": 'def main():\n    args = sys.argv[1:] or ["default"]\n',
            "edits": [],
            "delete": False,
        },
        {"path": "legacy.py", "content": "", "edits": [], "delete": True},
    ]
    assert pr_calls[0]["tool"] == "create_pull_request"
    assert pr_calls[0]["owner"] == "octocat"
    assert pr_calls[0]["name"] == "Hello-World"
    assert pr_calls[0]["title"] == "fix: default args so the app stops crashing"
    assert pr_calls[0]["head"].startswith("fix/issue-1-")
    assert pr_calls[0]["base"] == "main"
    assert pr_calls[0]["body"] == "## What\nAdds default CLI args.\n\n## Files\n- entrypoint.py"


async def test_pr_captures_flattened_metadata_without_raising() -> None:
    tools = [
        *PLAIN_TOOLS,
        make_tool("create_branch", json.dumps({"sha": "ABC123"})),
        make_tool("create_commit", json.dumps({"commit_sha": "c0ffee123456"})),
        make_tool("create_pull_request", RICH_PR_JSON),
    ]
    graph = create_agent_graph(
        model=FakeModel(MODEL_JSON, DESIGN_JSON, PR_SUMMARY_JSON),
        tools=tools,
        token="ght_test",
    )
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    # the run must complete — the PR is already open, metadata quirks
    # (string base/head) must never fail the run after PR creation
    assert final["pr_url"] == "https://github.com/octocat/Hello-World/pull/42"
    assert final["pr"] == {
        "number": "42",
        "url": "https://github.com/octocat/Hello-World/pull/42",
        "title": "fix: default args so the app stops crashing",
        "body": "## What\nAdds default CLI args.\n\n## Files\n- entrypoint.py",
        "state": "open",
        "author": "octocat",
        "created_at": "2026-08-28T00:36:10Z",
        "base": "main",
        "head": "fix/issue-1-1787877360",
        "changed_files": 1,
        "additions": 3,
        "deletions": 1,
    }


async def test_pr_opens_with_fallback_when_summary_fails() -> None:
    tools = [
        *PLAIN_TOOLS,
        make_tool("create_branch", json.dumps({"sha": "ABC123"})),
        make_tool("create_commit", json.dumps({"commit_sha": "c0ffee123456"})),
        make_tool("create_pull_request", PR_JSON),
    ]
    graph = create_agent_graph(
        model=FakeModel(MODEL_JSON, DESIGN_JSON, "this is not json"),
        tools=tools,
        token="ght_test",
    )
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["pr_url"] == "https://github.com/octocat/Hello-World/pull/42"
    summary = next(e for e in final["events"] if e.kind == "summary")
    assert summary.detail.startswith("Fix issue #1: App crashes when run without arguments.")


async def test_pr_missing_tool_ends_with_error() -> None:
    tools = [
        *PLAIN_TOOLS,
        make_tool("create_branch", json.dumps({"sha": "ABC123"})),
        make_tool("create_commit", json.dumps({"commit_sha": "c0ffee123456"})),
    ]
    graph = create_agent_graph(
        model=FakeModel(MODEL_JSON, DESIGN_JSON), tools=tools, token="ght_test"
    )
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["applied_branch"].startswith("fix/issue-1-")
    assert final["pr_url"] is None
    assert final["events"][-1].kind == "error"
    assert final["events"][-1].detail.startswith("MCP create_pull_request tool not available")


async def test_full_run_uses_custom_base_branch() -> None:
    branch_calls: list[dict[str, Any]] = []
    tools = [
        *PLAIN_TOOLS,
        make_tool("create_branch", json.dumps({"sha": "ABC123"}), branch_calls),
        make_tool("create_commit", json.dumps({"commit_sha": "c0ffee123456"})),
        make_tool("create_pull_request", PR_JSON),
    ]
    graph = create_agent_graph(
        model=FakeModel(MODEL_JSON, DESIGN_JSON),
        tools=tools,
        token="ght_test",
    )
    target = RunTarget(
        repo_full_name="octocat/Hello-World", kind="issue", number=9, base_branch="trunk"
    )
    final = await graph.ainvoke(dict(ISSUE_INPUT, target=target))
    assert final["applied_branch"].startswith("fix/issue-9-")
    assert final["pr_url"] is not None
    assert branch_calls[0]["base_branch"] == "trunk"


async def test_apply_without_token_stops_before_pr() -> None:
    graph = create_agent_graph(
        model=FakeModel(MODEL_JSON, DESIGN_JSON), tools=PLAIN_TOOLS, token=None
    )
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["applied_branch"] is None
    assert final["events"][-1].stage == "apply"
    assert [event.kind for event in final["events"]] == [
        "target",
        "context",
        "hypothesis",
        "design",
        "error",
    ]
    assert final["events"][-1].detail.startswith("GITHUB_TOKEN required")


async def test_apply_mcp_error_stops_run() -> None:
    tools = [
        *PLAIN_TOOLS,
        make_tool("create_branch", ToolException("boom")),
        make_tool("create_commit", json.dumps({"commit_sha": "c0ffee123456"})),
    ]
    graph = create_agent_graph(
        model=FakeModel(MODEL_JSON, DESIGN_JSON), tools=tools, token="ght_test"
    )
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["applied_branch"] is None
    assert final["events"][-1].kind == "error"
    assert final["events"][-1].detail.startswith("apply failed")


async def test_investigate_tool_error_falls_back_gracefully() -> None:
    tools = [make_tool("get_issue", ToolException("boom"))]
    graph = create_agent_graph(model=FakeModel(MODEL_JSON), tools=tools)
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["root_cause_hypothesis"] == ""
    assert "Could not load issue #1" in final["investigation"]
    assert final["events"][0].kind == "error"


async def test_investigate_invalid_model_json_falls_back() -> None:
    graph = create_agent_graph(model=FakeModel("definitely not json"), tools=PLAIN_TOOLS)
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["root_cause_hypothesis"] == ""
    assert "Analyzed" in final["investigation"]
    assert "error" in [event.kind for event in final["events"]]
