"""Tests for the fix-agent pipeline and the MCP tool adapter."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool, ToolException
from langchain_openai import ChatOpenAI

from agentos.agent.graph import create_agent_graph, create_agent_llm
from agentos.agent.mcp_adapter import GitHubMCPTools, json_schema_to_model
from agentos.agent.state import FileChange, RunTarget

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

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
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


async def test_full_run_creates_branch_and_commit() -> None:
    branch_calls: list[dict[str, Any]] = []
    commit_calls: list[dict[str, Any]] = []
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
    ]
    graph = create_agent_graph(
        model=FakeModel(MODEL_JSON, DESIGN_JSON), tools=tools, token="ght_test"
    )
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["applied_branch"] == "fix/issue-1"
    assert [event.kind for event in final["events"]] == [
        "target",
        "context",
        "hypothesis",
        "design",
        "branch",
        "commit",
        "pr",
    ]
    assert branch_calls[0] == {
        "tool": "create_branch",
        "owner": "octocat",
        "name": "Hello-World",
        "base_branch": "main",
        "new_branch": "fix/issue-1",
    }
    assert commit_calls[0]["branch"] == "fix/issue-1"
    assert commit_calls[0]["message"].startswith("AgentOS: fix issue #1")
    assert commit_calls[0]["changes"] == [
        {
            "path": "entrypoint.py",
            "content": 'def main():\n    args = sys.argv[1:] or ["default"]\n',
            "delete": False,
        },
        {"path": "legacy.py", "content": "", "delete": True},
    ]


async def test_full_run_uses_custom_base_branch() -> None:
    branch_calls: list[dict[str, Any]] = []
    tools = [
        *PLAIN_TOOLS,
        make_tool("create_branch", json.dumps({"sha": "ABC123"}), branch_calls),
        make_tool("create_commit", json.dumps({"commit_sha": "c0ffee123456"})),
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
    assert final["applied_branch"] == "fix/issue-9"
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
