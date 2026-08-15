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
from agentos.agent.state import RunTarget

ISSUE_INPUT: dict[str, Any] = {
    "target": RunTarget(repo_full_name="octocat/Hello-World", kind="issue", number=1),
    "messages": [],
    "events": [],
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
    """Minimal stand-in for ``BaseChatModel`` that returns canned JSON."""

    def __init__(self, content: str) -> None:
        self._content = content

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        return AIMessage(content=self._content)


def make_tool(name: str, body: str | Exception) -> StructuredTool:
    """A LangChain tool returning canned text or raising a given error."""

    async def invoke(**kwargs: Any) -> str:
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


async def test_investigate_uses_tools_and_model() -> None:
    graph = create_agent_graph(model=FakeModel(MODEL_JSON), tools=PLAIN_TOOLS)
    final = await graph.ainvoke(dict(ISSUE_INPUT))
    assert final["investigation"] == "App crashes when run without arguments."
    assert final["root_cause_hypothesis"] == "missing null check in the entrypoint"
    assert [event.kind for event in final["events"]] == [
        "target",
        "context",
        "hypothesis",
        "design",
        "apply",
        "pr",
    ]
    assert final["events"][0].detail.startswith("issue #1 loaded")
    assert final["events"][1].detail == "read 1 file(s)"


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
