"""Tests for the fix-agent pipeline and the MCP tool adapter."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
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


async def test_live_github_mcp_tools_exposes_all_five_tools() -> None:
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
