"""The fix-agent LangGraph: a linear pipeline of four staged nodes.

Topology is fixed here (``investigate -> design -> apply -> pr``). The node
bodies are deterministic stubs for now — they establish the state contract,
typed streaming, and trace events; the LLM/tool work lands in Stage 4
follow-ups (investigate: issue reads + RAG; design/apply: patch generation
and MCP write calls; pr: pull request + evaluation hook).
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from agentos.agent.state import AgentState, RunEvent
from agentos.core.config import get_settings


def create_agent_llm() -> BaseChatModel:
    """OpenRouter-bound chat model (``ChatOpenAI`` is OpenAI-compatible)."""
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=SecretStr(settings.openrouter_api_key),
        model=settings.openrouter_model,
    )


async def investigate_node(state: AgentState) -> dict[str, Any]:
    """Load and summarize the target issue/PR (LLM + RAG intrinsic in Stage 4.2)."""
    target = state["target"]
    return {
        "investigation": f"{target.kind} #{target.number} in {target.repo_full_name}",
        "root_cause_hypothesis": "",
        "events": [
            RunEvent(
                stage="investigate",
                kind="investigation",
                detail=f"target {target.kind} #{target.number} loaded",
            )
        ],
    }


async def design_node(state: AgentState) -> dict[str, Any]:
    """Produce the structured per-file fix (LLM + patch schema in Stage 4.3)."""
    return {
        "proposed_changes": [],
        "events": [RunEvent(stage="design", kind="design", detail="fix design pending")],
    }


async def apply_node(state: AgentState) -> dict[str, Any]:
    """Create the branch and commit the changes through MCP (Stage 4.3)."""
    return {
        "applied_branch": None,
        "events": [RunEvent(stage="apply", kind="apply", detail="changes not yet applied")],
    }


async def pr_node(state: AgentState) -> dict[str, Any]:
    """Open the pull request and finish the run (Stage 4.4)."""
    return {
        "pr_url": None,
        "events": [RunEvent(stage="pr", kind="pr", detail="pull request pending")],
    }


def create_agent_graph() -> CompiledStateGraph:
    """Build and compile the fix-agent pipeline."""
    builder = StateGraph(AgentState)
    builder.add_node("investigate", investigate_node)
    builder.add_node("design", design_node)
    builder.add_node("apply", apply_node)
    builder.add_node("pr", pr_node)
    builder.add_edge(START, "investigate")
    builder.add_edge("investigate", "design")
    builder.add_edge("design", "apply")
    builder.add_edge("apply", "pr")
    builder.add_edge("pr", END)
    return builder.compile()
