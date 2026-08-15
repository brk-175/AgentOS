"""The fix-agent LangGraph: a linear pipeline of four staged nodes.

Topology is fixed here (``investigate -> design -> apply -> pr``).
``investigate`` is implemented: it loads the target issue/PR through the MCP
tools, gathers a little repo context, and asks the LLM for the investigation
summary and root-cause hypothesis. ``design``/``apply``/``pr`` remain
deterministic stubs and gain their LLM/tool work in later steps.
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from agentos.agent.state import AgentState, RunEvent
from agentos.core.config import get_settings

MAX_CONTEXT_FILES = 2
MAX_FILE_CHARS = 3000
MAX_PROMPT_CONTEXT = 12_000
_INVESTIGATE_PROMPT = """You are the investigation stage of a GitHub code-fix agent.
You receive an issue/PR and some repository context. Determine what the problem
is and where it likely comes from.

Return STRICT JSON with exactly two keys:
- "investigation": a concise summary of the problem in the issue/PR
- "root_cause_hypothesis": the most likely root cause, naming files/line areas

No prose outside the JSON object."""

_PATCHABLE_NAME = re.compile(r"(?i)^(readme|contribut|license)")


def create_agent_llm() -> BaseChatModel:
    """OpenRouter-bound chat model (``ChatOpenAI`` is OpenAI-compatible)."""
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=SecretStr(settings.openrouter_api_key),
        model=settings.openrouter_model,
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from a model response (fence-tolerant)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("model output contained no JSON object")
    return json.loads(cleaned[start : end + 1])


def _tool(tools: Sequence[BaseTool], name: str) -> BaseTool | None:
    return next((tool for tool in tools if tool.name == name), None)


async def _fetch_target(tools: Sequence[BaseTool], state: AgentState) -> dict[str, Any]:
    """Load issue/PR details through the MCP tools; raises on failure."""
    target = state["target"]
    owner, _, repo = target.repo_full_name.partition("/")
    tool_name = "get_issue" if target.kind == "issue" else "get_pr"
    tool = _tool(tools, tool_name)
    if tool is None:
        raise ValueError(f"MCP tool '{tool_name}' not found among provided tools")
    payload = json.loads(
        await tool.ainvoke({"owner": owner, "name": repo, "number": target.number})
    )
    return payload


async def _gather_context(tools: Sequence[BaseTool], state: AgentState) -> tuple[list[str], int]:
    """Read a couple of promising files from the repo root (bounded)."""
    target = state["target"]
    owner, _, repo = target.repo_full_name.partition("/")
    listing_tool = _tool(tools, "list_repo_files")
    read_tool = _tool(tools, "read_file")
    if listing_tool is None or read_tool is None:
        return [], 0
    listing = json.loads(await listing_tool.ainvoke({"owner": owner, "name": repo, "path": ""}))
    entries = sorted(listing, key=lambda e: not bool(_PATCHABLE_NAME.match(e["name"])))
    picked = [
        entry["path"]
        for entry in entries
        if entry.get("kind") == "file" and entry.get("size", 0) <= MAX_FILE_CHARS
    ][:MAX_CONTEXT_FILES]
    contents: list[str] = []
    for path in picked:
        contents.append(
            f"### {path}\n{await read_tool.ainvoke({'owner': owner, 'name': repo, 'path': path})}"
        )
    return contents, len(picked)


async def investigate_node(
    state: AgentState,
    *,
    model: BaseChatModel | None,
    tools: Sequence[BaseTool],
) -> dict[str, Any]:
    """Investigate the target: load it via MCP, gather context, LLM analysis."""
    target = state["target"]
    events: list[RunEvent] = []
    if model is None and not tools:
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
    context_parts: list[str] = []
    files_read = 0
    try:
        details = await _fetch_target(tools, state)
        events.append(
            RunEvent(
                stage="investigate",
                kind="target",
                detail=f"{target.kind} #{target.number} loaded: {details.get('title', '')[:80]}",
            )
        )
        context_parts.append(
            f"### {target.kind.upper()} #{target.number}\n{json.dumps(details, indent=2)[:6000]}"
        )
    except Exception as exc:
        events.append(
            RunEvent(stage="investigate", kind="error", detail=f"target fetch failed: {exc}")
        )
        return {
            "investigation": f"Could not load {target.kind} #{target.number} of {target.repo_full_name}",
            "root_cause_hypothesis": "",
            "events": events,
        }

    try:
        context, files_read = await _gather_context(tools, state)
    except Exception as exc:
        events.append(
            RunEvent(stage="investigate", kind="error", detail=f"context fetch failed: {exc}")
        )
    if files_read:
        events.append(
            RunEvent(stage="investigate", kind="context", detail=f"read {files_read} file(s)")
        )
    context_parts.extend(context)

    if model is None:
        return {
            "investigation": f"{target.kind} #{target.number} loaded from {target.repo_full_name}"
            + (f": {details.get('title', '')}" if details.get("title") else ""),
            "root_cause_hypothesis": "",
            "events": events,
        }

    prompt = (
        f"Target: {target.repo_full_name} {target.kind} #{target.number}"
        f" ({target.title or 'no title'})\n\n" + "\n\n".join(context_parts)[:MAX_PROMPT_CONTEXT]
    )
    try:
        response = await model.ainvoke(
            [SystemMessage(content=_INVESTIGATE_PROMPT), HumanMessage(content=prompt)]
        )
        parsed = _extract_json(str(response.content))
        investigation = str(parsed.get("investigation") or "").strip()
        hypothesis = str(parsed.get("root_cause_hypothesis") or "").strip()
        if not investigation:
            raise ValueError("model returned an empty investigation")
    except Exception as exc:
        events.append(RunEvent(stage="investigate", kind="error", detail=f"analysis failed: {exc}"))
        return {
            "investigation": f"Analyzed {target.kind} #{target.number} of {target.repo_full_name}"
            + f": {details.get('title', '').strip()}",
            "root_cause_hypothesis": "",
            "events": events,
        }
    events.append(
        RunEvent(stage="investigate", kind="hypothesis", detail=hypothesis[:140] or "no hypothesis")
    )
    return {
        "investigation": investigation,
        "root_cause_hypothesis": hypothesis,
        "events": events,
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


def _investigate_with(
    model: BaseChatModel | None, tools: Sequence[BaseTool]
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Closure binding model+tools into the investigate node (LangGraph needs
    a real async callable — a sync wrapper returning a coroutine is not awaited)."""

    async def node(state: AgentState) -> dict[str, Any]:
        return await investigate_node(state, model=model, tools=tools)

    return node


def create_agent_graph(
    model: BaseChatModel | None = None,
    tools: Sequence[BaseTool] = (),
) -> CompiledStateGraph:
    """Build and compile the fix-agent pipeline.

    ``model`` and ``tools`` are bound into the investigate node; without
    them the graph runs in deterministic stub mode (useful for tests/demo).
    """
    builder = StateGraph(AgentState)
    builder.add_node("investigate", cast(Any, _investigate_with(model, tools)))
    builder.add_node("design", design_node)
    builder.add_node("apply", apply_node)
    builder.add_node("pr", pr_node)
    builder.add_edge(START, "investigate")
    builder.add_edge("investigate", "design")
    builder.add_edge("design", "apply")
    builder.add_edge("apply", "pr")
    builder.add_edge("pr", END)
    return builder.compile()
