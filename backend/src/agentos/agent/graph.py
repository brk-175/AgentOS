"""The fix-agent LangGraph: a staged pipeline of four nodes.

Topology: ``investigate -> design -> apply -> pr`` with a conditional edge
after ``apply`` (the run ends early when nothing could be applied).
``investigate`` loads the target through the MCP tools, gathers repo context
and asks the LLM for a summary + root-cause hypothesis; ``design`` asks the
LLM for the structured per-file changes; ``apply`` creates the branch and
commits the changes through MCP. ``pr`` remains a stub and gains its work in
Stage 4.4.
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

from agentos.agent.state import AgentState, ContextDoc, FileChange, RunEvent
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

_DESIGN_PROMPT = """You are the design stage of a GitHub code-fix agent.
Given the target, the investigation and the repo context, produce the exact
file changes that fix the problem and nothing else.

Return STRICT JSON: an ARRAY of change objects. Each object has:
- "path": repository-relative file path (required)
- "content": the FULL new file content (for edits and new files); ignored when
  "delete" is true
- "delete": optional boolean — true to delete the file (content must be "")
- "explanation": one short sentence about why this change fixes the issue

Never truncate a file: "content" must always be the complete file.
No prose outside the JSON array."""

_MAX_COMMIT_MESSAGE = 120


def create_agent_llm() -> BaseChatModel:
    """OpenRouter-bound chat model (``ChatOpenAI`` is OpenAI-compatible)."""
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.openrouter_base_url,
        api_key=SecretStr(settings.openrouter_api_key),
        model=settings.openrouter_model,
    )


def _extract_json(text: str) -> Any:
    """Extract the first JSON object/array from a model response
    (fence-tolerant)."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    positions = [index for index in (cleaned.find("{"), cleaned.find("[")) if index != -1]
    if not positions:
        raise ValueError("model output contained no JSON object or array")
    start = min(positions)
    closer = "}" if cleaned[start] == "{" else "]"
    end = cleaned.rfind(closer)
    if end == -1:
        raise ValueError("model output ended before closing its JSON block")
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


async def _gather_context(
    tools: Sequence[BaseTool], state: AgentState
) -> tuple[list[ContextDoc], int]:
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
    docs: list[ContextDoc] = []
    for path in picked:
        content = await read_tool.ainvoke({"owner": owner, "name": repo, "path": path})
        docs.append(ContextDoc(path=path, content=content))
    return docs, len(picked)


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
    context_docs: list[ContextDoc] = []
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
    context_docs.extend(context)
    context_parts.extend(f"### {doc.path}\n{doc.content}" for doc in context_docs)

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


async def design_node(
    state: AgentState,
    *,
    model: BaseChatModel | None,
) -> dict[str, Any]:
    """Design the fix: ask the LLM for exact per-file changes (strict JSON)."""
    target = state["target"]
    events: list[RunEvent] = []
    if model is None:
        return {
            "proposed_changes": [],
            "events": [RunEvent(stage="design", kind="design", detail="fix design pending")],
        }

    context_block = (
        "\n\n".join(f"### {doc.path}\n{doc.content}" for doc in state["context"])
        or "(no repo context was gathered)"
    )
    prompt = (
        f"Target: {target.repo_full_name} {target.kind} #{target.number}"
        f" ({target.title or 'no title'})\n"
        f"Investigation: {state['investigation'] or '(empty)'}\n"
        f"Root-cause hypothesis: {state['root_cause_hypothesis'] or '(none)'}\n\n"
        f"Repo context:\n{context_block}"
    )[:MAX_PROMPT_CONTEXT]
    try:
        response = await model.ainvoke(
            [SystemMessage(content=_DESIGN_PROMPT), HumanMessage(content=prompt)]
        )
        raw = _extract_json(str(response.content))
        if not isinstance(raw, list):
            raise ValueError("model patch must be a JSON array")
        changes: list[FileChange] = []
        for entry in raw:
            if not isinstance(entry, dict) or not entry.get("path"):
                continue
            changes.append(
                FileChange(
                    path=str(entry["path"]),
                    content=str(entry.get("content") or ""),
                    delete=bool(entry.get("delete", False)),
                    explanation=str(entry.get("explanation") or ""),
                )
            )
        if not changes:
            raise ValueError("no valid change entries in model patch")
    except Exception as exc:
        events.append(RunEvent(stage="design", kind="error", detail=f"design failed: {exc}"))
        return {"proposed_changes": [], "events": events}

    paths = ", ".join(c.path for c in changes[:5])
    events.append(
        RunEvent(
            stage="design",
            kind="design",
            detail=f"{len(changes)} change(s): {paths}" + ("…" if len(changes) > 5 else ""),
        )
    )
    return {"proposed_changes": changes, "events": events}


async def apply_node(
    state: AgentState,
    *,
    tools: Sequence[BaseTool],
    token: str | None,
) -> dict[str, Any]:
    """Apply the fix: create the branch, then commit the changes via MCP."""
    target = state["target"]
    changes = state["proposed_changes"]
    events: list[RunEvent] = []
    if token is None and not tools:
        return {
            "applied_branch": None,
            "events": [RunEvent(stage="apply", kind="apply", detail="changes not yet applied")],
        }
    if not changes:
        events.append(
            RunEvent(stage="apply", kind="error", detail="no changes to apply (design failed?)")
        )
        return {"applied_branch": None, "events": events}
    if token is None:
        events.append(
            RunEvent(stage="apply", kind="error", detail="GITHUB_TOKEN required to apply changes")
        )
        return {"applied_branch": None, "events": events}

    branch_tool = _tool(tools, "create_branch")
    commit_tool = _tool(tools, "create_commit")
    if branch_tool is None or commit_tool is None:
        events.append(RunEvent(stage="apply", kind="error", detail="MCP write tools not available"))
        return {"applied_branch": None, "events": events}

    owner, _, repo = target.repo_full_name.partition("/")
    branch = f"fix/{target.kind}-{target.number}"
    message = f"AgentOS: fix {target.kind} #{target.number}"
    if target.title:
        message = f"{message}: {target.title}"
    message = message[:_MAX_COMMIT_MESSAGE]
    payload = [change.model_dump(include={"path", "content", "delete"}) for change in changes]
    try:
        branch_result = json.loads(
            await branch_tool.ainvoke(
                {
                    "owner": owner,
                    "name": repo,
                    "base_branch": target.base_branch,
                    "new_branch": branch,
                }
            )
        )
        events.append(
            RunEvent(
                stage="apply",
                kind="branch",
                detail=f"created {branch} from {target.base_branch} "
                f"(sha {str(branch_result.get('sha', ''))[:8]})",
            )
        )
        commit_result = json.loads(
            await commit_tool.ainvoke(
                {
                    "owner": owner,
                    "name": repo,
                    "branch": branch,
                    "message": message,
                    "changes": payload,
                }
            )
        )
        events.append(
            RunEvent(
                stage="apply",
                kind="commit",
                detail=f"commit {str(commit_result.get('commit_sha', ''))[:12]}",
            )
        )
    except Exception as exc:
        events.append(RunEvent(stage="apply", kind="error", detail=f"apply failed: {exc}"))
        return {"applied_branch": None, "events": events}
    return {"applied_branch": branch, "events": events}


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


def _design_with(model: BaseChatModel | None) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Closure binding the model into the design node."""

    async def node(state: AgentState) -> dict[str, Any]:
        return await design_node(state, model=model)

    return node


def _apply_with(
    tools: Sequence[BaseTool], token: str | None
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Closure binding write tools + token into the apply node."""

    async def node(state: AgentState) -> dict[str, Any]:
        return await apply_node(state, tools=tools, token=token)

    return node


def _after_apply(pass_through: bool) -> Callable[[AgentState], str]:
    """Route after ``apply``: open the PR when a branch exists. In stub mode
    (pass_through) the pipeline keeps its linear demo shape."""

    def route(state: AgentState) -> str:
        if pass_through or state["applied_branch"]:
            return "pr"
        return "end"

    return route


def create_agent_graph(
    model: BaseChatModel | None = None,
    tools: Sequence[BaseTool] = (),
    token: str | None = None,
) -> CompiledStateGraph:
    """Build and compile the fix-agent pipeline.

    ``model`` and ``tools`` are bound into the investigate node, ``tools`` +
    ``token`` into apply. Without model/tools/token the graph runs in
    deterministic stub mode (useful for tests/demo).
    """
    pass_through = model is None and not tools and token is None
    builder = StateGraph(AgentState)
    builder.add_node("investigate", cast(Any, _investigate_with(model, tools)))
    builder.add_node("design", cast(Any, _design_with(model)))
    builder.add_node("apply", cast(Any, _apply_with(tools, token)))
    builder.add_node("pr", pr_node)
    builder.add_edge(START, "investigate")
    builder.add_edge("investigate", "design")
    builder.add_edge("design", "apply")
    builder.add_conditional_edges("apply", _after_apply(pass_through), {"pr": "pr", "end": END})
    builder.add_edge("pr", END)
    return builder.compile()
