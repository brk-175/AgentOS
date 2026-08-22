"""The fix-agent LangGraph: a staged pipeline of four nodes.

Topology: ``investigate -> design -> apply -> pr`` with a conditional edge
after ``apply`` (the run ends early when nothing could be applied).
``investigate`` loads the target through the MCP tools, gathers repo context
and asks the LLM for a summary + root-cause hypothesis; ``design`` asks the
LLM for the structured per-file changes; ``apply`` creates the branch and
commits the changes through MCP; ``pr`` writes the LLM-generated PR summary
and opens the pull request.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import SecretStr

from agentos.agent.state import (
    AgentState,
    ContextDoc,
    FileChange,
    Retriever,
    RunEvent,
    RunTarget,
)
from agentos.core.config import get_settings

MAX_CONTEXT_FILES = 50
MAX_FILE_CHARS = 10_000
MAX_PROMPT_CONTEXT = 1_00_000
MAX_RAG_CONTEXT = 50
MAX_DESIGN_CONTEXT_FILES = 8
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

You are editing a real repository, so follow these rules exactly:

1. MINIMAL SCOPE: change ONLY what the target issue/comments require. Do NOT
   refactor, reformat, reorder imports, rename props, change labels, buttons,
   styling, handlers or navigation unrelated to the issue. If something is not
   asked for, leave it byte-for-byte identical.
2. GROUND IN THE CURRENT FILE: you are given the full current content of each
   file you may touch. Reproduce that content EXACTLY and apply the smallest
   possible edit (a removed line, a commented-out block, an added guard). Never
   rewrite a whole component when only a line or two must change.
3. PRESERVE BEHAVIOR: never drop existing hooks/effects/state, imports used
   elsewhere, or routes. If you must remove an import, ensure no remaining code
   uses it. Prefer commenting out code over deleting it unless deletion is the
   stated intent.
4. Return STRICT JSON: an ARRAY of change objects. Each object has:
   - "path": repository-relative file path (required)
   - "content": the FULL new file content (for edits and new files); ignored when
     "delete" is true
   - "delete": optional boolean — true to delete the file (content must be "")
   - "explanation": one short sentence about why this change fixes the issue

Never truncate a file: "content" must always be the complete file.
No prose outside the JSON array."""

_MAX_COMMIT_MESSAGE = 120

_PR_PROMPT = """You are the pull-request stage of a GitHub code-fix agent.
Given the target, the investigation and the exact changes, write the pull
request summary a maintainer needs to review this fix.

Return STRICT JSON with exactly two keys:
- "title": a short, imperative summary of the change (max ~72 chars)
- "body": a concise markdown description: what was wrong, what changed
  (file by file), and how it was verified

No prose outside the JSON object."""


def create_agent_llm() -> BaseChatModel:
    """OpenCode-bound chat model (``ChatOpenAI`` is OpenAI-compatible)."""
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.opencode_base_url,
        api_key=SecretStr(settings.opencode_api_key),
        model=settings.opencode_model,
        max_tokens=settings.opencode_max_tokens,  # type: ignore[call-arg]  # pydantic field, stubs lag
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


_CONTEXT_MAX_DEPTH = 6  # depth 0 = repo root; each level descends one directory


async def _gather_context(
    tools: Sequence[BaseTool], state: AgentState
) -> tuple[list[ContextDoc], int]:
    """Read promising files, descending into subdirectories (bounded).

    Walks the file tree (up to ``_CONTEXT_MAX_DEPTH`` levels) and reads the
    FULL content of the most relevant files so the model edits against the
    true current source rather than a RAG snippet. Files that would be
    truncated are skipped. Never raises: on tool failure it degrades to a
    root listing (today's behavior).
    """
    target = state["target"]
    owner, _, repo = target.repo_full_name.partition("/")
    listing_tool = _tool(tools, "list_repo_files")
    read_tool = _tool(tools, "read_file")
    if listing_tool is None or read_tool is None:
        return [], 0

    async def walk(path: str, depth: int) -> list[dict[str, Any]]:
        try:
            listing_raw = await listing_tool.ainvoke(
                {"owner": owner, "name": repo, "path": path}
            )
        except Exception:
            return []
        listing = json.loads(listing_raw)
        entries = listing["items"] if isinstance(listing, dict) else listing
        found: list[dict[str, Any]] = []
        for entry in entries:
            kind = entry.get("kind")
            if kind == "dir" and depth > 0:
                found.extend(await walk(entry["path"], depth - 1))
            elif (
                kind == "file"
                and entry.get("size", 0) <= MAX_FILE_CHARS
            ):
                found.append(entry)
        return found

    try:
        entries = await walk("", _CONTEXT_MAX_DEPTH)
    except Exception:
        return [], 0
    picked = sorted(
        entries, key=lambda e: not bool(_PATCHABLE_NAME.match(e["name"] or ""))
    )[:MAX_CONTEXT_FILES]
    docs: list[ContextDoc] = []
    for entry in picked:
        try:
            content = await read_tool.ainvoke(
                {"owner": owner, "name": repo, "path": entry["path"]}
            )
        except Exception:
            continue
        docs.append(ContextDoc(path=entry["path"], content=content))
    return docs, len(docs)


async def _retrieve_context(
    retrieval: Retriever, state: AgentState
) -> tuple[list[ContextDoc], str]:
    """Semantic chunk search for the issue; never raises (degrade on failure)."""
    target = state["target"]
    query = target.title or f"{target.kind} #{target.number} in {target.repo_full_name}"
    docs = await retrieval(target.repo_full_name, query, MAX_RAG_CONTEXT)
    if not docs:
        return [], "no relevant chunks found"
    return docs, f"retrieved {len(docs)} relevant chunk(s)"


def _context_part(doc: ContextDoc) -> str:
    """Render one context doc as a prompt block (RAG docs carry provenance)."""
    if doc.score is None:
        return f"### {doc.path}\n{doc.content}"
    return f"### {doc.path} [chunk {doc.chunk_index}, relevance {doc.score:.2f}]\n{doc.content}"


async def investigate_node(
    state: AgentState,
    *,
    model: BaseChatModel | None,
    tools: Sequence[BaseTool],
    retrieval: Retriever | None = None,
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

    if retrieval is not None:
        try:
            docs, detail = await _retrieve_context(retrieval, state)
        except Exception as exc:
            events.append(
                RunEvent(stage="investigate", kind="rag", detail=f"retrieval failed: {exc}")
            )
        else:
            events.append(
                RunEvent(
                    stage="investigate", kind="rag", detail=detail or "no relevant chunks found"
                )
            )
            context_docs.extend(docs)

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
    context_parts.extend(_context_part(doc) for doc in context_docs)

    if model is None:
        return {
            "investigation": f"{target.kind} #{target.number} loaded from {target.repo_full_name}"
            + (f": {details.get('title', '')}" if details.get("title") else ""),
            "root_cause_hypothesis": "",
            "context": context_docs,
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
            "context": context_docs,
            "events": events,
        }
    events.append(
        RunEvent(stage="investigate", kind="hypothesis", detail=hypothesis[:140] or "no hypothesis")
    )
    return {
        "investigation": investigation,
        "root_cause_hypothesis": hypothesis,
        "context": context_docs,
        "events": events,
    }


_DESIGN_REPAIR_PROMPT = """You are the design stage of a GitHub code-fix agent,
performing a minimal-edit verification pass.

You previously proposed changes. For each file below you are given the EXACT
current content from the repository. If your proposed change rewrites this file,
re-emit it applying ONLY the minimal edit the issue requires: preserve every line
not directly related to the fix byte-for-byte. Do NOT rename props, restyle,
reorder imports, or drop hooks/effects/routes/handlers. If code must be hidden,
comment it out in place rather than deleting, unless deletion is the issue's
explicit intent. If the file did not change, return it unchanged.

Return STRICT JSON: an ARRAY of change objects — the same shape as before
(path, content, optional delete, explanation). No prose outside the JSON array."""


async def _reproduce_minimal(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    target: RunTarget,
    changes: list[FileChange],
    investigation: str,
) -> tuple[list[FileChange], bool]:
    """Re-read each to-be-touched file's full current content and let the model
    re-emit a minimal, grounded edit (returns ``(changes, repaired)``)."""
    read_tool = _tool(tools, "read_file")
    if read_tool is None:
        return changes, False
    owner, _, repo = target.repo_full_name.partition("/")
    need_fetch = [c for c in changes if not c.delete]
    if not need_fetch:
        return changes, False
    blocks: list[str] = []
    existed_paths: list[str] = []
    for change in need_fetch[:10]:
        try:
            content = await read_tool.ainvoke(
                {"owner": owner, "name": repo, "path": change.path}
            )
        except Exception:
            continue  # new file or unreadable — leave as the model wrote it
        existed_paths.append(change.path)
        blocks.append(f"### {change.path}\n{content}")
    if not existed_paths:
        return changes, False
    proposed = "\n\n".join(f"### {c.path}\n{c.content or '<deleted>'}" for c in changes)
    prompt = (
        f"Target: {target.repo_full_name} {target.kind} #{target.number}"
        f" ({target.title or 'no title'})\n"
        f"Investigation: {investigation or '(empty)'}\n\n"
        f"CURRENT FILE CONTENT (from the repo, authoritative):\n"
        + "\n\n".join(blocks)
        + f"\n\nPREVIOUSLY PROPOSED CHANGES:\n{proposed}"
    )[:MAX_PROMPT_CONTEXT]
    try:
        response = await model.ainvoke(
            [SystemMessage(content=_DESIGN_REPAIR_PROMPT), HumanMessage(content=prompt)]
        )
        raw = _extract_json(str(response.content))
        if not isinstance(raw, list):
            return changes, False
        fixed: list[FileChange] = []
        repaired: dict[str, dict[str, Any]] = {}
        for entry in raw:
            if not isinstance(entry, dict) or not entry.get("path"):
                continue
            repaired[str(entry["path"])] = entry
        for change in changes:
            if not change.delete and change.path in repaired and change.path in existed_paths:
                entry = repaired[change.path]
                fixed.append(
                    FileChange(
                        path=str(entry.get("path")),
                        content=str(entry.get("content") or ""),
                        delete=bool(entry.get("delete", False)),
                        explanation=str(entry.get("explanation") or ""),
                    )
                )
            else:
                fixed.append(change)
        if fixed == changes:
            return changes, False
        return fixed, True
    except Exception:
        return changes, False


def _design_relevant_context(state: AgentState) -> str:
    """Pick the context docs most relevant to the fix for the design prompt.

    The investigate stage may gather many files (e.g. ``MAX_CONTEXT_FILES``).
    Feeding all of them to design overloads the model and degrades its output.
    Order: docs whose path is explicitly named in the investigation first,
    then ranked RAG chunks, then the rest — capped at
    ``MAX_DESIGN_CONTEXT_FILES``.
    """
    docs = state["context"]
    if not docs:
        return "(no repo context was gathered)"
    probe = f"{state['investigation'] or ''} {state['root_cause_hypothesis'] or ''}".lower()

    def named(doc: ContextDoc) -> bool:
        basename = doc.path.rsplit("/", 1)[-1]
        stem = basename.rsplit(".", 1)[0].lower()
        return stem in probe or basename.lower() in probe

    named_docs = [doc for doc in docs if named(doc)]
    rest = [doc for doc in docs if not named(doc)]
    rest.sort(key=lambda doc: (doc.score is None, doc.score or 0.0), reverse=True)
    picked = (named_docs + rest)[:MAX_DESIGN_CONTEXT_FILES]
    return "\n\n".join(f"### {doc.path}\n{doc.content}" for doc in picked)


async def design_node(
    state: AgentState,
    *,
    model: BaseChatModel | None,
    tools: Sequence[BaseTool] = (),
) -> dict[str, Any]:
    """Design the fix: ask the LLM for exact per-file changes (strict JSON).

    When ``tools`` are available, the proposed changes go through a repair
    pass that re-reads each file's full current content and re-emits a
    minimal, grounded edit (guards against whole-file rewrites).
    """
    target = state["target"]
    events: list[RunEvent] = []
    if model is None:
        return {
            "proposed_changes": [],
            "events": [RunEvent(stage="design", kind="design", detail="fix design pending")],
        }

    context_block = _design_relevant_context(state)
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

    if tools and model is not None:
        ground, repaired = await _reproduce_minimal(
            model, tools, target, changes, state["investigation"] or ""
        )
        if repaired:
            changes = ground
            events.append(
                RunEvent(
                    stage="design",
                    kind="design",
                    detail="minimal-edit pass: re-grounded against current file(s)",
                )
            )

    paths = ", ".join(c.path for c in changes[:5])
    events.append(
        RunEvent(
            stage="design",
            kind="design",
            detail=f"{len(changes)} change(s): {paths}" + ("…" if len(changes) > 5 else ""),
        )
    )
    return {"proposed_changes": changes, "events": events}


def _branch_name(target: RunTarget, *, unique: bool) -> str:
    """Deterministic logical branch name, optionally suffixed for uniqueness.

    The suffix avoids collisions when the same issue is fixed more than once
    (or by parallel runs), so ``create_branch`` doesn't 422 on an existing ref.
    """
    base = f"fix/{target.kind}-{target.number}"
    return f"{base}-{int(time.time())}" if unique else base


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
    branch = _branch_name(target, unique=True)
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


async def pr_node(
    state: AgentState,
    *,
    model: BaseChatModel | None,
    tools: Sequence[BaseTool],
    token: str | None,
) -> dict[str, Any]:
    """Open the pull request: LLM-written title/body, created via MCP.

    Runs only when ``apply`` produced a branch (see ``_after_apply``);
    the fallbacks below cover partial/dry-run configurations defensively.
    """
    target = state["target"]
    events: list[RunEvent] = []
    if model is None and not tools and token is None:
        return {
            "pr_url": None,
            "events": [RunEvent(stage="pr", kind="pr", detail="pull request pending")],
        }
    apply_branch = state["applied_branch"]
    if not apply_branch:
        events.append(
            RunEvent(stage="pr", kind="error", detail="no applied branch to open a PR from")
        )
        return {"pr_url": None, "events": events}
    if token is None:
        events.append(
            RunEvent(stage="pr", kind="error", detail="GITHUB_TOKEN required to open a PR")
        )
        return {"pr_url": None, "events": events}
    pr_tool = _tool(tools, "create_pull_request")
    if pr_tool is None:
        events.append(
            RunEvent(stage="pr", kind="error", detail="MCP create_pull_request tool not available")
        )
        return {"pr_url": None, "events": events}

    changed = [
        change.model_dump(include={"path", "delete"}) for change in state["proposed_changes"]
    ]
    investigation = state["investigation"] or ""
    title = f"Fix {target.kind} #{target.number}: {investigation[:60]}"
    body = (
        f"Fixes {target.repo_full_name} {target.kind} #{target.number}.\n\n"
        f"## Investigation\n{investigation}\n\n"
        f"## Changes\n" + "\n".join(f"- {entry['path']}" for entry in changed)
    )
    if model is not None:
        try:
            response = await model.ainvoke(
                [
                    SystemMessage(content=_PR_PROMPT),
                    HumanMessage(
                        content=(
                            f"Target: {target.repo_full_name} {target.kind} #{target.number}\n"
                            f"Investigation: {state['investigation']}\n"
                            f"Root-cause hypothesis: {state['root_cause_hypothesis'] or '(none)'}\n"
                            f"Applicable changes: {json.dumps(changed)}"
                        )[:MAX_PROMPT_CONTEXT]
                    ),
                ]
            )
            parsed = _extract_json(str(response.content))
            llm_title = str(parsed.get("title") or "").strip()
            llm_body = str(parsed.get("body") or "").strip()
            if not llm_title or not llm_body:
                raise ValueError("model returned an incomplete PR summary")
            title, body = llm_title[:72], llm_body
        except Exception:
            pass
    events.append(RunEvent(stage="pr", kind="summary", detail=title))

    owner, _, repo = target.repo_full_name.partition("/")
    try:
        result = json.loads(
            await pr_tool.ainvoke(
                {
                    "owner": owner,
                    "name": repo,
                    "title": title,
                    "head": apply_branch,
                    "base": target.base_branch,
                    "body": body,
                }
            )
        )
        pr_url = str(result.get("url") or "")
        number = str(result.get("number") or "?")
    except Exception as exc:
        events.append(RunEvent(stage="pr", kind="error", detail=f"pr failed: {exc}"))
        return {"pr_url": None, "events": events}
    events.append(RunEvent(stage="pr", kind="pr", detail=f"opened PR #{number}: {pr_url}"))
    return {"pr_url": pr_url, "events": events}


def _investigate_with(
    model: BaseChatModel | None,
    tools: Sequence[BaseTool],
    retrieval: Retriever | None = None,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Closure binding model+tools into the investigate node (LangGraph needs
    a real async callable — a sync wrapper returning a coroutine is not awaited)."""

    async def node(state: AgentState) -> dict[str, Any]:
        return await investigate_node(state, model=model, tools=tools, retrieval=retrieval)

    return node


def _design_with(
    model: BaseChatModel | None, tools: Sequence[BaseTool]
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Closure binding model + tools into the design node."""

    async def node(state: AgentState) -> dict[str, Any]:
        return await design_node(state, model=model, tools=tools)

    return node


def _apply_with(
    tools: Sequence[BaseTool], token: str | None
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Closure binding write tools + token into the apply node."""

    async def node(state: AgentState) -> dict[str, Any]:
        return await apply_node(state, tools=tools, token=token)

    return node


def _pr_with(
    model: BaseChatModel | None, tools: Sequence[BaseTool], token: str | None
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Closure binding model+tools+token into the PR node."""

    async def node(state: AgentState) -> dict[str, Any]:
        return await pr_node(state, model=model, tools=tools, token=token)

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
    retrieval: Retriever | None = None,
) -> CompiledStateGraph:
    """Build and compile the fix-agent pipeline.

    ``model`` and ``tools`` are bound into the investigate node, ``tools`` +
    ``token`` into apply and the model/tools/token into the PR node. Without
    model/tools/token the graph runs in deterministic stub mode (useful for
    tests/demo). ``retrieval`` — when provided — is queried during
    ``investigate`` for semantically relevant chunks; failures degrade
    silently into today's file-only context.
    """
    pass_through = model is None and not tools and token is None
    builder = StateGraph(AgentState)
    builder.add_node("investigate", cast(Any, _investigate_with(model, tools, retrieval)))
    builder.add_node("design", cast(Any, _design_with(model, tools)))
    builder.add_node("apply", cast(Any, _apply_with(tools, token)))
    builder.add_node("pr", cast(Any, _pr_with(model, tools, token)))
    builder.add_edge(START, "investigate")
    builder.add_edge("investigate", "design")
    builder.add_edge("design", "apply")
    builder.add_conditional_edges("apply", _after_apply(pass_through), {"pr": "pr", "end": END})
    builder.add_edge("pr", END)
    return builder.compile()
