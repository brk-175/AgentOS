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

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
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
    EditPair,
    FileChange,
    Retriever,
    RunEvent,
    RunTarget,
    Stage,
)
from agentos.core.config import get_settings
from agentos.core.logging import get_logger
from agentos.services.rag import IndexSummary, significant_keywords

logger = get_logger(__name__)

MAX_CONTEXT_FILES = 50
MAX_FILE_CHARS = 40_000
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

# Live event sink: set by ``execute_run`` (or tests) while the graph streams.
# When set, every event is fired the moment it is created instead of waiting
# for a node-boundary snapshot — this is what makes the run page update in
# real time. When unset (plain graph runs / unit tests), events only land in
# the accumulated state, exactly as before.
_EventSink = Callable[[RunEvent], Awaitable[None]]
_event_sink: ContextVar[_EventSink | None] = ContextVar("agentos_event_sink", default=None)


def set_event_sink(sink: _EventSink | None) -> None:
    """Bind an async sink that receives every event as it is created."""
    _event_sink.set(sink)


async def _emit(
    events: list[RunEvent],
    stage: Stage,
    kind: str,
    detail: str,
) -> RunEvent:
    """Record an event in state AND fire the live sink synchronously."""
    event = RunEvent(stage=stage, kind=kind, detail=detail)
    events.append(event)
    sink = _event_sink.get()
    if sink is not None:
        await sink(event)
    return event

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
   - "edits": OPTIONAL list of {"before": "...", "after": "..."} pairs — the
     MINIMAL surgical change to an EXISTING file. Each "before" must be an
     exact substring of the current file, unique in the file, and reproduced
     verbatim; "after" is its replacement. PREFER "edits" for any fix to an
     existing file — never re-emit full content, and never include elision
     comments ("// ...") in an edit.
   - "content": ONLY for brand-new files or full rewrites: the complete new
     file content. Ignored when "edits" is present.
   - "delete": optional boolean — true to delete the file (content must be "")
   - "explanation": one short sentence about why this change fixes the issue

Never truncate a file and never write placeholder/elision code.
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


def _pr_ref(value: Any) -> str | None:
    """Extract a branch ref from a PR payload — GitHub API dict (``{"ref": ...}``)
    or the MCP server's flattened string form; anything else is ``None``."""
    if isinstance(value, dict):
        ref = value.get("ref") or value.get("name")
        return str(ref) if ref else None
    if isinstance(value, str) and value:
        return value
    return None


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
    tools: Sequence[BaseTool],
    state: AgentState,
    topic: str = "",
) -> tuple[list[ContextDoc], int]:
    """Read promising files, descending into subdirectories (bounded).

    Walks the file tree (up to ``_CONTEXT_MAX_DEPTH`` levels) and reads the
    FULL content of the most relevant files so the model edits against the
    true current source rather than a RAG snippet. Prioritizes files whose
    name matches the issue topic (title + body keywords), then README-ish
    files, then the rest (bounded by ``MAX_CONTEXT_FILES``). Never raises:
    on tool failure it degrades to an empty context.
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
            elif kind == "file" and entry.get("size", 0) <= MAX_FILE_CHARS:
                found.append(entry)
        return found

    topic_words = set(significant_keywords(topic, max_words=6))

    def is_readme(entry: dict[str, Any]) -> bool:
        return bool(_PATCHABLE_NAME.match(entry.get("name") or ""))

    def is_topic_match(entry: dict[str, Any]) -> bool:
        name = (entry.get("name") or "").lower()
        stem = name.rsplit(".", 1)[0]
        return any(word in name or word in stem for word in topic_words)

    try:
        entries = await walk("", _CONTEXT_MAX_DEPTH)
    except Exception:
        return [], 0
    picked = sorted(
        entries,
        key=lambda e: (not is_topic_match(e), not is_readme(e)),
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
    retrieval: Retriever, state: AgentState, topic: str
) -> tuple[list[ContextDoc], str]:
    """Retrieve chunks for the issue (semantic + literal keywords).

    ``topic`` combines the issue title and body so genuinely relevant terms
    ("patch files", "docs "), not just the headline, drive retrieval.
    Never raises: callers catch exceptions and degrade.
    """
    docs = await retrieval(state["target"].repo_full_name, topic, MAX_RAG_CONTEXT)
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
    indexer: Callable[[str], Awaitable[IndexSummary]] | None = None,
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

    async def loading(detail: str) -> None:
        await _emit(events, "investigate", "loading", detail)

    try:
        await loading(f"{target.kind} #{target.number} loading…")
        details = await _fetch_target(tools, state)
        await _emit(
            events,
            "investigate",
            "target",
            f"{target.kind} #{target.number} loaded: {details.get('title', '')[:80]}",
        )
        context_parts.append(
            f"### {target.kind.upper()} #{target.number}\n{json.dumps(details, indent=2)[:6000]}"
        )
    except Exception as exc:
        await _emit(events, "investigate", "error", f"target fetch failed: {exc}")
        return {
            "investigation": f"Could not load {target.kind} #{target.number} of {target.repo_full_name}",
            "root_cause_hypothesis": "",
            "events": events,
        }

    topic = " ".join(
        part
        for part in (
            str(details.get("title") or state["target"].title or ""),
            str(details.get("body") or ""),
        )
        if part
    ) or f"{target.kind} #{target.number} in {target.repo_full_name}"

    if retrieval is not None:
        await loading("searching repository context…")
        try:
            docs, detail = await _retrieve_context(retrieval, state, topic)
            if not docs and indexer is not None:
                # Cold or stale index: sync the repo chunks first, then
                # re-query — a run must never die because embeddings were
                # never generated (fresh DB) or the query just missed.
                await loading("indexing repository…")
                try:
                    summary = await indexer(state["target"].repo_full_name)
                except Exception as exc:
                    await _emit(
                        events,
                        "investigate",
                        "rag",
                        f"auto-indexing failed: {exc}",
                    )
                else:
                    await _emit(
                        events,
                        "investigate",
                        "indexed",
                        f"indexed {summary.files_indexed} file(s),"
                        f" {summary.chunks} chunk(s)",
                    )
                    docs, detail = await _retrieve_context(retrieval, state, topic)
        except Exception as exc:
            await _emit(events, "investigate", "rag", f"retrieval failed: {exc}")
        else:
            await _emit(
                events,
                "investigate",
                "rag",
                detail or "no relevant chunks found",
            )
            context_docs.extend(docs)

    await loading("reading repository files…")
    try:
        context, files_read = await _gather_context(tools, state, topic)
    except Exception as exc:
        await _emit(events, "investigate", "error", f"context fetch failed: {exc}")
    if files_read:
        await _emit(events, "investigate", "context", f"read {files_read} file(s)")
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
        await _emit(events, "investigate", "error", f"analysis failed: {exc}")
        return {
            "investigation": f"Analyzed {target.kind} #{target.number} of {target.repo_full_name}"
            + f": {details.get('title', '').strip()}",
            "root_cause_hypothesis": "",
            "context": context_docs,
            "events": events,
        }
    await _emit(events, "investigate", "hypothesis", hypothesis[:140] or "no hypothesis")
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

Return STRICT JSON: an ARRAY of change objects — the same shape as before.
Prefer "edits" ([{"before": "...", "after": "..."}] — exact substrings of the
current file, each unique) over full "content". Only new files use "content".
(path, optional edits, optional content, optional delete, explanation).
No prose outside the JSON array."""


async def _reproduce_minimal(
    model: BaseChatModel,
    tools: Sequence[BaseTool],
    target: RunTarget,
    changes: list[FileChange],
    investigation: str,
) -> tuple[list[FileChange] | None, bool]:
    """Re-read each to-be-touched file's full current content and let the model
    re-emit a minimal, grounded edit.

    Returns ``(None, False)`` when a target file cannot be read from the repo
    (two attempts) — a blind rewrite of an unknown file must never be
    committed, so the caller should treat design as failed.
    """
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
        content: str | None = None
        for _ in range(2):
            try:
                content = await read_tool.ainvoke(
                    {"owner": owner, "name": repo, "path": change.path}
                )
                break
            except Exception:
                await asyncio.sleep(1)
        if content is None:
            return None, False  # ungrounded target file — refuse a blind rewrite
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
                raw_edits = entry.get("edits") or []
                edits = [
                    EditPair(
                        before=str(pair.get("before", "")),
                        after=str(pair.get("after", "")),
                    )
                    for pair in raw_edits
                    if isinstance(pair, dict) and pair.get("before")
                ]
                fixed.append(
                    FileChange(
                        path=str(entry.get("path")),
                        content=str(entry.get("content") or ""),
                        edits=edits,
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


def _design_relevant_context(
    state: AgentState, *, max_files: int = MAX_DESIGN_CONTEXT_FILES
) -> str:
    """Pick the context docs most relevant to the fix for the design prompt.

    The investigate stage may gather many files (e.g. ``MAX_CONTEXT_FILES``)
    plus RAG/keyword chunks that duplicate paths. Feeding all of them to
    design overloads the model and degrades its output. Ranking is per-file
    (one doc per path — the highest-ranked chunk): paths explicitly named in
    the investigation first, then docs whose *content* literally contains
    investigation keywords (e.g. a keyword-retrieved file like ``page.tsx``
    for a "patch files" issue), then scored RAG chunks — capped at
    ``MAX_DESIGN_CONTEXT_FILES`` files.
    """
    docs = state["context"]
    if not docs:
        return "(no repo context was gathered)"
    probe = f"{state['investigation'] or ''} {state['root_cause_hypothesis'] or ''}"
    probe_lower = probe.lower()
    keywords = significant_keywords(probe)

    def content_hits(doc: ContextDoc) -> int:
        if not keywords:
            return 0
        lowered = doc.content.lower()
        return sum(1 for keyword in keywords if keyword in lowered)

    def rank(doc: ContextDoc) -> tuple[bool, int, bool, float, int]:
        basename = doc.path.rsplit("/", 1)[-1]
        stem = basename.rsplit(".", 1)[0].lower()
        is_named = stem in probe_lower or basename.lower() in probe_lower
        # longer content wins ties: full-file reads beat short RAG chunks
        return (is_named, content_hits(doc), doc.score is not None, doc.score or 0.0, len(doc.content))

    ranked = sorted(docs, key=rank, reverse=True)
    best_by_path: dict[str, ContextDoc] = {}
    for doc in ranked:
        best_by_path.setdefault(doc.path, doc)
    picked = list(best_by_path.values())[:max_files]
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

    async def loading(detail: str) -> None:
        await _emit(events, "design", "loading", detail)

    async def design_pass(block: str) -> list[FileChange]:
        prompt = (
            f"Target: {target.repo_full_name} {target.kind} #{target.number}"
            f" ({target.title or 'no title'})\n"
            f"Investigation: {state['investigation'] or '(empty)'}\n"
            f"Root-cause hypothesis: {state['root_cause_hypothesis'] or '(none)'}\n\n"
            f"Repo context:\n{block}"
        )[:MAX_PROMPT_CONTEXT]
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
            raw_edits = entry.get("edits") or []
            edits = [
                EditPair(before=str(pair.get("before", "")), after=str(pair.get("after", "")))
                for pair in raw_edits
                if isinstance(pair, dict) and pair.get("before")
            ]
            changes.append(
                FileChange(
                    path=str(entry["path"]),
                    content=str(entry.get("content") or ""),
                    edits=edits,
                    delete=bool(entry.get("delete", False)),
                    explanation=str(entry.get("explanation") or ""),
                )
            )
        if not changes:
            raise ValueError("no valid change entries in model patch")
        return changes

    try:
        changes = await design_pass(_design_relevant_context(state))
    except Exception as exc:
        await _emit(events, "design", "error", f"design failed: {exc}")
        # Retry once with a slimmer context — over-stuffed prompts are the
        # most common cause of a JSON-less response, so shrinking the files
        # given to the model frequently lands a valid patch on the second pass.
        await loading("design retrying with reduced context…")
        try:
            changes = await design_pass(
                _design_relevant_context(state, max_files=MAX_DESIGN_CONTEXT_FILES // 2)
            )
        except Exception as exc2:
            await _emit(events, "design", "error", f"design failed: {exc2}")
            return {"proposed_changes": [], "events": events}
        await _emit(events, "design", "design", "retry succeeded with reduced context")

    if tools and model is not None:
        ground, repaired = await _reproduce_minimal(
            model, tools, target, changes, state["investigation"] or ""
        )
        if ground is None:
            if not repaired:
                await _emit(
                    events,
                    "design",
                    "error",
                    "design aborted: could not read a target file to ground the edit",
                )
            return {"proposed_changes": [], "events": events}
        if repaired:
            changes = ground
            await _emit(
                events,
                "design",
                "design",
                "minimal-edit pass: re-grounded against current file(s)",
            )

    corrupt = _reject_corrupt_changes(changes)
    if corrupt:
        await _emit(
            events,
            "design",
            "error",
            f"design rejected: placeholder/truncated content in {', '.join(corrupt)}",
        )
        return {"proposed_changes": [], "events": events}

    paths = ", ".join(c.path for c in changes[:5])
    await _emit(
        events,
        "design",
        "design",
        f"{len(changes)} change(s): {paths}" + ("…" if len(changes) > 5 else ""),
    )
    return {"proposed_changes": changes, "events": events}


_CORRUPT_CONTENT_MARKERS = (
    "// ... other imports",  # LLM placeholder — not real code
    "// ... other state",
    "// ... rest of modes",
    "// ... other sections",
    "/* ... */",  # bare elision comment
    "<div className=\"...\">",  # placeholder JSX class
    "<header className=\"...\">",
    "// ... rest of file",
)


def _reject_corrupt_changes(changes: list[FileChange]) -> list[str]:
    """Return the paths whose ``content`` looks like LLM-truncated placeholder
    code (elision comments instead of real content). Such changes must never
    be committed — they silently delete real logic."""
    return [
        change.path
        for change in changes
        if not change.delete and any(marker in change.content for marker in _CORRUPT_CONTENT_MARKERS)
    ]


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
        await _emit(events, "apply", "error", "no changes to apply (design failed?)")
        return {"applied_branch": None, "events": events}
    if token is None:
        await _emit(events, "apply", "error", "GITHUB_TOKEN required to apply changes")
        return {"applied_branch": None, "events": events}

    branch_tool = _tool(tools, "create_branch")
    commit_tool = _tool(tools, "create_commit")
    if branch_tool is None or commit_tool is None:
        await _emit(events, "apply", "error", "MCP write tools not available")
        return {"applied_branch": None, "events": events}

    owner, _, repo = target.repo_full_name.partition("/")
    branch = _branch_name(target, unique=True)
    message = f"AgentOS: fix {target.kind} #{target.number}"
    if target.title:
        message = f"{message}: {target.title}"
    message = message[:_MAX_COMMIT_MESSAGE]
    payload = [
        change.model_dump(include={"path", "content", "edits", "delete"}) for change in changes
    ]
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
        await _emit(
            events,
            "apply",
            "branch",
            f"created {branch} from {target.base_branch} "
            f"(sha {str(branch_result.get('sha', ''))[:8]})",
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
        await _emit(
            events,
            "apply",
            "commit",
            f"commit {str(commit_result.get('commit_sha', ''))[:12]}",
        )
    except Exception as exc:
        await _emit(events, "apply", "error", f"apply failed: {exc}")
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
        await _emit(events, "pr", "error", "no applied branch to open a PR from")
        return {"pr_url": None, "events": events}
    if token is None:
        await _emit(events, "pr", "error", "GITHUB_TOKEN required to open a PR")
        return {"pr_url": None, "events": events}
    pr_tool = _tool(tools, "create_pull_request")
    if pr_tool is None:
        await _emit(events, "pr", "error", "MCP create_pull_request tool not available")
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
    await _emit(events, "pr", "summary", title)

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
        await _emit(events, "pr", "error", f"pr failed: {exc}")
        return {"pr_url": None, "events": events}

    # The PR is already open — metadata must never take the run down with it.
    # Payload shape varies (raw GitHub API vs the MCP server's flattened view),
    # so build the display dict defensively and degrade to a minimal card.
    try:
        author = result.get("author")
        if author is None and isinstance(result.get("user"), dict):
            author = result["user"].get("login")
        pr = {
            "number": number,
            "url": pr_url,
            "title": str(result.get("title") or title),
            "body": str(result.get("body") or body),
            "state": str(result.get("state") or "open"),
            "author": str(author) if author is not None else None,
            "created_at": str(result.get("created_at") or ""),
            "base": _pr_ref(result.get("base")),
            "head": _pr_ref(result.get("head")),
            "changed_files": result.get("changed_files"),
            "additions": result.get("additions"),
            "deletions": result.get("deletions"),
        }
    except Exception as exc:
        logger.warning("pr metadata mapping failed for %s: %s", pr_url, exc)
        await _emit(events, "pr", "error", f"pr metadata mapping failed (PR is open): {exc}")
        pr = {"number": number, "url": pr_url, "title": title, "body": body}

    await _emit(events, "pr", "pr", f"opened PR #{number}: {pr_url}")
    return {"pr_url": pr_url, "pr": pr, "events": events}


def _investigate_with(
    model: BaseChatModel | None,
    tools: Sequence[BaseTool],
    retrieval: Retriever | None = None,
    indexer: Callable[[str], Awaitable[IndexSummary]] | None = None,
) -> Callable[[AgentState], Awaitable[dict[str, Any]]]:
    """Closure binding model+tools into the investigate node (LangGraph needs
    a real async callable — a sync wrapper returning a coroutine is not awaited)."""

    async def node(state: AgentState) -> dict[str, Any]:
        return await investigate_node(state, model=model, tools=tools, retrieval=retrieval, indexer=indexer)

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
    indexer: Callable[[str], Awaitable[IndexSummary]] | None = None,
) -> CompiledStateGraph:
    """Build and compile the fix-agent pipeline.

    ``model`` and ``tools`` are bound into the investigate node, ``tools`` +
    ``token`` into apply and the model/tools/token into the PR node. Without
    model/tools/token the graph runs in deterministic stub mode (useful for
    tests/demo). ``retrieval`` — when provided — is queried during
    ``investigate`` for semantically relevant chunks; failures degrade
    silently into today's file-only context. ``indexer`` — when provided —
    re-syncs the repository chunk index when retrieval comes up empty, then
    ``investigate`` re-queries before falling back to raw file reads.
    """
    pass_through = model is None and not tools and token is None
    builder = StateGraph(AgentState)
    builder.add_node(
        "investigate", cast(Any, _investigate_with(model, tools, retrieval, indexer))
    )
    builder.add_node("design", cast(Any, _design_with(model, tools)))
    builder.add_node("apply", cast(Any, _apply_with(tools, token)))
    builder.add_node("pr", cast(Any, _pr_with(model, tools, token)))
    builder.add_edge(START, "investigate")
    builder.add_edge("investigate", "design")
    builder.add_edge("design", "apply")
    builder.add_conditional_edges("apply", _after_apply(pass_through), {"pr": "pr", "end": END})
    builder.add_edge("pr", END)
    return builder.compile()
