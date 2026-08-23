"""Judge-based fix evaluation: score a completed run with a separate LLM.

The judge model (``OPENCODE_JUDGE_MODEL``) reviews the investigation,
hypothesis and proposed changes *after* the pipeline finished and returns a
structured verdict — approved / changes_requested / failed with per-axis
scores. It is deliberately separate from the agent model (a stronger,
independent reviewer) and binds to opencode the same way ``create_agent_llm``
does. Evaluation never raises: ``evaluate_run`` returns a verdict object on
success and raises only on hard judge failure (the caller degrades into an
``eval/error`` event).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr

from agentos.agent.state import FileChange, RunTarget
from agentos.core.config import get_settings

_MAX_PROMPT_CHARS = 32_000


class JudgeScores(BaseModel):
    """Per-axis quality scores (0 = worst, 5 = best)."""

    correctness: float = Field(ge=0.0, le=5.0)
    minimality: float = Field(ge=0.0, le=5.0)
    behavior_preservation: float = Field(ge=0.0, le=5.0)
    grounding: float = Field(ge=0.0, le=5.0)


class JudgeVerdict(BaseModel):
    """Structured assessment of a fix by the judge model."""

    verdict: Literal["approved", "changes_requested", "failed"]
    scores: JudgeScores
    summary: str = ""
    issues: list[str] = []

    @property
    def passed(self) -> bool:
        return self.verdict == "approved"


_JUDGE_PROMPT = """You are the evaluation stage of a GitHub code-fix agent.

A fix pipeline just produced a pull-request change set for an issue. Review it
like a strict code reviewer and return STRICT JSON with exactly four keys:
- "verdict": "approved" | "changes_requested" | "failed"
- "scores": {"correctness": 0-5, "minimality": 0-5, "behavior_preservation":
  0-5, "grounding": 0-5}  (floats; 5 = perfect)
- "summary": 1-3 sentences for the human operator
- "issues": array of concrete problems found (empty if none)

HOW TO REVIEW (follow this algorithm exactly):

1. EDITS ARE FIND-AND-REPLACE PAIRS. Each change with "edits" means: the
   "FIND" block occurs EXACTLY once in the current file and is replaced
   VERBATIM by the "REPLACE WITH" block. Do NOT read a fragment as the whole
   file — a "FIND"/"REPLACE" piece may look unbalanced or incomplete in
   isolation; that is normal for surgical edits.
2. MENTALLY APPLY the pairs to the current content, then judge the RESULT:
   is the resulting file valid, and does it satisfy the issue?
3. Only flag a syntax/JSX/comment error if you can see it in the RESULT
   after applying the pairs — never infer breakage from a truncated-looking
   fragment.
4. Missing-but-related housekeeping (e.g. an import that became unused after
   removing the only usage) is a REAL but MINOR finding: lower correctness
   by ~0.5, never crash it to 0 — the primary requirement was met.

Scoring rules:
- correctness: does the change actually resolve the issue's requirements?
  Nearly-perfect resolution with one cosmetic leftover scores 4.5, not 0.
- minimality: does it touch ONLY what the issue asked (no scope creep, no
  unrelated refactors/rewrites, no hallucinated files)?
- behavior_preservation: does it keep unrelated behavior intact (habits,
  handlers, labels, imports still used elsewhere)?
- grounding: is the change anchored in real repo content (paths that exist,
  edits whose FIND matches the current file), not invented structure?

Be strict but FAIR: a full-file rewrite when a 2-line edit would do scores low
on minimality and behavior_preservation; a change to a path that does not
exist in the repo fails grounding. No prose outside the JSON object."""


_MAX_EDIT_CHARS = 4_000  # per FIND/REPLACE block — full surgical context


def _clip(text: str, limit: int) -> str:
    """Clip to ``limit`` chars WITHOUT cutting mid-way in a misleading spot."""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...({len(text) - limit} chars omitted)"


def _describe_change(change: FileChange) -> str:
    builder = [f"### CHANGE: {change.path}"]
    if change.delete:
        builder.append("ACTION: delete this file")
    for index, pair in enumerate(change.edits):
        builder.append(f"EDIT #{index + 1}")
        builder.append(f"FIND (must occur exactly once):\n{_clip(pair.before, _MAX_EDIT_CHARS)}")
        builder.append(f"REPLACE WITH:\n{_clip(pair.after, _MAX_EDIT_CHARS)}")
    if change.content and not change.edits:
        builder.append(f"FULL NEW FILE CONTENT:\n{_clip(change.content, _MAX_EDIT_CHARS)}")
    if change.explanation:
        builder.append(f"EXPLANATION: {change.explanation}")
    return "\n".join(builder)


def _build_prompt(
    target: RunTarget,
    *,
    investigation: str,
    hypothesis: str,
    changes: list[FileChange],
    applied_branch: str | None,
    pr_url: str | None,
    current_files: Sequence[tuple[str, str]] = (),
) -> str:
    lines = [
        f"Target: {target.repo_full_name} {target.kind} #{target.number}"
        f" ({target.title or 'no title'})\n",
        f"Applied branch: {applied_branch or '(none)'}",
        f"Pull request: {pr_url or '(not opened)'}\n",
        f"Investigation: {investigation or '(empty)'}\n",
        f"Root-cause hypothesis: {hypothesis or '(none)'}\n",
        "Proposed changes:",
    ]
    lines.extend(_describe_change(change) for change in changes or [])
    if current_files:
        lines.append("\nCURRENT FILE CONTENT (authoritative — apply the edits to this):")
        for path, content in current_files:
            lines.append(f"### {path}\n{_clip(content, _MAX_EDIT_CHARS)}")
    return "\n".join(lines)[:_MAX_PROMPT_CHARS]


def create_judge_llm() -> ChatOpenAI:
    """OpenCode-bound judge model (independent of the agent model)."""
    settings = get_settings()
    return ChatOpenAI(
        base_url=settings.opencode_base_url,
        api_key=SecretStr(settings.opencode_api_key),
        model=settings.opencode_judge_model,
        max_tokens=settings.opencode_max_tokens,  # type: ignore[call-arg]  # pydantic field, stubs lag
    )


def _parse_verdict(raw: str) -> JudgeVerdict:
    """Parse the judge's STRICT JSON into a verdict (raises on bad output)."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```", 2)[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("judge output contained no JSON object")
    payload = json.loads(cleaned[start : end + 1])
    return JudgeVerdict.model_validate(payload)


async def evaluate_run(
    target: RunTarget,
    *,
    investigation: str | None,
    hypothesis: str | None,
    changes: list[FileChange],
    applied_branch: str | None,
    pr_url: str | None,
    judge: Any,
    current_files: Sequence[tuple[str, str]] = (),
) -> JudgeVerdict:
    """Score a completed run with the judge model (raises on hard failure).

    ``current_files`` carries the authoritative current content of the files
    being edited (path, full content), so the judge can mentally apply the
    find/replace pairs and judge the RESULT instead of isolated fragments.
    """
    prompt = _build_prompt(
        target,
        investigation=investigation or "",
        hypothesis=hypothesis or "",
        changes=changes,
        applied_branch=applied_branch,
        pr_url=pr_url,
        current_files=current_files,
    )
    response = await judge.ainvoke(
        [SystemMessage(content=_JUDGE_PROMPT), HumanMessage(content=prompt)]
    )
    return _parse_verdict(str(response.content))
