"""Typed state and payloads flowing through the fix-agent graph.

The graph is a linear pipeline ``investigate -> design -> apply -> pr``.
Each node consumes the current state and returns partial updates that are
merged back. ``events`` accumulates a typed execution trace — the same
events will feed the SSE stream and the audit log in later stages.
"""

from __future__ import annotations

import operator
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

Stage = Literal["investigate", "design", "apply", "pr"]


class RunTarget(BaseModel):
    """The issue or pull request the run is fixing."""

    repo_full_name: str
    kind: Literal["issue", "pr"]
    number: int
    title: str = ""
    base_branch: str = "main"


class FileChange(BaseModel):
    """One file modification proposed by the agent (structured patch entry).

    ``content`` set + ``delete=False`` creates/updates the file;
    ``delete=True`` removes it. Consumed deterministically by the apply node.
    """

    path: str
    content: str = ""
    delete: bool = False
    explanation: str = ""


class ContextDoc(BaseModel):
    """A repository file/chunk the investigate node gathered for the agent.

    ``chunk_index``/``score`` are set when the doc comes from RAG retrieval
    (pgvector chunks); plain file reads leave them at their defaults.
    """

    path: str
    content: str
    chunk_index: int = 0
    score: float | None = None


Retriever = Callable[[str, str, int], Awaitable[list[ContextDoc]]]


class RunEvent(BaseModel):
    """A typed trace event emitted by a graph node (SSE payload later)."""

    stage: Stage
    kind: str
    detail: str = ""
    time: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AgentState(TypedDict):
    """Full run state threaded through the graph."""

    target: RunTarget
    messages: Annotated[list[AnyMessage], add_messages]
    events: Annotated[list[RunEvent], operator.add]
    context: list[ContextDoc]
    investigation: str | None
    root_cause_hypothesis: str | None
    proposed_changes: list[FileChange]
    applied_branch: str | None
    pr_url: str | None
