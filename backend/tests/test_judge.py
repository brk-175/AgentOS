"""Tests for the judge-based fix evaluation (Step 5.4)."""

from __future__ import annotations

import json
from typing import Any

import pytest
from langchain_core.messages import AI_MESSAGE_TYPE, AIMessage

from agentos.agent.state import FileChange, RunTarget
from agentos.services.judge import (
    JudgeVerdict,
    _build_prompt,
    _parse_verdict,
    create_judge_llm,
    evaluate_run,
)

TARGET = RunTarget(
    repo_full_name="octocat/Hello-World", kind="issue", number=1, title="App crashes on empty input"
)

VERDICT_JSON = json.dumps(
    {
        "verdict": "approved",
        "scores": {
            "correctness": 4.5,
            "minimality": 5.0,
            "behavior_preservation": 4.0,
            "grounding": 5.0,
        },
        "summary": "Fixes the crash with a minimal null guard.",
        "issues": [],
    }
)


class FakeJudge:
    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        self.messages = messages
        return AIMessage(content=VERDICT_JSON)


class JudgeResponse(str):
    pass


def test_create_judge_llm_is_openai_compatible_chat_model() -> None:
    model = create_judge_llm()
    assert model.model_name == __import__("agentos.core.config", fromlist=["get_settings"]).get_settings().opencode_judge_model


def test_parse_verdict_accepts_fenced_json() -> None:
    verdict = _parse_verdict(f"```json\n{VERDICT_JSON}\n```")
    assert verdict.verdict == "approved"
    assert verdict.scores.correctness == 4.5
    assert verdict.scores.minimality == 5.0
    assert verdict.passed


def test_parse_verdict_rejects_garbage() -> None:
    with pytest.raises(Exception):
        _parse_verdict("no json here")


def test_build_prompt_includes_changes_and_target() -> None:
    prompt = _build_prompt(
        TARGET,
        investigation="Prod crash on empty args.",
        hypothesis="Missing null check in entrypoint.",
        changes=[
            FileChange(
                path="entrypoint.py",
                edits=FileChange().edits,
                explanation="",
            ),
            FileChange(path="old.py", delete=True),
        ],
        applied_branch="fix/issue-1-123",
        pr_url="https://github.com/o/r/pull/1",
    )
    assert "App crashes on empty input" in prompt
    assert "entrypoint.py" in prompt
    assert "DELETE old.py" in prompt
    assert "fix/issue-1-123" in prompt


async def test_evaluate_run_returns_verdict() -> None:
    judge = FakeJudge()
    verdict = await evaluate_run(
        TARGET,
        investigation="x",
        hypothesis="y",
        changes=[FileChange(path="a.py", content="ok")],
        applied_branch="fix/issue-1-1",
        pr_url="http://pr",
        judge=judge,
    )
    assert isinstance(verdict, JudgeVerdict)
    assert verdict.passed
    assert judge.messages[0].type == "system"
