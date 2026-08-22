"""Tests for the Celery wiring: ``execute_run`` streaming + task registration."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.tools import StructuredTool, ToolException

from agentos.agent.state import ContextDoc, RunTarget
from agentos.tasks import celery_app, execute_run

TARGET = RunTarget(repo_full_name="octocat/Hello-World", kind="issue", number=1)

MODEL_JSON = (
    '{"investigation": "App crashes when run without arguments.",'
    ' "root_cause_hypothesis": "missing null check in the entrypoint"}'
)
DESIGN_JSON = json.dumps(
    [{"path": "entrypoint.py", "content": "def main():\n    pass\n", "explanation": "fix"}]
)
PR_SUMMARY_JSON = json.dumps({"title": "fix: entrypoint crash", "body": "Adds a null check."})
PR_JSON = json.dumps({"number": 42, "url": "https://github.com/octocat/Hello-World/pull/42"})


class FakeModel:
    def __init__(self, *contents: str) -> None:
        self._contents = contents
        self._index = 0

    async def ainvoke(self, messages: list[Any]) -> AIMessage:
        content = self._contents[min(self._index, len(self._contents) - 1)]
        self._index += 1
        return AIMessage(content=content)


def make_tool(name: str, body: str | Exception) -> StructuredTool:
    async def invoke(**kwargs: Any) -> str:
        if isinstance(body, Exception):
            raise body
        return body

    return StructuredTool.from_function(
        coroutine=invoke, name=name, description=name, infer_schema=False
    )


FULL_TOOLS = [
    make_tool("get_issue", json.dumps({"number": 1, "title": "crash", "body": "…"})),
    make_tool(
        "list_repo_files",
        json.dumps([{"kind": "file", "name": "README", "path": "README", "size": 20}]),
    ),
    make_tool("read_file", "# demo\n"),
    make_tool("create_branch", json.dumps({"sha": "ABC123"})),
    make_tool("create_commit", json.dumps({"commit_sha": "c0ffee"})),
    make_tool("create_pull_request", PR_JSON),
]


async def test_execute_run_publishes_every_event_then_final() -> None:
    published: list[dict[str, Any]] = []

    async def publish(payload: dict[str, Any]) -> None:
        published.append(payload)

    model = FakeModel(MODEL_JSON, DESIGN_JSON, PR_SUMMARY_JSON)
    result = await execute_run(
        "run-1",
        TARGET,
        "ght_test",
        publish=publish,
        model=model,
        tools=FULL_TOOLS,
    )
    assert published[0] == {"run_id": "run-1", "type": "start"}
    kinds = [item["kind"] for item in published if item["type"] == "event"]
    assert kinds == [
        "target",
        "context",
        "hypothesis",
        "design",
        "branch",
        "commit",
        "summary",
        "pr",
    ]
    assert all(item["run_id"] == "run-1" for item in published)
    final = published[-1]
    assert final["type"] == "final"
    assert final["state"]["pr_url"] == "https://github.com/octocat/Hello-World/pull/42"
    assert final["state"]["applied_branch"].startswith("fix/issue-1-")
    assert result["pr_url"] == "https://github.com/octocat/Hello-World/pull/42"


async def test_execute_run_surfaces_apply_failure() -> None:
    published: list[dict[str, Any]] = []

    async def publish(payload: dict[str, Any]) -> None:
        published.append(payload)

    tools = [
        *FULL_TOOLS[:3],
        make_tool("create_branch", ToolException("boom")),
        make_tool("create_commit", json.dumps({"commit_sha": "x"})),
        make_tool("create_pull_request", PR_JSON),
    ]
    await execute_run(
        "run-2",
        TARGET,
        "ght_test",
        publish=publish,
        model=FakeModel(MODEL_JSON, DESIGN_JSON, PR_SUMMARY_JSON),
        tools=tools,
    )
    kinds = [item["kind"] for item in published if item["type"] == "event"]
    assert "error" in kinds
    error = next(item for item in published if item["type"] == "event" and item["kind"] == "error")
    assert error["detail"].startswith("apply failed")
    assert published[-1]["type"] == "final"
    assert published[-1]["state"]["applied_branch"] is None


async def test_execute_run_with_retrieval_publishes_rag_event() -> None:
    published: list[dict[str, Any]] = []

    async def publish(payload: dict[str, Any]) -> None:
        published.append(payload)

    async def fake_retrieve(repo: str, query: str, top_k: int) -> list[ContextDoc]:
        return [
            ContextDoc(
                path="src/app.py",
                content="def run(): return None",
                chunk_index=3,
                score=0.9,
            )
        ]

    result = await execute_run(
        "run-3",
        TARGET,
        "ght_test",
        publish=publish,
        model=FakeModel(MODEL_JSON, DESIGN_JSON, PR_SUMMARY_JSON),
        tools=FULL_TOOLS,
        retrieval=fake_retrieve,
    )
    kinds = [item["kind"] for item in published if item["type"] == "event"]
    assert kinds == [
        "target",
        "rag",
        "context",
        "hypothesis",
        "design",
        "branch",
        "commit",
        "summary",
        "pr",
    ]
    rag_event = next(
        item for item in published if item["type"] == "event" and item["kind"] == "rag"
    )
    assert rag_event["detail"] == "retrieved 1 relevant chunk(s)"
    assert result["pr_url"] == "https://github.com/octocat/Hello-World/pull/42"


def test_celery_app_is_wired_to_settings() -> None:
    from agentos.core.config import get_settings

    settings = get_settings()
    assert celery_app.conf.broker_url == settings.celery_broker_url
    assert celery_app.conf.result_backend == settings.celery_result_backend
    assert "agentos.run_fix_workflow" in celery_app.tasks
