"""Thin MCP -> LangChain tool adapter for the GitHub MCP server.

Binds an ``mcp.Client`` (SDK v2) over stdio and exposes the server's tools
as ``langchain_core`` ``StructuredTool`` instances, so LangGraph nodes can
call GitHub with typed arguments. Deliberately replaces
``langchain-mcp-adapters`` (which lags the MCP v2 API) with ~80 local lines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool, ToolException
from mcp import Client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent, Tool
from pydantic import BaseModel, create_model

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEFAULT_EXE = _REPO_ROOT / "mcp-servers" / ".venv" / "Scripts" / "github-mcp-server.exe"


def default_github_mcp_command() -> tuple[str, list[str]]:
    """Return ``(command, args)`` that launch the project's GitHub MCP server."""
    if _DEFAULT_EXE.exists():
        return (str(_DEFAULT_EXE), [])
    raise FileNotFoundError(
        f"github-mcp-server executable not found at {_DEFAULT_EXE} — pass command/args explicitly"
    )


def _json_type(schema: dict[str, Any]) -> Any:
    """Map a JSON Schema type to a Python type."""
    match schema.get("type"):
        case "string":
            return str
        case "integer":
            return int
        case "number":
            return float
        case "boolean":
            return bool
        case "array":
            return list
        case "object":
            return dict
        case _:
            return Any


def json_schema_to_model(name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Convert a JSON Schema tool input into a Pydantic model for LangChain."""
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    fields: dict[str, Any] = {}
    for field_name, prop_schema in properties.items():
        py_type = _json_type(prop_schema)
        default = prop_schema.get("default")
        if field_name in required:
            fields[field_name] = (py_type, ...)
        elif default is not None:
            fields[field_name] = (py_type, default)
        else:
            fields[field_name] = (py_type | None, None)
    return create_model(name, **fields)


class GitHubMCPTools:
    """Async context manager binding LangChain tools to a running MCP server."""

    def __init__(
        self,
        command: str | None = None,
        args: list[str] | None = None,
        token: str | None = None,
        cwd: str | Path | None = None,
    ) -> None:
        self._command, self._args = (
            (command, args or []) if command else default_github_mcp_command()
        )
        self._token = token
        self._cwd = cwd
        self._client: Client | None = None
        self.tools: list[StructuredTool] = []

    async def __aenter__(self) -> GitHubMCPTools:
        env = {"GITHUB_TOKEN": self._token} if self._token else None
        params = StdioServerParameters(
            command=self._command, args=self._args, env=env, cwd=self._cwd
        )
        client = Client(stdio_client(params))
        await client.__aenter__()
        tools_result = await client.list_tools()
        self._client = client
        self.tools = [self._adapt(tool) for tool in tools_result.tools]
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._client is not None:
            await self._client.__aexit__(*exc_info)
            self._client = None

    def _adapt(self, tool: Tool) -> StructuredTool:
        """Wrap a single MCP tool as a typed LangChain ``StructuredTool``."""

        async def _invoke(**kwargs: Any) -> str:
            if self._client is None:
                raise ToolException("MCP client is not connected")
            result = await self._client.call_tool(tool.name, kwargs)
            text = "".join(part.text for part in result.content if isinstance(part, TextContent))
            if result.is_error:
                raise ToolException(f"MCP tool '{tool.name}' failed: {text or 'unknown error'}")
            return text

        return StructuredTool.from_function(
            coroutine=_invoke,
            name=tool.name,
            description=tool.description or f"GitHub tool: {tool.name}",
            args_schema=json_schema_to_model(tool.name, tool.input_schema),
            infer_schema=False,
        )
