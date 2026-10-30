#!/usr/bin/env python3
"""Tool stubs for the integration tests.

Important Classes:
    EchoTool: Echo tool that changes the repo so
        the agent loop terminates after tool execution.

Important Functions:
    function_definition(name, description, properties, required) -> dict:
        Build an OpenAI-compatible tool definition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from histrel.core.runtime import Runtime
from histrel.tools.tool import Tool, ToolContext, ToolExecution
from tests.integration.scripting import complete_and_change


def function_definition(
    name: str,
    description: str,
    properties: dict[str, Any] | None = None,
    required: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Build an OpenAI-compatible tool definition."""
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": properties or {},
    }
    if required:
        parameters["required"] = list(required)
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


class EchoTool(Tool):
    """Echo tool stub that changes the repo."""

    name = "echo"
    definition = function_definition(
        "echo",
        "Echo text back.",
        {"text": {"type": "string", "description": "Text to echo back."}},
        ("text",),
    )
    examples: list[str] = [
        '{"text": "hello"} -> "hello"',
        '{"text": "world"} -> "world"',
    ]

    def __init__(
        self,
        runtime: Runtime,
        workspace: Path,
        *,
        complete: bool = True,
    ) -> None:
        self._runtime = runtime
        self._workspace = workspace
        self._complete = complete

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        if self._complete:
            complete_and_change(self._workspace)
        return ToolExecution(status="ok", output=arguments.get("text", ""))
