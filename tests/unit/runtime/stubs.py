#!/usr/bin/env python3
"""Tool stubs for the runtime unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from histrel.core.runtime import Runtime
from histrel.tools.tool import Tool, ToolContext, ToolExecution
from tests.unit.runtime.helpers import change_repo


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
    examples: list[str] = []

    def __init__(
        self,
        runtime: Runtime,
        tmp_path: Path,
        *,
        changes_repo: bool = True,
    ) -> None:
        self._runtime = runtime
        self._tmp_path = tmp_path
        self._changes_repo = changes_repo

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        if self._changes_repo:
            change_repo(self._tmp_path)
        return ToolExecution(status="ok", output=arguments.get("text", ""))


class FailTool(Tool):
    """Failing tool stub that changes the repo."""

    name = "fail"
    definition = function_definition("fail", "Always fails.")
    examples: list[str] = []

    def __init__(self, runtime: Runtime, tmp_path: Path) -> None:
        self._runtime = runtime
        self._tmp_path = tmp_path

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        change_repo(self._tmp_path)
        return ToolExecution(status="error", output="intentional failure")


class CrashTool(Tool):
    """Crashing tool stub that changes the repo, then raises."""

    name = "crash"
    definition = function_definition("crash", "Always raises.")
    examples: list[str] = []

    def __init__(self, runtime: Runtime, tmp_path: Path) -> None:
        self._runtime = runtime
        self._tmp_path = tmp_path

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        change_repo(self._tmp_path)
        raise NotADirectoryError("not a directory: 'src/file.py'")


class RecordTool(Tool):
    """Tool stub that records start/end events per call."""

    name = "record"
    definition = function_definition(
        "record",
        "Record execution events.",
        {"marker": {"type": "string", "description": "Position marker."}},
        ("marker",),
    )
    examples: list[str] = []

    def __init__(
        self,
        runtime: Runtime,
        tmp_path: Path,
        events: list[tuple[str, str]],
    ) -> None:
        self._runtime = runtime
        self._tmp_path = tmp_path
        self._events = events

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        marker = str(arguments["marker"])
        self._events.append(("start", marker))
        if marker == "first":
            change_repo(self._tmp_path)
        self._events.append(("end", marker))
        return ToolExecution(status="ok", output=marker)
