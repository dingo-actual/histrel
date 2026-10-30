#!/usr/bin/env python3
"""
Tool protocol and shared types for the benchmark agent.

Defines the abstract base class that all tools implement, the context
passed to tool executions, and the path resolver that enforces workspace
isolation. Paths resolve relative to the current working directory,
which is the isolated workspace in the benchmark harness; paths that
resolve outside both CWD and /tmp are rejected.

    Important Classes:
        ToolContext: Context passed to every tool execution.
        ToolExecution: Result of a tool execution.
        Tool: Abstract base class that all tools implement.

    Important Functions:
        resolve_path(path_str) -> Path: Resolve a path; reject paths outside CWD and /tmp.
        build_tool_examples_section(tools) -> str: Build system-prompt tool examples section.

    Important Variables:
        MAX_TOOL_OUTPUT_CHARS (int): Maximum characters a tool returns
            before its output is truncated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from histrel.core.session_context import SessionContext

MAX_TOOL_OUTPUT_CHARS: int = 8192

# Scratch-space root where absolute paths are always accepted. Resolved
# so that platforms where /tmp is a symlink (e.g. macOS /private/tmp)
# compare correctly.
_ALLOWED_ABSOLUTE_ROOT: Path = Path("/tmp").resolve()


class ToolContext(BaseModel):
    """Context passed to every tool execution.

    Instance Attributes:
        session (SessionContext): Session-level context.
        message_index (int): Journal index of the message that triggered
            this tool call.
    """

    session: SessionContext
    message_index: int = 0


class ToolExecution(BaseModel):
    """Result of a tool execution.

    Instance Attributes:
        status (str): Execution status (ok, error, denied, duplicate,
            timeout).
        output (object): Tool output (string, dict, or other).
        metadata (dict[str, Any]): Additional execution metadata. Tools set
            ``metadata["informative"] = True`` on correctly-specified calls
            that fail with a prose, actionable reason (for example reading a
            nonexistent file); the runtime surfaces those failures to the
            model as observations.
    """

    status: str
    output: object
    metadata: dict[str, Any] = {}


class Tool(ABC):
    """Abstract base class that all tools implement.

    Class Attributes:
        name (str): Tool name.
        definition (dict[str, Any]): OpenAI-compatible tool definition.
    """

    name: str
    definition: dict[str, Any]

    @abstractmethod
    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        """Execute the tool.

        Args:
            arguments (dict[str, Any]): Tool arguments.
            context (ToolContext): Execution context.

        Returns:
            (ToolExecution) Execution result.
        """
        ...


def resolve_path(path_str: str) -> Path:
    """Resolve a path and verify it is under CWD or /tmp.

    The input path is resolved to an absolute path *before* the
    containment check so that directory traversal (``../../etc/passwd``)
    and symlinks that point outside the allowed roots are detected
    rather than followed silently. The resolved path must fall under
    the current working directory or under ``/tmp``; every other path
    is rejected.

    Args:
        path_str (str): Relative or absolute path from the model.

    Returns:
        (Path) Resolved path, guaranteed to exist on disk.

    Raises:
        ValueError: If the resolved path is not under CWD or /tmp.
        FileNotFoundError: If the path does not exist.
    """
    resolved = Path(path_str).resolve(strict=True)

    # Resolve before the containment check so that symlinks and
    # ``..`` traversal that escape the allowed roots are rejected.
    cwd = Path.cwd()
    is_outside_cwd = not resolved.is_relative_to(cwd)
    is_outside_tmp = not resolved.is_relative_to(_ALLOWED_ABSOLUTE_ROOT)
    if is_outside_cwd and is_outside_tmp:
        raise ValueError(
            f"path {resolved} is not under CWD ({cwd}) or {_ALLOWED_ABSOLUTE_ROOT}"
        )

    return resolved


def build_tool_examples_section(tools: list[Tool]) -> str:
    """Build a system-prompt section listing tool descriptions and examples.

    Neither OpenAI nor Fireworks AI supports an 'examples' field in tool
    definitions. This function appends a text section to the system prompt
    that lists all possible tools with their descriptions and usage examples.

    The section begins and ends with statements that not all tools shown
    will always be available.

    Args:
        tools (list[Tool]): All tools that may be available during the session.

    Returns:
        (str) Formatted tool examples section for the system prompt.
    """
    if not tools:
        return ""

    lines: list[str] = [
        "## Available Tools",
        "",
        (
            "The tools listed below may be available during this session. "
            "Not all of the tools shown will always be available — the set "
            "of tools offered on each turn depends on the current task state."
        ),
        "",
    ]

    for tool in tools:
        lines.append(f"### {tool.name}")
        desc = tool.definition.get("function", {}).get("description", "")
        if desc:
            lines.append(f"**Description:** {desc}")
            lines.append("")
        examples = getattr(tool, "examples", ())
        if examples:
            lines.append("**Usage examples:**")
            lines.append("")
            for example in examples:
                lines.append("```")
                lines.append(example)
                lines.append("```")
                lines.append("")
        else:
            lines.append("*(No examples provided)*")
            lines.append("")

    lines.append(
        "Remember: the availability of tools changes based on the current "
        "task state. Always check which tools are offered before using them."
    )

    return "\n".join(lines)
