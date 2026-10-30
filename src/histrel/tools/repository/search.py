#!/usr/bin/env python3
"""
Search tool for the benchmark agent.

    Important Classes:
        SearchTool: Search file contents with ripgrep.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from histrel.tools.repository.common import _display_path
from histrel.tools.tool import (
    MAX_TOOL_OUTPUT_CHARS,
    Tool,
    ToolContext,
    ToolExecution,
    resolve_path,
)


class SearchTool(Tool):
    """Search file contents with regular expressions or literals.

    Runs search with a regex pattern or literal, as well as
    optional path/glob filters. Returns matching lines with
    file paths and line numbers. Successful output is prefixed with a
    "Search for string pattern '{pattern}' in {path}" header line
    followed by a "---" separator, so the search parameters stay
    visible in the tool response even when the call arguments are not
    replayed.
    """

    name = "search"
    definition: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search for a text pattern in files by regular expression or "
                "literal. Returns\nmatching lines with file paths and line "
                "numbers."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Search pattern (regex or literal).",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory path to search in, relative to CWD or "
                            "an absolute path under /tmp.\nDefaults to CWD."
                        ),
                    },
                    "glob": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. '*.py').",
                    },
                },
                "required": ["pattern"],
            },
        },
    }
    examples: list[str] = [
        'run(name="search", args="{\\"pattern\\": \\"def hello\\", \\"path\\": \\"src/\\"}")',
        'run(name="search", args="{\\"pattern\\": \\"TODO|FIXME\\", \\"glob\\": \\"*.py\\"}")',
    ]

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        """Execute the search tool.

        Args:
            arguments (dict[str, Any]): Tool arguments.
            context (ToolContext): Execution context.

        Returns:
            (ToolExecution) Execution result.
        """
        pattern = arguments.get("pattern", "")
        if not pattern:
            return ToolExecution(
                status="error",
                output="'pattern' is required",
                metadata={},
            )

        cmd = ["rg", "--line-number", "--no-heading", "--color=never", pattern]

        path_str = arguments.get("path")
        search_path = _resolve_search_path(pattern, path_str)
        if isinstance(search_path, ToolExecution):
            return search_path
        if path_str:
            cmd.append(str(search_path))

        glob_pattern = arguments.get("glob")
        if glob_pattern:
            cmd.extend(["--glob", glob_pattern])

        result = _run_search_command(cmd, pattern, search_path)
        if isinstance(result, ToolExecution):
            return result
        return _shape_search_result(result, pattern, search_path)


def _search_path_error(pattern: str, path_str: str, reason: str) -> ToolExecution:
    """Build the informative error for an unusable search path."""
    return ToolExecution(
        status="error",
        output=(
            f"Attempt to search for text pattern '{pattern}' in "
            f"'{path_str}' failed due to {reason}."
        ),
        metadata={
            "pattern": pattern,
            "path": path_str,
            "informative": True,
        },
    )


def _resolve_search_path(pattern: str, path_str: Any) -> Path | ToolExecution:
    """Resolve the search path, defaulting to CWD.

    Args:
        pattern (str): Search pattern for error messages.
        path_str (Any): Raw path argument (may be absent).

    Returns:
        (Path | ToolExecution) The resolved directory, or an error result.
    """
    if not path_str:
        return Path.cwd()
    try:
        return resolve_path(path_str)
    except FileNotFoundError:
        return _search_path_error(pattern, path_str, "the search path not existing")
    except ValueError as exc:
        return _search_path_error(pattern, path_str, str(exc))


def _run_search_command(
    cmd: list[str], pattern: str, search_path: Path
) -> subprocess.CompletedProcess[str] | ToolExecution:
    """Run ripgrep, mapping backend failures to tool errors.

    Args:
        cmd (list[str]): Full ripgrep command.
        pattern (str): Search pattern for messages.
        search_path (Path): Resolved search directory.

    Returns:
        (subprocess.CompletedProcess[str] | ToolExecution) The process
        result, or an error/timeout result.
    """
    try:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        return ToolExecution(
            status="error",
            output=(
                f"Attempt to search for text pattern '{pattern}' in "
                f"'{_display_path(search_path)}' failed due to the "
                "search backend not being available."
            ),
            metadata={"pattern": pattern, "informative": True},
        )
    except subprocess.TimeoutExpired:
        return ToolExecution(
            status="timeout",
            output=(
                f"Attempt to search for text pattern '{pattern}' in "
                f"'{_display_path(search_path)}' failed due to the "
                "search timing out after 30 seconds."
            ),
            metadata={"pattern": pattern, "informative": True},
        )


def _shape_search_result(
    result: subprocess.CompletedProcess[str], pattern: str, search_path: Path
) -> ToolExecution:
    """Convert a ripgrep process result into a tool execution result."""
    if result.returncode not in (0, 1):
        # ripgrep exits 2+ for usage errors (bad regex, bad path)
        detail = result.stderr or (f"the search exiting with code {result.returncode}")
        return ToolExecution(
            status="error",
            output=(
                f"Attempt to search for text pattern '{pattern}' in "
                f"'{_display_path(search_path)}' failed due to: {detail}"
            ),
            metadata={
                "returncode": result.returncode,
                "pattern": pattern,
                "informative": True,
            },
        )

    # ripgrep exits 1 when nothing matched
    output = result.stdout if result.returncode == 0 else ""
    if not output.strip():
        output = "(no matches)"

    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n...[truncated]"

    header = (
        f"Search for string pattern '{pattern}' in {_display_path(search_path)}\n---\n"
    )
    return ToolExecution(
        status="ok",
        output=header + output,
        metadata={
            "returncode": result.returncode,
            "pattern": pattern,
        },
    )
