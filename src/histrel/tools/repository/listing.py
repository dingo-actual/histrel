#!/usr/bin/env python3
"""
Directory-listing and glob tools for the benchmark agent.

    Important Classes:
        ListDirTool: List directory contents.
        GlobTool: Find files and directories with fd.
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


class ListDirTool(Tool):
    """List directory contents.

    Lists files and directories at a given path relative to CWD or an
    absolute path under /tmp. Successful output is prefixed with a
    "Contents of directory {path}" header line followed by a "---"
    separator, so the path stays visible in the tool response even
    when the call arguments are not replayed.
    """

    name = "list_dir"
    definition: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": ("List files and directories at a given path."),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory path, relative to CWD or an "
                            "absolute path under /tmp. Defaults to CWD."
                        ),
                    },
                },
                "required": [],
            },
        },
    }
    examples: list[str] = [
        'run(name="list_dir", args="{\\"path\\": \\"src/\\"}")',
        'run(name="list_dir", args="{}")',
    ]

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        """Execute the list_dir tool.

        Args:
            arguments (dict[str, Any]): Tool arguments.
            context (ToolContext): Execution context.

        Returns:
            (ToolExecution) Execution result.
        """
        path_str = arguments.get("path", ".")
        try:
            dir_path = resolve_path(path_str)
        except FileNotFoundError:
            return ToolExecution(
                status="error",
                output=(
                    f"Attempt to list directory '{path_str}' failed due to "
                    "the directory not existing."
                ),
                metadata={"path": path_str, "informative": True},
            )
        except ValueError as exc:
            return ToolExecution(
                status="error",
                output=(f"Attempt to list directory '{path_str}' failed due to {exc}."),
                metadata={"path": path_str, "informative": True},
            )

        if not dir_path.is_dir():
            return ToolExecution(
                status="error",
                output=(
                    f"Attempt to list directory '{path_str}' failed due to "
                    "the path not being a directory."
                ),
                metadata={"path": path_str, "informative": True},
            )

        entries: list[str] = []
        for entry in sorted(dir_path.iterdir()):
            if entry.is_dir():
                entries.append(f"{entry.name}/")
            else:
                entries.append(entry.name)

        output = "\n".join(entries)
        if len(output) > MAX_TOOL_OUTPUT_CHARS:
            output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n...[truncated]"

        return ToolExecution(
            status="ok",
            output=f"Contents of directory {_display_path(dir_path)}\n---\n{output}",
            metadata={"path": str(dir_path), "count": len(entries)},
        )


class GlobTool(Tool):
    """Find files and directories with glob patterns.

    Runs file search against a glob pattern and optional path filter.
    Returns matching file and directory paths. A pattern without a path
    separator is matched against the file or directory name; a pattern
    containing one is matched against the path relative to the search
    directory. Successful output is prefixed with a
    "Search for file pattern '{pattern}' in {path}" header line
    followed by a "---" separator, so the search parameters stay
    visible in the tool response even when the call arguments are not
    replayed.
    """

    name = "glob"
    definition: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "glob",
            "description": (
                "Find files and directories matching a glob pattern. "
                "A pattern without '/' matches the file or directory "
                "name (e.g. '*.py' matches 'src/a.py'); a pattern with "
                "'/' matches the path relative to the search directory "
                "(e.g. '**/sub/*.py')."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Glob pattern (e.g. '*.py' for a name match, "
                            "'**/sub/*.py' for a path match)."
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Directory path to search in, relative to CWD or "
                            "an absolute path under /tmp.\nDefaults to CWD."
                        ),
                    },
                },
                "required": ["pattern"],
            },
        },
    }
    examples: list[str] = [
        'run(name="glob", args="{\\"pattern\\": \\"*.py\\", \\"path\\": \\"src/\\"}")',
        'run(name="glob", args="{\\"pattern\\": \\"**/*.json\\"}")',
        'run(name="glob", args="{\\"pattern\\": \\"**/test_distinct*/**\\"}")',
    ]

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        """Execute the glob tool.

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

        path_str = arguments.get("path")
        search_path = _resolve_glob_path(pattern, path_str)
        if isinstance(search_path, ToolExecution):
            return search_path

        # A pattern with a path separator is matched against the path
        # relative to the search directory; a bare name pattern is
        # matched against the name, where --full-path would find nothing.
        flags = ["--glob", "--full-path"] if "/" in pattern else ["--glob"]
        result = _run_glob_command(
            ["fd", *flags, pattern, str(search_path)], pattern, search_path
        )
        if isinstance(result, ToolExecution):
            return result
        return _shape_glob_result(result, pattern, search_path)


def _glob_path_error(pattern: str, path_str: str, reason: str) -> ToolExecution:
    """Build the informative error for an unusable glob path."""
    return ToolExecution(
        status="error",
        output=(
            f"Attempt to search for file pattern '{pattern}' in "
            f"'{path_str}' failed due to {reason}."
        ),
        metadata={
            "pattern": pattern,
            "path": path_str,
            "informative": True,
        },
    )


def _resolve_glob_path(pattern: str, path_str: Any) -> Path | ToolExecution:
    """Resolve the glob search path, defaulting to CWD.

    Args:
        pattern (str): Glob pattern for error messages.
        path_str (Any): Raw path argument (may be absent).

    Returns:
        (Path | ToolExecution) The resolved directory, or an error result.
    """
    if not path_str:
        return Path.cwd()
    try:
        return resolve_path(path_str)
    except FileNotFoundError:
        return _glob_path_error(pattern, path_str, "the search path not existing")
    except ValueError as exc:
        return _glob_path_error(pattern, path_str, str(exc))


def _run_glob_command(
    cmd: list[str], pattern: str, search_path: Path
) -> subprocess.CompletedProcess[str] | ToolExecution:
    """Run fd, mapping backend failures to tool errors.

    Args:
        cmd (list[str]): Full fd command.
        pattern (str): Glob pattern for messages.
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
                f"Attempt to search for file pattern '{pattern}' in "
                f"'{_display_path(search_path)}' failed due to the "
                "search backend not being available."
            ),
            metadata={"pattern": pattern, "informative": True},
        )
    except subprocess.TimeoutExpired:
        return ToolExecution(
            status="timeout",
            output=(
                f"Attempt to search for file pattern '{pattern}' in "
                f"'{_display_path(search_path)}' failed due to the "
                "search timing out after 30 seconds."
            ),
            metadata={"pattern": pattern, "informative": True},
        )


def _shape_glob_result(
    result: subprocess.CompletedProcess[str], pattern: str, search_path: Path
) -> ToolExecution:
    """Convert an fd process result into a tool execution result."""
    if result.returncode != 0:
        detail = result.stderr or (f"the search exiting with code {result.returncode}")
        return ToolExecution(
            status="error",
            output=(
                f"Attempt to search for file pattern '{pattern}' in "
                f"'{_display_path(search_path)}' failed due to: {detail}"
            ),
            metadata={
                "returncode": result.returncode,
                "pattern": pattern,
                "informative": True,
            },
        )

    output = result.stdout
    if not output.strip():
        output = "(no matches)"

    if len(output) > MAX_TOOL_OUTPUT_CHARS:
        output = output[:MAX_TOOL_OUTPUT_CHARS] + "\n...[truncated]"

    header = (
        f"Search for file pattern '{pattern}' in {_display_path(search_path)}\n---\n"
    )
    return ToolExecution(
        status="ok",
        output=header + output,
        metadata={
            "returncode": result.returncode,
            "pattern": pattern,
        },
    )
