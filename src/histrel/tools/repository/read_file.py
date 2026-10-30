#!/usr/bin/env python3
"""
Read-file tool for the benchmark agent.

    Important Classes:
        ReadFileTool: Read file contents.
"""

from __future__ import annotations

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


class ReadFileTool(Tool):
    """Read file contents.

    Reads a file relative to CWD (or an absolute path under /tmp) and
    returns its contents with line numbers, formatted as
    "{line_num}|{line}". Each line number is zero-padded to the width
    of the last line shown, so all numbers share one fixed width.
    Successful output is prefixed with a
    "Contents of file {path} (lines A-B)" header line followed by a
    "---" separator, so the path and displayed line range stay visible
    in the tool response even when the call arguments are not replayed.
    """

    name = "read"
    definition: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "read",
            "description": (
                "Read the contents of a file. Returns lines formatted as "
                "'line_num|content'\n(1-indexed). Use start_line and end_line"
                " to read a specific range. Results are\ntruncated after "
                f"{MAX_TOOL_OUTPUT_CHARS} characters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": (
                            "Path to the file to read, relative to CWD "
                            "or an absolute path under /tmp."
                        ),
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Starting line number (1-indexed). Defaults to 1.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Ending line number (1-indexed, inclusive). Defaults to last line.",
                    },
                },
                "required": ["path"],
            },
        },
    }
    examples: list[str] = [
        'run(name="read", args="{\\"path\\": \\"src/main.py\\"}")',
        'run(name="read", args="{\\"path\\": \\"src/utils.py\\", \\"start_line\\": 10, \\"end_line\\": 50}")',
    ]

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        """Execute the read tool.

        Args:
            arguments (dict[str, Any]): Tool arguments.
            context (ToolContext): Execution context.

        Returns:
            (ToolExecution) Execution result.
        """
        path_str = arguments.get("path", "")
        if not path_str:
            return _missing_path_error()

        file_path = _resolve_read_file(path_str)
        if isinstance(file_path, ToolExecution):
            return file_path

        content = _read_utf8_text(file_path, path_str)
        if isinstance(content, ToolExecution):
            return content

        lines = content.splitlines()
        if not lines:
            return _empty_file_result(file_path)

        start_line = arguments.get("start_line", 1)
        end_line = arguments.get("end_line", len(lines))
        bounds = _resolve_line_bounds(start_line, end_line, len(lines), path_str)
        if isinstance(bounds, ToolExecution):
            return bounds
        start_idx, end_idx = bounds

        if start_idx >= len(lines):
            return _start_beyond_end_error(path_str, start_line, len(lines))

        formatted, truncated = _format_numbered_lines(
            lines[start_idx:end_idx], start_idx, end_idx
        )
        return _read_success(
            file_path, start_idx, end_idx, len(lines), formatted, truncated
        )


def _missing_path_error() -> ToolExecution:
    """Return the malformed-call error for a missing path argument."""
    return ToolExecution(
        status="error",
        output="'path' is required",
        metadata={},
    )


def _resolve_read_file(path_str: str) -> Path | ToolExecution:
    """Resolve the read target, or return an informative error.

    Args:
        path_str (str): Path from the model.

    Returns:
        (Path | ToolExecution) The resolved file path, or an error result.
    """
    try:
        file_path = resolve_path(path_str)
    except FileNotFoundError:
        return ToolExecution(
            status="error",
            output=(
                f"Attempt to read file '{path_str}' failed due to "
                "the file not existing."
            ),
            metadata={"path": path_str, "informative": True},
        )
    except ValueError as exc:
        return ToolExecution(
            status="error",
            output=(f"Attempt to read file '{path_str}' failed due to {exc}."),
            metadata={"path": path_str, "informative": True},
        )

    if not file_path.is_file():
        return ToolExecution(
            status="error",
            output=(
                f"Attempt to read file '{path_str}' failed due to "
                "the path not being a file."
            ),
            metadata={"path": path_str, "informative": True},
        )
    return file_path


def _read_utf8_text(file_path: Path, path_str: str) -> str | ToolExecution:
    """Read and decode a file as UTF-8, or return an informative error.

    Args:
        file_path (Path): Resolved file path.
        path_str (str): Original path string for messages.

    Returns:
        (str | ToolExecution) The decoded content, or an error result.
    """
    try:
        return file_path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return ToolExecution(
            status="error",
            output=(
                f"Attempt to read file '{path_str}' failed due to "
                "the file not being valid UTF-8."
            ),
            metadata={"path": path_str, "informative": True},
        )


def _empty_file_result(file_path: Path) -> ToolExecution:
    """Return the success result for an empty file."""
    return ToolExecution(
        status="ok",
        output=f"Contents of file {_display_path(file_path)} (empty file)\n---\n",
        metadata={"path": str(file_path), "lines": "(empty file)"},
    )


def _resolve_line_bounds(
    start_line: Any, end_line: Any, total: int, path_str: str
) -> tuple[int, int] | ToolExecution:
    """Resolve inclusive line arguments to slice bounds.

    Args:
        start_line (Any): Raw start_line argument.
        end_line (Any): Raw end_line argument.
        total (int): Total number of lines in the file.
        path_str (str): Original path string for error metadata.

    Returns:
        (tuple[int, int] | ToolExecution) Zero-based start and end indices,
        or a malformed-argument error.
    """
    try:
        start_idx = max(0, int(start_line) - 1)
        end_idx = min(total, int(end_line))
    except (ValueError, TypeError):
        return ToolExecution(
            status="error",
            output="start_line and end_line must be integers",
            metadata={"path": path_str},
        )
    return start_idx, end_idx


def _start_beyond_end_error(
    path_str: str, start_line: Any, total: int
) -> ToolExecution:
    """Return the informative error for a start line past the file end."""
    return ToolExecution(
        status="error",
        output=(
            f"Attempt to read file '{path_str}' from line {start_line} "
            f"failed due to the file having only {total} lines."
        ),
        metadata={"path": path_str, "informative": True},
    )


def _format_numbered_lines(
    selected: list[str], start_idx: int, end_idx: int
) -> tuple[str, bool]:
    """Format selected lines with zero-padded line numbers.

    Args:
        selected (list[str]): Lines to display.
        start_idx (int): Zero-based index of the first selected line.
        end_idx (int): Zero-based index one past the last selected line.

    Returns:
        (tuple[str, bool]) The formatted block and whether it was truncated.
    """
    line_num_width = len(str(end_idx))
    formatted = "\n".join(
        f"{line_num:0{line_num_width}d}|{line}"
        for line_num, line in enumerate(selected, start=start_idx + 1)
    )
    truncated = len(formatted) > MAX_TOOL_OUTPUT_CHARS
    if truncated:
        formatted = formatted[:MAX_TOOL_OUTPUT_CHARS] + "\n...[truncated]"
    return formatted, truncated


def _read_success(
    file_path: Path,
    start_idx: int,
    end_idx: int,
    total: int,
    formatted: str,
    truncated: bool,
) -> ToolExecution:
    """Assemble the successful read result."""
    line_info = f"({start_idx + 1}-{min(end_idx, total)} of {total} lines)"
    return ToolExecution(
        status="ok",
        output=(
            f"Contents of file {_display_path(file_path)} "
            f"(lines {start_idx + 1}-{end_idx}"
            f"{'; truncated' if truncated else ''})\n---\n{formatted}"
        ),
        metadata={"path": str(file_path), "lines": line_info},
    )
