#!/usr/bin/env python3
"""
Bash tool for the benchmark agent.

    Important Classes:
        BashTool: Run one shell command with bash.

    Important Variables:
        BASH_TIMEOUT_SECONDS (float): Wall-clock budget for one command.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Any

from histrel.tools.tool import (
    MAX_TOOL_OUTPUT_CHARS,
    Tool,
    ToolContext,
    ToolExecution,
)

# A command that outlives this budget is killed with its whole process
# group, so a hung command cannot stall the turn.
BASH_TIMEOUT_SECONDS: float = 30.0


class BashTool(Tool):
    """Run one shell command with bash.

    The command runs as ``bash -c <command>`` in the current working
    directory (the isolated workspace). Stdout and stderr are merged, so
    the agent reads errors in the order they happened, and stdin is
    closed, so a command that reads input sees EOF instead of blocking.
    Output is truncated after ``MAX_TOOL_OUTPUT_CHARS`` characters. A
    non-zero exit code is an informative error carrying the output and
    the code; a command that outlives ``BASH_TIMEOUT_SECONDS`` is killed
    with its process group and reported as a timeout.
    """

    name = "bash"
    definition: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a shell command with bash in the current working "
                "directory. Returns the combined stdout and stderr.\n"
                f"Results are truncated after {MAX_TOOL_OUTPUT_CHARS} "
                f"characters. Commands are killed after {BASH_TIMEOUT_SECONDS:g} "
                "seconds."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command to run with bash -c.",
                    },
                },
                "required": ["command"],
            },
        },
    }
    examples: list[str] = [
        'bash(command="python -m pytest -q")',
        'bash(command="git diff --stat")',
    ]

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        """Execute the bash tool.

        Args:
            arguments (dict[str, Any]): Tool arguments.
            context (ToolContext): Execution context.

        Returns:
            (ToolExecution) Execution result.
        """
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return ToolExecution(
                status="error",
                output="'command' is required",
                metadata={},
            )

        text, code, timed_out = _run_command(command)
        output = _truncate(text.rstrip("\n"))
        if timed_out:
            return ToolExecution(
                status="timeout",
                output=_with_timeout(output),
                metadata={"informative": True},
            )
        if code == 0:
            return ToolExecution(
                status="ok",
                output=output,
                metadata={"exit_code": 0},
            )
        return ToolExecution(
            status="error",
            output=_with_exit_code(output, code),
            metadata={"exit_code": code, "informative": True},
        )


def _run_command(command: str) -> tuple[str, int, bool]:
    """Run one command, bounding its wall time and its process group.

    Args:
        command (str): Shell command to run.

    Returns:
        (tuple[str, int, bool]) Combined output, exit code, and whether
        the command hit the timeout.
    """
    process = subprocess.Popen(
        ["bash", "-c", command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        start_new_session=True,
    )
    try:
        combined, _ = process.communicate(timeout=BASH_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        # Kill the whole group: a grandchild holding the output pipe
        # would otherwise block the drain below.
        os.killpg(process.pid, signal.SIGKILL)
        combined, _ = process.communicate()
        return combined, -1, True
    return combined, process.returncode, False


def _truncate(output: str) -> str:
    """Truncate output to the shared maximum, marking the cut."""
    if len(output) <= MAX_TOOL_OUTPUT_CHARS:
        return output
    return output[:MAX_TOOL_OUTPUT_CHARS] + "\n...[truncated]"


def _with_timeout(output: str) -> str:
    """Append the timeout notice to a killed command's output."""
    notice = f"Command timed out after {BASH_TIMEOUT_SECONDS:g}s and was killed."
    return f"{output}\n{notice}" if output else notice


def _with_exit_code(output: str, code: int) -> str:
    """Append the exit code line to a failing command's output."""
    body = output.rstrip("\n")
    return f"{body}\nExit code: {code}" if body else f"Exit code: {code}"
