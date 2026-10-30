#!/usr/bin/env python3
"""Unit tests for the bash tool.

The tool runs one string command through bash and reports its combined
output plus the exit code.

    Test Classes:
        Test04Unit06Tools06BashCommand: Command execution, output capture,
            exit status, and argument validation.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path

from histrel.tools.registry import build_all_tools
from histrel.tools.repository import BashTool
from histrel.tools.tool import MAX_TOOL_OUTPUT_CHARS
from tests.unit.tools.helpers import make_context


class Test04Unit06Tools06BashCommand:
    """bash tool behavior."""

    def test_04_unit_06_tools_06_bash_01_morphism_01_returns_stdout(self) -> None:
        """A succeeding command returns its stdout and an ok status."""
        # Act
        execution = BashTool().execute({"command": "echo hello"}, make_context())

        # Assert
        assert execution.status == "ok"
        assert execution.output == "hello"
        assert execution.metadata["exit_code"] == 0

    def test_04_unit_06_tools_06_bash_01_morphism_02_returns_stderr(self) -> None:
        """Stderr is captured as output, not discarded."""
        # Act
        execution = BashTool().execute({"command": "echo oops >&2"}, make_context())

        # Assert
        assert execution.status == "ok"
        assert execution.output == "oops"

    def test_04_unit_06_tools_06_bash_02_boundary_01_preserves_line_order(self) -> None:
        """Multi-line output is returned in the order the command wrote it."""
        # Act
        execution = BashTool().execute({"command": "printf 'a\\nb\\n'"}, make_context())

        # Assert
        assert execution.output == "a\nb"

    def test_04_unit_06_tools_06_bash_03_failure_01_reports_exit_code(self) -> None:
        """A failing command is an informative error carrying its exit code."""
        # Act
        execution = BashTool().execute(
            {"command": "echo broken; exit 3"}, make_context()
        )

        # Assert
        assert execution.status == "error"
        assert execution.metadata["exit_code"] == 3
        assert execution.metadata["informative"] is True
        assert "broken" in str(execution.output)
        assert "Exit code: 3" in str(execution.output)

    def test_04_unit_06_tools_06_bash_04_boundary_02_syntax_error_does_not_raise(
        self,
    ) -> None:
        """A bash syntax error yields an error result instead of an exception."""
        # Act
        execution = BashTool().execute({"command": "if ["}, make_context())

        # Assert
        assert execution.status == "error"
        assert execution.metadata["exit_code"] != 0

    def test_04_unit_06_tools_06_bash_05_boundary_03_truncates_long_output(
        self,
    ) -> None:
        """Output longer than the shared maximum is truncated."""
        # Act
        execution = BashTool().execute(
            {"command": "head -c 9000 /dev/zero | tr '\\0' 'a'"}, make_context()
        )

        # Assert
        output = str(execution.output)
        assert len(output) < 9000
        assert output.endswith("...[truncated]")
        assert len(output) <= MAX_TOOL_OUTPUT_CHARS + len("\n...[truncated]")

    def test_04_unit_06_tools_06_bash_06_isolation_01_runs_in_cwd(
        self, isolated_workspace: Path
    ) -> None:
        """Commands run in the current working directory."""
        # Act
        execution = BashTool().execute({"command": "pwd -P"}, make_context())

        # Assert
        assert Path(str(execution.output).strip()).resolve() == (
            isolated_workspace.resolve()
        )

    def test_04_unit_06_tools_06_bash_07_contract_01_missing_command(self) -> None:
        """A call without a command argument is a malformed-call error."""
        # Act
        execution = BashTool().execute({}, make_context())

        # Assert
        assert execution.status == "error"
        assert execution.output == "'command' is required"
        assert "informative" not in execution.metadata

    def test_04_unit_06_tools_06_bash_07_contract_02_empty_command(self) -> None:
        """An empty command argument is a malformed-call error."""
        # Act
        execution = BashTool().execute({"command": "   "}, make_context())

        # Assert
        assert execution.status == "error"
        assert execution.output == "'command' is required"

    def test_04_unit_06_tools_06_bash_08_observability_01_definition(self) -> None:
        """The definition exposes exactly one required string parameter."""
        # Act
        function = BashTool.definition["function"]
        parameters = function["parameters"]

        # Assert
        assert function["name"] == "bash"
        assert list(parameters["properties"]) == ["command"]
        assert parameters["properties"]["command"]["type"] == "string"
        assert parameters["required"] == ["command"]

    def test_04_unit_06_tools_06_bash_08_observability_02_registered(self) -> None:
        """The registry offers the bash tool to the agent."""
        # Act
        registry = build_all_tools()

        # Assert
        assert "bash" in registry.all_tools

    def test_04_unit_06_tools_06_bash_09_failure_02_kills_timed_out_command(
        self, set_bash_timeout: Callable[[float], None]
    ) -> None:
        """A command that outlives the timeout is killed and reported."""
        # Arrange
        set_bash_timeout(0.5)

        # Act
        started = time.monotonic()
        execution = BashTool().execute({"command": "sleep 30"}, make_context())
        elapsed = time.monotonic() - started

        # Assert
        assert execution.status == "timeout"
        assert execution.metadata["informative"] is True
        assert "timed out" in str(execution.output)
        assert elapsed < 10

    def test_04_unit_06_tools_06_bash_09_failure_03_kills_whole_process_group(
        self, set_bash_timeout: Callable[[float], None]
    ) -> None:
        """A grandchild holding the output pipe cannot block the timeout."""
        # Arrange
        set_bash_timeout(0.5)

        # Act
        started = time.monotonic()
        execution = BashTool().execute({"command": "sleep 30 | cat"}, make_context())
        elapsed = time.monotonic() - started

        # Assert
        assert execution.status == "timeout"
        assert elapsed < 10

    def test_04_unit_06_tools_06_bash_10_boundary_04_stdin_is_closed(self) -> None:
        """A command reading stdin sees EOF instead of blocking the agent."""
        # Act
        started = time.monotonic()
        execution = BashTool().execute({"command": "cat"}, make_context())
        elapsed = time.monotonic() - started

        # Assert
        assert execution.status == "ok"
        assert execution.output == ""
        assert elapsed < 10
