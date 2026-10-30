#!/usr/bin/env python3
"""Unit tests for the tool protocol model (ToolContext, ToolExecution)."""

from __future__ import annotations

from histrel.tools.tool import (
    ToolExecution,
)
from tests.unit.tools.helpers import (
    make_context,
)


class Test04Unit01Tools01Protocol:
    """Tool protocol model validation."""

    def test_04_unit_01_tools_01_protocol_01_tool_context_defaults(self) -> None:
        # Act
        ctx = make_context()

        # Assert
        assert ctx.message_index == 0
        assert ctx.session.task_id == "test-task"

    def test_04_unit_01_tools_01_protocol_02_tool_execution_ok(self) -> None:
        # Act
        exec = ToolExecution(status="ok", output="result")

        # Assert
        assert exec.status == "ok"
        assert exec.output == "result"
        assert exec.metadata == {}

    def test_04_unit_01_tools_01_protocol_03_tool_execution_error(self) -> None:
        # Act
        exec = ToolExecution(
            status="error",
            output="file not found",
            metadata={"path": "missing.py"},
        )

        # Assert
        assert exec.status == "error"
        assert exec.metadata["path"] == "missing.py"
