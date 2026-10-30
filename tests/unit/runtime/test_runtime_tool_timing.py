#!/usr/bin/env python3
"""Unit tests for tool-execution wall-clock timing attribution.

Every dispatched tool call is attributed to the repository-tool timing
category, so the session's wall time adds up instead of leaking into an
unaccounted remainder.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from histrel.core.runtime import Runtime
from histrel.model.types import NormalizedToolCall
from tests.unit.runtime.stubs import EchoTool


class Test04Unit03Runtime03ToolTiming:
    """Tool executions attribute their wall time to the timing category."""

    def test_04_unit_03_runtime_03_timing_01_repository_tool_time_recorded(
        self,
        make_runtime: Callable[..., Runtime],
        tmp_path: Path,
    ) -> None:
        # Arrange — an ordinary repository tool
        runtime = make_runtime()
        runtime.ordinary_tools = [EchoTool(runtime, tmp_path, changes_repo=False)]

        # Act
        runtime._execute_tool_call(
            NormalizedToolCall(id="call_1", name="echo", arguments='{"text":"hi"}')
        )

        # Assert — the tool's wall time is attributed, not left at zero
        assert runtime.timings.get("repository_tool_wall_seconds") > 0.0
