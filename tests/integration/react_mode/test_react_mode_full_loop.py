#!/usr/bin/env python3
"""Integration tests for a full deterministic react-mode solve.

A full scripted solve with direct tool calls keeps the react request
shape — growing history and the static tool set — and terminates with
SUCCEEDED while fixing the fixture repository.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from histrel.core.runtime import run_agent
from histrel.terminal import SessionStatus
from tests.integration.helpers import (
    USER_MSG,
    PatchedRuntime,
    make_config,
)
from tests.integration.react_mode.helpers import (
    REACT_SYSTEM_PROMPT,
    SESSION_TOOL_NAMES,
    make_react_config,
)
from tests.integration.scripting import (
    solve_script,
    tool_call,
)


class Test01Integration06ReactMode01DeterministicFullLoop:
    """Deterministic full-loop react-mode solve."""

    def test_01_integration_06_react_mode_01_deterministic_03_full_loop_01_end_to_end_solve(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        """A full scripted solve with direct tool calls terminates with
        SUCCEEDED and fixes the fixture repository.
        """
        # Arrange
        config = make_react_config(make_config())
        session = patched_runtime(config)
        calculator = str(session.workspace / "calculator.py")
        fake_complete, _ = solve_script(session.runtime, calculator, tool_call)
        monkeypatch.setattr(session.adapter, "complete", fake_complete)

        # Act
        outcome = run_agent(USER_MSG, session.runtime)

        # Assert — session succeeded and the bug is fixed.
        assert outcome.status == SessionStatus.SUCCEEDED
        assert "return a + b" in (session.workspace / "calculator.py").read_text(
            encoding="utf-8"
        )

    def test_01_integration_06_react_mode_01_deterministic_03_full_loop_02_requests_keep_react_shape(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        """Every recorded call keeps the react shape: system first, the
        user message second, history strictly growing by appending, and
        the static tool set on every call.
        """
        # Arrange
        config = make_react_config(make_config())
        session = patched_runtime(config)
        calculator = str(session.workspace / "calculator.py")
        fake_complete, records = solve_script(session.runtime, calculator, tool_call)
        monkeypatch.setattr(session.adapter, "complete", fake_complete)

        # Act
        run_agent(USER_MSG, session.runtime)

        # Assert
        assert len(records) >= 3
        for request in records:
            assert request.messages[0].role == "system"
            assert request.messages[0].content == REACT_SYSTEM_PROMPT
            assert request.messages[1].role == "user"
            assert request.messages[1].content == USER_MSG
            tool_names = {tool.function["name"] for tool in request.tools}
            assert tool_names == SESSION_TOOL_NAMES
        for earlier, later in zip(records, records[1:], strict=False):
            expected_prefix = list(earlier.messages)
            assert list(later.messages)[: len(expected_prefix)] == expected_prefix

    def test_01_integration_06_react_mode_01_deterministic_03_full_loop_03_projected_tools(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        """Every iteration gets the same static tool set, and the final
        request replays the two prior exchanges.
        """
        # Arrange
        config = make_react_config(make_config())
        session = patched_runtime(config)
        calculator = str(session.workspace / "calculator.py")
        fake_complete, records = solve_script(session.runtime, calculator, tool_call)
        monkeypatch.setattr(session.adapter, "complete", fake_complete)

        # Act
        run_agent(USER_MSG, session.runtime)

        # Assert
        assert len(records) >= 3
        first_tool_names = {tool.function["name"] for tool in records[0].tools}
        assert first_tool_names == SESSION_TOOL_NAMES
        later_tool_names = {tool.function["name"] for tool in records[1].tools}
        assert later_tool_names == first_tool_names

        # The final request replays the two prior (assistant, tool
        # result) pairs after the shared system and user messages.
        assert len(records[2].messages) == 6
