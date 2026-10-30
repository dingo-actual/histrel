#!/usr/bin/env python3
"""Integration tests for a full deterministic user-message solve.

A full scripted solve through the run envelope keeps the two-message
request shape on every call, tracks the iteration counter, and
terminates with SUCCEEDED while fixing the fixture repository.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from histrel.core.journal import load_journal
from histrel.core.runtime import run_agent
from histrel.model.messages import ToolRequestPart
from histrel.model.types import ModelRequest
from histrel.terminal import SessionOutcome, SessionStatus
from tests.integration.helpers import (
    USER_MSG,
    PatchedRuntime,
    make_config,
    with_distinct_prompts,
)
from tests.integration.react_mode.helpers import SESSION_TOOL_NAMES
from tests.integration.scripting import (
    section,
    solve_script,
    tool_call,
)


def _solve(
    monkeypatch: pytest.MonkeyPatch,
    session: PatchedRuntime,
) -> tuple[SessionOutcome, list[ModelRequest]]:
    """Patch the session with a full-solve script and run the loop."""
    calculator = str(session.workspace / "calculator.py")
    fake_complete, records = solve_script(session.runtime, calculator, tool_call)
    monkeypatch.setattr(session.adapter, "complete", fake_complete)
    outcome = run_agent(USER_MSG, session.runtime)
    return outcome, records


class Test01Integration05UserMessages01DeterministicFullLoop:
    """Deterministic full-loop solve through the run envelope."""

    def test_01_integration_05_user_messages_01_deterministic_05_full_loop_01_end_to_end_solve(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        """A full scripted solve terminates with SUCCEEDED and fixes the
        fixture repository.
        """
        # Arrange
        config = with_distinct_prompts(make_config())
        session = patched_runtime(config)

        # Act
        outcome, _ = _solve(monkeypatch, session)

        # Assert — session succeeded and the bug is fixed.
        assert outcome.status == SessionStatus.SUCCEEDED
        assert "return a + b" in (session.workspace / "calculator.py").read_text(
            encoding="utf-8"
        )

    def test_01_integration_05_user_messages_01_deterministic_05_full_loop_02_requests_stay_stateless(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        """Every call keeps the two-message stateless shape with the
        static tool set, and the user message tracks the iteration
        counter.
        """
        # Arrange
        config = with_distinct_prompts(make_config())
        session = patched_runtime(config)

        # Act
        _, records = _solve(monkeypatch, session)

        # Assert
        assert len(records) >= 3
        for request in records[:3]:
            assert len(request.messages) == 2
            roles = [message.role for message in request.messages]
            assert roles == ["system", "user"]
            assert {
                tool.function["name"] for tool in request.tools
            } == SESSION_TOOL_NAMES
        for number, request in enumerate(records[1:3], start=2):
            assert f"Current iteration: {number}" in request.messages[1].content

    def test_01_integration_05_user_messages_01_deterministic_05_full_loop_03_journal_and_usage(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        """The journal records assistant tool-call messages before tool
        results, and model usage was tracked.
        """
        # Arrange
        config = with_distinct_prompts(make_config())
        session = patched_runtime(config)

        # Act
        _solve(monkeypatch, session)

        # Assert
        messages = list(load_journal(session.runtime.journal._path))
        assert any(
            message.role == "assistant"
            and any(isinstance(p, ToolRequestPart) for p in message.content)
            for message in messages
        )
        assert any(message.role == "tool" for message in messages)
        assert session.runtime.token_ledger.total_input > 0
        assert session.runtime.token_ledger.total_output > 0

    def test_01_integration_05_user_messages_01_deterministic_05_full_loop_04_edit_observation(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        """The edit observation names the call with its arguments and
        carries the tool's output under a separator.
        """
        # Arrange
        config = with_distinct_prompts(make_config())
        session = patched_runtime(config)
        calculator = str(session.workspace / "calculator.py")

        # Act
        _, records = _solve(monkeypatch, session)

        # Assert
        final_content = records[2].messages[1].content
        final_observations = section(final_content, "Observations")
        assert f'[Iteration 2] edit(file_path="{calculator}", ' in final_observations
        assert "file_edits=" in final_observations
        assert "---" in final_observations
        assert f"Edited file {calculator}" in final_observations
