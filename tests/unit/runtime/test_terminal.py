#!/usr/bin/env python3
"""Unit tests for terminal conditions.

Tests budget exhaustion and the SessionOutcome model.
"""

from __future__ import annotations

import pytest

from histrel.terminal import (
    SessionOutcome,
    SessionStatus,
    TerminalEvaluator,
)
from tests.unit.factories import build_agent_config


class Test04Unit01Terminal01Budgets:
    """Budget exhaustion terminal conditions."""

    @pytest.mark.parametrize(
        "max_wall, elapsed, max_turns, turns, max_tools, tools, should_terminate",
        [
            # No budgets set — never terminates on budget
            (0, 100.0, 0, 50, 0, 30, False),
            # Wall budget exceeded
            (60, 65.0, 0, 0, 0, 0, True),
            # Turns budget exceeded
            (0, 0.0, 20, 25, 0, 0, True),
            # Tool calls budget exceeded
            (0, 0.0, 0, 0, 10, 15, True),
            # All budgets within limits
            (60, 30.0, 20, 10, 10, 5, False),
            # Exactly at budget — not exceeded
            (60, 60.0, 20, 20, 10, 10, False),
        ],
        ids=[
            "no_budgets",
            "wall_exceeded",
            "turns_exceeded",
            "tools_exceeded",
            "all_within",
            "exactly_at",
        ],
    )
    def test_04_unit_01_terminal_01_budget_01_exhaustion(
        self,
        max_wall: int,
        elapsed: float,
        max_turns: int,
        turns: int,
        max_tools: int,
        tools: int,
        should_terminate: bool,
    ) -> None:
        # Arrange
        config = build_agent_config(
            max_wall_seconds=max_wall,
            max_turns=max_turns,
            max_tool_calls=max_tools,
        )
        evaluator = TerminalEvaluator(config)

        # Act
        outcome = evaluator.evaluate_budgets(
            elapsed_seconds=elapsed,
            turn_count=turns,
            tool_call_count=tools,
        )

        # Assert
        if should_terminate:
            assert outcome is not None
            assert outcome.status == SessionStatus.BUDGET_EXHAUSTED
        else:
            assert outcome is None


class Test04Unit04Terminal04SessionOutcome:
    """SessionOutcome model."""

    @pytest.mark.parametrize(
        "status, reason",
        [
            (SessionStatus.SUCCEEDED, "change_applied"),
            (SessionStatus.FAILED, "model_error"),
            (SessionStatus.BUDGET_EXHAUSTED, "max_turns_reached"),
            (SessionStatus.INFRASTRUCTURE_ERROR, "journal_write_failed"),
            (SessionStatus.CANCELLED, "harness_terminated"),
        ],
        ids=["succeeded", "failed", "budget", "infra", "cancelled"],
    )
    def test_04_unit_04_terminal_04_outcome_01_all_statuses(
        self, status: SessionStatus, reason: str
    ) -> None:
        # Act
        outcome = SessionOutcome(status=status, reason=reason)

        # Assert
        assert outcome.status == status
        assert outcome.reason == reason
