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


class Test04Unit01Terminal02TokenBudgets:
    """Token budget exhaustion terminal conditions."""

    @pytest.mark.parametrize(
        "token_budget, reserve, tokens_used, should_terminate",
        [
            # No token budget set — never terminates on tokens
            (0, 0, 10_000_000, False),
            # Below the working limit
            (10_000, 1_000, 8_999, False),
            # At the working limit — stops before overrunning the total
            (10_000, 1_000, 9_000, True),
            # Zero reserve makes the total the working limit
            (10_000, 0, 10_000, True),
            # Zero reserve, below the total
            (10_000, 0, 9_999, False),
        ],
        ids=[
            "unlimited",
            "below_working_limit",
            "at_working_limit",
            "no_reserve_at_total",
            "no_reserve_below_total",
        ],
    )
    def test_04_unit_01_terminal_02_token_budget_01_exhaustion(
        self,
        token_budget: int,
        reserve: int,
        tokens_used: int,
        should_terminate: bool,
    ) -> None:
        # Arrange
        config = build_agent_config(
            token_budget=token_budget,
            token_budget_buffer=reserve,
        )
        evaluator = TerminalEvaluator(config)

        # Act
        outcome = evaluator.evaluate_budgets(
            elapsed_seconds=0.0,
            turn_count=1,
            tool_call_count=1,
            tokens_used=tokens_used,
        )

        # Assert
        if should_terminate:
            assert outcome is not None
            assert outcome.status == SessionStatus.BUDGET_EXHAUSTED
            assert "token_budget" in outcome.reason
        else:
            assert outcome is None

    def test_04_unit_01_terminal_02_token_budget_02_reason_names_reserve(self) -> None:
        # Arrange
        config = build_agent_config(token_budget=1000, token_budget_buffer=400)
        evaluator = TerminalEvaluator(config)

        # Act
        outcome = evaluator.evaluate_budgets(
            elapsed_seconds=0.0,
            turn_count=1,
            tool_call_count=0,
            tokens_used=600,
        )

        # Assert — the working limit and the reserve are both reported
        assert outcome is not None
        assert "working limit 600" in outcome.reason
        assert "400" in outcome.reason

    def test_04_unit_01_terminal_02_token_budget_03_reserve_must_be_below_total(
        self,
    ) -> None:
        # Act / Assert — a reserve that swallows the budget is rejected
        with pytest.raises(ValueError, match="smaller than token_budget"):
            build_agent_config(token_budget=1_000, token_budget_buffer=1_000)


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
