#!/usr/bin/env python3
"""
Terminal conditions for the benchmark agent.

Evaluates budget exhaustion. The loop also ends when a model response
carries no tool calls and the repository differs from its committed
state, which the loop reports directly.

    Important Classes:
        SessionStatus: Enumeration of terminal session statuses.
        SessionOutcome: Terminal session outcome.
        TerminalEvaluator: Evaluates budget-based terminal conditions.

    Important Functions:
        None.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from histrel.core.config import AgentConfig


class SessionStatus(StrEnum):
    """Terminal session status."""

    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    CANCELLED = "cancelled"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


class SessionOutcome(BaseModel):
    """Terminal session outcome.

    Instance Attributes:
        status (SessionStatus): Terminal status.
        reason (str): Human-readable reason for the outcome.
    """

    status: SessionStatus
    reason: str


class TerminalEvaluator:
    """Evaluates budget-based terminal conditions.

    Instance Attributes:
        _config (AgentConfig): Agent configuration with budget limits.
    """

    def __init__(self, config: AgentConfig) -> None:
        """Initialize the evaluator.

        Args:
            config (AgentConfig): Agent configuration.
        """
        self._config = config

    def evaluate_budgets(
        self,
        *,
        elapsed_seconds: float,
        turn_count: int,
        tool_call_count: int,
    ) -> SessionOutcome | None:
        """Evaluate whether any budget is exhausted.

        A budget of 0 means unlimited. A budget is exhausted when the
        current count strictly exceeds the budget.

        Args:
            elapsed_seconds (float): Elapsed wall-clock seconds.
            turn_count (int): Number of model turns completed.
            tool_call_count (int): Number of tool calls executed.

        Returns:
            (SessionOutcome | None) Outcome if a budget is exhausted,
            None otherwise.
        """
        budgets = self._config.budgets

        if budgets.max_wall_seconds > 0 and elapsed_seconds > budgets.max_wall_seconds:
            return SessionOutcome(
                status=SessionStatus.BUDGET_EXHAUSTED,
                reason=f"max_wall_seconds ({budgets.max_wall_seconds}) exceeded: {elapsed_seconds:.1f}s elapsed",
            )

        if budgets.max_turns > 0 and turn_count > budgets.max_turns:
            return SessionOutcome(
                status=SessionStatus.BUDGET_EXHAUSTED,
                reason=f"max_turns ({budgets.max_turns}) exceeded: {turn_count} turns",
            )

        if budgets.max_tool_calls > 0 and tool_call_count > budgets.max_tool_calls:
            return SessionOutcome(
                status=SessionStatus.BUDGET_EXHAUSTED,
                reason=f"max_tool_calls ({budgets.max_tool_calls}) exceeded: {tool_call_count} calls",
            )

        return None
