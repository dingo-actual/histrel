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
        tokens_used: int = 0,
    ) -> SessionOutcome | None:
        """Evaluate whether any budget is exhausted.

        A budget of 0 means unlimited. A budget is exhausted when the
        current count strictly exceeds the budget, except the token
        budget, which is exhausted on reaching the working limit so a
        turn cannot overrun it.

        Args:
            elapsed_seconds (float): Elapsed wall-clock seconds.
            turn_count (int): Number of model turns completed.
            tool_call_count (int): Number of tool calls executed.
            tokens_used (int): Prompt plus completion tokens used so far.
                Defaults to 0.

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

        working_budget = self._working_token_budget()
        if working_budget > 0 and tokens_used >= working_budget:
            return SessionOutcome(
                status=SessionStatus.BUDGET_EXHAUSTED,
                reason=(
                    f"token_budget ({budgets.token_budget}) exhausted: "
                    f"{tokens_used} tokens used, working limit "
                    f"{working_budget} after {budgets.token_budget_buffer} "
                    "reserved for finalization"
                ),
            )

        return None

    def _working_token_budget(self) -> int:
        """Return the token count at which the session must stop.

        The reserve in ``token_budget_buffer`` keeps room below the total
        budget for the session to finish the turn in progress.

        Returns:
            (int) Working token limit; 0 means unlimited.
        """
        budgets = self._config.budgets
        if budgets.token_budget <= 0:
            return 0
        return budgets.token_budget - budgets.token_budget_buffer
