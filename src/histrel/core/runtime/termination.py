#!/usr/bin/env python3
"""
Terminal-condition evaluation for the agent loop.

Evaluates whether the session should terminate on budget exhaustion.

    Important Functions:
        evaluate_terminal_conditions(runtime) -> SessionOutcome | None:
            Terminal outcome, or None to continue.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from histrel.core.clock import monotonic_now
from histrel.terminal import SessionOutcome

if TYPE_CHECKING:
    from histrel.core.runtime.runtime import Runtime

log = logger.bind(component="bench_agent.runtime")


def evaluate_terminal_conditions(runtime: Runtime) -> SessionOutcome | None:
    """Evaluate whether the session should terminate.

    Checks budget exhaustion.

    Args:
        runtime (Runtime): The runtime session state.

    Returns:
        (SessionOutcome | None) Outcome if terminal, None otherwise.
    """
    elapsed = monotonic_now() - runtime.session_start
    return runtime.terminal_evaluator.evaluate_budgets(
        elapsed_seconds=elapsed,
        turn_count=runtime.turn_count,
        tool_call_count=runtime.tool_call_count,
    )
