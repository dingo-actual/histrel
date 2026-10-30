#!/usr/bin/env python3
"""
Per-tool invocation metrics for the benchmark agent.

Tracks invocation counts, successes, failures, and total elapsed time
per tool name.

    Important Classes:
        ToolMetrics: Per-tool invocation tracker.
"""

from __future__ import annotations


class ToolMetrics:
    """Per-tool invocation tracker.

    Instance Attributes:
        _tools (dict[str, dict[str, int]]): Per-tool metric counters.
    """

    def __init__(self) -> None:
        """Initialize an empty metrics tracker."""
        self._tools: dict[str, dict[str, int]] = {}

    def record(
        self,
        tool_name: str,
        *,
        elapsed_ms: int,
        success: bool,
    ) -> None:
        """Record a tool invocation.

        Args:
            tool_name (str): Name of the tool.
            elapsed_ms (int): Elapsed time in milliseconds.
            success (bool): Whether the invocation succeeded.
        """
        entry = self._tools.setdefault(
            tool_name,
            {
                "count": 0,
                "successes": 0,
                "failures": 0,
                "total_elapsed_ms": 0,
            },
        )
        entry["count"] += 1
        entry["total_elapsed_ms"] += elapsed_ms
        if success:
            entry["successes"] += 1
        else:
            entry["failures"] += 1

    def summary(self) -> dict[str, dict[str, int]]:
        """Return per-tool metric summaries.

        Returns:
            (dict[str, dict[str, int]]) Tool name to metrics dict.
        """
        return {name: dict(metrics) for name, metrics in self._tools.items()}
