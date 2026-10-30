#!/usr/bin/env python3
"""
Wall-clock timing instrumentation for the benchmark agent.

Tracks per-category wall-clock durations using a monotonic clock.
Categories may overlap (e.g., compaction may overlap with other work).

    Important Classes:
        Timings: Wall-clock category tracker.
"""

from __future__ import annotations

from typing import Any

_WALL_CLOCK_CATEGORIES = (
    "session_wall_seconds",
    "model_wall_seconds",
    "repository_tool_wall_seconds",
    "compaction_wall_seconds",
    "harness_overhead_wall_seconds",
)

_METADATA_BLOCK: dict[str, Any] = {
    "wall_clock_metric_priority": "secondary",
    "qualification": (
        "Elapsed time is diagnostic rather than a primary efficacy measure."
    ),
}


class Timings:
    """Wall-clock category tracker.

    Instance Attributes:
        _categories (dict[str, float]): Accumulated seconds per category.
    """

    def __init__(self) -> None:
        """Initialize all categories to zero."""
        self._categories: dict[str, float] = dict.fromkeys(_WALL_CLOCK_CATEGORIES, 0.0)

    def add(self, category: str, seconds: float) -> None:
        """Add elapsed seconds to a category.

        Args:
            category (str): Timing category name.
            seconds (float): Seconds to add.
        """
        if category in self._categories:
            self._categories[category] += seconds

    def get(self, category: str) -> float:
        """Return accumulated seconds for a category.

        Args:
            category (str): Timing category name.

        Returns:
            (float) Accumulated seconds, or 0.0 if unknown.
        """
        return self._categories.get(category, 0.0)

    def as_dict(self) -> dict[str, float]:
        """Return all categories as a dict.

        Returns:
            (dict[str, float]) Category name to seconds.
        """
        return dict(self._categories)

    def metadata_block(self) -> dict[str, Any]:
        """Return the wall-clock priority metadata block.

        Returns:
            (dict[str, Any]) Metadata describing wall-clock metric priority.
        """
        return dict(_METADATA_BLOCK)
