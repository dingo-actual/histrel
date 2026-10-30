#!/usr/bin/env python3
"""Unit tests for timings, tool metrics, and artifact paths.

Tests wall-clock category tracking, per-tool invocation metrics,
and artifact path structure. Uses real objects (no mocks).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from histrel.context.session_artifacts import ArtifactPaths
from histrel.context.timings import Timings
from histrel.context.tool_metrics import ToolMetrics


class Test04Unit01Timings01Categories:
    """Wall-clock category tracking."""

    @pytest.mark.parametrize(
        "category",
        [
            "session_wall_seconds",
            "model_wall_seconds",
            "test_wall_seconds",
            "repository_tool_wall_seconds",
            "compaction_wall_seconds",
            "harness_overhead_wall_seconds",
        ],
        ids=[
            "session",
            "model",
            "test",
            "repo_tool",
            "compaction",
            "harness",
        ],
    )
    def test_04_unit_01_timings_01_cat_01_all_categories_initialized(
        self, category: str
    ) -> None:
        # Act
        timings = Timings()

        # Assert
        assert timings.get(category) == 0.0

    def test_04_unit_01_timings_01_cat_02_add_increments(self) -> None:
        # Arrange
        timings = Timings()

        # Act
        timings.add("model_wall_seconds", 1.5)
        timings.add("model_wall_seconds", 2.5)

        # Assert
        assert timings.get("model_wall_seconds") == 4.0

    def test_04_unit_01_timings_01_cat_03_as_dict(self) -> None:
        # Arrange
        timings = Timings()
        timings.add("session_wall_seconds", 10.0)
        timings.add("model_wall_seconds", 5.0)

        # Act
        d = timings.as_dict()

        # Assert
        assert d["session_wall_seconds"] == 10.0
        assert d["model_wall_seconds"] == 5.0
        assert "harness_overhead_wall_seconds" in d

    def test_04_unit_01_timings_01_cat_04_metadata_block(self) -> None:
        # Act
        timings = Timings()

        # Assert — wall-clock priority metadata is included
        meta = timings.metadata_block()
        assert meta["wall_clock_metric_priority"] == "secondary"
        assert "qualification" in meta


class Test04Unit02ToolMetrics01Tracking:
    """Per-tool invocation metrics."""

    def test_04_unit_02_toolmetrics_01_track_01_records_invocation(self) -> None:
        # Arrange
        metrics = ToolMetrics()

        # Act
        metrics.record(tool_name="read_file", elapsed_ms=100, success=True)

        # Assert
        summary = metrics.summary()
        assert "read_file" in summary
        assert summary["read_file"]["count"] == 1
        assert summary["read_file"]["successes"] == 1
        assert summary["read_file"]["failures"] == 0

    def test_04_unit_02_toolmetrics_01_track_02_accumulates_multiple(self) -> None:
        # Arrange
        metrics = ToolMetrics()

        # Act
        metrics.record("read_file", elapsed_ms=100, success=True)
        metrics.record("read_file", elapsed_ms=200, success=True)
        metrics.record("write_file", elapsed_ms=500, success=False)

        # Assert
        summary = metrics.summary()
        assert summary["read_file"]["count"] == 2
        assert summary["read_file"]["successes"] == 2
        assert summary["write_file"]["count"] == 1
        assert summary["write_file"]["failures"] == 1

    def test_04_unit_02_toolmetrics_01_track_03_total_elapsed(self) -> None:
        # Arrange
        metrics = ToolMetrics()

        # Act
        metrics.record("read_file", elapsed_ms=100, success=True)
        metrics.record("read_file", elapsed_ms=300, success=True)

        # Assert
        summary = metrics.summary()
        assert summary["read_file"]["total_elapsed_ms"] == 400

    def test_04_unit_02_toolmetrics_01_track_04_empty_summary(self) -> None:
        # Act
        metrics = ToolMetrics()
        summary = metrics.summary()

        # Assert
        assert summary == {}


class Test04Unit03Artifacts01Paths:
    """Artifact directory path structure."""

    def test_04_unit_03_artifacts_01_paths_01_all_paths_defined(self) -> None:
        # Act
        paths = ArtifactPaths(base_dir=Path("/output/session-1/artifacts"))

        # Assert
        assert paths.manifest_resolved == Path(
            "/output/session-1/artifacts/manifest.resolved.json"
        )
        assert paths.messages == Path("/output/session-1/artifacts/messages.jsonl")
        assert paths.token_usage_calls == Path(
            "/output/session-1/artifacts/token_usage_calls.jsonl"
        )
        assert paths.costs == Path("/output/session-1/artifacts/costs.json")
        assert paths.timings == Path("/output/session-1/artifacts/timings.json")
        assert paths.tool_metrics == Path(
            "/output/session-1/artifacts/tool_metrics.json"
        )
        assert paths.outcome == Path("/output/session-1/artifacts/outcome.json")
        assert paths.repository_diff == Path(
            "/output/session-1/artifacts/repository.diff"
        )
        assert paths.compactions == Path(
            "/output/session-1/artifacts/compactions.jsonl"
        )
        assert paths.result_manifest == Path(
            "/output/session-1/artifacts/result_manifest.json"
        )
