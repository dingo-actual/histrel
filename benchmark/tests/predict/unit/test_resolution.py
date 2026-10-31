#!/usr/bin/env python3
"""Unit tests for resolution aggregation across repeated benchmark runs.

Naming follows the main suite taxonomy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from benchmark.common.io import read_json, write_json
from benchmark.predict.repeat import repeat_dir_name
from benchmark.predict.resolution import (
    RESOLUTION_FILE_NAME,
    aggregate_resolution,
    repeat_eval_run_id,
    summarize_resolution,
)


def _report(
    total: int,
    resolved: list[str],
    unresolved: list[str],
    *,
    errors: list[str] | None = None,
) -> dict[str, Any]:
    """Build a SWE-bench resolution report."""
    return {
        "total_instances": total,
        "submitted_instances": len(resolved) + len(unresolved),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(unresolved),
        "empty_patch_instances": 0,
        "error_instances": len(errors or []),
        "submitted_ids": sorted([*resolved, *unresolved, *(errors or [])]),
        "resolved_ids": sorted(resolved),
        "unresolved_ids": sorted(unresolved),
        "empty_patch_ids": [],
        "error_ids": sorted(errors or []),
        "schema_version": 2,
    }


def _write_reports(run_root: Path, reports: dict[int, dict[str, Any]]) -> None:
    """Write one SWE-bench report per repetition into the run directory."""
    for index, report in reports.items():
        run_id = repeat_eval_run_id(run_root.name, index)
        write_json(run_root / f"bench-agent.{run_id}.json", report)


class Test04Unit08Resolution01Summary:
    def test_04_unit_08_resolution_01_summary_01_resolve_rate_stats(self) -> None:
        # Arrange
        reports = [
            ("run-repeat-01", _report(4, ["a", "b"], ["c", "d"])),
            ("run-repeat-02", _report(4, ["a"], ["b", "c", "d"])),
        ]

        # Act
        summary = summarize_resolution(reports)

        # Assert
        assert summary["repetitions"] == 2
        assert summary["per_repetition"][0]["resolve_rate"] == 0.5
        assert summary["per_repetition"][1]["resolve_rate"] == 0.25
        assert summary["resolve_rate"]["mean"] == 0.375
        assert summary["resolve_rate"]["median"] == 0.375
        assert summary["resolve_rate"]["min"] == 0.25
        assert summary["resolve_rate"]["max"] == 0.5

    def test_04_unit_08_resolution_01_summary_02_instance_counts(self) -> None:
        # Arrange
        reports = [
            ("run-repeat-01", _report(2, ["a"], ["b"])),
            ("run-repeat-02", _report(2, ["a"], ["b"])),
            ("run-repeat-03", _report(2, [], ["a", "b"])),
        ]

        # Act
        summary = summarize_resolution(reports)

        # Assert
        assert summary["instances"]["a"] == {
            "runs": 3,
            "resolved": 2,
            "resolve_rate": 0.6667,
        }
        assert summary["instances"]["b"] == {
            "runs": 3,
            "resolved": 0,
            "resolve_rate": 0.0,
        }

    def test_04_unit_08_resolution_01_summary_03_errors_count_as_submitted(
        self,
    ) -> None:
        # Arrange
        reports = [("run-repeat-01", _report(2, [], [], errors=["a"]))]

        # Act
        summary = summarize_resolution(reports)

        # Assert
        assert summary["instances"]["a"]["runs"] == 1
        assert summary["instances"]["a"]["resolved"] == 0
        assert summary["per_repetition"][0]["error_instances"] == 1

    def test_04_unit_08_resolution_01_summary_04_empty_report_list(self) -> None:
        # Act
        summary = summarize_resolution([])

        # Assert
        assert summary["repetitions"] == 0
        assert summary["resolve_rate"]["count"] == 0
        assert summary["instances"] == {}


class Test04Unit08Resolution02Aggregate:
    def test_04_unit_08_resolution_02_aggregate_01_writes_file(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        run_root = tmp_path / "my-run"
        for index in (1, 2):
            (run_root / repeat_dir_name(index)).mkdir(parents=True)
        _write_reports(
            run_root,
            {
                1: _report(2, ["a"], ["b"]),
                2: _report(2, ["a", "b"], []),
            },
        )

        # Act
        path = aggregate_resolution(run_root)

        # Assert
        assert path == run_root / RESOLUTION_FILE_NAME
        written = read_json(path)
        assert written["repetitions"] == 2
        assert written["resolve_rate"]["mean"] == 0.75
        assert written["instances"]["b"]["resolved"] == 1

    def test_04_unit_08_resolution_02_aggregate_02_matches_run_id_file_name(
        self, tmp_path: Path
    ) -> None:
        # Arrange — the report file name carries both model and evaluation run ID
        run_root = tmp_path / "my-run"
        (run_root / repeat_dir_name(1)).mkdir(parents=True)
        write_json(
            run_root / "bench-agent.my-run-repeat-01.json",
            _report(1, ["a"], []),
        )

        # Act
        path = aggregate_resolution(run_root)

        # Assert
        assert read_json(path)["instances"]["a"]["resolved"] == 1

    def test_04_unit_08_resolution_02_aggregate_03_ignores_other_run_ids(
        self, tmp_path: Path
    ) -> None:
        # Arrange — a report of a different evaluation run must not be counted
        run_root = tmp_path / "my-run"
        (run_root / repeat_dir_name(1)).mkdir(parents=True)
        write_json(run_root / "bench-agent.other-run.json", _report(1, ["a"], []))

        # Act
        path = aggregate_resolution(run_root)

        # Assert
        assert read_json(path)["repetitions"] == 0

    def test_04_unit_08_resolution_02_aggregate_03_skips_missing_report(
        self, tmp_path: Path
    ) -> None:
        # Arrange — repetition 2 was never graded
        run_root = tmp_path / "my-run"
        for index in (1, 2, 3):
            (run_root / repeat_dir_name(index)).mkdir(parents=True)
        _write_reports(run_root, {1: _report(1, ["a"], []), 3: _report(1, [], ["a"])})

        # Act
        path = aggregate_resolution(run_root)

        # Assert
        written = read_json(path)
        assert written["repetitions"] == 2
        assert [entry["run_id"] for entry in written["per_repetition"]] == [
            "my-run-repeat-01",
            "my-run-repeat-03",
        ]
        assert written["per_repetition"][1]["repetition"] == 3

    def test_04_unit_08_resolution_02_aggregate_05_accepts_custom_reports_root(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        run_root = tmp_path / "my-run"
        (run_root / repeat_dir_name(1)).mkdir(parents=True)
        reports_root = tmp_path / "elsewhere"
        reports_root.mkdir()
        write_json(
            reports_root / "bench-agent.my-run-repeat-01.json",
            _report(1, ["a"], []),
        )

        # Act
        path = aggregate_resolution(run_root, reports_root)

        # Assert
        assert read_json(path)["instances"]["a"]["resolved"] == 1
