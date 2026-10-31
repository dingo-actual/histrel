#!/usr/bin/env python3
"""Unit tests for the SWE-bench scoring CLI.

Naming follows the main suite taxonomy.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from benchmark.common.io import read_json, write_json, write_jsonl
from benchmark.predict.repeat import PREDICTIONS_FILE_NAME, repeat_dir_name
from benchmark.predict.resolution import RESOLUTION_FILE_NAME
from benchmark.score import cli


def _make_run(
    run_root: Path,
    repeats: int,
    *,
    dataset_id: str = "dataset/split",
    split: str = "test",
    instance_ids: list[str] | None = None,
) -> None:
    """Create a repeated run with predictions and run metadata."""
    for index in range(1, repeats + 1):
        directory = run_root / repeat_dir_name(index)
        directory.mkdir(parents=True)
        write_jsonl(
            directory / PREDICTIONS_FILE_NAME,
            [
                {
                    "instance_id": "inst-1",
                    "model_name_or_path": "bench-agent",
                    "model_patch": "diff --git a/x b/x\n",
                }
            ],
        )
    write_json(
        run_root / repeat_dir_name(1) / "run_metadata.json",
        {
            "dataset_id": dataset_id,
            "split": split,
            "instance_ids": instance_ids or ["inst-1", "inst-2"],
        },
    )


def _report(
    run_root: Path,
    run_id: str,
    resolved: list[str],
    *,
    total: int = 2,
    submitted: list[str] | None = None,
) -> dict[str, Any]:
    """Write a SWE-bench resolution report and return it."""
    submitted_ids = submitted if submitted is not None else ["inst-1", "inst-2"]
    report = {
        "total_instances": total,
        "submitted_instances": len(submitted_ids),
        "resolved_instances": len(resolved),
        "unresolved_instances": len(submitted_ids) - len(resolved),
        "empty_patch_instances": 0,
        "error_instances": 0,
        "submitted_ids": submitted_ids,
        "resolved_ids": resolved,
        "unresolved_ids": [
            instance_id for instance_id in submitted_ids if instance_id not in resolved
        ],
        "empty_patch_ids": [],
        "error_ids": [],
        "schema_version": 2,
    }
    write_json(run_root / f"bench-agent.{run_id}.json", report)
    return report


def _fake_run(
    monkeypatch: Any, calls: list[dict[str, Any]], returncode: int = 0
) -> None:
    """Replace the evaluation subprocess with a recorder."""

    def run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"command": command, **kwargs})
        return subprocess.CompletedProcess(command, returncode)

    monkeypatch.setattr(cli.subprocess, "run", run)


class Test04Unit09ScoreCli01Command:
    def test_04_unit_09_score_cli_01_command_01_arguments(self, tmp_path: Path) -> None:
        # Arrange
        predictions = tmp_path / "repeat_01" / PREDICTIONS_FILE_NAME

        # Act
        command = cli.evaluation_command(
            predictions, "my-run-repeat-01", "ds", "test", 3, 900
        )

        # Assert
        assert command[:3] == [
            sys.executable,
            "-m",
            "swebench.harness.run_evaluation",
        ]
        assert (
            command[command.index("--predictions_path") + 1] == predictions.as_posix()
        )
        assert command[command.index("--run_id") + 1] == "my-run-repeat-01"
        assert command[command.index("--dataset_name") + 1] == "ds"
        assert command[command.index("--split") + 1] == "test"
        assert command[command.index("--max_workers") + 1] == "3"
        assert command[command.index("--timeout") + 1] == "900"
        assert "--instance_ids" not in command

    def test_04_unit_09_score_cli_01_command_02_instance_ids(
        self, tmp_path: Path
    ) -> None:
        # Arrange — a subset run must grade only its own instances
        predictions = tmp_path / "repeat_01" / PREDICTIONS_FILE_NAME

        # Act
        command = cli.evaluation_command(
            predictions,
            "my-run-repeat-01",
            "ds",
            "test",
            3,
            900,
            ["inst-a", "inst-b"],
        )

        # Assert
        index = command.index("--instance_ids")
        assert command[index + 1 :] == ["inst-a", "inst-b"]


class Test04Unit09ScoreCli02Grading:
    def test_04_unit_09_score_cli_02_grading_01_grades_every_repeat(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Arrange
        run_root = tmp_path / "my-run"
        _make_run(run_root, repeats=2)
        calls: list[dict[str, Any]] = []
        _fake_run(monkeypatch, calls)

        # Act
        exit_code = cli.main([str(run_root)])

        # Assert
        assert exit_code == 0
        assert [
            call["command"][call["command"].index("--run_id") + 1] for call in calls
        ] == [
            "my-run-repeat-01",
            "my-run-repeat-02",
        ]
        assert all(call["cwd"] == run_root.resolve() for call in calls)
        assert (run_root / RESOLUTION_FILE_NAME).exists()

    def test_04_unit_09_score_cli_02_grading_02_skips_existing_report(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Arrange
        run_root = tmp_path / "my-run"
        _make_run(run_root, repeats=2)
        _report(run_root, "my-run-repeat-01", ["inst-1"])
        calls: list[dict[str, Any]] = []
        _fake_run(monkeypatch, calls)

        # Act
        exit_code = cli.main([str(run_root)])

        # Assert
        assert exit_code == 0
        assert len(calls) == 1
        assert calls[0]["command"][calls[0]["command"].index("--run_id") + 1] == (
            "my-run-repeat-02"
        )

    def test_04_unit_09_score_cli_02_grading_03_force_regrades(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Arrange
        run_root = tmp_path / "my-run"
        _make_run(run_root, repeats=1)
        _report(run_root, "my-run-repeat-01", ["inst-1"])
        calls: list[dict[str, Any]] = []
        _fake_run(monkeypatch, calls)

        # Act
        exit_code = cli.main([str(run_root), "--force"])

        # Assert
        assert exit_code == 0
        assert len(calls) == 1

    def test_04_unit_09_score_cli_02_grading_06_force_discards_cached_results(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Arrange — SWE-bench skips an instance that already has a cached
        # report, so a forced re-grade must delete that cache.
        run_root = tmp_path / "my-run"
        _make_run(run_root, repeats=1)
        cached_report = (
            run_root
            / cli.EVAL_LOG_DIR
            / "my-run-repeat-01"
            / "bench-agent"
            / "inst-1"
            / "report.json"
        )
        cached_report.parent.mkdir(parents=True)
        cached_report.write_text("{}", encoding="utf-8")
        stale_report = _report(run_root, "my-run-repeat-01", ["inst-1"])
        assert stale_report is not None
        calls: list[dict[str, Any]] = []
        _fake_run(monkeypatch, calls)

        # Act
        exit_code = cli.main([str(run_root), "--force"])

        # Assert
        assert exit_code == 0
        assert len(calls) == 1
        assert not cached_report.parent.exists()
        assert not (run_root / "bench-agent.my-run-repeat-01.json").exists()

    def test_04_unit_09_score_cli_02_grading_07_no_force_keeps_cached_results(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Arrange
        run_root = tmp_path / "my-run"
        _make_run(run_root, repeats=1)
        cached_report = (
            run_root
            / cli.EVAL_LOG_DIR
            / "my-run-repeat-01"
            / "bench-agent"
            / "inst-1"
            / "report.json"
        )
        cached_report.parent.mkdir(parents=True)
        cached_report.write_text("{}", encoding="utf-8")
        calls: list[dict[str, Any]] = []
        _fake_run(monkeypatch, calls)

        # Act
        exit_code = cli.main([str(run_root)])

        # Assert — the repetition is graded, so its cache is untouched.
        assert exit_code == 0
        assert len(calls) == 1
        assert cached_report.exists()

    def test_04_unit_09_score_cli_02_grading_08_failed_repetition_returns_one(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Arrange
        run_root = tmp_path / "my-run"
        _make_run(run_root, repeats=1)
        calls: list[dict[str, Any]] = []
        _fake_run(monkeypatch, calls, returncode=1)

        # Act
        exit_code = cli.main([str(run_root)])

        # Assert
        assert exit_code == 1
        assert (run_root / RESOLUTION_FILE_NAME).exists()

    def test_04_unit_09_score_cli_02_grading_05_no_repeats_returns_one(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Arrange
        run_root = tmp_path / "my-run"
        run_root.mkdir()
        calls: list[dict[str, Any]] = []
        _fake_run(monkeypatch, calls)

        # Act
        exit_code = cli.main([str(run_root)])

        # Assert
        assert exit_code == 1
        assert calls == []


class Test04Unit09ScoreCli03Defaults:
    def test_04_unit_09_score_cli_03_defaults_01_dataset_from_metadata(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Arrange
        run_root = tmp_path / "my-run"
        _make_run(run_root, repeats=1, dataset_id="princeton-nlp/SWE-bench_Lite")
        calls: list[dict[str, Any]] = []
        _fake_run(monkeypatch, calls)

        # Act
        cli.main([str(run_root)])

        # Assert
        command = calls[0]["command"]
        assert command[command.index("--dataset_name") + 1] == (
            "princeton-nlp/SWE-bench_Lite"
        )
        assert command[command.index("--split") + 1] == "test"

    def test_04_unit_09_score_cli_03_defaults_02_flags_override_metadata(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Arrange
        run_root = tmp_path / "my-run"
        _make_run(run_root, repeats=1)
        calls: list[dict[str, Any]] = []
        _fake_run(monkeypatch, calls)

        # Act
        cli.main(
            [
                str(run_root),
                "--dataset-name",
                "other/dataset",
                "--split",
                "dev",
                "--max-workers",
                "1",
            ]
        )

        # Assert
        command = calls[0]["command"]
        assert command[command.index("--dataset_name") + 1] == "other/dataset"
        assert command[command.index("--split") + 1] == "dev"
        assert command[command.index("--max_workers") + 1] == "1"

    def test_04_unit_09_score_cli_03_defaults_03_aggregates_when_only_reports_exist(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Arrange — both repetitions were graded in an earlier batch
        run_root = tmp_path / "my-run"
        _make_run(run_root, repeats=2)
        _report(run_root, "my-run-repeat-01", ["inst-1"])
        _report(run_root, "my-run-repeat-02", [])
        calls: list[dict[str, Any]] = []
        _fake_run(monkeypatch, calls)

        # Act
        exit_code = cli.main([str(run_root)])

        # Assert
        assert exit_code == 0
        assert calls == []
        aggregate = read_json(run_root / RESOLUTION_FILE_NAME)
        assert aggregate["repetitions"] == 2
        assert aggregate["resolve_rate"]["mean"] == 0.25

    def test_04_unit_09_score_cli_03_defaults_04_grades_metadata_instance_ids(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Arrange — a subset run must ask SWE-bench for its own instances
        run_root = tmp_path / "my-run"
        _make_run(run_root, repeats=1, instance_ids=["inst-a", "inst-b"])
        calls: list[dict[str, Any]] = []
        _fake_run(monkeypatch, calls)

        # Act
        cli.main([str(run_root)])

        # Assert
        command = calls[0]["command"]
        index = command.index("--instance_ids")
        assert command[index + 1 :] == ["inst-a", "inst-b"]

    def test_04_unit_09_score_cli_03_defaults_05_denominator_is_intended_set(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Arrange — the report totals the whole dataset (5) while the run
        # intended 2 instances and one of them was never predicted
        run_root = tmp_path / "my-run"
        _make_run(run_root, repeats=1, instance_ids=["inst-1", "inst-2"])
        _report(
            run_root,
            "my-run-repeat-01",
            ["inst-1"],
            total=5,
            submitted=["inst-1"],
        )
        calls: list[dict[str, Any]] = []
        _fake_run(monkeypatch, calls)

        # Act
        exit_code = cli.main([str(run_root)])

        # Assert — one of two intended instances resolved, not one of five
        assert exit_code == 0
        aggregate = read_json(run_root / RESOLUTION_FILE_NAME)
        assert aggregate["per_repetition"][0]["total_instances"] == 2
        assert aggregate["per_repetition"][0]["resolve_rate"] == 0.5
        assert aggregate["coverage"]["intended_instances"] == 2
        assert aggregate["coverage"]["missing_instances"] == ["inst-2"]
        assert aggregate["instances"]["inst-2"]["runs"] == 1

    def test_04_unit_09_score_cli_03_defaults_06_flags_unexpected_instances(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        # Arrange — a report that graded something outside the run
        run_root = tmp_path / "my-run"
        _make_run(run_root, repeats=1, instance_ids=["inst-1"])
        _report(run_root, "my-run-repeat-01", ["inst-9"], submitted=["inst-9"])
        calls: list[dict[str, Any]] = []
        _fake_run(monkeypatch, calls)

        # Act
        cli.main([str(run_root)])

        # Assert
        aggregate = read_json(run_root / RESOLUTION_FILE_NAME)
        assert aggregate["coverage"]["unexpected_instances"] == ["inst-9"]
