#!/usr/bin/env python3
"""Unit tests for repeated benchmark runs and their aggregate artifacts.

Naming follows the main suite taxonomy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmark.common.io import write_jsonl
from benchmark.predict import cli
from benchmark.predict.config import PredictorConfig
from benchmark.predict.repeat import (
    AGGREGATE_USAGE_FILE_NAME,
    PREDICTIONS_FILE_NAME,
    USAGE_FILE_NAME,
    aggregate_repetitions,
    collect_predictions,
    collect_usage,
    repeat_dir_name,
)


def _prediction_row(instance_id: str, patch: str) -> dict[str, str]:
    """Build a prediction record for a repetition."""
    return {
        "instance_id": instance_id,
        "model_name_or_path": "bench-agent",
        "model_patch": patch,
    }


def _usage_row(instance_id: str, tokens_total: int) -> dict[str, Any]:
    """Build a usage record for a repetition."""
    return {
        "instance_id": instance_id,
        "execution_time": float(tokens_total),
        "tokens_input": tokens_total,
        "tokens_output": tokens_total * 2,
        "tokens_cache_read": 0,
        "tokens_cache_write": 0,
        "tokens_total": tokens_total,
        "cost_total": 0.0,
        "maximum_context": tokens_total * 10,
    }


def _write_repetition(
    run_root: Path,
    index: int,
    *,
    predictions: list[dict[str, str]] | None = None,
    usage: list[dict[str, Any]] | None = None,
) -> Path:
    """Write one repetition directory with its artifacts."""
    directory = run_root / repeat_dir_name(index)
    directory.mkdir(parents=True, exist_ok=True)
    if predictions is not None:
        write_jsonl(directory / PREDICTIONS_FILE_NAME, predictions)
    if usage is not None:
        write_jsonl(directory / USAGE_FILE_NAME, usage)
    return directory


class Test04Unit01Repeat01AggregatePredictions:
    def test_04_unit_01_repeat_01_aggregate_predictions_01_all_runs(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        first = _write_repetition(
            tmp_path,
            1,
            predictions=[
                _prediction_row("inst-1", "patch-a"),
                _prediction_row("inst-2", "patch-b"),
            ],
        )
        second = _write_repetition(
            tmp_path,
            2,
            predictions=[
                _prediction_row("inst-1", "patch-c"),
                _prediction_row("inst-2", "patch-d"),
            ],
        )

        # Act
        records = collect_predictions([first, second])

        # Assert
        assert [record["model_patch"] for record in records] == [
            "patch-a",
            "patch-b",
            "patch-c",
            "patch-d",
        ]
        assert [record["repetition"] for record in records] == [1, 1, 2, 2]

    def test_04_unit_01_repeat_01_aggregate_predictions_02_missing_file(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        empty = _write_repetition(tmp_path, 1)
        populated = _write_repetition(
            tmp_path, 2, predictions=[_prediction_row("inst-1", "patch-a")]
        )

        # Act
        records = collect_predictions([empty, populated])

        # Assert
        assert records == [
            {
                "instance_id": "inst-1",
                "model_name_or_path": "bench-agent",
                "model_patch": "patch-a",
                "repetition": 2,
            }
        ]


class Test04Unit01Repeat02AggregateUsage:
    def test_04_unit_01_repeat_02_aggregate_usage_01_sum_mean_median(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        dirs = [
            _write_repetition(tmp_path, 1, usage=[_usage_row("inst-1", 10)]),
            _write_repetition(tmp_path, 2, usage=[_usage_row("inst-1", 20)]),
            _write_repetition(tmp_path, 3, usage=[_usage_row("inst-1", 31)]),
        ]

        # Act
        stats = collect_usage(dirs)["instances"]["inst-1"]

        # Assert
        assert stats["count"] == 3
        tokens = stats["metrics"]["tokens_total"]
        assert tokens["sum"] == pytest.approx(61)
        assert tokens["mean"] == pytest.approx(20.3333)
        assert tokens["median"] == pytest.approx(20)

    def test_04_unit_01_repeat_02_aggregate_usage_02_overall(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        first = _write_repetition(
            tmp_path,
            1,
            usage=[_usage_row("inst-1", 10), _usage_row("inst-2", 5)],
        )
        second = _write_repetition(
            tmp_path,
            2,
            usage=[_usage_row("inst-1", 20), _usage_row("inst-2", 5)],
        )

        # Act
        summary = collect_usage([first, second])

        # Assert
        assert summary["repetitions"] == 2
        assert summary["overall"]["count"] == 4
        overall_tokens = summary["overall"]["metrics"]["tokens_total"]
        assert overall_tokens["sum"] == pytest.approx(40)
        assert overall_tokens["mean"] == pytest.approx(10)
        assert summary["instances"]["inst-2"]["count"] == 2
        assert summary["instances"]["inst-2"]["metrics"]["tokens_total"] == {
            "count": 2,
            "sum": 10.0,
            "mean": 5.0,
            "median": 5.0,
        }

    def test_04_unit_01_repeat_02_aggregate_usage_03_unknown_costs_skipped(
        self, tmp_path: Path
    ) -> None:
        # Arrange — one attempt has a known cost, one failed with an unknown
        # cost marked by the runner, and one carries the legacy -1 sentinel.
        known = {**_usage_row("inst-1", 10), "cost_total": 0.5, "cost_complete": True}
        unknown = {
            **_usage_row("inst-1", 30),
            "cost_total": None,
            "cost_complete": False,
        }
        legacy = {**_usage_row("inst-1", 50), "cost_total": -1.0}
        dirs = [
            _write_repetition(tmp_path, 1, usage=[known]),
            _write_repetition(tmp_path, 2, usage=[unknown]),
            _write_repetition(tmp_path, 3, usage=[legacy]),
        ]

        # Act
        summary = collect_usage(dirs)

        # Assert — only the known cost is an observation; the unknown costs
        # are counted, not averaged in as 0 or -1.
        overall = summary["overall"]
        assert overall["count"] == 3
        assert overall["unknown_costs"] == 2
        assert overall["cost_complete"] is False
        assert overall["metrics"]["cost_total"] == {
            "count": 1,
            "sum": 0.5,
            "mean": 0.5,
            "median": 0.5,
        }
        # Token statistics still use every attempt.
        assert overall["metrics"]["tokens_total"]["count"] == 3
        assert overall["metrics"]["tokens_total"]["sum"] == pytest.approx(90)


class Test04Unit09Repeat03AggregateArtifacts:
    def test_04_unit_09_repeat_03_aggregate_artifacts_01_written(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        dirs = [
            _write_repetition(
                tmp_path,
                index,
                predictions=[_prediction_row("inst-1", f"patch-{index}")],
                usage=[_usage_row("inst-1", index)],
            )
            for index in (1, 2)
        ]

        # Act
        paths = aggregate_repetitions(tmp_path, dirs)

        # Assert
        assert paths.predictions == tmp_path / PREDICTIONS_FILE_NAME
        assert paths.usage == tmp_path / AGGREGATE_USAGE_FILE_NAME
        assert paths.predictions.exists()
        assert paths.usage.exists()


class Test04Unit04Repeat04CliContract:
    def test_04_unit_04_repeat_04_cli_contract_01_repeat_default(self) -> None:
        # Arrange & Act
        args = cli.build_parser().parse_args([])

        # Assert
        assert args.repeat == 1

    def test_04_unit_04_repeat_04_cli_contract_02_repeat_must_be_positive(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        instances_path = tmp_path / "instances.jsonl"
        write_jsonl(instances_path, [{"instance_id": "inst-1"}])

        # Act & Assert
        with pytest.raises(SystemExit):
            cli.main(["--repeat", "0", "--instances", str(instances_path)])


class Test04Unit02Repeat05CliExecution:
    @pytest.fixture
    def recorded_runs(self, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
        """Patch the runner with a recording fake and return its call log."""
        calls: list[dict[str, object]] = []

        async def _fake_run_prediction(  # noqa: RUF029
            config: PredictorConfig, instances: list[dict], *, build_images: bool = True
        ) -> Path:
            run_name = config.run_name
            run_dir = Path(config.output_dir) / run_name
            calls.append({"run_name": run_name, "build_images": build_images})
            write_jsonl(
                run_dir / PREDICTIONS_FILE_NAME,
                [_prediction_row("inst-1", f"patch-{run_name}")],
            )
            write_jsonl(run_dir / USAGE_FILE_NAME, [_usage_row("inst-1", 10)])
            return run_dir

        monkeypatch.setattr(cli, "run_prediction", _fake_run_prediction)
        return calls

    @pytest.fixture
    def configured_instances(self, tmp_path: Path) -> tuple[Path, Path]:
        """Write a predictor config and an instances file."""
        config_path = tmp_path / "bench_settings.toml"
        config_path.write_text(
            "[benchmark]\n"
            f'output_dir = "{tmp_path.as_posix()}"\n'
            'run_name = "run-x"\n'
            "\n[benchmark.swe_bench]\n"
            "max_instances = 1\n",
            encoding="utf-8",
        )
        instances_path = tmp_path / "instances.jsonl"
        write_jsonl(instances_path, [{"instance_id": "inst-1"}])
        return config_path, instances_path

    def test_04_unit_02_repeat_05_cli_execution_01_builds_images_once(
        self,
        recorded_runs: list[dict[str, object]],
        configured_instances: tuple[Path, Path],
    ) -> None:
        # Arrange
        config_path, instances_path = configured_instances

        # Act
        exit_code = cli.main(
            [
                "--config",
                str(config_path),
                "--instances",
                str(instances_path),
                "--repeat",
                "3",
            ]
        )

        # Assert
        assert exit_code == 0
        assert [call["build_images"] for call in recorded_runs] == [
            True,
            False,
            False,
        ]

    def test_04_unit_02_repeat_05_cli_execution_02_aggregates_every_run(
        self,
        recorded_runs: list[dict[str, object]],
        configured_instances: tuple[Path, Path],
    ) -> None:
        # Arrange
        config_path, instances_path = configured_instances

        # Act
        cli.main(
            [
                "--config",
                str(config_path),
                "--instances",
                str(instances_path),
                "--repeat",
                "2",
            ]
        )

        # Assert
        run_root = config_path.parent / "run-x"
        assert [call["run_name"] for call in recorded_runs] == [
            "repeat_01",
            "repeat_02",
        ]
        predictions = (
            (run_root / PREDICTIONS_FILE_NAME).read_text(encoding="utf-8").splitlines()
        )
        assert len(predictions) == 2
        assert "patch-repeat_01" in predictions[0]
        assert "patch-repeat_02" in predictions[1]
