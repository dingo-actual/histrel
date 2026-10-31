#!/usr/bin/env python3
"""Unit tests for per-attempt usage accounting in the predictor runner.

A scheduled attempt must stay visible to the aggregate even when the
container run raises before reporting any usage: it is recorded with unknown
measurements and an unknown cost, never as a free or zero-token attempt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from benchmark.common.io import read_jsonl
from benchmark.predict.config import PredictorConfig
from benchmark.predict.runner import (
    _empty_usage_accumulator,
    _run_agent_instance,
)
from benchmark.predict.types import SessionUsage, get_accumulator_from_agent_data

_INSTANCE_ID = "django__django-11099"


class _FailingDockerRunner:
    """Docker runner double whose container run always raises."""

    def __init__(self, error: str) -> None:
        self.error = error

    def run_agent(
        self,
        instance: dict[str, Any],
        image: str,
        config: PredictorConfig,
        *,
        timeout: int,
        repetition: int,
    ) -> tuple[Path, dict, int]:
        raise RuntimeError(self.error)


def _session_usage() -> SessionUsage:
    """Return zeroed session totals."""
    return {
        "session_id": None,
        "none": _empty_usage_accumulator(),
        "total": _empty_usage_accumulator(),
    }


def _run_failing_attempt(
    *,
    usage_file: Path,
    predictions_file: Path,
    session_usage: SessionUsage,
    predictor_config: PredictorConfig,
) -> None:
    """Run one attempt whose container execution raises."""
    _run_agent_instance(
        run_name="run-x",
        instance={"instance_id": _INSTANCE_ID},
        instance_id=_INSTANCE_ID,
        image="swebench/sweb.env.py.x86_64.abc123:latest",
        config=predictor_config,
        docker_runner=_FailingDockerRunner(  # type: ignore[arg-type]
            "container exited before writing usage"
        ),
        predictions_file=predictions_file,
        usage_file=usage_file,
        session_usage=session_usage,
        repetition=1,
    )


class Test04Unit02RunnerAccounting01FailedAttempt:
    def test_04_unit_02_runner_accounting_01_failed_attempt_01_record_written(
        self, tmp_path: Path, predictor_config: PredictorConfig
    ) -> None:
        # Arrange
        usage_file = tmp_path / "bench_agent_usage.jsonl"
        predictions_file = tmp_path / "bench_agent_predictions.jsonl"

        # Act
        _run_failing_attempt(
            usage_file=usage_file,
            predictions_file=predictions_file,
            session_usage=_session_usage(),
            predictor_config=predictor_config,
        )

        # Assert
        records = read_jsonl(usage_file)
        assert len(records) == 1
        assert records[0]["instance_id"] == _INSTANCE_ID

    def test_04_unit_02_runner_accounting_01_failed_attempt_02_measurements_unknown(
        self, tmp_path: Path, predictor_config: PredictorConfig
    ) -> None:
        # Arrange
        usage_file = tmp_path / "bench_agent_usage.jsonl"

        # Act
        _run_failing_attempt(
            usage_file=usage_file,
            predictions_file=tmp_path / "bench_agent_predictions.jsonl",
            session_usage=_session_usage(),
            predictor_config=predictor_config,
        )

        # Assert
        record = read_jsonl(usage_file)[0]
        assert record["cost_total"] is None
        assert record["cost_complete"] is False
        assert record["tokens_total"] == -1
        assert record["execution_time"] == pytest.approx(-1.0)

    def test_04_unit_02_runner_accounting_01_failed_attempt_03_session_totals_counted(
        self, tmp_path: Path, predictor_config: PredictorConfig
    ) -> None:
        # Arrange
        session_usage = _session_usage()

        # Act
        _run_failing_attempt(
            usage_file=tmp_path / "bench_agent_usage.jsonl",
            predictions_file=tmp_path / "bench_agent_predictions.jsonl",
            session_usage=session_usage,
            predictor_config=predictor_config,
        )

        # Assert
        none_totals = get_accumulator_from_agent_data(session_usage, "none")
        total_totals = get_accumulator_from_agent_data(session_usage, "total")
        assert none_totals is not None
        assert total_totals is not None
        assert none_totals["cost_unknown_instances"] == 1
        assert total_totals["cost_unknown_instances"] == 1
        assert total_totals["tokens_total"] == 0

    def test_04_unit_02_runner_accounting_01_failed_attempt_04_no_prediction(
        self, tmp_path: Path, predictor_config: PredictorConfig
    ) -> None:
        # Arrange
        predictions_file = tmp_path / "bench_agent_predictions.jsonl"

        # Act
        _run_failing_attempt(
            usage_file=tmp_path / "bench_agent_usage.jsonl",
            predictions_file=predictions_file,
            session_usage=_session_usage(),
            predictor_config=predictor_config,
        )

        # Assert — the attempt stays retryable on resume because it produced
        # no prediction.
        assert not predictions_file.exists()
