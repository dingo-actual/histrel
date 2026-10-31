#!/usr/bin/env python3
"""Repetition support for benchmark prediction runs.

A repeated run executes the same instance set N times. Each repetition is a
normal run in its own directory, ``<output_dir>/<run_name>/repeat_NN/``, so its
artifacts stay inspectable on their own. This module merges those directories
into aggregate files at ``<output_dir>/<run_name>/``:

- ``bench_agent_predictions.jsonl``: every prediction from every repetition,
  tagged with its 1-based ``repetition`` number.
- ``bench_agent_usage_aggregate.json``: per-instance and overall sum, mean,
  and median of each numeric usage metric.

    Important Functions:
        repeat_dir_name(index: int) -> str
        aggregate_repetitions(run_root, repeat_dirs) -> AggregatePaths
        collect_predictions(repeat_dirs) -> list[dict[str, Any]]
        collect_usage(repeat_dirs) -> dict[str, Any]
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any, NamedTuple

from benchmark.common.io import read_jsonl, write_json, write_jsonl
from loguru import logger

#: Predictions file name, used inside each repetition and for the aggregate.
PREDICTIONS_FILE_NAME = "bench_agent_predictions.jsonl"
#: Per-repetition usage file written by the runner.
USAGE_FILE_NAME = "bench_agent_usage.jsonl"
#: Aggregate usage file with sum, mean, and median across repetitions.
AGGREGATE_USAGE_FILE_NAME = "bench_agent_usage_aggregate.json"

#: Numeric UsageRecord fields that can be aggregated.
_STAT_FIELDS = (
    "execution_time",
    "tokens_input",
    "tokens_output",
    "tokens_cache_read",
    "tokens_cache_write",
    "tokens_total",
    "cost_total",
    "maximum_context",
)

#: Statistic names reported for each aggregated metric.
_STAT_NAMES = ("sum", "mean", "median")


class AggregatePaths(NamedTuple):
    """Paths to the aggregate artifacts of a repeated run.

    Instance Attributes:
        predictions (Path): Aggregate predictions JSONL path.
        usage (Path): Aggregate usage JSON path.
    """

    predictions: Path
    usage: Path


def repeat_dir_name(index: int) -> str:
    """Return the directory name of a repetition.

    Args:
        index (int): 1-based repetition index.

    Returns:
        (str) Repetition directory name, e.g. ``repeat_02``.
    """
    return f"repeat_{index:02d}"


def _read_records(directory: Path, file_name: str) -> list[dict[str, Any]]:
    """Read a JSONL artifact from a repetition directory.

    A repetition that produced no records (for example, every instance was
    skipped) is tolerated and contributes nothing.

    Args:
        directory (Path): Repetition directory.
        file_name (str): Artifact file name.

    Returns:
        (list[dict[str, Any]]) Records, or an empty list when the file is
        missing.
    """
    path = directory / file_name
    if not path.exists():
        logger.bind(
            component="benchmark.predict.repeat",
            origin="bench_harness_predict",
        ).warning(
            f"Aggregating repetition without {file_name}", directory=str(directory)
        )
        return []
    return read_jsonl(path)


def collect_predictions(repeat_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    """Collect predictions from every repetition.

    Args:
        repeat_dirs (Sequence[Path]): Repetition directories, in run order.

    Returns:
        (list[dict[str, Any]]) Predictions in repetition order, each with an
        added ``repetition`` field.
    """
    return [
        {**record, "repetition": index}
        for index, directory in enumerate(repeat_dirs, start=1)
        for record in _read_records(directory, PREDICTIONS_FILE_NAME)
    ]


def _metric_stats(values: Sequence[float]) -> dict[str, float]:
    """Return sum, mean, and median of a metric sample.

    Args:
        values (Sequence[float]): Non-empty sample of metric values.

    Returns:
        (dict[str, float]) Statistics keyed by stat name, rounded for
        readability.
    """
    stats = {
        "sum": math.fsum(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
    }
    return {name: round(stats[name], 4) for name in _STAT_NAMES}


def _summarize(samples: dict[str, list[float]], count: int) -> dict[str, Any]:
    """Build the sample statistics block for one scope.

    Args:
        samples (dict[str, list[float]]): Metric name to sampled values.
        count (int): Number of usage records summarized.

    Returns:
        (dict[str, Any]) Record count and per-metric statistics.
    """
    return {
        "count": count,
        "metrics": {
            field: _metric_stats(values) for field, values in samples.items() if values
        },
    }


def collect_usage(repeat_dirs: Sequence[Path]) -> dict[str, Any]:
    """Aggregate usage records across repetitions.

    Args:
        repeat_dirs (Sequence[Path]): Repetition directories, in run order.

    Returns:
        (dict[str, Any]) Per-instance statistics, overall statistics, and the
        repetition count.
    """
    per_instance: dict[str, dict[str, list[float]]] = {}
    per_instance_count: dict[str, int] = {}
    overall: dict[str, list[float]] = {field: [] for field in _STAT_FIELDS}
    overall_count = 0

    for directory in repeat_dirs:
        for record in _read_records(directory, USAGE_FILE_NAME):
            instance_id = record.get("instance_id")
            if not isinstance(instance_id, str):
                continue
            samples = per_instance.setdefault(
                instance_id, {field: [] for field in _STAT_FIELDS}
            )
            per_instance_count[instance_id] = per_instance_count.get(instance_id, 0) + 1
            overall_count += 1
            for field in _STAT_FIELDS:
                value = record.get(field)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    samples[field].append(float(value))
                    overall[field].append(float(value))

    return {
        "repetitions": len(repeat_dirs),
        "instances": {
            instance_id: _summarize(samples, per_instance_count[instance_id])
            for instance_id, samples in per_instance.items()
        },
        "overall": _summarize(overall, overall_count),
    }


def aggregate_repetitions(
    run_root: Path, repeat_dirs: Sequence[Path]
) -> AggregatePaths:
    """Write the aggregate artifacts of a repeated run.

    Args:
        run_root (Path): Run directory that holds the repetition directories.
        repeat_dirs (Sequence[Path]): Repetition directories, in run order.

    Returns:
        (AggregatePaths) Paths to the written aggregate artifacts.
    """
    paths = AggregatePaths(
        predictions=run_root / PREDICTIONS_FILE_NAME,
        usage=run_root / AGGREGATE_USAGE_FILE_NAME,
    )
    predictions = collect_predictions(repeat_dirs)
    write_jsonl(paths.predictions, predictions)
    write_json(paths.usage, collect_usage(repeat_dirs))

    logger.bind(
        component="benchmark.predict.repeat",
        origin="bench_harness_predict",
        event="repeat.aggregated",
    ).info(
        "Aggregated repetitions",
        repetitions=len(repeat_dirs),
        predictions=len(predictions),
        predictions_path=str(paths.predictions),
        usage_path=str(paths.usage),
    )
    return paths
