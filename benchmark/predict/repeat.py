#!/usr/bin/env python3
"""Repetition support for benchmark prediction runs.

A repeated run executes the same instance set N times. Each repetition is a
normal run in its own directory, ``<output_dir>/<run_name>/repeat_NN/``, so its
artifacts stay inspectable on their own. This module merges those directories
into aggregate files at ``<output_dir>/<run_name>/``:

- ``bench_agent_predictions.jsonl``: every prediction from every repetition,
  tagged with its 1-based ``repetition`` number. This merged file is an
  analysis artifact, not a grading input: SWE-bench keys predictions by
  ``instance_id``, so grading it would keep only one prediction per instance.
  Grade each ``repeat_NN/bench_agent_predictions.jsonl`` under its own
  evaluation run ID, then aggregate with ``benchmark.predict.resolution``.
- ``bench_agent_usage_aggregate.json``: per-instance and overall sample
  count, sum, mean, and median of each numeric usage metric, plus the
  number of records whose cost is unknown. Records with an unknown cost
  (missing, negative, or marked incomplete) are not cost observations, so
  they never pull the reported cost average toward zero.

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
from benchmark.common.schemas import read_run_metadata
from benchmark.predict.usage_helpers import known_cost, known_measurement
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


def _scheduled_instances(directory: Path) -> list[str]:
    """Return the instance IDs a repetition was scheduled to run.

    Args:
        directory (Path): Repetition directory.

    Returns:
        (list[str]) Scheduled instance IDs, or an empty list when the
        repetition wrote no run metadata to describe its work.
    """
    path = directory / "run_metadata.json"
    if not path.exists():
        return []
    try:
        return list(read_run_metadata(path).instance_ids)
    except (OSError, ValueError) as exc:
        logger.bind(
            component="benchmark.predict.repeat",
            event="repeat.metadata.unreadable",
            origin="bench_harness_predict",
        ).warning(
            f"Cannot reconcile repetition against run metadata: {exc}",
            directory=str(directory),
        )
        return []


def _metric_stats(values: Sequence[float]) -> dict[str, float | int]:
    """Return the sample size, sum, mean, and median of a metric.

    Args:
        values (Sequence[float]): Non-empty sample of metric values.

    Returns:
        (dict[str, float | int]) Statistics keyed by stat name, rounded for
        readability.
    """
    stats: dict[str, float | int] = {"count": len(values)}
    stats["sum"] = round(math.fsum(values), 4)
    stats["mean"] = round(statistics.fmean(values), 4)
    stats["median"] = round(statistics.median(values), 4)
    return stats


def _cost_is_unknown(record: dict[str, Any]) -> bool:
    """Return whether a usage record's cost is not an observation.

    A cost is unknown when it is absent, negative (the legacy -1 sentinel),
    or explicitly marked incomplete by the runner.

    Args:
        record (dict[str, Any]): One usage record.

    Returns:
        (bool) True when the record must not contribute to cost statistics.
    """
    return (
        record.get("cost_complete") is False
        or known_cost(record.get("cost_total")) is None
    )


def _summarize(
    samples: dict[str, list[float]],
    count: int,
    unknown_costs: int,
    missing_attempts: int = 0,
) -> dict[str, Any]:
    """Build the sample statistics block for one scope.

    Args:
        samples (dict[str, list[float]]): Metric name to sampled values.
        count (int): Number of usage records summarized.
        unknown_costs (int): Number of records with an unknown cost.
        missing_attempts (int): Number of scheduled attempts without a
            record. Defaults to 0, used when the schedule is unknown.

    Returns:
        (dict[str, Any]) Record count, cost completeness flag, unknown-cost
        count, missing-attempt count, and per-metric statistics.
    """
    return {
        "count": count,
        "cost_complete": unknown_costs == 0 and missing_attempts == 0,
        "unknown_costs": unknown_costs,
        "missing_attempts": missing_attempts,
        "metrics": {
            field: _metric_stats(values) for field, values in samples.items() if values
        },
    }


def collect_usage(repeat_dirs: Sequence[Path]) -> dict[str, Any]:
    """Aggregate usage records across repetitions.

    Every scheduled instance/repetition pair is accounted for: pairs with a
    record contribute samples, pairs without one are counted as
    ``missing_attempts`` (the attempt never started).

    Args:
        repeat_dirs (Sequence[Path]): Repetition directories, in run order.

    Returns:
        (dict[str, Any]) Per-instance statistics, overall statistics, and the
        repetition count. ``count`` plus ``missing_attempts`` is the number
        of scheduled attempts.
    """
    per_instance: dict[str, dict[str, list[float]]] = {}
    per_instance_count: dict[str, int] = {}
    per_instance_unknown_costs: dict[str, int] = {}
    per_instance_missing: dict[str, int] = {}
    overall: dict[str, list[float]] = {field: [] for field in _STAT_FIELDS}
    overall_count = 0
    overall_unknown_costs = 0
    overall_missing = 0

    for directory in repeat_dirs:
        recorded: set[str] = set()
        for record in _read_records(directory, USAGE_FILE_NAME):
            instance_id = record.get("instance_id")
            if not isinstance(instance_id, str):
                continue
            recorded.add(instance_id)
            samples = per_instance.setdefault(
                instance_id, {field: [] for field in _STAT_FIELDS}
            )
            per_instance_count[instance_id] = per_instance_count.get(instance_id, 0) + 1
            overall_count += 1
            cost_unknown = _cost_is_unknown(record)
            if cost_unknown:
                overall_unknown_costs += 1
                per_instance_unknown_costs[instance_id] = (
                    per_instance_unknown_costs.get(instance_id, 0) + 1
                )
            for field in _STAT_FIELDS:
                if field == "cost_total" and cost_unknown:
                    continue
                measurement = known_measurement(record.get(field))
                if measurement is None:
                    continue
                samples[field].append(float(measurement))
                overall[field].append(float(measurement))

        for instance_id in _scheduled_instances(directory):
            if instance_id in recorded:
                continue
            per_instance.setdefault(instance_id, {field: [] for field in _STAT_FIELDS})
            per_instance_missing[instance_id] = (
                per_instance_missing.get(instance_id, 0) + 1
            )
            overall_missing += 1

    return {
        "repetitions": len(repeat_dirs),
        "instances": {
            instance_id: _summarize(
                samples,
                per_instance_count.get(instance_id, 0),
                per_instance_unknown_costs.get(instance_id, 0),
                per_instance_missing.get(instance_id, 0),
            )
            for instance_id, samples in per_instance.items()
        },
        "overall": _summarize(
            overall, overall_count, overall_unknown_costs, overall_missing
        ),
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
