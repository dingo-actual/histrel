#!/usr/bin/env python3
"""Resolution aggregation for repeated SWE-bench runs.

SWE-bench turns a predictions file into a dict keyed by ``instance_id``
(``swebench/harness/run_evaluation.py:main``), so a file that holds several
repetitions of the same instances can never be graded as repetitions: only the
last prediction for each instance survives. Each repetition is therefore graded
in its own evaluation run, with its own evaluation run ID, from its own
``repeat_NN/bench_agent_predictions.jsonl``. This module combines those
resolution reports into ``resolution_aggregate.json``.

SWE-bench writes the report of a run to its working directory as
``<model>.<run_id>.json`` (``swebench/harness/reporting.py:make_run_report``),
so run each evaluation from the run directory and the reports land side by side
with the repetition directories.

The merged ``bench_agent_predictions.jsonl`` at the run root is an analysis
artifact only; it is not a grading input.

    Important Functions:
        repeat_eval_run_id(run_name: str, index: int) -> str
        summarize_resolution(reports: Sequence[tuple[str, Mapping]]) -> dict
        aggregate_resolution(run_root: Path) -> Path
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.common.io import read_json, write_json
from loguru import logger

#: Aggregate resolution file written at the run root.
RESOLUTION_FILE_NAME = "resolution_aggregate.json"


def repeat_eval_run_id(run_name: str, index: int) -> str:
    """Return the SWE-bench evaluation run ID of a repetition.

    Args:
        run_name (str): Run name, i.e. the run root directory name.
        index (int): 1-based repetition index.

    Returns:
        (str) Evaluation run ID, e.g. ``my-run-repeat-02``.
    """
    return f"{run_name}-repeat-{index:02d}"


def _read_report(directory: Path, run_id: str) -> dict[str, Any] | None:
    """Read the resolution report of one repetition.

    Args:
        directory (Path): Directory that holds the report; SWE-bench writes it
            to its working directory.
        run_id (str): Evaluation run ID of the repetition.

    Returns:
        (dict[str, Any] | None) The ``<model>.<run_id>.json`` report, or None
        when it is missing or ambiguous.
    """
    candidates = sorted(directory.glob(f"*.{run_id}.json"))
    if len(candidates) != 1:
        logger.bind(
            component="benchmark.predict.resolution",
            origin="bench_harness_predict",
            directory=str(directory),
            run_id=run_id,
            candidates=len(candidates),
        ).warning("No unambiguous resolution report found; skipping repetition")
        return None
    return read_json(candidates[0])


def _report_instances(report: Mapping[str, Any]) -> set[str]:
    """Return the instance IDs a report covers.

    Args:
        report (Mapping[str, Any]): SWE-bench resolution report.

    Returns:
        (set[str]) Submitted instance IDs, or all mentioned IDs when the
        report lacks the submitted list.
    """
    submitted = report.get("submitted_ids")
    if isinstance(submitted, list) and submitted:
        return {str(instance_id) for instance_id in submitted}
    mentioned: set[str] = set()
    for key in ("resolved_ids", "unresolved_ids", "error_ids", "empty_patch_ids"):
        values = report.get(key)
        if isinstance(values, list):
            mentioned.update(str(instance_id) for instance_id in values)
    return mentioned


def _resolve_rate(resolved: int, total: int) -> float:
    """Return the resolve rate of one evaluation run."""
    return round(resolved / total, 4) if total else 0.0


def _repetition_of(run_id: str) -> int:
    """Return the repetition number encoded in an evaluation run ID."""
    suffix = run_id.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def _per_repetition(
    index: int, run_id: str, report: Mapping[str, Any]
) -> dict[str, Any]:
    """Summarize one repetition's resolution report.

    Args:
        index (int): 1-based repetition index.
        run_id (str): Evaluation run ID of the repetition.
        report (Mapping[str, Any]): SWE-bench resolution report.

    Returns:
        (dict[str, Any]) Counts and resolve rate of that repetition.
    """
    resolved = len(report.get("resolved_ids", []))
    total = int(report.get("total_instances", 0))
    return {
        "repetition": index,
        "run_id": run_id,
        "total_instances": total,
        "resolved_instances": resolved,
        "unresolved_instances": len(report.get("unresolved_ids", [])),
        "empty_patch_instances": len(report.get("empty_patch_ids", [])),
        "error_instances": len(report.get("error_ids", [])),
        "resolve_rate": _resolve_rate(resolved, total),
    }


def summarize_resolution(
    reports: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Summarize the resolution reports of every graded repetition.

    Args:
        reports (Sequence[tuple[str, Mapping[str, Any]]]): Pairs of evaluation
            run ID and report, in repetition order.

    Returns:
        (dict[str, Any]) Per-repetition counts, resolve-rate statistics across
        repetitions, and per-instance resolution counts.
    """
    per_repetition = [
        _per_repetition(_repetition_of(run_id), run_id, report)
        for run_id, report in reports
    ]
    rates = [entry["resolve_rate"] for entry in per_repetition]
    instance_runs: dict[str, int] = {}
    instance_resolved: dict[str, int] = {}
    for _run_id, report in reports:
        resolved_ids = {
            str(instance_id) for instance_id in report.get("resolved_ids", [])
        }
        for instance_id in _report_instances(report):
            instance_runs[instance_id] = instance_runs.get(instance_id, 0) + 1
            if instance_id in resolved_ids:
                instance_resolved[instance_id] = (
                    instance_resolved.get(instance_id, 0) + 1
                )

    return {
        "repetitions": len(per_repetition),
        "per_repetition": per_repetition,
        "resolve_rate": {
            "count": len(rates),
            "mean": round(statistics.fmean(rates), 4) if rates else 0.0,
            "median": round(statistics.median(rates), 4) if rates else 0.0,
            "min": min(rates) if rates else 0.0,
            "max": max(rates) if rates else 0.0,
        },
        "instances": {
            instance_id: {
                "runs": runs,
                "resolved": instance_resolved.get(instance_id, 0),
                "resolve_rate": _resolve_rate(
                    instance_resolved.get(instance_id, 0), runs
                ),
            }
            for instance_id, runs in sorted(instance_runs.items())
        },
    }


def aggregate_resolution(run_root: Path, reports_root: Path | None = None) -> Path:
    """Aggregate the per-repetition resolution reports of a repeated run.

    Args:
        run_root (Path): Run directory that holds the repetition directories.
        reports_root (Path | None): Directory that holds the
            ``<model>.<run_id>.json`` reports. Defaults to None, which uses
            ``run_root``.

    Returns:
        (Path) Path to the written ``resolution_aggregate.json``.
    """
    reports_root = reports_root or run_root
    reports: list[tuple[str, dict[str, Any]]] = []
    repeat_dirs = sorted(path for path in run_root.glob("repeat_*") if path.is_dir())
    for index in range(1, len(repeat_dirs) + 1):
        run_id = repeat_eval_run_id(run_root.name, index)
        report = _read_report(reports_root, run_id)
        if report is not None:
            reports.append((run_id, report))

    path = run_root / RESOLUTION_FILE_NAME
    write_json(path, summarize_resolution(reports))
    logger.bind(
        component="benchmark.predict.resolution",
        origin="bench_harness_predict",
        event="repeat.resolution_aggregated",
    ).info(
        "Aggregated resolution reports",
        graded_repetitions=len(reports),
        path=str(path),
    )
    return path


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for resolution aggregation."""
    parser = argparse.ArgumentParser(
        prog="histrel-bench-resolution",
        description=(
            "Aggregate the per-repetition SWE-bench resolution reports of a "
            "repeated run into resolution_aggregate.json."
        ),
    )
    parser.add_argument("run_root", type=str, help="Repeated run directory.")
    parser.add_argument(
        "--reports-root",
        type=str,
        help="Directory that holds the <model>.<run_id>.json reports "
        "(default: the run root).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for resolution aggregation."""
    args = build_parser().parse_args(argv)
    reports_root = Path(args.reports_root) if args.reports_root else None
    aggregate_resolution(Path(args.run_root), reports_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
