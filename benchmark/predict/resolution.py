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

A report that graded instances outside the run's intended set is rejected:
scoring it would count foreign resolutions, so no aggregate is written.

    Important Functions:
        repeat_eval_run_id(run_name: str, index: int) -> str
        summarize_resolution(reports, instance_ids) -> dict
        aggregate_resolution(run_root: Path) -> Path
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmark.common.io import read_json, write_json
from benchmark.predict.repeat import repeat_dir_name
from loguru import logger

#: Aggregate resolution file written at the run root.
RESOLUTION_FILE_NAME = "resolution_aggregate.json"

#: Historical name a stale aggregate is moved to when a report is rejected.
REJECTED_AGGREGATE_FILE_NAME = "resolution_aggregate.rejected.json"


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


def _intended_instance_ids(run_root: Path) -> set[str]:
    """Return the instance IDs the run intended to grade.

    Args:
        run_root (Path): Run directory that holds the repetition directories.

    Returns:
        (set[str]) IDs recorded in the first repetition's run metadata; empty
        when the metadata is missing or holds no IDs.
    """
    path = run_root / repeat_dir_name(1) / "run_metadata.json"
    if not path.exists():
        return set()
    metadata = read_json(path)
    ids = metadata.get("instance_ids") if isinstance(metadata, dict) else None
    if not isinstance(ids, list):
        return set()
    return {str(instance_id) for instance_id in ids}


def _resolve_rate(resolved: int, total: int, complete: bool = True) -> float | None:
    """Return the resolve rate of one evaluation run.

    Args:
        resolved (int): Number of resolved attempts.
        total (int): Number of attempts the rate is measured over.
        complete (bool): Whether every scheduled repetition was reported.
            Defaults to True. An incomplete schedule yields no rate.

    Returns:
        (float | None) The resolve rate, or None when it is withheld.
    """
    if not complete:
        return None
    return round(resolved / total, 4) if total else 0.0


def _repetition_of(run_id: str) -> int:
    """Return the repetition number encoded in an evaluation run ID."""
    suffix = run_id.rsplit("-", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def _per_repetition(
    index: int,
    run_id: str,
    report: Mapping[str, Any],
    intended: Collection[str] = (),
) -> dict[str, Any]:
    """Summarize one repetition's resolution report.

    Args:
        index (int): 1-based repetition index.
        run_id (str): Evaluation run ID of the repetition.
        report (Mapping[str, Any]): SWE-bench resolution report.
        intended (Collection[str]): Instance IDs the run intended to grade.
            Defaults to empty, which uses the report total.

    Returns:
        (dict[str, Any]) Counts and resolve rate of that repetition.
    """
    resolved = len(report.get("resolved_ids", []))
    # Grade the intended set: an instance without a prediction, or with a
    # failed attempt, still counts against the run. Fall back to the report
    # total when no metadata records the intended IDs.
    graded = set(intended) | _report_instances(report)
    total = len(graded) if graded else int(report.get("total_instances", 0))
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
    instance_ids: Sequence[str] | None = None,
    repetitions: int | None = None,
) -> dict[str, Any]:
    """Summarize the resolution reports of every graded repetition.

    When the intended instance IDs are known, they bound the denominator of
    every repetition: an instance without a prediction, or with a failed
    attempt, still counts against the run. The reports are checked against
    that set so a report graded over a wider dataset cannot inflate the
    rates.

    A per-instance rate is measured over the scheduled repetitions, not over
    the reports that happen to mention the instance: a repetition that never
    produced a prediction is a failed attempt, not a missing measurement.

    The cross-repetition statistics and the per-instance rates are withheld
    unless every scheduled repetition produced a report. A repetition whose
    evaluation never ran is unobserved, so scoring it as unresolved would
    report the agent as failing harder than it did.

    Args:
        reports (Sequence[tuple[str, Mapping[str, Any]]]): Pairs of evaluation
            run ID and report, in repetition order.
        instance_ids (Sequence[str] | None): Instance IDs the run intended to
            grade. Defaults to None, which trusts each report's own total.
        repetitions (int | None): Number of scheduled repetitions. Defaults to
            None, which uses the number of reports.

    Returns:
        (dict[str, Any]) Per-repetition counts, whether every scheduled
        repetition was reported, the resolve-rate statistics across
        repetitions, per-instance resolution counts, and coverage of the
        intended set. Both the overall and the per-instance rates are None
        when a scheduled repetition has no report.

    Raises:
        ValueError: If a report mentions an instance outside the intended set.
    """
    intended = {str(instance_id) for instance_id in instance_ids or []}
    _reject_unexpected(reports, intended)
    scheduled = repetitions if repetitions is not None else len(reports)
    per_repetition = [
        _per_repetition(_repetition_of(run_id), run_id, report, intended)
        for run_id, report in reports
    ]
    reported = {entry["repetition"] for entry in per_repetition}
    complete = reported == set(range(1, scheduled + 1))
    # Seed the intended IDs so an instance without a prediction or with a
    # failed attempt is reported with zero resolutions instead of vanishing.
    instance_ids_all = set(intended)
    instance_resolved: dict[str, int] = {}
    for _run_id, report in reports:
        resolved_ids = {
            str(instance_id) for instance_id in report.get("resolved_ids", [])
        }
        report_ids = _report_instances(report)
        instance_ids_all |= report_ids
        for instance_id in report_ids:
            if instance_id in resolved_ids:
                instance_resolved[instance_id] = (
                    instance_resolved.get(instance_id, 0) + 1
                )

    return {
        "repetitions": len(per_repetition),
        "scheduled_repetitions": scheduled,
        "reports_complete": complete,
        "per_repetition": per_repetition,
        "resolve_rate": _overall_rate(
            [entry["resolve_rate"] for entry in per_repetition], complete
        ),
        "instances": {
            instance_id: {
                "runs": scheduled,
                "resolved": instance_resolved.get(instance_id, 0),
                "resolve_rate": _resolve_rate(
                    instance_resolved.get(instance_id, 0), scheduled, complete
                ),
            }
            for instance_id in sorted(instance_ids_all)
        },
        "coverage": _coverage(reports, intended, scheduled),
    }


def _overall_rate(rates: Sequence[float], complete: bool) -> dict[str, Any]:
    """Return the cross-repetition resolve-rate statistics.

    Args:
        rates (Sequence[float]): Resolve rate of every reported repetition.
        complete (bool): Whether every scheduled repetition was reported.

    Returns:
        (dict[str, Any]) Whether the statistics are complete and, only when
        they are, the sample count and the mean, median, min, and max. An
        incomplete set reports None for every statistic instead of
        summarizing a subset of the schedule.
    """
    if not complete:
        return {
            "complete": False,
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
        }
    return {
        "complete": True,
        "count": len(rates),
        "mean": round(statistics.fmean(rates), 4) if rates else 0.0,
        "median": round(statistics.median(rates), 4) if rates else 0.0,
        "min": min(rates) if rates else 0.0,
        "max": max(rates) if rates else 0.0,
    }


def _reject_unexpected(
    reports: Sequence[tuple[str, Mapping[str, Any]]], intended: set[str]
) -> None:
    """Reject any report that graded instances outside the intended set.

    Scoring such a report would add foreign resolutions to the numerator
    while the denominator only holds the intended instances, so no score is
    computed at all.

    Args:
        reports (Sequence[tuple[str, Mapping[str, Any]]]): Pairs of evaluation
            run ID and report.
        intended (set[str]): Instance IDs the run intended to grade. An empty
            set cannot prove a report wrong, so nothing is rejected.

    Raises:
        ValueError: If a report mentions an instance outside ``intended``.
    """
    for run_id, report in reports:
        unexpected = sorted(_report_instances(report) - intended)
        if not intended or not unexpected:
            continue
        logger.bind(
            component="benchmark.predict.resolution",
            origin="bench_harness_predict",
            event="repeat.coverage_mismatch",
        ).error(
            "Rejected resolution report outside the run",
            run_id=run_id,
            unexpected_instances=unexpected,
            intended_instances=len(intended),
        )
        raise ValueError(
            f"Report {run_id} graded instances outside the run: {', '.join(unexpected)}"
        )


def _preserve_stale_aggregate(path: Path) -> None:
    """Move a stale aggregate aside so it cannot pass for the current one.

    Args:
        path (Path): Aggregate path written by an earlier run, if any.
    """
    if not path.exists():
        return
    rejected = path.with_name(REJECTED_AGGREGATE_FILE_NAME)
    path.replace(rejected)
    logger.bind(
        component="benchmark.predict.resolution",
        origin="bench_harness_predict",
        event="repeat.aggregate_preserved",
    ).warning(
        "Preserved stale resolution aggregate after a rejected report",
        path=str(rejected),
    )


def _coverage(
    reports: Sequence[tuple[str, Mapping[str, Any]]],
    intended: set[str],
    scheduled: int,
) -> dict[str, Any]:
    """Check that the reports cover every scheduled instance/repetition pair.

    Coverage is tracked per ``(instance_id, repetition)`` pair: an instance
    that appears in one repetition but not in another has a missing
    prediction for the other one.

    Args:
        reports (Sequence[tuple[str, Mapping[str, Any]]]): Pairs of evaluation
            run ID and report.
        intended (set[str]): Instance IDs the run intended to grade.
        scheduled (int): Number of scheduled repetitions.

    Returns:
        (dict[str, Any]) Intended count, scheduled attempt count, the missing
        pairs, and the instances with at least one missing repetition.
    """
    covered: set[tuple[str, int]] = set()
    for run_id, report in reports:
        repetition = _repetition_of(run_id)
        covered |= {
            (instance_id, repetition) for instance_id in _report_instances(report)
        }
    expected = {
        (instance_id, repetition)
        for instance_id in intended
        for repetition in range(1, scheduled + 1)
    }
    missing = sorted(expected - covered)
    return {
        "intended_instances": len(intended),
        "scheduled_attempts": len(expected),
        "missing_coverage": [
            {"instance_id": instance_id, "repetition": repetition}
            for instance_id, repetition in missing
        ],
        "missing_instances": sorted({instance_id for instance_id, _ in missing}),
    }


def aggregate_resolution(run_root: Path, reports_root: Path | None = None) -> Path:
    """Aggregate the per-repetition resolution reports of a repeated run.

    The instance IDs recorded in the run metadata are the denominator, and
    every report is checked against them. Per-instance rates are measured
    over the scheduled repetitions, which are the repetition directories in
    the run root. The cross-repetition statistics and the per-instance rates
    are withheld when any of those repetitions has no report.

    Args:
        run_root (Path): Run directory that holds the repetition directories.
        reports_root (Path | None): Directory that holds the
            ``<model>.<run_id>.json`` reports. Defaults to None, which uses
            ``run_root``.

    Returns:
        (Path) Path to the written ``resolution_aggregate.json``.

    Raises:
        ValueError: If a report mentions an instance outside the intended
            set. An aggregate from an earlier run is then moved to
            ``resolution_aggregate.rejected.json`` so no stale file keeps
            looking current.
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
    intent = sorted(_intended_instance_ids(run_root))
    try:
        summary = summarize_resolution(reports, intent, len(repeat_dirs))
    except ValueError:
        _preserve_stale_aggregate(path)
        raise
    write_json(path, summary)
    logger.bind(
        component="benchmark.predict.resolution",
        origin="bench_harness_predict",
        event="repeat.resolution_aggregated",
    ).info(
        "Aggregated resolution reports",
        graded_repetitions=len(reports),
        intended_instances=len(intent),
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
