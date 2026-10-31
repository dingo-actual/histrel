#!/usr/bin/env python3
"""SWE-bench scoring CLI for repeated benchmark runs.

SWE-bench keys predictions by ``instance_id``, so the merged
``bench_agent_predictions.jsonl`` at a run root cannot be graded as
repetitions. This CLI grades every ``repeat_NN/bench_agent_predictions.jsonl``
in its own evaluation run, with its own run ID, then aggregates the reports
into ``resolution_aggregate.json``.

    Important Functions:
        build_parser() -> argparse.ArgumentParser
        evaluation_command(...) -> list[str]
        grade_repetition(...) -> bool
        main(argv: list[str] | None) -> int
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from benchmark.common.io import read_json
from benchmark.predict.repeat import PREDICTIONS_FILE_NAME, repeat_dir_name
from benchmark.predict.resolution import aggregate_resolution, repeat_eval_run_id
from loguru import logger

#: Fallback dataset when the run wrote no metadata.
DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"

#: Where the SWE-bench harness caches per-run evaluation output, relative to
#: the evaluation working directory (the run root).
EVAL_LOG_DIR = Path("logs") / "run_evaluation"

_LOG = dict(component="benchmark.score.cli", origin="bench_harness_score")


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for SWE-bench scoring."""
    parser = argparse.ArgumentParser(
        prog="histrel-bench-score",
        description=(
            "Grade every repetition of a benchmark run with SWE-bench and "
            "aggregate the resolution reports."
        ),
    )
    parser.add_argument("run_root", type=str, help="Repeated run directory.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="SWE-bench dataset (default: the run metadata, else "
        f"{DEFAULT_DATASET_NAME}).",
    )
    parser.add_argument(
        "--split", type=str, help="Dataset split (default: run metadata)."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Parallel evaluation workers (default: 5).",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=1800,
        help="Per-instance test timeout in seconds (default: 1800).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Discard cached evaluation results and re-grade every repetition.",
    )
    return parser


def repeat_indices(run_root: Path) -> list[int]:
    """Return the repetition indices present in a run root.

    Args:
        run_root (Path): Run directory that holds the repetition directories.

    Returns:
        (list[int]) 1-based repetition indices, in order.
    """
    directories = [path for path in run_root.glob("repeat_*") if path.is_dir()]
    return list(range(1, len(directories) + 1))


def report_files(run_root: Path, run_id: str) -> list[Path]:
    """Return the resolution report files of an evaluation run.

    Args:
        run_root (Path): Run directory that holds the reports.
        run_id (str): Evaluation run ID.

    Returns:
        (list[Path]) Matching ``<model>.<run_id>.json`` files.
    """
    return sorted(run_root.glob(f"*.{run_id}.json"))


def evaluation_command(
    predictions_path: Path,
    run_id: str,
    dataset_name: str,
    split: str,
    max_workers: int,
    timeout_seconds: int,
    instance_ids: Sequence[str] | None = None,
) -> list[str]:
    """Build the SWE-bench evaluation command of one repetition.

    Args:
        predictions_path (Path): Predictions file of the repetition.
        run_id (str): Evaluation run ID of the repetition.
        dataset_name (str): SWE-bench dataset name.
        split (str): Dataset split.
        max_workers (int): Parallel evaluation workers.
        timeout_seconds (int): Per-instance test timeout.
        instance_ids (Sequence[str] | None): Instance IDs to grade. SWE-bench
            reports the dataset filtered by these IDs as ``total_instances``,
            so passing the run's IDs keeps the denominator at the intended
            subset instead of the whole dataset. Defaults to None.

    Returns:
        (list[str]) Command to run the SWE-bench harness.
    """
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        predictions_path.as_posix(),
        "--run_id",
        run_id,
        "--max_workers",
        str(max_workers),
        "--timeout",
        str(timeout_seconds),
    ]
    if instance_ids:
        command.extend(["--instance_ids", *instance_ids])
    return command


def _evaluation_env() -> dict[str, str]:
    """Return the environment for the evaluation subprocess.

    The vendored SWE-bench snapshot keeps its pinned build resources, so it
    must be the import source for ``swebench``.

    Returns:
        (dict[str, str]) Environment with the vendor root on ``PYTHONPATH``.
    """
    env = os.environ.copy()
    vendor_root = Path(__file__).resolve().parents[1] / "vendor" / "SWE-bench"
    if not (vendor_root / "swebench" / "__init__.py").exists():
        return env
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        f"{vendor_root}{os.pathsep}{existing}" if existing else vendor_root.as_posix()
    )
    return env


def _run_metadata(run_root: Path) -> dict[str, Any]:
    """Read the run metadata of the first repetition.

    Args:
        run_root (Path): Run directory that holds the repetition directories.

    Returns:
        (dict[str, Any]) Metadata, or an empty dict when it is missing.
    """
    path = run_root / repeat_dir_name(1) / "run_metadata.json"
    if not path.exists():
        return {}
    metadata = read_json(path)
    return metadata if isinstance(metadata, dict) else {}


def invalidate_cached_evaluation(run_root: Path, run_id: str) -> None:
    """Delete the cached evaluation results of one repetition.

    SWE-bench skips an instance whose ``report.json`` already exists under
    ``logs/run_evaluation/<run_id>/``, so a forced re-grade must remove that
    cache directory and the resolution report written from it. Otherwise a
    changed prediction keeps its old score.

    Args:
        run_root (Path): Run directory that holds the repetition directories
            and the evaluation working directory.
        run_id (str): Evaluation run ID of the repetition.
    """
    cache_dir = run_root / EVAL_LOG_DIR / run_id
    removed_cache = cache_dir.exists()
    if removed_cache:
        shutil.rmtree(cache_dir)
    reports = report_files(run_root, run_id)
    for report in reports:
        report.unlink()
    logger.bind(**_LOG, event="score.cache.invalidated").info(
        "Invalidated cached evaluation results",
        run_id=run_id,
        cache_dir=str(cache_dir),
        removed_cache=removed_cache,
        removed_reports=[report.name for report in reports],
    )


def grade_repetition(
    run_root: Path,
    index: int,
    *,
    dataset_name: str,
    split: str,
    max_workers: int,
    timeout_seconds: int,
    instance_ids: Sequence[str] | None = None,
    force: bool = False,
) -> bool:
    """Grade one repetition with the SWE-bench harness.

    Args:
        run_root (Path): Run directory that holds the repetition directories.
        index (int): 1-based repetition index.
        dataset_name (str): SWE-bench dataset name.
        split (str): Dataset split.
        max_workers (int): Parallel evaluation workers.
        timeout_seconds (int): Per-instance test timeout.
        instance_ids (Sequence[str] | None): Instance IDs to grade. Defaults to
            None, which grades the whole dataset.
        force (bool): Discard this repetition's cached evaluation results
            before grading. Defaults to False.

    Returns:
        (bool) True when the evaluation finished successfully.
    """
    predictions_path = run_root / repeat_dir_name(index) / PREDICTIONS_FILE_NAME
    run_id = repeat_eval_run_id(run_root.name, index)
    if not predictions_path.exists():
        logger.bind(**_LOG).error(
            "Missing predictions file; skipping repetition",
            run_id=run_id,
            predictions_path=str(predictions_path),
        )
        return False

    if force:
        invalidate_cached_evaluation(run_root, run_id)

    command = evaluation_command(
        predictions_path,
        run_id,
        dataset_name,
        split,
        max_workers,
        timeout_seconds,
        instance_ids,
    )
    logger.bind(**_LOG).info(
        "Grading repetition", run_id=run_id, command=" ".join(command)
    )
    completed = subprocess.run(
        command, cwd=run_root, env=_evaluation_env(), check=False
    )
    if completed.returncode != 0:
        logger.bind(**_LOG).error(
            "SWE-bench evaluation failed",
            run_id=run_id,
            returncode=completed.returncode,
        )
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    """Entry point for SWE-bench scoring.

    Args:
        argv (list[str] | None): Command-line arguments. Defaults to None,
            which uses ``sys.argv``.

    Returns:
        (int) 0 when every repetition was graded, 1 otherwise.
    """
    args = build_parser().parse_args(argv)
    run_root = Path(args.run_root).resolve()
    indices = repeat_indices(run_root)
    if not indices:
        logger.bind(**_LOG).error(
            "No repeat_* directories found; nothing to grade", run_root=str(run_root)
        )
        return 1

    metadata = _run_metadata(run_root)
    dataset_name = (
        args.dataset_name or metadata.get("dataset_id") or DEFAULT_DATASET_NAME
    )
    split = args.split or metadata.get("split") or "test"
    # Grade the same subset the run intended: without these IDs SWE-bench
    # reports the whole dataset as total_instances and the resolve rate is
    # divided by the wrong denominator.
    recorded_ids = metadata.get("instance_ids")
    instance_ids = (
        [str(instance_id) for instance_id in recorded_ids]
        if isinstance(recorded_ids, list) and recorded_ids
        else None
    )
    if instance_ids is None:
        logger.bind(**_LOG).warning(
            "Run metadata holds no instance IDs; grading the whole dataset",
            run_root=str(run_root),
        )

    graded = 0
    for index in indices:
        run_id = repeat_eval_run_id(run_root.name, index)
        if not args.force and report_files(run_root, run_id):
            logger.bind(**_LOG).info("Report already exists; skipping", run_id=run_id)
            graded += 1
            continue
        if grade_repetition(
            run_root,
            index,
            dataset_name=dataset_name,
            split=split,
            max_workers=args.max_workers,
            timeout_seconds=args.timeout_seconds,
            instance_ids=instance_ids,
            force=args.force,
        ):
            graded += 1

    aggregate_resolution(run_root)
    return 0 if graded == len(indices) else 1


if __name__ == "__main__":
    raise SystemExit(main())
