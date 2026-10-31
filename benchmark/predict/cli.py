#!/usr/bin/env python3
"""
Histrel Benchmark Predictor CLI.

Command-line interface for generating predictions without scoring.
Platform independent.

    Important Functions:
        build_parser() -> argparse.ArgumentParser
        main(argv: list[str] | None) -> int
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from benchmark.common.io import ensure_dir, read_jsonl, run_output_dir
from benchmark.predict.config import PredictorConfig, load_predictor_config
from benchmark.predict.repeat import (
    AggregatePaths,
    aggregate_repetitions,
    repeat_dir_name,
)
from benchmark.predict.runner import run_prediction
from loguru import logger


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for Histrel benchmark predictor."""
    parser = argparse.ArgumentParser(
        prog="histrel-bench-predict",
        description="Generate predictions for SWE-bench evaluation.",
    )

    # Common options
    parser.add_argument("-c", "--config", type=str, help="Path to config file.")
    parser.add_argument(
        "--mode",
        choices=("stateless", "react"),
        help="Agent loop mode passed to the agent (default: stateless).",
    )
    parser.add_argument("-o", "--output-dir", type=str, help="Override output dir.")
    parser.add_argument("--run-name", type=str, help="Override run name.")
    parser.add_argument("--seed", type=int, help="Override seed.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Run the full instance set this many times (default: 1). Each "
            "repetition writes to its own directory; aggregate predictions "
            "and token usage are written at the run root."
        ),
    )

    # SWE-bench specific options
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--dataset-id", type=str)
    parser.add_argument("--max-instances", type=int, default=25)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument(
        "--instances", type=str, help="Path to SWE-bench instances JSONL"
    )
    parser.add_argument(
        "--instance-ids",
        type=str,
        help="Comma-separated list of specific instance IDs to run",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode with container log streaming to stderr",
    )
    parser.add_argument(
        "--show-container-streams",
        action="store_true",
        default=False,
        help="Stream container stdout/stderr to terminal (useful for debugging)",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        default=False,
        help="Force rebuild of Docker images",
    )
    parser.add_argument("--dry-run", action="store_true")

    return parser


def _apply_overrides(cfg, args: argparse.Namespace):
    """Return a new config with CLI overrides applied."""
    updates = {}
    if args.mode:
        updates["mode"] = args.mode
    if args.output_dir:
        updates["output_dir"] = args.output_dir
    if args.run_name:
        updates["run_name"] = args.run_name.replace("_", "-")
    if args.seed is not None:
        updates["seed"] = args.seed

    # SWE-bench overrides
    swe_updates = {}
    if hasattr(args, "split") and args.split:
        swe_updates["split"] = args.split
    if hasattr(args, "dataset_id") and args.dataset_id:
        swe_updates["dataset_id"] = args.dataset_id
    if hasattr(args, "max_instances") and args.max_instances is not None:
        swe_updates["max_instances"] = args.max_instances
    if hasattr(args, "timeout_seconds") and args.timeout_seconds is not None:
        swe_updates["timeout_seconds"] = args.timeout_seconds

    if swe_updates:
        updates["swe_bench"] = cfg.swe_bench.model_copy(update=swe_updates)

    # Handle verbose flag
    if hasattr(args, "verbose") and args.verbose:
        updates["verbose"] = True

    # Handle show_container_streams flag
    if hasattr(args, "show_container_streams") and args.show_container_streams:
        updates["show_container_streams"] = True

    # Handle rebuild flag
    if hasattr(args, "rebuild") and args.rebuild:
        updates["rebuild"] = True

    if updates:
        return cfg.model_copy(update=updates)
    return cfg


def _run_repeated(
    cfg: PredictorConfig,
    instances: list[dict],
    repeats: int,
) -> AggregatePaths:
    """Run the instance set ``repeats`` times and aggregate the results.

    Each repetition is a normal run in ``<output_dir>/<run_name>/repeat_NN``.

    Args:
        cfg (PredictorConfig): Predictor configuration.
        instances (list[dict]): Instances to run in every repetition.
        repeats (int): Number of repetitions (at least 2).

    Returns:
        (AggregatePaths) Paths to the aggregate artifacts.
    """
    run_root = Path(cfg.output_dir) / cfg.run_name
    repeat_dirs: list[Path] = []

    for index in range(1, repeats + 1):
        repetition_cfg = cfg.model_copy(
            update={
                "output_dir": run_root.as_posix(),
                "run_name": repeat_dir_name(index),
            }
        )
        logger.bind(
            component="benchmark.predict.cli",
            origin="bench_harness_predict",
            event="repeat.start",
        ).info(
            f"Starting repetition {index}/{repeats}",
            run_name=repetition_cfg.run_name,
            output_dir=repetition_cfg.output_dir,
        )
        repeat_dirs.append(
            asyncio.run(
                run_prediction(
                    repetition_cfg,
                    instances,
                    build_images=index == 1,
                    repetition=index,
                )
            )
        )

    return aggregate_repetitions(run_root, repeat_dirs)


def main(argv: list[str] | None = None) -> int:
    """Entry point for the predictor CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be at least 1")

    cfg = load_predictor_config(args.config)
    cfg = _apply_overrides(cfg, args)

    logger.bind(component="benchmark.predict.cli", origin="bench_harness_predict").info(
        "Configuration loaded successfully",
        run_name=cfg.run_name,
        output_dir=cfg.output_dir,
        verbose=cfg.verbose,
        max_instances=cfg.swe_bench.max_instances,
        dataset_id=cfg.swe_bench.dataset_id,
        split=cfg.swe_bench.split,
        timeout=cfg.swe_bench.timeout_seconds,
    )

    out_dir = run_output_dir(cfg.output_dir, cfg.run_name, "swe-bench")
    ensure_dir(out_dir)

    logger.bind(component="benchmark.predict.cli", origin="bench_harness_predict").info(
        "Output directory created",
        out_dir=str(out_dir),
    )

    # Dry-run: write a simple plan and exit
    if args.dry_run:
        plan_lines = [
            f"Run: {cfg.run_name}",
            f"Output: {out_dir.as_posix()}",
            f"Dataset: {getattr(args, 'instances', '')}",
            "Produces predictions.jsonl and metadata.json",
        ]
        (out_dir / "PLAN.md").write_text("\n".join(plan_lines) + "\n", encoding="utf-8")
        return 0

    # Ensure instances path, fetching from HF if omitted
    instances_path = getattr(args, "instances", None)
    if not instances_path:
        try:
            from benchmark.common.swebench_data import materialize_instances_jsonl

            dataset_id = cfg.swe_bench.dataset_id
            dest = (out_dir / "instances.jsonl").as_posix()
            logger.bind(
                component="histrel.benchmark.predict.cli",
                origin="bench_harness_predict",
            ).info(
                f"Fetching SWE-bench instances from {dataset_id} (split={cfg.swe_bench.split})"
            )
            materialize_instances_jsonl(
                dataset_id, split=cfg.swe_bench.split, dest_path=dest
            )
            instances_path = dest
            logger.bind(
                component="histrel.benchmark.predict.cli",
                origin="bench_harness_predict",
            ).info(f"Instances written to {instances_path}")
        except Exception as exc:
            logger.bind(
                component="histrel.benchmark.predict.cli",
                origin="bench_harness_predict",
            ).error(f"Unable to fetch SWE-bench instances automatically: {exc}")
            return 2

    try:
        instances = read_jsonl(instances_path)
    except Exception as exc:
        logger.bind(
            component="histrel.benchmark.predict.cli", origin="bench_harness_predict"
        ).error(f"Failed to read instances: {exc}")
        return 2

    # Filter instances if specific IDs were requested
    if hasattr(args, "instance_ids") and args.instance_ids:
        requested_ids = [id.strip() for id in args.instance_ids.split(",")]
        instances = [
            inst for inst in instances if inst.get("instance_id") in requested_ids
        ]
        if not instances:
            logger.bind(
                component="histrel.benchmark.predict.cli",
                origin="bench_harness_predict",
            ).error(
                f"No instances matched the requested IDs: {', '.join(requested_ids)}"
            )
            return 2
        logger.bind(
            component="histrel.benchmark.predict.cli", origin="bench_harness_predict"
        ).info(f"Filtered to {len(instances)} instances matching requested IDs")

    # Limit instances if max_instances is set
    if cfg.swe_bench.max_instances and len(instances) > cfg.swe_bench.max_instances:
        instances = instances[: cfg.swe_bench.max_instances]
        logger.bind(
            component="histrel.benchmark.predict.cli", origin="bench_harness_predict"
        ).info(
            f"Limited to {len(instances)} instances (max_instances={cfg.swe_bench.max_instances})"
        )

    # Run prediction generation
    try:
        if args.repeat == 1:
            output_path = asyncio.run(run_prediction(cfg, instances))
            logger.bind(
                component="histrel.benchmark.predict.cli",
                origin="bench_harness_predict",
            ).info(f"Prediction generation complete. Output: {output_path}")
        else:
            aggregates = _run_repeated(cfg, instances, args.repeat)
            logger.bind(
                component="histrel.benchmark.predict.cli",
                origin="bench_harness_predict",
            ).info(
                f"Repeated prediction generation complete over {args.repeat} runs. "
                f"Predictions: {aggregates.predictions}",
                usage_aggregate=str(aggregates.usage),
            )
        return 0
    except Exception as exc:
        if cfg.verbose:
            import traceback

            sys.stderr.write("\n" + "=" * 60 + "\n")
            sys.stderr.write("FULL TRACEBACK (verbose mode):\n")
            sys.stderr.write("=" * 60 + "\n")
            traceback.print_exc(file=sys.stderr)
            sys.stderr.write("=" * 60 + "\n\n")
        logger.bind(
            component="histrel.benchmark.predict.cli", origin="bench_harness_predict"
        ).opt(exception=True).error(f"Prediction generation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
