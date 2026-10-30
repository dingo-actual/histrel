#!/usr/bin/env python3
"""
Benchmark predictor runner orchestration module.

Coordinates Docker image builds and benchmark agent execution for
prediction generation with Rich progress reporting and resumability
support. Tracks token usage for each instance and supports resuming
interrupted runs.

    Important Functions:
        run_prediction(config: PredictorConfig, instances: list[dict]) -> Path
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import docker
import docker.errors
from benchmark.common.io import append_jsonl, read_jsonl, write_run_metadata
from benchmark.common.progress import (
    create_progress,
    ensure_task_complete,
    setup_benchmark_file_sink,
    setup_benchmark_logging,
)
from benchmark.common.schemas import (
    PredictionRecord,
    RunSpec,
    UsageRecord,
    read_run_spec,
    write_run_spec,
)
from benchmark.predict.docker.runner import AGENT_TYPE, DockerRunner
from benchmark.predict.swebench.build_orchestrator import BuildOrchestrator
from benchmark.predict.swebench.testspec_adapter import TestSpecAdapter
from benchmark.predict.types import SessionUsage, UsageData, UsageRecordData
from benchmark.predict.usage_helpers import accumulate_usage
from loguru import logger

if TYPE_CHECKING:
    from benchmark.predict.config import PredictorConfig


def _empty_usage_accumulator() -> dict[str, int | float]:
    """Return a zeroed usage accumulator for session totals."""
    return {
        "tokens_input": 0,
        "tokens_output": 0,
        "tokens_cache_read": 0,
        "tokens_cache_write": 0,
        "tokens_total": 0,
        "cost_total": 0.0,
        "maximum_context": 0,
    }


def _record_prediction(
    *,
    run_name: str,
    instance_id: str,
    patch_path: Path,
    token_usage: dict[str, float | UsageData | None],
    exit_code: int,
    predictions_file: Path,
    usage_file: Path,
    session_usage: SessionUsage,
) -> None:
    """Read the container's patch, then record prediction and usage."""
    logger.bind(
        run_name=run_name,
        instance_id=instance_id,
        agent_type=AGENT_TYPE,
        component="benchmark.predict.runner",
        event="patch.read.attempting",
        patch_path=str(patch_path),
        exit_code=exit_code,
        origin="bench_harness_predict",
    ).info("Attempting to read patch file from container")

    patch_text = ""
    patch_read_success = False
    if patch_path.exists():
        try:
            patch_data = json.loads(patch_path.read_text(encoding="utf-8").strip())
            patch_text = patch_data.get("model_patch", "")
            patch_read_success = True

            logger.bind(
                run_name=run_name,
                instance_id=instance_id,
                agent_type=AGENT_TYPE,
                component="benchmark.predict.runner",
                event="patch.read.success",
                patch_length=len(patch_text),
                patch_empty=len(patch_text) == 0,
                origin="bench_harness_predict",
            ).info(f"Patch read successfully (length={len(patch_text)})")
        except Exception as exc:
            logger.bind(
                run_name=run_name,
                instance_id=instance_id,
                agent_type=AGENT_TYPE,
                component="benchmark.predict.runner",
                event="patch.read.failed",
                error=str(exc),
                error_type=type(exc).__name__,
                origin="bench_harness_predict",
            ).error(f"Failed to parse patch file: {exc}")
    else:
        logger.bind(
            run_name=run_name,
            instance_id=instance_id,
            agent_type=AGENT_TYPE,
            component="benchmark.predict.runner",
            event="patch.file_not_found",
            patch_path=str(patch_path),
            exit_code=exit_code,
            origin="bench_harness_predict",
        ).error("Patch file NOT FOUND - container likely failed or produced no output")

    prediction = PredictionRecord(
        instance_id=instance_id,
        model_name_or_path=AGENT_TYPE,
        model_patch=patch_text,
    )
    append_jsonl(predictions_file, prediction.model_dump())

    logger.bind(
        run_name=run_name,
        instance_id=instance_id,
        agent_type=AGENT_TYPE,
        component="benchmark.predict.runner",
        event="prediction.written",
        predictions_file=str(predictions_file),
        patch_length=len(patch_text),
        patch_read_success=patch_read_success,
        origin="bench_harness_predict",
    ).info(f"Prediction recorded for {AGENT_TYPE}")

    # Track usage
    usage_data_raw = token_usage.get("usage_data")
    usage_data: UsageData = {}
    if isinstance(usage_data_raw, dict):
        usage_data = usage_data_raw  # type: ignore[assignment]

    total_data: UsageRecordData = usage_data.get("total", {})

    exec_time = token_usage.get("execution_time")
    usage_record = UsageRecord(
        instance_id=instance_id,
        execution_time=exec_time if isinstance(exec_time, float) else 0.0,
        tokens_input=int(total_data.get("tokens_input", 0)),
        tokens_output=int(total_data.get("tokens_output", 0)),
        tokens_cache_read=int(total_data.get("tokens_cache_read", 0)),
        tokens_cache_write=int(total_data.get("tokens_cache_write", 0)),
        tokens_total=int(total_data.get("tokens_total", 0)),
        cost_total=float(total_data.get("cost_total", 0.0)),
        maximum_context=int(total_data.get("maximum_context", 0)),
        raw_usage_data=usage_data,  # type: ignore[arg-type]
    )
    append_jsonl(usage_file, usage_record.model_dump())

    # Accumulate usage data for session totals
    if isinstance(usage_data_raw, dict):
        # Extract session_id from "none" or "total" records
        for record_type in ["none", "total"]:
            record_data = usage_data_raw.get(record_type)
            if (
                record_data is not None
                and isinstance(record_data, dict)
                and "session_id" in record_data
            ):
                if session_usage["session_id"] is None:
                    session_usage["session_id"] = record_data["session_id"]
                break

        # Store complete usage data for this instance
        session_usage[instance_id] = usage_data_raw
        accumulate_usage(session_usage, usage_data_raw)


def _run_agent_instance(
    *,
    run_name: str,
    instance: dict,
    instance_id: str,
    image: str,
    config: PredictorConfig,
    docker_runner: DockerRunner,
    predictions_file: Path,
    usage_file: Path,
    session_usage: SessionUsage,
) -> None:
    """Run the benchmark agent for one instance and record its results."""

    try:
        patch_path, token_usage, exit_code = docker_runner.run_agent(
            instance,
            image,
            config,
            timeout=config.swe_bench.timeout_seconds,
        )

        _record_prediction(
            run_name=run_name,
            instance_id=instance_id,
            patch_path=patch_path,
            token_usage=token_usage,
            exit_code=exit_code,
            predictions_file=predictions_file,
            usage_file=usage_file,
            session_usage=session_usage,
        )

        with contextlib.suppress(Exception):
            logger.bind(
                run_name=run_name,
                instance_id=instance_id,
                agent_type=AGENT_TYPE,
                component="benchmark.predict.runner",
                origin="bench_harness_predict",
            ).debug(
                f"Generated {AGENT_TYPE} patch",
                tokens_total=token_usage.get("tokens_total"),
            )
    except Exception as exc:
        with contextlib.suppress(Exception):
            logger.bind(
                run_name=run_name,
                instance_id=instance_id,
                agent_type=AGENT_TYPE,
                component="benchmark.predict.runner",
                origin="bench_harness_predict",
            ).error(f"{AGENT_TYPE} patch generation failed", error=str(exc))


async def run_prediction(config: PredictorConfig, instances: list[dict]) -> Path:
    """Main prediction orchestration with Rich progress bars.

    Coordinates Docker image building and agent execution for prediction
    generation with real-time progress feedback and structured logging.
    Supports resumability through spec files.

    Args:
        config (PredictorConfig): Predictor configuration.
        instances (list[dict]): List of SWE-bench test instances to process.

    Returns:
        (Path) Path to output directory containing predictions and artifacts.

    Raises:
        docker.errors.DockerException: If Docker client cannot be initialized.
        RuntimeError: If critical failures prevent prediction generation.
    """
    # Sort instances by instance_id for stable ordering
    instances = sorted(instances, key=lambda inst: inst.get("instance_id", ""))

    run_name = config.run_name

    # Validate verbose mode requires run_name
    if config.verbose and not config.run_name:
        raise RuntimeError(
            "run_name is required when --verbose is enabled. "
            "Provide it via CLI argument --run-name or config file."
        )

    # Configure logging
    setup_benchmark_logging(verbose=config.verbose, level="INFO")
    setup_benchmark_file_sink()  # Always-active file sink at TRACE level

    # Test log to verify the log sinks are working
    logger.bind(
        component="benchmark.predict.runner",
        origin="bench_harness_predict",
        event="test.sink_verification",
    ).info("Logging sink test message")

    with contextlib.suppress(Exception):
        logger.bind(
            run_name=run_name,
            component="benchmark.predict.runner",
            origin="bench_harness_predict",
            event="prediction.run_start",
        ).info(
            "Starting prediction generation",
            num_instances=len(instances),
            agent_type=AGENT_TYPE,
            verbose=config.verbose,
        )

    # Initialize Docker client
    try:
        client = docker.from_env()
        docker_runner = DockerRunner(client)  # type: ignore[arg-type]
    except docker.errors.DockerException as exc:
        with contextlib.suppress(Exception):
            logger.bind(
                run_name=run_name,
                component="benchmark.predict.runner",
                origin="bench_harness_predict",
            ).error("Failed to initialize Docker client", error=str(exc))
        raise RuntimeError(f"Docker initialization failed: {exc}") from exc

    # Build SWE-bench images using BuildOrchestrator
    histrel_repo_path = Path(__file__).resolve().parent.parent.parent

    orchestrator = BuildOrchestrator(client, histrel_repo_path)

    logger.bind(
        run_name=run_name,
        component="benchmark.predict.runner",
        origin="bench_harness_predict",
        event="image.build_start",
    ).info("Building SWE-bench images")

    spec_adapters: list[TestSpecAdapter] = []
    for instance in instances:
        adapter = TestSpecAdapter.from_instance(instance, histrel_repo_path)
        adapter.harden_git_operations()
        adapter.inject_agent_env()
        adapter.inject_repo_configs()
        _, has_test_runner = adapter.inject_test_runners()
        instance["has_test_runner"] = has_test_runner
        spec_adapters.append(adapter)

    orchestrator.build_images(
        spec_adapters,
        force_rebuild_env=config.rebuild,
        force_rebuild_instances=config.rebuild,
    )

    logger.bind(
        run_name=run_name,
        component="benchmark.predict.runner",
        event="image.build_complete",
        origin="bench_harness_predict",
    ).info("SWE-bench images built successfully")

    # Build mapping from instance_id to SWE-bench instance image tag
    instance_images: dict[str, str] = {
        adapter.instance_id: adapter.instance_image_key for adapter in spec_adapters
    }

    # Set up output paths
    output_dir = Path(config.output_dir)
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(parents=True, exist_ok=True)

    predictions_file = run_dir / "bench_agent_predictions.jsonl"
    usage_file = run_dir / "bench_agent_usage.jsonl"

    # Initialize session_usage for comprehensive usage tracking
    session_usage: SessionUsage = {
        "session_id": None,
        "none": _empty_usage_accumulator(),
        "total": _empty_usage_accumulator(),
    }

    # Spec file management for resumability
    spec_path = run_dir / f"{run_name}_spec.json"
    resume_run = False

    if spec_path.exists():
        try:
            existing_spec = read_run_spec(spec_path)
            # Reconcile config and instances
            current_spec = RunSpec(
                config=config.model_dump(),
                instances=instances,
            )
            if (
                existing_spec.config != current_spec.config
                or existing_spec.instances != current_spec.instances
            ):
                with contextlib.suppress(Exception):
                    logger.bind(
                        run_name=run_name,
                        component="benchmark.predict.runner",
                        origin="bench_harness_predict",
                    ).warning(
                        "Run spec mismatch detected - ignoring existing spec and starting fresh"
                    )
            else:
                resume_run = True
                with contextlib.suppress(Exception):
                    logger.bind(
                        run_name=run_name,
                        component="benchmark.predict.runner",
                        origin="bench_harness_predict",
                    ).info("Resuming existing run based on spec file")
        except Exception as exc:
            with contextlib.suppress(Exception):
                logger.bind(
                    run_name=run_name,
                    component="benchmark.predict.runner",
                    origin="bench_harness_predict",
                ).warning("Failed to read existing spec file", error=str(exc))

    if not resume_run:
        # Clear existing prediction files for this run
        for path in (predictions_file, usage_file):
            with contextlib.suppress(Exception):
                if path.exists():
                    path.unlink()
                    logger.bind(
                        run_name=run_name,
                        component="benchmark.predict.runner",
                        origin="bench_harness_predict",
                    ).info(f"Cleared existing file: {path}")

        # Create new spec file
        spec = RunSpec(
            config=config.model_dump(),
            instances=instances,
        )
        write_run_spec(run_dir, run_name, spec)
        with contextlib.suppress(Exception):
            logger.bind(
                run_name=run_name,
                component="benchmark.predict.runner",
                origin="bench_harness_predict",
            ).info("Created run spec file for resumability")

    # Load existing predictions if resuming
    completed: set[str] = set()

    if resume_run and predictions_file.exists():
        with contextlib.suppress(Exception):
            existing_predictions = read_jsonl(predictions_file)
            completed = {
                pred["instance_id"]
                for pred in existing_predictions
                if "instance_id" in pred
            }
            with contextlib.suppress(Exception):
                logger.bind(
                    run_name=run_name,
                    component="benchmark.predict.runner",
                    origin="bench_harness_predict",
                ).info(f"Found {len(completed)} existing predictions")

    with create_progress(verbose=config.verbose) as progress:
        build_task = progress.add_task("[cyan]Building Docker image...", total=1)

        logger.bind(
            run_name=run_name,
            component="benchmark.predict.runner",
            event="agent_execution.preparing",
            agent_type=AGENT_TYPE,
            origin="bench_harness_predict",
        ).info("Preparing agent execution loop")

        progress.update(build_task, advance=1)

        # Ensure build task reaches 100%
        ensure_task_complete(progress, build_task)

        # Process instances
        instance_task = progress.add_task(
            f"[green]Processing {len(instances)} instances...", total=len(instances)
        )

        for ix, instance in enumerate(instances, start=1):
            instance_id = instance.get("instance_id", f"<unknown_{ix}>")

            with contextlib.suppress(Exception):
                logger.bind(
                    run_name=run_name,
                    instance_id=instance_id,
                    component="benchmark.predict.runner",
                    event="prediction.instance_start",
                    origin="bench_harness_predict",
                ).info(
                    f"Processing instance {ix}/{len(instances)}",
                    repo=instance.get("repo"),
                )

            progress.update(
                instance_task,
                description=f"[green]Processing {instance_id} ({ix}/{len(instances)})...",
            )

            image = instance_images.get(instance_id)
            if not image:
                logger.bind(
                    run_name=run_name,
                    instance_id=instance_id,
                    agent_type=AGENT_TYPE,
                    component="benchmark.predict.runner",
                    event="agent.execution_skipped",
                    origin="bench_harness_predict",
                ).error(
                    "Agent execution SKIPPED - no image found for instance "
                    f"{instance_id}. Check if instance image was built successfully."
                )
            elif instance_id in completed:
                with contextlib.suppress(Exception):
                    logger.bind(
                        run_name=run_name,
                        instance_id=instance_id,
                        agent_type=AGENT_TYPE,
                        component="benchmark.predict.runner",
                        origin="bench_harness_predict",
                    ).info("Skipping instance - prediction already exists")
            else:
                _run_agent_instance(
                    run_name=run_name,
                    instance=instance,
                    instance_id=instance_id,
                    image=image,
                    config=config,
                    docker_runner=docker_runner,
                    predictions_file=predictions_file,
                    usage_file=usage_file,
                    session_usage=session_usage,
                )

            progress.update(instance_task, advance=1)

        # Ensure instance task reaches 100%
        progress.update(
            instance_task,
            description=f"[green]Completed {len(instances)} instances",
        )
        ensure_task_complete(progress, instance_task)

    # Cleanup spec file after successful completion
    if spec_path.exists():
        with contextlib.suppress(Exception):
            spec_path.unlink()
            logger.bind(
                run_name=run_name,
                component="benchmark.predict.runner",
                origin="bench_harness_predict",
            ).info("Run completed successfully - removed spec file")

    # Write run metadata file
    predictions_files = [predictions_file.name] if predictions_file.exists() else []
    instance_ids = [inst.get("instance_id", "") for inst in instances]

    with contextlib.suppress(Exception):
        metadata_path = write_run_metadata(
            output_dir=run_dir,
            run_name=run_name,
            dataset_id=config.swe_bench.dataset_id,
            split=config.swe_bench.split,
            instance_ids=instance_ids,
            predictions_files=predictions_files,
        )
        logger.bind(
            run_name=run_name,
            component="benchmark.predict.runner",
            origin="bench_harness_predict",
        ).info("Wrote run metadata file", metadata_path=str(metadata_path))

    # Validate that predictions were actually written with content
    total_predictions = 0
    empty_predictions = 0

    if predictions_file.exists():
        try:
            predictions = read_jsonl(predictions_file)
            total_predictions = len(predictions)
            empty_predictions = sum(1 for p in predictions if not p.get("model_patch"))
            logger.bind(
                run_name=run_name,
                component="benchmark.predict.runner",
                event="prediction.summary",
                agent_type=AGENT_TYPE,
                total_predictions=total_predictions,
                empty_predictions=empty_predictions,
                predictions_file=str(predictions_file),
                origin="bench_harness_predict",
            ).info(
                f"Predictions summary: {total_predictions} total, "
                f"{empty_predictions} empty patches"
            )
        except Exception as exc:
            logger.bind(
                run_name=run_name,
                component="benchmark.predict.runner",
                event="prediction.summary.failed",
                agent_type=AGENT_TYPE,
                error=str(exc),
                origin="bench_harness_predict",
            ).error(f"Failed to read predictions: {exc}")
    else:
        logger.bind(
            run_name=run_name,
            component="benchmark.predict.runner",
            event="prediction.summary.missing",
            agent_type=AGENT_TYPE,
            predictions_file=str(predictions_file),
            origin="bench_harness_predict",
        ).warning("Predictions file does not exist")

    # Overall summary with warning if no predictions
    if total_predictions == 0:
        logger.bind(
            run_name=run_name,
            component="benchmark.predict.runner",
            event="prediction.summary.critical",
            origin="bench_harness_predict",
        ).error(
            "CRITICAL: No predictions generated! Check previous logs for "
            "'execution_skipped' or 'patch file not found' errors."
        )
    else:
        logger.bind(
            run_name=run_name,
            component="benchmark.predict.runner",
            event="prediction.summary.success",
            total_predictions=total_predictions,
            origin="bench_harness_predict",
        ).info(
            f"Prediction generation complete: {total_predictions} total predictions generated"
        )

    with contextlib.suppress(Exception):
        logger.bind(
            run_name=run_name,
            component="benchmark.predict.runner",
            event="prediction.run_complete",
            origin="bench_harness_predict",
        ).info(
            "Prediction generation complete",
            predictions_file=str(predictions_file),
        )

    # Set final session_id
    session_usage["session_id"] = session_usage["session_id"] or "MISSING"

    # Write session_usage.json with accumulated values
    usage_summary_path = run_dir / "session_usage.json"
    usage_summary_path.write_text(json.dumps(session_usage, indent=2))
    logger.bind(
        run_name=run_name,
        component="benchmark.predict.runner",
        session_usage_path=str(usage_summary_path),
        origin="bench_harness_predict",
    ).info("Session usage data written")

    return run_dir
