#!/usr/bin/env python3
"""Docker container orchestration for benchmark agents.

Provides DockerRunner to execute the benchmark agent in a SWE-bench
instance container with mounted volumes and environment, returning the
path to the generated patch file and the container exit code.
"""

from __future__ import annotations

import contextlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import cast

from benchmark.common.docker_utils import (
    DockerClientLike,
    run_container,
)
from benchmark.common.io import ensure_dir
from benchmark.predict.config import PredictorConfig
from benchmark.predict.docker.instance_record import build_agent_instance
from benchmark.predict.types import (
    AGENT_TYPE,
    AttemptIdentity,
    UsageData,
    build_attempt_identity,
    identity_matches,
)
from histrel.context.cost_calculator import PRICING_CACHE_ENV
from loguru import logger


def _read_attempt_usage(usage_path: Path, expected: AttemptIdentity) -> dict | None:
    """Read this attempt's usage record and verify its identity.

    Args:
        usage_path (Path): Usage file inside the attempt's artifact dir.
        expected (AttemptIdentity): Identity of the attempt being ingested.

    Returns:
        (dict | None) Parsed usage data, or None when the file is missing,
            unreadable, or belongs to a different attempt. Usage from
            another instance, repetition, or mode is never attributed to
            this one.
    """
    try:
        data = json.loads(usage_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    identity = data.get("identity")
    if not identity_matches(identity, expected):
        logger.bind(
            component="benchmark.predict.docker.runner",
            event="usage.identity_mismatch",
            origin="bench_harness_predict",
            usage_path=str(usage_path),
            recorded_identity=identity,
            expected_identity=dict(expected),
        ).error("Discarding usage record that does not match this attempt")
        return None
    return data


class DockerRunner:
    """Run the benchmark agent inside SWE-bench instance containers."""

    def __init__(self, client: DockerClientLike):
        self.client = client

    # --- Execution -------------------------------------------------------
    def run_agent(
        self,
        instance: dict,
        image: str | None,
        config: PredictorConfig,
        *,
        output_dir: Path | None = None,
        env: dict[str, str] | None = None,
        network_mode: str = "bridge",
        timeout: int | None = None,
        repetition: int = 1,
    ) -> tuple[Path, dict[str, float | UsageData | None], int]:
        """Run an agent container and return (patch_path, token_usage, exit_code).

        Args:
            instance: SWE-bench instance data.
            image: Docker image tag to use.
            config: Predictor configuration.
            output_dir: Artifacts root. Each attempt gets its own
                subdirectory named after the instance. Defaults to None.
            env: Additional environment variables. Defaults to None.
            network_mode: Docker network mode. Defaults to "bridge".
            timeout: Timeout in seconds. Defaults to None.
            repetition: 1-based repetition number of this attempt. Defaults
                to 1.

        Returns:
            (Path, dict, int) Tuple of (patch_path, token_usage, exit_code).

        Raises:
            RuntimeError: If container execution fails critically.
        """
        if image is None:
            raise RuntimeError("An instance image is required to run the agent")

        # Each attempt writes into its own artifact directory, created empty
        # right before launch: a record left by an earlier instance or an
        # earlier attempt of this instance can never be read as this one's.
        artifacts_root = output_dir or (
            Path(config.output_dir) / config.run_name / "artifacts"
        )
        instance_id = instance.get("instance_id", "unknown")
        out_dir = artifacts_root / instance_id
        if out_dir.exists():
            shutil.rmtree(out_dir)
        ensure_dir(out_dir)

        # Patch path as written by the entrypoint
        patch_path = out_dir / f"{AGENT_TYPE}_{instance_id}.jsonl"

        # Agent-facing instance record: allowlisted task fields only, written
        # outside the rw-mounted output dir. The container sees the same host
        # tree at /workspace/output, so a full dataset record placed there
        # would hand the agent the reference patch and eval metadata.
        instance_mount_dir = artifacts_root / "instance_mounts" / instance_id
        ensure_dir(instance_mount_dir)
        (instance_mount_dir / "instance.json").write_text(
            json.dumps(build_agent_instance(instance)), encoding="utf-8"
        )

        # Earlier runs kept the full record here; a resumed run must not
        # expose it through the output mount.
        with contextlib.suppress(OSError):
            shutil.rmtree(artifacts_root / "tmp_instances")

        # Volumes mapping
        volumes = {
            str(out_dir.resolve()): {"bind": "/workspace/output", "mode": "rw"},
            str(instance_mount_dir.resolve()): {
                "bind": "/workspace/instance",
                "mode": "ro",
            },
        }
        # Mount benchmark/bench_settings.toml if present (read-only)
        bench_toml = (
            Path(__file__).resolve().parent.parent.parent / "bench_settings.toml"
        )
        if bench_toml.exists():
            volumes[str(bench_toml.resolve())] = {
                "bind": "/workspace/bench_settings.toml",
                "mode": "ro",
            }

        # Environment
        environment = dict(env or {})
        environment.setdefault("PYTHONUNBUFFERED", "1")

        # Pass API keys from host environment to container
        api_key_vars = [
            "ANTHROPIC_API_KEY",
            "EXA_API_KEY",
            "FIREWORKS_API_KEY",
            "GEMINI_API_KEY",
            "MISTRAL_API_KEY",
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "TAVILY_API_KEY",
        ]
        for key in api_key_vars:
            if key in os.environ:
                environment[key] = os.environ[key]

        # Set benchmark context environment variables
        # Note: run_id is generated by Histrel MCP server, not here
        session_id = "-".join(
            map(
                lambda st: st.replace("_", "-"),
                [config.run_name, AGENT_TYPE, instance_id],
            )
        )
        environment["RUN_NAME"] = config.run_name
        environment["AGENT_TYPE"] = AGENT_TYPE
        environment["AGENT_MODE"] = config.mode
        environment["INSTANCE_ID"] = instance_id
        environment["SESSION_ID"] = session_id
        environment["REPETITION"] = str(repetition)

        # Set TIMEOUT_SECONDS environment variable for subprocess timeout
        if timeout:
            environment["TIMEOUT_SECONDS"] = str(timeout)

        # Set HAS_TEST_RUNNER based on instance metadata
        environment["HAS_TEST_RUNNER"] = (
            "true" if instance.get("has_test_runner", False) else "false"
        )

        # Share one Fireworks pricing cache file across the attempt's
        # agent processes: /workspace/output is the attempt's rw mount, so
        # the cache survives a crash and lands with the attempt's
        # artifacts. Both the benchmark agent and Histrel core read this
        # variable when they resolve model pricing.
        environment[PRICING_CACHE_ENV] = (
            "/workspace/output/fireworks_pricing_cache.json"
        )

        # Run the agent entrypoint
        entrypoint_script = Path(__file__).parent / "entrypoint_bench_agent.py"

        # Mount the entrypoint script
        volumes[str(entrypoint_script.resolve())] = {
            "bind": "/workspace/entrypoint.py",
            "mode": "ro",
        }

        # Create container_logs directory and file handles
        logs_dir = Path(config.output_dir) / config.run_name / "container_logs"
        ensure_dir(logs_dir)

        stdout_log_path = logs_dir / f"{session_id}_stdout.log"
        stderr_log_path = logs_dir / f"{session_id}_stderr.log"

        stdout_handle = stdout_log_path.open("wb")
        stderr_handle = stderr_log_path.open("wb")

        start_time = time.time()
        try:
            exit_code, _container_id = run_container(
                self.client,
                image=image,
                command=[
                    "conda",
                    "run",
                    "-n",
                    "agent",
                    "--no-capture-output",
                    "python",
                    "/workspace/entrypoint.py",
                ],
                volumes=volumes,
                environment=environment,
                network_mode=network_mode,
                timeout=timeout,
                stdout_file=stdout_handle,
                stderr_file=stderr_handle,
            )
        finally:
            stdout_handle.close()
            stderr_handle.close()

        elapsed_time = time.time() - start_time

        # Extract token usage from the attempt's own usage.json, but only
        # when the record identifies this exact attempt (run, repetition,
        # mode, instance, session).
        usage_path = out_dir / f"{AGENT_TYPE}_usage.json"
        token_usage: dict[str, float | UsageData | None] = {
            "execution_time": elapsed_time,
            "usage_data": None,  # Will contain full usage.json content
        }
        expected_identity = build_attempt_identity(
            run_name=config.run_name,
            repetition=repetition,
            mode=config.mode,
            instance_id=instance_id,
            session_id=session_id,
        )
        usage_data = _read_attempt_usage(usage_path, expected_identity)
        if usage_data is not None:
            token_usage["usage_data"] = cast(UsageData, usage_data)

        return patch_path, token_usage, exit_code
