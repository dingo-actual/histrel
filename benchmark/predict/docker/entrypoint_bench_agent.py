#!/usr/bin/env python3
"""Docker entrypoint for benchmark agent execution.

Reads instance JSON, delegates execution to BenchAgentWrapper, writes results.
The wrapper reads the harness bench_settings.toml for model settings, generates
an agent_config.toml from the template at ``histrel.toml`` in the repository
root, and invokes the benchmark agent as ``python -m histrel``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import cast

from benchmark.predict.agents import BenchAgentWrapper
from benchmark.predict.config import Mode
from benchmark.predict.types import AGENT_TYPE
from benchmark.common.logging import safe_log


def main() -> int:
    """Run benchmark agent and write results."""
    component_str = ("benchmark.predict.docker.entrypoint_bench_agent",)

    instance_path = Path("/workspace/instance/instance.json")
    if not instance_path.exists():
        safe_log(
            "Instance file not found",
            "error",
            bind_values={
                "component": component_str,
                "instance_path": str(instance_path),
            },
        )
        return 1

    instance = json.loads(instance_path.read_text(encoding="utf-8"))
    if not isinstance(instance, dict):
        safe_log(
            "Instance file contents invalid",
            "error",
            bind_values={
                "component": component_str,
                "instance_path": str(instance_path),
            },
        )
        return 1

    run_name = os.environ.get("RUN_NAME") or ""
    agent_type = AGENT_TYPE

    instance_id = instance.get("instance_id", os.environ.get("INSTANCE_ID")) or ""
    repo_slug = instance.get("repo") or ""
    base_commit = instance.get("base_commit") or ""

    if not instance_id and repo_slug and base_commit:
        safe_log(
            "Invalid instance (instance_id, repo slug, and base commit must all be specified)",
            "error",
            bind_values={
                "component": component_str,
                "run_name": run_name,
                "agent_type": agent_type,
                "instance_id": instance_id,
                "repo_slug": repo_slug,
                "base_commit": base_commit,
            },
        )
        return 1

    safe_log(
        "Instance loaded successfully",
        "info",
        bind_values={
            "component": component_str,
            "run_name": run_name,
            "agent_type": agent_type,
            "instance_id": instance_id,
            "repo": repo_slug,
            "base_commit": base_commit,
        },
    )

    config_path = Path("/workspace/bench_settings.toml")
    if not config_path.exists():
        safe_log(
            "Config file not found",
            "error",
            bind_values={
                "component": component_str,
                "run_name": run_name,
                "agent_type": agent_type,
                "instance_id": instance_id,
                "config_path": str(config_path),
            },
        )
        return 1

    # Histrel is always enabled for the benchmark agent.
    session_id = os.environ.get("SESSION_ID") or "-".join(
        map(lambda st: st.replace("_", "-"), [run_name, agent_type, instance_id])
    )
    timeout_seconds = int(os.environ.get("TIMEOUT_SECONDS", "7200"))

    # SWE-bench places repos at /testbed
    repo_dir = Path("/testbed")

    if not repo_dir.exists():
        safe_log(
            "Repository not found",
            "error",
            bind_values={
                "component": component_str,
                "run_name": run_name,
                "agent_type": agent_type,
                "instance_id": instance_id,
                "session_id": session_id,
                "repo_dir": str(repo_dir),
            },
        )
        return 1

    safe_log(
        "Repository validation successful",
        "info",
        bind_values={
            "component": component_str,
            "run_name": run_name,
            "agent_type": agent_type,
            "instance_id": instance_id,
            "session_id": session_id,
            "repo_dir": str(repo_dir),
        },
    )

    # Read has_test_runner from environment
    has_test_runner_env = os.environ.get("HAS_TEST_RUNNER") or ""
    if has_test_runner_env:
        has_test_runner = has_test_runner_env.lower() == "true"
    else:
        has_test_runner = instance.get("has_test_runner", False)

    wrapper = BenchAgentWrapper(config_path)
    patch_text, usage_data = wrapper.generate_patch(
        instance,
        repo_dir,
        session_id,
        has_test_runner=has_test_runner,
        timeout=timeout_seconds,
        mode=cast(Mode, os.environ.get("AGENT_MODE") or "stateless"),
    )

    output_dir = Path("/workspace/output")

    patch_path = output_dir / f"{agent_type}_{instance_id}.jsonl"
    patch_record = {
        "instance_id": instance_id,
        "model_name_or_path": agent_type,
        "model_patch": patch_text,
    }
    patch_path.write_text(json.dumps(patch_record) + "\n", encoding="utf-8")

    usage_path = output_dir / f"{agent_type}_usage.json"
    usage_path.write_text(json.dumps(usage_data), encoding="utf-8")

    safe_log(
        "Patch written",
        "info",
        bind_values={
            "component": component_str,
            "run_name": run_name,
            "agent_type": agent_type,
            "instance_id": instance_id,
            "session_id": session_id,
            "patch_path": str(patch_path),
            "token_usage": usage_data,
        },
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
