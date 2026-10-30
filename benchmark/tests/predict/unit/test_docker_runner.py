#!/usr/bin/env python3
"""Unit tests for benchmark.docker.runner.DockerRunner."""

from __future__ import annotations

from pathlib import Path

from benchmark.predict.config import PredictorConfig
from benchmark.predict.docker.runner import DockerRunner
from benchmark.tests.fixtures.docker_stubs import DockerClientStub


class Test04Unit02Docker02Runner02RunAgent:
    def test_04_unit_02_docker_02_runner_02_run_agent_01_generates_patch(
        self, docker_client_stub: DockerClientStub, tmp_path: Path
    ) -> None:
        cfg = PredictorConfig(
            output_dir=str(tmp_path / "results"), run_name="bench-local", verbose=True
        )
        runner = DockerRunner(docker_client_stub)

        instance = {
            "instance_id": "dummy",
            "repo": "owner/repo",
            "problem_statement": "fix issue",
        }
        out_dir = Path(cfg.output_dir) / cfg.run_name / "artifacts"

        patch_path, token_usage, exit_code = runner.run_agent(
            instance,
            image="swebench/test:latest",
            config=cfg,
            output_dir=out_dir,
            env={"FOO": "BAR"},
            network_mode="bridge",
            timeout=5,
        )

        # Patch path is determined by {agent}_{instance_id}.jsonl format
        assert patch_path.name == "bench-agent_dummy.jsonl"
        # Token usage extracted - new format with usage_data
        assert "execution_time" in token_usage
        assert "usage_data" in token_usage
        # Container invoked with the API key and harness env passed through
        assert docker_client_stub.containers.run_calls
        call = docker_client_stub.containers.run_calls[0]
        assert call["environment"].get("FOO") == "BAR"

    def test_04_unit_02_docker_02_runner_02_run_agent_02_shared_pricing_cache_env(
        self, docker_client_stub: DockerClientStub, tmp_path: Path
    ) -> None:
        """HISTREL_FIREWORKS_PRICING_CACHE points into the shared rw mount."""
        cfg = PredictorConfig(
            output_dir=str(tmp_path / "results"),
            run_name="bench-pricing-cache",
        )
        runner = DockerRunner(docker_client_stub)

        instance = {
            "instance_id": "dummy",
            "repo": "owner/repo",
            "problem_statement": "fix issue",
        }
        out_dir = Path(cfg.output_dir) / cfg.run_name / "artifacts"

        runner.run_agent(
            instance,
            image="swebench/test:latest",
            config=cfg,
            output_dir=out_dir,
            timeout=5,
        )

        assert docker_client_stub.containers.run_calls
        call = docker_client_stub.containers.run_calls[0]
        assert call["environment"].get("HISTREL_FIREWORKS_PRICING_CACHE") == (
            "/workspace/output/fireworks_pricing_cache.json"
        )

    def test_04_unit_02_docker_02_runner_02_run_agent_03_mode_env_passed(
        self, docker_client_stub: DockerClientStub, tmp_path: Path
    ) -> None:
        """The configured agent mode reaches the container as AGENT_MODE."""
        cfg = PredictorConfig(
            output_dir=str(tmp_path / "results"),
            run_name="bench-mode",
            mode="react",
        )
        runner = DockerRunner(docker_client_stub)

        instance = {
            "instance_id": "dummy",
            "repo": "owner/repo",
            "problem_statement": "fix issue",
        }
        out_dir = Path(cfg.output_dir) / cfg.run_name / "artifacts"

        runner.run_agent(
            instance,
            image="swebench/test:latest",
            config=cfg,
            output_dir=out_dir,
            timeout=5,
        )

        assert docker_client_stub.containers.run_calls
        call = docker_client_stub.containers.run_calls[0]
        assert call["environment"].get("AGENT_MODE") == "react"
