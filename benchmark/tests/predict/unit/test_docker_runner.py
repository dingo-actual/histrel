#!/usr/bin/env python3
"""Unit tests for benchmark.docker.runner.DockerRunner."""

from __future__ import annotations

import json
from pathlib import Path

from benchmark.predict.config import PredictorConfig
from benchmark.predict.docker.instance_record import build_agent_instance
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

    def test_04_unit_02_docker_02_runner_02_run_agent_04_instance_mount_allowlisted(
        self, docker_client_stub: DockerClientStub, tmp_path: Path
    ) -> None:
        """The instance mount holds the task record, not the ground truth."""
        cfg = PredictorConfig(
            output_dir=str(tmp_path / "results"), run_name="bench-instance-mount"
        )
        runner = DockerRunner(docker_client_stub)

        instance = {
            "instance_id": "dummy",
            "repo": "owner/repo",
            "base_commit": "abc123",
            "problem_statement": "fix issue",
            "patch": "diff --git a/src b/src\n+reference fix",
            "test_patch": "diff --git a/tests b/tests\n+reference test",
            "FAIL_TO_PASS": '["tests/test_x.py::test_y"]',
            "PASS_TO_PASS": "[]",
        }
        out_dir = Path(cfg.output_dir) / cfg.run_name / "artifacts"
        stale_dir = out_dir / "tmp_instances"
        stale_dir.mkdir(parents=True)
        (stale_dir / "instance.json").write_text(json.dumps(instance), encoding="utf-8")

        runner.run_agent(
            instance,
            image="swebench/test:latest",
            config=cfg,
            output_dir=out_dir,
            timeout=5,
        )

        call = docker_client_stub.containers.run_calls[0]
        mounts = {spec["bind"]: Path(host) for host, spec in call["volumes"].items()}

        record = json.loads(
            (mounts["/workspace/instance"] / "instance.json").read_text()
        )
        assert record == {
            "instance_id": "dummy",
            "repo": "owner/repo",
            "base_commit": "abc123",
            "problem_statement": "fix issue",
        }

        # No path under the rw-mounted output dir may carry the full record.
        assert mounts["/workspace/output"] == out_dir.resolve()
        assert not mounts["/workspace/instance"].is_relative_to(out_dir.resolve())
        assert not stale_dir.exists()


class Test04Unit02Docker03InstanceRecord:
    def test_04_unit_02_docker_03_instance_record_01_strips_ground_truth(
        self,
    ) -> None:
        """Only allowlisted task fields survive the projection."""
        instance = {
            "instance_id": "django__django-11099",
            "repo": "django/django",
            "base_commit": "abc123",
            "problem_statement": "Crash on empty input",
            "hints_text": "",
            "has_test_runner": True,
            "version": "4.2",
            "patch": "diff --git a/secret",
            "test_patch": "diff --git a/tests",
            "FAIL_TO_PASS": '["tests/test_x.py::test_y"]',
            "PASS_TO_PASS": "[]",
            "environment_setup_commit": "def456",
            "created_at": "2023-01-01",
        }

        record = build_agent_instance(instance)

        assert record == {
            "instance_id": "django__django-11099",
            "repo": "django/django",
            "base_commit": "abc123",
            "problem_statement": "Crash on empty input",
            "hints_text": "",
            "has_test_runner": True,
        }

    def test_04_unit_02_docker_03_instance_record_02_omits_absent_fields(
        self,
    ) -> None:
        """Absent optional fields are omitted, not synthesized."""
        assert build_agent_instance({"instance_id": "dummy"}) == {
            "instance_id": "dummy"
        }
