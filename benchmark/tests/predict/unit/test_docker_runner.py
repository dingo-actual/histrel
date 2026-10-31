#!/usr/bin/env python3
"""Unit tests for benchmark.docker.runner.DockerRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import benchmark.predict.docker.runner as runner_module
from benchmark.predict.config import HistrelConfig, PredictorConfig
from benchmark.predict.docker.instance_record import build_agent_instance
from benchmark.predict.docker.runner import DockerRunner
from benchmark.predict.types import build_attempt_identity
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
        attempt_dir = out_dir / "dummy"
        assert mounts["/workspace/output"] == attempt_dir.resolve()
        assert not mounts["/workspace/instance"].is_relative_to(attempt_dir.resolve())
        assert not stale_dir.exists()

    def test_04_unit_02_docker_02_runner_02_run_agent_04b_settings_mount(
        self, docker_client_stub: DockerClientStub, tmp_path: Path
    ) -> None:
        """The container mounts the resolved run settings, not the repo file."""
        # Arrange — a custom config the repository file does not contain
        cfg = PredictorConfig(
            output_dir=str(tmp_path / "results"),
            run_name="bench-custom-config",
            histrel=HistrelConfig(models="accounts/fireworks/models/custom-model"),
        )
        cfg.write_bench_settings(Path(cfg.output_dir) / cfg.run_name)
        runner = DockerRunner(docker_client_stub)
        out_dir = Path(cfg.output_dir) / cfg.run_name / "artifacts"

        # Act
        runner.run_agent(
            {"instance_id": "dummy", "repo": "owner/repo"},
            image="swebench/test:latest",
            config=cfg,
            output_dir=out_dir,
            timeout=5,
        )

        # Assert
        call = docker_client_stub.containers.run_calls[0]
        mounts = {spec["bind"]: Path(host) for host, spec in call["volumes"].items()}
        mounted = mounts["/workspace/bench_settings.toml"]
        assert (
            mounted
            == (Path(cfg.output_dir) / cfg.run_name / "bench_settings.toml").resolve()
        )
        assert "custom-model" in mounted.read_text(encoding="utf-8")

    def test_04_unit_02_docker_02_runner_02_run_agent_05_own_attempt_dir(
        self, docker_client_stub: DockerClientStub, tmp_path: Path
    ) -> None:
        """Each attempt mounts and writes its own artifact directory."""
        cfg = PredictorConfig(
            output_dir=str(tmp_path / "results"), run_name="bench-attempts"
        )
        runner = DockerRunner(docker_client_stub)
        out_dir = Path(cfg.output_dir) / cfg.run_name / "artifacts"

        for instance_id in ("inst-a", "inst-b"):
            runner.run_agent(
                {"instance_id": instance_id, "repo": "owner/repo"},
                image="swebench/test:latest",
                config=cfg,
                output_dir=out_dir,
                timeout=5,
                repetition=2,
            )

        output_mounts = [
            Path(host)
            for call in docker_client_stub.containers.run_calls
            for host, spec in call["volumes"].items()
            if spec["bind"] == "/workspace/output"
        ]
        assert output_mounts == [
            (out_dir / "inst-a").resolve(),
            (out_dir / "inst-b").resolve(),
        ]
        assert (
            docker_client_stub.containers.run_calls[0]["environment"]["REPETITION"]
            == "2"
        )

    def test_04_unit_02_docker_02_runner_02_run_agent_06_stale_usage_dropped(
        self, docker_client_stub: DockerClientStub, tmp_path: Path
    ) -> None:
        """A usage record left by an earlier attempt is never ingested."""
        cfg = PredictorConfig(
            output_dir=str(tmp_path / "results"), run_name="bench-stale"
        )
        runner = DockerRunner(docker_client_stub)
        out_dir = Path(cfg.output_dir) / cfg.run_name / "artifacts"
        stale_usage = out_dir / "dummy" / "bench-agent_usage.json"
        stale_usage.parent.mkdir(parents=True)
        stale_usage.write_text(
            json.dumps({"total": {"session_id": "other", "tokens_total": 999}}),
            encoding="utf-8",
        )

        _, token_usage, _ = runner.run_agent(
            {"instance_id": "dummy", "repo": "owner/repo"},
            image="swebench/test:latest",
            config=cfg,
            output_dir=out_dir,
            timeout=5,
        )

        assert token_usage["usage_data"] is None
        assert not stale_usage.exists()

    def test_04_unit_02_docker_02_runner_02_run_agent_07_identity_checked(
        self,
        docker_client_stub: DockerClientStub,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Only a record identifying this attempt (run, repetition, mode) is kept."""
        cfg = PredictorConfig(
            output_dir=str(tmp_path / "results"),
            run_name="repeat_02",
            mode="react",
        )
        runner = DockerRunner(docker_client_stub)
        out_dir = Path(cfg.output_dir) / cfg.run_name / "artifacts"
        instance_id = "dummy"
        session_id = f"{cfg.run_name.replace('_', '-')}-bench-agent-{instance_id}"
        written: dict[str, dict] = {}

        def _fake_run_container(client, *, volumes, **kwargs):
            host_dir = Path(
                next(
                    host
                    for host, spec in volumes.items()
                    if spec["bind"] == "/workspace/output"
                )
            )
            (host_dir / "bench-agent_usage.json").write_text(
                json.dumps(written["payload"]), encoding="utf-8"
            )
            return 0, "container-id"

        monkeypatch.setattr(runner_module, "run_container", _fake_run_container)

        # A record from another repetition must be discarded.
        written["payload"] = {
            "identity": build_attempt_identity(
                run_name=cfg.run_name,
                repetition=1,
                mode=cfg.mode,
                instance_id=instance_id,
                session_id=session_id,
            ),
            "total": {"session_id": session_id, "tokens_total": 42},
        }
        _, mismatch_usage, _ = runner.run_agent(
            {"instance_id": instance_id, "repo": "owner/repo"},
            image="swebench/test:latest",
            config=cfg,
            output_dir=out_dir,
            timeout=5,
            repetition=2,
        )

        # A record that matches this attempt is ingested.
        written["payload"]["identity"]["repetition"] = 2
        _, match_usage, _ = runner.run_agent(
            {"instance_id": instance_id, "repo": "owner/repo"},
            image="swebench/test:latest",
            config=cfg,
            output_dir=out_dir,
            timeout=5,
            repetition=2,
        )

        assert mismatch_usage["usage_data"] is None
        assert match_usage["usage_data"] == written["payload"]


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
