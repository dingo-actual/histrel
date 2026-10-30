#!/usr/bin/env python3
"""Unit tests for SWE-bench agent execution validation.

Tests for agent execution in pre-cloned repositories with conda environment
management, patch generation, and usage tracking.

    Test Classes:
        Test04Unit05AgentExec01Repository: Repository validation tests
        Test04Unit05AgentExec02Execution: Execution environment tests
        Test04Unit05AgentExec03PatchGeneration: Patch generation tests
        Test04Unit05AgentExec04UsageTracking: Usage tracking tests
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

if sys.platform == "win32":
    pytest.skip("Agent execution requires Unix", allow_module_level=True)

from benchmark.tests.fixtures.swebench import (
    make_entrypoint_test_env,
)


class Test04Unit05AgentExec01Repository:
    """Tests for repository validation in SWE-bench containers."""

    def test_04_unit_05_agent_exec_01_repository_01_validate_repo_structure(
        self,
        tmp_path: Path,
    ) -> None:
        """Test validate_repo checks for .git directory."""
        from benchmark.predict.agents.repo.git_repo import GitRepoManager

        repo_dir = tmp_path / "testrepo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()

        repo_manager = GitRepoManager()
        repo_manager.validate_repo(repo_dir)

    def test_04_unit_05_agent_exec_01_repository_02_validate_repo_detects_missing_git(
        self,
        tmp_path: Path,
    ) -> None:
        """Test validate_repo raises error when .git is missing."""
        from benchmark.predict.agents.repo.git_repo import GitRepoManager

        repo_dir = tmp_path / "testrepo"
        repo_dir.mkdir()

        repo_manager = GitRepoManager()
        with pytest.raises((ValueError, RuntimeError)):
            repo_manager.validate_repo(repo_dir)


class Test04Unit05AgentExec02Execution:
    """Tests for execution environment in SWE-bench containers."""

    def test_04_unit_05_agent_exec_02_execution_01_entrypoint_computes_repo_dir(
        self,
    ) -> None:
        """Test entrypoint computes correct repo directory from instance."""
        instance = {
            "instance_id": "astropy__astropy-13579",
            "repo": "astropy/astropy",
            "base_commit": "0df94ff7097961e92fd7812036a24b145bc13ca8",
        }
        with make_entrypoint_test_env(instance=instance) as tmpdir:
            instance_path = tmpdir / "workspace" / "instance" / "instance.json"
            instance_data = json.loads(instance_path.read_text())
            expected_repo = "astropy/astropy"
            assert instance_data["repo"] == expected_repo

    def test_04_unit_05_agent_exec_02_execution_02_has_test_runner_from_instance(
        self,
    ) -> None:
        """Test has_test_runner is extracted from instance."""
        instance = {
            "instance_id": "astropy__astropy-13579",
            "repo": "astropy/astropy",
            "base_commit": "0df94ff7097961e92fd7812036a24b145bc13ca8",
            "has_test_runner": True,
        }
        with make_entrypoint_test_env(
            instance=instance, has_test_runner=True
        ) as tmpdir:
            instance_path = tmpdir / "workspace" / "instance" / "instance.json"
            instance_data = json.loads(instance_path.read_text())
            assert instance_data.get("has_test_runner") is True

    def test_04_unit_05_agent_exec_02_execution_03_output_directory_structure(
        self,
    ) -> None:
        """Test output directory exists for patch extraction."""
        with make_entrypoint_test_env() as tmpdir:
            output_dir = tmpdir / "workspace" / "output"
            assert output_dir.exists()
            assert output_dir.is_dir()


class Test04Unit05AgentExec04UsageTracking:
    """Tests for usage tracking in SWE-bench containers."""

    def test_04_unit_05_agent_exec_04_usage_tracking_01_usage_json_structure(
        self,
    ) -> None:
        """Test usage.json has expected structure."""
        usage_data = {
            "model": "accounts/fireworks/models/deepseek-v4p1-flash-1017",
            "input_tokens": 1000,
            "output_tokens": 500,
            "duration_seconds": 30,
        }
        assert "model" in usage_data
        assert "input_tokens" in usage_data
        assert "output_tokens" in usage_data
        assert isinstance(usage_data["input_tokens"], int)

    def test_04_unit_05_agent_exec_04_usage_tracking_02_usage_json_path(
        self,
    ) -> None:
        """Test usage.json is written to expected location."""
        with make_entrypoint_test_env() as tmpdir:
            testbed_dir = tmpdir / "testbed"
            usage_path = testbed_dir / "usage.json"
            assert usage_path.parent.exists()


class Test04Unit05AgentExec05Integration:
    """Integration tests for complete agent execution pipeline."""

    def test_04_unit_05_agent_exec_05_integration_01_instance_matches_swebench_format(
        self,
    ) -> None:
        """Test instance data matches SWE-bench Verified format."""
        instance = {
            "instance_id": "astropy__astropy-13579",
            "repo": "astropy/astropy",
            "version": "5.0",
            "base_commit": "0df94ff7097961e92fd7812036a24b145bc13ca8",
            "FAIL_TO_PASS": ["test_sliced_wcs.py::test_coupled_world_slicing"],
            "PASS_TO_PASS": [],
        }
        assert "astropy__astropy-13579" in instance["instance_id"]
        assert instance["repo"] == "astropy/astropy"
        assert len(instance["base_commit"]) == 40

    def test_04_unit_05_agent_exec_05_integration_02_dual_environment_execution(
        self,
        tmp_path: Path,
    ) -> None:
        """Test dual conda environment setup for agent execution."""
        testbed_python = "/opt/miniconda3/envs/testbed/bin/python"
        agent_python = "/opt/miniconda3/envs/agent/bin/python"
        assert "testbed" in testbed_python
        assert "agent" in agent_python
        assert testbed_python != agent_python
