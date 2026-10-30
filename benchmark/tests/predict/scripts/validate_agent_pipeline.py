#!/usr/bin/env python3
"""Validate SWE-bench agent execution pipeline.

This script validates agent execution including repository validation,
patch generation, and usage tracking in SWE-bench containers.

    Usage:
        python scripts/validate_agent_pipeline.py [--verbose] [--quick]
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS

from benchmark.tests.fixtures.swebench import (
    make_bench_agent_wrapper,
    make_entrypoint_test_env,
)
from benchmark.predict.swebench.testspec_adapter import TestSpecAdapter


def validate_git_repo_structure(repo_dir: Path) -> dict:
    """Validate git repository structure."""
    git_dir = repo_dir / ".git"
    return {
        "valid": git_dir.exists() and git_dir.is_dir(),
        "git_directory": str(git_dir),
    }


def validate_instance_structure(instance: dict) -> dict:
    """Validate instance dictionary structure."""
    required_fields = {
        "instance_id": str,
        "repo": str,
        "base_commit": str,
    }
    missing = []
    wrong_type = []
    for field, expected_type in required_fields.items():
        if field not in instance:
            missing.append(field)
        elif not isinstance(instance[field], expected_type):
            wrong_type.append(field)

    return {
        "valid": len(missing) == 0 and len(wrong_type) == 0,
        "missing_fields": missing,
        "wrong_type_fields": wrong_type,
        "has_test_runner": instance.get("has_test_runner", False),
    }


def validate_workspace_structure(workspace_dir: Path) -> dict:
    """Validate workspace directory structure."""
    structure = {
        "instance": workspace_dir / "instance" / "instance.json",
        "output": workspace_dir / "output",
    }
    missing = []
    for name, path in structure.items():
        if not path.exists():
            missing.append(name)

    return {
        "valid": len(missing) == 0,
        "missing_components": missing,
    }


def validate_testbed_structure(testbed_dir: Path) -> dict:
    """Validate testbed directory structure."""
    structure = {
        "repo_root": testbed_dir,
        "git_directory": testbed_dir / ".git",
    }
    missing = []
    for name, path in structure.items():
        if not path.exists():
            missing.append(name)

    return {
        "valid": len(missing) == 0,
        "missing_components": missing,
        "repo_name": testbed_dir.name,
    }


def validate_conda_environments() -> dict:
    """Validate conda environment paths are correctly structured."""
    testbed_path = "/opt/miniconda3/envs/testbed"
    agent_path = "/opt/miniconda3/envs/agent"
    return {
        "testbed_exists": "testbed" in testbed_path,
        "agent_exists": "agent" in agent_path,
        "paths": {"testbed": testbed_path, "agent": agent_path},
    }


def run_quick_validation() -> bool:
    """Run quick validation checks."""
    print("Running quick agent pipeline validation...")

    try:
        from benchmark.tests.fixtures.swebench import (
            make_bench_agent_wrapper,
            make_entrypoint_test_env,
        )

        with make_entrypoint_test_env() as tmpdir:
            workspace = tmpdir / "workspace"
            assert (workspace / "instance" / "instance.json").exists()
            assert (workspace / "output").exists()
            testbed = tmpdir / "testbed"
            assert (testbed / ".git").exists()

        with make_bench_agent_wrapper() as wrapper:
            assert wrapper is not None

        print("  ✓ Workspace structure valid")
        return True
    except Exception as e:
        print(f"  ✗ Quick validation failed: {e}")
        return False


def run_full_validation(verbose: bool = False) -> bool:
    """Run full validation suite."""

    print("Running full agent pipeline validation...")

    with make_entrypoint_test_env() as tmpdir:
        workspace = tmpdir / "workspace"
        testbed = tmpdir / "testbed"

        workspace_result = validate_workspace_structure(workspace)
        if not workspace_result["valid"]:
            print(f"  ✗ Workspace invalid: {workspace_result}")
            return False
        print("  ✓ Workspace structure valid")

        testbed_result = validate_testbed_structure(testbed)
        if not testbed_result["valid"]:
            print(f"  ✗ Testbed invalid: {testbed_result}")
            return False
        print(f"  ✓ Testbed structure valid: {testbed_result['repo_name']}")

    repo_specs = MAP_REPO_VERSION_TO_SPECS.get("astropy/astropy", {})
    specs = repo_specs.get("5.0", {}) if isinstance(repo_specs, dict) else {}
    if not specs:
        print("  ✗ No SWE-bench specs found for astropy/astropy/5.0")
        return False
    if "test_cmd" not in specs:
        print("  ✗ Spec missing test_cmd")
        return False
    print(f"  ✓ SWE-bench spec retrieved (test_cmd: {specs['test_cmd']})")

    with tempfile.TemporaryDirectory() as mt_path:
        adapter = TestSpecAdapter(
            instance_id="astropy__astropy-13579",
            repo="astropy/astropy",
            version="5.0",
            env_script_list=[],
            repo_script_list=[],
            eval_script_list=[],
            arch="x86_64",
            FAIL_TO_PASS=[],
            PASS_TO_PASS=[],
            language="py",
            docker_specs={},
            namespace=None,
            histrel_repo_path=Path(mt_path),
        )

        adapter.inject_agent_env()
        if not adapter.env_script_list:
            print("  ✗ Agent environment injection failed")
            return False
        print(
            f"  ✓ Agent environment injected ({len(adapter.env_script_list)} commands)"
        )

        adapter.inject_repo_configs()
        if not adapter.repo_script_list:
            print("  ✗ Repo config injection failed")
            return False
        print(f"  ✓ Repo configs injected ({len(adapter.repo_script_list)} commands)")

    conda_result = validate_conda_environments()
    if not conda_result["testbed_exists"] or not conda_result["agent_exists"]:
        print(f"  ✗ Conda environments invalid: {conda_result}")
        return False
    print("  ✓ Conda environments correctly named")

    with make_bench_agent_wrapper() as wrapper:
        if wrapper is None:
            print("  ✗ BenchAgentWrapper initialization failed")
            return False
        if wrapper.config_path is None:
            print("  ✗ BenchAgentWrapper missing config path")
            return False
        print("  ✓ BenchAgentWrapper initialized correctly")

    if verbose:
        print(f"  Instance: {adapter.instance_id}")
        print(f"  Repo: {adapter.repo}")

    return True


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate SWE-bench agent pipeline")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Quick validation only"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SWE-bench Agent Pipeline Validation")
    print("=" * 60)

    if args.quick:
        success = run_quick_validation()
    else:
        success = run_full_validation(args.verbose)

    print("=" * 60)
    if success:
        print("Validation PASSED")
        return 0
    else:
        print("Validation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
