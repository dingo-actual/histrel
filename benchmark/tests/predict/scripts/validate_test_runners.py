#!/usr/bin/env python3
"""Validate SWE-bench test runner scripts.

This script validates test runner script generation for the single
run_tests runner with proper test framework support.

    Usage:
        python scripts/validate_test_runners.py [--verbose] [--quick]
"""

from __future__ import annotations

import argparse
import json
import sys


def validate_runner_script(script: str) -> dict:
    """Validate the runner script has required structure."""
    results = {
        "shebang": script.startswith("#!/opt/miniconda3/envs/testbed/bin/python"),
        "imports": "import json" in script,
        "main_function": "def main():" in script,
        "result_output": "/tmp/test_run_" in script,
        "delta_output": "test_run_delta_" in script,
        "spec_reading": "test_spec.json" in script,
        "summary": "REGRESSIONS (new failures vs run" in script,
    }

    return {
        "valid": all(results.values()),
        "checks": results,
    }


def validate_test_framework_support(test_cmd: str) -> dict:
    """Validate test command has supported framework."""
    frameworks = {
        "pytest": "python -m pytest" in test_cmd or "pytest" in test_cmd,
        "django": "django test" in test_cmd.lower(),
        "tox": "tox" in test_cmd.lower(),
        "unittest": "python -m unittest" in test_cmd,
    }
    detected = [f for f, detected in frameworks.items() if detected]
    return {
        "supported": len(detected) > 0,
        "frameworks": frameworks,
        "primary_framework": detected[0] if detected else None,
    }


def validate_spec_format(spec_data: dict) -> dict:
    """Validate test specification format."""
    if not isinstance(spec_data, dict):
        return {"valid": False, "error": "Spec must be a dictionary"}

    for file_path, test_list in spec_data.items():
        if not isinstance(file_path, str):
            return {"valid": False, "error": "File path must be string"}
        if not isinstance(test_list, list):
            return {"valid": False, "error": "Test list must be array"}

    return {
        "valid": True,
        "test_file_count": len(spec_data),
        "total_tests": sum(len(tests) for tests in spec_data.values()),
    }


def validate_install_command(install_cmd: str | None) -> dict:
    """Validate install command format."""
    if install_cmd is None:
        return {"valid": True, "reason": "No install command"}
    valid_patterns = ["pip install", "conda install", "npm install", "poetry add"]
    detected = [p for p in valid_patterns if p in install_cmd]
    return {
        "valid": len(detected) > 0,
        "install_type": detected[0] if detected else "unknown",
        "command": install_cmd,
    }


def run_quick_validation() -> bool:
    """Run quick validation checks."""
    print("Running quick test runner validation...")

    try:
        from benchmark.predict.swebench.test_runner_generators import (
            TestRunnerGenerator,
        )
        from benchmark.tests.fixtures.swebench import (
            make_test_runner_execution_env,
        )

        generator = TestRunnerGenerator(
            test_cmd="python -m pytest",
            install_cmd="pip install -e .",
        )
        script = generator.generate_runner()

        assert script.startswith("#!/opt/miniconda3/envs/testbed/bin/python")
        assert "import json" in script
        assert script.count("print(") > 0

        with make_test_runner_execution_env() as tmpdir:
            spec_file = tmpdir / "test_spec.json"
            spec = json.loads(spec_file.read_text())
            assert isinstance(spec, dict)

        print("  ✓ All runner scripts generate correctly")
        return True
    except Exception as e:
        print(f"  ✗ Quick validation failed: {e}")
        return False


def run_full_validation(verbose: bool = False) -> bool:
    """Run full validation suite."""
    print("Running full test runner validation...")

    from benchmark.predict.swebench.test_runner_generators import (
        TestRunnerGenerator,
    )
    from benchmark.tests.fixtures.swebench import (
        make_test_runner_execution_env,
    )

    all_passed = True

    generator = TestRunnerGenerator(
        test_cmd="python -m pytest",
        install_cmd="pip install -e .",
        eval_commands=["export VAR=value"],
    )

    script = generator.generate_runner()

    if verbose:
        print(f"  Runner script length: {len(script)} chars")

    script_result = validate_runner_script(script)

    assert script_result["valid"], f"Runner validation failed: {script_result}"
    print("  ✓ Runner structure valid")

    framework_check = validate_test_framework_support(generator.test_cmd)
    assert framework_check["supported"], f"Framework not detected: {framework_check}"
    print(f"  ✓ Framework detected: {framework_check['primary_framework']}")

    with make_test_runner_execution_env() as tmpdir:
        spec_file = tmpdir / "test_spec.json"
        spec_data = json.loads(spec_file.read_text())
        spec_result = validate_spec_format(spec_data)
        assert spec_result["valid"], f"Spec format invalid: {spec_result}"
        print(f"  ✓ Test spec valid: {spec_result['test_file_count']} files")

    install_result = validate_install_command(generator.install_cmd)
    assert install_result["valid"], f"Install validation failed: {install_result}"
    print(f"  ✓ Install command valid: {install_result['install_type']}")

    return all_passed


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate SWE-bench test runners")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Quick validation only"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SWE-bench Test Runner Validation")
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
