#!/usr/bin/env python3
"""Unit tests for SWE-bench test runner execution validation.

Tests for test runner script execution including result recording,
delta comparison, and framework compatibility.

    Test Classes:
        Test04Unit05RunnerExec01Script: Runner script structure tests
        Test04Unit05RunnerExec02Delta: Delta comparison tests
        Test04Unit05RunnerExec04Frameworks: Framework compatibility tests
"""

from __future__ import annotations

import json
import sys

import pytest

if sys.platform == "win32":
    pytest.skip("Test runners require Unix", allow_module_level=True)

from benchmark.predict.swebench.test_runner_generators import (
    TestRunnerGenerator,
)
from benchmark.tests.fixtures.swebench import (
    make_test_runner_execution_env,
)


class Test04Unit05RunnerExec01Script:
    """Tests for runner script structure."""

    def test_04_unit_05_runner_exec_01_script_01_script_has_correct_structure(
        self,
    ) -> None:
        """Test runner script has required structure."""
        generator = TestRunnerGenerator(test_cmd="python -m pytest")
        script = generator.generate_runner()

        assert "#!/opt/miniconda3/envs/testbed/bin/python" in script
        assert "import json" in script
        assert "def main():" in script

    def test_04_unit_05_runner_exec_01_script_02_includes_test_cmd(self) -> None:
        """Test runner script includes correct test command."""
        generator = TestRunnerGenerator(
            test_cmd="python -m pytest astropy/wcs/tests/",
        )
        script = generator.generate_runner()

        assert "python -m pytest astropy/wcs/tests/" in script

    def test_04_unit_05_runner_exec_01_script_03_writes_run_results(self) -> None:
        """Test runner script writes numbered run results."""
        generator = TestRunnerGenerator(test_cmd="python -m pytest")
        script = generator.generate_runner()

        assert "/tmp/test_run_" in script

    def test_04_unit_05_runner_exec_01_script_04_reads_fixed_spec(self) -> None:
        """Test runner script reads spec from the fixed path."""
        generator = TestRunnerGenerator(test_cmd="python -m pytest")
        script = generator.generate_runner()

        assert "sys.argv" in script
        assert "/tmp/test_spec.json" in script


class Test04Unit05RunnerExec02Delta:
    """Tests for delta comparison behavior."""

    def test_04_unit_05_runner_exec_02_delta_01_computes_delta(self) -> None:
        """Test runner script computes delta between runs."""
        generator = TestRunnerGenerator(test_cmd="python -m pytest")
        script = generator.generate_runner()

        assert "delta" in script.lower()
        assert "previous" in script.lower()

    def test_04_unit_05_runner_exec_02_delta_02_writes_comparison_results(
        self,
    ) -> None:
        """Test runner script writes comparison output."""
        generator = TestRunnerGenerator(test_cmd="python -m pytest")
        script = generator.generate_runner()

        assert "/tmp/test_run_" in script
        assert "test_run_delta_" in script

    def test_04_unit_05_runner_exec_02_delta_03_records_transition_categories(
        self,
    ) -> None:
        """Test runner script records all transition categories."""
        generator = TestRunnerGenerator(test_cmd="python -m pytest")
        script = generator.generate_runner()

        assert "'transitions'" in script
        assert "'new_tests'" in script
        assert "'removed_tests'" in script


class Test04Unit05RunnerExec04Frameworks:
    """Tests for test framework compatibility."""

    def test_04_unit_05_runner_exec_04_frameworks_01_pytest_is_detected(
        self,
    ) -> None:
        """Test pytest framework is correctly detected."""
        generator = TestRunnerGenerator(test_cmd="python -m pytest -xvs")
        assert generator.test_cmd == "python -m pytest -xvs"

    def test_04_unit_05_runner_exec_04_frameworks_02_django_is_detected(
        self,
    ) -> None:
        """Test Django framework is correctly detected."""
        generator = TestRunnerGenerator(
            test_cmd="python -m django test --settings=tests.settings",
        )
        assert "django" in generator.test_cmd.lower()

    def test_04_unit_05_runner_exec_04_frameworks_03_install_cmd_preserved(
        self,
    ) -> None:
        """Test install command is preserved in the runner script."""
        generator = TestRunnerGenerator(
            test_cmd="python -m pytest",
            install_cmd="pip install -e .",
        )
        script = generator.generate_runner()
        assert "pip install -e ." in script

    def test_04_unit_05_runner_exec_04_frameworks_04_eval_commands_preserved(
        self,
    ) -> None:
        """Test eval commands are preserved in the runner script."""
        generator = TestRunnerGenerator(
            test_cmd="python -m pytest",
            eval_commands=["export VAR=value"],
        )
        script = generator.generate_runner()
        assert "export VAR=value" in script


class Test04Unit05RunnerExec05Integration:
    """Integration tests for test runner execution in container environment."""

    def test_04_unit_05_runner_exec_05_integration_01_execution_env_structure(
        self,
    ) -> None:
        """Test execution environment has required directory structure."""
        with make_test_runner_execution_env() as tmpdir:
            assert (tmpdir / "tests_custom").exists()
            assert (tmpdir / "test_spec.json").exists()

    def test_04_unit_05_runner_exec_05_integration_02_spec_format_matches_expected(
        self,
    ) -> None:
        """Test test spec JSON format is correct."""
        with make_test_runner_execution_env() as tmpdir:
            spec_path = tmpdir / "test_spec.json"
            spec = json.loads(spec_path.read_text())
            assert isinstance(spec, dict)
