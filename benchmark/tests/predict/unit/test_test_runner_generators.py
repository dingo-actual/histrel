#!/usr/bin/env python3
"""Unit tests for test runner script generators.

Tests for TestRunnerGenerator creating the single executable Python
`run_tests` script that runs all tests specified in /tmp/test_spec.json,
records results, and computes a delta against the previous run inside
SWE-bench containers.

    Test Classes:
        Test04Unit02RunnerGen01Creation: Generator creation and initialization tests
        Test04Unit02RunnerGen02Script: Single runner script structure tests
        Test04Unit02RunnerGen03Results: Result recording tests
        Test04Unit02RunnerGen04Delta: Delta computation tests
        Test04Unit02RunnerGen05Summary: Summary output tests
        Test04Unit02RunnerGen06Integration: Integration and edge case tests
        Test04Unit02RunnerGen07Py35Compat: Python 3.5 compatibility tests
"""

from __future__ import annotations

import ast

from benchmark.predict.swebench.test_runner_generators import (
    TestRunnerGenerator,
)


class Test04Unit02RunnerGen01Creation:
    """Tests for TestRunnerGenerator creation and initialization."""

    def test_04_unit_02_runner_gen_01_creation_01_basic_initialization(self) -> None:
        """Test generator can be created with minimal parameters."""
        generator = TestRunnerGenerator(test_cmd="pytest")

        assert generator.test_cmd == "pytest"
        assert generator.install_cmd is None
        assert generator.eval_commands == []

    def test_04_unit_02_runner_gen_01_creation_02_full_initialization(self) -> None:
        """Test generator can be created with all parameters."""
        generator = TestRunnerGenerator(
            test_cmd="python -m pytest -v",
            install_cmd="pip install -e .",
            eval_commands=["export VAR=value"],
        )

        assert generator.test_cmd == "python -m pytest -v"
        assert generator.install_cmd == "pip install -e ."
        assert generator.eval_commands == ["export VAR=value"]

    def test_04_unit_02_runner_gen_01_creation_03_empty_eval_commands(self) -> None:
        """Test generator handles empty eval_commands list."""
        generator = TestRunnerGenerator(
            test_cmd="pytest",
            eval_commands=[],
        )

        assert generator.eval_commands == []


class Test04Unit02RunnerGen02Script:
    """Tests for single runner script structure."""

    def test_04_unit_02_runner_gen_02_script_01_script_structure(self) -> None:
        """Test script has required structure."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "#!/opt/miniconda3/envs/testbed/bin/python" in script
        assert "import json" in script
        assert "import subprocess" in script
        assert "def main():" in script

    def test_04_unit_02_runner_gen_02_script_02_has_shebang(self) -> None:
        """Test script starts with correct shebang."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        lines = script.split("\n")
        assert lines[0] == "#!/opt/miniconda3/envs/testbed/bin/python"

    def test_04_unit_02_runner_gen_02_script_03_test_cmd_included(self) -> None:
        """Test script includes test command."""
        generator = TestRunnerGenerator(test_cmd="python -m pytest -xvs")
        script = generator.generate_runner()

        assert "python -m pytest -xvs" in script

    def test_04_unit_02_runner_gen_02_script_04_fixed_spec_path(self) -> None:
        """Test script always reads the fixed spec path."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "/tmp/test_spec.json" in script

    def test_04_unit_02_runner_gen_02_script_05_takes_no_arguments(self) -> None:
        """Test script does not read any positional command line arguments."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "sys.argv[1]" not in script

    def test_04_unit_02_runner_gen_02_script_06_custom_test_discovery(self) -> None:
        """Test script discovers custom tests from /tmp/tests_custom/."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "/tmp/tests_custom" in script
        assert "*.py" in script

    def test_04_unit_02_runner_gen_02_script_07_help_flag_short(self) -> None:
        """Test script handles -h and prints usage."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "'-h' in sys.argv" in script
        assert "'--help' in sys.argv" in script
        assert "Usage: run_tests" in script

    def test_04_unit_02_runner_gen_02_script_08_help_returns_zero(self) -> None:
        """Test help branch returns 0 before any file access."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        lines = script.split("\n")
        help_idx = next(i for i, line in enumerate(lines) if "'-h' in sys.argv" in line)
        return_idx = next(
            i
            for i, line in enumerate(lines[help_idx:], start=help_idx)
            if "return 0" in line
        )
        open_idx = next(
            i
            for i, line in enumerate(lines[help_idx:], start=help_idx)
            if ".open()" in line
        )
        assert return_idx < open_idx

    def test_04_unit_02_runner_gen_02_script_09_pins_testbed_environment(self) -> None:
        """Test script pins child commands to the testbed env."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "import os" in script
        assert (
            "os.environ['PATH'] = '/opt/miniconda3/envs/testbed/bin:'"
            " + os.environ.get('PATH', '')" in script
        )

    def test_04_unit_02_runner_gen_02_script_10_no_install_skips_marker(self) -> None:
        """Test script without install command skips marker logic."""
        generator = TestRunnerGenerator(test_cmd="pytest", install_cmd=None)
        script = generator.generate_runner()

        assert "swebench_install_done" not in script

    def test_04_unit_02_runner_gen_02_script_11_with_install_has_marker(self) -> None:
        """Test script with install command includes marker logic."""
        generator = TestRunnerGenerator(
            test_cmd="pytest",
            install_cmd="pip install -e .",
        )
        script = generator.generate_runner()

        assert "swebench_install_done" in script
        assert "pip install -e ." in script

    def test_04_unit_02_runner_gen_02_script_12_no_eval_skips_section(self) -> None:
        """Test script without eval commands skips eval section."""
        generator = TestRunnerGenerator(test_cmd="pytest", eval_commands=[])
        script = generator.generate_runner()

        assert "Run eval commands" not in script

    def test_04_unit_02_runner_gen_02_script_13_with_eval_has_section(self) -> None:
        """Test script with eval commands includes eval section."""
        generator = TestRunnerGenerator(
            test_cmd="pytest",
            eval_commands=["export DJANGO_SETTINGS_MODULE=test.settings"],
        )
        script = generator.generate_runner()

        assert "Run eval commands" in script
        assert "DJANGO_SETTINGS_MODULE" in script


class Test04Unit02RunnerGen03Results:
    """Tests for result recording behavior."""

    def test_04_unit_02_runner_gen_03_results_01_increments_counter(self) -> None:
        """Test script increments the run counter."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "test_run_counter" in script
        assert "run_number =" in script

    def test_04_unit_02_runner_gen_03_results_02_writes_run_results(self) -> None:
        """Test script writes full results with run number."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "test_run_{run_number}.json" in script

    def test_04_unit_02_runner_gen_03_results_03_no_baseline_results_file(self) -> None:
        """Test script does not write the legacy baseline results file."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "/tmp/tests_baseline.json" not in script

    def test_04_unit_02_runner_gen_03_results_04_reads_previous_run(self) -> None:
        """Test script reads the previous run's results when present."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "'/tmp/test_run_{0}.json'.format(run_number - 1)" in script

    def test_04_unit_02_runner_gen_03_results_05_skips_delta_without_previous(
        self,
    ) -> None:
        """Test script writes no delta file for the first run."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        # The delta path is written only when the previous results file exists.
        assert "if not previous_path.exists():" in script

    def test_04_unit_02_runner_gen_03_results_06_executes_spec_files(self) -> None:
        """Test script executes every test file in the spec."""
        generator = TestRunnerGenerator(test_cmd="python -m pytest")
        script = generator.generate_runner()

        assert "for test_file, test_names in test_spec.items():" in script
        assert "cwd='/testbed'" in script

    def test_04_unit_02_runner_gen_03_results_07_empty_list_runs_all_tests(
        self,
    ) -> None:
        """Test script treats an empty test name list as run-all for the file."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "if not test_names:" in script

    def test_04_unit_02_runner_gen_03_results_08_exit_code_heuristic(self) -> None:
        """Test script records PASS or FAIL per test from the exit code."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "'PASS' if result.returncode == 0 else 'FAIL'" in script


class Test04Unit02RunnerGen04Delta:
    """Tests for delta computation behavior."""

    def test_04_unit_02_runner_gen_04_delta_01_writes_delta_file(self) -> None:
        """Test script writes the delta file with the run number."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "test_run_delta_{run_number}.json" in script

    def test_04_unit_02_runner_gen_04_delta_02_records_all_transitions(self) -> None:
        """Test delta records all four status transitions, including unchanged."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "'from': previous_status" in script
        assert "'to': status" in script

    def test_04_unit_02_runner_gen_04_delta_03_records_new_tests(self) -> None:
        """Test delta records new tests with their resulting status."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "'new_tests': new_tests" in script

    def test_04_unit_02_runner_gen_04_delta_04_records_removed_tests(self) -> None:
        """Test delta records removed tests."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "'removed_tests': removed_tests" in script

    def test_04_unit_02_runner_gen_04_delta_05_no_delta_without_previous(
        self,
    ) -> None:
        """Test the delta write block sits inside the has-previous branch."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        # The delta path is written only when the previous results file exists.
        assert script.count("if not previous_path.exists():") >= 1
        assert script.index("if not previous_path.exists():") < script.index(
            "delta_path = Path("
        )


class Test04Unit02RunnerGen05Summary:
    """Tests for stdout summary behavior."""

    def test_04_unit_02_runner_gen_05_summary_01_prints_run_banner(self) -> None:
        """Test summary prints a banner with the run number."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "Test Run" in script
        assert "RESULTS:" in script

    def test_04_unit_02_runner_gen_05_summary_02_prints_transition_counts(self) -> None:
        """Test summary prints counts for every transition category."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "PASS -> PASS:" in script
        assert "PASS -> FAIL:" in script
        assert "FAIL -> PASS:" in script
        assert "FAIL -> FAIL:" in script
        assert "NEW TESTS:" in script
        assert "REMOVED TESTS:" in script

    def test_04_unit_02_runner_gen_05_summary_03_prints_regressions_line(self) -> None:
        """Test summary prints a REGRESSIONS line that names its basis.

        A bare ``REGRESSIONS: 0`` next to ``TOTALS: 7 tests: 2 PASS, 5
        FAIL`` reads as "the suite has no failures", when it only means
        that nothing got worse since the previous run.
        """
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "REGRESSIONS (new failures vs run {0}): {1}" in script

    def test_04_unit_02_runner_gen_05_summary_04_prints_transition_details(
        self,
    ) -> None:
        """Test summary lists per-test transition details."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "PASS -> FAIL" in script
        assert "FAIL -> PASS" in script

    def test_04_unit_02_runner_gen_05_summary_05_prints_new_and_removed_details(
        self,
    ) -> None:
        """Test summary lists new failing tests and removed tests."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "NEW FAILING TESTS:" in script
        assert "REMOVED TESTS" in script

    def test_04_unit_02_runner_gen_05_summary_06_prints_totals(self) -> None:
        """Test summary prints overall PASS and FAIL totals."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "TOTALS:" in script

    def test_04_unit_02_runner_gen_05_summary_07_first_run_has_no_previous(
        self,
    ) -> None:
        """Test summary notes the first run has no previous comparison."""
        generator = TestRunnerGenerator(test_cmd="pytest")
        script = generator.generate_runner()

        assert "No previous run" in script


class Test04Unit02RunnerGen06Integration:
    """Integration and edge case tests for runner generators."""

    def test_04_unit_02_runner_gen_06_integration_01_single_runner_generated(
        self,
    ) -> None:
        """Test the generator produces exactly one run_tests script."""
        generator = TestRunnerGenerator(
            test_cmd="python -m pytest",
            install_cmd="pip install -e .",
            eval_commands=["export VAR=value"],
        )

        script = generator.generate_runner()

        assert script
        assert "run_tests" in script

    def test_04_unit_02_runner_gen_06_integration_02_complex_test_cmd(self) -> None:
        """Test generator handles complex test commands with flags."""
        generator = TestRunnerGenerator(
            test_cmd="python -m pytest -xvs --tb=short --no-header",
        )
        script = generator.generate_runner()

        assert "-xvs --tb=short --no-header" in script

    def test_04_unit_02_runner_gen_06_integration_03_empty_strings(self) -> None:
        """Test generator handles empty string parameters gracefully."""
        generator = TestRunnerGenerator(
            test_cmd="",
            install_cmd="",
            eval_commands=[""],
        )
        script = generator.generate_runner()

        # Should still generate valid script
        assert "def main():" in script

    def test_04_unit_02_runner_gen_06_integration_04_special_chars_in_cmd(self) -> None:
        """Test generator handles special characters in commands."""
        generator = TestRunnerGenerator(
            test_cmd="pytest -k 'test_name or other_test'",
        )
        script = generator.generate_runner()

        # Should properly escape/quote the command
        assert "pytest" in script

    def test_04_unit_02_runner_gen_06_integration_05_env_setup_precedes_subprocesses(
        self,
    ) -> None:
        """Test script sets the testbed PATH before any subprocess."""
        generator = TestRunnerGenerator(
            test_cmd="pytest",
            install_cmd="pip install -e .",
            eval_commands=["export VAR=value"],
        )

        script = generator.generate_runner()
        lines = script.split("\n")
        env_idx = next(
            i for i, line in enumerate(lines) if "os.environ['PATH']" in line
        )
        subprocess_idx = next(
            i for i, line in enumerate(lines) if "subprocess.run" in line
        )
        assert env_idx < subprocess_idx

    def test_04_unit_02_runner_gen_06_integration_06_compiles_to_valid_python(
        self,
    ) -> None:
        """Test generated script is syntactically valid Python."""
        generator = TestRunnerGenerator(
            test_cmd="python -m pytest",
            install_cmd="pip install -e .",
            eval_commands=["export VAR=value"],
        )
        script = generator.generate_runner()

        compile(script, "run_tests", "exec")


class Test04Unit02RunnerGen07Py35Compat:
    """Tests that the generated script stays compatible with Python 3.5.

    Testbed interpreters span Python 3.5 to 3.11 across SWE-bench
    instances, so the script must not use newer language features.
    """

    def _generate_full_runner(self) -> str:
        """Generate a runner with every optional section present."""
        generator = TestRunnerGenerator(
            test_cmd="python -m pytest",
            install_cmd="pip install -e .",
            eval_commands=["export VAR=value"],
        )
        return generator.generate_runner()

    def test_04_unit_02_runner_gen_07_py35_compat_01_no_f_strings(self) -> None:
        """f-strings require Python 3.6 (PEP 498), so none may appear."""
        script = self._generate_full_runner()

        tree = ast.parse(script)
        joined = [node for node in ast.walk(tree) if isinstance(node, ast.JoinedStr)]
        assert joined == []

    def test_04_unit_02_runner_gen_07_py35_compat_03_parses_as_3_5_grammar(
        self,
    ) -> None:
        """The script parses under the Python 3.5 grammar."""
        script = self._generate_full_runner()

        ast.parse(script, feature_version=(3, 5))

    def test_04_unit_02_runner_gen_07_py35_compat_02_no_py37_subprocess_kwargs(
        self,
    ) -> None:
        """subprocess.run capture_output/text arguments require Python 3.7."""
        script = self._generate_full_runner()

        assert "capture_output" not in script
        assert "text=True" not in script
        assert "stdout=subprocess.PIPE" in script
        assert "universal_newlines=True" in script
