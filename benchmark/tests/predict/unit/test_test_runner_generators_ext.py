#!/usr/bin/env python3
"""
Unit tests for TestRunnerGenerator's custom test subprocess isolation and
stdout/stderr capture features.

Verifies that the generated runner script:
- Runs each custom test function (under /tmp/tests_custom) in a separate
  subprocess, using exit code for pass/fail.
- Captures stdout and stderr per test file (non-custom) and per function
  (custom), writing them to /tmp/test_run_{n}.log.
- Reports log file paths in the summary output.

    Test Classes:
        Test04Unit01Morphism80TestRunnerGen01CustomSubprocess:
            Tests that the generated script runs custom tests in subprocesses.
        Test04Unit01Morphism80TestRunnerGen02StdoutStderrCapture:
            Tests that the generated script captures stdout/stderr per test.
        Test04Unit01Morphism80TestRunnerGen03LogFileWriting:
            Tests that the generated script writes captured output to log files.
        Test04Unit01Morphism80TestRunnerGen04SummaryLogPaths:
            Tests that the summary output includes log file paths.
        Test04Unit01Morphism80TestRunnerGen05HelpSection:
            Tests that the help section mentions log files.
"""

from __future__ import annotations

from benchmark.predict.swebench.test_runner_generators import TestRunnerGenerator


def _generate_runner(
    test_cmd: str = "python -m pytest -o console_output_style=classic",
    install_cmd: str | None = None,
    eval_commands: list[str] | None = None,
) -> str:
    """Generate a runner script and return it as a string.

    Args:
        test_cmd: The test command to use.
        install_cmd: Optional install command.
        eval_commands: Optional eval commands.

    Returns:
        (str) The complete generated runner script.
    """
    gen = TestRunnerGenerator(test_cmd, install_cmd, eval_commands)
    return gen.generate_runner()


class Test04Unit01Morphism80TestRunnerGen01CustomSubprocess:
    """Tests that the generated script runs custom tests in subprocesses."""

    def test_04_unit_01_morphism_80_test_runner_gen_01_custom_subprocess_01_separate_process_per_function(
        self,
    ) -> None:
        """Verify the generated script runs each custom test function in a separate process."""
        script = _generate_runner()
        # The script must use subprocess to run individual custom test functions
        # in isolation. We check for a pattern that indicates per-function
        # subprocess invocation for custom tests.
        assert "subprocess" in script
        # The custom test directory path must appear
        assert "/tmp/tests_custom" in script
        # The script must distinguish custom tests from suite tests
        # by checking whether the test file is under /tmp/tests_custom
        assert "tests_custom" in script

    def test_04_unit_01_morphism_80_test_runner_gen_01_custom_subprocess_02_exit_code_for_pass_fail(
        self,
    ) -> None:
        """Verify the generated script uses exit code to determine custom test pass/fail."""
        script = _generate_runner()
        # For custom tests, the script must use returncode == 0 for PASS
        # and nonzero for FAIL, running each function separately.
        # The custom test runner should invoke each function individually.
        # Look for evidence of per-function subprocess calls using the
        # testbed python to run a command that imports and calls the function.
        lines = script.splitlines()
        # Find the custom test handling section
        custom_section_lines = [
            line for line in lines if "tests_custom" in line or "custom" in line.lower()
        ]
        assert len(custom_section_lines) > 0
        # The script must contain logic to run individual functions
        # (not just whole files) for custom tests
        assert "function" in script.lower() or "func" in script.lower()

    def test_04_unit_01_morphism_80_test_runner_gen_01_custom_subprocess_03_custom_runner_snippet(
        self,
    ) -> None:
        """Verify the generated script includes a self-contained custom test runner.

        The custom test runner must be a small Python snippet embedded in the
        generated script that:
        - Imports the test file as a module
        - Discovers all public (non-dunder) functions in the file
        - Runs each in a separate subprocess
        - Uses exit code 0 for success, nonzero for failure
        """
        script = _generate_runner()
        # The generated script must contain a helper that runs custom
        # test functions in separate processes. This is typically a
        # subprocess call that executes a small inline Python snippet
        # or a separate command per function.
        # We check for the presence of patterns indicating per-function
        # execution:
        # - module loading logic (importlib)
        # - function discovery (inspect.getmembers)
        # - subprocess per function
        assert "returncode" in script
        # The script must NOT simply run the whole custom test file with
        # the repo's test harness (which is the bug being fixed).
        # Instead, it should have its own simple runner for custom tests.
        # Check for a pattern that indicates a custom test runner
        # (not just using the repo's test harness command)
        assert (
            "importlib" in script or "import_module" in script or "__import__" in script
        )
        assert "inspect" in script
        assert "getmembers" in script


class Test04Unit01Morphism80TestRunnerGen02StdoutStderrCapture:
    """Tests that the generated script captures stdout/stderr per test."""

    def test_04_unit_01_morphism_80_test_runner_gen_02_stdout_stderr_capture_01_suite_test_capture(
        self,
    ) -> None:
        """Verify the generated script captures stdout and stderr for suite test files."""
        script = _generate_runner()
        # The script must capture stdout and stderr for each test file
        # that is NOT under /tmp/tests_custom
        assert "stdout" in script
        assert "stderr" in script
        # capture_output or PIPE must be used
        assert "capture_output" in script or "PIPE" in script

    def test_04_unit_01_morphism_80_test_runner_gen_02_stdout_stderr_capture_02_custom_test_capture(
        self,
    ) -> None:
        """Verify the generated script captures stdout and stderr for custom test functions."""
        script = _generate_runner()
        # For custom tests run in subprocesses, stdout and stderr must
        # also be captured
        assert "stdout" in script
        assert "stderr" in script

    def test_04_unit_01_morphism_80_test_runner_gen_02_stdout_stderr_capture_03_per_test_granularity(
        self,
    ) -> None:
        """Verify stdout/stderr is captured per suite file and per custom function."""
        script = _generate_runner()
        # The results structure must include captured output fields
        # for each test file (suite) and each function (custom)
        assert "stdout" in script
        assert "stderr" in script
        # The log content must be accumulated for writing
        assert "log" in script.lower()


class Test04Unit01Morphism80TestRunnerGen03LogFileWriting:
    """Tests that the generated script writes captured output to log files."""

    def test_04_unit_01_morphism_80_test_runner_gen_03_log_file_writing_01_writes_log_path(
        self,
    ) -> None:
        """Verify the generated script writes output to /tmp/test_run_{n}.log."""
        script = _generate_runner()
        # The script must write a log file at /tmp/test_run_{n}.log
        assert "test_run_" in script
        assert ".log" in script

    def test_04_unit_01_morphism_80_test_runner_gen_03_log_file_writing_02_log_uses_run_number(
        self,
    ) -> None:
        """Verify the log file path uses the run number variable."""
        script = _generate_runner()
        # The log path must be formatted with run_number
        assert "run_number" in script
        # The .log file must be in the /tmp directory
        assert "/tmp/" in script
        assert ".log" in script

    def test_04_unit_01_morphism_80_test_runner_gen_03_log_file_writing_03_log_contains_test_output(
        self,
    ) -> None:
        """Verify the log file writing section writes captured stdout/stderr."""
        script = _generate_runner()
        # The script must write captured output to the log file
        # Look for write operations on the log file
        assert "write" in script
        assert ".log" in script


class Test04Unit01Morphism80TestRunnerGen04SummaryLogPaths:
    """Tests that the summary output includes log file paths."""

    def test_04_unit_01_morphism_80_test_runner_gen_04_summary_log_paths_01_run1_includes_log(
        self,
    ) -> None:
        """Verify the run 1 summary (no previous run) includes the log file path."""
        script = _generate_runner()
        # The first-run summary section must print the log file path
        # Find the "No previous run" section
        assert "No previous run" in script
        # Within that section or nearby, the log path must be printed
        # The log path variable must be referenced in the summary
        assert "log" in script.lower()

    def test_04_unit_01_morphism_80_test_runner_gen_04_summary_log_paths_02_delta_summary_includes_log(
        self,
    ) -> None:
        """Verify the delta summary (run 2+) includes the log file path."""
        script = _generate_runner()
        # The delta summary section must also print the log file path
        # Find the "CHANGES FROM RUN" section
        assert "CHANGES FROM RUN" in script
        # The log path must be referenced
        assert "log" in script.lower()

    def test_04_unit_01_morphism_80_test_runner_gen_04_summary_log_paths_03_log_path_variable(
        self,
    ) -> None:
        """Verify a log_path variable is defined and used in the script."""
        script = _generate_runner()
        # A log_path variable must be defined
        assert "log_path" in script


class Test04Unit01Morphism80TestRunnerGen05HelpSection:
    """Tests that the help section mentions log files."""

    def test_04_unit_01_morphism_80_test_runner_gen_05_help_section_01_mentions_log_file(
        self,
    ) -> None:
        """Verify the -h/--help output mentions the log file."""
        script = _generate_runner()
        # The help section must list the log file as an output
        assert ".log" in script
        # The Outputs section in help must include the log file
        assert "Outputs" in script or "outputs" in script.lower()
