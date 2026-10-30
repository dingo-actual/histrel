#!/usr/bin/env python3
"""Test runner script generator for SWE-bench integration.

This module contains the generator for the single executable Python
`run_tests` script that runs all tests specified in /tmp/test_spec.json
inside SWE-bench Docker containers. The script is embedded in instance
images and executed by the benchmark agent. Every invocation writes the run's full results to
/tmp/test_run_{n}.json; from the second run on, it also writes a delta
against /tmp/test_run_{n-1}.json to /tmp/test_run_delta_{n}.json and
prints a change summary.

    Important Classes:
        TestRunnerGenerator: Generates the test runner script for SWE-bench instances
"""

from __future__ import annotations

# The spec path is fixed inside the container; the script takes no spec
# path argument.
_SPEC_PATH = "/tmp/test_spec.json"

# Child commands (eval, install, and the test command) run through a
# shell that inherits the session environment. In the benchmark container
# the session PATH points at the agent conda env, whose Python version
# differs from the testbed Python the repository was built for, so the
# generated script pins child commands to the testbed env by putting its
# bin directory first on PATH.
_TESTBED_BIN = "/opt/miniconda3/envs/testbed/bin"


class TestRunnerGenerator:
    """Generates the test runner script for SWE-bench instances.

    This class encapsulates the test command configuration and generates
    a single script that runs every invocation as a numbered run:
    - First run: establishes the baseline (results only, no delta)
    - Later runs: records results and the delta against the previous run

    Attributes:
        test_cmd: The test command from MAP_REPO_VERSION_TO_SPECS
        install_cmd: Optional install command to run before tests
        eval_commands: List of eval commands for environment setup
    """

    __test__ = False  # prevent pytest from trying to collect this as a test class

    # The injected script runs on the testbed interpreter, which varies from
    # Python 3.5 to 3.11 across SWE-bench instances. The generated script must
    # therefore avoid f-strings (3.6+, PEP 498) and the subprocess.run
    # ``capture_output`` and ``text`` keyword arguments (3.7+); use
    # ``str.format`` and ``stdout=PIPE``/``stderr=PIPE`` with
    # ``universal_newlines=True`` instead.

    def __init__(
        self,
        test_cmd: str,
        install_cmd: str | None = None,
        eval_commands: list[str] | None = None,
    ) -> None:
        """Initialize generator with test configuration.

        Args:
            test_cmd: The test command from MAP_REPO_VERSION_TO_SPECS
            install_cmd: Optional install command
            eval_commands: Optional list of eval commands
        """
        self.test_cmd = test_cmd
        self.install_cmd = install_cmd
        self.eval_commands = eval_commands or []

    def _generate_help_check(
        self,
        usage: str,
        description: str,
        sections: dict[str, list[str]],
    ) -> list[str]:
        """Generate -h/--help argument handling section.

        Args:
            usage: Usage line to display.
            description: One-line description of the script's purpose.
            sections: Ordered sections to display, mapping section headers
                to entry lines.

        Returns:
            List of script lines that check for -h/--help and print usage.
        """
        lines = [
            "    # Check for help flag",
            "    if '-h' in sys.argv or '--help' in sys.argv:",
            f"        print('{usage}')",
            "        print()",
            f"        print('{description}')",
        ]
        for header, entries in sections.items():
            lines.append("        print()")
            lines.append(f"        print('{header}:')")
            for entry in entries:
                lines.append(f"        print('    {entry}')")
        lines.append("        return 0")
        lines.append("    ")
        return lines

    def generate_runner(self) -> str:
        """Generate the Python script for test execution with delta tracking.

        The generated script takes no arguments and always reads the test
        spec from /tmp/test_spec.json. Run 1 records results only; every
        later run also writes a delta against the previous run.

        Returns:
            Complete Python script as string that can be embedded in instance image
        """
        lines = self._generate_header()
        lines.extend(self._generate_main())
        return "\n".join(lines)

    def _generate_header(self) -> list[str]:
        """Generate the header section of the runner."""
        return [
            "#!/opt/miniconda3/envs/testbed/bin/python",
            '"""Test runner for SWE-bench instances."""',
            "import importlib",
            "import inspect",
            "import json",
            "import os",
            "import subprocess",
            "import sys",
            "from pathlib import Path",
            "",
        ]

    def _generate_testbed_env_setup(self) -> list[str]:
        """Generate environment setup pinning children to the testbed env.

        Returns:
            List of script lines that prepend the testbed environment's
            bin directory to PATH.
        """
        return [
            "    # Pin child commands to the testbed environment: the",
            "    # session PATH points at the agent env, whose Python",
            "    # differs from the Python the repository was built for.",
            f"    os.environ['PATH'] = '{_TESTBED_BIN}:' + os.environ.get('PATH', '')",
            "    ",
        ]

    def _generate_main(self) -> list[str]:
        """Generate the main function of the runner."""
        lines = [
            "def main():",
            '    """Execute the specified tests, record results, and diff runs."""',
        ]
        lines.extend(self._generate_testbed_env_setup())
        lines.extend(
            self._generate_help_check(
                usage="Usage: run_tests",
                description=(
                    "Run all tests specified in /tmp/test_spec.json and record results."
                ),
                sections={
                    "Outputs": [
                        "/tmp/test_run_N.json        - Full results for run N",
                        "/tmp/test_run_delta_N.json  - Changes from run N-1 (runs 2+ only)",
                        "/tmp/test_run_N.log         - Captured stdout/stderr per test for run N",
                    ],
                },
            )
        )
        lines.extend(
            [
                f"    # Read test spec (which tests to run) from {_SPEC_PATH}",
                f"    with Path('{_SPEC_PATH}').open() as f:",
                "        test_spec = json.load(f)",
                "    ",
                "    # Discover and merge custom tests from /tmp/tests_custom/",
                "    custom_dir = Path('/tmp/tests_custom')",
                "    if custom_dir.exists():",
                "        for test_file in custom_dir.glob('*.py'):",
                "            # Add custom test file with full path to spec if not already present",
                "            if str(test_file) not in test_spec:",
                "                test_spec[str(test_file)] = []  # Empty list means run all functions in file",
                "    ",
                "    # Read and increment run counter",
                "    counter_path = Path('/tmp/test_run_counter')",
                "    if counter_path.exists():",
                "        run_number = int(counter_path.read_text().strip()) + 1",
                "    else:",
                "        run_number = 1",
                "    counter_path.write_text(str(run_number))",
                "    ",
            ]
        )

        lines.extend(self._generate_eval_section())
        lines.extend(self._generate_install_section())
        lines.extend(self._generate_test_execution())
        lines.extend(self._generate_results())
        lines.extend(self._generate_footer())

        return lines

    def _generate_eval_section(self) -> list[str]:
        """Generate eval commands section if present."""
        if not self.eval_commands:
            return []

        lines = ["    # Run eval commands for environment setup"]
        for cmd in self.eval_commands:
            lines.append(f"    subprocess.run({repr(cmd)}, shell=True, cwd='/testbed')")
        lines.append("")
        return lines

    def _generate_install_section(self) -> list[str]:
        """Generate conditional install section if present."""
        if not self.install_cmd:
            return []

        return [
            "    # Conditional install (run once only)",
            "    install_marker = Path('/tmp/swebench_install_done')",
            "    if not install_marker.exists():",
            f"        # Run install: {self.install_cmd}",
            "        result = subprocess.run(",
            f"            {repr(self.install_cmd)},",
            "            shell=True,",
            "            cwd='/testbed',",
            "            stdout=subprocess.PIPE,",
            "            stderr=subprocess.PIPE,",
            "        )",
            "        if result.returncode != 0:",
            "            print('Install failed: {0}'.format(result.stderr.decode()),",
            "                  file=sys.stderr)",
            "            return 1",
            "        install_marker.touch()",
            "    ",
        ]

    def _generate_test_execution(self) -> list[str]:
        """Generate test execution logic.

        Two execution modes:

        1. **Suite tests** (files NOT under /tmp/tests_custom): run with the
           repo's own test command.  stdout and stderr are captured per file
           and appended to the log.

        2. **Custom tests** (files under /tmp/tests_custom): each test
           function is run in a separate subprocess via the testbed Python.
           Exit code 0 means PASS; nonzero means FAIL.  stdout and stderr are
           captured per function and appended to the log.
        """
        return [
            "    # Execute tests for each file in spec",
            "    results = {}",
            "    log_lines = []",
            "    custom_dir = Path('/tmp/tests_custom')",
            "    for test_file, test_names in test_spec.items():",
            "        results[test_file] = {}",
            "        is_custom = str(custom_dir) in test_file",
            "        ",
            "        if is_custom:",
            "            # --- Custom tests: run each function in a separate",
            "            #     subprocess using the testbed Python. ---",
            "            test_path = Path(test_file)",
            "            if not test_path.exists():",
            "                results[test_file]['all'] = 'FAIL'",
            "                log_lines.append(",
            "                    '\\n=== [{0}] (file not found) ===\\n'.format(test_file)",
            "                )",
            "                continue",
            "            # Import the module to discover test functions",
            "            import sys as _sys",
            "            _sys.path.insert(0, str(test_path.parent))",
            "            try:",
            "                mod = importlib.import_module(test_path.stem)",
            "            except Exception as e:",
            "                results[test_file]['all'] = 'FAIL'",
            "                log_lines.append(",
            "                    '\\n=== [{0}] (import error: {1}) ===\\n'.format(",
            "                        test_file, e",
            "                    ),",
            "                )",
            "                continue",
            "            # Discover all public (non-dunder) functions in the module",
            "            all_funcs = [",
            "                name for name, obj in inspect.getmembers(mod, inspect.isfunction)",
            "                if not name.startswith('_')",
            "            ]",
            "            # If test_names is non-empty, run only those; otherwise run all",
            "            funcs_to_run = test_names if test_names else all_funcs",
            "            for func_name in funcs_to_run:",
            "                # Run each function in its own subprocess so a",
            "                # crash in one does not affect others.  The",
            "                # subprocess starts with a clean sys.path, so we",
            "                # must add the test file's parent directory before",
            "                # importing the module.",
            "                runner_code = (",
            "                    'import sys; sys.path.insert(0, {0!r}); '",
            "                    'import {1} as m; m.{2}();'",
            "                ).format(",
            "                    str(test_path.parent), test_path.stem, func_name",
            "                )",
            "                result = subprocess.run(",
            "                    ['/opt/miniconda3/envs/testbed/bin/python', '-c', runner_code],",
            "                    stdout=subprocess.PIPE,",
            "                    stderr=subprocess.PIPE,",
            "                    universal_newlines=True,",
            "                    cwd='/testbed',",
            "                )",
            "                status = 'PASS' if result.returncode == 0 else 'FAIL'",
            "                results[test_file][func_name] = status",
            "                log_lines.append(",
            "                    '\\n=== [{0}] {1} ({2}) ==='.format(test_file, func_name, status)",
            "                )",
            "                log_lines.append('--- stdout ---')",
            "                log_lines.append(result.stdout)",
            "                log_lines.append('--- stderr ---')",
            "                log_lines.append(result.stderr)",
            "        else:",
            "            # --- Suite tests: run with the repo's test command. ---",
            f"            base_cmd = {repr(self.test_cmd)}",
            "            cmd = base_cmd + ' ' + test_file",
            "            result = subprocess.run(",
            "                cmd, shell=True, cwd='/testbed',",
            "                stdout=subprocess.PIPE, stderr=subprocess.PIPE,",
            "                universal_newlines=True,",
            "            )",
            "            # Parse results using exit code as heuristic",
            "            if not test_names:",
            "                results[test_file]['all'] = 'PASS' if result.returncode == 0 else 'FAIL'",
            "            else:",
            "                for test_name in test_names:",
            "                    results[test_file][test_name] = 'PASS' if result.returncode == 0 else 'FAIL'",
            "            log_lines.append('\\n=== [{0}] ==='.format(test_file))",
            "            log_lines.append('--- stdout ---')",
            "            log_lines.append(result.stdout)",
            "            log_lines.append('--- stderr ---')",
            "            log_lines.append(result.stderr)",
            "    ",
        ]

    def _generate_results(self) -> list[str]:
        """Generate result writing, delta computation, and summary logic."""
        return [
            "    # Write full results",
            "    results_path = Path('/tmp/test_run_{run_number}.json'.format(run_number=run_number))",
            "    with results_path.open('w') as f:",
            "        json.dump(results, f, indent=2)",
            "    ",
            "    # Write captured stdout/stderr log",
            "    log_path = Path('/tmp/test_run_{run_number}.log'.format(run_number=run_number))",
            "    with log_path.open('w') as f:",
            "        f.write('\\n'.join(log_lines))",
            "    ",
            "    total = sum(len(file_results) for file_results in results.values())",
            "    passed = sum(",
            "        1 for file_results in results.values() for status in file_results.values()",
            "        if status == 'PASS'",
            "    )",
            "    ",
            "    # Select the previous run's results; run 1 has no delta to compute",
            "    previous_path = Path('/tmp/test_run_{0}.json'.format(run_number - 1))",
            "    if not previous_path.exists():",
            "        print('----------------------')",
            "        print('|   Test Run {run_number}     |'.format(run_number=run_number))",
            "        print('----------------------')",
            "        print()",
            "        print('RESULTS:')",
            "        print('  Full: {0}'.format(results_path))",
            "        print('  Log: {0}'.format(log_path))",
            "        print()",
            "        print('TOTALS: {0} tests: {1} PASS, {2} FAIL'.format(",
            "            total, passed, total - passed))",
            "        print('No previous run; results recorded.')",
            "        return 0",
            "    ",
            "    with previous_path.open() as f:",
            "        previous = json.load(f)",
            "    ",
            "    # Compute the delta against the previous run: every test present",
            "    # in both runs records its status transition, tests only in this",
            "    # run are new, and tests only in the previous run are removed.",
            "    transitions = {}",
            "    new_tests = {}",
            "    removed_tests = {}",
            "    for test_file, test_results in results.items():",
            "        if test_file not in previous:",
            "            new_tests[test_file] = dict(test_results)",
            "            continue",
            "        file_transitions = {}",
            "        for test_name, status in test_results.items():",
            "            previous_status = previous[test_file].get(test_name)",
            "            if previous_status is None:",
            "                new_tests.setdefault(test_file, {})[test_name] = status",
            "            else:",
            "                file_transitions[test_name] = {'from': previous_status, 'to': status}",
            "        if file_transitions:",
            "            transitions[test_file] = file_transitions",
            "    for test_file, previous_results in previous.items():",
            "        if test_file not in results:",
            "            removed_tests[test_file] = sorted(previous_results)",
            "            continue",
            "        missing = [",
            "            test_name for test_name in previous_results",
            "            if test_name not in results[test_file]",
            "        ]",
            "        if missing:",
            "            removed_tests[test_file] = sorted(missing)",
            "    ",
            "    # Write delta",
            "    delta = {",
            "        'transitions': transitions,",
            "        'new_tests': new_tests,",
            "        'removed_tests': removed_tests,",
            "    }",
            "    delta_path = Path('/tmp/test_run_delta_{run_number}.json'.format(run_number=run_number))",
            "    with delta_path.open('w') as f:",
            "        json.dump(delta, f, indent=2)",
            "    ",
        ]

    def _generate_footer(self) -> list[str]:
        """Generate the summary printing and script entry point."""
        return [
            "    # Collect tests by change category for the summary",
            "    pass_to_pass = []",
            "    pass_to_fail = []",
            "    fail_to_pass = []",
            "    fail_to_fail = []",
            "    for test_file, file_transitions in transitions.items():",
            "        for test_name, change in file_transitions.items():",
            "            entry = (test_file, test_name, change['from'], change['to'])",
            "            if change == {'from': 'PASS', 'to': 'PASS'}:",
            "                pass_to_pass.append(entry)",
            "            elif change == {'from': 'PASS', 'to': 'FAIL'}:",
            "                pass_to_fail.append(entry)",
            "            elif change == {'from': 'FAIL', 'to': 'PASS'}:",
            "                fail_to_pass.append(entry)",
            "            else:",
            "                fail_to_fail.append(entry)",
            "    new_failing = [",
            "        (test_file, test_name, status)",
            "        for test_file, tests in new_tests.items()",
            "        for test_name, status in tests.items()",
            "        if status == 'FAIL'",
            "    ]",
            "    regression_count = len(pass_to_fail) + len(new_failing)",
            "    ",
            "    # Print human-readable summary",
            "    print('----------------------')",
            "    print('|   Test Run {run_number}     |'.format(run_number=run_number))",
            "    print('----------------------')",
            "    print()",
            "    print('RESULTS:')",
            "    print('  Full: {0}'.format(results_path))",
            "    print('  Changes: {0}'.format(delta_path))",
            "    print('  Log: {0}'.format(log_path))",
            "    print()",
            "    print('CHANGES FROM RUN {0}:'.format(run_number - 1))",
            "    print('  PASS -> PASS: {0}'.format(len(pass_to_pass)))",
            "    print('  PASS -> FAIL: {0}'.format(len(pass_to_fail)))",
            "    print('  FAIL -> PASS: {0}'.format(len(fail_to_pass)))",
            "    print('  FAIL -> FAIL: {0}'.format(len(fail_to_fail)))",
            "    new_pass_count = sum(",
            "        1 for status in (",
            "            status for tests in new_tests.values() for status in tests.values()",
            "        ) if status == 'PASS'",
            "    )",
            "    new_count = sum(len(tests) for tests in new_tests.values())",
            "    removed_count = sum(len(tests) for tests in removed_tests.values())",
            "    print('  NEW TESTS: {0} (PASS: {1}, FAIL: {2})'.format(",
            "              new_count, new_pass_count, len(new_failing)))",
            "    print('  REMOVED TESTS: {0}'.format(removed_count))",
            # The basis is named explicitly: a bare count next to the
            # TOTALS line reads as "the suite has no failures" rather
            # than "nothing got worse since the previous run".
            "    print('  REGRESSIONS (new failures vs run {0}): {1}'.format(",
            "              run_number - 1, regression_count))",
            "    print('TOTALS: {0} tests: {1} PASS, {2} FAIL'.format(",
            "        total, passed, total - passed))",
            "    print()",
            "    for label, entries in (",
            "        ('PASS -> FAIL:', pass_to_fail),",
            "        ('FAIL -> PASS:', fail_to_pass),",
            "    ):",
            "        if entries:",
            "            print(label)",
            "            for test_file, test_name, old_status, new_status in entries:",
            "                print('  [{0}]'.format(test_file))",
            "                print('    {0}: {1} -> {2}'.format(",
            "                    test_name, old_status, new_status))",
            "            print()",
            "    if new_failing:",
            "        print('NEW FAILING TESTS:')",
            "        for test_file, test_name, _status in new_failing:",
            "            print('  [{0}]'.format(test_file))",
            "            print('    {0}'.format(test_name))",
            "        print()",
            "    removed_entries = [",
            "        (test_file, test_name)",
            "        for test_file, test_names in removed_tests.items()",
            "        for test_name in test_names",
            "    ]",
            "    if removed_entries:",
            "        print('REMOVED TESTS:')",
            "        for test_file, test_name in removed_entries:",
            "            print('  [{0}]'.format(test_file))",
            "            print('    {0}'.format(test_name))",
            "        print()",
            "    return 0",
            "",
            'if __name__ == "__main__":',
            "    sys.exit(main())",
        ]
