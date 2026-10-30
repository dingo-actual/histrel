#!/usr/bin/env python3
"""TestSpec adapter for SWE-bench integration with Histrel requirements.

Extends SWE-bench's TestSpec dataclass with Histrel-specific functionality
including agent environment injection, test runner generation, and dependency
hashing for cache invalidation.

    Important Classes:
        TestSpecAdapter: Extended TestSpec with Histrel benchmark requirements
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from loguru import logger

from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS
from swebench.harness.test_spec.test_spec import TestSpec

from benchmark.predict.swebench.test_runner_generators import (
    TestRunnerGenerator,
)


def _repo_version_specs(repo: str, version: str) -> dict[str, Any]:
    """Return the SWE-bench spec mapping for a repo version.

    Args:
        repo (str): Repository name, e.g. "astropy/astropy".
        version (str): Repository version.

    Returns:
        (dict[str, Any]) Spec mapping, or an empty dict when absent.
    """
    try:
        versions = MAP_REPO_VERSION_TO_SPECS[repo]
    except KeyError:
        return {}
    if not isinstance(versions, dict):
        return {}
    specs = versions.get(version)
    return specs if isinstance(specs, dict) else {}


class TestSpecAdapter(TestSpec):
    """Extended TestSpec for Histrel benchmark integration.

    Inherits from SWE-bench's TestSpec and adds:
    - has_test_runner: bool attribute for prompt construction
    - Agent environment injection methods
    - Test runner generation
    - Configuration file placement
    - Compute hash methods for env_image_key cache invalidation
    """

    __test__ = False  # prevent pytest from trying to collect this as a test class

    has_test_runner: bool = False
    repo_script_list: list[str]

    def __init__(
        self,
        instance_id: str,
        repo: str,
        version: str,
        env_script_list: list[str],
        repo_script_list: list[str],
        eval_script_list: list[str],
        arch: str,
        FAIL_TO_PASS: list[str],
        PASS_TO_PASS: list[str],
        language: str,
        docker_specs: dict,
        namespace: str | None,
        histrel_repo_path: Path,
        base_image_tag: str = "latest",
        env_image_tag: str = "latest",
        instance_image_tag: str = "latest",
    ) -> None:
        """Initialize TestSpecAdapter with required TestSpec fields.

        Args:
            instance_id: Unique instance identifier
            repo: Repository name (e.g., "test/repo")
            version: Version string
            env_script_list: Environment setup script commands
            repo_script_list: Repository setup script commands
            eval_script_list: Evaluation script commands
            arch: Architecture (e.g., "x86_64")
            FAIL_TO_PASS: Tests that should fail
            PASS_TO_PASS: Tests that should pass
            language: Programming language
            docker_specs: Docker specifications
            namespace: Kubernetes namespace
            base_image_tag: Base image tag
            env_image_tag: Environment image tag
            instance_image_tag: Instance image tag
            histrel_repo_path: Path to Histrel repository root
        """
        super().__init__(
            instance_id=instance_id,
            repo=repo,
            version=version,
            env_script_list=env_script_list,
            repo_script_list=repo_script_list,
            eval_script_list=eval_script_list,
            arch=arch,
            FAIL_TO_PASS=FAIL_TO_PASS,
            PASS_TO_PASS=PASS_TO_PASS,
            language=language,
            docker_specs=docker_specs,
            namespace=namespace,
            base_image_tag=base_image_tag,
            env_image_tag=env_image_tag,
            instance_image_tag=instance_image_tag,
        )
        self.histrel_repo_path = histrel_repo_path

        # Assume test runner is known until proven otherwise (defensive)
        self.has_test_runner = False

    @classmethod
    def from_instance(
        cls,
        instance: dict,
        histrel_repo_path: Path,
    ) -> TestSpecAdapter:
        """Create TestSpecAdapter from SWE-bench instance dict.

        Args:
            instance: SWE-bench instance dictionary containing repo, version, etc.

        Returns:
            TestSpecAdapter with all fields copied from base TestSpec
        """
        from swebench.harness.test_spec.test_spec import make_test_spec

        logger.bind(
            component="benchmark.predict.swebench.testspec_adapter",
            instance_id=instance.get("instance_id", "unknown"),
            repo=instance.get("repo", "unknown"),
            origin="bench_harness_predict",
        ).info("Creating TestSpecAdapter from instance")

        spec = make_test_spec(instance)  # type: ignore[arg-type]

        adapter = cls(
            instance_id=spec.instance_id,
            repo=spec.repo,
            version=spec.version,
            env_script_list=list(spec.env_script_list),
            repo_script_list=list(spec.repo_script_list),
            eval_script_list=list(spec.eval_script_list),
            arch=spec.arch,
            FAIL_TO_PASS=list(spec.FAIL_TO_PASS),
            PASS_TO_PASS=list(spec.PASS_TO_PASS),
            language=spec.language,
            docker_specs=dict(spec.docker_specs) if spec.docker_specs else {},
            namespace=spec.namespace,
            base_image_tag=spec.base_image_tag,
            env_image_tag=spec.env_image_tag,
            instance_image_tag=spec.instance_image_tag,
            histrel_repo_path=histrel_repo_path,
        )

        return adapter

    def harden_git_operations(self) -> TestSpecAdapter:
        """Harden fragile git operations in repo_script_list.

        SWE-bench's ``make_repo_script_list_py`` generates a ``git clone``
        followed by ``git gc --prune=now --aggressive``.  Cloning over HTTP
        can silently produce corrupt pack files, and ``git gc --aggressive``
        fails fatally (exit 128) when it encounters a corrupt object.

        This method replaces the fragile clone with a verified clone that
        re-clones on ``git fsck`` failure, and replaces ``git gc --aggressive``
        with a non-fatal variant so a corrupt object does not abort the
        build.

        Returns:
            Self with hardened git commands in repo_script_list
        """
        if not self.repo_script_list:
            return self

        hardened: list[str] = []
        for cmd in self.repo_script_list:
            if cmd.startswith("git clone -o origin"):
                hardened.append(self._harden_clone_command(cmd))
            elif cmd == "git gc --prune=now --aggressive":
                hardened.append("git gc --prune=now --aggressive || true")
            else:
                hardened.append(cmd)

        self.repo_script_list = hardened

        logger.bind(
            component="benchmark.predict.swebench.testspec_adapter",
            instance_id=self.instance_id,
            origin="bench_harness_predict",
        ).info("Hardened git operations in repo_script_list")
        return self

    @staticmethod
    def _harden_clone_command(clone_cmd: str) -> str:
        """Wrap a git clone command with fsck verification and re-clone retry.

        Some upstream repositories (for example, psf/requests) contain historical
        commits with non-standard time zones. ``git fsck --full`` treats these
        as fatal ``badTimezone`` errors even though the repository is otherwise
        intact. This wrapper ignores that specific message while still catching
        real object corruption.

        The loop runs from a known safe working directory (``/``) so that
        ``rm -rf <target>`` cannot delete the shell's own CWD. ``git clone``
        itself reads the CWD of the calling process, so removing ``/testbed``
        while CWD is ``/testbed`` causes ``fatal: Unable to read current
        working directory``.

        Args:
            clone_cmd: The original ``git clone`` command string.

        Returns:
            A bash snippet that clones, verifies with ``git fsck``, and
            re-clones up to three times on corruption.
        """
        target = clone_cmd.split()[-1]
        return (
            f"for _clone_attempt in 1 2 3; do\n"
            f"    cd /\n"
            f"    rm -rf {target}\n"
            f"    if {clone_cmd} && "
            f"cd {target} && git -c fsck.badTimezone=ignore fsck --full; then\n"
            f"        cd /\n"
            f"        break\n"
            f"    fi\n"
            f"    if [ $_clone_attempt -eq 3 ]; then exit 128; fi\n"
            f"    sleep 2\n"
            f"done"
        )

    def _get_agent_env_commands(self) -> list[str]:
        """Get list of commands to set up agent environment.

        Returns:
            List of shell commands for tools and conda setup
        """
        return [
            "apt-get update",
            "apt-get install -y curl ca-certificates libmagic1 unzip",
            "curl -fsSL -o /tmp/rg.deb https://github.com/BurntSushi/ripgrep/releases/download/15.1.0/ripgrep_15.1.0-1_amd64.deb",
            "dpkg -i /tmp/rg.deb",
            "rm /tmp/rg.deb",
            "curl -fsSL -o /tmp/fd.deb https://github.com/sharkdp/fd/releases/download/v10.3.0/fd_10.3.0_amd64.deb",
            "dpkg -i /tmp/fd.deb",
            "rm /tmp/fd.deb",
            "curl -fsSL -o /tmp/tokei.tar.gz https://github.com/XAMPPRocky/tokei/releases/download/v12.0.0/tokei-x86_64-unknown-linux-gnu.tar.gz",
            "tar -xzf /tmp/tokei.tar.gz -C /usr/local/bin tokei",
            "chmod +x /usr/local/bin/tokei",
            "rm /tmp/tokei.tar.gz",
            "source /opt/miniconda3/bin/activate",
            "conda create -y -n agent python=3.11.12",
            "conda run -n agent python -m pip install /root/histrel",
        ]

    def inject_agent_env(self) -> TestSpecAdapter:
        """Inject agent runtime dependencies into env_script_list.

        Adds commands to install system tools and create the conda agent
        environment for benchmark agent execution.

        Returns:
            Self with agent environment commands appended to env_script_list
        """
        logger.bind(
            component="benchmark.predict.swebench.testspec_adapter",
            instance_id=self.instance_id,
            language=self.language,
            origin="bench_harness_predict",
        ).info("Injecting agent environment dependencies")

        if self.language != "py":
            logger.bind(
                component="benchmark.predict.swebench.testspec_adapter",
                instance_id=self.instance_id,
                origin="bench_harness_predict",
            ).debug("Skipping non-Python instance")
            return self

        commands = self._get_agent_env_commands()
        self.env_script_list.extend(commands)

        histrel_hash = self.compute_histrel_hash()
        self.env_script_list.append(f"# histrel_hash: {histrel_hash}")

        logger.bind(
            component="benchmark.predict.swebench.testspec_adapter",
            instance_id=self.instance_id,
            commands_added=len(commands),
            origin="bench_harness_predict",
        ).info("Agent environment dependencies injected successfully")
        return self

    def inject_repo_configs(self) -> TestSpecAdapter:
        """Inject configuration file placement into repo_script_list.

        Copies bench_settings.toml to both repo root and benchmark/ subdirectory.

        Returns:
            Self with config placement commands appended to repo_script_list
        """
        logger.bind(
            component="benchmark.predict.swebench.testspec_adapter",
            instance_id=self.instance_id,
            origin="bench_harness_predict",
        ).info("Injecting configuration file placements")

        # SWE-bench clones the repo directly to /testbed/, not /testbed/{repo_name}/
        # So we place configs at /testbed/ directly
        config_commands = [
            "cp /root/bench_settings.toml /testbed/bench_settings.toml",
            "mkdir -p /testbed/benchmark",
            "cp /root/bench_settings.toml /testbed/benchmark/bench_settings.toml",
        ]

        self.repo_script_list.extend(config_commands)

        logger.bind(
            component="benchmark.predict.swebench.testspec_adapter",
            instance_id=self.instance_id,
            config_commands=len(config_commands),
            origin="bench_harness_predict",
        ).info("Configuration file placements injected")
        return self

    def inject_test_runners(self) -> tuple[TestSpecAdapter, bool]:
        """Inject the test runner script if test_cmd is available and framework is known.

        Generates the single run_tests script conditionally based on test_cmd
        availability and framework detection.

        Returns:
            Tuple of (self, has_test_runner)
        """
        test_cmd, _install_cmd, _eval_commands = self._extract_test_specs()

        # Definitive condition: framework detection must succeed
        framework = self._detect_test_framework(test_cmd)
        if framework is None:
            logger.bind(
                component="benchmark.predict.swebench.testspec_adapter",
                instance_id=self.instance_id,
                test_cmd=test_cmd if test_cmd else "not_found",
                origin="bench_harness_predict",
            ).info("No known test framework detected, skipping test runner generation")
            self.has_test_runner = False
        else:
            logger.bind(
                component="benchmark.predict.swebench.testspec_adapter",
                instance_id=self.instance_id,
                framework=framework,
                origin="bench_harness_predict",
            ).info("Test framework detected, generating runner scripts")
            self.has_test_runner = True

            commands: list[str] = []

            commands.extend(self._create_test_directories())
            commands.extend(self._create_test_runner_scripts())
            commands.extend(self._create_test_documentation())

            self.repo_script_list.extend(commands)

            logger.bind(
                component="benchmark.predict.swebench.testspec_adapter",
                instance_id=self.instance_id,
                commands_added=len(commands),
                framework=framework,
                origin="bench_harness_predict",
            ).info("Test runner scripts injected successfully")
        return self, self.has_test_runner

    def _extract_test_specs(self) -> tuple[str, str | None, list[str]]:
        """Extract test command, install command, and eval commands from specs.

        Returns:
            Tuple of (test_cmd, install_cmd, eval_commands)
        """
        try:
            specs = _repo_version_specs(self.repo, self.version)
            test_cmd = specs["test_cmd"]
            install_cmd = specs.get("install")
            eval_commands = specs.get("eval_commands", [])
            return test_cmd, install_cmd, eval_commands
        except (KeyError, AttributeError):
            return "", None, []

    def _detect_test_framework(self, test_cmd: str) -> str | None:
        """Detect test framework from command pattern.

        Args:
            test_cmd: The test command string from specs

        Returns:
            Framework type: 'pytest', 'django', 'sympy', 'tox', 'astropy', 'seaborn',
            or None if no pattern matches
        """
        # Order matters - check more specific patterns first
        if "./tests/runtests.py" in test_cmd:
            return "django"
        elif "bin/test" in test_cmd:
            return "sympy"
        elif "tox" in test_cmd:
            return "tox"
        elif "-o console_output_style=classic" in test_cmd:
            return "astropy"
        elif "--no-header" in test_cmd:
            return "seaborn"
        elif "pytest" in test_cmd:
            return "pytest"
        else:
            return None

    def _test_command_available(self) -> bool:
        """Check if test command is available for this instance.

        Returns:
            True if test_cmd exists and is non-empty, False otherwise
        """
        specs = _repo_version_specs(self.repo, self.version)
        test_cmd = specs.get("test_cmd")
        if not isinstance(test_cmd, str):
            return False
        return test_cmd.strip() != ""

    def _create_test_directories(self) -> list[str]:
        """Create test directories and placeholder test spec.

        Returns:
            List of commands to create test directories and files
        """
        return [
            "mkdir -p /tmp/tests_custom",
            "chmod 777 /tmp/tests_custom",
            "chmod 777 /tmp",
        ]

    def _create_test_runner_scripts(self) -> list[str]:
        """Generate the test runner Python script.

        Returns:
            List of commands to create the executable test runner script
        """
        test_cmd, install_cmd, eval_commands = self._extract_test_specs()
        generator = TestRunnerGenerator(test_cmd, install_cmd, eval_commands)

        script = generator.generate_runner()

        return [
            "cat > /usr/local/bin/run_tests << 'EOF_SCRIPT'",
            script,
            "EOF_SCRIPT",
            "chmod +x /usr/local/bin/run_tests",
        ]

    def _create_test_documentation(self) -> list[str]:
        """Create testing documentation file and empty spec files.

        Returns:
            List of commands to create testing README and empty spec files
        """
        readme_content = """# Testing Infrastructure Guide

This environment includes a test runner script at `/usr/local/bin/run_tests`:
 - Runs all tests specified in `/tmp/test_spec.json` and records the results
 - Takes no arguments
 - Every run writes full results to `/tmp/test_run_N.json` (N is an
   auto-incrementing run counter; run 1 establishes the baseline)
 - From run 2 on, also writes `/tmp/test_run_delta_N.json` with the changes
   from run N-1
 - Every run writes captured stdout and stderr per test to
   `/tmp/test_run_N.log`
 - Prints a summary of the run's results to stdout, including the paths of
   all written files (results JSON, delta JSON, and log file)

## Test Specification Format

Specified tests can include those in the existing test suite, as well
as custom tests (see Test Directories).

```json
{
    "test_file_path_01": ["test_name_01", "test_name_02", ...],
    "test_file_path_02": ["test_name_nn", "test_name_mm", ...],
    ...
}
```

For custom tests (files under `/tmp/tests_custom`), the function names in
the list refer to functions defined in the file.  An empty list means run
all public (non-dunder) functions in the file.

## Test Result Format

`/tmp/test_run_N.json`:

```json
{
    "test_file_path_01": {
        "test_name_01": "PASS|FAIL|ERROR",
        "test_name_02": "PASS|FAIL|ERROR",
        ...
    },
    "test_file_path_02": {
        "test_name_nn": "PASS|FAIL|ERROR",
        "test_name_mm": "PASS|FAIL|ERROR",
        ...
    },
    ...
}
```

## Test Log Format

`/tmp/test_run_N.log` contains the captured stdout and stderr for each
test, organized as sections:

```
=== [test_file_path_01] ===
--- stdout ---
<captured stdout for the test file>
--- stderr ---
<captured stderr for the test file>

=== [/tmp/tests_custom/test_foo.py] my_func (PASS) ===
--- stdout ---
<captured stdout for the function>
--- stderr ---
<captured stderr for the function>
```

Each section is delimited by `=== [file_path] ===` for suite tests, or
`=== [file_path] function_name (status) ===` for custom tests.

## Test Delta Format

`/tmp/test_run_delta_N.json` (written from run 2 on) compares run N against
run N-1:

```json
{
    "transitions": {
        "test_file_path_01": {
            "test_name_01": {"from": "PASS", "to": "PASS"},
            "test_name_02": {"from": "PASS", "to": "FAIL"},
            "test_name_03": {"from": "FAIL", "to": "PASS"},
            "test_name_04": {"from": "FAIL", "to": "FAIL"}
        }
    },
    "new_tests": {
        "test_file_path_02": {
            "test_name_nn": "PASS|FAIL"
        }
    },
    "removed_tests": {
        "test_file_path_01": ["test_name_gone"]
    }
}
```

- `transitions`: every test present in both runs, with its previous and
  current status (including unchanged tests)
- `new_tests`: tests present in run N but not run N-1, with their status
- `removed_tests`: tests present in run N-1 but not run N

## Run Summary

`run_tests` prints a summary of each run's results to stdout, including the
written file paths (results JSON, delta JSON, and log file), counts for every
change category (PASS -> PASS, PASS -> FAIL, FAIL -> PASS, FAIL -> FAIL,
NEW TESTS, REMOVED TESTS), a REGRESSIONS count (tests now failing that passed
in the previous run), and per-test details for PASS -> FAIL, FAIL -> PASS,
new failing, and removed tests.

## Directories

The testing tool uses the following directory structure:

- `/tmp/tests_custom`: Place all tests for solution correctness here
  (*not* tests for regressions; specify existing tests for that via
  the test spec JSON file)
  - Each function specified in the test spec (or all public functions
    if the spec lists is empty) is run in a separate subprocess using
    the testbed Python; exit code 0 means PASS, nonzero means FAIL
  - Executed via `run_tests`

Custom tests placed in `/tmp/tests_custom` are automatically discovered by
`run_tests`, as are tests in the existing test suite.

Custom tests are executed differently from suite tests:

- **Suite tests** (in the codebase's test suite) are run with the repo's
  own test command (the instance's built-in test harness).  stdout and
  stderr are captured per test file.
- **Custom tests** (under `/tmp/tests_custom`) are run function-by-function
  in separate subprocesses, using the exit code to determine pass or fail.
  stdout and stderr are captured per function.

## Workflows

### Custom Tests

1. Write custom tests to `/tmp/tests_custom` that check for solution correctness
   - Specify each test as a function that returns 0 or `None` on success,
     and either returns nonzero (and non-null) or throws an exception on
     failure
2. Find existing tests in the test suite to check for regressions
3. Add all tests to the test spec JSON at `/tmp/test_spec.json`
4. Call `run_tests` (run 1 establishes the baseline)
5. After making changes to the codebase, call `run_tests` again; the delta
   against the previous run reports regressions (PASS -> FAIL) and
   improvements (FAIL -> PASS)
6. Read `/tmp/test_run_N.log` to see the captured stdout and stderr for
   each test when diagnosing failures

"""
        return [
            "cat > /tmp/TESTING_README.txt << 'EOF_README'",
            readme_content,
            "EOF_README",
            "cat > /tmp/test_spec.json << 'EOF_SPEC'",
            "{}",
            "EOF_SPEC",
        ]

    def compute_histrel_hash(self) -> str:
        """Compute hash of Histrel dependencies for env image key.

        Returns:
            SHA256 hash (first 22 chars) of all Histrel dependencies
        """
        hasher = hashlib.sha256()

        if self.histrel_repo_path is None:
            return hasher.hexdigest()[:22]

        src_dir = self.histrel_repo_path / "src"
        if src_dir.exists():
            for py_file in sorted(src_dir.rglob("*.py")):
                if "__pycache__" not in py_file.parts:
                    hasher.update(py_file.read_bytes())

        benchmark_dir = self.histrel_repo_path / "benchmark"
        if benchmark_dir.exists():
            for py_file in sorted(benchmark_dir.rglob("*.py")):
                if "vendor" not in py_file.parts and "__pycache__" not in py_file.parts:
                    hasher.update(py_file.read_bytes())

        toml_path = self.histrel_repo_path / "benchmark/bench_settings.toml"
        if toml_path.exists():
            hasher.update(toml_path.read_bytes())

        # The env image pip-installs the repo, so packaging metadata and the
        # root-level agent config template shape what lands in the image.
        for extra_path in (
            self.histrel_repo_path / "pyproject.toml",
            self.histrel_repo_path / "histrel.toml",
        ):
            if extra_path.exists():
                hasher.update(extra_path.read_bytes())

        return hasher.hexdigest()[:22]
