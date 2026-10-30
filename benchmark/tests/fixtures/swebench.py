#!/usr/bin/env python3
"""Shared test fixture factories for SWE-bench integration.

Provides factory functions that create TestSpecAdapter, EnvBuildContextManager,
and BuildOrchestrator instances with sensible default values for testing.

    Factory Functions:
        make_test_spec_adapter: Create TestSpecAdapter with default or custom values
        create_env_build_context_manager: Create EnvBuildContextManager with default values
        make_build_orchestrator: Create BuildOrchestrator with default values
"""

from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    import docker

    from benchmark.tests.fixtures.docker_stubs import DockerClientStub

from benchmark.predict.agents.bench_agent_wrapper import BenchAgentWrapper
from benchmark.predict.swebench.testspec_adapter import TestSpecAdapter


@contextmanager
def make_test_spec_adapter(
    instance_id: str = "astropy__astropy-13579",
    repo: str = "astropy/astropy",
    version: str = "5.0",
    env_script_list: list[str] | None = None,
    repo_script_list: list[str] | None = None,
    eval_script_list: list[str] | None = None,
    arch: str = "x86_64",
    FAIL_TO_PASS: list[str] | None = None,
    PASS_TO_PASS: list[str] | None = None,
    language: str = "py",
    docker_specs: dict | None = None,
    namespace: str | None = None,
    base_image_tag: str = "latest",
    env_image_tag: str = "latest",
    instance_image_tag: str = "latest",
    histrel_repo_path: Path | None = None,
) -> Iterator[TestSpecAdapter]:
    """Factory context manager to create TestSpecAdapter with sensible defaults.

    Args:

        instance_id: Unique instance identifier
        repo: Repository name
        version: Version string
        env_script_list: Environment setup script commands
        repo_script_list: Repository setup script commands
        eval_script_list: Evaluation script commands
        arch: Architecture
        FAIL_TO_PASS: Tests that should fail
        PASS_TO_PASS: Tests that should pass
        language: Programming language
        docker_specs: Docker specifications
        namespace: Kubernetes namespace
        base_image_tag: Base image tag
        env_image_tag: Environment image tag
        instance_image_tag: Instance image tag
        histrel_repo_path: Path to Histrel repository root

    Yields:
        TestSpecAdapter instance
    """
    with ExitStack() as stack:
        if histrel_repo_path is None:
            histrel_repo_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))

        yield TestSpecAdapter(
            instance_id=instance_id,
            repo=repo,
            version=version,
            env_script_list=env_script_list or [],
            repo_script_list=repo_script_list or [],
            eval_script_list=eval_script_list or [],
            arch=arch,
            FAIL_TO_PASS=FAIL_TO_PASS or [],
            PASS_TO_PASS=PASS_TO_PASS or [],
            language=language,
            docker_specs=docker_specs or {},
            namespace=namespace,
            base_image_tag=base_image_tag,
            env_image_tag=env_image_tag,
            instance_image_tag=instance_image_tag,
            histrel_repo_path=histrel_repo_path,
        )


def create_env_build_context_manager(
    histrel_repo_path: Path | None = None,
):
    """Factory function to create EnvBuildContextManager with sensible defaults.

    Args:
        histrel_repo_path: Path to Histrel repository root

    Returns:
        EnvBuildContextManager instance
    """
    from benchmark.predict.swebench.env_builder import EnvBuildContextManager

    if histrel_repo_path is None:
        tmpdir = tempfile.mkdtemp()
        histrel_repo_path = Path(tmpdir) / "histrel"
        histrel_repo_path.mkdir()

    return EnvBuildContextManager(histrel_repo_path)


@contextmanager
def make_build_orchestrator(
    histrel_repo_path: Path | None = None,
    client: DockerClientStub | None = None,
) -> Iterator:
    """Factory context manager to create BuildOrchestrator with sensible defaults.

    Args:
        histrel_repo_path: Path to Histrel repository root
        client: Docker client to inject (defaults to DockerClientStub)

    Yields:
        BuildOrchestrator instance
    """
    from benchmark.predict.swebench.build_orchestrator import BuildOrchestrator
    from benchmark.tests.fixtures.docker_stubs import DockerClientStub

    with ExitStack() as stack:
        if histrel_repo_path is None:
            histrel_repo_path = Path(stack.enter_context(tempfile.TemporaryDirectory()))

        if client is None:
            client = DockerClientStub()

        yield BuildOrchestrator(cast("docker.DockerClient", client), histrel_repo_path)


@contextmanager
def make_bench_agent_wrapper(
    config_path: Path | None = None,
) -> Iterator[BenchAgentWrapper]:
    """Factory context manager to create BenchAgentWrapper with temporary config.

    Args:
        config_path: Path to bench_settings.toml config file (created if None)

    Yields:
        BenchAgentWrapper instance
    """
    with ExitStack() as stack:
        if config_path is None:
            tmpdir = stack.enter_context(tempfile.TemporaryDirectory())
            config_path = Path(tmpdir) / "histrel.toml"
            config_path.write_text(
                '[histrel]\nmodels = "accounts/fireworks/models/deepseek-v4p1-flash-1017"\n'
            )

        yield BenchAgentWrapper(config_path)


@contextmanager
def make_entrypoint_test_env(
    instance: dict | None = None,
    repo_name: str = "testrepo",
    has_test_runner: bool = False,
) -> Iterator[Path]:
    """Factory context manager to create entrypoint test environment.

    Creates a temporary directory structure with:
    - workspace/instance/instance.json
    - workspace/bench_settings.toml
    - testbed/.git

    Args:
        instance: Instance dict to write to instance.json (created if None)
        repo_name: Repository name for testbed directory
        has_test_runner: Whether to include has_test_runner in instance

    Yields:
        Path to the temporary root directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create default instance if not provided
        if instance is None:
            instance = {
                "instance_id": "astropy__astropy-13579",
                "repo": "astropy/astropy",
                "base_commit": "0df94ff7097961e92fd7812036a24b145bc13ca8",
            }
        if has_test_runner:
            instance["has_test_runner"] = True

        # Create workspace structure
        workspace_dir = tmpdir_path / "workspace"
        instance_dir = workspace_dir / "instance"
        instance_dir.mkdir(parents=True)
        instance_file = instance_dir / "instance.json"
        instance_file.write_text(json.dumps(instance))

        bench_settings_file = workspace_dir / "bench_settings.toml"
        bench_settings_file.write_text("[histrel]\n")

        # Create output directory (needed by entrypoint)
        output_dir = workspace_dir / "output"
        output_dir.mkdir(parents=True)

        # Create testbed structure
        testbed_dir = tmpdir_path / "testbed"
        testbed_dir.mkdir(parents=True)
        (testbed_dir / ".git").mkdir()

        yield tmpdir_path


@contextmanager
def make_image_build_test_env(
    instance_id: str = "astropy__astropy-13579",
    repo: str = "astropy/astropy",
    version: str = "5.0",
    arch: str = "x86_64",
) -> Iterator[tuple[Path, Path]]:
    """Factory context manager for image build testing with real SWE-bench data.

    Creates temporary directories for the Histrel repo and test specs using
    values from the SWE-bench Verified dataset.

    Args:
        instance_id: SWE-bench instance identifier
        repo: Repository name
        version: Version string
        arch: Architecture

    Yields:
        Tuple of (histrel_repo_path, temp_dir)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        histrel_path = tmpdir_path / "histrel"
        histrel_path.mkdir()
        (histrel_path / "src").mkdir()
        (histrel_path / "benchmark").mkdir()

        yield histrel_path, tmpdir_path


@contextmanager
def make_test_runner_execution_env(
    test_cmd: str = "python -m pytest",
    install_cmd: str | None = None,
) -> Iterator[Path]:
    """Factory context manager for test runner execution testing.

    Creates temporary directories simulating SWE-bench container environment
    with test directories, spec files, and test infrastructure.

    Args:
        test_cmd: Test command from MAP_REPO_VERSION_TO_SPECS
        install_cmd: Optional install command

    Yields:
        Path to temporary root directory
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        testbed_dir = tmpdir_path / "testbed"
        testbed_dir.mkdir(parents=True)

        (tmpdir_path / "tests_custom").mkdir()

        spec_file = tmpdir_path / "test_spec.json"
        spec_file.write_text('{"test_file.py": []}')

        yield tmpdir_path
