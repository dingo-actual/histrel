#!/usr/bin/env python3
"""Integration tests for benchmark harness use of the benchmark agent.

Tests the full integration chain: harness config -> BenchAgentWrapper ->
agent_config.toml -> agent CLI -> patch extraction -> usage data.

Non-live tests use patched subprocess execution and temporary filesystem
structures. Live tests (marked ``requires_live_llm``) make real API calls
to Fireworks AI and skip unless ``--llm`` is provided.

Test taxonomy coverage:
    1. Contract Conformance — wrapper interface, entrypoint contract
    2. Boundary Traversal — config propagation, patch/usage handoff
    3. Configuration Coherence — model from bench_settings.toml reaches agent
    4. Failure Propagation — missing files, invalid instances, error exit
    5. Emergent Behavior — full wrapper-to-agent-to-patch round trip
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.parse import urlparse

import pytest

if sys.platform == "win32":
    pytest.skip("SWE-bench requires Unix", allow_module_level=True)

from benchmark.predict.agents import BenchAgentWrapper
from benchmark.predict.config import PredictorConfig, SWEbenchConfig
from benchmark.predict.docker.runner import DockerRunner
from benchmark.predict.types import AGENT_TYPE
from benchmark.predict.docker.entrypoint_bench_agent import main as entrypoint_main
from histrel.core.config import (
    AgentConfig,
    Budgets,
    load_config,
    SystemPrompts,
)
from histrel.core.runtime import Runtime, run_agent
from histrel.model.adapter_openai_compat import OpenAICompatAdapter
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
    NormalizedToolCall,
    ProviderMessage,
    Usage,
)
from histrel.terminal import SessionStatus
from benchmark.tests.fixtures.docker_stubs import DockerClientStub
from dulwich.porcelain import add, commit
from dulwich.repo import Repo

# User request message used across integration tests.
_USER_MSG = "Fix the failing test."


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FIREWORKS_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash-1017"
_FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# Per-request bound for live calls. The production bound is 600s; a stalled
# proxy would otherwise block the test suite for many minutes.
_LIVE_TIMEOUT_SECONDS = 60.0

_MINIMAL_TOML = (
    '[histrel]\nmodels = "accounts/fireworks/models/deepseek-v4p1-flash-1017"\n'
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _env(name: str) -> str | None:
    val = os.getenv(name)
    return val if val and val.strip() else None


def _server_listening(url: str, *, timeout: float = 2.0) -> bool:
    """Report whether a TCP connection to ``url`` succeeds.

    Egress may go through an HTTP proxy, so the proxy endpoint is probed
    as a fallback when the direct connection fails.
    """
    import contextlib

    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    targets = [(host, port)]
    proxy = urlparse(os.getenv("https_proxy") or os.getenv("HTTPS_PROXY") or "")
    if proxy.hostname:
        targets.append((proxy.hostname, proxy.port or 3128))
    for target in targets:
        with (
            contextlib.suppress(Exception),
            socket.create_connection(target, timeout=timeout),
        ):
            return True
    return False


def _require_live_fireworks() -> str:
    """Return the Fireworks API key, skipping when live calls are unavailable."""
    api_key = _env("FIREWORKS_API_KEY")
    if not api_key:
        pytest.skip("FIREWORKS_API_KEY not set")
    if not _server_listening(_FIREWORKS_BASE_URL):
        pytest.skip("Fireworks API unreachable")
    return api_key


def _init_fixture_repo(path: Path) -> None:
    """Initialize a tiny fixture repository with a defect."""
    Repo.init(str(path))

    (path / "calculator.py").write_text(
        '"""Simple calculator module with a bug."""\n'
        "\n"
        "\n"
        "def add(a: int, b: int) -> int:\n"
        '    """Add two numbers. BUG: returns a - b instead of a + b."""\n'
        "    return a - b\n"
        "\n"
        "\n"
        "def multiply(a: int, b: int) -> int:\n"
        '    """Multiply two numbers (correct)."""\n'
        "    return a * b\n",
        encoding="utf-8",
    )

    (path / "test_calculator.py").write_text(
        '"""Failing test for the calculator module."""\n'
        "\n"
        "from calculator import add, multiply\n"
        "\n"
        "\n"
        "def test_add():\n"
        '    """add(2, 3) should return 5, but the bug returns -1."""\n'
        "    assert add(2, 3) == 5\n"
        "\n"
        "\n"
        "def test_multiply():\n"
        '    """Adjacent behavior: multiply is correct."""\n'
        "    assert multiply(2, 3) == 6\n",
        encoding="utf-8",
    )

    add(str(path), [b"calculator.py", b"test_calculator.py"])
    commit(
        str(path),
        message=b"initial: calculator with add bug",
        author=b"Test <test@test.com>",
        committer=b"Test <test@test.com>",
    )


def _make_agent_config(
    max_turns: int = 0,
    max_tool_calls: int = 0,
    model_name: str = _FIREWORKS_MODEL,
) -> AgentConfig:
    """Build a minimal AgentConfig for integration tests."""
    return AgentConfig(
        schema_version=1,
        session_id="integration-test-session",
        task_id="fixture-calculator-bug",
        agent_system_prompt=SystemPrompts(
            initialize=(
                "You are a coding agent. Fix the bug in the repository so all tests pass."
            ),
            step=(
                "You are a coding agent. Fix the bug in the repository so all tests pass."
            ),
        ),
        model_provider="fireworks",  # type: ignore[arg-type]
        model_name=model_name,
        model_temperature=0.0,
        model_top_p=1.0,
        model_reasoning_effort=None,
        model_context_limit_tokens=128000,
        model_max_tokens_output=16384,
        model_api_key_env="FIREWORKS_API_KEY",
        budgets=Budgets(
            max_wall_seconds=0,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
        ),
    )


def _make_fake_response(
    text: str = "Working on it.",
    tool_calls: tuple[NormalizedToolCall, ...] = (),
) -> ModelResponse:
    """Create a fake model response for patching."""
    return ModelResponse(
        text_parts=(text,) if text else (),
        tool_calls=tool_calls,
        usage=Usage(input=1000, output=500, total=1500),
        provider_request_id="req-fake",
        finish_reason="stop",
    )


def _make_entrypoint_env(
    tmpdir: Path,
    instance: dict | None = None,
    agent_type: str = "bench-agent",
    run_name: str = "test-run",
    has_test_runner: bool = False,
    timeout_seconds: int = 7200,
) -> dict[str, str]:
    """Create environment variables for entrypoint tests."""
    env = {
        "RUN_NAME": run_name,
        "AGENT_TYPE": agent_type,
        "INSTANCE_ID": instance.get("instance_id", "test-id")
        if instance
        else "test-id",
        "SESSION_ID": "-".join(
            s.replace("_", "-")
            for s in [
                run_name,
                agent_type,
                instance.get("instance_id", "test-id") if instance else "test-id",
            ]
        ),
        "TIMEOUT_SECONDS": str(timeout_seconds),
        "HAS_TEST_RUNNER": "true" if has_test_runner else "false",
    }
    if agent_type == "bench-agent-no-mt":
        env["DISABLE_HISTREL"] = "true"
    return env


# ===========================================================================
# 1. Contract Conformance Tests
# ===========================================================================


class Test04Integration01BenchAgent01ContractConformance:
    """Verify that components honor their declared interfaces when composed."""

    def test_04_int_01_bench_agent_01_contract_01_wrapper_has_generate_patch(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "bench_settings.toml"
        config_path.write_text(_MINIMAL_TOML)

        # Act
        wrapper = BenchAgentWrapper(config_path)

        # Assert — interface contract: generate_patch exists and is callable
        assert callable(getattr(wrapper, "generate_patch", None))
        assert callable(getattr(wrapper, "_write_agent_config", None))
        assert callable(getattr(wrapper, "_load_usage_data", None))

    def test_04_int_01_bench_agent_01_contract_02_wrapper_init_signature(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "bench_settings.toml"
        config_path.write_text(_MINIMAL_TOML)

        # Act
        wrapper = BenchAgentWrapper(config_path)

        # Assert
        assert wrapper.config_path == config_path

    def test_04_int_01_bench_agent_01_contract_03_entrypoint_returns_int(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — entrypoint should return an int exit code
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            workspace_dir = tmpdir_path / "workspace"
            instance_dir = workspace_dir / "instance"
            instance_dir.mkdir(parents=True)
            (instance_dir / "instance.json").write_text(
                json.dumps(
                    {"instance_id": "test-1", "repo": "test/repo", "base_commit": "abc"}
                )
            )
            (workspace_dir / "bench_settings.toml").write_text(_MINIMAL_TOML)
            (workspace_dir / "output").mkdir()

            testbed_dir = tmpdir_path / "testbed"
            testbed_dir.mkdir(parents=True)
            (testbed_dir / ".git").mkdir()

            def mock_path(path_str):
                if isinstance(path_str, str):
                    if path_str.startswith("/workspace"):
                        return workspace_dir / path_str.replace("/workspace/", "")
                    elif path_str == "/testbed":
                        return testbed_dir
                return Path(path_str)

            with (
                patch(
                    "benchmark.predict.docker.entrypoint_bench_agent.Path",
                    side_effect=mock_path,
                ),
                patch(
                    "benchmark.predict.docker.entrypoint_bench_agent.BenchAgentWrapper"
                ) as mock_w,
            ):
                mock_w.return_value.generate_patch.return_value = ("patch", {})
                monkeypatch.setenv("AGENT_TYPE", "bench-agent")
                monkeypatch.setenv("RUN_NAME", "test")
                monkeypatch.setenv("SESSION_ID", "test-bench-agent-test-1")

                # Act
                result = entrypoint_main()

                # Assert — returns int, not None or other type
                assert isinstance(result, int)
                assert result == 0

    def test_04_int_01_bench_agent_01_contract_04_runner_accepts_bench_agent_types(
        self, docker_client_stub: DockerClientStub, tmp_path: Path
    ) -> None:
        # Arrange
        cfg = PredictorConfig(
            output_dir=str(tmp_path / "results"),
            run_name="bench-contract",
        )
        runner = DockerRunner(docker_client_stub)
        instance = {"instance_id": "test-1", "repo": "test/repo"}
        out_dir = Path(cfg.output_dir) / cfg.run_name / "artifacts"

        for _ in range(1):
            # Act — should not raise
            patch_path, _, exit_code = runner.run_agent(
                instance,
                image="test:latest",
                config=cfg,
                output_dir=out_dir,
                timeout=5,
            )

            # Assert
            assert patch_path.name == "bench-agent_test-1.jsonl"

    def test_04_int_01_bench_agent_01_contract_05_config_accepts_bench_agent_types(
        self,
    ) -> None:
        # Act
        cfg = SWEbenchConfig()

        # Assert
        assert isinstance(cfg.tools, list) and cfg.tools

    def test_04_int_01_bench_agent_01_contract_06_cli_has_no_agent_flag(self) -> None:
        # Arrange
        from benchmark.predict.cli import build_parser

        parser = build_parser()

        # Act / Assert — the single agent type is implicit, no flag needed
        options = {action.dest for action in parser._actions}
        assert "agents" not in options
        args = parser.parse_args([])
        assert not hasattr(args, "agents")


# ===========================================================================
# 2. Boundary Traversal Tests
# ===========================================================================


class Test04Integration02BenchAgent02BoundaryTraversal:
    """Verify correct behavior as data crosses component boundaries."""

    def test_04_int_02_bench_agent_02_boundary_01_config_propagates_condition(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "bench_settings.toml"
        config_path.write_text(_MINIMAL_TOML)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        # Act
        wrapper = BenchAgentWrapper(config_path)
        agent_config_path = wrapper._write_agent_config(repo_dir)

        # Assert — the written agent config is valid for the agent CLI
        content = tomllib.loads(agent_config_path.read_text(encoding="utf-8"))
        assert content["agent_system_prompt"]["initialize"]
        assert content["agent_system_prompt"]["step"]

    def test_04_int_02_bench_agent_02_boundary_02_patch_flows_from_git_to_wrapper(
        self, tmp_path: Path
    ) -> None:
        # Arrange — create a repo with changes
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_fixture_repo(repo_dir)

        # Make a change (fix the bug)
        (repo_dir / "calculator.py").write_text(
            "def add(a, b):\n    return a + b\n", encoding="utf-8"
        )

        config_path = tmp_path / "bench_settings.toml"
        config_path.write_text(_MINIMAL_TOML)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fake_result = MagicMock(returncode=0, stdout="", stderr="")
        real_subprocess_run = subprocess.run

        def selective_run(cmd, **kwargs):
            # Only mock the agent execution (bash -c), let git commands through
            if isinstance(cmd, list) and cmd and cmd[0] == "bash":
                return fake_result
            return real_subprocess_run(cmd, **kwargs)

        with (
            patch.object(
                BenchAgentWrapper,
                "_load_usage_data",
                return_value={"total": {"tokens_total": 0}},
            ),
            patch("subprocess.run", side_effect=selective_run),
            patch("benchmark.predict.agents.bench_agent_wrapper.Path") as mock_path,
        ):
            mock_path.return_value = output_dir
            wrapper = BenchAgentWrapper(config_path)

            # Act
            patch_text, _ = wrapper.generate_patch(
                instance={"instance_id": "test-1"},
                repo_dir=repo_dir,
                session_id="test-session",
            )

            # Assert — patch extracted from git diff
            assert "def add" in patch_text
            assert "a + b" in patch_text

    def test_04_int_02_bench_agent_02_boundary_03_usage_flows_from_agent_to_harness(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — simulate agent artifact directory
        monkeypatch.setenv("SESSION_ID", "test-usage-sess")
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        costs_file = artifact_dir / "costs.json"
        costs_file.write_text(json.dumps({"total_cost": 0.15}))
        calls_file = artifact_dir / "token_usage_calls.jsonl"
        calls_file.write_text(
            json.dumps({"usage": {"input": 5000, "output": 2000, "cached": 100}}) + "\n"
        )

        # Act
        wrapper = BenchAgentWrapper(tmp_path / "bench_settings.toml")
        wrapper.config_path = tmp_path / "bench_settings.toml"
        wrapper.config_path.write_text(_MINIMAL_TOML)
        usage_data = wrapper._load_usage_data(costs_file)

        # Assert — usage converted to harness format
        assert usage_data["total"]["tokens_input"] == 5000
        assert usage_data["total"]["tokens_output"] == 2000
        assert usage_data["total"]["tokens_total"] == 7000
        assert usage_data["total"]["cost_total"] == 0.15
        assert usage_data["total"]["session_id"] == "test-usage-sess"
        # "none" mirrors "total"
        assert usage_data["none"]["tokens_total"] == 7000

    def test_04_int_02_bench_agent_02_boundary_04_env_vars_cross_runner_to_container(
        self, docker_client_stub: DockerClientStub, tmp_path: Path
    ) -> None:
        # Arrange
        cfg = PredictorConfig(
            output_dir=str(tmp_path / "results"),
            run_name="bench-env",
            verbose=True,
        )
        runner = DockerRunner(docker_client_stub)
        instance = {"instance_id": "inst-1", "repo": "test/repo"}
        out_dir = Path(cfg.output_dir) / cfg.run_name / "artifacts"

        # Act
        runner.run_agent(
            instance,
            image="test:latest",
            config=cfg,
            output_dir=out_dir,
            timeout=300,
        )

        # Assert — all expected env vars propagated to container
        env = docker_client_stub.containers.run_calls[0]["environment"]
        assert env["RUN_NAME"] == "bench-env"
        assert env["AGENT_TYPE"] == "bench-agent"
        assert env["INSTANCE_ID"] == "inst-1"
        assert env["SESSION_ID"] == "bench-env-bench-agent-inst-1"
        assert env["TIMEOUT_SECONDS"] == "300"

    def test_04_int_02_bench_agent_02_boundary_05_entrypoint_passes_session_id_to_wrapper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            workspace_dir = tmpdir_path / "workspace"
            instance_dir = workspace_dir / "instance"
            instance_dir.mkdir(parents=True)
            (instance_dir / "instance.json").write_text(
                json.dumps(
                    {
                        "instance_id": "test-sid",
                        "repo": "test/repo",
                        "base_commit": "abc",
                    }
                )
            )
            (workspace_dir / "bench_settings.toml").write_text(_MINIMAL_TOML)
            (workspace_dir / "output").mkdir()

            testbed_dir = tmpdir_path / "testbed"
            testbed_dir.mkdir(parents=True)
            (testbed_dir / ".git").mkdir()

            monkeypatch.setenv("RUN_NAME", "my-run")
            monkeypatch.setenv("AGENT_TYPE", "bench-agent")
            monkeypatch.setenv("SESSION_ID", "my-run-bench-agent-test-sid")
            monkeypatch.setenv("TIMEOUT_SECONDS", "3600")

            def mock_path(path_str):
                if isinstance(path_str, str):
                    if path_str.startswith("/workspace"):
                        return workspace_dir / path_str.replace("/workspace/", "")
                    elif path_str == "/testbed":
                        return testbed_dir
                return Path(path_str)

            with (
                patch(
                    "benchmark.predict.docker.entrypoint_bench_agent.Path",
                    side_effect=mock_path,
                ),
                patch(
                    "benchmark.predict.docker.entrypoint_bench_agent.BenchAgentWrapper"
                ) as mock_w,
            ):
                mock_w.return_value.generate_patch.return_value = ("patch", {})
                # Act
                entrypoint_main()

                # Assert — session_id passed to generate_patch
                call_args = mock_w.return_value.generate_patch.call_args
                assert call_args[0][2] == "my-run-bench-agent-test-sid"


# ===========================================================================
# 3. Configuration Coherence Tests
# ===========================================================================


class Test04Integration03BenchAgent03ConfigCoherence:
    """Verify consistent configuration across all components."""

    def test_04_int_03_bench_agent_03_config_01_provider_is_always_fireworks(
        self, tmp_path: Path
    ) -> None:
        # Arrange — even if bench_settings.toml has an OpenRouter provider block
        config_path = tmp_path / "bench_settings.toml"
        config_path.write_text(
            "[histrel]\n"
            'models = "accounts/fireworks/models/deepseek-v4p1-flash-1017"\n'
            "[histrel.provider]\n"
            'provider = "openrouter"\n'
        )
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        # Act
        wrapper = BenchAgentWrapper(config_path)
        agent_config_path = wrapper._write_agent_config(repo_dir)

        # Assert — agent config always uses fireworks
        content = tomllib.loads(agent_config_path.read_text(encoding="utf-8"))
        assert content["model_provider"] == "fireworks"
        assert content["model_api_key_env"] == "FIREWORKS_API_KEY"

    def test_04_int_03_bench_agent_03_config_02_agent_cli_reads_env_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — use agent-format TOML (not harness bench_settings.toml)
        config_path = tmp_path / "agent_config.toml"
        config_path.write_text(
            "schema_version = 1\n"
            'session_id = "placeholder"\n'
            'task_id = "placeholder"\n'
            'agent_system_prompt = { initialize = "Fix the bug", step = "Fix the bug" }\n'
            'model_provider = "fireworks"\n'
            'model_name = "accounts/fireworks/models/deepseek-v4p1-flash-1017"\n'
            "model_temperature = 0.0\n"
            "model_top_p = 1.0\n"
            "model_context_limit_tokens = 128000\n"
            "model_max_tokens_output = 16384\n"
            'model_api_key_env = "FIREWORKS_API_KEY"\n'
            "\n[budgets]\n"
            "max_wall_seconds = 0\n"
            "max_turns = 0\n"
            "max_tool_calls = 0\n"
        )

        monkeypatch.setenv("SESSION_ID", "env-session-override")
        monkeypatch.setenv("INSTANCE_ID", "env-instance-override")
        monkeypatch.setenv("TIMEOUT_SECONDS", "60")
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")

        # Act — load config through the agent CLI path
        from histrel.cli import read_env, build_config_from_env

        env = read_env()
        config = build_config_from_env(config_path, env)

        # Assert — env overrides take effect
        assert config.session_id == "env-session-override"
        assert config.task_id == "env-instance-override"
        assert config.budgets.max_wall_seconds == 60

    def test_04_int_03_bench_agent_03_config_03_entrypoint_accepts_bench_agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — bench-agent should pass validation and reach generate_patch
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            workspace_dir = tmpdir_path / "workspace"
            instance_dir = workspace_dir / "instance"
            instance_dir.mkdir(parents=True)
            (instance_dir / "instance.json").write_text(
                json.dumps(
                    {"instance_id": "test-1", "repo": "test/repo", "base_commit": "abc"}
                )
            )
            (workspace_dir / "bench_settings.toml").write_text(_MINIMAL_TOML)
            (workspace_dir / "output").mkdir()

            testbed_dir = tmpdir_path / "testbed"
            testbed_dir.mkdir(parents=True)
            (testbed_dir / ".git").mkdir()

            monkeypatch.setenv("AGENT_TYPE", "bench-agent")
            monkeypatch.setenv("RUN_NAME", "test")
            monkeypatch.setenv("SESSION_ID", "test-bench-agent-test-1")

            def mock_path(path_str):
                if isinstance(path_str, str):
                    if path_str.startswith("/workspace"):
                        return workspace_dir / path_str.replace("/workspace/", "")
                    elif path_str == "/testbed":
                        return testbed_dir
                return Path(path_str)

            with (
                patch(
                    "benchmark.predict.docker.entrypoint_bench_agent.Path",
                    side_effect=mock_path,
                ),
                patch(
                    "benchmark.predict.docker.entrypoint_bench_agent.BenchAgentWrapper"
                ) as mock_w,
            ):
                mock_w.return_value.generate_patch.return_value = ("patch", {})
                # Act
                result = entrypoint_main()

                # Assert — bench-agent passes validation
                assert result == 0
                mock_w.assert_called_once()


# ===========================================================================
# 4. Failure Propagation Tests
# ===========================================================================


class Test04Integration04BenchAgent04FailurePropagation:
    """Verify system resilience when individual components fail."""

    def test_04_int_04_bench_agent_04_failure_01_missing_instance_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — no instance file
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            workspace_dir = tmpdir_path / "workspace"
            workspace_dir.mkdir()

            def mock_path(path_str):
                if isinstance(path_str, str) and path_str.startswith("/workspace"):
                    return workspace_dir / path_str.replace("/workspace/", "")
                return Path(path_str)

            with patch(
                "benchmark.predict.docker.entrypoint_bench_agent.Path",
                side_effect=mock_path,
            ):
                # Act
                result = entrypoint_main()

                # Assert
                assert result == 1

    def test_04_int_04_bench_agent_04_failure_02_invalid_instance_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            workspace_dir = tmpdir_path / "workspace"
            instance_dir = workspace_dir / "instance"
            instance_dir.mkdir(parents=True)
            (instance_dir / "instance.json").write_text("not valid json")
            (workspace_dir / "bench_settings.toml").write_text(_MINIMAL_TOML)

            def mock_path(path_str):
                if isinstance(path_str, str) and path_str.startswith("/workspace"):
                    return workspace_dir / path_str.replace("/workspace/", "")
                return Path(path_str)

            with (
                patch(
                    "benchmark.predict.docker.entrypoint_bench_agent.Path",
                    side_effect=mock_path,
                ),
                pytest.raises(json.JSONDecodeError),
            ):
                entrypoint_main()

    def test_04_int_04_bench_agent_04_failure_03_missing_config_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            workspace_dir = tmpdir_path / "workspace"
            instance_dir = workspace_dir / "instance"
            instance_dir.mkdir(parents=True)
            (instance_dir / "instance.json").write_text(
                json.dumps(
                    {"instance_id": "test-1", "repo": "test/repo", "base_commit": "abc"}
                )
            )
            # No bench_settings.toml
            (workspace_dir / "output").mkdir()

            testbed_dir = tmpdir_path / "testbed"
            testbed_dir.mkdir(parents=True)
            (testbed_dir / ".git").mkdir()

            monkeypatch.setenv("AGENT_TYPE", "bench-agent")
            monkeypatch.setenv("RUN_NAME", "test")
            monkeypatch.setenv("SESSION_ID", "test-bench-agent-test-1")

            def mock_path(path_str):
                if isinstance(path_str, str):
                    if path_str.startswith("/workspace"):
                        return workspace_dir / path_str.replace("/workspace/", "")
                    elif path_str == "/testbed":
                        return testbed_dir
                return Path(path_str)

            with patch(
                "benchmark.predict.docker.entrypoint_bench_agent.Path",
                side_effect=mock_path,
            ):
                # Act
                result = entrypoint_main()

                # Assert
                assert result == 1

    def test_04_int_04_bench_agent_04_failure_04_missing_repo_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            workspace_dir = tmpdir_path / "workspace"
            instance_dir = workspace_dir / "instance"
            instance_dir.mkdir(parents=True)
            (instance_dir / "instance.json").write_text(
                json.dumps(
                    {"instance_id": "test-1", "repo": "test/repo", "base_commit": "abc"}
                )
            )
            (workspace_dir / "bench_settings.toml").write_text(_MINIMAL_TOML)
            (workspace_dir / "output").mkdir()

            # No testbed directory

            monkeypatch.setenv("AGENT_TYPE", "bench-agent")
            monkeypatch.setenv("RUN_NAME", "test")
            monkeypatch.setenv("SESSION_ID", "test-bench-agent-test-1")

            def mock_path(path_str):
                if isinstance(path_str, str):
                    if path_str.startswith("/workspace"):
                        return workspace_dir / path_str.replace("/workspace/", "")
                    elif path_str == "/testbed":
                        return tmpdir_path / "nonexistent_testbed"
                return Path(path_str)

            with patch(
                "benchmark.predict.docker.entrypoint_bench_agent.Path",
                side_effect=mock_path,
            ):
                # Act
                result = entrypoint_main()

                # Assert
                assert result == 1

    def test_04_int_04_bench_agent_04_failure_05_agent_nonzero_exit_still_returns_patch(
        self, tmp_path: Path
    ) -> None:
        # Arrange — agent exits with error but git diff still has changes
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_fixture_repo(repo_dir)
        (repo_dir / "calculator.py").write_text("def add(a, b):\n    return a + b\n")

        config_path = tmp_path / "bench_settings.toml"
        config_path.write_text(_MINIMAL_TOML)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fake_result = MagicMock(returncode=1, stdout="error output", stderr="error")
        real_subprocess_run = subprocess.run

        def selective_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd and cmd[0] == "bash":
                return fake_result
            return real_subprocess_run(cmd, **kwargs)

        with (
            patch.object(
                BenchAgentWrapper,
                "_load_usage_data",
                return_value={"total": {"tokens_total": -1}},
            ),
            patch("subprocess.run", side_effect=selective_run),
            patch("benchmark.predict.agents.bench_agent_wrapper.Path") as mock_path,
        ):
            mock_path.return_value = output_dir
            wrapper = BenchAgentWrapper(config_path)

            # Act — should not raise even when agent fails
            patch_text, usage = wrapper.generate_patch(
                instance={"instance_id": "test-1"},
                repo_dir=repo_dir,
                session_id="test-session",
            )

            # Assert — patch still extracted from git, usage shows error
            assert "a + b" in patch_text
            assert usage["total"]["tokens_total"] == -1

    def test_04_int_04_bench_agent_04_failure_06_subprocess_timeout_handled(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_fixture_repo(repo_dir)

        config_path = tmp_path / "bench_settings.toml"
        config_path.write_text(_MINIMAL_TOML)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        real_subprocess_run = subprocess.run

        def selective_timeout(cmd, **kwargs):
            if isinstance(cmd, list) and cmd and cmd[0] == "bash":
                raise subprocess.TimeoutExpired(cmd="test", timeout=1)
            return real_subprocess_run(cmd, **kwargs)

        with (
            patch.object(
                BenchAgentWrapper,
                "_load_usage_data",
                return_value={"total": {"tokens_total": -1}},
            ),
            patch("subprocess.run", side_effect=selective_timeout),
            patch("benchmark.predict.agents.bench_agent_wrapper.Path") as mock_path,
        ):
            mock_path.return_value = output_dir
            wrapper = BenchAgentWrapper(config_path)

            # Act — should not raise on timeout
            patch_text, _ = wrapper.generate_patch(
                instance={"instance_id": "test-1"},
                repo_dir=repo_dir,
                session_id="test-session",
                timeout=1,
            )

            # Assert — empty patch from no changes, no crash
            assert isinstance(patch_text, str)


# ===========================================================================
# 5. Emergent Behavior Tests
# ===========================================================================


class Test04Integration05BenchAgent05EmergentBehavior:
    """Verify properties that only exist when components interact as a system."""

    def test_04_int_05_bench_agent_05_emergent_01_full_wrapper_to_patch_roundtrip(
        self, tmp_path: Path
    ) -> None:
        # Arrange — full flow: config -> wrapper -> agent_config -> fake exec -> patch
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_fixture_repo(repo_dir)

        real_subprocess_run = subprocess.run

        def fake_subprocess_run(cmd, **kwargs):
            # Only mock the agent execution (bash -c), let git commands through
            if isinstance(cmd, list) and cmd and cmd[0] == "bash":
                # Simulate the agent writing the fix
                (repo_dir / "calculator.py").write_text(
                    "def add(a, b):\n    return a + b\n", encoding="utf-8"
                )
                return MagicMock(returncode=0, stdout="Fixed!", stderr="")
            return real_subprocess_run(cmd, **kwargs)

        config_path = tmp_path / "bench_settings.toml"
        config_path.write_text(_MINIMAL_TOML)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        with (
            patch.object(
                BenchAgentWrapper,
                "_load_usage_data",
                return_value={"total": {"tokens_total": 100}},
            ),
            patch("subprocess.run", side_effect=fake_subprocess_run),
            patch("benchmark.predict.agents.bench_agent_wrapper.Path") as mock_path,
        ):
            mock_path.return_value = output_dir
            wrapper = BenchAgentWrapper(config_path)

            # Act
            patch_text, usage = wrapper.generate_patch(
                instance={"instance_id": "calc-1", "repo": "test/calc"},
                repo_dir=repo_dir,
                session_id="integration-session",
            )

            # Assert — full round trip: config read, agent_config written,
            # agent executed, fix applied, patch extracted, usage loaded
            assert (repo_dir / "agent_config.toml").exists()
            assert "a + b" in patch_text
            # The diff should show the removal of "return a - b"
            assert "-    return a - b" in patch_text
            assert "+    return a + b" in patch_text
            assert usage["total"]["tokens_total"] == 100

    def test_04_int_05_bench_agent_05_emergent_02_entrypoint_writes_patch_and_usage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            workspace_dir = tmpdir_path / "workspace"
            instance_dir = workspace_dir / "instance"
            instance_dir.mkdir(parents=True)
            instance = {
                "instance_id": "test-ep",
                "repo": "test/repo",
                "base_commit": "abc",
            }
            (instance_dir / "instance.json").write_text(json.dumps(instance))
            (workspace_dir / "bench_settings.toml").write_text(_MINIMAL_TOML)
            output_dir = workspace_dir / "output"
            output_dir.mkdir()

            testbed_dir = tmpdir_path / "testbed"
            testbed_dir.mkdir(parents=True)
            (testbed_dir / ".git").mkdir()
            (testbed_dir / "calculator.py").write_text(
                "def add(a, b):\n    return a - b\n"
            )

            monkeypatch.setenv("AGENT_TYPE", "bench-agent")
            monkeypatch.setenv("RUN_NAME", "ep-test")
            monkeypatch.setenv("SESSION_ID", "ep-test-bench-agent-test-ep")
            monkeypatch.setenv("TIMEOUT_SECONDS", "60")
            monkeypatch.delenv("HAS_TEST_RUNNER", raising=False)

            def mock_path(path_str):
                if isinstance(path_str, str):
                    if path_str.startswith("/workspace"):
                        return workspace_dir / path_str.replace("/workspace/", "")
                    elif path_str == "/testbed":
                        return testbed_dir
                return Path(path_str)

            def fake_generate_patch(instance, repo_dir, session_id, **kwargs):
                # Simulate the agent fixing the bug
                (testbed_dir / "calculator.py").write_text(
                    "def add(a, b):\n    return a + b\n", encoding="utf-8"
                )
                return (
                    "diff --git a/calculator.py b/calculator.py\n",
                    {"total": {"tokens_total": 500}},
                )

            with (
                patch(
                    "benchmark.predict.docker.entrypoint_bench_agent.Path",
                    side_effect=mock_path,
                ),
                patch(
                    "benchmark.predict.docker.entrypoint_bench_agent.BenchAgentWrapper"
                ) as mock_w,
            ):
                mock_w.return_value.generate_patch.side_effect = fake_generate_patch

                # Act
                result = entrypoint_main()

                # Assert
                assert result == 0
                patch_file = output_dir / "bench-agent_test-ep.jsonl"
                assert patch_file.exists()
                patch_record = json.loads(patch_file.read_text())
                assert patch_record["instance_id"] == "test-ep"
                assert patch_record["model_name_or_path"] == "bench-agent"
                assert "diff" in patch_record["model_patch"]

                usage_file = output_dir / "bench-agent_usage.json"
                assert usage_file.exists()
                usage_data = json.loads(usage_file.read_text())
                assert usage_data["total"]["tokens_total"] == 500

    def test_04_int_05_bench_agent_05_emergent_03_empty_patch_when_no_changes(
        self, tmp_path: Path
    ) -> None:
        # Arrange — repo with no changes after agent execution
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        _init_fixture_repo(repo_dir)

        config_path = tmp_path / "bench_settings.toml"
        config_path.write_text(_MINIMAL_TOML)
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fake_result = MagicMock(returncode=0, stdout="", stderr="")
        real_subprocess_run = subprocess.run

        def selective_run(cmd, **kwargs):
            if isinstance(cmd, list) and cmd and cmd[0] == "bash":
                return fake_result
            return real_subprocess_run(cmd, **kwargs)

        with (
            patch.object(
                BenchAgentWrapper,
                "_load_usage_data",
                return_value={"total": {"tokens_total": 0}},
            ),
            patch("subprocess.run", side_effect=selective_run),
            patch("benchmark.predict.agents.bench_agent_wrapper.Path") as mock_path,
        ):
            mock_path.return_value = output_dir
            wrapper = BenchAgentWrapper(config_path)

            # Act — no changes made to repo
            patch_text, _ = wrapper.generate_patch(
                instance={"instance_id": "test-1"},
                repo_dir=repo_dir,
                session_id="test-session",
            )

            # Assert — empty patch
            assert patch_text == ""

    def test_04_int_05_bench_agent_05_emergent_04_single_agent_in_run_config(
        self,
    ) -> None:
        # Arrange / Act
        cfg = PredictorConfig(swe_bench=SWEbenchConfig())

        # Assert — only the single supported agent type is configured
        assert AGENT_TYPE == "bench-agent"
        assert not hasattr(cfg.swe_bench, "agents")


# ===========================================================================
# 6. Live LLM Integration Tests (skip unless --llm)
# ===========================================================================


@pytest.mark.requires_live_llm
class Test04Integration06BenchAgent06LiveLlm:
    """Live LLM integration tests using Fireworks AI.

    These tests make real API calls to verify the full integration chain
    works with actual LLM responses. They skip unless --llm (or --all) is
    provided and FIREWORKS_API_KEY is set.

    Tests cover:
    - Agent CLI reads TOML and produces valid config
    - Agent loop runs with real Fireworks API and produces artifacts
    - Token usage is tracked from real API responses
    - Patch is extracted from git diff after agent modifies repo
    - BenchAgentWrapper config template produces valid AgentConfig
    - Entry point would accept the generated agent_config.toml
    """

    def test_04_int_06_bench_agent_06_live_llm_01_agent_config_from_template(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "bench_settings.toml"
        config_path.write_text(f'[histrel]\nmodels = "{_FIREWORKS_MODEL}"\n')
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        # Act
        wrapper = BenchAgentWrapper(config_path)
        agent_config_path = wrapper._write_agent_config(repo_dir)

        # Assert
        config = load_config(agent_config_path)
        assert config.session_id

    def test_04_int_06_bench_agent_06_live_llm_02_fireworks_adapter_returns_response(
        self,
    ) -> None:
        # Arrange — skip if no API key or server unreachable
        api_key = _require_live_fireworks()

        from openai import OpenAI

        client = OpenAI(base_url=_FIREWORKS_BASE_URL, api_key=api_key)
        adapter = OpenAICompatAdapter(
            client=client,
            model_name=_FIREWORKS_MODEL,
            provider="fireworks",
            request_timeout=_LIVE_TIMEOUT_SECONDS,
        )

        # Act — make a real model call
        request = ModelRequest(
            messages=(
                ProviderMessage(
                    role="user", content="What is 2 + 3? Answer with just the number."
                ),
            ),
            temperature=0.0,
            top_p=1.0,
            max_tokens_output=100,
        )

        response = adapter.complete(request)

        # Assert — response has text and usage
        assert len(response.text_parts) > 0
        text = " ".join(response.text_parts)
        assert "5" in text
        assert response.usage.input > 0
        assert response.usage.output > 0
        assert response.usage.total > 0

    def test_04_int_06_bench_agent_06_live_llm_03_agent_loop_e2e_with_fireworks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange — skip if no API key or server unreachable
        api_key = _require_live_fireworks()

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _init_fixture_repo(workspace)
        # The harness runs the agent with the workspace as CWD; the tools that
        # resolve paths or spawn commands depend on it.
        monkeypatch.chdir(workspace)
        output = tmp_path / "output"

        config = _make_agent_config(
            max_turns=5,
            model_name=_FIREWORKS_MODEL,
        )

        from openai import OpenAI

        client = OpenAI(base_url=_FIREWORKS_BASE_URL, api_key=api_key)
        adapter = OpenAICompatAdapter(
            client=client,
            model_name=_FIREWORKS_MODEL,
            provider="fireworks",
            request_timeout=_LIVE_TIMEOUT_SECONDS,
        )

        runtime = Runtime(config=config, workspace_root=workspace, output_dir=output)
        runtime.initialize(_USER_MSG)
        runtime.model_adapter = adapter

        # Act — run the full agent loop with real LLM
        outcome = run_agent(_USER_MSG, runtime)

        # Assert — session completed without infrastructure failure
        assert outcome.status in (
            SessionStatus.SUCCEEDED,
            SessionStatus.BUDGET_EXHAUSTED,
        )

        # Artifacts written
        artifacts_dir = output / config.session_id / "artifacts"
        assert (artifacts_dir / "messages.jsonl").exists()
        assert (artifacts_dir / "outcome.json").exists()

        # Token ledger recorded usage from real calls
        assert runtime.token_ledger.total_input > 0
        assert runtime.token_ledger.total_output > 0

    def test_04_int_06_bench_agent_06_live_llm_04_agent_produces_patch(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange — skip if no API key or server unreachable
        api_key = _require_live_fireworks()

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _init_fixture_repo(workspace)
        # The harness runs the agent with the workspace as CWD; tools that
        # resolve paths or spawn commands depend on it.
        monkeypatch.chdir(workspace)
        output = tmp_path / "output"

        config = _make_agent_config(
            max_turns=10,
            model_name=_FIREWORKS_MODEL,
        )

        from openai import OpenAI

        client = OpenAI(base_url=_FIREWORKS_BASE_URL, api_key=api_key)
        adapter = OpenAICompatAdapter(
            client=client,
            model_name=_FIREWORKS_MODEL,
            provider="fireworks",
            request_timeout=_LIVE_TIMEOUT_SECONDS,
        )

        runtime = Runtime(config=config, workspace_root=workspace, output_dir=output)
        runtime.initialize(_USER_MSG)
        runtime.model_adapter = adapter

        # Act
        run_agent(_USER_MSG, runtime)

        # Assert — check for repository changes
        diff_result = subprocess.run(
            ["git", "diff"],
            cwd=str(workspace),
            capture_output=True,
            text=True,
            timeout=10,
        )
        # The agent may or may not produce changes, but artifacts should exist
        artifacts_dir = output / config.session_id / "artifacts"
        assert (artifacts_dir / "repository.diff").exists()

        # If the agent made changes, verify the diff is non-empty
        if diff_result.stdout.strip():
            diff_text = diff_result.stdout
            # The diff should modify calculator.py
            assert "calculator.py" in diff_text

    def test_04_int_06_bench_agent_06_live_llm_05_wrapper_generates_valid_config_for_live(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — verify the full wrapper config chain works with real model
        api_key = _require_live_fireworks()

        # Use the actual bench_settings.toml model
        config_path = tmp_path / "bench_settings.toml"
        config_path.write_text(f'[histrel]\nmodels = "{_FIREWORKS_MODEL}"\n')
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        # Act — generate config through the wrapper
        wrapper = BenchAgentWrapper(config_path)
        agent_config_path = wrapper._write_agent_config(repo_dir)

        # Assert — config loads and has correct fields for live execution
        config = load_config(agent_config_path)
        assert config.model_provider == "fireworks"
        assert config.model_api_key_env == "FIREWORKS_API_KEY"

        # The config should be usable by the Runtime
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _init_fixture_repo(workspace)
        output = tmp_path / "output"

        monkeypatch.setenv("FIREWORKS_API_KEY", api_key)

        runtime = Runtime(
            config=config,
            workspace_root=workspace,
            output_dir=output,
        )
        runtime.initialize(_USER_MSG)

        # Verify the runtime can construct an adapter with this config
        assert runtime.model_adapter is not None

    def test_04_int_06_bench_agent_06_live_llm_06_histrel_condition_adapter(
        self, tmp_path: Path
    ) -> None:
        # Arrange — verify the histrel condition config is valid for live use
        api_key = _require_live_fireworks()

        config_path = tmp_path / "bench_settings.toml"
        config_path.write_text(f'[histrel]\nmodels = "{_FIREWORKS_MODEL}"\n')
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        # Act — generate the agent config
        wrapper = BenchAgentWrapper(config_path)
        agent_config_path = wrapper._write_agent_config(repo_dir)

        # Assert — config loads
        config = load_config(agent_config_path)
        assert config.session_id

        # Verify the adapter works with a real API call
        from openai import OpenAI

        client = OpenAI(base_url=_FIREWORKS_BASE_URL, api_key=api_key)
        adapter = OpenAICompatAdapter(
            client=client,
            model_name=_FIREWORKS_MODEL,
            provider="fireworks",
            request_timeout=_LIVE_TIMEOUT_SECONDS,
        )

        request = ModelRequest(
            messages=(
                ProviderMessage(
                    role="user",
                    content="Write a Python function that adds two numbers. Just the function.",
                ),
            ),
            temperature=0.0,
            top_p=1.0,
            # Reasoning tokens count against max_tokens on Fireworks, and
            # reasoning runs before any text is emitted.
            # A small budget (e.g. 200) is exhausted by reasoning alone
            # (finish_reason="length") before the answer is produced.
            max_tokens_output=4096,
        )

        response = adapter.complete(request)

        # Assert — real response received
        assert len(response.text_parts) > 0
        text = " ".join(response.text_parts)
        assert "def" in text.lower() or "add" in text.lower()
        assert response.usage.total > 0

    def test_04_int_06_bench_agent_06_live_llm_07_usage_data_from_real_agent_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange — verify token usage tracking from a real agent run
        api_key = _require_live_fireworks()

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _init_fixture_repo(workspace)
        monkeypatch.chdir(workspace)
        output = tmp_path / "output"

        config = _make_agent_config(
            max_turns=3,
            model_name=_FIREWORKS_MODEL,
        )

        from openai import OpenAI

        client = OpenAI(base_url=_FIREWORKS_BASE_URL, api_key=api_key)
        adapter = OpenAICompatAdapter(
            client=client,
            model_name=_FIREWORKS_MODEL,
            provider="fireworks",
            request_timeout=_LIVE_TIMEOUT_SECONDS,
        )

        runtime = Runtime(config=config, workspace_root=workspace, output_dir=output)
        runtime.initialize(_USER_MSG)
        runtime.model_adapter = adapter

        # Act
        run_agent(_USER_MSG, runtime)

        # Assert — artifacts contain real token usage
        artifacts_dir = output / config.session_id / "artifacts"
        costs_file = artifacts_dir / "costs.json"
        assert costs_file.exists()

        costs = json.loads(costs_file.read_text(encoding="utf-8"))
        assert "total_cost" in costs

        calls_file = artifacts_dir / "token_usage_calls.jsonl"
        assert calls_file.exists()
        calls_content = calls_file.read_text(encoding="utf-8").strip()
        if calls_content:
            first_call = json.loads(calls_content.splitlines()[0])
            assert "usage" in first_call
            assert first_call["usage"]["input"] > 0
            assert first_call["usage"]["output"] > 0

    def test_04_int_06_bench_agent_06_live_llm_08_wrapper_loads_usage_from_real_artifacts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange — run the agent, then use the wrapper to load usage
        api_key = _require_live_fireworks()

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _init_fixture_repo(workspace)
        monkeypatch.chdir(workspace)
        output = tmp_path / "output"

        config = _make_agent_config(
            max_turns=3,
            model_name=_FIREWORKS_MODEL,
        )

        from openai import OpenAI

        client = OpenAI(base_url=_FIREWORKS_BASE_URL, api_key=api_key)
        adapter = OpenAICompatAdapter(
            client=client,
            model_name=_FIREWORKS_MODEL,
            provider="fireworks",
            request_timeout=_LIVE_TIMEOUT_SECONDS,
        )

        runtime = Runtime(config=config, workspace_root=workspace, output_dir=output)
        runtime.initialize(_USER_MSG)
        runtime.model_adapter = adapter

        # Act — run agent
        run_agent(_USER_MSG, runtime)

        # Then load usage through the wrapper's _load_usage_data
        artifacts_dir = output / config.session_id / "artifacts"
        costs_file = artifacts_dir / "costs.json"

        wrapper = BenchAgentWrapper(tmp_path / "bench_settings.toml")
        usage_data = wrapper._load_usage_data(costs_file)

        # Assert — usage data has real token counts
        assert usage_data["total"]["tokens_input"] > 0
        assert usage_data["total"]["tokens_output"] > 0
        assert usage_data["total"]["tokens_total"] > 0
        assert usage_data["total"]["cost_total"] >= 0.0

    def test_04_int_06_bench_agent_06_live_llm_09_full_wrapper_roundtrip_live(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange — full end-to-end: wrapper writes config, agent runs with
        # real LLM, patch extracted, usage loaded — all through the wrapper
        api_key = _require_live_fireworks()

        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _init_fixture_repo(workspace)
        monkeypatch.chdir(workspace)

        config_path = tmp_path / "bench_settings.toml"
        config_path.write_text(f'[histrel]\nmodels = "{_FIREWORKS_MODEL}"\n')

        monkeypatch.setenv("FIREWORKS_API_KEY", api_key)
        monkeypatch.setenv("SESSION_ID", "live-roundtrip-sess")
        monkeypatch.setenv("INSTANCE_ID", "live-roundtrip-inst")
        monkeypatch.setenv("TIMEOUT_SECONDS", "120")
        monkeypatch.setenv("AGENT_TYPE", "bench-agent")

        # Act — write agent config through the wrapper
        wrapper = BenchAgentWrapper(config_path)
        agent_config_path = wrapper._write_agent_config(workspace)

        # Load it as an AgentConfig
        config = load_config(agent_config_path)
        config = config.model_copy(
            update={
                "session_id": "live-roundtrip-sess",
                "task_id": "live-roundtrip-inst",
                "budgets": Budgets(max_wall_seconds=120, max_turns=5, max_tool_calls=0),
            }
        )

        output = tmp_path / "output"

        from openai import OpenAI

        client = OpenAI(base_url=_FIREWORKS_BASE_URL, api_key=api_key)
        adapter = OpenAICompatAdapter(
            client=client,
            model_name=_FIREWORKS_MODEL,
            provider="fireworks",
            request_timeout=_LIVE_TIMEOUT_SECONDS,
        )

        runtime = Runtime(config=config, workspace_root=workspace, output_dir=output)
        runtime.initialize(_USER_MSG)
        runtime.model_adapter = adapter

        outcome = run_agent(_USER_MSG, runtime)

        # Assert — full round trip completed without infrastructure failure
        assert outcome.status in (
            SessionStatus.SUCCEEDED,
            SessionStatus.BUDGET_EXHAUSTED,
        )

        artifacts_dir = output / config.session_id / "artifacts"
        assert (artifacts_dir / "messages.jsonl").exists()
        assert (artifacts_dir / "outcome.json").exists()

        # Load usage through wrapper
        costs_file = artifacts_dir / "costs.json"
        usage_data = wrapper._load_usage_data(costs_file)
        assert usage_data["total"]["tokens_total"] > 0
