#!/usr/bin/env python3
"""Unit tests for the CLI entrypoint.

Tests environment variable reading, config construction, and the
main entry point. Uses monkeypatch for environment variables and
config file creation (no mocks).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
from dulwich.porcelain import add, commit
from dulwich.repo import Repo

from benchmark.predict.agents.prompts import get_bench_agent_sys_prompts
from histrel.cli import (
    build_config_from_env,
    build_parser,
    main,
    read_env,
    run,
)
from histrel.core.config import Mode
from histrel.core.runtime import Runtime
from histrel.terminal import SessionOutcome, SessionStatus

_MODES: list[Mode] = ["stateless", "react"]


def _write_toml(path: Path, content: str) -> Path:
    """Write a TOML file and return its path."""
    path.write_text(content, encoding="utf-8")
    return path


def _baseline_toml() -> str:
    """Return a minimal agent TOML config."""
    return """\
schema_version = 1
session_id = "test-session"
task_id = "test-task"
agent_system_prompt = { initialize = "You are a coding agent.", step = "You are a coding agent." }

model_provider = "openai"
model_name = "gpt-4o"
model_temperature = 0.0
model_top_p = 1.0
model_context_limit_tokens = 128000
model_max_tokens_output = 16384
model_api_key_env = "OPENAI_API_KEY"
"""


def _init_workspace(workspace: Path) -> None:
    """Initialize a git repository with one committed file."""
    Repo.init(str(workspace))
    (workspace / "main.py").write_text("print('hello')\n")
    add(str(workspace), [b"main.py"])
    commit(
        str(workspace),
        message=b"initial",
        author=b"T <t@t.com>",
        committer=b"T <t@t.com>",
    )


def _patch_cli_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the environment the CLI main entry point requires.

    The real file sink writes to /tmp and is covered by its own tests,
    so it is patched out to keep this test hermetic.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("SESSION_ID", "test-session")
    monkeypatch.setenv("INSTANCE_ID", "test-task")
    monkeypatch.setenv("TIMEOUT_SECONDS", "0")
    monkeypatch.setattr("histrel.cli.setup_file_sink", lambda: None)


def _fake_run_agent(
    workspace: Path, captured: list[Runtime] | None = None
) -> Callable[[str, Runtime], SessionOutcome]:
    """Build a run_agent replacement that reports a successful session.

    The CLI test validates wiring, not the agent loop itself (which has
    its own tests), so the replacement avoids real model calls and
    terminates deterministically.

    Args:
        workspace (Path): Workspace the replacement edits.
        captured (list[Runtime] | None): When given, receives the runtime
            the CLI constructed. Defaults to None.
    """

    def fake_run_agent(user_msg: str, runtime: Runtime) -> SessionOutcome:
        if captured is not None:
            captured.append(runtime)
        runtime.initialize(user_msg)
        (workspace / "main.py").write_text("print('fixed')\n")
        return runtime.finalize(
            SessionOutcome(
                status=SessionStatus.SUCCEEDED,
                reason="no_tool_calls_with_repo_changes",
            )
        )

    return fake_run_agent


class Test04Unit01Cli01Env:
    """Environment variable reading."""

    def test_04_unit_01_cli_01_env_01_reads_all_vars(self, monkeypatch) -> None:
        # Arrange
        monkeypatch.setenv("RUN_NAME", "bench-run-001")
        monkeypatch.setenv("AGENT_TYPE", "bench-agent")
        monkeypatch.setenv("INSTANCE_ID", "django--django-14007")
        monkeypatch.setenv("SESSION_ID", "sess-001")
        monkeypatch.setenv("TIMEOUT_SECONDS", "3600")
        monkeypatch.setenv("FIREWORKS_API_KEY", "fw-key-123")

        # Act
        env = read_env()

        # Assert
        assert env.run_name == "bench-run-001"
        assert env.agent_type == "bench-agent"
        assert env.instance_id == "django--django-14007"
        assert env.session_id == "sess-001"
        assert env.timeout_seconds == 3600
        assert env.fireworks_api_key == "fw-key-123"

    def test_04_unit_01_cli_01_env_02_defaults_when_unset(self, monkeypatch) -> None:
        # Arrange
        monkeypatch.delenv("RUN_NAME", raising=False)
        monkeypatch.delenv("AGENT_TYPE", raising=False)
        monkeypatch.delenv("INSTANCE_ID", raising=False)
        monkeypatch.delenv("SESSION_ID", raising=False)
        monkeypatch.delenv("TIMEOUT_SECONDS", raising=False)
        monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)

        # Act
        env = read_env()

        # Assert
        assert env.run_name == ""
        assert env.agent_type == ""
        assert env.instance_id == ""
        assert env.session_id == ""
        assert env.timeout_seconds == 0
        assert env.fireworks_api_key == ""

    def test_04_unit_01_cli_01_env_03_timeout_seconds_parsed_as_int(
        self, monkeypatch
    ) -> None:
        # Arrange
        monkeypatch.setenv("TIMEOUT_SECONDS", "7200")

        # Act
        env = read_env()

        # Assert
        assert env.timeout_seconds == 7200
        assert isinstance(env.timeout_seconds, int)


class Test04Unit02Cli02ConfigFromEnv:
    """Config construction from environment variables and TOML."""

    def test_04_unit_02_cli_02_config_01_builds_baseline_config(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        toml_path = tmp_path / "config.toml"
        _write_toml(toml_path, _baseline_toml())

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("SESSION_ID", "sess-from-env")
        monkeypatch.setenv("INSTANCE_ID", "task-from-env")

        # Act
        config = build_config_from_env(toml_path, read_env())

        # Assert — env vars override TOML for session_id and task_id
        assert config.session_id == "sess-from-env"
        assert config.task_id == "task-from-env"
        assert config.model_provider == "openai"

    def test_04_unit_02_cli_02_config_02_timeout_overrides_budget(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        toml_path = tmp_path / "config.toml"
        toml_content = _baseline_toml() + "\n[budgets]\nmax_wall_seconds = 0\n"
        _write_toml(toml_path, toml_content)

        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("TIMEOUT_SECONDS", "1800")

        # Act
        config = build_config_from_env(toml_path, read_env())

        # Assert — TIMEOUT_SECONDS overrides max_wall_seconds
        assert config.budgets.max_wall_seconds == 1800


class Test04Unit03Cli03Main:
    """CLI main entry point integration."""

    def test_04_unit_03_cli_03_main_01_runs_and_terminates(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange — create a config TOML and a workspace with a git repo
        toml_path = _write_toml(tmp_path / "config.toml", _baseline_toml())
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        output = tmp_path / "output"
        _init_workspace(workspace)
        _patch_cli_env(monkeypatch)
        monkeypatch.setattr("histrel.cli.run_agent", _fake_run_agent(workspace))

        # Act
        main(
            user_msg="Fix the failing test.",
            config_path=toml_path,
            workspace_root=workspace,
            output_dir=output,
        )

        # Assert — artifacts were written
        artifacts_dir = output / "test-session" / "artifacts"
        assert (artifacts_dir / "outcome.json").exists()
        assert (artifacts_dir / "result_manifest.json").exists()
        source_context = json.loads(
            (artifacts_dir / "source_context.json").read_text(encoding="utf-8")
        )
        assert source_context["mode"] == "stateless"

    def test_04_unit_03_cli_03_main_02_mode_flag_sets_config_mode(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange
        toml_path = _write_toml(tmp_path / "config.toml", _baseline_toml())
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        output = tmp_path / "output"
        _init_workspace(workspace)
        _patch_cli_env(monkeypatch)
        monkeypatch.setattr("histrel.cli.run_agent", _fake_run_agent(workspace))

        # Act
        main(
            user_msg="Fix the failing test.",
            config_path=toml_path,
            workspace_root=workspace,
            output_dir=output,
            mode="react",
        )

        # Assert — the CLI flag wins over the config default
        source_context = json.loads(
            (output / "test-session" / "artifacts" / "source_context.json").read_text(
                encoding="utf-8"
            )
        )
        assert source_context["mode"] == "react"


class Test04Unit03Cli04ModeParser:
    """``python -m histrel`` argument parsing."""

    def test_04_unit_03_cli_04_mode_parser_01_defaults_to_stateless(self) -> None:
        # Act
        args = build_parser().parse_args(["cfg.toml", "workspace", "out", "Fix it."])

        # Assert
        assert args.mode == "stateless"

    def test_04_unit_03_cli_04_mode_parser_02_accepts_react(self) -> None:
        # Act
        args = build_parser().parse_args(
            ["--mode", "react", "cfg.toml", "workspace", "out", "Fix it."]
        )

        # Assert
        assert args.mode == "react"

    def test_04_unit_03_cli_04_mode_parser_03_keeps_dashed_message_positional(
        self,
    ) -> None:
        # Act
        args = build_parser().parse_args(
            ["--mode", "react", "cfg.toml", "workspace", "out", "--", "--None--"]
        )

        # Assert
        assert args.user_msg == "--None--"

    def test_04_unit_03_cli_04_mode_parser_04_rejects_unknown_mode(self) -> None:
        # Act & Assert
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                ["--mode", "agentic", "cfg.toml", "workspace", "out", "Fix it."]
            )

    def test_04_unit_03_cli_04_mode_parser_05_run_takes_arguments(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Arrange — the console-script target must accept a plain argv
        toml_path = _write_toml(tmp_path / "config.toml", _baseline_toml())
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _init_workspace(workspace)
        _patch_cli_env(monkeypatch)
        monkeypatch.setattr("histrel.cli.run_agent", _fake_run_agent(workspace))

        # Act
        exit_code = run(
            [
                str(toml_path),
                str(workspace),
                str(tmp_path / "output"),
                "Fix the failing test.",
            ]
        )

        # Assert
        assert exit_code == 0


class Test04Unit03Cli06Prompts:
    """System prompts come from the harness prompt module, not the TOML."""

    @pytest.mark.parametrize("mode", _MODES)
    def test_04_unit_03_cli_06_prompts_01_mode_selects_rendered_prompts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: Mode
    ) -> None:
        # Arrange
        toml_path = _write_toml(tmp_path / "config.toml", _baseline_toml())
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        _init_workspace(workspace)
        _patch_cli_env(monkeypatch)
        captured: list[Runtime] = []
        monkeypatch.setattr(
            "histrel.cli.run_agent", _fake_run_agent(workspace, captured)
        )

        # Act
        main(
            user_msg="Fix the failing test.",
            config_path=toml_path,
            workspace_root=workspace,
            output_dir=tmp_path / "output",
            mode=mode,
        )

        # Assert — the rendered harness prompts replace the TOML text
        expected = get_bench_agent_sys_prompts(workspace, mode)
        prompts = captured[0].config.agent_system_prompt
        assert prompts.initialize == expected["initialize"]
        assert prompts.step == expected["step"]
        assert prompts.initialize != "You are a coding agent."
        assert str(workspace) in prompts.initialize
