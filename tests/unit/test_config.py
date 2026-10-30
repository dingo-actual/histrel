#!/usr/bin/env python3
"""Unit tests for benchmark agent configuration (AgentConfig).

Tests TOML loading, Pydantic v2 validation, strict unknown-field rejection,
and the hard-coded provider guard that rejects openai for histrel
conditions.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from histrel.core.config import (
    AgentConfig,
    Budgets,
    Compaction,
    SystemPrompts,
    load_config,
)


def _write_toml(tmpdir: Path, content: str) -> Path:
    """Write a TOML file and return its path."""
    path = tmpdir / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


class Test04Unit01Config01Schema:
    """AgentConfig model validation."""

    def test_04_unit_01_config_01_schema_01_valid_baseline_config(self) -> None:
        # Arrange
        config = AgentConfig(
            schema_version=1,
            session_id="test-session",
            task_id="test-instance",
            agent_system_prompt=SystemPrompts(
                initialize="You are a coding agent.", step="You are a coding agent."
            ),
            model_provider="openai",
            model_name="gpt-4o",
            model_temperature=0.0,
            model_top_p=1.0,
            model_reasoning_effort=None,
            model_context_limit_tokens=128000,
            model_max_tokens_output=16384,
            model_api_key_env="OPENAI_API_KEY",
            model_base_url=None,
        )

        # Assert
        assert config.model_provider == "openai"
        assert config.model_context_limit_tokens == 128000

    def test_04_unit_01_config_01_schema_02_valid_histrel_config(self) -> None:
        # Arrange
        config = AgentConfig(
            schema_version=1,
            session_id="test-session",
            task_id="test-instance",
            agent_system_prompt=SystemPrompts(
                initialize="You are a coding agent.", step="You are a coding agent."
            ),
            model_provider="fireworks",
            model_name="accounts/fireworks/models/deepseek-v4p1-flash-1017",
            model_temperature=0.0,
            model_top_p=1.0,
            model_reasoning_effort="max",
            model_context_limit_tokens=1048576,
            model_max_tokens_output=131072,
            model_api_key_env="FIREWORKS_API_KEY",
            model_base_url="https://api.fireworks.ai/inference/v1",
        )

        # Assert
        assert config.model_provider == "fireworks"
        assert config.model_reasoning_effort == "max"

    def test_04_unit_01_config_01_schema_03_rejects_unknown_fields(self) -> None:
        # Act & Assert — extra="forbid"
        with pytest.raises(ValidationError):
            AgentConfig(
                schema_version=1,
                session_id="s",
                task_id="t",
                agent_system_prompt=SystemPrompts(initialize="x", step="x"),
                model_provider="openai",
                model_name="m",
                model_temperature=0.0,
                model_top_p=1.0,
                model_reasoning_effort=None,
                model_context_limit_tokens=128000,
                model_max_tokens_output=16384,
                model_api_key_env="OPENAI_API_KEY",
                model_base_url=None,
                unknown_field="bad",  # type: ignore[call-arg]
            )


class Test04Unit02Config02ProviderGuard:
    """Hard-coded provider guard: only openai and fireworks are accepted."""

    def test_04_unit_02_config_02_provider_guard_01_openai_ok(
        self,
    ) -> None:
        # Act
        config = AgentConfig(
            schema_version=1,
            session_id="s",
            task_id="t",
            agent_system_prompt=SystemPrompts(initialize="x", step="x"),
            model_provider="openai",
            model_name="gpt-4o",
            model_temperature=0.0,
            model_top_p=1.0,
            model_reasoning_effort=None,
            model_context_limit_tokens=128000,
            model_max_tokens_output=16384,
            model_api_key_env="OPENAI_API_KEY",
            model_base_url=None,
        )

        # Assert — no error raised
        assert config.model_provider == "openai"

    def test_04_unit_02_config_02_provider_guard_04_fireworks_ok(
        self,
    ) -> None:
        # Act
        config = AgentConfig(
            schema_version=1,
            session_id="s",
            task_id="t",
            agent_system_prompt=SystemPrompts(initialize="x", step="x"),
            model_provider="fireworks",
            model_name="accounts/fireworks/models/deepseek-v4p1-flash-1017",
            model_temperature=0.0,
            model_top_p=1.0,
            model_reasoning_effort="max",
            model_context_limit_tokens=1048576,
            model_max_tokens_output=131072,
            model_api_key_env="FIREWORKS_API_KEY",
            model_base_url="https://api.fireworks.ai/inference/v1",
        )

        # Assert
        assert config.model_provider == "fireworks"

    def test_04_unit_02_config_02_provider_guard_06_unknown_provider_rejected(
        self,
    ) -> None:
        # Act & Assert
        with pytest.raises(ValidationError):
            AgentConfig(
                schema_version=1,
                session_id="s",
                task_id="t",
                agent_system_prompt=SystemPrompts(initialize="x", step="x"),
                model_provider="claude",  # type: ignore[arg-type]
                model_name="m",
                model_temperature=0.0,
                model_top_p=1.0,
                model_reasoning_effort=None,
                model_context_limit_tokens=128000,
                model_max_tokens_output=16384,
                model_api_key_env="CLAUDE_API_KEY",
                model_base_url=None,
            )


class Test04Unit03Config03TomlLoading:
    """TOML file loading via tomllib."""

    def test_04_unit_03_config_03_toml_01_load_valid_file(self) -> None:
        # Arrange
        toml_content = """\
schema_version = 1
session_id = "from-toml"
task_id = "instance-42"
agent_system_prompt = { initialize = "You are a test agent.", step = "You are a test agent." }

model_provider = "openai"
model_name = "gpt-4o"
model_temperature = 0.0
model_top_p = 1.0
model_context_limit_tokens = 128000
model_max_tokens_output = 16384
model_api_key_env = "OPENAI_API_KEY"
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_toml(Path(tmp), toml_content)

            # Act
            config = load_config(path)

            # Assert
            assert config.session_id == "from-toml"
            assert config.task_id == "instance-42"
            assert config.model_provider == "openai"
            assert config.model_context_limit_tokens == 128000

    def test_04_unit_03_config_03_toml_02_load_with_sections(self) -> None:
        # Arrange
        toml_content = """\
schema_version = 1
session_id = "sec-test"
task_id = "inst-1"
agent_system_prompt = { initialize = "Prompt text.", step = "Prompt text." }

model_provider = "fireworks"
model_name = "accounts/fireworks/models/deepseek-v4p1-flash-1017"
model_temperature = 1.0
model_top_p = 1.0
model_reasoning_effort = "max"
model_context_limit_tokens = 1048576
model_max_tokens_output = 131072
model_api_key_env = "FIREWORKS_API_KEY"
model_base_url = "https://api.fireworks.ai/inference/v1"

[budgets]
max_wall_seconds = 3600
max_turns = 20

[compaction]
compact_at_fraction = 0.75
target_fraction = 0.35
compact_retain_tail = 8
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_toml(Path(tmp), toml_content)

            # Act
            config = load_config(path)

            # Assert
            assert config.model_reasoning_effort == "max"
            assert config.budgets.max_wall_seconds == 3600
            assert config.budgets.max_turns == 20
            assert config.compaction.compact_at_fraction == 0.75
            assert config.compaction.target_fraction == 0.35
            assert config.compaction.compact_retain_tail == 8

    def test_04_unit_03_config_03_toml_03_missing_file_raises(self) -> None:
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/path/config.toml"))

    def test_04_unit_03_config_03_toml_04_system_prompt_from_config(
        self,
    ) -> None:
        # Arrange
        toml_content = """\
schema_version = 1
session_id = "s"
task_id = "t"
agent_system_prompt = { initialize = "Initialize prompt from TOML.", step = "Step prompt from TOML." }

model_provider = "openai"
model_name = "gpt-4o"
model_temperature = 0.0
model_top_p = 1.0
model_context_limit_tokens = 128000
model_max_tokens_output = 16384
model_api_key_env = "OPENAI_API_KEY"
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_toml(Path(tmp), toml_content)

            # Act
            config = load_config(path)

            # Assert — system prompts must come from TOML, not hardcoded
            assert (
                config.agent_system_prompt.initialize == "Initialize prompt from TOML."
            )
            assert config.agent_system_prompt.step == "Step prompt from TOML."


class Test04Unit04Config04SubModels:
    """Sub-model defaults and validation."""

    def test_04_unit_04_config_04_sub_models_01_budgets_defaults(self) -> None:
        # Act
        budgets = Budgets()

        # Assert — 0 means unlimited
        assert budgets.max_wall_seconds == 0
        assert budgets.max_turns == 0
        assert budgets.max_tool_calls == 0

    def test_04_unit_04_config_04_sub_models_02_compaction_defaults(self) -> None:
        # Act
        comp = Compaction()

        # Assert
        assert comp.compact_at_fraction == 0.75
        assert comp.target_fraction == 0.35
        assert comp.compact_retain_tail == 10


class Test04Unit05Config05DuplicateWindow:
    """duplicate_call_window validation and TOML loading."""

    def test_04_unit_05_config_05_window_01_default_is_five(self) -> None:
        # Arrange
        config = AgentConfig(
            session_id="test-session",
            task_id="test-instance",
            agent_system_prompt=SystemPrompts(
                initialize="You are a coding agent.", step="You are a coding agent."
            ),
            model_provider="openai",  # type: ignore[arg-type]
            model_name="gpt-4o",
        )

        # Act
        window = config.duplicate_call_window

        # Assert
        assert window == 5

    def test_04_unit_05_config_05_window_02_zero_allowed(self) -> None:
        # Arrange — zero disables the window-based duplicate rules.
        config = AgentConfig(
            session_id="test-session",
            task_id="test-instance",
            agent_system_prompt=SystemPrompts(
                initialize="You are a coding agent.", step="You are a coding agent."
            ),
            model_provider="openai",  # type: ignore[arg-type]
            model_name="gpt-4o",
            duplicate_call_window=0,
        )

        # Act
        window = config.duplicate_call_window

        # Assert
        assert window == 0

    def test_04_unit_05_config_05_window_03_negative_rejected(self) -> None:
        # Arrange & Act & Assert
        with pytest.raises(ValidationError, match="greater than or equal"):
            AgentConfig(
                session_id="test-session",
                task_id="test-instance",
                agent_system_prompt=SystemPrompts(
                    initialize="You are a coding agent.", step="You are a coding agent."
                ),
                model_provider="openai",  # type: ignore[arg-type]
                model_name="gpt-4o",
                duplicate_call_window=-1,
            )

    def test_04_unit_05_config_05_window_04_loaded_from_toml(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        content = """\n            schema_version = 1
            session_id = "test-session"
            task_id = "test-instance"
            agent_system_prompt = { initialize = "You are a coding agent.", step = "You are a coding agent." }
            model_provider = "openai"
            model_name = "gpt-4o"
            duplicate_call_window = 3
            """
        path = _write_toml(tmp_path, content)

        # Act
        config = load_config(path)

        # Assert
        assert config.duplicate_call_window == 3
