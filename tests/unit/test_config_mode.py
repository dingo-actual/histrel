#!/usr/bin/env python3
"""Unit tests for the mode field on AgentConfig.

Tests the agent loop mode ("stateless" or "react"): default value,
TOML loading, and rejection of invalid values.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from histrel.core.config import AgentConfig, SystemPrompts, load_config


def _write_toml(tmpdir: Path, content: str) -> Path:
    """Write a TOML file and return its path."""
    path = tmpdir / "config.toml"
    path.write_text(content, encoding="utf-8")
    return path


class Test04Unit35Config35AgentMode:
    """mode field on AgentConfig."""

    def test_04_unit_35_config_35_agent_mode_01_default_is_stateless(self) -> None:
        # Act — mode omitted
        config = AgentConfig(
            session_id="s",
            task_id="t",
            agent_system_prompt=SystemPrompts(initialize="x", step="x"),
            model_provider="openai",  # type: ignore[arg-type]
            model_name="gpt-4o",
        )

        # Assert
        assert config.mode == "stateless"

    def test_04_unit_35_config_35_agent_mode_02_explicit_stateless_ok(self) -> None:
        # Act
        config = AgentConfig(
            session_id="s",
            task_id="t",
            agent_system_prompt=SystemPrompts(initialize="x", step="x"),
            model_provider="openai",  # type: ignore[arg-type]
            model_name="gpt-4o",
            mode="stateless",
        )

        # Assert
        assert config.mode == "stateless"

    def test_04_unit_35_config_35_agent_mode_03_react_ok(self) -> None:
        # Act
        config = AgentConfig(
            session_id="s",
            task_id="t",
            agent_system_prompt=SystemPrompts(initialize="x", step="x"),
            model_provider="openai",  # type: ignore[arg-type]
            model_name="gpt-4o",
            mode="react",
        )

        # Assert
        assert config.mode == "react"

    def test_04_unit_35_config_35_agent_mode_04_invalid_rejected(self) -> None:
        # Act & Assert
        with pytest.raises(ValidationError):
            AgentConfig(
                session_id="s",
                task_id="t",
                agent_system_prompt=SystemPrompts(initialize="x", step="x"),
                model_provider="openai",  # type: ignore[arg-type]
                model_name="gpt-4o",
                mode="agentic",  # type: ignore[arg-type]
            )

    def test_04_unit_35_config_35_agent_mode_05_loaded_from_toml(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        content = """\
schema_version = 1
session_id = "s"
task_id = "t"
mode = "react"
agent_system_prompt = { initialize = "Prompt.", step = "Prompt." }

model_provider = "openai"
model_name = "gpt-4o"
model_api_key_env = "OPENAI_API_KEY"
"""
        path = _write_toml(tmp_path, content)

        # Act
        config = load_config(path)

        # Assert
        assert config.mode == "react"

    def test_04_unit_35_config_35_agent_mode_06_toml_default_when_omitted(
        self, tmp_path: Path
    ) -> None:
        # Arrange — no mode key in TOML
        content = """\
schema_version = 1
session_id = "s"
task_id = "t"
agent_system_prompt = { initialize = "Prompt.", step = "Prompt." }

model_provider = "openai"
model_name = "gpt-4o"
model_api_key_env = "OPENAI_API_KEY"
"""
        path = _write_toml(tmp_path, content)

        # Act
        config = load_config(path)

        # Assert
        assert config.mode == "stateless"

    def test_04_unit_35_config_35_agent_mode_07_toml_invalid_rejected(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        content = """\
schema_version = 1
session_id = "s"
task_id = "t"
mode = "agentic"
agent_system_prompt = { initialize = "Prompt.", step = "Prompt." }

model_provider = "openai"
model_name = "gpt-4o"
model_api_key_env = "OPENAI_API_KEY"
"""
        path = _write_toml(tmp_path, content)

        # Act & Assert
        with pytest.raises(ValidationError):
            load_config(path)
