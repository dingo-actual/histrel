#!/usr/bin/env python3
"""Unit tests for Runtime initialization."""

from __future__ import annotations

from histrel.core.journal import load_journal
from histrel.core.runtime import (
    Runtime,
)
from tests.unit.factories import (
    build_agent_config,
    init_fixture_repo,
)
from tests.unit.runtime.helpers import (
    USER_MSG,
    outside_output_dir,
)


class Test04Unit01Runtime01Init:
    """Runtime initialization."""

    def test_04_unit_01_runtime_01_init_01_initializes_components(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config = build_agent_config()
        init_fixture_repo(tmp_path)

        # Act
        runtime = Runtime(
            config=config, workspace_root=tmp_path, output_dir=tmp_path / "output"
        )
        runtime.initialize(USER_MSG)

        # Assert
        assert runtime.message_factory.next_message_index > 0  # task message committed
        assert runtime.turn_count == 0
        assert runtime.tool_call_count == 0
        messages = list(load_journal(runtime.journal._path))
        assert len(messages) >= 1  # at least the task message

    def test_04_unit_01_runtime_01_init_02_repo_checker_reports_clean_state(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config = build_agent_config()
        init_fixture_repo(tmp_path)

        # Act
        runtime = Runtime(
            config=config,
            workspace_root=tmp_path,
            output_dir=outside_output_dir(tmp_path),
        )
        runtime.initialize(USER_MSG)

        # Assert — the checker reports no changes against the clean
        # committed state
        assert runtime.repo_checker.has_changes() is False
