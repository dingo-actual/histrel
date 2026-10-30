#!/usr/bin/env python3
"""Unit tests for session event recording.

Tests compaction event recording, the compactor's compaction callback, and
the source-context snapshot using real objects (no mocks).
"""

from __future__ import annotations

from pathlib import Path

from histrel.context.compactor import CompactedState
from histrel.context.session_events import (
    SessionEvents,
    build_source_context,
)
from tests.unit.context.helpers import CompactorHarness
from tests.unit.factories import build_agent_config


class Test04Unit05SessionEvents01Compactions:
    """Compaction event recording."""

    def test_04_unit_05_sessionevents_01_compactions_01_records_counts(self) -> None:
        # Arrange
        events = SessionEvents()
        state = CompactedState(
            system_prompt="system",
            original_task="task",
            recent_message_ids=("m7", "m8"),
            source_message_index_start=3,
            source_message_index_end=6,
            working_summary="Earlier work.",
        )

        # Act
        events.record_compaction(state)

        # Assert
        assert len(events.compactions) == 1
        event = events.compactions[0]
        assert event.compacted_message_count == 4
        assert event.retained_message_count == 2
        assert event.summary == "Earlier work."

    def test_04_unit_05_sessionevents_01_compactions_02_no_middle_counts_zero(
        self,
    ) -> None:
        # Arrange
        events = SessionEvents()
        state = CompactedState(
            system_prompt="system",
            original_task="task",
            source_message_index_start=-1,
            source_message_index_end=-1,
        )

        # Act
        events.record_compaction(state)

        # Assert
        assert events.compactions[0].compacted_message_count == 0

    def test_04_unit_05_sessionevents_01_compactions_03_compactor_invokes_hook(
        self,
        make_compactor,
        make_messages,
    ) -> None:
        # Arrange
        events = SessionEvents()
        harness: CompactorHarness = make_compactor(
            on_compaction=events.record_compaction
        )
        messages = make_messages(
            ("user", "task"),
            ("assistant", "step one"),
            ("assistant", "step two"),
            ("assistant", "step three"),
        )

        # Act
        harness.compact(messages, recent_tail_count=1)

        # Assert — the produced state was recorded as one event
        assert len(events.compactions) == 1
        assert events.compactions[0].compacted_message_count == 2


class Test04Unit05SessionEvents02SourceContext:
    """Source-context snapshot."""

    def test_04_unit_05_sessionevents_02_sourcecontext_01_records_model_and_tools(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config = build_agent_config()

        # Act
        context = build_source_context(
            config=config,
            workspace_root=tmp_path,
            tool_names=["read", "bash"],
        )

        # Assert
        assert context["session_id"] == config.session_id
        assert context["task_id"] == config.task_id
        assert context["mode"] == config.mode
        assert context["model"]["name"] == config.model_name
        assert context["tools"] == ["bash", "read"]

    def test_04_unit_05_sessionevents_02_sourcecontext_02_records_workspace_root(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        workspace = tmp_path / "workspace"
        workspace.mkdir()

        # Act
        context = build_source_context(
            config=build_agent_config(),
            workspace_root=workspace,
            tool_names=[],
        )

        # Assert — a directory that is not a git repository has no revision
        assert context["workspace"]["root"] == str(workspace)
        assert context["workspace"]["head_commit"] is None
        assert context["workspace"]["branch"] is None

    def test_04_unit_05_sessionevents_02_sourcecontext_03_records_budgets(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config = build_agent_config(max_turns=7, max_tool_calls=11)

        # Act
        context = build_source_context(
            config=config,
            workspace_root=tmp_path,
            tool_names=[],
        )

        # Assert
        assert context["budgets"]["max_turns"] == 7
        assert context["budgets"]["max_tool_calls"] == 11
