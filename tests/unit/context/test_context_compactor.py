#!/usr/bin/env python3
"""Unit tests for compactor triggering and CompactedState shape."""

from __future__ import annotations

from collections.abc import Callable

from histrel.context.compactor import Compactor
from histrel.model.messages import Message
from tests.unit.context.helpers import CompactorHarness
from tests.unit.factories import build_agent_config


class Test04Unit03Context03Compactor:
    """Compaction trigger, protected head, and index/tail semantics."""

    def test_04_unit_03_context_03_compactor_01_no_compact_below_threshold(
        self,
    ) -> None:
        # Arrange
        config = build_agent_config(context_limit=128000, max_output=16384)
        compactor = Compactor(config)

        # Act — well below the compaction threshold
        should = compactor.should_compact(
            estimated_tokens=1000,
            usable_context=100000,
        )

        # Assert
        assert should is False

    def test_04_unit_03_context_03_compactor_02_compact_above_threshold(
        self,
    ) -> None:
        # Arrange
        config = build_agent_config(context_limit=128000, max_output=16384)
        compactor = Compactor(config)

        # Act — above 75% of usable context
        should = compactor.should_compact(
            estimated_tokens=80000,
            usable_context=100000,
        )

        # Assert
        assert should is True

    def test_04_unit_03_context_03_compactor_03_compact_at_exact_threshold(
        self,
    ) -> None:
        # Arrange
        config = build_agent_config(context_limit=128000, max_output=16384)
        compactor = Compactor(config)

        # Act — exactly at 75% threshold
        should = compactor.should_compact(
            estimated_tokens=75000,
            usable_context=100000,
        )

        # Assert — at or above the threshold triggers compaction
        assert should is True

    def test_04_unit_03_context_03_compactor_04_protected_head_preserved(
        self,
        make_compactor: Callable[..., CompactorHarness],
        make_messages: Callable[..., list[Message]],
    ) -> None:
        # Arrange
        harness = make_compactor()
        messages = make_messages(
            ("user", "Original task text"),
            ("assistant", "Working on it"),
        )

        # Act
        compacted = harness.compact(
            messages,
            original_task="Original task text",
            recent_tail_count=1,
        )

        # Assert — system prompt and original task are preserved verbatim
        assert compacted.system_prompt == "You are a coding agent."
        assert compacted.original_task == "Original task text"

    def test_04_unit_03_context_03_compactor_05_index_range_covers_middle_only(
        self,
        make_compactor: Callable[..., CompactorHarness],
        make_messages: Callable[..., list[Message]],
    ) -> None:
        # Arrange — index 0 is the protected head, index 3 the recent tail
        harness = make_compactor()
        messages = make_messages(
            ("user", "task"),
            ("assistant", "step1"),
            ("assistant", "step2"),
            ("user", "recent"),
        )

        # Act
        compacted = harness.compact(messages, recent_tail_count=1)

        # Assert — index range covers only the compacted middle (1-2)
        assert compacted.source_message_index_start == 1
        assert compacted.source_message_index_end == 2

    def test_04_unit_03_context_03_compactor_07_recent_message_ids_retained(
        self,
        make_compactor: Callable[..., CompactorHarness],
        make_messages: Callable[..., list[Message]],
    ) -> None:
        # Arrange — indices 2 and 3 are the recent tail
        harness = make_compactor()
        messages = make_messages(
            ("user", "task"),
            ("assistant", "middle"),
            ("user", "recent1"),
            ("assistant", "recent2"),
        )

        # Act
        compacted = harness.compact(messages, recent_tail_count=2)

        # Assert — recent message IDs cover indices 2 and 3
        assert len(compacted.recent_message_ids) == 2

    def test_04_unit_03_context_03_compactor_08_empty_middle_when_all_tail(
        self,
        make_compactor: Callable[..., CompactorHarness],
        make_messages: Callable[..., list[Message]],
    ) -> None:
        # Arrange — all messages are head or tail, no middle
        harness = make_compactor()
        messages = make_messages(
            ("user", "task"),
            ("assistant", "recent"),
        )

        # Act — recent_tail_count covers all non-head messages
        compacted = harness.compact(messages, recent_tail_count=1)

        # Assert — no middle to compact
        assert compacted.source_message_index_start == -1
        assert compacted.source_message_index_end == -1
