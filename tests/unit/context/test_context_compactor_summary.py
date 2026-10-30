#!/usr/bin/env python3
"""Unit tests for compactor summary and ledger side effects."""

from __future__ import annotations

from collections.abc import Callable

from histrel.model.messages import Message
from histrel.model.types import Usage
from tests.unit.context.helpers import CompactorHarness


class Test04Unit03Context03CompactorSummary:
    """Compaction summary generation and token-ledger recording."""

    def test_04_unit_03_context_03_compactor_09_summary_from_llm(
        self,
        make_compactor: Callable[..., CompactorHarness],
        make_messages: Callable[..., list[Message]],
    ) -> None:
        # Arrange — the compaction summary comes from the model
        harness = make_compactor(summary="LLM-generated compaction summary.")
        messages = make_messages(
            ("user", "task"),
            ("assistant", "step1"),
            ("assistant", "step2"),
        )

        # Act
        compacted = harness.compact(messages, recent_tail_count=1)

        # Assert — working_summary contains the LLM output
        assert "LLM-generated compaction summary" in compacted.working_summary

    def test_04_unit_03_context_03_compactor_10_ledger_records_compaction_call(
        self,
        make_compactor: Callable[..., CompactorHarness],
        make_messages: Callable[..., list[Message]],
    ) -> None:
        # Arrange — compaction usage is recorded in the token ledger
        harness = make_compactor(usage=Usage(input=500, output=200, total=700))
        messages = make_messages(
            ("user", "task"),
            ("assistant", "middle content"),
            ("assistant", "recent"),
        )

        # Act
        harness.compact(messages, recent_tail_count=1)

        # Assert — the compaction call is recorded with call_kind="compaction"
        assert harness.ledger.entry_count == 1
        assert harness.ledger.compaction_input == 500
        assert harness.ledger.model_input == 0
        entries = harness.ledger.entries
        assert entries[0].call_kind == "compaction"

    def test_04_unit_03_context_03_compactor_11_empty_middle_skips_llm_call(
        self,
        make_compactor: Callable[..., CompactorHarness],
        make_messages: Callable[..., list[Message]],
    ) -> None:
        # Arrange — no middle to compact, so no LLM call should happen
        harness = make_compactor()
        messages = make_messages(
            ("user", "task"),
            ("assistant", "recent"),
        )

        # Act — recent_tail_count=1 means all non-head is tail, no middle
        compacted = harness.compact(messages, recent_tail_count=1)

        # Assert — no LLM call was made, no ledger entry recorded
        assert harness.call_count == 0
        assert harness.ledger.entry_count == 0
        assert compacted.working_summary == ""
