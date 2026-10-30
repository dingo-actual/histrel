#!/usr/bin/env python3
"""Unit tests for the token ledger.

Tests cumulative token tracking from real model response Usage objects.
The ledger accumulates actual input/output/reasoning/cached tokens
from model responses rather than approximating with a tokenizer.
"""

from __future__ import annotations

from histrel.context.token_ledger import (
    TokenLedger,
)
from histrel.model.types import Usage


def _make_usage(
    input: int = 1000,
    output: int = 500,
    reasoning: int = 0,
    cached: int = 0,
) -> Usage:
    return Usage(
        input=input,
        output=output,
        reasoning=reasoning,
        total=input + output,
        cached=cached,
    )


class Test04Unit01Ledger01Record:
    """Token ledger entry recording."""

    def test_04_unit_01_ledger_01_record_01_single_entry(self) -> None:
        # Arrange
        ledger = TokenLedger()

        # Act
        ledger.record(
            message_index=0,
            usage=_make_usage(input=1000, output=500),
            provider_request_id="req-1",
        )

        # Assert
        assert ledger.total_input == 1000
        assert ledger.total_output == 500
        assert ledger.total_reasoning == 0
        assert ledger.total_cached == 0

    def test_04_unit_01_ledger_01_record_02_multiple_entries_accumulate(
        self,
    ) -> None:
        # Arrange
        ledger = TokenLedger()

        # Act
        ledger.record(0, _make_usage(input=1000, output=500), "req-1")
        ledger.record(1, _make_usage(input=2000, output=800), "req-2")
        ledger.record(2, _make_usage(input=1500, output=300), "req-3")

        # Assert
        assert ledger.total_input == 4500
        assert ledger.total_output == 1600
        assert ledger.entry_count == 3

    def test_04_unit_01_ledger_01_record_03_reasoning_accumulates(self) -> None:
        # Arrange
        ledger = TokenLedger()

        # Act
        ledger.record(
            0,
            _make_usage(input=1000, output=500, reasoning=200),
            "req-1",
        )
        ledger.record(
            1,
            _make_usage(input=2000, output=800, reasoning=400),
            "req-2",
        )

        # Assert
        assert ledger.total_reasoning == 600

    def test_04_unit_01_ledger_01_record_04_cached_accumulates(self) -> None:
        # Arrange
        ledger = TokenLedger()

        # Act
        ledger.record(
            0,
            _make_usage(input=1000, output=500, cached=300),
            "req-1",
        )

        # Assert
        assert ledger.total_cached == 300

    def test_04_unit_01_ledger_01_record_05_last_input_tracked(self) -> None:
        # Arrange
        ledger = TokenLedger()

        # Act
        ledger.record(0, _make_usage(input=1000, output=500), "req-1")
        ledger.record(1, _make_usage(input=5000, output=1000), "req-2")

        # Assert — last input is the most recent, not cumulative
        assert ledger.last_input == 5000

    def test_04_unit_01_ledger_01_record_06_empty_ledger(self) -> None:
        # Act
        ledger = TokenLedger()

        # Assert
        assert ledger.total_input == 0
        assert ledger.total_output == 0
        assert ledger.total_reasoning == 0
        assert ledger.total_cached == 0
        assert ledger.last_input == 0
        assert ledger.entry_count == 0


class Test04Unit02Ledger02Entries:
    """Token ledger entry retrieval."""

    def test_04_unit_02_ledger_02_entries_01_returns_entries(self) -> None:
        # Arrange
        ledger = TokenLedger()
        ledger.record(0, _make_usage(input=100, output=50), "req-1")
        ledger.record(1, _make_usage(input=200, output=80), "req-2")

        # Act
        entries = ledger.entries

        # Assert
        assert len(entries) == 2
        assert entries[0].message_index == 0
        assert entries[0].usage.input == 100
        assert entries[1].message_index == 1

    def test_04_unit_02_ledger_02_entries_02_returns_copy(self) -> None:
        # Arrange
        ledger = TokenLedger()
        ledger.record(0, _make_usage(input=100, output=50), "req-1")

        # Act
        entries = ledger.entries
        entries.clear()

        # Assert — mutating the returned list does not affect the ledger
        assert len(ledger.entries) == 1

    def test_04_unit_02_ledger_02_entries_03_entry_has_provider_id(self) -> None:
        # Arrange
        ledger = TokenLedger()
        ledger.record(0, _make_usage(), "req-correlation-123")

        # Act
        entries = ledger.entries

        # Assert
        assert entries[0].provider_request_id == "req-correlation-123"


class Test04Unit03Ledger03CompactionUsage:
    """Token ledger records compaction model calls."""

    def test_04_unit_03_ledger_03_compaction_01_records_with_call_kind(
        self,
    ) -> None:
        # Arrange
        ledger = TokenLedger()

        # Act
        ledger.record(
            message_index=5,
            usage=_make_usage(input=3000, output=1000),
            provider_request_id="req-compaction-1",
            call_kind="compaction",
        )

        # Assert
        entries = ledger.entries
        assert entries[0].call_kind == "compaction"
        assert ledger.total_input == 3000
        assert ledger.total_output == 1000

    def test_04_unit_03_ledger_03_compaction_02_separate_from_model_calls(
        self,
    ) -> None:
        # Arrange
        ledger = TokenLedger()
        ledger.record(0, _make_usage(input=1000, output=500), "req-1")
        ledger.record(
            5,
            _make_usage(input=3000, output=1000),
            "req-compaction",
            call_kind="compaction",
        )
        ledger.record(6, _make_usage(input=2000, output=800), "req-2")

        # Act
        compaction_input = ledger.compaction_input
        model_input = ledger.model_input

        # Assert — compaction usage is tracked separately
        assert compaction_input == 3000
        assert model_input == 1000 + 2000
        assert ledger.total_input == 1000 + 3000 + 2000
