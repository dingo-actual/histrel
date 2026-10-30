#!/usr/bin/env python3
"""Unit tests for benchmark agent identifiers and clock utilities.

Tests the NewType identifiers, UUID generation, and monotonic/UTC clock helpers
that form the foundation of the agent's message and session model.
"""

from __future__ import annotations

import time

from histrel.core.clock import (
    monotonic_now,
    utc_now,
    utc_timestamp,
)
from histrel.core.ids import (
    MessageId,
    SessionId,
    new_message_id,
)


class Test04Unit01Ids01Generation:
    """Identifier generation and type semantics."""

    def test_04_unit_01_ids_01_generation_01_message_id_is_str(self) -> None:
        # Act
        mid = new_message_id()

        # Assert
        assert isinstance(mid, str)
        assert len(mid) > 0

    def test_04_unit_01_ids_01_generation_02_message_id_unique(self) -> None:
        # Act
        a = new_message_id()
        b = new_message_id()

        # Assert
        assert a != b

    def test_04_unit_01_ids_01_generation_05_newtype_is_str_subclass(self) -> None:
        # Assert — NewType produces str at runtime
        sid: SessionId = SessionId("test-session")
        assert isinstance(sid, str)

        mid: MessageId = MessageId("msg-1")
        assert isinstance(mid, str)


class Test04Unit02Clock01Timestamps:
    """Clock utilities for timing and correlation."""

    def test_04_unit_02_clock_01_timestamps_01_utc_now_returns_float(self) -> None:
        # Act
        ts = utc_now()

        # Assert
        assert isinstance(ts, float)
        assert ts > 0

    def test_04_unit_02_clock_01_timestamps_02_utc_timestamp_returns_int(self) -> None:
        # Act
        ts = utc_timestamp()

        # Assert
        assert isinstance(ts, int)
        assert ts > 0

    def test_04_unit_02_clock_01_timestamps_03_monotonic_increases(self) -> None:
        # Act
        a = monotonic_now()
        time.sleep(0.001)
        b = monotonic_now()

        # Assert
        assert b > a

    def test_04_unit_02_clock_01_timestamps_04_utc_timestamp_approximate(
        self,
    ) -> None:
        # Arrange
        expected = int(time.time())

        # Act
        actual = utc_timestamp()

        # Assert — within 2 seconds of system time
        assert abs(actual - expected) <= 2
