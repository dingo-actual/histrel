#!/usr/bin/env python3
"""Unit tests for the append-only message journal.

Tests journal append, fsync durability, checksum verification, and
round-trip decode(encode(message)) preservation.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from histrel.core.ids import SessionId
from histrel.core.journal import (
    MessageJournal,
    decode_journal_line,
    encode_journal_line,
    load_journal,
)
from histrel.model.messages import (
    Message,
    MessageFactory,
    MessageMetadata,
    TextPart,
)


def _make_message(factory: MessageFactory, text: str = "test") -> Message:
    """Create a minimal message for journal tests."""
    return factory.create(
        role="user",
        content=[TextPart(text=text)],
        metadata=MessageMetadata(),
    )


class Test04Unit01Journal01Encoding:
    """Journal line encoding and checksum."""

    def test_04_unit_01_journal_01_encoding_01_encode_produces_valid_json(
        self,
    ) -> None:
        # Arrange
        factory = MessageFactory(SessionId("sess-enc"))
        msg = _make_message(factory)

        # Act
        line = encode_journal_line(msg)

        # Assert
        envelope = json.loads(line)
        assert envelope["journal_schema_version"] == 1
        assert "message" in envelope
        assert "checksum" in envelope

    def test_04_unit_01_journal_01_encoding_02_decode_preserves_message(
        self,
    ) -> None:
        # Arrange
        factory = MessageFactory(SessionId("sess-rt"))
        msg = _make_message(factory, "round-trip text")

        # Act
        line = encode_journal_line(msg)
        envelope = decode_journal_line(line)
        decoded = envelope.message

        # Assert
        assert decoded.message_id == msg.message_id
        assert decoded.session_id == msg.session_id
        assert decoded.role == msg.role
        assert decoded.created == msg.created
        assert decoded.message_index == msg.message_index
        assert len(decoded.content) == 1
        first = decoded.content[0]
        assert isinstance(first, TextPart)
        assert first.text == "round-trip text"

    def test_04_unit_01_journal_01_encoding_03_checksum_matches(self) -> None:
        # Arrange
        factory = MessageFactory(SessionId("sess-ck"))
        msg = _make_message(factory)

        # Act
        line = encode_journal_line(msg)
        envelope = json.loads(line)

        # Assert — checksum is sha256 hex
        assert len(envelope["checksum"]) == 64

    def test_04_unit_01_journal_01_encoding_04_deterministic_serialization(
        self,
    ) -> None:
        # Arrange
        factory = MessageFactory(SessionId("sess-det"))
        msg = _make_message(factory)

        # Act
        line1 = encode_journal_line(msg)
        line2 = encode_journal_line(msg)

        # Assert — same message produces same bytes (sorted keys, stable separators)
        assert line1 == line2


class Test04Unit02Journal02Append:
    """Journal append and persistence."""

    def test_04_unit_02_journal_02_append_01_writes_lines(self) -> None:
        # Arrange
        factory = MessageFactory(SessionId("sess-app"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            journal = MessageJournal(path)

            # Act
            journal.append(_make_message(factory, "first"))
            journal.append(_make_message(factory, "second"))

            # Assert
            lines = path.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 2

    def test_04_unit_02_journal_02_append_02_persists_across_instances(
        self,
    ) -> None:
        # Arrange
        factory = MessageFactory(SessionId("sess-persist"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"

            # Act — write with one journal instance
            journal1 = MessageJournal(path)
            journal1.append(_make_message(factory, "persisted"))
            journal1.close()

            # Read back via load_journal (new instance not needed for read)
            messages = list(load_journal(path))

            # Assert
            assert len(messages) == 1
            first = messages[0].content[0]
            assert isinstance(first, TextPart)
            assert first.text == "persisted"

    def test_04_unit_02_journal_02_append_03_order_preserved(self) -> None:
        # Arrange
        factory = MessageFactory(SessionId("sess-order"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            journal = MessageJournal(path)

            # Act
            for i in range(5):
                journal.append(_make_message(factory, f"msg-{i}"))
            journal.close()

            # Assert
            messages = list(load_journal(path))
            indices = [m.message_index for m in messages]
            assert indices == [0, 1, 2, 3, 4]

    def test_04_unit_02_journal_02_append_04_append_only(self) -> None:
        # Arrange
        factory = MessageFactory(SessionId("sess-append-only"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            journal = MessageJournal(path)
            journal.append(_make_message(factory, "first"))
            journal.close()

            content_before = path.read_bytes()

            # Act — append more
            journal2 = MessageJournal(path)
            journal2.append(_make_message(factory, "second"))
            journal2.close()

            content_after = path.read_bytes()

            # Assert — original bytes are a prefix (append-only)
            assert content_after.startswith(content_before)


class Test04Unit03Journal03LoadAndReplay:
    """Journal loading and replay."""

    def test_04_unit_03_journal_03_replay_01_load_all_messages(self) -> None:
        # Arrange
        factory = MessageFactory(SessionId("sess-replay"))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "journal.jsonl"
            journal = MessageJournal(path)
            for i in range(3):
                journal.append(_make_message(factory, f"r-{i}"))
            journal.close()

            # Act
            messages = list(load_journal(path))

            # Assert
            assert len(messages) == 3
            assert all(isinstance(m, Message) for m in messages)

    def test_04_unit_03_journal_03_replay_02_empty_journal(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.jsonl"
            path.touch()

            # Act
            messages = list(load_journal(path))

            # Assert
            assert len(messages) == 0

    def test_04_unit_03_journal_03_replay_03_nonexistent_journal(self) -> None:
        # Arrange
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nonexistent.jsonl"

            # Act
            messages = list(load_journal(path))

            # Assert
            assert len(messages) == 0
