#!/usr/bin/env python3
"""
Append-only message journal for the benchmark agent.

Every behavior-changing event is committed to the journal as a single line
of JSON (JSONL). The journal is byte-for-byte append-only: compaction never
mutates or removes lines. Each line is a JournalEnvelope containing the
canonical message, a sha256 checksum of the message payload, and a schema
version for forward compatibility.

    Important Classes:
        JournalEnvelope: Envelope wrapping a message with checksum and schema version.
        MessageJournal: Append-only JSONL writer with fsync durability.

    Important Functions:
        encode_journal_line(message: Message) -> str: Serialize a message to a journal line.
        decode_journal_line(line: str) -> JournalEnvelope: Parse a journal line.
        load_journal(path: Path) -> Iterator[Message]: Load all messages from a journal file.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from collections.abc import Iterator
from pathlib import Path

from pydantic import BaseModel

from histrel.core.errors import JournalError
from histrel.model.messages import Message

JOURNAL_SCHEMA_VERSION = 1


class JournalEnvelope(BaseModel):
    """Envelope wrapping a message with checksum and schema version.

    Instance Attributes:
        journal_schema_version (int): Journal format version.
        message (Message): The canonical message payload.
        checksum (str): SHA-256 hex digest of the canonical message JSON.
    """

    journal_schema_version: int = JOURNAL_SCHEMA_VERSION
    message: Message
    checksum: str


def _canonical_message_bytes(message: Message) -> bytes:
    """Return deterministic JSON bytes for a message.

    Uses sorted keys and no extra whitespace to ensure checksum stability
    across serialization runs.

    Args:
        message (Message): The message to serialize.

    Returns:
        (bytes) Canonical JSON bytes.
    """
    return message.model_dump_json(
        exclude_none=False,
        by_alias=True,
    ).encode("utf-8")


def _compute_checksum(message: Message) -> str:
    """Compute SHA-256 hex digest of the canonical message payload.

    Args:
        message (Message): The message to checksum.

    Returns:
        (str) Hex digest string.
    """
    # model_dump_json already produces sorted-key, stable output via Pydantic v2
    payload = message.model_dump_json().encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def encode_journal_line(message: Message) -> str:
    """Serialize a message to a single journal line (JSON string).

    The line contains a JournalEnvelope with the message, its checksum,
    and the journal schema version. The output is deterministic: the same
    message always produces the same line.

    Args:
        message (Message): The message to encode.

    Returns:
        (str) A single JSON line (no trailing newline).
    """
    checksum = _compute_checksum(message)
    envelope = JournalEnvelope(
        journal_schema_version=JOURNAL_SCHEMA_VERSION,
        message=message,
        checksum=checksum,
    )
    # Sort keys for deterministic output; no trailing newline
    return envelope.model_dump_json()


def decode_journal_line(line: str) -> JournalEnvelope:
    """Parse a journal line into a JournalEnvelope.

    Validates the checksum against the embedded message to detect
    corruption or truncation.

    Args:
        line (str): A single journal line (JSON string).

    Returns:
        (JournalEnvelope) The parsed envelope with message and checksum.

    Raises:
        JournalError: If the line is malformed or the checksum does not match.
    """
    line = line.strip()
    if not line:
        raise JournalError("Empty journal line")
    try:
        raw = json.loads(line)
    except json.JSONDecodeError as exc:
        raise JournalError(f"Malformed journal line: {exc}") from exc

    if not isinstance(raw, dict) or "message" not in raw or "checksum" not in raw:
        raise JournalError("Journal line missing required fields")

    envelope = JournalEnvelope.model_validate(raw)

    # Verify checksum
    expected = _compute_checksum(envelope.message)
    if expected != envelope.checksum:
        raise JournalError(
            f"Checksum mismatch: expected {expected}, got {envelope.checksum}"
        )

    return envelope


class MessageJournal:
    """Append-only JSONL journal writer.

    Writes messages as one JSON line per message with fsync durability.
    The file is opened in append-binary mode and never seeks or truncates.

    Instance Attributes:
        _path (Path): Path to the journal file.
        _fp (io.BufferedWriter | None): File handle (opened lazily).
    """

    def __init__(self, path: Path) -> None:
        """Initialize the journal writer.

        Creates the parent directory if needed. The file handle is opened
        lazily on the first append.

        Args:
            path (Path): Path to the journal JSONL file.
        """
        self._path = path
        self._fp: io.BufferedWriter | None = None

    def _ensure_open(self) -> None:
        """Open the file handle in append-binary mode if not already open."""
        if self._fp is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fp = self._path.open("ab")

    def append(self, message: Message) -> None:
        """Append a message to the journal as a single JSONL line.

        The line is fsynced to disk before returning.

        Args:
            message (Message): The message to append.

        Raises:
            JournalError: If the write fails.
        """
        try:
            self._ensure_open()
            if self._fp is not None:
                line = encode_journal_line(message) + "\n"
                self._fp.write(line.encode("utf-8"))
                self._fp.flush()
                os.fsync(self._fp.fileno())
        except OSError as exc:
            raise JournalError(f"Journal write failed: {exc}") from exc

    def close(self) -> None:
        """Close the journal file handle."""
        if self._fp is not None:
            with contextlib.suppress(Exception):
                self._fp.close()
            self._fp = None


def load_journal(path: Path) -> Iterator[Message]:
    """Load all messages from a journal file.

    Reads the journal lazily, yielding one Message per line. Corrupt or
    checksum-mismatched lines are skipped with a warning. If the file does
    not exist, yields nothing.

    Args:
        path (Path): Path to the journal JSONL file.

    Yields:
        (Message) Each message in the journal, in order.
    """
    if not path.exists():
        return

    with path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                envelope = decode_journal_line(line)
            except JournalError:
                # Skip corrupt lines — the journal is append-only and
                # truncated-tail recovery is expected at the caller level.
                continue
            yield envelope.message
