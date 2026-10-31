#!/usr/bin/env python3
"""
Token ledger for the benchmark agent.

Accumulates actual token usage from model responses. Does not approximate
tokens with a tokenizer — the ledger only records real usage data returned
by the provider in streaming response chunks.

    Important Classes:
        LedgerEntry: A single token usage record.
        TokenLedger: Cumulative token usage tracker.

    Important Functions:
        entry_payload: Serialize a ledger entry for artifact output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from histrel.model.types import Usage


class LedgerEntry(BaseModel):
    """A single token usage record.

    Instance Attributes:
        message_index (int): Message index when the usage was recorded.
        usage (Usage): Token usage from the model response.
        provider_request_id (str | None): Provider request ID for audit.
        call_kind (str): Call type ("model", "compaction",
            or "routine").
    """

    message_index: int
    usage: Usage
    provider_request_id: str | None = None
    call_kind: str = "model"


def entry_payload(entry: LedgerEntry) -> dict[str, Any]:
    """Serialize a ledger entry for a JSONL usage log.

    Args:
        entry (LedgerEntry): The entry to serialize.

    Returns:
        (dict[str, Any]) JSON-ready payload for one usage record.
    """
    return {
        "message_index": entry.message_index,
        "provider_request_id": entry.provider_request_id,
        "call_kind": entry.call_kind,
        "usage": entry.usage.model_dump(),
    }


class TokenLedger:
    """Cumulative token usage tracker.

    Records real usage from model responses. Compaction calls are tracked
    separately from model calls.

    Instance Attributes:
        _entries (list[LedgerEntry]): Recorded usage entries.
        _total_input (int): Cumulative input tokens.
        _total_output (int): Cumulative output tokens.
        _total_reasoning (int): Cumulative reasoning tokens.
        _total_cached (int): Cumulative cached tokens.
        _last_input (int): Most recent input token count.
        _usage_log_path (Path | None): Append-only log written on every
            recorded entry, so usage survives a session that never
            finalizes.
    """

    def __init__(self, usage_log_path: Path | None = None) -> None:
        """Initialize an empty token ledger.

        Args:
            usage_log_path (Path | None): Append-only JSONL log that
                receives each entry as it is recorded.
        """
        self._usage_log_path = usage_log_path
        self._entries: list[LedgerEntry] = []
        self._total_input: int = 0
        self._total_output: int = 0
        self._total_reasoning: int = 0
        self._total_cached: int = 0
        self._last_input: int = 0

    def record(
        self,
        message_index: int,
        usage: Usage,
        provider_request_id: str | None = None,
        call_kind: str = "model",
    ) -> None:
        """Record a token usage entry.

        Args:
            message_index (int): Message index when the usage was recorded.
            usage (Usage): Token usage from the model response.
            provider_request_id (str | None): Provider request ID.
            call_kind (str): Call type ("model", "compaction",
                or "routine").
        """
        entry = LedgerEntry(
            message_index=message_index,
            usage=usage,
            provider_request_id=provider_request_id,
            call_kind=call_kind,
        )
        self._entries.append(entry)
        self._total_input += usage.input
        self._total_output += usage.output
        self._total_reasoning += usage.reasoning
        self._total_cached += usage.cached
        if call_kind == "model":
            self._last_input = usage.input
        self._append(entry)

    def _append(self, entry: LedgerEntry) -> None:
        """Append one entry to the usage log, if a log is configured.

        Args:
            entry (LedgerEntry): The entry to persist.
        """
        if self._usage_log_path is None:
            return
        try:
            self._usage_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._usage_log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(entry_payload(entry)) + "\n")
        except OSError as exc:
            logger.bind(
                component="histrel.context.token_ledger",
                event="usage.persist_failed",
            ).warning(
                "Failed to append usage to the usage log",
                path=str(self._usage_log_path),
                error=str(exc),
            )

    @property
    def total_input(self) -> int:
        """Return cumulative input tokens across all calls."""
        return self._total_input

    @property
    def total_output(self) -> int:
        """Return cumulative output tokens across all calls."""
        return self._total_output

    @property
    def total_reasoning(self) -> int:
        """Return cumulative reasoning tokens across all calls."""
        return self._total_reasoning

    @property
    def total_cached(self) -> int:
        """Return cumulative cached tokens across all calls."""
        return self._total_cached

    @property
    def last_input(self) -> int:
        """Return the most recent model call input token count."""
        return self._last_input

    @property
    def entry_count(self) -> int:
        """Return the number of recorded entries."""
        return len(self._entries)

    @property
    def entries(self) -> list[LedgerEntry]:
        """Return a copy of all recorded entries."""
        return list(self._entries)

    @property
    def compaction_input(self) -> int:
        """Return cumulative input tokens from compaction calls only."""
        return sum(e.usage.input for e in self._entries if e.call_kind == "compaction")

    @property
    def model_input(self) -> int:
        """Return cumulative input tokens from model calls only."""
        return sum(e.usage.input for e in self._entries if e.call_kind == "model")
