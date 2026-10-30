#!/usr/bin/env python3
"""
Per-tool execution recording for duplicate detection.

Each function stores the snapshot needed to detect a later duplicate call.

    Important Functions:
        _record_read(state, arguments, llm_call_index) -> None
        _record_list_dir(state, arguments, llm_call_index) -> None
        _record_search(state, tool_name, arguments, llm_call_index) -> None
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from histrel.core.duplicate_guard.keys import (
    _list_dir_key,
    _read_key,
    _search_key,
)
from histrel.core.duplicate_guard.snapshots import (
    _directory_listing,
    _file_digest,
)
from histrel.core.duplicate_guard.types import GuardState


def _record_read(
    state: GuardState,
    arguments: dict[str, Any],
    llm_call_index: int,
) -> None:
    """Record a read execution and the file's digest.

    Args:
        state (GuardState): Guard bookkeeping.
        arguments (dict[str, Any]): Parsed read tool arguments.
        llm_call_index (int): Index of the LLM call that produced it.
    """
    key = _read_key(arguments)
    if key is None:
        return
    digest = _file_digest(Path(key[0]))
    if digest is None:
        return
    state.read_entries[key] = (llm_call_index, digest)


def _record_list_dir(
    state: GuardState,
    arguments: dict[str, Any],
    llm_call_index: int,
) -> None:
    """Record a list_dir execution and the directory's listing.

    Args:
        state (GuardState): Guard bookkeeping.
        arguments (dict[str, Any]): Parsed list_dir tool arguments.
        llm_call_index (int): Index of the LLM call that produced it.
    """
    key = _list_dir_key(arguments)
    if key is None:
        return
    listing = _directory_listing(Path(key))
    if listing is None:
        return
    state.list_dir_entries[key] = (llm_call_index, listing)


def _record_search(
    state: GuardState,
    tool_name: str,
    arguments: dict[str, Any],
    llm_call_index: int,
) -> None:
    """Record a search or glob execution and the workspace change sequence.

    Args:
        state (GuardState): Guard bookkeeping.
        tool_name (str): Tool name ("search" or "glob").
        arguments (dict[str, Any]): Parsed tool arguments.
        llm_call_index (int): Index of the LLM call that produced it.
    """
    key = _search_key(tool_name, arguments)
    if key is None:
        return
    state.search_entries[key] = (llm_call_index, state.workspace_change_seq)
