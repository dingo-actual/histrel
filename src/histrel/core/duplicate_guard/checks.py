#!/usr/bin/env python3
"""
Per-tool duplicate-detection checks.

Each function inspects the guard state for a specific tool and returns a
verdict when the call repeats a recorded execution.

    Important Functions:
        _check_read(state, arguments, llm_call_index) -> DuplicateVerdict | None
        _check_list_dir(state, arguments, llm_call_index) -> DuplicateVerdict | None
        _check_search(state, tool_name, arguments, llm_call_index) -> DuplicateVerdict | None
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
    _within_window,
)
from histrel.core.duplicate_guard.types import (
    DuplicateVerdict,
    GuardState,
)


def _check_read(
    state: GuardState,
    arguments: dict[str, Any],
    llm_call_index: int,
) -> DuplicateVerdict | None:
    """Check a read call against recorded reads.

    Args:
        state (GuardState): Guard bookkeeping.
        arguments (dict[str, Any]): Parsed read tool arguments.
        llm_call_index (int): Index of the current LLM call.

    Returns:
        (DuplicateVerdict | None) Verdict when the read repeats, else None.
    """
    key = _read_key(arguments)
    if key is None:
        return None
    entry = state.read_entries.get(key)
    if entry is None:
        return None
    recorded_call, digest = entry
    if not _within_window(recorded_call, llm_call_index, state.window):
        return None
    if _file_digest(Path(key[0])) != digest:
        return None
    start = key[1]
    end = key[2] if key[2] is not None else "EOF"
    return DuplicateVerdict(
        reason=(
            f"read of {key[0]} lines {start}-{end} repeated within "
            f"{state.window} LLM calls without the file being edited"
        )
    )


def _check_list_dir(
    state: GuardState,
    arguments: dict[str, Any],
    llm_call_index: int,
) -> DuplicateVerdict | None:
    """Check a list_dir call against recorded listings.

    Args:
        state (GuardState): Guard bookkeeping.
        arguments (dict[str, Any]): Parsed list_dir tool arguments.
        llm_call_index (int): Index of the current LLM call.

    Returns:
        (DuplicateVerdict | None) Verdict when the listing repeats, else None.
    """
    key = _list_dir_key(arguments)
    if key is None:
        return None
    entry = state.list_dir_entries.get(key)
    if entry is None:
        return None
    recorded_call, listing = entry
    if not _within_window(recorded_call, llm_call_index, state.window):
        return None
    if _directory_listing(Path(key)) != listing:
        return None
    return DuplicateVerdict(
        reason=(
            f"list_dir of {key} repeated within {state.window} LLM "
            "calls without files being added or removed"
        )
    )


def _check_search(
    state: GuardState,
    tool_name: str,
    arguments: dict[str, Any],
    llm_call_index: int,
) -> DuplicateVerdict | None:
    """Check a search or glob call against recorded searches.

    Args:
        state (GuardState): Guard bookkeeping.
        tool_name (str): Tool name ("search" or "glob").
        arguments (dict[str, Any]): Parsed tool arguments.
        llm_call_index (int): Index of the current LLM call.

    Returns:
        (DuplicateVerdict | None) Verdict when the search repeats, else None.
    """
    key = _search_key(tool_name, arguments)
    if key is None:
        return None
    entry = state.search_entries.get(key)
    if entry is None:
        return None
    recorded_call, change_seq = entry
    if not _within_window(recorded_call, llm_call_index, state.window):
        return None
    if change_seq != state.workspace_change_seq:
        return None
    path_str = arguments.get("path") or "."
    return DuplicateVerdict(
        reason=(
            f"{tool_name} for pattern {key[1]!r} in {path_str} repeated "
            f"within {state.window} LLM calls without files being changed"
        )
    )
