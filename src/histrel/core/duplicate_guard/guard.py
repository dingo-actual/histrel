#!/usr/bin/env python3
"""
Duplicate-call guard dispatcher.

    Important Classes:
        DuplicateCallGuard: Tracks executed calls to block duplicates.
"""

from __future__ import annotations

from typing import Any

from histrel.core.duplicate_guard.checks import (
    _check_list_dir,
    _check_read,
    _check_search,
)
from histrel.core.duplicate_guard.records import (
    _record_list_dir,
    _record_read,
    _record_search,
)
from histrel.core.duplicate_guard.snapshots import _within_window
from histrel.core.duplicate_guard.types import (
    _RECALL_WINDOW,
    DuplicateVerdict,
    GuardState,
)


class DuplicateCallGuard:
    """Blocks duplicate tool calls before execution.

    Implemented rules:
        * read: same resolved path and line range within the recall
          window without the file having been edited.
        * list_dir: same resolved directory within the recall window
          without files having been added or removed.
        * search/glob: same pattern over the same directory within the
          recall window without files having been changed, added, or
          removed.

    Instance Attributes:
        _state (GuardState): Mutable duplicate-detection bookkeeping.
    """

    def __init__(self, window: int = _RECALL_WINDOW) -> None:
        """Initialize an empty guard with no recorded executions.

        Args:
            window (int): LLM calls within which repeated calls are
                duplicates. 0 disables the window-based rules.

        Raises:
            ValueError: If window is negative.
        """
        if window < 0:
            raise ValueError("duplicate call window must not be negative")
        self._state = GuardState(window=window)

    @property
    def window(self) -> int:
        """Return the configured recall window in LLM calls."""
        return self._state.window

    def check(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        llm_call_index: int,
    ) -> DuplicateVerdict | None:
        """Check whether a tool call is a duplicate.

        Args:
            tool_name (str): Name of the tool to call.
            arguments (dict[str, Any]): Parsed tool arguments.
            llm_call_index (int): Index of the current LLM call.

        Returns:
            (DuplicateVerdict | None) Verdict when the call is a
            duplicate, None when it should execute.
        """
        state = self._state
        if tool_name == "read":
            return _check_read(state, arguments, llm_call_index)
        if tool_name == "list_dir":
            return _check_list_dir(state, arguments, llm_call_index)
        if tool_name in ("search", "glob"):
            return _check_search(state, tool_name, arguments, llm_call_index)
        return None

    def record(
        self,
        *,
        tool_name: str,
        arguments: dict[str, Any],
        llm_call_index: int,
    ) -> None:
        """Record a successful tool execution for duplicate detection.

        Args:
            tool_name (str): Name of the executed tool.
            arguments (dict[str, Any]): Parsed tool arguments.
            llm_call_index (int): Index of the LLM call that produced
                the execution.
        """
        self._prune(llm_call_index)
        state = self._state
        if tool_name == "edit":
            state.workspace_change_seq += 1
            return
        if tool_name == "read":
            _record_read(state, arguments, llm_call_index)
            return
        if tool_name == "list_dir":
            _record_list_dir(state, arguments, llm_call_index)
            return
        if tool_name in ("search", "glob"):
            _record_search(state, tool_name, arguments, llm_call_index)
            return

    def _prune(self, llm_call_index: int) -> None:
        """Drop entries that can no longer fall within the window."""
        state = self._state
        state.read_entries = {
            key: entry
            for key, entry in state.read_entries.items()
            if _within_window(entry[0], llm_call_index, state.window)
        }
        state.list_dir_entries = {
            key: entry
            for key, entry in state.list_dir_entries.items()
            if _within_window(entry[0], llm_call_index, state.window)
        }
        state.search_entries = {
            key: entry
            for key, entry in state.search_entries.items()
            if _within_window(entry[0], llm_call_index, state.window)
        }
