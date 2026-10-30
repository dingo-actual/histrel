#!/usr/bin/env python3
"""
Shared types and bookkeeping state for duplicate-call detection.

    Important Classes:
        DuplicateVerdict: Duplicate-call decision with an explanation.
        GuardState: Mutable duplicate-detection bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# A recorded execution stays reachable for duplicate detection for
# this many LLM calls, counting the call that produced it.
_RECALL_WINDOW = 5

_DirectoryListing = tuple[tuple[str, bool], ...]
_ReadKey = tuple[str, int, int | None]
_SearchKey = tuple[str, str, str, str | None]


@dataclass(frozen=True)
class DuplicateVerdict:
    """Duplicate-call decision for one tool call.

    Instance Attributes:
        reason (str): Explanation of the matched duplicate rule.
    """

    reason: str


@dataclass
class GuardState:
    """Mutable bookkeeping shared by the duplicate-detection strategies.

    Instance Attributes:
        window (int): Configured recall window in LLM calls.
        read_entries (dict[_ReadKey, tuple[int, str]]): Read key to
            (LLM call index, file digest) for recorded reads.
        list_dir_entries (dict[str, tuple[int, _DirectoryListing]]):
            Resolved directory to (LLM call index, listing snapshot).
        search_entries (dict[_SearchKey, tuple[int, int]]): Search key
            to (LLM call index, workspace change sequence).
        workspace_change_seq (int): Counter bumped on successful
            edits and test runs.
    """

    window: int = _RECALL_WINDOW
    read_entries: dict[_ReadKey, tuple[int, str]] = field(default_factory=dict)
    list_dir_entries: dict[str, tuple[int, _DirectoryListing]] = field(
        default_factory=dict
    )
    search_entries: dict[_SearchKey, tuple[int, int]] = field(default_factory=dict)
    workspace_change_seq: int = 0
