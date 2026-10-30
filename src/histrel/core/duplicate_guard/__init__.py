#!/usr/bin/env python3
"""Duplicate tool call detection for the benchmark agent.

This facade re-exports every name that was importable from the former
``histrel.core.duplicate_guard`` module so import paths
stay stable.

Important Classes:
    DuplicateVerdict: Duplicate-call decision with an explanation.
    DuplicateCallGuard: Tracks executed calls to block duplicates.
"""

from __future__ import annotations

from histrel.core.duplicate_guard.checks import (
    _check_list_dir,
    _check_read,
    _check_search,
)
from histrel.core.duplicate_guard.guard import DuplicateCallGuard
from histrel.core.duplicate_guard.keys import (
    _list_dir_key,
    _read_key,
    _search_key,
)
from histrel.core.duplicate_guard.records import (
    _record_list_dir,
    _record_read,
    _record_search,
)
from histrel.core.duplicate_guard.snapshots import (
    _directory_listing,
    _file_digest,
    _within_window,
)
from histrel.core.duplicate_guard.types import (
    _RECALL_WINDOW,
    DuplicateVerdict,
    GuardState,
    _DirectoryListing,
    _ReadKey,
    _SearchKey,
)

__all__ = [
    "DuplicateCallGuard",
    "DuplicateVerdict",
    "GuardState",
    "_DirectoryListing",
    "_RECALL_WINDOW",
    "_ReadKey",
    "_SearchKey",
    "_check_list_dir",
    "_check_read",
    "_check_search",
    "_directory_listing",
    "_file_digest",
    "_list_dir_key",
    "_read_key",
    "_record_list_dir",
    "_record_read",
    "_record_search",
    "_search_key",
    "_within_window",
]
