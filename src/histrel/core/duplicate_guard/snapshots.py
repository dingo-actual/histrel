#!/usr/bin/env python3
"""
Filesystem snapshots used for duplicate detection.

    Important Functions:
        _file_digest(path) -> str | None: SHA-256 digest of a file.
        _directory_listing(path) -> _DirectoryListing | None: One directory level.
        _within_window(recorded_call, current_call, window) -> bool: Window test.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from histrel.core.duplicate_guard.types import (
    _DirectoryListing,
)


def _file_digest(path: Path) -> str | None:
    """Return the SHA-256 digest of a file.

    Args:
        path (Path): File to digest.

    Returns:
        (str | None) Hex digest, or None when the file is unreadable.
    """
    try:
        data = path.read_bytes()
    except OSError:
        return None
    return hashlib.sha256(data).hexdigest()


def _directory_listing(path: Path) -> _DirectoryListing | None:
    """Return sorted (name, is_dir) entries for one directory level.

    Args:
        path (Path): Directory to snapshot.

    Returns:
        (_DirectoryListing | None) Directory entries, or None when the
        directory is unreadable.
    """
    try:
        with os.scandir(path) as entries:
            scanned = sorted(entries, key=lambda entry: entry.name)
    except OSError:
        return None
    listing: list[tuple[str, bool]] = []
    for entry in scanned:
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            return None
        listing.append((entry.name, is_dir))
    return tuple(listing)


def _within_window(recorded_call: int, current_call: int, window: int) -> bool:
    """Check whether a recorded call is within the recall window.

    Args:
        recorded_call (int): LLM call index of the recorded execution.
        current_call (int): LLM call index of the checked call.
        window (int): Number of LLM calls duplicates stay detectable.

    Returns:
        (bool) True when fewer than `window` calls separate them.
    """
    return current_call - recorded_call < window
