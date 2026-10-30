#!/usr/bin/env python3
"""
Error hierarchy for the patching engine.

    Important Classes:
        PatchError: Base error for patching operations.
        MatchNotFoundError: Raised when search block not found.
        AmbiguousMatchError: Raised when multiple matches found.
        OverlapError: Raised when patch ranges overlap.
"""

from __future__ import annotations


class PatchError(Exception):
    """Base exception for all patching errors."""

    def __init__(self, path: str, msg: str) -> None:
        super().__init__(f"[{path}] {msg}")
        self.path = path
        self.msg = msg


class MatchNotFoundError(PatchError):
    """Raised when a search block cannot be found in the file."""


class AmbiguousMatchError(PatchError):
    """Raised when a search block matches multiple locations."""


class OverlapError(PatchError):
    """Raised when patch ranges overlap."""
