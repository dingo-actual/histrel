#!/usr/bin/env python3
"""
Core patching engine for content-addressable search/replace operations.

Implements the matching and indentation logic from the search/replace
specification. Uses a 3-tier matching strategy (exact, normalized, fuzzy)
to locate search blocks and applies replacements with intelligent
indentation reconstruction.

One local divergence from the specification: exact matches (whitespace
included) take precedence over the normalized tier, and exact-match
replacements are applied verbatim.

This facade re-exports every name that was importable from the former
``histrel.tools.patch_engine`` module so import paths
stay stable.

    Important Classes:
        PatchError: Base error for patching operations.
        MatchNotFoundError: Raised when search block not found.
        AmbiguousMatchError: Raised when multiple matches found.
        OverlapError: Raised when patch ranges overlap.
        BlockMatcher: Locates text blocks using 3-tier matching strategy.
        Indenter: Handles indentation reconstruction for replacement blocks.
        PatchTransaction: Manages atomic patch application to files.

    Important Functions:
        apply_patch(file_path, changes) -> str: Apply patch and return result.
"""

from __future__ import annotations

from histrel.tools.patch_engine.errors import (
    AmbiguousMatchError,
    MatchNotFoundError,
    OverlapError,
    PatchError,
)
from histrel.tools.patch_engine.indenter import Indenter
from histrel.tools.patch_engine.matcher import BlockMatcher
from histrel.tools.patch_engine.transaction import (
    PatchTransaction,
    apply_patch,
)

__all__ = [
    "AmbiguousMatchError",
    "BlockMatcher",
    "Indenter",
    "MatchNotFoundError",
    "OverlapError",
    "PatchError",
    "PatchTransaction",
    "apply_patch",
]
