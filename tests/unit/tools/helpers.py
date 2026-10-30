#!/usr/bin/env python3
"""Shared helpers for the tool unit tests."""

from __future__ import annotations

from pathlib import Path

from histrel.core.session_context import ModelInfo, SessionContext
from histrel.tools.tool import ToolContext

# Repository root: the nearest ancestor that is neither /tmp nor CWD for the
# containment tests.
REPO_ROOT = Path(__file__).resolve().parents[3]


def make_context() -> ToolContext:
    """Create a ToolContext for tests."""
    return ToolContext(
        session=SessionContext(
            session_id="test-session",
            task_id="test-task",
            model=ModelInfo(provider="openai", name="gpt-4o"),
        ),
    )


def edit_block(search: str, replace: str) -> str:
    """Build a SEARCH/REPLACE edit block string for edit-tool arguments.

    The block uses all four delimiting markers; the marker lines' own
    line terminators are not part of the block content.
    """
    search_text = search if not search or search.endswith("\n") else search + "\n"
    replace_text = replace if not replace or replace.endswith("\n") else replace + "\n"
    return (
        "/***SEARCH***/\n"
        f"{search_text}"
        "/***END_SEARCH***/\n"
        "/***REPLACE***/\n"
        f"{replace_text}"
        "/***END_REPLACE***/"
    )
