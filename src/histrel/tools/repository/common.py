#!/usr/bin/env python3
"""
Shared helpers for the repository tools package.

    Important Functions:
        _display_path(path) -> str: Display path for tool output headers.
"""

from __future__ import annotations

from pathlib import Path


def _display_path(path: Path) -> str:
    """Return a path for display in tool output headers.

    Paths under CWD are shown relative to CWD. Paths outside CWD (for
    example, allowed absolute paths under /tmp) are shown as-is, so
    header construction never raises for a valid resolved path.

    Args:
        path (Path): Resolved path to display.

    Returns:
        (str) Posix-style display path.
    """
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()
