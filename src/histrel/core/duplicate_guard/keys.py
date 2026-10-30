#!/usr/bin/env python3
"""
Argument-to-key normalization for duplicate detection.

    Important Functions:
        _read_key(arguments) -> _ReadKey | None: Key for a read call.
        _list_dir_key(arguments) -> str | None: Key for a list_dir call.
        _search_key(tool_name, arguments) -> _SearchKey | None: Key for a search call.
"""

from __future__ import annotations

from typing import Any

from histrel.core.duplicate_guard.types import _ReadKey, _SearchKey
from histrel.tools.tool import resolve_path


def _read_key(arguments: dict[str, Any]) -> _ReadKey | None:
    """Build the duplicate-detection key for a read call.

    Args:
        arguments (dict[str, Any]): Parsed read tool arguments.

    Returns:
        (_ReadKey | None) Resolved path, start line, and end line, or
        None when the arguments are not usable.
    """
    path_str = arguments.get("path")
    if not isinstance(path_str, str) or not path_str:
        return None
    start = arguments.get("start_line", 1)
    end = arguments.get("end_line")
    if isinstance(start, bool) or not isinstance(start, int):
        return None
    if end is not None and (isinstance(end, bool) or not isinstance(end, int)):
        return None
    try:
        resolved = resolve_path(path_str)
    except (ValueError, FileNotFoundError):
        return None
    return (str(resolved), start, end)


def _list_dir_key(arguments: dict[str, Any]) -> str | None:
    """Build the duplicate-detection key for a list_dir call.

    Args:
        arguments (dict[str, Any]): Parsed list_dir tool arguments.

    Returns:
        (str | None) Resolved directory path, or None when the arguments
        are not usable.
    """
    path_str = arguments.get("path", ".")
    if not isinstance(path_str, str) or not path_str:
        return None
    try:
        return str(resolve_path(path_str))
    except (ValueError, FileNotFoundError):
        return None


def _search_key(tool_name: str, arguments: dict[str, Any]) -> _SearchKey | None:
    """Build the duplicate-detection key for a search or glob call.

    Args:
        tool_name (str): Tool name ("search" or "glob").
        arguments (dict[str, Any]): Parsed tool arguments.

    Returns:
        (_SearchKey | None) Tool name, pattern, resolved path, and glob
        filter, or None when the arguments are not usable.
    """
    pattern = arguments.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return None
    path_str = arguments.get("path")
    glob_filter = arguments.get("glob")
    if path_str is not None and not isinstance(path_str, str):
        return None
    if glob_filter is not None and not isinstance(glob_filter, str):
        return None
    if path_str:
        try:
            path_key = str(resolve_path(path_str))
        except (ValueError, FileNotFoundError):
            path_key = path_str
    else:
        path_key = ""
    return (tool_name, pattern, path_key, glob_filter)
