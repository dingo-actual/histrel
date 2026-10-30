#!/usr/bin/env python3
"""
Repository tools package for the benchmark agent.

Provides tools for reading files, editing and creating files
(SEARCH/REPLACE with empty-search file creation), searching with
ripgrep, listing directories, and globbing with fd. All tools subclass
Tool (no mocks; real file and subprocess access).

This facade re-exports every name that was importable from the former
``histrel.tools.repository`` module so import paths
stay stable.

    Important Functions:
        _display_path: Display path for tool output headers.

    Important Classes:
        ReadFileTool: Read file contents.
        EditFileTool: Apply SEARCH/REPLACE patches to edit or create a single file.
        SearchTool: Search file contents with ripgrep.
        ListDirTool: List directory contents.
        GlobTool: Find files with fd.
"""

from __future__ import annotations

from histrel.tools.repository.bash import BashTool
from histrel.tools.repository.common import _display_path
from histrel.tools.repository.edit_file import EditFileTool
from histrel.tools.repository.listing import GlobTool, ListDirTool
from histrel.tools.repository.read_file import ReadFileTool
from histrel.tools.repository.search import SearchTool

__all__ = [
    "BashTool",
    "EditFileTool",
    "GlobTool",
    "ListDirTool",
    "ReadFileTool",
    "SearchTool",
    "_display_path",
]
