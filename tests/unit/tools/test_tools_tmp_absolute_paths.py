#!/usr/bin/env python3
"""Unit tests for absolute /tmp path support across the tools."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from histrel.tools.repository import (
    EditFileTool,
    GlobTool,
    ListDirTool,
    ReadFileTool,
    SearchTool,
)
from tests.unit.tools.helpers import (
    edit_block,
    make_context,
)


class Test04Unit09Tools09TmpAbsolutePaths:
    """Tools accept absolute paths under /tmp in addition to CWD-relative paths."""

    def test_04_unit_09_tools_09_tmp_01_read_tmp_absolute_path(
        self, isolated_workspace: Path
    ) -> None:
        """read() accepts an absolute /tmp path outside the workspace."""
        with tempfile.TemporaryDirectory(dir="/tmp") as scratch:
            scratch_note = Path(scratch) / "note.txt"
            scratch_note.write_text("scratch data\n", encoding="utf-8")

            # Act — CWD is the isolated workspace, the file is elsewhere in /tmp
            execution = ReadFileTool().execute(
                {"path": str(scratch_note)}, make_context()
            )

            # Assert — the header repeats the absolute /tmp path argument
            assert execution.status == "ok"
            assert str(execution.output).startswith(
                f"Contents of file {scratch_note} (lines 1-1)\n---\n"
            )
            assert "scratch data" in str(execution.output)

    def test_04_unit_09_tools_09_tmp_02_edit_tmp_absolute_path(
        self, isolated_workspace: Path
    ) -> None:
        """edit() accepts an absolute /tmp path outside the workspace."""
        with tempfile.TemporaryDirectory(dir="/tmp") as scratch:
            scratch_note = Path(scratch) / "note.txt"
            scratch_note.write_text("value = 1\n", encoding="utf-8")

            # Act
            execution = EditFileTool().execute(
                {
                    "file_path": str(scratch_note),
                    "file_edits": [
                        edit_block("value = 1", "value = 2"),
                    ],
                },
                make_context(),
            )

            # Assert
            assert execution.status == "ok"
            assert scratch_note.read_text(encoding="utf-8") == "value = 2\n"

    def test_04_unit_09_tools_09_tmp_03_list_dir_tmp_absolute_path(
        self, isolated_workspace: Path
    ) -> None:
        """list_dir() accepts an absolute /tmp directory outside the workspace."""
        with tempfile.TemporaryDirectory(dir="/tmp") as scratch:
            (Path(scratch) / "scratch_file.txt").touch()

            # Act
            execution = ListDirTool().execute({"path": scratch}, make_context())

            # Assert — the header repeats the absolute /tmp path argument
            assert execution.status == "ok"
            assert str(execution.output).startswith(
                f"Contents of directory {scratch}\n---\n"
            )
            assert "scratch_file.txt" in str(execution.output)

    @pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
    def test_04_unit_09_tools_09_tmp_04_search_tmp_absolute_path(
        self, isolated_workspace: Path
    ) -> None:
        """search() accepts an absolute /tmp directory outside the workspace."""
        with tempfile.TemporaryDirectory(dir="/tmp") as scratch:
            (Path(scratch) / "a.py").write_text("needle = True\n", encoding="utf-8")

            # Act
            execution = SearchTool().execute(
                {"pattern": "needle", "path": scratch}, make_context()
            )

            # Assert — the header repeats the pattern and /tmp path arguments
            assert execution.status == "ok"
            assert str(execution.output).startswith(
                f"Search for string pattern 'needle' in {scratch}\n---\n"
            )
            assert "needle" in str(execution.output)

    @pytest.mark.skipif(shutil.which("fd") is None, reason="fd not installed")
    def test_04_unit_09_tools_09_tmp_05_glob_tmp_absolute_path(
        self, isolated_workspace: Path
    ) -> None:
        """glob() accepts an absolute /tmp directory outside the workspace."""
        with tempfile.TemporaryDirectory(dir="/tmp") as scratch:
            (Path(scratch) / "target.py").write_text("x = 1\n", encoding="utf-8")

            # Act
            execution = GlobTool().execute(
                {"pattern": "*.py", "path": scratch}, make_context()
            )

            # Assert — the header repeats the pattern and /tmp path arguments
            assert execution.status == "ok"
            assert str(execution.output).startswith(
                f"Search for file pattern '*.py' in {scratch}\n---\n"
            )
            assert "target.py" in str(execution.output)
