#!/usr/bin/env python3
"""Unit tests for edit tool file-creation behavior."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from histrel.tools.tool import ToolExecution
from tests.unit.tools.helpers import edit_block


class Test04Unit10Tools10EditCreateFile:
    """edit tool file-creation behavior: empty search creates new files.

    When the file_path does not exist, exactly one edit is given, and
    that edit has an empty search block, edit creates a new file with
    the replace block as its content. These tests cover creation,
    error handling, and containment.
    """

    def test_04_unit_10_tools_10_edit_create_01_creates_new_file_with_content(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """An edit with empty search on a nonexistent file creates it."""
        # Act
        execution = run_edit("main.py", [edit_block("", "value = 1\n")])

        # Assert
        assert execution.status == "ok"
        assert "Created file" in str(execution.output)
        content = (isolated_workspace / "main.py").read_text(encoding="utf-8")
        assert content == "value = 1\n"

    def test_04_unit_10_tools_10_edit_create_02_creates_in_existing_subdirectory(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """A path inside an existing subdirectory resolves relative to CWD."""
        (isolated_workspace / "src").mkdir()

        # Act
        execution = run_edit("src/new_module.py", [edit_block("", "x = 1\n")])

        # Assert
        assert execution.status == "ok"
        content = (isolated_workspace / "src" / "new_module.py").read_text(
            encoding="utf-8"
        )
        assert content == "x = 1\n"

    def test_04_unit_10_tools_10_edit_create_03_empty_replace_creates_empty_file(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """An empty replace block creates an empty file."""
        # Act
        execution = run_edit("empty.py", [edit_block("", "")])

        # Assert
        assert execution.status == "ok"
        assert (isolated_workspace / "empty.py").read_bytes() == b""

    def test_04_unit_10_tools_10_edit_create_04_utf8_multibyte_content(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """Multibyte UTF-8 content round-trips through the edit create path."""
        # Act
        execution = run_edit("unicode.txt", [edit_block("", "café → ✓\n")])

        # Assert
        assert execution.status == "ok"
        content = (isolated_workspace / "unicode.txt").read_text(encoding="utf-8")
        assert content == "café → ✓\n"

    def test_04_unit_10_tools_10_edit_create_08_tmp_absolute_path_creates(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """edit() accepts an absolute /tmp path outside the workspace."""
        with tempfile.TemporaryDirectory(dir="/tmp") as scratch:
            scratch_note = Path(scratch) / "note.txt"

            # Act — CWD is the isolated workspace, the file is elsewhere in /tmp
            execution = run_edit(str(scratch_note), [edit_block("", "scratch data\n")])

            # Assert
            assert execution.status == "ok"
            assert scratch_note.read_text(encoding="utf-8") == "scratch data\n"
