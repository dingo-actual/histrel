#!/usr/bin/env python3
"""Unit tests for edit tool block application and persistence."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from histrel.tools.tool import ToolExecution
from tests.unit.tools.helpers import edit_block


class Test04Unit05Tools05EditFileTool:
    """edit tool behavior: persistence, error handling, and containment."""

    def test_04_unit_05_tools_05_edit_01_persists_patched_content(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """A successful edit writes the patched content to disk."""
        (isolated_workspace / "main.py").write_text(
            "def hello():\n    print('hello')\n    return None\n",
            encoding="utf-8",
        )

        # Act
        execution = run_edit(
            "main.py", [edit_block("print('hello')", "print('world')")]
        )

        # Assert — status reports success and the file content changed
        assert execution.status == "ok"
        assert "Edited file" in str(execution.output)
        content = (isolated_workspace / "main.py").read_text(encoding="utf-8")
        assert "print('world')" in content
        assert "print('hello')" not in content

    def test_04_unit_05_tools_05_edit_02_persists_multiple_changes(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """All changes in one edit call persist to disk."""
        (isolated_workspace / "multi.py").write_text(
            "x = 1\ny = 2\nz = 3\n", encoding="utf-8"
        )

        # Act
        execution = run_edit(
            "multi.py",
            [
                edit_block("x = 1", "x = 10"),
                edit_block("z = 3", "z = 30"),
            ],
        )

        # Assert
        assert execution.status == "ok"
        content_lines = (
            (isolated_workspace / "multi.py").read_text(encoding="utf-8").split("\n")
        )
        assert "x = 10" in content_lines
        assert "z = 30" in content_lines
        assert "x = 1" not in content_lines
        assert "z = 3" not in content_lines

    def test_04_unit_05_tools_05_edit_03_preserves_crlf_on_disk(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """Persisted content keeps the file's original CRLF line endings."""
        (isolated_workspace / "crlf.py").write_text(
            "x = 1\r\ny = 2\r\n", encoding="utf-8", newline=""
        )

        # Act
        execution = run_edit("crlf.py", [edit_block("x = 1", "x = 10")])

        # Assert
        assert execution.status == "ok"
        data = (isolated_workspace / "crlf.py").read_bytes()
        assert b"x = 10\r\n" in data
        assert b"x = 1\r" not in data

    def test_04_unit_05_tools_05_edit_07_multiline_block_replaces_matched_text(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """A multi-line SEARCH/REPLACE block replaces the whole matched text."""
        (isolated_workspace / "divide.py").write_text(
            "def divide(a, b):\n    return a / b\n\nx = divide(1, 2)\n",
            encoding="utf-8",
        )

        # Act
        execution = run_edit(
            "divide.py",
            [
                edit_block(
                    "def divide(a, b):\n    return a / b\n",
                    (
                        "def divide(a, b):\n"
                        "    if b == 0:\n"
                        "        raise ValueError('division by zero')\n"
                        "    return a / b\n"
                    ),
                )
            ],
        )

        # Assert — the two matched lines are replaced verbatim
        assert execution.status == "ok"
        content = (isolated_workspace / "divide.py").read_text(encoding="utf-8")
        assert content == (
            "def divide(a, b):\n"
            "    if b == 0:\n"
            "        raise ValueError('division by zero')\n"
            "    return a / b\n"
            "\nx = divide(1, 2)\n"
        )

    def test_04_unit_05_tools_05_edit_08_normalizes_lf_blocks_to_crlf_file(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """LF newlines in the blocks are normalized to the CRLF file."""
        (isolated_workspace / "total.py").write_text(
            "def total(items):\r\n    return 0\r\n",
            encoding="utf-8",
            newline="",
        )

        # Act — the block uses LF newlines against a CRLF file
        execution = run_edit(
            "total.py",
            [
                edit_block(
                    "def total(items):\n    return 0\n",
                    "def total(items):\n    return sum(items)\n",
                )
            ],
        )

        # Assert — the search matched and the file keeps CRLF endings
        assert execution.status == "ok"
        data = (isolated_workspace / "total.py").read_bytes()
        assert data == b"def total(items):\r\n    return sum(items)\r\n"

    def test_04_unit_05_tools_05_edit_09_normalizes_crlf_blocks_to_lf_file(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """CRLF newlines in the blocks are normalized to the LF file."""
        (isolated_workspace / "pairs.py").write_text("x = 1\ny = 2\n", encoding="utf-8")

        # Act — the block uses CRLF newlines against an LF file
        execution = run_edit(
            "pairs.py",
            [edit_block("x = 1\r\ny = 2\r\n", "x = 10\r\ny = 20\r\n")],
        )

        # Assert — the search matched and the file keeps LF endings
        assert execution.status == "ok"
        data = (isolated_workspace / "pairs.py").read_bytes()
        assert data == b"x = 10\ny = 20\n"

    def test_04_unit_05_tools_05_edit_10_empty_replace_block_deletes_search(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """An empty REPLACE block deletes the SEARCH text from the file."""
        (isolated_workspace / "lines.py").write_text(
            "x = 1\ny = 2\nz = 3\n", encoding="utf-8"
        )

        # Act
        execution = run_edit("lines.py", [edit_block("y = 2", "")])

        # Assert — the matched line is removed with no stray blank line
        assert execution.status == "ok"
        content = (isolated_workspace / "lines.py").read_text(encoding="utf-8")
        assert content == "x = 1\nz = 3\n"

    def test_04_unit_05_tools_05_edit_12c_stray_text_outside_blocks_ignored(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """Text outside the declared blocks is ignored and the edit applies."""
        (isolated_workspace / "hello.py").write_text(
            "print('hello')\n", encoding="utf-8"
        )

        # Act — prose wraps the block on both sides
        execution = run_edit(
            "hello.py",
            [
                (
                    "Thought: fix the greeting.\n"
                    "/***SEARCH***/\n"
                    "print('hello')\n"
                    "/***END_SEARCH***/\n"
                    "/***REPLACE***/\n"
                    "print('world')\n"
                    "/***END_REPLACE***/\n"
                    "Done fixing the greeting."
                ),
            ],
        )

        # Assert — the edit applies to the declared blocks only
        assert execution.status == "ok"
        content = (isolated_workspace / "hello.py").read_text(encoding="utf-8")
        assert content == "print('world')\n"
