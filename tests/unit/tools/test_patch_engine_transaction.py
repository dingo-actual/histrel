#!/usr/bin/env python3
"""Unit tests for PatchTransaction atomic application and verbatim replacement."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from histrel.tools.patch_engine import (
    AmbiguousMatchError,
    MatchNotFoundError,
    PatchError,
    apply_patch,
)
from histrel.tools.patch_utils import Change, detect_newline_style


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    """Create a temporary directory with git initialized.

    Args:
        tmp_path: Pytest temporary path fixture.

    Returns:
        Path to temporary directory with git repository initialized.
    """
    subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(tmp_path),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(tmp_path),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "config", "merge.conflictstyle", "merge"],
        cwd=str(tmp_path),
        check=False,
        capture_output=True,
    )
    return tmp_path


def git_commit_file(file_path: Path, git_root: Path) -> None:
    """Add and commit a file to git.

    Args:
        file_path: Path to file to commit.
        git_root: Path to git repository root.
    """
    subprocess.run(
        ["git", "add", str(file_path.name)],
        cwd=str(git_root),
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", f"Add {file_path.name}"],
        cwd=str(git_root),
        capture_output=True,
        check=True,
    )


class Test04Unit03Patch03Transaction:
    """PatchTransaction atomic application."""

    def test_04_unit_03_patch_03_txn_01_apply_single_change(
        self, git_repo: Path
    ) -> None:
        # Arrange
        test_file = git_repo / "test.py"
        test_file.write_text(
            "def hello():\n    print('hello')\n    return None\n", encoding="utf-8"
        )
        git_commit_file(test_file, git_repo)

        # Act
        result = apply_patch(
            str(test_file),
            [Change(search="print('hello')", replace="print('world')")],
        )

        # Assert
        assert "print('world')" in result
        assert "print('hello')" not in result

    def test_04_unit_03_patch_03_txn_02_apply_multiple_changes(
        self, git_repo: Path
    ) -> None:
        # Arrange
        test_file = git_repo / "multi.py"
        test_file.write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8")
        git_commit_file(test_file, git_repo)

        # Act — apply bottom-up (z first, then x)
        result = apply_patch(
            str(test_file),
            [
                Change(search="x = 1", replace="x = 10"),
                Change(search="z = 3", replace="z = 30"),
            ],
        )

        # Assert
        result_lines = result.split("\n")
        assert "x = 10" in result_lines
        assert "z = 30" in result_lines
        assert "x = 1" not in result_lines
        assert "z = 3" not in result_lines

    def test_04_unit_03_patch_03_txn_03_overlap_raises(self, git_repo: Path) -> None:
        # Arrange
        test_file = git_repo / "overlap.py"
        test_file.write_text("a\nb\nc\nd\n", encoding="utf-8")
        git_commit_file(test_file, git_repo)

        # Act & Assert — overlapping ranges
        with pytest.raises(PatchError, match="Overlap"):
            apply_patch(
                str(test_file),
                [
                    Change(search="a\nb", replace="X\nY"),
                    Change(search="b\nc", replace="Y\nZ"),
                ],
            )

    def test_04_unit_03_patch_03_txn_04_not_found_raises(self, git_repo: Path) -> None:
        # Arrange
        test_file = git_repo / "notfound.py"
        test_file.write_text("print('hello')\n", encoding="utf-8")
        git_commit_file(test_file, git_repo)

        # Act & Assert
        with pytest.raises(MatchNotFoundError):
            apply_patch(
                str(test_file),
                [Change(search="def nonexistent()", replace="pass")],
            )

    def test_04_unit_03_patch_03_txn_05_preserves_newline_style(
        self, git_repo: Path
    ) -> None:
        # Arrange
        test_file = git_repo / "crlf.py"
        test_file.write_text("x = 1\r\ny = 2\r\n", encoding="utf-8", newline="")
        git_commit_file(test_file, git_repo)

        # Act
        result = apply_patch(
            str(test_file),
            [Change(search="x = 1", replace="x = 10")],
        )

        # Assert — CRLF preserved
        assert "\r\n" in result
        assert "x = 10" in result

    def test_04_unit_03_patch_03_txn_06_file_not_found_raises(self) -> None:
        # Act & Assert
        with pytest.raises(PatchError, match="not found"):
            apply_patch("/nonexistent/file.py", [])

    def test_04_unit_03_patch_03_txn_07_relative_indent_increase(
        self, git_repo: Path
    ) -> None:
        # Arrange — replace has MORE indentation than search
        test_file = git_repo / "indent_inc.py"
        test_file.write_text("def foo():\n    bar()\n", encoding="utf-8", newline="\n")
        git_commit_file(test_file, git_repo)

        # Act
        result = apply_patch(
            str(test_file),
            [
                Change(
                    search="def foo():\n    bar()",
                    replace="def foo():\n    if True:\n        bar()",
                ),
            ],
        )

        # Assert — bar() should be indented 8 spaces (relative to 4-space search)
        assert "    if True:" in result
        assert "        bar()" in result

    def test_04_unit_03_patch_03_txn_08_relative_indent_decrease(
        self, git_repo: Path
    ) -> None:
        # Arrange — replace has LESS indentation than search
        test_file = git_repo / "indent_dec.py"
        test_file.write_text(
            "def foo():\n        bar()\n", encoding="utf-8", newline="\n"
        )
        git_commit_file(test_file, git_repo)

        # Act
        result = apply_patch(
            str(test_file),
            [
                Change(
                    search="def foo():\n        bar()",
                    replace="def foo():\n    bar()",
                ),
            ],
        )

        # Assert — bar() should be at 4 spaces (relative to 8-space search)
        assert "    bar()" in result


class Test04Unit06Patch06NewlineStyle:
    """Newline style detection by dominant terminator count."""

    def test_04_unit_06_patch_06_style_01_crlf_dominant_wins(self) -> None:
        # Arrange — two CRLF terminators against one bare LF
        content = "a\r\nb\r\nc\nd"

        # Act & Assert
        assert detect_newline_style(content) == "\r\n"

    def test_04_unit_06_patch_06_style_02_lf_dominant_wins(self) -> None:
        # Arrange — two bare LF terminators against one CRLF
        content = "a\nb\nc\r\nd"

        # Act & Assert
        assert detect_newline_style(content) == "\n"

    def test_04_unit_06_patch_06_style_03_cr_dominant_wins(self) -> None:
        # Arrange — two bare CR terminators against one LF
        content = "a\rb\rc\nd"

        # Act & Assert
        assert detect_newline_style(content) == "\r"

    def test_04_unit_06_patch_06_style_04_no_terminators_defaults_lf(self) -> None:
        # Act & Assert
        assert detect_newline_style("single line") == "\n"


class Test04Unit03Patch04Verbatim:
    """Verbatim replacement for exactly matched search blocks."""

    def test_04_unit_03_patch_04_verbatim_01_exact_search_replace_verbatim(
        self, tmp_path: Path
    ) -> None:
        # Arrange — replace block de-indents to a new method at class level
        test_file = tmp_path / "verbatim_deindent.py"
        test_file.write_text(
            "class C:\n"
            "    def run(self):\n"
            "        with cursor():\n"
            "            for sql in self.as_sql():\n"
            "                pass\n"
            "            if not self.rows:\n"
            "                return []\n"
            "            return [(self.last_id(),)]\n",
            encoding="utf-8",
            newline="\n",
        )
        search = "            if not self.rows:\n                return []\n            return [(self.last_id(),)]"
        replace = (
            "            if not self.rows:\n"
            "                return []\n"
            "            return list(self.convert([(self.last_id(),)]))\n"
            "\n"
            "    def convert(self, rows):\n"
            "        return list(rows)\n"
        )

        # Act
        result = apply_patch(str(test_file), [Change(search=search, replace=replace)])

        # Assert — body lines keep their file indentation and the new
        # method lands at class level, exactly as written in the block
        assert "            if not self.rows:" in result
        assert "            return list(self.convert([(self.last_id(),)]))" in result
        assert "\n    def convert(self, rows):\n        return list(rows)\n" in result
        assert "                if not self.rows:" not in result

    def test_04_unit_03_patch_04_verbatim_02_non_exact_search_still_reindents(
        self, tmp_path: Path
    ) -> None:
        # Arrange — search indentation differs from the file, so the
        # normalized tier must locate it and relative reindentation applies
        test_file = tmp_path / "fallback_reindent.py"
        test_file.write_text(
            "class C:\n"
            "    def run(self):\n"
            "        with cursor():\n"
            "            a()\n"
            "            b()\n",
            encoding="utf-8",
            newline="\n",
        )
        search = "                a()\n        b()"

        # Act
        result = apply_patch(
            str(test_file), [Change(search=search, replace="a()\nb()")]
        )

        # Assert — both lines re-indented to the matched location's indent
        assert "            a()\n" in result
        assert "            b()\n" in result

    def test_04_unit_03_patch_04_verbatim_03_empty_replace_deletes_block(
        self, tmp_path: Path
    ) -> None:
        # Arrange — exact match deleted with an empty replace block
        test_file = tmp_path / "delete.py"
        test_file.write_text("x = 1\ny = 2\nz = 3\n", encoding="utf-8", newline="\n")

        # Act
        result = apply_patch(str(test_file), [Change(search="y = 2", replace="")])

        # Assert — block removed with no stray blank line left behind
        assert result == "x = 1\nz = 3\n"

    def test_04_unit_03_patch_04_verbatim_04_replace_without_trailing_newline(
        self, tmp_path: Path
    ) -> None:
        # Arrange — replacement lacks a trailing newline mid-file
        test_file = tmp_path / "no_newline.py"
        test_file.write_text("a = 1\nb = 2\n", encoding="utf-8", newline="\n")

        # Act
        result = apply_patch(str(test_file), [Change(search="a = 1", replace="a = 10")])

        # Assert — lines do not join
        assert result == "a = 10\nb = 2\n"

    def test_04_unit_03_patch_04_verbatim_05_multiple_exact_matches_raise(
        self, tmp_path: Path
    ) -> None:
        # Arrange — the search block matches exactly twice
        test_file = tmp_path / "ambiguous.py"
        test_file.write_text("x = 1\nx = 1\n", encoding="utf-8", newline="\n")

        # Act & Assert
        with pytest.raises(AmbiguousMatchError):
            apply_patch(str(test_file), [Change(search="x = 1", replace="x = 10")])

    def test_04_unit_03_patch_04_verbatim_06_empty_replace_non_exact_deletes(
        self, tmp_path: Path
    ) -> None:
        """An empty replace block deletes matched lines on any match tier.

        A search block whose indentation differs from the file exercises
        the normalized tier, so the empty replace takes the reindent path
        and must still delete the matched range with no stray newline.
        """
        # Arrange
        test_file = tmp_path / "delete_non_exact.py"
        test_file.write_text(
            "a = 1\n    b = 2\nc = 3\n",
            encoding="utf-8",
            newline="\n",
        )

        # Act — over-indented search text forces the normalized tier
        result = apply_patch(
            str(test_file), [Change(search="        b = 2\n", replace="")]
        )

        # Assert — the matched line is removed with no blank line left
        assert result == "a = 1\nc = 3\n"
