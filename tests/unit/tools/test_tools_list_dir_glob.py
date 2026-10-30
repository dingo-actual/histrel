#!/usr/bin/env python3
"""Unit tests for list_dir, search and glob directory tools."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from histrel.tools.repository import (
    GlobTool,
    ListDirTool,
    SearchTool,
)
from tests.unit.tools.helpers import (
    make_context,
)


class Test04Unit04Tools04ListDirTool:
    """list_dir behavior on directories and non-directories."""

    def test_04_unit_02_tools_04_list_dir_01_file_path_returns_error(
        self,
    ) -> None:
        """list_dir on a file path reports a prose informative failure."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "serialization.py").touch()

            old_cwd = Path.cwd()
            os.chdir(workspace)
            try:
                # Act
                execution = ListDirTool().execute(
                    {"path": "serialization.py"}, make_context()
                )
            finally:
                os.chdir(old_cwd)

            # Assert — prose informative failure for a well-specified call
            assert execution.status == "error"
            assert str(execution.output) == (
                "Attempt to list directory 'serialization.py' failed due to "
                "the path not being a directory."
            )
            assert execution.metadata["informative"] is True

    def test_04_unit_02_tools_04_list_dir_02_directory_listing_works(
        self,
    ) -> None:
        """list_dir on a directory returns entries with type indicators."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "src").mkdir()
            (workspace / "main.py").touch()

            old_cwd = Path.cwd()
            os.chdir(workspace)
            try:
                # Act
                execution = ListDirTool().execute({}, make_context())
            finally:
                os.chdir(old_cwd)

            # Assert — success prepends the default path header to entries
            assert execution.status == "ok"
            assert str(execution.output).startswith("Contents of directory .\n---\n")
            assert "src/" in str(execution.output)
            assert "main.py" in str(execution.output)

    def test_04_unit_02_tools_04_list_dir_03_missing_directory_reports_informative_failure(
        self,
    ) -> None:
        """list_dir on a nonexistent path reports a prose informative failure."""
        # Act — resolve fails before any directory access
        execution = ListDirTool().execute({"path": "nope"}, make_context())

        # Assert
        assert execution.status == "error"
        assert str(execution.output) == (
            "Attempt to list directory 'nope' failed due to the directory not existing."
        )
        assert execution.metadata["informative"] is True


@pytest.mark.skipif(shutil.which("rg") is None, reason="ripgrep not installed")
class Test04Unit07Tools07SearchTool:
    """search tool: matches, no-match reporting, and error handling."""

    def test_04_unit_07_tools_07_search_01_finds_matches(self) -> None:
        """A matching pattern returns matches with file paths and lines."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "src").mkdir()
            (workspace / "src" / "a.py").write_text("def hello():\n", encoding="utf-8")

            old_cwd = Path.cwd()
            os.chdir(workspace)
            try:
                # Act
                execution = SearchTool().execute({"pattern": "hello"}, make_context())
            finally:
                os.chdir(old_cwd)

            # Assert — success prepends the pattern/path header to matches
            assert execution.status == "ok"
            assert str(execution.output).startswith(
                "Search for string pattern 'hello' in .\n---\n"
            )
            assert "src/a.py" in str(execution.output)
            assert "hello" in str(execution.output)

    def test_04_unit_07_tools_07_search_02_no_matches_reports_no_matches(self) -> None:
        """A successful search with no hits reports 'no matches' explicitly."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "a.py").write_text("content\n", encoding="utf-8")

            old_cwd = Path.cwd()
            os.chdir(workspace)
            try:
                # Act
                execution = SearchTool().execute(
                    {"pattern": "zzznotfound"}, make_context()
                )
            finally:
                os.chdir(old_cwd)

            # Assert — success prepends the pattern/path header to the
            # no-matches placeholder
            assert execution.status == "ok"
            assert (
                str(execution.output)
                == "Search for string pattern 'zzznotfound' in .\n---\n(no matches)"
            )

    def test_04_unit_07_tools_07_search_03_backend_error_returns_error(self) -> None:
        """A ripgrep failure (invalid regex) is an error, not a silent ok."""
        # Act
        execution = SearchTool().execute({"pattern": "( ["}, make_context())

        # Assert — prose informative failure names the operation attempt,
        # then the backend's error detail
        assert execution.status == "error"
        assert str(execution.output).startswith(
            "Attempt to search for text pattern '( [' in '.' failed due to:"
        )
        assert execution.metadata["informative"] is True

    def test_04_unit_07_tools_07_search_04_missing_path_returns_error(self) -> None:
        """A nonexistent search path is an error, not a silent ok."""
        # Act
        execution = SearchTool().execute(
            {"pattern": "hello", "path": "nope"}, make_context()
        )

        # Assert
        assert execution.status == "error"
        assert str(execution.output) == (
            "Attempt to search for text pattern 'hello' in 'nope' failed due to "
            "the search path not existing."
        )
        assert execution.metadata["informative"] is True

    def test_04_unit_07_tools_07_search_05_absolute_path_rejected(self) -> None:
        """Absolute search paths are rejected so searches stay in the workspace."""
        # Act
        execution = SearchTool().execute(
            {"pattern": "root", "path": "/etc"}, make_context()
        )

        # Assert — rejected with a prose reason, and nothing from the host
        # filesystem leaks
        assert execution.status == "error"
        assert str(execution.output).startswith(
            "Attempt to search for text pattern 'root' in '/etc' failed due to path"
        )
        assert "not under" in str(execution.output)
        assert execution.metadata["informative"] is True
        assert "root:x:0:0" not in str(execution.output)


@pytest.mark.skipif(shutil.which("fd") is None, reason="fd not installed")
class Test04Unit08Tools08GlobTool:
    """glob tool: glob-pattern matching, no-match reporting, error handling."""

    def test_04_unit_08_tools_08_glob_01_glob_pattern_matches_files(self) -> None:
        """The pattern is treated as a glob, so '*.py' matches Python files."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "src").mkdir()
            (workspace / "src" / "a.py").write_text("x = 1\n", encoding="utf-8")
            (workspace / "b.txt").write_text("y = 2\n", encoding="utf-8")

            old_cwd = Path.cwd()
            os.chdir(workspace)
            try:
                # Act
                execution = GlobTool().execute({"pattern": "*.py"}, make_context())
            finally:
                os.chdir(old_cwd)

            # Assert — success prepends the pattern/path header to matches
            assert execution.status == "ok"
            assert str(execution.output).startswith(
                "Search for file pattern '*.py' in .\n---\n"
            )
            assert "src/a.py" in str(execution.output)
            assert "b.txt" not in str(execution.output)

    def test_04_unit_08_tools_08_glob_02_no_matches_reports_no_matches(self) -> None:
        """A successful glob with no hits reports 'no matches' explicitly."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "b.txt").write_text("y = 2\n", encoding="utf-8")

            old_cwd = Path.cwd()
            os.chdir(workspace)
            try:
                # Act
                execution = GlobTool().execute({"pattern": "*.zzz"}, make_context())
            finally:
                os.chdir(old_cwd)

            # Assert — success prepends the pattern/path header to the
            # no-matches placeholder
            assert execution.status == "ok"
            assert (
                str(execution.output)
                == "Search for file pattern '*.zzz' in .\n---\n(no matches)"
            )

    def test_04_unit_08_tools_08_glob_03_bad_path_returns_error(self) -> None:
        """A nonexistent search path is an error, not a silent ok."""
        # Act
        execution = GlobTool().execute(
            {"pattern": "*.py", "path": "nope"}, make_context()
        )

        # Assert
        assert execution.status == "error"
        assert str(execution.output) == (
            "Attempt to search for file pattern '*.py' in 'nope' failed due to "
            "the search path not existing."
        )
        assert execution.metadata["informative"] is True

    def test_04_unit_08_tools_08_glob_04_absolute_path_rejected(self) -> None:
        """Absolute search paths are rejected so globs stay in the workspace."""
        # Act
        execution = GlobTool().execute(
            {"pattern": "passwd", "path": "/etc"}, make_context()
        )

        # Assert — rejected with a prose reason, and nothing from the host
        # filesystem leaks
        assert execution.status == "error"
        assert str(execution.output).startswith(
            "Attempt to search for file pattern 'passwd' in '/etc' failed due to path"
        )
        assert "not under" in str(execution.output)
        assert execution.metadata["informative"] is True
        # The prose echoes the pattern argument by design; only host
        # filesystem content must not leak
        assert "root:" not in str(execution.output)
