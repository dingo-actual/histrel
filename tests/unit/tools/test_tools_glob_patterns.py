#!/usr/bin/env python3
"""Unit tests for glob patterns matched against names and paths.

A pattern without a path separator is matched against the file or
directory name, so it finds a test directory by name. A pattern with a
separator is matched against the path relative to the search directory,
so an anchored pattern like ``**/sub/*.py`` works. Neither form filters
by file type, so directories are reported as well.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from histrel.tools.repository import GlobTool
from tests.unit.tools.helpers import make_context


@contextmanager
def _workspace(files: tuple[str, ...]) -> Iterator[Path]:
    """Create a temp workspace with the given files and chdir into it."""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp)
        for relative in files:
            target = workspace / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x = 1\n", encoding="utf-8")
        old_cwd = Path.cwd()
        os.chdir(workspace)
        try:
            yield workspace
        finally:
            os.chdir(old_cwd)


class Test04Unit08Tools08GlobPatterns:
    """Glob name patterns and path patterns."""

    def test_04_unit_08_tools_08_globpat_01_directory_name_matches_directory(
        self,
    ) -> None:
        """A name pattern matches a directory, not only the files in it."""
        # Arrange — the agent searches for a test directory by name
        with _workspace(("tests/distinct_on/d.py",)):
            # Act
            execution = GlobTool().execute({"pattern": "*distinct*"}, make_context())

            # Assert
            assert execution.status == "ok"
            assert "tests/distinct_on" in str(execution.output)

    def test_04_unit_08_tools_08_globpat_02_path_pattern_matches_nested_files(
        self,
    ) -> None:
        """A pattern containing '/' is matched against the workspace path."""
        # Arrange
        with _workspace(("src/sub/c.py", "src/other.py")):
            # Act
            execution = GlobTool().execute({"pattern": "**/sub/*.py"}, make_context())

            # Assert
            assert execution.status == "ok"
            assert "src/sub/c.py" in str(execution.output)
            assert "src/other.py" not in str(execution.output)

    def test_04_unit_08_tools_08_globpat_03_bare_name_pattern_matches_nested_file(
        self,
    ) -> None:
        """A bare name pattern still matches files in nested directories."""
        # Arrange
        with _workspace(("src/sub/c.py",)):
            # Act
            execution = GlobTool().execute({"pattern": "*.py"}, make_context())

            # Assert
            assert execution.status == "ok"
            assert "src/sub/c.py" in str(execution.output)
