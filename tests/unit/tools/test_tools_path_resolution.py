#!/usr/bin/env python3
"""Unit tests for CWD-relative path resolution and containment."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from histrel.tools.tool import (
    resolve_path,
)
from tests.unit.tools.helpers import (
    REPO_ROOT,
)


class Test04Unit02Tools02PathResolution:
    """Path resolution relative to CWD."""

    def test_04_unit_02_tools_02_path_01_relative_path_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "src").mkdir()
            (workspace / "src" / "main.py").touch()

            # Act — resolve relative to CWD (chdir into workspace)
            old_cwd = Path.cwd()
            os.chdir(workspace)
            try:
                resolved = resolve_path("src/main.py")
            finally:
                os.chdir(old_cwd)

            # Assert
            assert resolved.exists()
            assert resolved.name == "main.py"

    def test_04_unit_02_tools_02_path_02_absolute_path_rejected(self) -> None:
        # Act & Assert — paths outside CWD and /tmp are not allowed
        with pytest.raises(ValueError, match="not under"):
            resolve_path("/etc/passwd")

    def test_04_unit_02_tools_02_path_03_nonexistent_path_raises(self) -> None:
        # Act & Assert — strict resolution fails for nonexistent paths
        with pytest.raises(FileNotFoundError):
            resolve_path("nonexistent/file.py")

    def test_04_unit_02_tools_02_path_04_absolute_tmp_path_resolves(
        self, isolated_workspace: Path
    ) -> None:
        """Absolute /tmp paths resolve even though CWD is elsewhere."""
        with tempfile.TemporaryDirectory(dir="/tmp") as scratch:
            scratch_file = Path(scratch) / "note.txt"
            scratch_file.write_text("data\n", encoding="utf-8")

            # Act — CWD is the isolated workspace, not the file's directory
            resolved = resolve_path(str(scratch_file))

            # Assert
            assert resolved == scratch_file.resolve()

    def test_04_unit_02_tools_02_path_05_tmp_root_resolves(
        self, isolated_workspace: Path
    ) -> None:
        """The /tmp root itself is an allowed absolute path."""
        # Act
        resolved = resolve_path("/tmp")

        # Assert
        assert resolved == Path("/tmp").resolve()

    def test_04_unit_02_tools_02_path_06_tmp_symlink_escape_rejected(
        self, isolated_workspace: Path
    ) -> None:
        """A /tmp symlink pointing outside /tmp is rejected after resolution."""
        with tempfile.TemporaryDirectory(dir="/tmp") as scratch:
            link = Path(scratch) / "escape"
            link.symlink_to("/etc")

            # Act & Assert — the resolved target is outside both CWD and /tmp
            with pytest.raises(ValueError, match="not under"):
                resolve_path(str(link))

    def test_04_unit_02_tools_02_path_07_nonexistent_tmp_path_raises(
        self, isolated_workspace: Path
    ) -> None:
        """Strict resolution still applies to allowed /tmp absolute paths."""
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            resolve_path("/tmp/histrel-does-not-exist-check")

    def test_04_unit_02_tools_02_path_08_traversal_escape_rejected(
        self, isolated_workspace: Path
    ) -> None:
        """A relative path that traverses outside CWD and /tmp is rejected.

        The input is a *relative* path, so the old absolute-path check
        would have let it through.  The new logic resolves to an absolute
        path first and then rejects it because the resolved target is
        neither under CWD nor under /tmp.
        """
        with tempfile.TemporaryDirectory(dir="/tmp") as scratch:
            # Create a file outside the workspace but inside /tmp so we
            # can build a ..-chain that escapes *both* CWD and /tmp.
            bait = Path(scratch) / "bait.txt"
            bait.write_text("secret\n", encoding="utf-8")

            # CWD is isolated_workspace (inside /tmp).  Build a relative
            # path that climbs past /tmp to /etc/passwd.
            depth = len(isolated_workspace.relative_to(Path("/tmp").resolve()).parts)
            escape = "../" * (depth + 1) + "etc/passwd"

            # Act & Assert
            with pytest.raises(ValueError, match="not under"):
                resolve_path(escape)

    def test_04_unit_02_tools_02_path_09_absolute_path_under_cwd_resolves(
        self,
    ) -> None:
        """An absolute path under CWD is accepted even when CWD is not /tmp."""
        # Arrange — use the repo root (not under /tmp) as the parent for
        # a temporary CWD so the CWD-or-/tmp check is exercised independently.
        repo_root = REPO_ROOT
        with tempfile.TemporaryDirectory(dir=repo_root) as cwd_dir:
            cwd = Path(cwd_dir)
            (cwd / "target.py").write_text("x = 1\n", encoding="utf-8")
            abs_target = cwd / "target.py"

            old_cwd = Path.cwd()
            os.chdir(cwd)
            try:
                # Act
                resolved = resolve_path(str(abs_target))
            finally:
                os.chdir(old_cwd)

            # Assert
            assert resolved == abs_target.resolve()

    def test_04_unit_02_tools_02_path_10_cwd_symlink_escape_rejected(
        self,
    ) -> None:
        """A CWD-relative symlink pointing outside CWD and /tmp is rejected."""
        repo_root = REPO_ROOT
        with tempfile.TemporaryDirectory(dir=repo_root) as cwd_dir:
            cwd = Path(cwd_dir)
            (cwd / "src").mkdir()
            link = cwd / "src" / "escape"
            link.symlink_to("/etc")

            old_cwd = Path.cwd()
            os.chdir(cwd)
            try:
                # Act & Assert — resolved target is outside both CWD and /tmp
                with pytest.raises(ValueError, match="not under"):
                    resolve_path("src/escape")
            finally:
                os.chdir(old_cwd)

    def test_04_unit_02_tools_02_path_11_dotdot_within_cwd_resolves(
        self, isolated_workspace: Path
    ) -> None:
        """A relative path using .. that stays inside CWD resolves."""
        # Arrange
        (isolated_workspace / "a").mkdir()
        (isolated_workspace / "b").mkdir()
        (isolated_workspace / "b" / "file.txt").write_text("ok\n", encoding="utf-8")

        # Act — a/../b/file.txt stays within CWD
        resolved = resolve_path("a/../b/file.txt")

        # Assert
        assert resolved == (isolated_workspace / "b" / "file.txt").resolve()
