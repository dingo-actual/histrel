#!/usr/bin/env python3
"""Unit tests for edit tool file-creation failure diagnostics."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from pathlib import Path

from histrel.tools.tool import ToolExecution
from tests.unit.tools.helpers import REPO_ROOT, edit_block


class Test04Unit10Tools10EditCreateFileDiagnostics:
    """edit tool creation failure diagnostics: containment and invalid targets."""

    def test_04_unit_10_tools_10_edit_create_05_missing_parent_returns_error(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """A nonexistent parent directory is an error; no directories are created."""
        # Act
        execution = run_edit("nope/new.py", [edit_block("", "x = 1\n")])

        # Assert — the parent directory is missing, so no path resolves;
        # the resolve failure is reported before create intent is known
        assert execution.status == "error"
        assert str(execution.output) == (
            "Attempt to edit file 'nope/new.py' failed due to "
            "the parent directory not existing."
        )
        assert execution.metadata["informative"] is True
        assert not (isolated_workspace / "nope").exists()

    def test_04_unit_10_tools_10_edit_create_06_directory_path_returns_error(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """Creating a file at a path that is a directory is an error."""
        (isolated_workspace / "sub").mkdir()

        # Act
        execution = run_edit("sub", [edit_block("", "x = 1\n")])

        # Assert — the existing path is a directory, so the edit attempt
        # fails informatively
        assert execution.status == "error"
        assert str(execution.output) == (
            "Attempt to edit file 'sub' failed due to the path being a directory."
        )
        assert execution.metadata["informative"] is True

    def test_04_unit_10_tools_10_edit_create_07_absolute_path_outside_tmp_rejected(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """Absolute paths outside /tmp are rejected so edits stay contained."""
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as external_dir:
            external = Path(external_dir) / "external.py"

            # Act
            execution = run_edit(str(external), [edit_block("", "value = 1\n")])

            # Assert — error reported with the containment rule stated, and
            # no file was created outside /tmp; containment is rejected
            # during path resolution, before create intent is known
            assert execution.status == "error"
            assert str(execution.output).startswith(
                f"Attempt to edit file '{external}' failed due to "
                "the path being outside the allowed roots"
            )
            assert "only edits in CWD" in str(execution.output)
            assert execution.metadata["informative"] is True
            assert not external.exists()

    def test_04_unit_10_tools_10_edit_create_09_nonempty_search_on_missing_file_error(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """A non-empty search on a nonexistent file is still an error."""
        # Act
        execution = run_edit("missing.py", [edit_block("foo", "bar")])

        # Assert — a non-empty search cannot create the file, so the
        # edit attempt fails informatively
        assert execution.status == "error"
        assert str(execution.output) == (
            "Attempt to edit file 'missing.py' failed due to the file not existing."
        )
        assert execution.metadata["informative"] is True
        assert not (isolated_workspace / "missing.py").exists()

    def test_04_unit_10_tools_10_edit_create_10_multiple_changes_rejected_for_create(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """Multiple edits with empty search on a missing file is an error.

        File creation requires exactly one edit. Providing more than
        one edit (even if the first has an empty search) is rejected.
        """
        # Act
        execution = run_edit(
            "new.py",
            [
                edit_block("", "x = 1\n"),
                edit_block("x = 1", "x = 2\n"),
            ],
        )

        # Assert
        assert execution.status == "error"
        assert str(execution.output) == (
            "Attempt to edit file 'new.py' failed due to the file not existing."
        )
        assert execution.metadata["informative"] is True
        assert not (isolated_workspace / "new.py").exists()

    def test_04_unit_10_tools_10_edit_create_11_existing_file_with_empty_search_error(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """An empty search on an existing file does not overwrite it."""
        (isolated_workspace / "existing.py").write_text(
            "original = True\n", encoding="utf-8"
        )

        # Act
        execution = run_edit("existing.py", [edit_block("", "overwritten = True\n")])

        # Assert — an empty search on an existing file is an informative
        # failure, not an overwrite
        assert execution.status == "error"
        assert str(execution.output).startswith(
            "Attempt to edit file 'existing.py' failed due to"
        )
        assert execution.metadata["informative"] is True
        content = (isolated_workspace / "existing.py").read_text(encoding="utf-8")
        assert content == "original = True\n"
