#!/usr/bin/env python3
"""Unit tests for edit tool marker and argument diagnostics."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path

from histrel.tools.repository import EditFileTool
from histrel.tools.tool import ToolExecution
from tests.unit.tools.helpers import REPO_ROOT, edit_block


class Test04Unit05Tools05EditFileToolDiagnostics:
    """edit tool failure diagnostics for containment and malformed blocks."""

    def test_04_unit_05_tools_05_edit_04_missing_file_returns_error(
        self,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """Editing a nonexistent file returns a recoverable error, not a crash."""
        # Act
        execution = run_edit("missing.py", [edit_block("a", "b")])

        # Assert — prose informative failure for a well-specified call
        assert execution.status == "error"
        assert str(execution.output) == (
            "Attempt to edit file 'missing.py' failed due to the file not existing."
        )
        assert execution.metadata["informative"] is True

    def test_04_unit_05_tools_05_edit_05_absolute_path_outside_tmp_rejected(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """Absolute paths outside /tmp are rejected so edits stay contained."""
        # The external file must live outside /tmp, the one location where
        # absolute paths remain rejected. The repo root supplies that
        # location and TemporaryDirectory removes it afterwards.
        with tempfile.TemporaryDirectory(dir=REPO_ROOT) as external_dir:
            external = Path(external_dir) / "external.py"
            external.write_text("value = 1\n", encoding="utf-8")

            # Act
            execution = run_edit(str(external), [edit_block("value = 1", "value = 2")])

            # Assert — error reported with the containment rule stated, and
            # the external file untouched
            assert execution.status == "error"
            assert str(execution.output).startswith(
                f"Attempt to edit file '{external}' failed due to "
                "the path being outside the allowed roots"
            )
            assert "only edits in CWD" in str(execution.output)
            assert execution.metadata["informative"] is True
            assert external.read_text(encoding="utf-8") == "value = 1\n"

    def test_04_unit_05_tools_05_edit_06_search_block_not_found_reports_informative_failure(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """A search block absent from the file reports a prose informative failure."""
        (isolated_workspace / "main.py").write_text("x = 1\n", encoding="utf-8")

        # Act
        execution = run_edit("main.py", [edit_block("not there", "anything")])

        # Assert — the failure names the operation attempt and reason
        assert execution.status == "error"
        assert str(execution.output).startswith(
            "Attempt to edit file 'main.py' failed due to"
        )
        assert "not found" in str(execution.output)
        assert execution.metadata["informative"] is True

    def test_04_unit_05_tools_05_edit_11_missing_search_marker_reports_error(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """A block without the SEARCH marker fails with a parse error."""
        (isolated_workspace / "main.py").write_text("x = 1\n", encoding="utf-8")

        # Act
        execution = run_edit(
            "main.py",
            [
                ("/***END_SEARCH***/\n/***REPLACE***/\nx = 10\n/***END_REPLACE***/"),
            ],
        )

        # Assert — the malformed block is rejected before any file access
        assert execution.status == "error"
        assert str(execution.output).startswith("Error parsing file_edits:")
        assert "/***SEARCH***/" in str(execution.output)
        assert (isolated_workspace / "main.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_04_unit_05_tools_05_edit_12_missing_replace_marker_reports_error(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """A block without the REPLACE marker fails with a parse error."""
        (isolated_workspace / "main.py").write_text("x = 1\n", encoding="utf-8")

        # Act
        execution = run_edit(
            "main.py",
            [
                ("/***SEARCH***/\nx = 1\n/***END_SEARCH***/\n/***END_REPLACE***/"),
            ],
        )

        # Assert
        assert execution.status == "error"
        assert str(execution.output).startswith("Error parsing file_edits:")
        assert "/***REPLACE***/" in str(execution.output)
        assert (isolated_workspace / "main.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_04_unit_05_tools_05_edit_12b_duplicate_marker_fails_informatively(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """A duplicated marker fails with an informative duplicate cause."""
        (isolated_workspace / "main.py").write_text("x = 1\n", encoding="utf-8")

        # Act — the SEARCH marker appears twice
        execution = run_edit(
            "main.py",
            [
                (
                    "/***SEARCH***/\n"
                    "x = 1\n"
                    "/***SEARCH***/\n"
                    "/***END_SEARCH***/\n"
                    "/***REPLACE***/\n"
                    "x = 10\n"
                    "/***END_REPLACE***/"
                ),
            ],
        )

        # Assert — prose informative failure naming the offending marker
        assert execution.status == "error"
        assert str(execution.output) == (
            "Attempt to edit file 'main.py' failed due to multiple "
            "occurrences of marker(s) '/***SEARCH***/' in a file_edits block."
        )
        assert execution.metadata["informative"] is True
        assert (isolated_workspace / "main.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_04_unit_05_tools_05_edit_13_non_string_block_reports_error(
        self,
        isolated_workspace: Path,
        run_edit: Callable[..., ToolExecution],
    ) -> None:
        """A file_edits element that is not a string fails with a parse error."""
        (isolated_workspace / "main.py").write_text("x = 1\n", encoding="utf-8")

        # Act
        execution = run_edit("main.py", [edit_block("x = 1", "x = 10"), 123])

        # Assert
        assert execution.status == "error"
        assert str(execution.output) == (
            "Error parsing file_edits: file_edits[1] must be a string but got int"
        )
        assert (isolated_workspace / "main.py").read_text(encoding="utf-8") == "x = 1\n"

    def test_04_unit_05_tools_05_edit_14_definition_declares_block_strings(
        self,
    ) -> None:
        """The edit definition declares file_edits as block strings."""
        # Act
        tool = EditFileTool()

        # Assert — the schema exposes file_edits as a list of strings and
        # every usage example is a block-formatted edit call
        params = tool.definition["function"]["parameters"]
        assert params["required"] == ["file_path", "file_edits"]
        assert params["properties"]["file_edits"]["items"]["type"] == "string"
        assert tool.examples
        for example in tool.examples:
            assert 'name="edit"' in example
            args_json = example.split('args="', 1)[1].rsplit('"', 1)[0]
            parsed_args = json.loads(args_json)
            assert parsed_args["file_path"]
            for block in parsed_args["file_edits"]:
                assert block.startswith("/***SEARCH***/")
                assert "/***END_SEARCH***/" in block
                assert "/***REPLACE***/" in block
                assert block.endswith("/***END_REPLACE***/")
