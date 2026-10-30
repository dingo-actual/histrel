#!/usr/bin/env python3
"""Unit tests for read tool edge cases and line formatting."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from histrel.tools.repository import (
    ReadFileTool,
)
from tests.unit.tools.helpers import (
    make_context,
)


class Test04Unit06Tools06ReadFileEdgeCases:
    """read tool edge cases: empty files, binary files, directories, line format."""

    def test_04_unit_06_tools_06_read_01_empty_file_returns_ok(self) -> None:
        """Reading an empty file succeeds with empty output."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "empty.py").write_text("", encoding="utf-8")

            old_cwd = Path.cwd()
            os.chdir(workspace)
            try:
                # Act
                execution = ReadFileTool().execute({"path": "empty.py"}, make_context())
            finally:
                os.chdir(old_cwd)

            # Assert — success prepends the path header to the empty body
            assert execution.status == "ok"
            assert execution.output == "Contents of file empty.py (empty file)\n---\n"

    def test_04_unit_06_tools_06_read_02_non_utf8_returns_error(self) -> None:
        """Reading a non-UTF-8 file returns a recoverable error, not a crash."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "bin.dat").write_bytes(b"\xff\xfe\x00binary")

            old_cwd = Path.cwd()
            os.chdir(workspace)
            try:
                # Act
                execution = ReadFileTool().execute({"path": "bin.dat"}, make_context())
            finally:
                os.chdir(old_cwd)

            # Assert — prose informative failure for a well-specified call
            assert execution.status == "error"
            assert str(execution.output) == (
                "Attempt to read file 'bin.dat' failed due to "
                "the file not being valid UTF-8."
            )
            assert execution.metadata["informative"] is True

    def test_04_unit_06_tools_06_read_03_directory_returns_error(self) -> None:
        """Reading a directory returns a recoverable error, not a crash."""
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            (workspace / "sub").mkdir()

            old_cwd = Path.cwd()
            os.chdir(workspace)
            try:
                # Act
                execution = ReadFileTool().execute({"path": "sub"}, make_context())
            finally:
                os.chdir(old_cwd)

            # Assert — prose informative failure for a well-specified call
            assert execution.status == "error"
            assert str(execution.output) == (
                "Attempt to read file 'sub' failed due to the path not being a file."
            )
            assert execution.metadata["informative"] is True

    def test_04_unit_06_tools_06_read_04_line_numbers_zero_padded_to_fixed_width(
        self, isolated_workspace: Path
    ) -> None:
        """Line numbers are zero-padded so every number shares one width."""
        lines = [f"line-{n}" for n in range(1, 12)]
        (isolated_workspace / "sample.py").write_text(
            "\n".join(lines), encoding="utf-8"
        )

        # Act
        execution = ReadFileTool().execute({"path": "sample.py"}, make_context())

        # Assert — 11 lines need two digits, so lines 1-9 gain a leading zero;
        # the output is prefixed with the path header
        expected_body = "\n".join(f"{n:02d}|line-{n}" for n in range(1, 12))
        assert execution.status == "ok"
        assert str(execution.output) == (
            f"Contents of file sample.py (lines 1-11)\n---\n{expected_body}"
        )

    def test_04_unit_06_tools_06_read_05_single_digit_line_numbers_unpadded(
        self, isolated_workspace: Path
    ) -> None:
        """Files where all line numbers fit in one digit gain no padding."""
        (isolated_workspace / "small.py").write_text(
            "alpha\nbeta\ngamma", encoding="utf-8"
        )

        # Act
        execution = ReadFileTool().execute({"path": "small.py"}, make_context())

        # Assert
        assert execution.status == "ok"
        assert str(execution.output) == (
            "Contents of file small.py (lines 1-3)\n---\n1|alpha\n2|beta\n3|gamma"
        )

    def test_04_unit_06_tools_06_read_06_partial_range_pads_to_last_line_width(
        self, isolated_workspace: Path
    ) -> None:
        """Partial reads pad to the width of the last line shown in range."""
        lines = [f"line-{n}" for n in range(1, 101)]
        (isolated_workspace / "big.py").write_text("\n".join(lines), encoding="utf-8")

        # Act
        execution = ReadFileTool().execute(
            {"path": "big.py", "start_line": 98, "end_line": 100}, make_context()
        )

        # Assert — the range ends at line 100 (three digits), so 98 and 99
        # are padded to the same width as 100; the output is prefixed
        # with the path header
        assert execution.status == "ok"
        assert str(execution.output) == (
            "Contents of file big.py (lines 98-100)\n---\n"
            "098|line-98\n099|line-99\n100|line-100"
        )

    def test_04_unit_06_tools_06_read_07_missing_file_reports_informative_failure(
        self,
    ) -> None:
        """Reading a nonexistent file reports a prose informative failure."""
        # Act — resolve fails before any file access
        execution = ReadFileTool().execute({"path": "missing.py"}, make_context())

        # Assert
        assert execution.status == "error"
        assert str(execution.output) == (
            "Attempt to read file 'missing.py' failed due to the file not existing."
        )
        assert execution.metadata["informative"] is True

    def test_04_unit_06_tools_06_read_08_missing_path_argument_is_malformed(
        self,
    ) -> None:
        """A call without the 'path' argument is malformed, not informative."""
        # Act
        execution = ReadFileTool().execute({}, make_context())

        # Assert — no informative flag because the call was not correctly
        # specified
        assert execution.status == "error"
        assert str(execution.output) == "'path' is required"
        assert "informative" not in execution.metadata

    def test_04_unit_06_tools_06_read_09_start_line_beyond_end_reports_informative_failure(
        self, isolated_workspace: Path
    ) -> None:
        """A start_line beyond the last line reports a prose informative failure."""
        (isolated_workspace / "small.py").write_text(
            "alpha\nbeta\ngamma", encoding="utf-8"
        )

        # Act
        execution = ReadFileTool().execute(
            {"path": "small.py", "start_line": 10}, make_context()
        )

        # Assert
        assert execution.status == "error"
        assert str(execution.output) == (
            "Attempt to read file 'small.py' from line 10 failed due to "
            "the file having only 3 lines."
        )
        assert execution.metadata["informative"] is True
