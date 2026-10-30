#!/usr/bin/env python3
"""Unit tests for duplicate list_dir detection."""

from __future__ import annotations

from pathlib import Path

from histrel.core.duplicate_guard import (
    DuplicateCallGuard,
)
from tests.unit.duplicate_guard.helpers import (
    list_dir_args,
)


class Test04Unit36DuplicateGuard02ListDir:
    """list_dir: duplicate detection for repeated directory listings."""

    def _make_tree(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("a\n")
        (src / "b.py").write_text("b\n")

    def test_04_unit_36_dupguard_02_listdir_01_next_turn_repeat_denied(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        monkeypatch.chdir(tmp_path)
        self._make_tree(tmp_path)
        guard = DuplicateCallGuard()
        args = list_dir_args()

        guard.record(tool_name="list_dir", arguments=args, llm_call_index=1)

        # Act
        verdict = guard.check(tool_name="list_dir", arguments=args, llm_call_index=2)

        # Assert
        assert verdict is not None

    def test_04_unit_36_dupguard_02_listdir_02_allowed_after_window_expiry(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        monkeypatch.chdir(tmp_path)
        self._make_tree(tmp_path)
        guard = DuplicateCallGuard()
        args = list_dir_args()

        guard.record(tool_name="list_dir", arguments=args, llm_call_index=1)

        # Act
        verdict = guard.check(tool_name="list_dir", arguments=args, llm_call_index=6)

        # Assert
        assert verdict is None

    def test_04_unit_36_dupguard_02_listdir_03_allowed_after_file_added(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange — a file was added after the recorded listing, so the
        # listing result would differ.
        monkeypatch.chdir(tmp_path)
        self._make_tree(tmp_path)
        guard = DuplicateCallGuard()
        args = list_dir_args()

        guard.record(tool_name="list_dir", arguments=args, llm_call_index=1)
        (tmp_path / "src" / "c.py").write_text("c\n")

        # Act
        verdict = guard.check(tool_name="list_dir", arguments=args, llm_call_index=2)

        # Assert
        assert verdict is None

    def test_04_unit_36_dupguard_02_listdir_04_allowed_after_file_removed(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange — a file was removed after the recorded listing.
        monkeypatch.chdir(tmp_path)
        self._make_tree(tmp_path)
        guard = DuplicateCallGuard()
        args = list_dir_args()

        guard.record(tool_name="list_dir", arguments=args, llm_call_index=1)
        (tmp_path / "src" / "a.py").unlink()

        # Act
        verdict = guard.check(tool_name="list_dir", arguments=args, llm_call_index=2)

        # Assert
        assert verdict is None

    def test_04_unit_36_dupguard_02_listdir_05_different_directory_allowed(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        monkeypatch.chdir(tmp_path)
        self._make_tree(tmp_path)
        (tmp_path / "tests").mkdir()
        guard = DuplicateCallGuard()
        args = list_dir_args()

        guard.record(tool_name="list_dir", arguments=args, llm_call_index=1)

        # Act
        verdict = guard.check(
            tool_name="list_dir",
            arguments={"path": "tests"},
            llm_call_index=2,
        )

        # Assert
        assert verdict is None
