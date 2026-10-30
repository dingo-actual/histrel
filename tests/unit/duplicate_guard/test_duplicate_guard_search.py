#!/usr/bin/env python3
"""Unit tests for duplicate search and glob detection."""

from __future__ import annotations

from pathlib import Path

from histrel.core.duplicate_guard import (
    DuplicateCallGuard,
)
from tests.unit.duplicate_guard.helpers import (
    glob_args,
    search_args,
)


class Test04Unit36DuplicateGuard03Search:
    """search and glob: duplicate detection for repeated patterns."""

    def _make_tree(self, tmp_path: Path) -> None:
        src = tmp_path / "src"
        src.mkdir()
        (src / "a.py").write_text("# TODO fix\n")

    def test_04_unit_36_dupguard_03_search_01_next_turn_repeat_denied(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        monkeypatch.chdir(tmp_path)
        self._make_tree(tmp_path)
        guard = DuplicateCallGuard()
        args = search_args()

        guard.record(tool_name="search", arguments=args, llm_call_index=1)

        # Act
        verdict = guard.check(tool_name="search", arguments=args, llm_call_index=2)

        # Assert
        assert verdict is not None

    def test_04_unit_36_dupguard_03_search_02_allowed_after_window_expiry(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        monkeypatch.chdir(tmp_path)
        self._make_tree(tmp_path)
        guard = DuplicateCallGuard()
        args = search_args()

        guard.record(tool_name="search", arguments=args, llm_call_index=1)

        # Act
        verdict = guard.check(tool_name="search", arguments=args, llm_call_index=6)

        # Assert
        assert verdict is None

    def test_04_unit_36_dupguard_03_search_03_allowed_after_edit(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange — a successful edit means file contents may have
        # changed, so repeated searches are re-run.
        monkeypatch.chdir(tmp_path)
        self._make_tree(tmp_path)
        guard = DuplicateCallGuard()
        args = search_args()

        guard.record(tool_name="search", arguments=args, llm_call_index=1)
        guard.record(
            tool_name="edit",
            arguments={"file_path": "src/a.py", "file_edits": []},
            llm_call_index=2,
        )

        # Act
        verdict = guard.check(
            tool_name="search",
            arguments=search_args(),
            llm_call_index=3,
        )

        # Assert
        assert verdict is None

    def test_04_unit_36_dupguard_03_search_04_different_pattern_allowed(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        monkeypatch.chdir(tmp_path)
        self._make_tree(tmp_path)
        guard = DuplicateCallGuard()
        other_args = {"pattern": "FIXME", "path": "src", "glob": "*.py"}
        args = search_args()

        guard.record(tool_name="search", arguments=args, llm_call_index=1)

        # Act
        verdict = guard.check(
            tool_name="search", arguments=other_args, llm_call_index=2
        )

        # Assert
        assert verdict is None

    def test_04_unit_36_dupguard_03_search_05_search_glob_filter_distinguishes(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange — the same pattern under a different glob filter is a
        # different search.
        monkeypatch.chdir(tmp_path)
        self._make_tree(tmp_path)
        guard = DuplicateCallGuard()
        other_args = {"pattern": "TODO", "path": "src", "glob": "*.md"}
        args = search_args()

        guard.record(tool_name="search", arguments=args, llm_call_index=1)

        # Act
        verdict = guard.check(
            tool_name="search", arguments=other_args, llm_call_index=2
        )

        # Assert
        assert verdict is None

    def test_04_unit_36_dupguard_03_search_06_glob_repeat_denied(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        monkeypatch.chdir(tmp_path)
        self._make_tree(tmp_path)
        guard = DuplicateCallGuard()
        args = glob_args()

        guard.record(tool_name="glob", arguments=args, llm_call_index=1)

        # Act
        verdict = guard.check(tool_name="glob", arguments=args, llm_call_index=2)

        # Assert
        assert verdict is not None

    def test_04_unit_36_dupguard_03_search_07_search_and_glob_not_confused(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange — a search and a glob for the same pattern and path
        # are different tools and never duplicate each other.
        monkeypatch.chdir(tmp_path)
        self._make_tree(tmp_path)
        guard = DuplicateCallGuard()

        guard.record(
            tool_name="search",
            arguments={"pattern": "*.py", "path": "src"},
            llm_call_index=1,
        )

        # Act
        verdict = guard.check(
            tool_name="glob",
            arguments={"pattern": "*.py", "path": "src"},
            llm_call_index=2,
        )

        # Assert
        assert verdict is None
