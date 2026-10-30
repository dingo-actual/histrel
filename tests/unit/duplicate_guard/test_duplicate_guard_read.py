#!/usr/bin/env python3
"""Unit tests for duplicate read detection."""

from __future__ import annotations

from pathlib import Path

import pytest

from histrel.core.duplicate_guard import (
    DuplicateCallGuard,
    DuplicateVerdict,
)
from tests.unit.duplicate_guard.helpers import (
    read_args,
)


class Test04Unit36DuplicateGuard01Read:
    """read: duplicate detection for repeated line ranges."""

    def test_04_unit_36_dupguard_01_read_01_same_turn_repeat_denied(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange — a successful read, then the same call again in the
        # same LLM call batch.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "main.py").write_text("print('a')\n")
        guard = DuplicateCallGuard()
        args = read_args()

        guard.record(tool_name="read", arguments=args, llm_call_index=4)

        # Act
        verdict = guard.check(tool_name="read", arguments=args, llm_call_index=4)

        # Assert
        assert isinstance(verdict, DuplicateVerdict)
        assert verdict.reason

    def test_04_unit_36_dupguard_01_read_02_next_turn_repeat_denied(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        monkeypatch.chdir(tmp_path)
        (tmp_path / "main.py").write_text("print('a')\n")
        guard = DuplicateCallGuard()
        args = read_args()

        guard.record(tool_name="read", arguments=args, llm_call_index=1)

        # Act
        verdict = guard.check(tool_name="read", arguments=args, llm_call_index=2)

        # Assert
        assert verdict is not None

    def test_04_unit_36_dupguard_01_read_03_denied_through_window_edge(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange — four LLM calls have passed since the recorded read;
        # the recorded call is still one of the past five.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "main.py").write_text("print('a')\n")
        guard = DuplicateCallGuard()
        args = read_args()

        guard.record(tool_name="read", arguments=args, llm_call_index=1)

        # Act
        verdict = guard.check(tool_name="read", arguments=args, llm_call_index=5)

        # Assert
        assert verdict is not None

    def test_04_unit_36_dupguard_01_read_04_allowed_after_window_expiry(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange — five LLM calls have passed since the recorded read,
        # so the read is no longer within the past five calls.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "main.py").write_text("print('a')\n")
        guard = DuplicateCallGuard()
        args = read_args()

        guard.record(tool_name="read", arguments=args, llm_call_index=1)

        # Act
        verdict = guard.check(tool_name="read", arguments=args, llm_call_index=6)

        # Assert
        assert verdict is None

    def test_04_unit_36_dupguard_01_read_05_allowed_after_file_edited(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange — the file content changed since the recorded read,
        # so the read is no longer a duplicate.
        monkeypatch.chdir(tmp_path)
        (tmp_path / "main.py").write_text("print('a')\n")
        guard = DuplicateCallGuard()
        args = read_args()

        guard.record(tool_name="read", arguments=args, llm_call_index=1)
        (tmp_path / "main.py").write_text("print('changed')\n")

        # Act
        verdict = guard.check(tool_name="read", arguments=args, llm_call_index=2)

        # Assert
        assert verdict is None

    def test_04_unit_36_dupguard_01_read_06_different_range_allowed(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        monkeypatch.chdir(tmp_path)
        (tmp_path / "main.py").write_text("a\nb\nc\nd\ne\n")
        guard = DuplicateCallGuard()

        guard.record(
            tool_name="read",
            arguments={"path": "main.py", "start_line": 1, "end_line": 5},
            llm_call_index=1,
        )

        # Act
        verdict = guard.check(
            tool_name="read",
            arguments={"path": "main.py", "start_line": 2, "end_line": 3},
            llm_call_index=2,
        )

        # Assert
        assert verdict is None

    def test_04_unit_36_dupguard_01_read_07_different_file_allowed(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        monkeypatch.chdir(tmp_path)
        (tmp_path / "main.py").write_text("print('a')\n")
        (tmp_path / "other.py").write_text("print('b')\n")
        guard = DuplicateCallGuard()
        other_args = {"path": "other.py", "start_line": 1, "end_line": 1}
        args = read_args()

        guard.record(tool_name="read", arguments=args, llm_call_index=1)

        # Act
        verdict = guard.check(tool_name="read", arguments=other_args, llm_call_index=2)

        # Assert
        assert verdict is None

    def test_04_unit_36_dupguard_01_read_08_unresolvable_path_allowed(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange — a path that cannot be resolved is never a duplicate;
        # the tool itself reports the resolution failure.
        monkeypatch.chdir(tmp_path)
        guard = DuplicateCallGuard()

        # Act
        verdict = guard.check(
            tool_name="read",
            arguments={"path": "missing.py", "start_line": 1, "end_line": 1},
            llm_call_index=1,
        )

        # Assert
        assert verdict is None


class Test04Unit36DuplicateGuard07Window:
    """Configurable recall window for window-based duplicate rules."""

    def _make_tree(self, tmp_path: Path) -> None:
        tmp_path.mkdir(exist_ok=True)
        (tmp_path / "main.py").write_text("print('a')\n")

    def test_04_unit_36_dupguard_07_win_01_two_call_window_boundary(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange — with window 2 a repeat is denied only while fewer
        # than 2 LLM calls separate it from the recorded call.
        monkeypatch.chdir(tmp_path)
        self._make_tree(tmp_path)
        guard = DuplicateCallGuard(window=2)
        args = read_args()

        guard.record(tool_name="read", arguments=args, llm_call_index=1)

        # Act
        within = guard.check(tool_name="read", arguments=args, llm_call_index=2)
        outside = guard.check(tool_name="read", arguments=args, llm_call_index=3)

        # Assert
        assert within is not None
        assert outside is None

    def test_04_unit_36_dupguard_07_win_02_zero_disables_window_rules(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange — window 0 disables window-based duplicates.
        monkeypatch.chdir(tmp_path)
        self._make_tree(tmp_path)
        guard = DuplicateCallGuard(window=0)
        args = read_args()

        guard.record(tool_name="read", arguments=args, llm_call_index=1)

        # Act
        verdict = guard.check(tool_name="read", arguments=args, llm_call_index=1)

        # Assert
        assert verdict is None

    def test_04_unit_36_dupguard_07_win_03_negative_window_rejected(self) -> None:
        # Act & Assert
        with pytest.raises(ValueError, match="window"):
            DuplicateCallGuard(window=-1)
