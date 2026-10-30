#!/usr/bin/env python3
"""Unit tests for BlockMatcher three-tier matching."""

from __future__ import annotations

import pytest

from histrel.tools.patch_engine import (
    AmbiguousMatchError,
    BlockMatcher,
    MatchNotFoundError,
)


class Test04Unit01Patch01BlockMatcher:
    """BlockMatcher 3-tier matching strategy."""

    def test_04_unit_01_patch_01_match_01_exact_match(self) -> None:
        # Arrange
        content = "def hello():\n    print('hello')\n    return None\n"
        matcher = BlockMatcher(content)

        # Act
        start, end = matcher.find("def hello():\n    print('hello')")

        # Assert
        assert start == 0
        assert end == 2

    def test_04_unit_01_patch_01_match_02_normalized_match(self) -> None:
        # Arrange — extra whitespace differences
        content = "  def foo():\n      x = 1\n      return x\n"
        matcher = BlockMatcher(content)

        # Act
        start, end = matcher.find("def foo():\n    x = 1")

        # Assert
        assert start == 0
        assert end == 2

    def test_04_unit_01_patch_01_match_03_not_found_raises(self) -> None:
        # Arrange
        matcher = BlockMatcher("print('hello')\n")

        # Act & Assert
        with pytest.raises(MatchNotFoundError):
            matcher.find("def nonexistent():\n    pass")

    def test_04_unit_01_patch_01_match_04_ambiguous_raises(self) -> None:
        # Arrange — two identical blocks
        content = "x = 1\nx = 1\n"
        matcher = BlockMatcher(content)

        # Act & Assert
        with pytest.raises(AmbiguousMatchError):
            matcher.find("x = 1")

    def test_04_unit_01_patch_01_match_05_single_line_substring(self) -> None:
        # Arrange
        content = "import os\nimport sys\n"
        matcher = BlockMatcher(content)

        # Act
        start, end = matcher.find("os")

        # Assert — single-line substring match
        assert start == 0
        assert end == 1

    def test_04_unit_01_patch_01_match_06_empty_search_raises(self) -> None:
        # Arrange
        matcher = BlockMatcher("content\n")

        # Act & Assert
        with pytest.raises(ValueError, match="Empty"):
            matcher.find("")


class Test04Unit01Patch02FindExact:
    """BlockMatcher exact (indentation-significant) matching."""

    def test_04_unit_01_patch_02_find_exact_01_single_exact_match(self) -> None:
        # Arrange
        matcher = BlockMatcher("def hello():\n    print('hi')\n")
        search_lines = "def hello():\n    print('hi')".splitlines(keepends=True)

        # Act
        ranges = matcher.find_exact(search_lines)

        # Assert
        assert ranges == [(0, 2)]

    def test_04_unit_01_patch_02_find_exact_02_indent_difference_no_match(
        self,
    ) -> None:
        # Arrange — search carries different leading whitespace than the file
        matcher = BlockMatcher("def foo():\n      x = 1\n")
        search_lines = "def foo():\n    x = 1".splitlines(keepends=True)

        # Act
        ranges = matcher.find_exact(search_lines)

        # Assert
        assert ranges == []

    def test_04_unit_01_patch_02_find_exact_03_blank_lines_significant(self) -> None:
        # Arrange — the file block contains a blank line
        matcher = BlockMatcher("a\n\nb\n")

        # Act
        without_blank = matcher.find_exact("a\nb".splitlines(keepends=True))
        with_blank = matcher.find_exact("a\n\nb".splitlines(keepends=True))

        # Assert
        assert without_blank == []
        assert with_blank == [(0, 3)]

    def test_04_unit_01_patch_02_find_exact_04_multiple_matches_returned(self) -> None:
        # Arrange
        matcher = BlockMatcher("x = 1\ny\nx = 1\n")
        search_lines = "x = 1".splitlines(keepends=True)

        # Act
        ranges = matcher.find_exact(search_lines)

        # Assert
        assert ranges == [(0, 1), (2, 3)]
