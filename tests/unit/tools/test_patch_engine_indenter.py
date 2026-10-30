#!/usr/bin/env python3
"""Unit tests for Indenter relative indentation reconstruction."""

from __future__ import annotations

from histrel.tools.patch_engine import (
    Indenter,
)


class Test04Unit02Patch02Indenter:
    """Indenter relative indentation reconstruction."""

    def test_04_unit_02_patch_02_indent_01_get_indent(self) -> None:
        # Act & Assert
        assert Indenter.get_indent("    hello") == "    "
        assert Indenter.get_indent("\thello") == "\t"
        assert Indenter.get_indent("hello") == ""

    def test_04_unit_02_patch_02_indent_02_reindent_preserves_relative(self) -> None:
        # Arrange — replace block has 4-space base indent
        replace_str = "    result = compute()\n    return result\n"
        base_indent = "        "  # 8 spaces (deeper nesting)

        # Act
        reindented = Indenter.reindent(replace_str, base_indent)

        # Assert — min indent (4) stripped, base indent (8) applied
        lines = reindented.rstrip("\n").split("\n")
        assert lines[0] == "        result = compute()"
        assert lines[1] == "        return result"

    def test_04_unit_02_patch_02_indent_03_empty_lines_preserved(self) -> None:
        # Arrange
        replace_str = "x = 1\n\ny = 2\n"
        base_indent = "    "

        # Act
        reindented = Indenter.reindent(replace_str, base_indent)

        # Assert — empty line stays empty
        lines = reindented.rstrip("\n").split("\n")
        assert lines[0] == "    x = 1"
        assert lines[1] == ""
        assert lines[2] == "    y = 2"

    def test_04_unit_02_patch_02_indent_04_no_non_empty_lines(self) -> None:
        # Arrange — all empty lines
        replace_str = "\n\n"
        base_indent = "    "

        # Act
        result = Indenter.reindent(replace_str, base_indent)

        # Assert — returned as-is with trailing newline
        assert result == "\n\n"
