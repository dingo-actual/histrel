#!/usr/bin/env python3
"""
Indentation reconstruction for replacement blocks.

    Important Classes:
        Indenter: Handles indentation reconstruction for replacement blocks.
"""

from __future__ import annotations


class Indenter:
    """Handles indentation reconstruction for replacement blocks.

    Uses a relative indentation algorithm that preserves the structural
    relationship between search and replace blocks. The minimum indentation
    from the replacement block is stripped, and the base indentation from
    the matched file location is applied to all lines.
    """

    @staticmethod
    def get_indent(line: str) -> str:
        """Returns leading whitespace string.

        Args:
            line: Line of text to extract indentation from.

        Returns:
            Leading whitespace from the line.
        """
        return line[: len(line) - len(line.lstrip())]

    @staticmethod
    def reindent(replace_str: str, base_indent: str) -> str:
        """Reindent replacement text using relative indentation algorithm.

        Strips the minimum indentation from the replacement block and
        applies the base indentation from the matched location.

        Args:
            replace_str: Replacement text to reindent.
            base_indent: Base indentation from matched location (first line).

        Returns:
            Reindented replacement text with trailing newline.
        """
        lines = replace_str.splitlines()
        non_empty = [line for line in lines if line.strip()]
        if not non_empty:
            if not replace_str:
                # Deletion: an empty replace block removes the matched range.
                return ""
            return replace_str if replace_str.endswith("\n") else replace_str + "\n"

        min_indent = Indenter._calculate_min_indent(non_empty)
        reindented_lines = Indenter._apply_reindentation(lines, min_indent, base_indent)

        return "\n".join(reindented_lines) + "\n"

    @staticmethod
    def _calculate_min_indent(non_empty_lines: list[str]) -> int:
        """Calculate minimum leading whitespace across non-empty lines.

        Args:
            non_empty_lines: List of non-empty lines.

        Returns:
            Minimum indentation level in spaces.
        """
        return min(len(line) - len(line.lstrip()) for line in non_empty_lines)

    @staticmethod
    def _apply_reindentation(
        lines: list[str], min_indent: int, base_indent: str
    ) -> list[str]:
        """Apply reindentation to all lines.

        Strips min_indent from each line and applies base_indent.

        Args:
            lines: All lines including empty ones.
            min_indent: Minimum indentation to strip.
            base_indent: Base indentation to apply.

        Returns:
            Reindented lines.
        """
        result = []
        for line in lines:
            if not line.strip():
                result.append("")
                continue

            relative_content = line[min_indent:]
            result.append(base_indent + relative_content)

        return result
