#!/usr/bin/env python3
"""
Atomic patch application with validation and conflict detection.

    Important Classes:
        PatchTransaction: Manages atomic patch application to files.

    Important Functions:
        apply_patch(file_path, changes) -> str: Apply patch and return result.
"""

from __future__ import annotations

from pathlib import Path

from histrel.tools.patch_engine.errors import (
    AmbiguousMatchError,
    OverlapError,
    PatchError,
)
from histrel.tools.patch_engine.indenter import Indenter
from histrel.tools.patch_engine.matcher import BlockMatcher
from histrel.tools.patch_utils import (
    Change,
    PendingChange,
    detect_newline_style,
    normalize_newlines,
)


class PatchTransaction:
    """Manages atomic patch application with validation and conflict detection."""

    def __init__(self, file_path: str) -> None:
        """Initialize transaction for a file.

        Args:
            file_path (str): Path to the file to patch.

        Raises:
            PatchError: If file cannot be read.
        """
        try:
            self.content = Path(file_path).read_bytes().decode("utf-8")
        except FileNotFoundError as err:
            raise PatchError(file_path, "File not found") from err
        except Exception as err:
            raise PatchError(file_path, f"Cannot read file: {err}") from err

        self.path = file_path
        self.newline_style = detect_newline_style(self.content)
        self.matcher = BlockMatcher(self.content)
        self.pending_ops: list[PendingChange] = []

    def plan(self, changes: list[Change]) -> None:
        """Validate and locate all changes without modifying the file.

        Args:
            changes (list[Change]): List of changes to validate and locate.

        Raises:
            PatchError: If any change cannot be located or validated.
            OverlapError: If changes overlap.
        """
        for change in changes:
            pending = self._locate_and_prepare_change(change)
            self.pending_ops.append(pending)

        self._validate_no_overlaps()

    def commit(self) -> str:
        """Apply all planned changes and return modified content.

        Returns:
            Modified file content with original newline style preserved.
        """
        self.pending_ops.sort(key=lambda op: op.start_line, reverse=True)
        working_lines = self.content.splitlines(keepends=True)

        for operation in self.pending_ops:
            working_lines = self._apply_single_operation(operation, working_lines)

        return "".join(working_lines)

    def _locate_and_prepare_change(self, change: Change) -> PendingChange:
        """Locate change in file and prepare replacement text.

        Args:
            change (Change): Change to locate and prepare.

        Returns:
            PendingChange ready for application.

        Raises:
            PatchError: If change cannot be located.
        """
        search_lines = change.search.splitlines(keepends=True)
        exact_ranges = self.matcher.find_exact(search_lines)

        if len(exact_ranges) > 1:
            raise AmbiguousMatchError(
                self.path, f"Found {len(exact_ranges)} exact matches"
            )

        if exact_ranges:
            # Exact match (including indentation): the replace block is
            # trusted verbatim, because it was written against the real
            # file content. Relative re-indentation only applies when the
            # search block did not match the file's exact whitespace.
            start_line, end_line = exact_ranges[0]
            matched_block = self.matcher.lines[start_line:end_line]
            final_text = self._verbatim_new_text(change.replace, matched_block[-1])
        else:
            try:
                start_line, end_line = self.matcher.find(change.search)
            except PatchError as err:
                raise type(err)(self.path, str(err)) from err

            matched_block = self.matcher.lines[start_line:end_line]
            base_indent = Indenter.get_indent(matched_block[0])
            final_text = Indenter.reindent(change.replace, base_indent)

        return PendingChange(
            start_line=start_line, end_line=end_line, new_text=final_text
        )

    @staticmethod
    def _verbatim_new_text(replace_str: str, matched_last_line: str) -> str:
        """Prepare a verbatim replacement for an exactly matched block.

        The replace block keeps its own indentation. A newline is appended
        only when the replaced block's last line was terminated but the
        replacement is not, so replacement lines never join. The newline
        style itself is normalized by ``commit``.

        Args:
            replace_str: Replacement text as provided by the caller.
            matched_last_line: Last line of the matched block in the file.

        Returns:
            (str) Replacement text with (commit-safe) line termination.
        """
        if replace_str.endswith(("\n", "\r")) or not replace_str:
            return replace_str
        if matched_last_line.endswith(("\n", "\r")):
            return replace_str + "\n"
        return replace_str

    def _validate_no_overlaps(self) -> None:
        """Check for overlapping changes.

        Raises:
            OverlapError: If any changes overlap.
        """
        sorted_ops = sorted(self.pending_ops, key=lambda op: op.start_line)

        for idx in range(len(sorted_ops) - 1):
            current = sorted_ops[idx]
            next_op = sorted_ops[idx + 1]
            if current.end_line > next_op.start_line:
                raise OverlapError(
                    self.path, f"Overlapping change at line {current.end_line}"
                )

    def _apply_single_operation(
        self, operation: PendingChange, working_lines: list[str]
    ) -> list[str]:
        """Apply a single operation to working lines.

        Args:
            operation: Operation to apply.
            working_lines: Current state of file lines.

        Returns:
            Updated working lines.
        """
        new_lines_raw = operation.new_text.splitlines(keepends=True)

        if self.newline_style != "\n":
            # Idempotent conversion: lines already in the target style
            # (normalized upstream) pass through unchanged.
            new_lines = [
                normalize_newlines(line, self.newline_style) for line in new_lines_raw
            ]
        else:
            new_lines = new_lines_raw

        working_lines[operation.start_line : operation.end_line] = new_lines
        return working_lines


def apply_patch(file_path: str, changes: list[Change]) -> str:
    """Apply patches to a file and return the result.

    Args:
        file_path (str): Path to the file to patch.
        changes (list[Change]): List of changes to apply.

    Returns:
        Modified file content.

    Raises:
            PatchError: If patching fails.
    """
    transaction = PatchTransaction(file_path)
    transaction.plan(changes)
    return transaction.commit()
