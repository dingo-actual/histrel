#!/usr/bin/env python3
"""
Text-block locator using the 3-tier matching strategy.

    Important Classes:
        BlockMatcher: Locates text blocks using 3-tier matching strategy.
"""

from __future__ import annotations

from collections.abc import Sequence

import rapidfuzz.fuzz

from histrel.tools.patch_engine.errors import (
    AmbiguousMatchError,
    MatchNotFoundError,
)


class BlockMatcher:
    """Locates text blocks in files using 3-tier matching strategy."""

    def __init__(self, file_content: str) -> None:
        """Initialize the matcher.

        Args:
            file_content (str): The full content of the file to search.
        """
        self.lines = file_content.splitlines(keepends=True)
        self.norm_lines = [
            (line_idx, line.strip()) for line_idx, line in enumerate(self.lines)
        ]

    def find(self, search_text: str) -> tuple[int, int]:
        """Find search block and return (start_line, end_line).

        Uses a 3-tier strategy:
        1. Normalized match (ignoring whitespace differences).
        2. Fuzzy match (using rapidfuzz for approximate matching).
        3. Single-line substring containment (fallback).

        Args:
            search_text (str): Text block to search for.

        Returns:
            (tuple[int, int]) (start_line, end_line) indices.

        Raises:
            ValueError: If search_text is empty.
            MatchNotFoundError: If block not found.
            AmbiguousMatchError: If multiple matches found.
        """
        search_lines = search_text.splitlines(keepends=True)
        if not search_lines:
            raise ValueError("Empty search block")

        candidates = self._match_normalized(search_lines)
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise AmbiguousMatchError("", f"Found {len(candidates)} matches")

        match = self._match_fuzzy(search_lines)
        if match:
            return match

        # Fallback: for single-line searches, allow substring containment
        if len(search_lines) == 1:
            needle = search_lines[0].strip()
            if needle:
                for line_idx, line in enumerate(self.lines):
                    if needle in line.strip():
                        return line_idx, line_idx + 1

        raise MatchNotFoundError("", "Search block not found in file")

    @staticmethod
    def _strip_eol(line: str) -> str:
        """Return a line without its line terminator.

        Args:
            line: Line possibly ending in a line terminator.

        Returns:
            (str) The line content with trailing newline characters removed.
        """
        return line.rstrip("\r\n")

    def find_exact(self, search_lines: Sequence[str]) -> list[tuple[int, int]]:
        """Find ranges whose lines equal the search lines exactly.

        Comparison includes leading whitespace; only line terminators are
        ignored. Unlike ``find``, blank lines are significant.

        Args:
            search_lines: Lines of the search block (with line terminators).

        Returns:
            (list[tuple[int, int]]) All matching (start_line, end_line)
            ranges; an empty list means no exact match exists.
        """
        window_size = len(search_lines)
        if window_size == 0:
            return []

        candidates = []
        for start in range(len(self.lines) - window_size + 1):
            if all(
                self._strip_eol(self.lines[start + offset])
                == self._strip_eol(search_lines[offset])
                for offset in range(window_size)
            ):
                candidates.append((start, start + window_size))
        return candidates

    def _match_normalized(self, search_lines: list[str]) -> list[tuple[int, int]]:
        """Sliding window search ignoring whitespace."""
        search_vec = [line.strip() for line in search_lines if line.strip()]
        if not search_vec:
            return []

        file_vec = [
            (line_idx, stripped) for line_idx, stripped in self.norm_lines if stripped
        ]
        window_size = len(search_vec)
        candidates = []

        for window_start in range(len(file_vec) - window_size + 1):
            block_match = self._check_window_match(
                file_vec, search_vec, window_start, window_size
            )
            if block_match:
                start_line = file_vec[window_start][0]
                end_line = file_vec[window_start + window_size - 1][0] + 1
                candidates.append((start_line, end_line))

        return candidates

    def _check_window_match(
        self,
        file_vec: list[tuple[int, str]],
        search_vec: list[str],
        window_start: int,
        window_size: int,
    ) -> bool:
        """Check if window matches search vector."""
        return all(
            file_vec[window_start + offset][1] == search_vec[offset]
            for offset in range(window_size)
        )

    def _match_fuzzy(self, search_lines: list[str]) -> tuple[int, int] | None:
        """RapidFuzz anchored search with longest line as anchor."""
        anchor_result = self._select_anchor_line(search_lines)
        if anchor_result is None:
            return None

        anchor_idx, anchor_txt, anchor_clean = anchor_result
        block_str = "".join(search_lines)
        matches = self._find_fuzzy_candidates(
            anchor_idx, anchor_txt, anchor_clean, block_str, len(search_lines)
        )

        return self._select_best_fuzzy_match(matches)

    def _select_anchor_line(
        self, search_lines: list[str]
    ) -> tuple[int, str, str] | None:
        """Select longest non-empty line as anchor."""
        anchor_idx, anchor_txt = max(
            enumerate(search_lines), key=lambda pair: len(pair[1].strip())
        )
        anchor_clean = anchor_txt.strip()
        if len(anchor_clean) < 5:
            return None
        return anchor_idx, anchor_txt, anchor_clean

    def _find_fuzzy_candidates(
        self,
        anchor_idx: int,
        anchor_txt: str,
        anchor_clean: str,
        block_str: str,
        search_line_count: int,
    ) -> list[tuple[float, tuple[int, int]]]:
        """Find fuzzy match candidates using anchor line."""
        matches = []

        for line_idx, line in enumerate(self.lines):
            if abs(len(line) - len(anchor_txt)) > max(40, int(len(anchor_txt) * 0.75)):
                continue

            score = rapidfuzz.fuzz.ratio(anchor_clean, line.strip())
            if score <= 80.0:
                continue

            window_range = self._calculate_window_range(
                line_idx, anchor_idx, search_line_count
            )
            if window_range is None:
                continue

            window_start, window_end = window_range
            window_str = "".join(self.lines[window_start:window_end])
            block_score = rapidfuzz.fuzz.ratio(block_str, window_str)

            if block_score > 80.0:
                matches.append((block_score, (window_start, window_end)))

        return matches

    def _calculate_window_range(
        self, line_idx: int, anchor_idx: int, search_line_count: int
    ) -> tuple[int, int] | None:
        """Calculate window range for fuzzy matching."""
        window_start = max(0, line_idx - anchor_idx)
        window_end = min(len(self.lines), window_start + search_line_count)

        if (window_end - window_start) < search_line_count * 0.8:
            return None

        return window_start, window_end

    def _select_best_fuzzy_match(
        self, matches: list[tuple[float, tuple[int, int]]]
    ) -> tuple[int, int] | None:
        """Select best fuzzy match, ensuring no ambiguity."""
        matches.sort(key=lambda match_pair: match_pair[0], reverse=True)
        if not matches:
            return None

        best_match = matches[0]
        if len(matches) > 1 and (best_match[0] == matches[1][0]):
            raise AmbiguousMatchError("", "Fuzzy match ambiguous")

        return best_match[1]
