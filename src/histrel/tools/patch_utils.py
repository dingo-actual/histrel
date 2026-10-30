#!/usr/bin/env python3
"""
Utilities for patch operations.

Provides Pydantic models and helper functions for patch transactions,
including parsing of SEARCH/REPLACE edit blocks and newline
normalization to the target file's line endings.

    Important Classes:
        Change: Represents a single search/replace change.
        PendingChange: Represents a validated and located change.
        DuplicateMarkerError: Raised when an edit block contains a
            marker more than once.

    Important Variables:
        SEARCH_MARKER (str): Marker that opens the search text in an
            edit block.
        END_SEARCH_MARKER (str): Marker that closes the search text.
        REPLACE_MARKER (str): Marker that opens the replace text.
        END_REPLACE_MARKER (str): Marker that closes the replace text.

    Important Functions:
        parse_edit_block(block) -> Change: Parse one SEARCH/REPLACE
            edit block string into a Change.
        normalize_newlines(text, newline) -> str: Convert every line
            terminator in text to one newline style.
        normalize_change_newlines(change, file_content) -> Change:
            Normalize a change's block newlines to the file's line
            endings.
        detect_newline_style(content) -> str: Detect the line ending
            style used in text content.
"""

from __future__ import annotations

from loguru import logger
from pydantic import BaseModel, Field

log = logger.bind(
    component="bench_agent.tools.patch_utils", event="edit_block_stray_text_ignored"
)

# Markers delimiting the search and replace text inside an edit block.
SEARCH_MARKER = "/***SEARCH***/"
END_SEARCH_MARKER = "/***END_SEARCH***/"
REPLACE_MARKER = "/***REPLACE***/"
END_REPLACE_MARKER = "/***END_REPLACE***/"

_ALL_MARKERS: tuple[str, ...] = (
    SEARCH_MARKER,
    END_SEARCH_MARKER,
    REPLACE_MARKER,
    END_REPLACE_MARKER,
)


class Change(BaseModel):
    """Represents a single search/replace change.

    Instance Attributes:
        search (str): Text block to search for.
        replace (str): Replacement text block.
    """

    search: str = Field(..., description="Text block to search for")
    replace: str = Field(..., description="Replacement text block")


class PendingChange(BaseModel):
    """Represents a validated and located change ready for application.

    Instance Attributes:
        start_line (int): Starting line index (0-based).
        end_line (int): Ending line index (exclusive).
        new_text (str): Processed replacement text.
    """

    start_line: int = Field(..., description="Starting line index (0-based)")
    end_line: int = Field(..., description="Ending line index (exclusive)")
    new_text: str = Field(..., description="Processed replacement text")


class DuplicateMarkerError(ValueError):
    """Raised when an edit block contains a marker more than once."""


def _strip_leading_newline(text: str) -> str:
    """Remove one line terminator from the start of text.

    Args:
        text (str): Text that may start with a line terminator.

    Returns:
        (str) Text without its leading line terminator, or the text
        unchanged when it does not start with one.
    """
    for eol in ("\r\n", "\n", "\r"):
        if text.startswith(eol):
            return text[len(eol) :]
    return text


def parse_edit_block(block: str) -> Change:
    """Parse one SEARCH/REPLACE edit block string into a Change.

    The block has the format::

        /***SEARCH***/
        {text to replace, including newlines}
        /***END_SEARCH***/
        /***REPLACE***/
        {text of replacement, including newlines}
        /***END_REPLACE***/

    Every marker must occur exactly once: a marker occurring more than
    once raises ``DuplicateMarkerError``, and a missing marker raises
    ``ValueError``. Text outside the declared SEARCH and REPLACE
    blocks is ignored with a logged warning, because models
    occasionally wrap the block in prose or code fences. The marker
    lines' own line terminators are not part of the block content:
    one terminator after each opening marker is stripped, and the
    terminator before each closing marker terminates the block's last
    line. A whitespace-only search block is treated as empty so the
    file-creation form also tolerates a blank line after the SEARCH
    marker.

    Args:
        block (str): Edit block string in the SEARCH/REPLACE format.

    Returns:
        (Change) Parsed change with search and replace text.

    Raises:
        DuplicateMarkerError: If any marker occurs more than once.
        ValueError: If any required marker is missing.
    """
    duplicate_markers = [marker for marker in _ALL_MARKERS if block.count(marker) > 1]
    if duplicate_markers:
        marker_list = ", ".join(f"'{marker}'" for marker in duplicate_markers)
        raise DuplicateMarkerError(
            f"multiple occurrences of marker(s) {marker_list} in a file_edits block"
        )

    missing_markers = [marker for marker in _ALL_MARKERS if marker not in block]
    if missing_markers:
        raise ValueError(f"edit block must contain '{missing_markers[0]}'")

    i_search = block.find(SEARCH_MARKER)
    i_end_search = block.find(END_SEARCH_MARKER)
    i_replace = block.find(REPLACE_MARKER)
    i_end_replace = block.find(END_REPLACE_MARKER)

    _warn_stray_text(
        before=block[:i_search],
        between=block[i_end_search + len(END_SEARCH_MARKER) : i_replace],
        after=block[i_end_replace + len(END_REPLACE_MARKER) :],
    )

    search_text = _strip_leading_newline(
        block[i_search + len(SEARCH_MARKER) : i_end_search]
    )
    replace_text = _strip_leading_newline(
        block[i_replace + len(REPLACE_MARKER) : i_end_replace]
    )
    if not search_text.strip():
        search_text = ""
    return Change(search=search_text, replace=replace_text)


def _warn_stray_text(before: str, between: str, after: str) -> None:
    """Log a warning for text outside the declared edit blocks.

    Each segment excludes its leading marker-line terminator, so a
    well-formed block — where the markers sit on adjacent lines —
    produces no warning.

    Args:
        before (str): Text preceding the SEARCH marker.
        between (str): Text between the END_SEARCH and REPLACE markers.
        after (str): Text following the END_REPLACE marker.
    """
    segments: list[str] = []
    if before.strip():
        segments.append(f"before the SEARCH marker: {before.strip()!r}")
    between_stripped = _strip_leading_newline(between).strip()
    if between_stripped:
        segments.append(
            f"between the END_SEARCH and REPLACE markers: {between_stripped!r}"
        )
    after_stripped = _strip_leading_newline(after).strip()
    if after_stripped:
        segments.append(f"after the END_REPLACE marker: {after_stripped!r}")
    if segments:
        log.warning(
            "Ignored text outside the declared edit blocks",
            stray_segments=segments,
        )


def normalize_newlines(text: str, newline: str) -> str:
    """Convert every line terminator in text to one newline style.

    Args:
        text (str): Text possibly mixing CRLF, LF, and CR terminators.
        newline (str): Target terminator ("\\r\\n", "\\n", or "\\r").

    Returns:
        (str) Text with all line terminators converted to newline.
    """
    flattened = text.replace("\r\n", "\n").replace("\r", "\n")
    return flattened if newline == "\n" else flattened.replace("\n", newline)


def normalize_change_newlines(change: Change, file_content: str) -> Change:
    """Normalize a change's block newlines to the file's line endings.

    Args:
        change (Change): Change whose search and replace text may use
            different line endings than the target file.
        file_content (str): Full content of the target file.

    Returns:
        (Change) Copy with search and replace text rewritten to the
        file's detected newline style.
    """
    newline = detect_newline_style(file_content)
    return change.model_copy(
        update={
            "search": normalize_newlines(change.search, newline),
            "replace": normalize_newlines(change.replace, newline),
        }
    )


def detect_newline_style(content: str) -> str:
    """Detect the most frequent line ending style in text content.

    Counts CRLF pairs, bare CR terminators (CR not followed by LF),
    and bare LF terminators (LF not preceded by CR), then returns the
    style with the highest count. Ties break in CRLF, LF, CR order,
    matching the precedence used before count-based detection.

    Args:
        content (str): Text content to analyze.

    Returns:
        (str) The dominant newline style: '\\r\\n' (CRLF), '\\n' (LF),
        or '\\r' (CR). Returns '\\n' when the content has no line
        terminators.
    """
    crlf_count = content.count("\r\n")
    lf_count = content.count("\n") - crlf_count
    cr_count = content.count("\r") - crlf_count
    if crlf_count == 0 and lf_count == 0 and cr_count == 0:
        return "\n"
    if crlf_count >= lf_count and crlf_count >= cr_count:
        return "\r\n"
    if lf_count >= cr_count:
        return "\n"
    return "\r"
