#!/usr/bin/env python3
"""
Edit/create-file tool for the benchmark agent.

    Important Classes:
        EditFileTool: Apply SEARCH/REPLACE patches to edit or create a single file.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from histrel.tools.patch_engine import apply_patch
from histrel.tools.patch_utils import (
    Change,
    DuplicateMarkerError,
    normalize_change_newlines,
    parse_edit_block,
)
from histrel.tools.tool import (
    Tool,
    ToolContext,
    ToolExecution,
    resolve_path,
)


class EditFileTool(Tool):
    """Apply SEARCH/REPLACE patches to edit or create a single file.

    Accepts a file path and a list of SEARCH/REPLACE edit blocks as
    strings (see ``parse_edit_block`` in patch_utils for the exact
    format). Text outside the declared blocks is ignored, newlines in
    the SEARCH and REPLACE text are normalized to the target file's
    line endings, and edits apply atomically. Relative paths resolve
    against CWD; absolute paths are allowed only under /tmp.

    When the target file does not exist, exactly one edit is provided,
    and that edit has an empty SEARCH block, the REPLACE block is
    written as the new file's content. An empty REPLACE block deletes
    the SEARCH block's contents from the file. When a SEARCH block
    matches the file exactly (including indentation), the REPLACE
    block is applied verbatim, preserving its own indentation.
    """

    name = "edit"
    definition: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "edit",
            "description": (
                "Edit or create a single file using SEARCH/REPLACE blocks. "
                "Provide\n'file_path' and a list of 'file_edits' strings, each "
                "one SEARCH/REPLACE\nblock in the format described for "
                "'file_edits'. Newlines in the SEARCH and\nREPLACE text are "
                "normalized to match the file's line endings. When the\ntarget "
                "file does not exist, exactly one edit is provided, and "
                "that\nedit's SEARCH block is empty, the edit's REPLACE block "
                "is written as\nthe new file's content, creating the file.\n"
                "  IMPORTANT: Be careful to use the correct number of escapes "
                "when escaping\n  characters that JSON requires to be escaped "
                'inside string values: double quotes\n  (\\"), '
                "backslashes (\\\\), and control characters like newline "
                "(\\n) and tab\n  (\\t). Do NOT escape single quotes — they "
                "are valid unescaped in JSON strings.\n  Do NOT add "
                "unnecessary backslashes before the double quotes that delimit"
                " JSON\n  keys or string values (except for nested JSON "
                "strings); those quotes are\n  structural and must remain "
                "unescaped."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the file to edit or create.",
                    },
                    "file_edits": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "List of SEARCH/REPLACE edit blocks to apply. Each "
                            "string is one\nedit block in the exact format:\n"
                            "/***SEARCH***/\n"
                            "{text to replace, including newlines}\n"
                            "/***END_SEARCH***/\n"
                            "/***REPLACE***/\n"
                            "{text of replacement, including newlines}\n"
                            "/***END_REPLACE***/\n"
                            "Each marker must appear exactly once per block; "
                            "text outside the\ndeclared SEARCH and REPLACE "
                            "blocks is ignored. The SEARCH text must\nmatch the "
                            "file exactly (including indentation). An empty "
                            "REPLACE block\ndeletes the contents of the SEARCH "
                            "block in the file."
                        ),
                    },
                },
                "required": ["file_path", "file_edits"],
            },
        },
    }
    _EXAMPLE_SINGLE_BLOCK: dict[str, Any] = {
        "file_path": "src/math_utils.py",
        "file_edits": [
            (
                "/***SEARCH***/\n"
                "def divide(a, b):\n"
                "    return a / b\n"
                "/***END_SEARCH***/\n"
                "/***REPLACE***/\n"
                "def divide(a, b):\n"
                "    if b == 0:\n"
                "        raise ValueError('division by zero')\n"
                "    return a / b\n"
                "/***END_REPLACE***/"
            )
        ],
    }
    _EXAMPLE_MULTI_BLOCK: dict[str, Any] = {
        "file_path": "src/report.py",
        "file_edits": [
            (
                "/***SEARCH***/\n"
                "title = 'Report'\n"
                "/***END_SEARCH***/\n"
                "/***REPLACE***/\n"
                "title = 'Quarterly Report'\n"
                "/***END_REPLACE***/"
            ),
            (
                "/***SEARCH***/\n"
                "    print('Report generated')\n"
                "/***END_SEARCH***/\n"
                "/***REPLACE***/\n"
                "    print(f'Report generated: {len(rows)} rows')\n"
                "/***END_REPLACE***/"
            ),
            (
                "/***SEARCH***/\n"
                "def summarize(rows):\n"
                "    # Legacy summary kept only for reference.\n"
                "    lines = []\n"
                "    for row in rows:\n"
                "        lines.append(str(row))\n"
                "    return '\\n'.join(lines)\n"
                "/***END_SEARCH***/\n"
                "/***REPLACE***/\n"
                "/***END_REPLACE***/"
            ),
        ],
    }
    _EXAMPLE_CREATE_FILE: dict[str, Any] = {
        "file_path": "src/new_module.py",
        "file_edits": [
            (
                "/***SEARCH***/\n"
                "/***END_SEARCH***/\n"
                "/***REPLACE***/\n"
                "def hello():\n"
                "    print('hello')\n"
                "/***END_REPLACE***/"
            )
        ],
    }
    examples: list[str] = [
        f'run(name="edit", args="{json.dumps(_EXAMPLE_SINGLE_BLOCK)}")',
        f'run(name="edit", args="{json.dumps(_EXAMPLE_MULTI_BLOCK)}")',
        f'run(name="edit", args="{json.dumps(_EXAMPLE_CREATE_FILE)}")',
    ]

    @staticmethod
    def _resolve_file_path(file_path_str: str) -> Path:
        """Resolve a file path, tolerating nonexistent files.

        If the path exists, resolve it strictly via ``resolve_path``.
        If it does not, resolve the parent directory strictly and
        append the filename, applying the same containment rules.

        Args:
            file_path_str (str): Path from the model.

        Returns:
            (Path) Resolved path (may not exist yet).

        Raises:
            ValueError: If the path or parent is outside the allowed root.
            FileNotFoundError: If the parent directory does not exist.
        """
        try:
            return resolve_path(file_path_str)
        except FileNotFoundError:
            candidate = Path(file_path_str)
            parent_dir = resolve_path(str(candidate.parent))
            return parent_dir / candidate.name

    @staticmethod
    def _try_create_file(
        file_path: Path, file_path_str: str, changes: list[Change]
    ) -> tuple[bool, str] | None:
        """Attempt to create a new file when the path does not exist.

        Creation proceeds only when exactly one edit is provided and
        that edit has an empty search block. The replace content from
        that edit becomes the new file's content.

        Args:
            file_path (Path): Resolved path that does not exist.
            file_path_str (str): Original path string for messages.
            changes (list[Change]): Parsed changes.

        Returns:
            (tuple[bool, str] | None) (success, message) pair, or None if
            creation was not requested (non-empty search or multiple edits).
        """
        if len(changes) != 1 or changes[0].search:
            return None

        try:
            encoded = changes[0].replace.encode("utf-8")
        except UnicodeEncodeError:
            return (
                False,
                (
                    f"Attempt to create file '{file_path_str}' failed due to "
                    "the content not being valid UTF-8."
                ),
            )

        file_path.write_bytes(encoded)
        return (True, f"Created file {file_path_str}")

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        """Execute the edit tool.

        Args:
            arguments (dict[str, Any]): Tool arguments.
            context (ToolContext): Execution context.

        Returns:
            (ToolExecution) Execution result.
        """
        file_path_str = arguments.get("file_path", "")
        file_edits = arguments.get("file_edits")

        argument_error = _validate_edit_arguments(file_path_str, file_edits)
        if argument_error is not None:
            return argument_error

        changes = _parse_edit_changes(file_path_str, file_edits)
        if isinstance(changes, ToolExecution):
            return changes

        file_path = _resolve_edit_target(file_path_str, self._resolve_file_path)
        if isinstance(file_path, ToolExecution):
            return file_path

        if not file_path.exists():
            return _edit_missing_file(
                file_path, file_path_str, changes, self._try_create_file
            )

        if file_path.is_dir():
            return _edit_directory_error(file_path_str)

        return _apply_edit(file_path, file_path_str, changes)


def _validate_edit_arguments(
    file_path_str: Any, file_edits: Any
) -> ToolExecution | None:
    """Validate the required edit arguments.

    Args:
        file_path_str (Any): Raw file_path argument.
        file_edits (Any): Raw file_edits argument.

    Returns:
        (ToolExecution | None) A malformed-call error, or None when valid.
    """
    if not file_path_str:
        return ToolExecution(
            status="error",
            output="'file_path' is required",
            metadata={},
        )
    if not isinstance(file_path_str, str):
        return ToolExecution(
            status="error",
            output="'file_path' must be a string",
            metadata={},
        )
    if not file_edits or not isinstance(file_edits, list):
        return ToolExecution(
            status="error",
            output="'file_edits' is required and must be a non-empty list",
            metadata={},
        )
    return None


def _parse_edit_changes(
    file_path_str: str, file_edits: Any
) -> list[Change] | ToolExecution:
    """Parse SEARCH/REPLACE blocks into Change objects.

    Args:
        file_path_str (str): Original path string for messages.
        file_edits (Any): Raw edit-block strings (already validated as a list).

    Returns:
        (list[Change] | ToolExecution) Parsed changes, or an error result.
    """
    changes: list[Change] = []
    try:
        for idx, block in enumerate(file_edits):
            if not isinstance(block, str):
                raise ValueError(
                    f"file_edits[{idx}] must be a string but got {type(block).__name__}"
                )
            changes.append(parse_edit_block(block))
    except DuplicateMarkerError as exc:
        # A duplicated marker is a well-intentioned call with one
        # specific defect, so it fails informatively to let the
        # model retry with a corrected block.
        return ToolExecution(
            status="error",
            output=(f"Attempt to edit file '{file_path_str}' failed due to {exc}."),
            metadata={"informative": True},
        )
    except ValueError as exc:
        return ToolExecution(
            status="error",
            output=f"Error parsing file_edits: {exc}",
            metadata={},
        )
    return changes


def _resolve_edit_target(
    file_path_str: str, resolver: Callable[[str], Path]
) -> Path | ToolExecution:
    """Resolve the edit target, mapping resolution failures to errors.

    Args:
        file_path_str (str): Original path string for messages.
        resolver (Callable[[str], Path]): Path resolver to apply.

    Returns:
        (Path | ToolExecution) The resolved path, or an error result.
    """
    try:
        return resolver(file_path_str)
    except ValueError:
        # State the containment rule explicitly: the path argument was
        # valid, but only workspace and /tmp targets are editable.
        return ToolExecution(
            status="error",
            output=(
                f"Attempt to edit file '{file_path_str}' failed due to "
                "the path being outside the allowed roots; only edits "
                f"in CWD ({Path.cwd()}) and /tmp are allowed."
            ),
            metadata={"informative": True},
        )
    except FileNotFoundError:
        return ToolExecution(
            status="error",
            output=(
                f"Attempt to edit file '{file_path_str}' failed due to "
                "the parent directory not existing."
            ),
            metadata={"informative": True},
        )


def _edit_missing_file(
    file_path: Path,
    file_path_str: str,
    changes: list[Change],
    create: Callable[[Path, str, list[Change]], tuple[bool, str] | None],
) -> ToolExecution:
    """Handle an edit aimed at a nonexistent file.

    Args:
        file_path (Path): Resolved path that does not exist.
        file_path_str (str): Original path string for messages.
        changes (list[Change]): Parsed changes.
        create (Callable): File-creation helper.

    Returns:
        (ToolExecution) The creation result, or a missing-file error.
    """
    create_result = create(file_path, file_path_str, changes)
    if create_result is not None:
        created, message = create_result
        return ToolExecution(
            status="ok" if created else "error",
            output=message,
            metadata={} if created else {"informative": True},
        )
    return ToolExecution(
        status="error",
        output=(
            f"Attempt to edit file '{file_path_str}' failed due to "
            "the file not existing."
        ),
        metadata={"informative": True},
    )


def _edit_directory_error(file_path_str: str) -> ToolExecution:
    """Return the informative error for an edit aimed at a directory."""
    return ToolExecution(
        status="error",
        output=(
            f"Attempt to edit file '{file_path_str}' failed due to "
            "the path being a directory."
        ),
        metadata={"informative": True},
    )


def _apply_edit(
    file_path: Path, file_path_str: str, changes: list[Change]
) -> ToolExecution:
    """Apply parsed changes to an existing file.

    Args:
        file_path (Path): Resolved existing file path.
        file_path_str (str): Original path string for messages.
        changes (list[Change]): Parsed changes.

    Returns:
        (ToolExecution) The edit result.
    """
    try:
        file_content = file_path.read_bytes().decode("utf-8")
    except UnicodeDecodeError:
        return ToolExecution(
            status="error",
            output=(
                f"Attempt to edit file '{file_path_str}' failed due to "
                "the file not being valid UTF-8."
            ),
            metadata={"path": file_path_str, "informative": True},
        )

    normalized = [normalize_change_newlines(change, file_content) for change in changes]
    try:
        patched_content = apply_patch(str(file_path), normalized)
        file_path.write_bytes(patched_content.encode("utf-8"))
        return ToolExecution(
            status="ok",
            output=f"Edited file {file_path_str}",
            metadata={},
        )
    except Exception as exc:
        # Patch engine errors carry a "[path] " prefix that the prose
        # already names; prefer the bare message when present.
        reason = str(getattr(exc, "msg", exc))
        return ToolExecution(
            status="error",
            output=(f"Attempt to edit file '{file_path_str}' failed due to {reason}."),
            metadata={"informative": True},
        )
