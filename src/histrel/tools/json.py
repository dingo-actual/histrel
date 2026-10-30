#!/usr/bin/env python3
"""
JSON parsing and repair for tool-call argument strings.

LLM tool calls sometimes carry argument strings with repairable JSON
malformations: over-escaped double quotes, invalid single-quote
escapes, or a JSON object wrapped in quotes as a string value. This
module parses such strings, applying the smallest repair that yields
a JSON object, and records each repair as a structured log event.

    Important Functions:
        repair_and_parse(raw, llm_fix) -> dict[str, Any] | str: Parse raw
            JSON arguments, repairing fixable malformations. Returns
            the parsed dict or a descriptive error string. When every
            local repair fails and an ``llm_fix`` hook is supplied, the
            hook (the fix_json_arguments LLM routine, wired by the
            runtime) is tried as a final fallback: its output must
            itself survive the same local repairs, else the original
            error string is returned.
        repair_edit_arguments(raw: str) -> dict[str, Any] | str:
            Repair edit-tool arguments: string-level repairs, nested
            repair of a file_edits value emitted as a JSON string, and
            structural validation of what cannot be repaired.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from loguru import logger

log = logger.bind(component="bench_agent.tools.json", event="json_repaired")

# A failing LLM repair routine is not a completed repair, so its
# warning carries its own event.
llm_repair_log = logger.bind(
    component="bench_agent.tools.json", event="json_llm_repair_failed"
)

# (pattern, replacement, repair description), applied in this order
# when a raw string fails to parse as-is.
_ESCAPE_FIXES: tuple[tuple[str, str, str], ...] = (
    ("\\'", "'", "removed invalid single-quote escapes"),
    ('\\"', '"', "removed over-escaped double quotes"),
)


def _parse_with_repairs(raw: str) -> tuple[Any, list[str]] | None:
    """Parse a string, applying escape repairs until one candidate parses.

    Candidates are tried in order of increasing repair, so a string
    that parses as-is is never modified. A repair counts only when it
    yields parseable JSON; repairs whose result still fails to parse
    are discarded.

    Args:
        raw (str): Raw string to parse.

    Returns:
        (tuple[Any, list[str]] | None) The parsed value and the
        descriptions of the repairs applied, or None when no candidate
        parses.
    """
    candidates: list[tuple[str, list[str]]] = [(raw, [])]
    applied: list[str] = []
    candidate = raw
    for pattern, replacement, description in _ESCAPE_FIXES:
        if pattern not in candidate:
            continue
        candidate = candidate.replace(pattern, replacement)
        applied = [*applied, description]
        candidates.append((candidate, applied))

    for candidate, applied in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return parsed, applied
    return None


def _repair_json(raw: str) -> dict[str, Any] | str:
    """Parse raw JSON, applying every local (non-LLM) repair.

    Repairs applied, in order of increasing intervention: over-escaped
    double quotes and invalid single-quote escapes during parsing, then
    unwrapping of a JSON object that was emitted wrapped in quotes as a
    string value. Input that remains unparseable, or that does not
    describe a JSON object, cannot be repaired and yields a descriptive
    error string.

    Args:
        raw (str): Raw JSON arguments string from the model.

    Returns:
        (dict[str, Any] | str) The parsed arguments dict on success, or
        a descriptive error string when no JSON object can be
        recovered.
    """
    try:
        parsed: Any = json.loads(raw)
        applied: list[str] = []
    except json.JSONDecodeError as exc:
        repaired = _parse_with_repairs(raw)
        if repaired is None:
            return f"Invalid JSON: {exc.msg} at position {exc.pos}"
        parsed, applied = repaired

    if isinstance(parsed, str):
        inner = _parse_with_repairs(parsed)
        if inner is None or not isinstance(inner[0], dict):
            name = type(inner[0]).__name__ if inner is not None else "a plain string"
            return (
                f"Expected a JSON object (dict) but got a string containing "
                f"{name}. Provide a bare JSON object."
            )
        parsed, inner_applied = inner
        applied = [*applied, *inner_applied, "unwrapped a string-wrapped JSON object"]

    if not isinstance(parsed, dict):
        return f"Expected a JSON object (dict) but got {type(parsed).__name__}"

    if applied:
        log.info("Repaired malformed JSON tool arguments", repairs=applied)

    return parsed


def repair_and_parse(
    raw: str,
    *,
    llm_fix: Callable[[str], str] | None = None,
) -> dict[str, Any] | str:
    """Parse raw JSON arguments, repairing fixable malformations.

    When every local repair fails and an ``llm_fix`` hook is supplied,
    the hook — the fix_json_arguments LLM routine, wired by the runtime
    with the called tool's schema — is tried as a final fallback: it
    receives the raw string, and its output must itself survive the
    same local repairs and describe a JSON object. A fallback output
    that still fails, or a hook error, is discarded: the error string
    that would have been returned without the routine is returned
    instead. Empty input never reaches the hook; there is nothing to
    infer an intended JSON from.

    Args:
        raw (str): Raw JSON arguments string from the model.
        llm_fix (Callable[[str], str] | None): Optional LLM fallback
            returning reconstructed JSON for the raw string. Defaults
            to None.

    Returns:
        (dict[str, Any] | str) The parsed arguments dict on success, or
        a descriptive error string when no JSON object can be recovered.
    """
    if not raw or not raw.strip():
        return "Empty arguments: no JSON provided"

    parsed = _repair_json(raw)
    if isinstance(parsed, dict) or llm_fix is None:
        return parsed

    try:
        fixed = llm_fix(raw)
    except Exception as exc:
        # The routine is a last-resort attempt: its failures settle
        # back to the local-repair outcome instead of raising.
        llm_repair_log.warning(
            "LLM JSON repair routine failed",
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return parsed

    retried = _repair_json(fixed) if fixed and fixed.strip() else None
    if isinstance(retried, dict):
        log.info(
            "Repaired malformed JSON tool arguments",
            repairs=["reconstructed the JSON with an LLM routine"],
        )
        return retried
    return parsed


def _validate_edit_structure(data: dict[str, Any]) -> list[str]:
    """Report unrepairable structural defects in edit arguments.

    Checks presence, type, and validity of the required ``file_path``
    and ``file_edits`` keys and the structure of each edit block
    within ``file_edits``: each element must be a non-empty string
    (one SEARCH/REPLACE block).

    Args:
        data (dict[str, Any]): Parsed and repaired arguments dict.

    Returns:
        (list[str]) Defect description strings; an empty list means
        the structure is valid.
    """
    defects: list[str] = []

    # Check file_path
    file_path = data.get("file_path")
    if file_path is None:
        defects.append("Missing required key 'file_path'")
    elif not isinstance(file_path, str):
        defects.append(
            f"Key 'file_path' must be a string but got {type(file_path).__name__}"
        )
    elif len(file_path) == 0:
        defects.append("Key 'file_path' must not be empty")

    # Check file_edits
    file_edits = data.get("file_edits")
    if file_edits is None:
        defects.append("Missing required key 'file_edits'")
    elif not isinstance(file_edits, list):
        defects.append(
            f"Key 'file_edits' must be a list but got {type(file_edits).__name__}"
        )
    elif len(file_edits) == 0:
        defects.append("Key 'file_edits' must not be an empty list")
    else:
        # Check each edit block
        for idx, edit in enumerate(file_edits):
            if not isinstance(edit, str):
                defects.append(
                    f"file_edits[{idx}] must be a string but got {type(edit).__name__}"
                )
            elif len(edit) == 0:
                defects.append(f"file_edits[{idx}] must not be an empty string")

    return defects


def repair_edit_arguments(
    raw: str,
    *,
    llm_fix: Callable[[str], str] | None = None,
) -> dict[str, Any] | str:
    """Repair edit-tool JSON arguments, fixing fixable malformations.

    Parses the raw string with ``repair_and_parse`` (which also runs
    the ``llm_fix`` LLM fallback when local repairs fail), then repairs
    structure-level malformations: a ``file_edits`` value emitted as a
    JSON string is parsed into the list it should have been. Structural
    defects that cannot be repaired (missing keys, wrong types, empty
    values) yield a descriptive error string. The runtime retries an
    unrepairable structure through ``llm_fix`` (see
    ``core/runtime/edit_arguments.py``).

    Args:
        raw (str): Raw JSON arguments string from the model.
        llm_fix (Callable[[str], str] | None): Optional LLM fallback
            forwarding to ``repair_and_parse``. Defaults to None.

    Returns:
        (dict[str, Any] | str) The repaired arguments dict, or a
        descriptive error string for the unrepairable defects.
    """
    parsed = repair_and_parse(raw, llm_fix=llm_fix)
    if isinstance(parsed, str):
        return parsed

    # A file_edits value emitted as a JSON string (one level of the
    # same malformations repair_and_parse fixes at the top level) is
    # repaired by parsing it back into the list it should have been.
    file_edits = parsed.get("file_edits")
    if isinstance(file_edits, str):
        restored = _parse_with_repairs(file_edits)
        if restored is not None and isinstance(restored[0], list):
            log.info(
                "Repaired malformed JSON tool arguments",
                repairs=["parsed file_edits from a JSON string"],
            )
            parsed = {**parsed, "file_edits": restored[0]}

    defects = _validate_edit_structure(parsed)
    if defects:
        return "; ".join(defects)

    return parsed
