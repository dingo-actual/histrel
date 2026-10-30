#!/usr/bin/env python3
"""Unit tests for repair_edit_arguments JSON repair."""

from __future__ import annotations

import json

from histrel.tools.json import (
    repair_edit_arguments,
)
from tests.unit.tools.helpers import (
    edit_block,
)


class Test04Unit11Tools11EditRepairArguments:
    """repair_edit_arguments: JSON parsing with defect repair.

    repair_edit_arguments in tools.json is what the runtime calls with
    the raw JSON string from the LLM before delegating to execute().
    Malformed JSON with a repairable form (over-escaped quotes, invalid
    single-quote escapes, string-wrapped objects, file_edits emitted
    as a JSON string) is repaired rather than rejected.

    On success it returns a dict with the repaired arguments. On
    failure it returns a descriptive error string for the defects
    that cannot be repaired.
    """

    # ── Valid JSON, correct structure → returns dict ─────────────

    def test_04_unit_11_tools_11_parse_01_valid_json_returns_dict(self) -> None:
        """Correct JSON with the expected structure is returned as a dict."""
        raw = json.dumps(
            {
                "file_path": "main.py",
                "file_edits": [
                    edit_block("old", "new"),
                ],
            }
        )
        result = repair_edit_arguments(raw)
        assert isinstance(result, dict)
        assert result == {
            "file_path": "main.py",
            "file_edits": [edit_block("old", "new")],
        }

    def test_04_unit_11_tools_11_parse_02_valid_json_with_escaped_quotes(self) -> None:
        """Correct JSON with escaped double quotes in content is preserved."""
        raw = json.dumps(
            {
                "file_path": "main.py",
                "file_edits": [
                    edit_block('print("hello")', 'print("world")'),
                ],
            }
        )
        result = repair_edit_arguments(raw)
        assert isinstance(result, dict)
        assert result["file_edits"][0] == edit_block('print("hello")', 'print("world")')

    def test_04_unit_11_tools_11_parse_03_valid_json_with_newline_escape(self) -> None:
        """Correct JSON with \\n escapes inside string values is preserved."""
        raw = json.dumps(
            {
                "file_path": "main.py",
                "file_edits": [
                    edit_block("line1\nline2", "lineA\nlineB"),
                ],
            }
        )
        result = repair_edit_arguments(raw)
        assert isinstance(result, dict)
        assert result["file_edits"][0] == edit_block("line1\nline2", "lineA\nlineB")

    # ── Over-escaped JSON → repaired and returned as dict ────────

    def test_04_unit_11_tools_11_parse_06_over_escaped_is_repaired(self) -> None:
        """Over-escaped structural quotes are repaired and parsed.

        The LLM outputs extra backslashes before structural quotes.
        repair_edit_arguments removes the extra escaping level and
        returns the parsed arguments dict.
        """
        correct = json.dumps(
            {
                "file_path": "main.py",
                "file_edits": [
                    edit_block("old", "new"),
                ],
            }
        )
        over_escaped = correct.replace('"', '\\"')
        result = repair_edit_arguments(over_escaped)
        assert isinstance(result, dict)
        assert result == {
            "file_path": "main.py",
            "file_edits": [edit_block("old", "new")],
        }

    def test_04_unit_11_tools_11_parse_07_over_escaped_with_content_quotes(
        self,
    ) -> None:
        """Over-escaped JSON with content quotes is repaired and parsed."""
        correct = json.dumps(
            {
                "file_path": "main.py",
                "file_edits": [
                    edit_block('print("hello")', 'print("world")'),
                ],
            }
        )
        over_escaped = correct.replace('"', '\\"')
        result = repair_edit_arguments(over_escaped)
        assert isinstance(result, dict)
        assert result["file_edits"][0] == edit_block('print("hello")', 'print("world")')

    def test_04_unit_11_tools_11_parse_08_over_escaped_with_newline_escape(
        self,
    ) -> None:
        """Over-escaped JSON with \\n in content is repaired and parsed."""
        correct = json.dumps(
            {
                "file_path": "main.py",
                "file_edits": [
                    edit_block("line1\nline2", "lineA\nlineB"),
                ],
            }
        )
        over_escaped = correct.replace('"', '\\"')
        result = repair_edit_arguments(over_escaped)
        assert isinstance(result, dict)
        assert result["file_edits"][0] == edit_block("line1\nline2", "lineA\nlineB")

    def test_04_unit_11_tools_11_parse_09_single_quote_escape_repaired(self) -> None:
        """Invalid \\' escapes are repaired, not rejected."""
        correct = json.dumps(
            {
                "file_path": "main.py",
                "file_edits": [
                    edit_block("print('hello')", "print('world')"),
                ],
            }
        )
        over_escaped = correct.replace('"', '\\"').replace("'", "\\'")
        result = repair_edit_arguments(over_escaped)
        assert isinstance(result, dict)
        assert result["file_edits"][0] == edit_block("print('hello')", "print('world')")

    # ── Wrapped-in-quotes JSON → unwrapped and returned as dict ──

    def test_04_unit_11_tools_11_parse_15_wrapped_in_quotes_is_repaired(self) -> None:
        """JSON wrapped entirely in quotes is unwrapped and parsed.

        The LLM sometimes wraps the entire JSON object in quotes,
        producing a JSON string value instead of an object. The inner
        JSON object is recovered without the wrapping quotes.
        """
        correct = json.dumps(
            {
                "file_path": "main.py",
                "file_edits": [
                    edit_block("old", "new"),
                ],
            }
        )
        wrapped = json.dumps(correct)
        result = repair_edit_arguments(wrapped)
        assert isinstance(result, dict)
        assert result == {
            "file_path": "main.py",
            "file_edits": [edit_block("old", "new")],
        }

    # ── Valid JSON, wrong structure → error string when unrepairable ──

    def test_04_unit_11_tools_11_parse_04_file_edits_as_string_repaired(
        self,
    ) -> None:
        """A file_edits value emitted as a JSON string is repaired and parsed."""
        inner = json.dumps([edit_block("old", "new")])
        raw = json.dumps({"file_path": "main.py", "file_edits": inner})
        result = repair_edit_arguments(raw)
        assert isinstance(result, dict)
        assert result["file_edits"] == [edit_block("old", "new")]

    def test_04_unit_11_tools_11_parse_05_missing_file_path_rejected(self) -> None:
        """Missing file_path key yields a descriptive error string."""
        raw = json.dumps({"file_edits": [edit_block("old", "new")]})
        result = repair_edit_arguments(raw)
        assert isinstance(result, str)
        assert "file_path" in result

    def test_04_unit_11_tools_11_parse_16_missing_file_edits_rejected(self) -> None:
        """Missing file_edits key yields a descriptive error string."""
        raw = json.dumps({"file_path": "main.py"})
        result = repair_edit_arguments(raw)
        assert isinstance(result, str)
        assert "file_edits" in result

    def test_04_unit_11_tools_11_parse_17_edit_element_not_string_rejected(
        self,
    ) -> None:
        """A file_edits element that is not a string yields an error string."""
        raw = json.dumps(
            {
                "file_path": "main.py",
                # Old-format edit objects are no longer valid elements.
                "file_edits": [{"search": "old", "replace": "new"}],
            }
        )
        result = repair_edit_arguments(raw)
        assert isinstance(result, str)
        assert "file_edits[0] must be a string" in result

    def test_04_unit_11_tools_11_parse_18_empty_edit_block_rejected(
        self,
    ) -> None:
        """An empty file_edits string element yields an error string."""
        raw = json.dumps(
            {
                "file_path": "main.py",
                "file_edits": [""],
            }
        )
        result = repair_edit_arguments(raw)
        assert isinstance(result, str)
        assert "file_edits[0] must not be an empty string" in result

    def test_04_unit_11_tools_11_parse_19_empty_file_edits_rejected(self) -> None:
        """An empty file_edits list yields a descriptive error string."""
        raw = json.dumps({"file_path": "main.py", "file_edits": []})
        result = repair_edit_arguments(raw)
        assert isinstance(result, str)
        assert "file_edits" in result

    def test_04_unit_11_tools_11_parse_20_empty_file_path_rejected(self) -> None:
        """An empty file_path string yields a descriptive error string."""
        raw = json.dumps({"file_path": "", "file_edits": [edit_block("a", "b")]})
        result = repair_edit_arguments(raw)
        assert isinstance(result, str)
        assert "file_path" in result

    # ── Unparseable input → error string for unrepairable JSON ──

    def test_04_unit_11_tools_11_parse_11_unparseable_returns_error_string(
        self,
    ) -> None:
        """Completely unparseable input yields a descriptive error string."""
        result = repair_edit_arguments("this is not json at all")
        assert isinstance(result, str)
        assert "json" in result.lower() or "parse" in result.lower()

    def test_04_unit_11_tools_11_parse_12_empty_string_returns_error_string(
        self,
    ) -> None:
        """An empty string yields a descriptive error string."""
        result = repair_edit_arguments("")
        assert isinstance(result, str)

    def test_04_unit_11_tools_11_parse_13_parses_but_wrong_type_returns_error(
        self,
    ) -> None:
        """JSON that parses to a non-dict (e.g. a list) yields an error string."""
        result = repair_edit_arguments('["not", "an", "object"]')
        assert isinstance(result, str)
        assert "object" in result.lower() or "dict" in result.lower()

    # ── Correct JSON with complex content → returned unchanged ───

    def test_04_unit_11_tools_11_parse_10_correct_json_never_modified(self) -> None:
        """Correct JSON with complex content is returned as a dict.

        This is the zero false-positive guarantee: if the JSON is valid
        and has the expected structure, it is returned as a dict with
        no modification.
        """
        raw = json.dumps(
            {
                "file_path": "main.py",
                "file_edits": [
                    edit_block('print("hello\\nworld")', "x = 1\\ny = 2"),
                ],
            }
        )
        result = repair_edit_arguments(raw)
        assert isinstance(result, dict)
        assert result["file_edits"][0] == edit_block(
            'print("hello\\nworld")', "x = 1\\ny = 2"
        )
