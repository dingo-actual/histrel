#!/usr/bin/env python3
"""Unit tests for the repair_and_parse LLM fallback hook."""

from __future__ import annotations

import json
from collections.abc import Callable

from histrel.tools.json import (
    repair_and_parse,
    repair_edit_arguments,
)


class Test04Unit14Tools14JsonLlmFallback:
    """repair_and_parse: LLM routine fallback after local repairs fail.

    When every local repair fails, repair_and_parse calls the optional
    ``llm_fix`` hook (wired by the runtime to the fix_json_arguments
    LLM routine) with the raw string and re-applies the same local
    repair machinery to the routine's output. A routine result that
    still fails to parse, or a routine error, keeps the error string
    that would have been returned without the routine.
    """

    @staticmethod
    def _recording_llm_fix(
        fixed_output: str,
    ) -> tuple[Callable[[str], str], list[str]]:
        """Build an llm_fix hook that records its inputs."""
        calls: list[str] = []

        def llm_fix(raw: str) -> str:
            calls.append(raw)
            return fixed_output

        return llm_fix, calls

    # ── Local repairs fail → routine fixes the JSON ──────────

    def test_04_unit_14_tools_14_llm_01_morphism_01_unparseable_json_is_fixed(
        self,
    ) -> None:
        """JSON no local repair can fix is fixed by the LLM routine."""
        raw = '{"file_path": "main.py"'  # truncated: unrepairable
        llm_fix, calls = self._recording_llm_fix(
            json.dumps({"file_path": "main.py", "file_edits": []})
        )

        result = repair_and_parse(raw, llm_fix=llm_fix)

        assert result == {"file_path": "main.py", "file_edits": []}
        assert calls == [raw]

    def test_04_unit_14_tools_14_llm_02_boundary_01_parseable_non_dict_triggers_llm(
        self,
    ) -> None:
        """JSON that parses to a non-dict also triggers the routine."""
        raw = json.dumps([1, 2])
        llm_fix, calls = self._recording_llm_fix(json.dumps({"path": "main.py"}))

        result = repair_and_parse(raw, llm_fix=llm_fix)

        assert result == {"path": "main.py"}
        assert calls == [raw]

    # ── Routine output unusable → the original error survives ─

    def test_04_unit_14_tools_14_llm_03_failure_01_unparseable_output_keeps_error(
        self,
    ) -> None:
        """A routine result that still fails to parse keeps the error."""
        raw = '{"file_path": "main.py"'
        first_error = repair_and_parse(raw)
        assert isinstance(first_error, str)
        llm_fix, _ = self._recording_llm_fix("still not json")

        result = repair_and_parse(raw, llm_fix=llm_fix)

        assert result == first_error

    def test_04_unit_14_tools_14_llm_04_failure_01_routine_error_keeps_error(
        self,
    ) -> None:
        """A crashing routine is contained; the error string survives."""
        raw = '{"file_path": "main.py"'
        first_error = repair_and_parse(raw)
        assert isinstance(first_error, str)

        def crashing_llm_fix(raw: str) -> str:
            raise RuntimeError("routine offline")

        result = repair_and_parse(raw, llm_fix=crashing_llm_fix)

        assert result == first_error

    # ── Repairable inputs → no LLM call ─────────────────────

    def test_04_unit_14_tools_14_llm_05_isolation_01_valid_json_never_calls_llm(
        self,
    ) -> None:
        """Valid JSON never reaches the routine."""
        llm_fix, calls = self._recording_llm_fix("{}")

        result = repair_and_parse(json.dumps({"path": "main.py"}), llm_fix=llm_fix)

        assert result == {"path": "main.py"}
        assert calls == []

    def test_04_unit_14_tools_14_llm_06_boundary_02_empty_input_never_calls_llm(
        self,
    ) -> None:
        """Empty input fails before the routine: nothing to infer from."""
        llm_fix, calls = self._recording_llm_fix("{}")

        result = repair_and_parse("   ", llm_fix=llm_fix)

        assert isinstance(result, str)
        assert "Empty arguments" in result
        assert calls == []

    # ── No hook → prior behavior ──────────────────────────────

    def test_04_unit_14_tools_14_llm_07_contract_01_no_hook_preserves_error(
        self,
    ) -> None:
        """Without a hook the unrepairable error string is returned."""
        result = repair_and_parse('{"file_path": "main.py"')

        assert isinstance(result, str)
        assert result.startswith("Invalid JSON:")

    # ── Wrappers forward the hook to repair_and_parse ──────

    def test_04_unit_14_tools_14_llm_08_morphism_02_edit_arguments_forward_hook(
        self,
    ) -> None:
        """repair_edit_arguments forwards the hook to its parse stage."""
        raw = '{"file_path": "main.py", "file_edits": ["x"]'  # truncated
        fixed = json.dumps({"file_path": "main.py", "file_edits": ["x"]})
        llm_fix, calls = self._recording_llm_fix(fixed)

        result = repair_edit_arguments(raw, llm_fix=llm_fix)

        assert result == {"file_path": "main.py", "file_edits": ["x"]}
        assert calls == [raw]
