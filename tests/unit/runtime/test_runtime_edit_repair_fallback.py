#!/usr/bin/env python3
"""Unit tests for edit-arguments repair through the LLM routine.

An edit call the local repairs cannot fix — arguments that parse to a
non-object, or valid JSON whose inner ``file_edits`` string is mangled
beyond repair — is offered to the fix_json_arguments routine constrained
by the edit tool's schema, and the repaired arguments are validated and
applied. Unusable routine output keeps the original structural error.

    Test Classes:
        Test04Unit43Runtime43EditJsonFallback: Edit-call fallback.

    Test Subtypes:
        Morphism, Failure Mode
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from histrel.core.runtime import Runtime
from tests.unit.runtime.helpers import (
    assert_single_routine_usage,
    tool_result_parts,
)
from tests.unit.runtime.json_fallback_support import (
    NON_OBJECT_EDITS_CALL,
    UNREPAIRABLE_EDITS_CALL,
    assert_edit_applied,
    assert_single_edit_routine_request,
    edit_routine_text,
    routine_requests,
    run_edit_case,
)


class Test04Unit43Runtime43EditJsonFallback:
    """run_agent: unrepairable edit arguments get one LLM fix attempt."""

    def test_04_unit_43_runtime_43_edit_01_morphism_01_routine_consulted_for_non_object(
        self,
        make_json_runtime: Callable[..., Runtime],
    ) -> None:
        # Arrange — react mode with an edit call whose arguments parse
        # to a list, which the local repairs reject
        # Act
        _runtime, requests = run_edit_case(
            make_json_runtime, "react", NON_OBJECT_EDITS_CALL, edit_routine_text()
        )

        # Assert — the routine ran once under the edit tool's schema
        assert_single_edit_routine_request(requests)

    def test_04_unit_43_runtime_43_edit_02_morphism_02_non_object_arguments_applied(
        self,
        make_json_runtime: Callable[..., Runtime],
        tmp_path: Path,
    ) -> None:
        # Arrange — the routine returns the intended edit arguments
        # Act
        runtime, _requests = run_edit_case(
            make_json_runtime, "react", NON_OBJECT_EDITS_CALL, edit_routine_text()
        )

        # Assert — the repaired arguments passed validation and executed
        assert_edit_applied(runtime, tmp_path)
        assert_single_routine_usage(runtime)

    def test_04_unit_43_runtime_43_edit_03_morphism_03_routine_consulted_for_bad_inner_edits(
        self,
        make_json_runtime: Callable[..., Runtime],
    ) -> None:
        # Arrange — stateless mode with valid outer JSON whose
        # file_edits string no local repair can parse
        # Act
        _runtime, requests = run_edit_case(
            make_json_runtime,
            "stateless",
            UNREPAIRABLE_EDITS_CALL,
            edit_routine_text(),
        )

        # Assert — the routine ran once under the edit tool's schema
        assert_single_edit_routine_request(requests)

    def test_04_unit_43_runtime_43_edit_04_morphism_04_unrepairable_inner_edits_applied(
        self,
        make_json_runtime: Callable[..., Runtime],
        tmp_path: Path,
    ) -> None:
        # Arrange — the routine returns the intended edit arguments
        # Act
        runtime, _requests = run_edit_case(
            make_json_runtime,
            "stateless",
            UNREPAIRABLE_EDITS_CALL,
            edit_routine_text(),
        )

        # Assert — the repaired arguments passed validation and executed
        assert_edit_applied(runtime, tmp_path)
        assert_single_routine_usage(runtime)

    def test_04_unit_43_runtime_43_edit_05_failure_02_unusable_fix_records_error(
        self,
        make_json_runtime: Callable[..., Runtime],
    ) -> None:
        # Arrange — the routine cannot recover the malformed edits
        # Act
        runtime, requests = run_edit_case(
            make_json_runtime, "stateless", UNREPAIRABLE_EDITS_CALL, "not json"
        )

        # Assert — the routine was attempted once and the original
        # structural error survives as the tool result.
        assert len(routine_requests(requests)) == 1
        parts = tool_result_parts(runtime, tool_name="edit")
        assert len(parts) == 1
        assert parts[0].success is False
        assert parts[0].error is not None
        assert "file_edits" in parts[0].error
