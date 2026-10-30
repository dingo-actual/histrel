#!/usr/bin/env python3
"""Shared fixtures and assertions for the JSON-repair fallback tests.

The edit-arguments fallback tests use one fake completer, one set of
malformed tool calls, and one schema assertion, so they live here rather
than being duplicated across modules.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from histrel.core.runtime import Runtime, run_agent
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
    NormalizedToolCall,
)
from histrel.tools.repository import EditFileTool
from tests.unit.runtime.helpers import (
    USER_MSG,
    make_fake_response,
    tool_result_parts,
)

NON_OBJECT_EDITS_CALL = NormalizedToolCall(
    id="call_1",
    name="edit",
    arguments='["not", "an", "object"]',
)

MAIN_EDIT = (
    "/***SEARCH***/\n"
    "print('hello')\n"
    "/***END_SEARCH***/\n"
    "/***REPLACE***/\n"
    "print('fixed')\n"
    "/***END_REPLACE***/"
)

# Valid JSON whose inner file_edits value is itself malformed beyond
# local repair, so only the LLM routine can recover the edit.
UNREPAIRABLE_EDITS_CALL = NormalizedToolCall(
    id="call_1",
    name="edit",
    arguments=json.dumps(
        {"file_path": "main.py", "file_edits": '["\n  /***SEARCH***/\n'}
    ),
)


def make_instructions(
    requests: list[ModelRequest],
    first_call: NormalizedToolCall,
    routine_text: str,
) -> Callable[[ModelRequest], ModelResponse]:
    """Build a fake complete that emits one malformed call, then stops.

    Main-loop requests (which carry tools) return the first tool call
    once and a text-only response after; routine requests (which carry
    a response_format) return ``routine_text``.
    """

    def fake_complete(request: ModelRequest) -> ModelResponse:
        requests.append(request)
        if request.tools:
            if len([r for r in requests if r.tools]) == 1:
                return make_fake_response(tool_calls=(first_call,))
            return make_fake_response(text="Done.")
        return make_fake_response(text=routine_text)

    return fake_complete


def routine_requests(requests: list[ModelRequest]) -> list[ModelRequest]:
    """Collect the fix-json routine requests from captured requests."""
    return [r for r in requests if r.response_format is not None]


def run_edit_case(
    make_json_runtime: Callable[..., Runtime],
    mode: str,
    call: NormalizedToolCall,
    routine_text: str,
) -> tuple[Runtime, list[ModelRequest]]:
    """Run one session whose edit call needs the LLM repair routine.

    Args:
        make_json_runtime (Callable[..., Runtime]): Runtime factory
            fixture from the runtime test conftest.
        mode (str): Agent loop mode.
        call (NormalizedToolCall): The malformed edit call to emit.
        routine_text (str): The routine's response text.

    Returns:
        (tuple[Runtime, list[ModelRequest]]) The runtime and the
        requests the model adapter received.
    """
    requests: list[ModelRequest] = []
    runtime = make_json_runtime(mode, make_instructions(requests, call, routine_text))
    run_agent(USER_MSG, runtime)
    return runtime, requests


def edit_routine_text() -> str:
    """Return the routine response that repairs the fixture edit."""
    return json.dumps({"file_path": "main.py", "file_edits": [MAIN_EDIT]})


def assert_edit_applied(runtime: Runtime, tmp_path: Any) -> None:
    """Assert the repaired edit executed and rewrote the fixture file."""
    parts = tool_result_parts(runtime, tool_name="edit")
    assert len(parts) == 1
    assert parts[0].success is True
    assert (tmp_path / "main.py").read_text(encoding="utf-8") == "print('fixed')\n"


def assert_single_edit_routine_request(requests: list[ModelRequest]) -> None:
    """Assert one routine request, constrained to the edit tool schema."""
    routine = routine_requests(requests)
    assert len(routine) == 1
    assert routine[0].response_format == {
        "type": "json_schema",
        "json_schema": {
            "name": "edit",
            "schema": EditFileTool.definition["function"]["parameters"],
        },
    }
