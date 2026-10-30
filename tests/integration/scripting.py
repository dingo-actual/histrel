#!/usr/bin/env python3
"""Scripting helpers for deterministic integration-test sessions.

Important Classes:
    CallBuilder (Protocol): Signature of a tool-call builder.

Important Functions:
    make_fake_response(text, tool_calls) -> ModelResponse: Build a fake
        model response for patching.
    make_completer(tool_calls, text) -> Callable: Build a fake complete
        function returning one fixed response.
    tool_call(tool_name, arguments, call_id) -> NormalizedToolCall: Build
        a normalized tool call from raw arguments.
    complete_and_change(workspace) -> None: Modify the repo.
    scripted_complete(responses) -> tuple: Scripted complete function and
        its recorded-request log.
    section(content, name) -> str: Extract a ``## <name>`` section body.
    respond(text, call, name, arguments) -> ModelResponse: Response with
        one tool call.
    solve_turns(runtime, calculator, call) -> list: The scripted turns of a
        full solve.
    solve_script(runtime, calculator, call) -> tuple: Full-solve script and
        its recorded-request log.

Important Variables:
    EDIT_BLOCK (str): SEARCH/REPLACE edit that fixes the add function.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

from histrel.core.runtime import Runtime
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
    NormalizedToolCall,
    Usage,
)

# SEARCH/REPLACE edit that replaces the defective subtraction.
EDIT_BLOCK = (
    "/***SEARCH***/\n"
    "    return a - b\n"
    "/***END_SEARCH***/\n"
    "/***REPLACE***/\n"
    "    return a + b\n"
    "/***END_REPLACE***/"
)


class CallBuilder(Protocol):
    """Signature of a tool-call builder."""

    def __call__(
        self,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        call_id: str = "call_1",
    ) -> NormalizedToolCall:
        """Build a tool call."""
        ...


def make_fake_response(
    text: str = "Working on it.",
    tool_calls: tuple[NormalizedToolCall, ...] = (),
) -> ModelResponse:
    """Create a fake model response for patching."""
    return ModelResponse(
        text_parts=(text,) if text else (),
        tool_calls=tool_calls,
        usage=Usage(input=1000, output=500, total=1500),
        provider_request_id="req-fake",
        finish_reason="stop",
    )


def make_completer(
    *tool_calls: NormalizedToolCall,
    text: str = "",
) -> Callable[[ModelRequest], ModelResponse]:
    """Build a fake complete function returning one fixed response."""
    response = make_fake_response(text=text, tool_calls=tool_calls)

    def complete(request: ModelRequest) -> ModelResponse:
        return response

    return complete


def tool_call(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    call_id: str = "call_1",
) -> NormalizedToolCall:
    """Build a normalized tool call from raw arguments."""
    return NormalizedToolCall(
        id=call_id,
        name=tool_name,
        arguments=json.dumps(arguments or {}),
    )


def complete_and_change(workspace: Path) -> None:
    """Modify the repo so the session's terminal condition is met."""
    (workspace / "calculator.py").write_text("print('fixed')\n")


def scripted_complete(
    responses: list[ModelResponse],
) -> tuple[Callable[[ModelRequest], ModelResponse], list[ModelRequest]]:
    """Build a deterministic provider-call patch target and its log.

    Records every ModelRequest the agent loop submits; returns the
    scripted responses in order, repeating the final response for any
    calls beyond the script.
    """
    recorded: list[ModelRequest] = []

    def complete(request: ModelRequest) -> ModelResponse:
        recorded.append(request)
        index = min(len(recorded) - 1, len(responses) - 1)
        return responses[index]

    return complete, recorded


def section(content: str, name: str) -> str:
    """Extract a ``## <name>`` section body from a rendered user message.

    The section body runs from just after the ``## <name>`` header up
    to the next ``## `` section, or to the end of the message.
    """
    body = content.split(f"## {name}", 1)[1]
    boundary = body.find("\n## ")
    return body[:boundary] if boundary != -1 else body


def respond(
    text: str,
    call: CallBuilder,
    name: str,
    arguments: dict[str, Any],
) -> ModelResponse:
    """Build a model response carrying one tool call."""
    return make_fake_response(text=text, tool_calls=(call(name, arguments),))


def solve_turns(
    runtime: Runtime,
    calculator: str,
    call: CallBuilder,
) -> list[Callable[[], ModelResponse]]:
    """Build the scripted turns of a full solve.

    The turns read the defective module, fix it with a real edit, and
    then answer "Done." forever.
    """
    edit_args: dict[str, Any] = {
        "file_path": calculator,
        "file_edits": [EDIT_BLOCK],
    }
    return [
        lambda: respond(
            "Reading calculator.py to confirm the defect.",
            call,
            "read",
            {"path": calculator},
        ),
        lambda: respond(
            "The add function subtracts; replacing with a + b.",
            call,
            "edit",
            edit_args,
        ),
        lambda: make_fake_response(text="Done."),
    ]


def solve_script(
    runtime: Runtime,
    calculator: str,
    call: CallBuilder,
) -> tuple[Callable[[ModelRequest], ModelResponse], list[ModelRequest]]:
    """Build a full-solve script and its recorded-request log."""
    turns = solve_turns(runtime, calculator, call)
    recorded: list[ModelRequest] = []

    def complete(request: ModelRequest) -> ModelResponse:
        recorded.append(request)
        index = min(len(recorded) - 1, len(turns) - 1)
        return turns[index]()

    return complete, recorded
