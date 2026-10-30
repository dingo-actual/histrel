#!/usr/bin/env python3
"""Shared helpers for the runtime unit tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from openai import OpenAI

from histrel.context.token_ledger import LedgerEntry
from histrel.core.journal import load_journal
from histrel.core.runtime import Runtime
from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
)
from histrel.model.messages import (
    ToolRequestPart,
    ToolResultPart,
)
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
    NormalizedToolCall,
    Usage,
)
from histrel.tools.tool import ToolExecution

# User request message used across runtime tests.
USER_MSG = "Fix the failing test."

# Usage reported by make_fake_response.
FAKE_USAGE = Usage(input=1000, output=500, reasoning=0, total=1500, cached=0)

# Arguments for a minimal analyze_query call.
ANALYZE_QUERY_ARGS = json.dumps(
    {
        "objective": "test",
        "initial_summary": "test",
        "relevant_information": [],
    }
)


def outside_output_dir(tmp_path: Path) -> Path:
    """Output dir outside the repo, like the production /workspace/output.

    Journal artifacts inside the repo would count as untracked changes
    against HEAD.
    """
    return tmp_path.parent / (tmp_path.name + "-output")


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


def tool_call(
    name: str,
    arguments: str = "{}",
    call_id: str = "call_1",
) -> NormalizedToolCall:
    """Build a normalized tool call."""
    return NormalizedToolCall(id=call_id, name=name, arguments=arguments)


def patched_adapter(
    monkeypatch: pytest.MonkeyPatch,
    complete: Any,
) -> OpenAICompatAdapter:
    """Build a baseline adapter whose complete method is patched."""
    adapter = OpenAICompatAdapter(
        client=OpenAI(api_key="test-key"),
        model_name="gpt-4o",
        provider="openai",
    )
    monkeypatch.setattr(adapter, "complete", complete)
    return adapter


def make_completer(
    *tool_calls: NormalizedToolCall,
    text: str = "",
) -> Any:
    """Build a fake complete function returning one fixed response."""
    response = make_fake_response(text=text, tool_calls=tool_calls)

    def complete(request: ModelRequest) -> ModelResponse:
        return response

    return complete


def first_tool_request(runtime: Runtime) -> ToolRequestPart:
    """Return the first tool-request part recorded in the journal."""
    messages = list(load_journal(runtime.journal._path))
    message = next(
        m for m in messages if any(isinstance(p, ToolRequestPart) for p in m.content)
    )
    return next(p for p in message.content if isinstance(p, ToolRequestPart))


def tool_result_parts(
    runtime: Runtime,
    tool_name: str | None = None,
) -> list[ToolResultPart]:
    """Return the tool-result parts recorded in the journal.

    Args:
        runtime (Runtime): Runtime whose journal is read.
        tool_name (str | None): When given, keep only that tool's parts.

    Returns:
        (list[ToolResultPart]) The matching tool-result parts.
    """
    messages = list(load_journal(runtime.journal._path))
    return [
        part
        for message in messages
        if message.role == "tool"
        for part in message.content
        if isinstance(part, ToolResultPart)
        and (tool_name is None or part.tool_name == tool_name)
    ]


def first_tool_result(runtime: Runtime) -> ToolResultPart:
    """Return the first tool-result part recorded in the journal."""
    return tool_result_parts(runtime)[0]


def execute_tool(
    runtime: Runtime,
    name: str,
    arguments: str = "{}",
) -> ToolExecution:
    """Execute one tool call against the runtime, bypassing the loop."""
    return runtime._execute_tool_call(
        NormalizedToolCall(id="call_1", name=name, arguments=arguments)
    )


def user_message_contents(request: ModelRequest) -> list[str]:
    """Return the user message contents of one model request."""
    return [m.content for m in request.messages if m.role == "user"]


def routine_ledger_entries(runtime: Runtime) -> list[LedgerEntry]:
    """Return the routine-kind entries from the runtime token ledger."""
    return [e for e in runtime.token_ledger.entries if e.call_kind == "routine"]


def assert_single_routine_usage(runtime: Runtime) -> None:
    """Assert exactly one routine call's usage was recorded."""
    entries = routine_ledger_entries(runtime)
    assert len(entries) == 1
    assert entries[0].usage == FAKE_USAGE
    assert entries[0].provider_request_id == "req-fake"


def change_repo(tmp_path: Path) -> None:
    """Modify main.py so the repo differs from its committed state."""
    (tmp_path / "main.py").write_text("print('changed')\n")
