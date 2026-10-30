#!/usr/bin/env python3
"""Shared helpers for the duplicate-guard unit tests.

Important Functions:
    make_fake_response(text, tool_calls) -> ModelResponse: Build a fake
        model response for patching.
    make_runtime(monkeypatch, tmp_path, config) -> tuple: Build an
        initialized runtime with a real adapter over a fixture repo.
    make_tool_call(call_id, name, arguments) -> NormalizedToolCall:
        Build a normalized tool call from raw arguments.
    read_call(call_id, start_line) -> NormalizedToolCall: Build a read
        tool call for a single line of main.py.
    read_args() -> dict: Fresh read arguments for main.py, line 1 only.
    list_dir_args() -> dict: Fresh list_dir arguments for src.
    search_args() -> dict: Fresh search arguments for TODO in src/*.py.
    glob_args() -> dict: Fresh glob arguments for *.py in src.
    tool_results(runtime, tool_name) -> list[ToolResultPart]: Tool results
        for one tool from the session journal.
    assert_duplicate_skipped(results) -> None: Assert the second result is
        a non-informative duplicate audit result.

Important Variables:
    USER_MSG (str): User message every runtime test initializes with.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from openai import OpenAI

from histrel.core.config import AgentConfig
from histrel.core.journal import load_journal
from histrel.core.runtime import Runtime
from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
)
from histrel.model.messages import ToolResultPart
from histrel.model.types import (
    ModelResponse,
    NormalizedToolCall,
    Usage,
)
from tests.unit.factories import init_fixture_repo

USER_MSG = "Fix the failing test."


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


def make_tool_call(call_id: str, name: str, arguments: dict) -> NormalizedToolCall:
    """Build a normalized tool call from raw arguments."""
    return NormalizedToolCall(
        id=call_id,
        name=name,
        arguments=json.dumps(arguments),
    )


def read_call(
    call_id: str, start_line: int = 1, path: str = "main.py"
) -> NormalizedToolCall:
    """Build a read tool call for a single line of a workspace file."""
    return make_tool_call(
        call_id,
        "read",
        {"path": path, "start_line": start_line, "end_line": start_line},
    )


def read_args() -> dict[str, Any]:
    """Fresh read arguments for main.py, line 1 only."""
    return {"path": "main.py", "start_line": 1, "end_line": 1}


def list_dir_args() -> dict[str, Any]:
    """Fresh list_dir arguments for the src directory."""
    return {"path": "src"}


def search_args() -> dict[str, Any]:
    """Fresh search arguments for TODO in src/*.py."""
    return {"pattern": "TODO", "path": "src", "glob": "*.py"}


def glob_args() -> dict[str, Any]:
    """Fresh glob arguments for *.py in src."""
    return {"pattern": "*.py", "path": "src"}


def make_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    config: AgentConfig,
) -> tuple[Runtime, OpenAICompatAdapter]:
    """Build an initialized runtime with a real adapter over a fixture repo."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    init_fixture_repo(tmp_path)
    monkeypatch.chdir(tmp_path)

    adapter = OpenAICompatAdapter(
        client=OpenAI(api_key="test-key"),
        model_name="gpt-4o",
        provider="openai",
    )
    runtime = Runtime(
        config=config, workspace_root=tmp_path, output_dir=tmp_path / "output"
    )
    runtime.initialize(USER_MSG)
    runtime.model_adapter = adapter
    return runtime, adapter


def tool_results(runtime: Runtime, tool_name: str) -> list[ToolResultPart]:
    """Return the journaled tool results for one tool name."""
    messages = list(load_journal(runtime.journal._path))
    parts: list[ToolResultPart] = []
    for message in messages:
        if message.role != "tool":
            continue
        for part in message.content:
            if isinstance(part, ToolResultPart) and part.tool_name == tool_name:
                parts.append(part)
    return parts


def assert_duplicate_skipped(results: list[ToolResultPart], *, count: int = 2) -> None:
    """Assert the first result ran and the second was a skipped duplicate."""
    assert len(results) == count
    assert results[0].success is True
    assert results[0].error is None
    assert results[1].success is False
    assert results[1].informative is False
    assert "Duplicate tool call" in str(results[1].output)
