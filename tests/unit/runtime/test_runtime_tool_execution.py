#!/usr/bin/env python3
"""Unit tests for runtime tool execution, commit ordering, history, and payload shapes."""

from __future__ import annotations

import json
from collections.abc import Callable

import pytest

from histrel.core.journal import load_journal
from histrel.core.runtime import (
    Runtime,
    run_agent,
)
from histrel.model.messages import (
    ToolRequestPart,
    ToolResultPart,
)
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
    NormalizedToolCall,
)
from tests.unit.runtime.helpers import (
    USER_MSG,
    make_completer,
    make_fake_response,
    patched_adapter,
    tool_call,
)
from tests.unit.runtime.stubs import (
    CrashTool,
    EchoTool,
    FailTool,
    RecordTool,
)


class Test04Unit03Runtime03ToolExecution:
    """Runtime tool execution and commit ordering."""

    def test_04_unit_03_runtime_03_tool_01_commits_assistant_before_tool(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        # Arrange
        runtime = make_runtime()
        runtime.model_adapter = patched_adapter(
            monkeypatch,
            make_completer(
                tool_call("echo", '{"text":"hi"}'),
                text="Let me read the file.",
            ),
        )
        # A simple echo tool that completes the todo and makes a repo
        # change as a side effect, so the terminal condition is satisfied
        # after tool execution.
        runtime.ordinary_tools = [EchoTool(runtime, tmp_path)]

        # Act
        run_agent(USER_MSG, runtime)

        # Assert — the journal should contain the assistant message before the tool result
        messages = list(load_journal(runtime.journal._path))
        assistant_indices = [
            i
            for i, m in enumerate(messages)
            if m.role == "assistant"
            and any(isinstance(p, ToolRequestPart) for p in m.content)
        ]
        tool_indices = [i for i, m in enumerate(messages) if m.role == "tool"]
        assert len(assistant_indices) > 0
        assert len(tool_indices) > 0
        assert assistant_indices[0] < tool_indices[0]

    def test_04_unit_03_runtime_03_tool_02_commits_tool_result_even_on_error(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        # Arrange
        runtime = make_runtime()
        runtime.model_adapter = patched_adapter(
            monkeypatch,
            make_completer(tool_call("fail")),
        )
        runtime.ordinary_tools = [FailTool(runtime, tmp_path)]

        # Act
        run_agent(USER_MSG, runtime)

        # Assert — tool result is committed even on error
        messages = list(load_journal(runtime.journal._path))
        tool_results = [
            m
            for m in messages
            if m.role == "tool"
            and any(isinstance(p, ToolResultPart) and not p.success for p in m.content)
        ]
        assert len(tool_results) > 0

    def test_04_unit_03_runtime_03_tool_03_crashing_tool_returns_error_result(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        """A tool that raises is contained as an error result, not a crash.

        The agent run must survive an unexpected tool exception (for example
        NotADirectoryError from list_dir on a file path) so one bad tool
        call cannot destroy an expensive live session.
        """
        # Arrange
        runtime = make_runtime()
        runtime.model_adapter = patched_adapter(
            monkeypatch,
            make_completer(tool_call("crash")),
        )
        runtime.ordinary_tools = [CrashTool(runtime, tmp_path)]

        # Act
        run_agent(USER_MSG, runtime)

        # Assert — the run completed and the crash surfaced as a failed
        # tool result the model could recover from.
        messages = list(load_journal(runtime.journal._path))
        tool_results = [
            m
            for m in messages
            if m.role == "tool"
            and any(
                isinstance(p, ToolResultPart)
                and not p.success
                and "tool crashed" in (p.output or "")
                for p in m.content
            )
        ]
        assert len(tool_results) > 0


class Test04Unit03Runtime03ToolExecutionHistory:
    """Runtime tool execution: history records and payload shapes."""

    def test_04_unit_03_runtime_03_tool_05_tool_result_contains_tool_name_and_arguments(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path,
    ) -> None:
        # Arrange — the tool result message committed to the journal must
        # carry the tool name and arguments so the LLM can correlate the
        # result with the call that produced it.
        runtime = make_runtime()
        runtime.model_adapter = patched_adapter(
            monkeypatch,
            make_completer(
                tool_call("echo", '{"text":"hello"}'),
                text="Reading the file.",
            ),
        )
        runtime.ordinary_tools = [EchoTool(runtime, tmp_path)]

        # Act
        run_agent(USER_MSG, runtime)

        # Assert — the journal's tool message has a ToolResultPart with
        # the tool name and arguments from the original call.
        messages = list(load_journal(runtime.journal._path))
        tool_msgs = [m for m in messages if m.role == "tool"]
        assert len(tool_msgs) >= 1
        result_part = None
        for part in tool_msgs[0].content:
            if isinstance(part, ToolResultPart):
                result_part = part
                break
        assert result_part is not None
        assert result_part.tool_name == "echo"
        assert result_part.arguments == {"text": "hello"}


@pytest.fixture
def record_session(
    make_runtime: Callable[..., Runtime],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> tuple[Runtime, list[tuple[str, str]]]:
    """Run a session that emits three serial ``record`` calls."""
    runtime = make_runtime()
    events: list[tuple[str, str]] = []
    calls = tuple(
        NormalizedToolCall(
            id=f"call_{ix}",
            name="record",
            arguments=json.dumps({"marker": marker}),
        )
        for ix, marker in enumerate(("first", "second", "third"), 1)
    )
    responses = [
        make_fake_response(tool_calls=calls),
        make_fake_response(text="Done."),
    ]

    def complete(request: ModelRequest) -> ModelResponse:
        return responses.pop(0) if len(responses) > 1 else responses[0]

    runtime.model_adapter = patched_adapter(monkeypatch, complete)
    runtime.ordinary_tools = [RecordTool(runtime, tmp_path, events)]
    run_agent(USER_MSG, runtime)
    return runtime, events


class Test04Unit03Runtime03ToolExecutionOrder:
    """Calls from one LLM response execute serially, in emission order."""

    def test_04_unit_03_runtime_03_tool_06_calls_never_overlap(
        self,
        record_session: tuple[Runtime, list[tuple[str, str]]],
    ) -> None:
        # Assert — strictly serial, strictly in LLM emission order: each
        # start/end pair closes before the next call begins, and the
        # markers appear exactly in the order the model sent them.
        _, events = record_session
        assert events == [
            ("start", "first"),
            ("end", "first"),
            ("start", "second"),
            ("end", "second"),
            ("start", "third"),
            ("end", "third"),
        ]

    def test_04_unit_03_runtime_03_tool_07_journal_commits_in_call_order(
        self,
        record_session: tuple[Runtime, list[tuple[str, str]]],
    ) -> None:
        # Assert — the journal commits one tool result per call, in call
        # order, carrying each call's arguments.
        runtime, _ = record_session
        messages = list(load_journal(runtime.journal._path))
        tool_parts = [
            part
            for msg in messages
            if msg.role == "tool"
            for part in msg.content
            if isinstance(part, ToolResultPart)
        ]
        assert [part.tool_id for part in tool_parts] == [
            "call_1",
            "call_2",
            "call_3",
        ]
        markers = [(part.arguments or {}).get("marker") for part in tool_parts]
        assert markers == [
            "first",
            "second",
            "third",
        ]
