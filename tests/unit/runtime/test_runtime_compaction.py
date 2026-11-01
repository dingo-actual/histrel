#!/usr/bin/env python3
"""Unit tests for runtime context compaction.

Context preparation checks the estimated outgoing prompt against the
usable input budget. When it is exceeded, the compactor summarizes the
middle of the presented history, the state is kept on the runtime, and
the request is rebuilt from that state.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from histrel.core.config import Compaction
from histrel.core.journal import load_journal
from histrel.core.runtime import (
    Runtime,
    run_agent,
)
from histrel.core.runtime.provider_context import CONTEXT_CHECKPOINT_PREFIX
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
)
from histrel.terminal import SessionStatus
from tests.unit.runtime.helpers import (
    USER_MSG,
    make_fake_response,
    tool_call,
)
from tests.unit.runtime.stubs import EchoTool

# The compaction summary request is recognized by its system prompt.
SUMMARIZATION_MARKER = "You are a summarization agent"

SUMMARY_TEXT = "Earlier work: edited main.py."


def _is_summary_request(request: ModelRequest) -> bool:
    """Report whether a request asks for a compaction summary."""
    return request.messages[0].content.startswith(SUMMARIZATION_MARKER)


def _request_text(request: ModelRequest) -> str:
    """Flatten a request's messages and tool calls into one string."""
    parts = [message.content for message in request.messages]
    for message in request.messages:
        for call in message.tool_calls or ():
            parts.append(json.dumps(call, sort_keys=True))
    return "\n".join(parts)


def _patch_completer(
    runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
    responses: list[ModelResponse],
    *,
    summary_fails: bool = False,
) -> tuple[list[ModelRequest], list[ModelRequest]]:
    """Serve the responses in order and answer summary requests.

    Returns:
        (tuple) The model requests and the summary requests seen.
    """
    queue = list(responses)
    calls: list[ModelRequest] = []
    summaries: list[ModelRequest] = []

    def fake_complete(request: ModelRequest) -> ModelResponse:
        if _is_summary_request(request):
            summaries.append(request)
            if summary_fails:
                raise RuntimeError("summarization failed")
            return make_fake_response(text=SUMMARY_TEXT)
        calls.append(request)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(runtime.model_adapter, "complete", fake_complete)
    return calls, summaries


def _compacting_config(*, retain_tail: int = 2) -> Compaction:
    """Build compaction settings that always trigger."""
    return Compaction(compact_at_fraction=0.0, compact_retain_tail=retain_tail)


def _two_tool_turns_then_done() -> list[ModelResponse]:
    """Return responses that fill the journal, then end the session."""
    return [
        make_fake_response(tool_calls=(tool_call("echo", '{"text":"first"}'),)),
        make_fake_response(tool_calls=(tool_call("echo", '{"text":"second"}'),)),
        make_fake_response(text="All work is done."),
    ]


class Test04Unit04Runtime08Compaction:
    """Compaction during context preparation."""

    def test_04_unit_04_runtime_08_compaction_01_no_state_below_threshold(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — the default threshold is far above this prompt
        runtime = make_runtime(mode="react", max_turns=1)
        _, summaries = _patch_completer(
            runtime, monkeypatch, [make_fake_response(text="All work is done.")]
        )

        # Act
        run_agent(USER_MSG, runtime)

        # Assert — nothing is compacted and no summary call was made
        assert runtime.compacted_state is None
        assert summaries == []
        assert runtime.session_events.compactions == []

    def test_04_unit_04_runtime_08_compaction_02_react_keeps_only_retained(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Arrange — the third iteration exceeds the threshold with two
        # turns of history to compact
        runtime = make_runtime(
            mode="react", max_turns=50, compaction=_compacting_config()
        )
        runtime.ordinary_tools = [EchoTool(runtime, tmp_path)]
        calls, summaries = _patch_completer(
            runtime, monkeypatch, _two_tool_turns_then_done()
        )

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert — the summary replaced the compacted middle
        assert outcome.status == SessionStatus.SUCCEEDED
        assert len(summaries) == 1
        assert runtime.compacted_state is not None
        assert runtime.compacted_state.working_summary == SUMMARY_TEXT
        last_request = _request_text(calls[-1])
        assert f"{CONTEXT_CHECKPOINT_PREFIX}{SUMMARY_TEXT}" in last_request
        assert "second" in last_request
        assert "first" not in last_request
        # The compacted turn was visible before the compaction happened
        assert "first" in _request_text(calls[1])

    def test_04_unit_04_runtime_08_compaction_03_records_event(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Arrange
        runtime = make_runtime(
            mode="react", max_turns=50, compaction=_compacting_config()
        )
        runtime.ordinary_tools = [EchoTool(runtime, tmp_path)]
        _patch_completer(runtime, monkeypatch, _two_tool_turns_then_done())

        # Act
        run_agent(USER_MSG, runtime)

        # Assert — one event covers the two compacted messages
        assert len(runtime.session_events.compactions) == 1
        event = runtime.session_events.compactions[0]
        assert event.compacted_message_count == 2
        assert event.summary == SUMMARY_TEXT

    def test_04_unit_04_runtime_08_compaction_04_stateless_renders_summary(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Arrange — stateless mode renders a fresh step prompt each turn
        runtime = make_runtime(
            mode="stateless", max_turns=50, compaction=_compacting_config()
        )
        runtime.ordinary_tools = [EchoTool(runtime, tmp_path)]
        calls, _ = _patch_completer(runtime, monkeypatch, _two_tool_turns_then_done())

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert — the prompt carries the summary and only the retained
        # observation
        assert outcome.status == SessionStatus.SUCCEEDED
        step_prompt = calls[-1].messages[-1].content
        assert "## Context Checkpoint" in step_prompt
        assert SUMMARY_TEXT in step_prompt
        assert "second" in step_prompt
        assert "first" not in step_prompt
        assert "first" in calls[1].messages[-1].content

    def test_04_unit_04_runtime_08_compaction_05_summary_failure_is_infrastructure_error(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Arrange — one compactable message, and the summary call fails.
        # Preparation makes that call, so it must share the error path.
        runtime = make_runtime(
            mode="react", max_turns=50, compaction=_compacting_config(retain_tail=1)
        )
        runtime.ordinary_tools = [EchoTool(runtime, tmp_path)]
        _patch_completer(
            runtime,
            monkeypatch,
            _two_tool_turns_then_done(),
            summary_fails=True,
        )

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert — the failure ends the session as an infrastructure error
        # instead of escaping the loop
        assert outcome.status == SessionStatus.INFRASTRUCTURE_ERROR
        assert "summarization failed" in outcome.reason
        assert runtime.compacted_state is None
        assert any(
            message.error is not None for message in load_journal(runtime.journal._path)
        )
