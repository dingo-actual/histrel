#!/usr/bin/env python3
"""Unit tests for the tool-call reminder after a text-only turn.

A tolerated turn with no tool calls leaves the provider context
unchanged, so the loop arms a reminder that the next iteration must
carry a tool call. Stateless mode renders it into the step prompt;
react mode commits it as a user message.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from histrel.core.journal import load_journal
from histrel.core.runtime import (
    Runtime,
    run_agent,
)
from histrel.core.runtime.provider_context import (
    TOOL_CALL_REQUIRED_REMINDER,
)
from histrel.model.messages import (
    Message,
    TextPart,
)
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
)
from histrel.terminal import (
    SessionStatus,
)
from tests.unit.runtime.helpers import (
    USER_MSG,
    make_fake_response,
    tool_call,
)
from tests.unit.runtime.stubs import EchoTool


def _message_text(message: Message) -> str:
    """Concatenate the text parts of a committed message."""
    return "".join(part.text for part in message.content if isinstance(part, TextPart))


def _committed_reminders(runtime: Runtime) -> list[Message]:
    """Return the committed user messages that carry the reminder."""
    return [
        msg
        for msg in load_journal(runtime.journal._path)
        if msg.role == "user" and TOOL_CALL_REQUIRED_REMINDER in _message_text(msg)
    ]


def _replayed_reminders(request: ModelRequest) -> list[str]:
    """Return the provider messages that carry the reminder."""
    return [
        msg.content
        for msg in request.messages
        if TOOL_CALL_REQUIRED_REMINDER in msg.content
    ]


def _patch_completer(
    runtime: Runtime,
    monkeypatch: pytest.MonkeyPatch,
    responses: list[ModelResponse],
) -> list[ModelRequest]:
    """Serve the responses in order, repeating the last one thereafter."""
    queue = list(responses)
    calls: list[ModelRequest] = []

    def fake_complete(request: ModelRequest) -> ModelResponse:
        calls.append(request)
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(runtime.model_adapter, "complete", fake_complete)
    return calls


class Test04Unit07Runtime07ToolCallReminder:
    """The reminder that follows a text-only turn."""

    def test_04_unit_07_runtime_07_reminder_01_stateless_step_reminds_required(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Arrange — a tolerated text-only turn must not leave the next
        # stateless prompt unchanged, or the model has nothing new to
        # steer it back to tool use.
        runtime = make_runtime(mode="stateless", max_turns=1)
        calls = _patch_completer(
            runtime, monkeypatch, [make_fake_response(text="All work is done.")]
        )

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert — the initialize prompt carries no reminder, the step
        # prompt that follows the tolerated turn does.
        assert outcome.status == SessionStatus.BUDGET_EXHAUSTED
        assert len(calls) == 2
        assert TOOL_CALL_REQUIRED_REMINDER not in calls[0].messages[-1].content
        assert TOOL_CALL_REQUIRED_REMINDER in calls[1].messages[-1].content

    def test_04_unit_07_runtime_07_reminder_02_react_commits_message(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — react mode replays the conversation, so the reminder
        # must be a committed message rather than a prompt injection.
        runtime = make_runtime(mode="react", max_turns=1)
        _patch_completer(
            runtime, monkeypatch, [make_fake_response(text="All work is done.")]
        )

        # Act
        run_agent(USER_MSG, runtime)

        # Assert — one user reminder is committed per tolerated turn
        assert len(_committed_reminders(runtime)) == 2

    def test_04_unit_07_runtime_07_reminder_02b_react_replays_message(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        runtime = make_runtime(mode="react", max_turns=1)
        calls = _patch_completer(
            runtime, monkeypatch, [make_fake_response(text="All work is done.")]
        )

        # Act
        run_agent(USER_MSG, runtime)

        # Assert — the committed reminder reaches the next request
        assert len(calls) == 2
        assert len(_replayed_reminders(calls[1])) == 1

    def test_04_unit_07_runtime_07_reminder_03_tool_call_clears_it(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Arrange — a tolerated text-only turn arms the reminder, then a
        # tool-calling turn must clear it again.
        runtime = make_runtime(mode="stateless", max_turns=1)
        runtime.ordinary_tools = [EchoTool(runtime, tmp_path)]
        calls = _patch_completer(
            runtime,
            monkeypatch,
            [
                make_fake_response(text="Let me look at the code."),
                make_fake_response(tool_calls=(tool_call("echo", '{"text":"hi"}'),)),
            ],
        )

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert — request 2 follows the tolerated turn and carries the
        # reminder; the tool-calling turn that follows clears the flag.
        assert outcome.status == SessionStatus.BUDGET_EXHAUSTED
        assert len(calls) == 2
        assert _replayed_reminders(calls[1])
        assert runtime.tool_call_required is False
