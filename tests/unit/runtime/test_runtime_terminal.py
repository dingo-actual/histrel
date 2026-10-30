#!/usr/bin/env python3
"""Unit tests for runtime terminal condition evaluation."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from histrel.core.runtime import (
    Runtime,
    run_agent,
)
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
)
from histrel.terminal import SessionStatus
from tests.unit.runtime.helpers import (
    USER_MSG,
    make_fake_response,
    patched_adapter,
    tool_call,
)

# A minimal edit payload that modifies the tracked main.py. An edit
# that only creates an untracked file leaves the repository unchanged
# from git diff's point of view, so it cannot satisfy the terminal
# condition.
_EDIT_ARGS = json.dumps(
    {
        "file_path": "main.py",
        "file_edits": [
            (
                "/***SEARCH***/\nprint('hello')\n/***END_SEARCH***/\n"
                "/***REPLACE***/\nprint('done')\n/***END_REPLACE***/"
            )
        ],
    }
)


class Test04Unit02Runtime02TerminalConditions:
    """Runtime terminal condition evaluation in the loop."""

    def test_04_unit_02_runtime_02_term_01_budget_turns_terminates(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        runtime = make_runtime(max_turns=1)
        calls: list[ModelRequest] = []

        def fake_complete(request: ModelRequest) -> ModelResponse:
            calls.append(request)
            return make_fake_response(text="thinking...")

        runtime.model_adapter = patched_adapter(monkeypatch, fake_complete)

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert — terminates after 2 turns (max_turns=1, exceeded at 2)
        assert outcome.status == SessionStatus.BUDGET_EXHAUSTED
        assert "max_turns" in outcome.reason
        assert len(calls) == 2  # two model calls before budget exceeded

    def test_04_unit_02_runtime_02_term_02_text_only_after_change_succeeds(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Arrange — the edit changes the repo, the following text-only
        # response ends the session.
        runtime = make_runtime()
        monkeypatch.chdir(tmp_path)

        responses = [
            make_fake_response(tool_calls=(tool_call("edit", _EDIT_ARGS, "call_1"),)),
            make_fake_response(text="Done."),
        ]
        call_count = 0

        def fake_complete(request: ModelRequest) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            return responses[min(call_count - 1, len(responses) - 1)]

        runtime.model_adapter = patched_adapter(monkeypatch, fake_complete)

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert
        assert outcome.status == SessionStatus.SUCCEEDED
        assert outcome.reason == "no_tool_calls_with_repo_changes"
        assert call_count == 2
        assert (tmp_path / "main.py").read_text() == "print('done')\n"

    def test_04_unit_02_runtime_02_term_03_text_only_without_change_continues(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — no tool call and no repo change, so the loop arms the
        # tool-call reminder and continues until the turn budget ends it.
        runtime = make_runtime(max_turns=2)
        calls: list[ModelRequest] = []

        def fake_complete(request: ModelRequest) -> ModelResponse:
            calls.append(request)
            return make_fake_response(text="Nothing to do.")

        runtime.model_adapter = patched_adapter(monkeypatch, fake_complete)

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert
        assert outcome.status == SessionStatus.BUDGET_EXHAUSTED
        assert len(calls) == 3

    def test_04_unit_02_runtime_02_term_04_static_tools_advertised(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Arrange — every session tool is advertised to the provider; a
        # text-only response after the edit ends the session.
        runtime = make_runtime()
        requested_tool_names: list[set[str]] = []

        responses = [
            make_fake_response(tool_calls=(tool_call("edit", _EDIT_ARGS, "call_1"),)),
            make_fake_response(text="Done."),
        ]

        def fake_complete(request: ModelRequest) -> ModelResponse:
            requested_tool_names.append({d.function["name"] for d in request.tools})
            return responses[min(len(requested_tool_names) - 1, len(responses) - 1)]

        runtime.model_adapter = patched_adapter(monkeypatch, fake_complete)
        monkeypatch.chdir(tmp_path)

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert
        assert requested_tool_names, "no model calls were made"
        assert requested_tool_names[0] == {t.name for t in runtime.ordinary_tools}
        assert outcome.status == SessionStatus.SUCCEEDED
