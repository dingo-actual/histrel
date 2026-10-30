#!/usr/bin/env python3
"""Unit tests for the baseline agent loop over the fixture repository.

Uses a patched model adapter to exercise the full Runtime loop without
real API calls. The agent discovers the defect, fixes it, and the
session terminates successfully.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from histrel.core.config import SystemPrompts
from histrel.core.journal import load_journal
from histrel.core.runtime import run_agent
from histrel.model.messages import (
    ToolRequestPart,
    ToolResultPart,
)
from histrel.model.types import ModelRequest, ModelResponse
from histrel.terminal import SessionStatus
from tests.integration.helpers import (
    USER_MSG,
    PatchedRuntime,
    make_config,
)
from tests.integration.repo import (
    FIXED_CALCULATOR_SOURCE,
)
from tests.integration.scripting import (
    complete_and_change,
    make_completer,
    make_fake_response,
    tool_call,
)
from tests.integration.stubs import EchoTool


class Test01Integration01AgentLoop01Baseline:
    """Baseline agent loop behavior over the fixture repository."""

    def test_01_integration_01_agent_loop_01_baseline_01_solves_fixture(
        self,
        monkeypatch: pytest.MonkeyPatch,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        # Arrange
        session = patched_runtime(make_config())
        runtime = session.runtime
        workspace = session.workspace
        call_count = 0

        def fake_complete(request: ModelRequest) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First turn: read the file
                return make_fake_response(
                    text="Let me look at the code.",
                    tool_calls=(
                        tool_call("read_file", {"path": "calculator.py"}, "call_1"),
                    ),
                )
            if call_count == 2:
                # Second turn: fix the bug and complete
                (workspace / "calculator.py").write_text(
                    FIXED_CALCULATOR_SOURCE, encoding="utf-8"
                )
                complete_and_change(workspace)
                return make_fake_response(text="Fixed the bug.")
            return make_fake_response(text="Done.")

        monkeypatch.setattr(session.adapter, "complete", fake_complete)

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert — session terminated, journal has messages
        assert outcome.status in (
            SessionStatus.SUCCEEDED,
            SessionStatus.BUDGET_EXHAUSTED,
        )
        messages = list(load_journal(runtime.journal._path))
        assert len(messages) >= 2  # task message + at least one assistant

    def test_01_integration_01_agent_loop_01_baseline_02_journal_integrity(
        self,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        # Arrange
        config = make_config(max_turns=3)
        session = patched_runtime(config, complete=make_completer(text="thinking..."))
        runtime = session.runtime

        # Act
        run_agent(USER_MSG, runtime)

        # Assert — journal messages are sequential and well-formed
        messages = list(load_journal(runtime.journal._path))
        indices = [m.message_index for m in messages]
        assert indices == sorted(indices)  # monotonically increasing
        assert all(m.session_id == config.session_id for m in messages)
        assert all(m.source == "bench-agent" for m in messages)

    def test_01_integration_01_agent_loop_01_baseline_05_iteration_system_prompts(
        self,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        # Arrange — distinct initialize and step prompts make the
        # first-iteration selection observable in the built context.
        config = make_config(max_turns=1)
        config = config.model_copy(
            update={
                "agent_system_prompt": SystemPrompts(
                    initialize="INITIALIZE-PROMPT",
                    step="STEP-PROMPT",
                )
            }
        )
        received_system_prompts: list[str] = []

        def fake_complete(request: ModelRequest) -> ModelResponse:
            received_system_prompts.append(request.messages[0].content)
            return make_fake_response(text="thinking...")

        session = patched_runtime(config, complete=fake_complete)

        # Act — max_turns=1 (strict excess) yields exactly two model calls
        run_agent(USER_MSG, session.runtime)

        # Assert — the first iteration receives the initialize prompt and
        # every subsequent iteration receives the step prompt.
        assert received_system_prompts[0] == "INITIALIZE-PROMPT"
        assert received_system_prompts[1:] == ["STEP-PROMPT"] * len(
            received_system_prompts[1:]
        )


class Test01Integration01AgentLoop01BaselineArtifacts:
    """Artifact writing and the tool-execution commit boundary."""

    def test_01_integration_01_agent_loop_01_baseline_03_artifacts_written(
        self,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        # Arrange
        config = make_config()
        session = patched_runtime(config, complete=make_completer(text="Done."))
        runtime = session.runtime

        # Change the repo so the text-only response ends the session
        complete_and_change(session.workspace)

        # Act
        run_agent(USER_MSG, runtime)

        # Assert — all artifact files exist
        artifacts_dir = session.output / config.session_id / "artifacts"
        assert (artifacts_dir / "outcome.json").exists()
        assert (artifacts_dir / "result_manifest.json").exists()
        assert (artifacts_dir / "messages.jsonl").exists()

    def test_01_integration_01_agent_loop_01_baseline_04_tool_execution_boundary(
        self,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        # Arrange — verify assistant message is committed before tool result
        session = patched_runtime(
            make_config(),
            complete=make_completer(
                tool_call("echo", {"text": "hello"}, "call_1"),
                text="Let me read the file.",
            ),
        )
        runtime = session.runtime
        runtime.ordinary_tools = [EchoTool(runtime, session.workspace)]

        # Act
        run_agent(USER_MSG, runtime)

        # Assert — assistant message committed before tool result
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

        # Verify tool result content
        tool_msg = messages[tool_indices[0]]
        tool_part = next(p for p in tool_msg.content if isinstance(p, ToolResultPart))
        assert tool_part.success is True
        assert tool_part.output == "hello"
