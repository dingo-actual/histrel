#!/usr/bin/env python3
"""Live LLM integration tests for the user-message pipeline.

These tests make real API calls to Fireworks AI using the model
"accounts/fireworks/models/deepseek-v4p1-flash-1017". They skip unless --llm (or
--all) is provided and FIREWORKS_API_KEY is set. Assertions are limited
to what holds regardless of the live model's choices: no assertion
depends on live tool outcomes.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from histrel.core.journal import load_journal
from histrel.core.runtime import run_agent
from histrel.terminal import SessionStatus
from tests.integration.environment import (
    FIREWORKS_MODEL,
    RecordingAdapter,
)
from tests.integration.helpers import (
    USER_MSG,
    PatchedRuntime,
    make_config,
    with_distinct_prompts,
)
from tests.integration.react_mode.helpers import SESSION_TOOL_NAMES
from tests.integration.scripting import (
    section,
)


@pytest.mark.requires_live_llm
class Test01Integration05UserMessages02LiveLlm:
    """Live LLM tests for the user-message pipeline using Fireworks AI."""

    def test_01_integration_05_user_messages_02_live_llm_01_first_iteration_01_fireworks_init_shape(
        self,
        fireworks_recording_adapter: RecordingAdapter,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        """A live first iteration receives the init system prompt plus
        only the rendered user_first.j2 message, with every session
        tool passed.
        """
        # Arrange
        config = with_distinct_prompts(
            make_config(
                max_turns=1,
                model_provider="fireworks",
                model_name=FIREWORKS_MODEL,
                api_key_env="FIREWORKS_API_KEY",
            )
        )
        session = patched_runtime(
            config,
            adapter=fireworks_recording_adapter,
            api_key_env=None,
        )
        adapter = fireworks_recording_adapter

        # Act
        outcome = run_agent(USER_MSG, session.runtime)

        # Assert — the live session ran and the first request kept the
        # stateless two-message shape with the init system prompt.
        assert outcome.status != SessionStatus.INFRASTRUCTURE_ERROR
        assert len(adapter.recorded) >= 1
        first = adapter.recorded[0]
        assert [m.role for m in first.messages] == ["system", "user"]
        assert first.messages[0].content == "INITIALIZE-PROMPT"
        user_content = first.messages[1].content
        assert "# Inputs" in user_content
        assert USER_MSG in user_content
        assert "## Observations" not in user_content
        assert {tool.function["name"] for tool in first.tools} == SESSION_TOOL_NAMES

    def test_01_integration_05_user_messages_02_live_llm_01_first_iteration_02_fireworks_usage_and_journal(
        self,
        fireworks_recording_adapter: RecordingAdapter,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        """A live session tracks real token usage and commits at least
        the task message plus one assistant message to the journal.
        """
        # Arrange
        config = with_distinct_prompts(
            make_config(
                max_turns=1,
                model_provider="fireworks",
                model_name=FIREWORKS_MODEL,
                api_key_env="FIREWORKS_API_KEY",
            )
        )
        session = patched_runtime(
            config,
            adapter=fireworks_recording_adapter,
            api_key_env=None,
        )

        # Act
        run_agent(USER_MSG, session.runtime)

        # Assert — real usage and journal content from the provider.
        assert session.runtime.token_ledger.total_input > 0
        assert session.runtime.token_ledger.total_output > 0
        assert len(list(load_journal(session.runtime.journal._path))) >= 2

    def test_01_integration_05_user_messages_02_live_llm_02_step_iteration_01_fireworks_step_shape(
        self,
        fireworks_recording_adapter: RecordingAdapter,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        """Live later iterations receive the step system prompt plus the
        rendered user_step.j2 message with the iteration counter and
        observations section, keeping the two-message shape throughout.
        """
        # Arrange
        config = with_distinct_prompts(
            make_config(
                max_turns=3,
                model_provider="fireworks",
                model_name=FIREWORKS_MODEL,
                api_key_env="FIREWORKS_API_KEY",
            )
        )
        session = patched_runtime(
            config,
            adapter=fireworks_recording_adapter,
            api_key_env=None,
        )
        adapter = fireworks_recording_adapter

        # Act
        run_agent(USER_MSG, session.runtime)

        # Assert — every live call kept the stateless two-message shape
        # and the static tool set. Tool-less requests are repair-routine
        # calls, which are not part of the agent loop.
        loop_requests = [request for request in adapter.recorded if request.tools]
        assert len(loop_requests) >= 2
        for request in loop_requests:
            assert [m.role for m in request.messages] == ["system", "user"]
            assert {
                tool.function["name"] for tool in request.tools
            } == SESSION_TOOL_NAMES

        second = loop_requests[1]
        assert second.messages[0].content == "STEP-PROMPT"
        user_content = second.messages[1].content
        assert "# Inputs" in user_content
        assert USER_MSG in user_content
        observations = section(user_content, "Observations")
        assert "Current iteration: 2" in observations
        assert "- Name:" not in observations
