#!/usr/bin/env python3
"""Integration tests for stateless user-message rendering.

Tests the per-iteration user-message pipeline: the first iteration sends
the system message plus the rendered ``user_first.j2`` message; later
iterations send the system message plus the rendered ``user_step.j2``
message with observations, reasoning state, and the iteration
counter. Every LLM call receives only the two messages and the static
session tool set.

Tool-success assertions are deliberately conservative: scenarios use
tool calls that succeed, and no assertion depends on how tool-call
failure outcomes are filtered (a refactor will change that logic).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from histrel.core.runtime import run_agent
from histrel.model.types import ModelRequest
from tests.integration.helpers import (
    USER_MSG,
    PatchedRuntime,
    make_config,
    with_distinct_prompts,
)
from tests.integration.react_mode.helpers import SESSION_TOOL_NAMES
from tests.integration.scripting import (
    make_fake_response,
    section,
    tool_call,
)


class Test01Integration05UserMessages01Deterministic:
    """Deterministic agent-loop tests for the user-message pipeline.

    Uses a patched model adapter (last resort) with plausible scripted
    responses, so the full Runtime loop runs without real API calls.
    """

    def test_01_integration_05_user_messages_01_deterministic_01_first_iteration_01_init_message_shape(  # noqa: E501
        self,
        scripted_session: Callable[..., tuple[PatchedRuntime, list[ModelRequest]]],
    ) -> None:
        """The first LLM call gets the init system prompt plus only the
        rendered user_first.j2 message, with every session tool passed.
        """
        # Arrange
        config = with_distinct_prompts(make_config())
        session, recorded = scripted_session(
            make_fake_response(text="Planning the fix."), config=config
        )

        # Act
        run_agent(USER_MSG, session.runtime)

        # Assert — first call: system + user only, init system prompt.
        assert len(recorded) >= 1
        first = recorded[0]
        assert len(first.messages) == 2
        assert first.messages[0].role == "system"
        assert first.messages[0].content == "INITIALIZE-PROMPT"
        assert first.messages[1].role == "user"

        # The user message is the rendered user_first.j2 output.
        user_content = first.messages[1].content
        assert "# Inputs" in user_content
        assert USER_MSG in user_content
        assert "## Observations" not in user_content

        # The static session tools are passed on every call.
        assert {tool.function["name"] for tool in first.tools} == SESSION_TOOL_NAMES

    def test_01_integration_05_user_messages_01_deterministic_02_step_iteration_01_request_shape(  # noqa: E501
        self,
        tmp_path: Path,
        scripted_session: Callable[..., tuple[PatchedRuntime, list[ModelRequest]]],
    ) -> None:
        """A later LLM call switches to the step system prompt and keeps
        the two-message stateless shape with the static tool set.
        """
        # Arrange — turn 1 reads the file via the run envelope; turn 2
        # is text-only, so iteration 2 renders the step message.
        calculator = str(tmp_path / "workspace" / "calculator.py")
        config = with_distinct_prompts(make_config())
        session, recorded = scripted_session(
            make_fake_response(
                text="Initial reasoning-state: inspect the calculator.",
                tool_calls=(tool_call("read", {"path": calculator}, "call_1"),),
            ),
            make_fake_response(text="Second reasoning-state."),
            config=config,
        )

        # Act
        run_agent(USER_MSG, session.runtime)

        # Assert — second call: system + user only, step system prompt.
        assert len(recorded) >= 2
        second = recorded[1]
        assert len(second.messages) == 2
        assert second.messages[0].role == "system"
        assert second.messages[0].content == "STEP-PROMPT"
        assert second.messages[1].role == "user"

        # The instance problem is repeated in every iteration.
        user_content = second.messages[1].content
        assert "# Inputs" in user_content
        assert USER_MSG in user_content

        # The static session tools are passed on every call.
        assert {tool.function["name"] for tool in second.tools} == SESSION_TOOL_NAMES

    def test_01_integration_05_user_messages_01_deterministic_02_step_iteration_02_observation_and_reasoning(  # noqa: E501
        self,
        tmp_path: Path,
        scripted_session: Callable[..., tuple[PatchedRuntime, list[ModelRequest]]],
    ) -> None:
        """The step message carries the iteration-prefixed observation,
        the raw read output, the most recent assistant text as the
        reasoning state, and the call header naming the tool and its
        arguments.
        """
        # Arrange
        calculator = str(tmp_path / "workspace" / "calculator.py")
        config = with_distinct_prompts(make_config())
        session, recorded = scripted_session(
            make_fake_response(
                text="Initial reasoning-state: inspect the calculator.",
                tool_calls=(tool_call("read", {"path": calculator}, "call_1"),),
            ),
            make_fake_response(text="Second reasoning-state."),
            config=config,
        )

        # Act
        run_agent(USER_MSG, session.runtime)

        # Assert — the observations section carries the iteration prefix,
        # the call the agent made with its arguments, and the raw
        # read-tool output under a separator.
        assert len(recorded) >= 2
        user_content = recorded[1].messages[1].content
        observations = section(user_content, "Observations")
        assert "Current iteration: 2" in observations
        assert f'[Iteration 1] read(path="{calculator}")' in observations
        assert "---" in observations
        assert "def add" in observations

        # The reasoning state is the most recent assistant text.
        assert "Initial reasoning-state: inspect the calculator." in section(
            user_content, "Current Reasoning-State"
        )


class Test01Integration05UserMessages01DeterministicState:
    """Observation accumulation across iterations."""

    def test_01_integration_05_user_messages_01_deterministic_04_step_iteration_03_after_one_read(  # noqa: E501
        self,
        tmp_path: Path,
        scripted_session: Callable[..., tuple[PatchedRuntime, list[ModelRequest]]],
    ) -> None:
        """After one tool call, the iteration-2 observation section holds
        exactly that one observation.
        """
        # Arrange
        calculator = str(tmp_path / "workspace" / "calculator.py")
        readme = str(tmp_path / "workspace" / "README.md")
        config = with_distinct_prompts(make_config())
        session, recorded = scripted_session(
            make_fake_response(
                text="Reading the module under test.",
                tool_calls=(tool_call("read", {"path": calculator}, "call_1"),),
            ),
            make_fake_response(
                text="Reading the README for context.",
                tool_calls=(tool_call("read", {"path": readme}, "call_2"),),
            ),
            make_fake_response(text="No further reads needed."),
            config=config,
        )

        # Act
        run_agent(USER_MSG, session.runtime)

        # Assert — iteration 2 sees only the first observation.
        assert len(recorded) >= 3
        second = section(recorded[1].messages[1].content, "Observations")
        assert f'[Iteration 1] read(path="{calculator}")' in second
        assert "[Iteration 2]" not in second

    def test_01_integration_05_user_messages_01_deterministic_04_step_iteration_03_after_two_reads(  # noqa: E501
        self,
        tmp_path: Path,
        scripted_session: Callable[..., tuple[PatchedRuntime, list[ModelRequest]]],
    ) -> None:
        """After two tool calls, the iteration-3 observation section holds
        both observations in commit order.
        """
        # Arrange
        calculator = str(tmp_path / "workspace" / "calculator.py")
        readme = str(tmp_path / "workspace" / "README.md")
        config = with_distinct_prompts(make_config())
        session, recorded = scripted_session(
            make_fake_response(
                text="Reading the module under test.",
                tool_calls=(tool_call("read", {"path": calculator}, "call_1"),),
            ),
            make_fake_response(
                text="Reading the README for context.",
                tool_calls=(tool_call("read", {"path": readme}, "call_2"),),
            ),
            make_fake_response(text="No further reads needed."),
            config=config,
        )

        # Act
        run_agent(USER_MSG, session.runtime)

        # Assert — iteration 3 sees both observations, in order.
        assert len(recorded) >= 3
        third_content = recorded[2].messages[1].content
        third = section(third_content, "Observations")
        assert f'[Iteration 1] read(path="{calculator}")' in third
        assert f'[Iteration 2] read(path="{readme}")' in third
        assert third.index(f'[Iteration 1] read(path="{calculator}")') < third.index(
            f'[Iteration 2] read(path="{readme}")'
        )
