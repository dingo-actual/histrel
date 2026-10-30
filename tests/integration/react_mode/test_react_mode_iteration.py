#!/usr/bin/env python3
"""Integration tests for react-mode request shape and history replay.

The first iteration sends the system message plus the user message with
the real (converted) tool definitions for the projected tools. Later
iterations replay the full conversation history — assistant messages,
tool calls, and tool call results — instead of a projected stateless
user message.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from histrel.core.runtime import run_agent
from histrel.model.types import ModelRequest
from tests.integration.helpers import (
    USER_MSG,
    PatchedRuntime,
    make_config,
)
from tests.integration.react_mode.helpers import (
    REACT_SYSTEM_PROMPT,
    SESSION_TOOL_NAMES,
    make_react_config,
)
from tests.integration.scripting import (
    make_fake_response,
)


class Test01Integration06ReactMode01Deterministic:
    """Deterministic react-mode agent loop tests."""

    def test_01_integration_06_react_mode_01_deterministic_01_first_iteration_01_request_shape(
        self,
        scripted_session: Callable[..., tuple[PatchedRuntime, list[ModelRequest]]],
    ) -> None:
        """The first react-mode LLM call sends the system prompt, the
        passed-in user message, and the definitions of every session
        tool.
        """
        # Arrange
        config = make_react_config(make_config())
        session, recorded = scripted_session(
            make_fake_response(text="Not done yet."),
            make_fake_response(text="Still working."),
            make_fake_response(text="Still working."),
            config=config,
        )

        # Act
        run_agent(USER_MSG, session.runtime)

        # Assert — system prompt plus the raw user message, no history.
        assert len(recorded) >= 1
        first = recorded[0]
        assert len(first.messages) == 2
        assert first.messages[0].role == "system"
        assert first.messages[0].content == REACT_SYSTEM_PROMPT
        assert first.messages[1].role == "user"
        assert first.messages[1].content == USER_MSG

        # The static session tools are passed on every call; the
        # converted definitions carry their function payloads.
        tool_names = {tool.function["name"] for tool in first.tools}
        assert tool_names == SESSION_TOOL_NAMES
        for tool in first.tools:
            assert tool.type == "function"
            assert "parameters" in tool.function
            assert tool.function["description"]

    def test_01_integration_06_react_mode_01_deterministic_02_step_iteration_01_history_roles(
        self,
        history_session: tuple[str, list[ModelRequest]],
    ) -> None:
        """A later react-mode LLM call replays the shared system prompt,
        the user message, then the prior assistant and tool messages.
        """
        # Arrange
        _, recorded = history_session

        # Assert — history replay: system, user, assistant, tool.
        assert len(recorded) >= 2
        second = recorded[1]
        roles = [message.role for message in second.messages]
        assert roles == ["system", "user", "assistant", "tool"]

        # The system prompt and user message repeat verbatim.
        assert second.messages[0].content == REACT_SYSTEM_PROMPT
        assert second.messages[1].content == USER_MSG

    def test_01_integration_06_react_mode_01_deterministic_02_step_iteration_02_history_tool_call(
        self,
        history_session: tuple[str, list[ModelRequest]],
    ) -> None:
        """The replayed assistant message carries its text and raw tool
        call, and the tool result follows it keyed by call ID.
        """
        # Arrange
        calculator, recorded = history_session

        # Assert — the assistant message carries its text and raw call.
        assert len(recorded) >= 2
        second = recorded[1]
        assistant = second.messages[2]
        assert assistant.content == "Reading the module."
        assert assistant.tool_calls is not None
        assert len(assistant.tool_calls) == 1
        call = assistant.tool_calls[0]
        assert call["id"] == "call_1"
        assert call["type"] == "function"
        assert call["function"]["name"] == "read"
        assert json.loads(call["function"]["arguments"]) == {"path": calculator}

        # The tool result follows its tool call, keyed by call ID.
        tool_message = second.messages[3]
        assert tool_message.role == "tool"
        assert tool_message.tool_call_id == "call_1"
        assert "Tool: read" in tool_message.content
        assert "def add" in tool_message.content
