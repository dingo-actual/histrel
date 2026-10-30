#!/usr/bin/env python3
"""Unit tests for the agent loop's static tool set.

Every model call carries the same tool definitions — the session's tools
— and calls are dispatched straight to the named tool. No envelope is
unwrapped and no per-iteration tool list exists.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from histrel.core.runtime import (
    Runtime,
    run_agent,
)
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
)
from histrel.terminal import (
    SessionStatus,
)
from histrel.tools.repository import ReadFileTool
from tests.unit.runtime.helpers import (
    USER_MSG,
    first_tool_request,
    first_tool_result,
    make_fake_response,
    tool_call,
)


class Test04Unit11Runtime11StaticToolset:
    """run_agent: the same tool definitions are sent on every call."""

    def test_04_unit_11_runtime_11_static_01_every_request_carries_the_session_tools(
        self,
        make_stub_runtime: Callable[..., Runtime],
    ) -> None:
        # Arrange
        requests: list[ModelRequest] = []

        def fake_complete(request: ModelRequest) -> ModelResponse:
            requests.append(request)
            if len(requests) == 1:
                return make_fake_response(
                    tool_calls=(tool_call("echo", json.dumps({"text": "hi"})),),
                )
            return make_fake_response(text="Done.")

        runtime = make_stub_runtime(fake_complete)

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert — the session terminated and every model request
        # carried the same single tool definition.
        assert outcome.status == SessionStatus.SUCCEEDED
        assert len(requests) >= 2
        for request in requests:
            assert [t.function.get("name") for t in request.tools] == ["echo"]

    def test_04_unit_11_runtime_11_static_02_call_dispatches_named_tool(
        self,
        make_stub_runtime: Callable[..., Runtime],
    ) -> None:
        # Arrange
        calls: list[ModelRequest] = []

        def fake_complete(request: ModelRequest) -> ModelResponse:
            calls.append(request)
            if len(calls) == 1:
                return make_fake_response(
                    tool_calls=(tool_call("echo", json.dumps({"text": "hello"})),),
                )
            return make_fake_response(text="Done.")

        runtime = make_stub_runtime(fake_complete)

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert — the journal records the call as made, and its result
        # under the same tool name.
        assert outcome.status == SessionStatus.SUCCEEDED
        request_part = first_tool_request(runtime)
        assert request_part.tool_name == "echo"
        assert request_part.arguments == {"text": "hello"}
        result_part = first_tool_result(runtime)
        assert result_part.tool_id == "call_1"
        assert result_part.tool_name == "echo"
        assert result_part.arguments == {"text": "hello"}
        assert result_part.success is True
        assert result_part.output == "hello"

    def test_04_unit_11_runtime_11_static_03_malformed_arguments_record_error(
        self,
        make_stub_runtime: Callable[..., Runtime],
    ) -> None:
        # Arrange — a call with unrepairable JSON is committed as an
        # error tool result instead of crashing the loop.
        calls: list[ModelRequest] = []

        def fake_complete(request: ModelRequest) -> ModelResponse:
            calls.append(request)
            if len(calls) == 1:
                return make_fake_response(tool_calls=(tool_call("echo", "not json"),))
            return make_fake_response(text="Done.")

        runtime = make_stub_runtime(
            fake_complete,
            changes_repo=False,
            max_turns=1,
        )

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert
        assert outcome.status == SessionStatus.BUDGET_EXHAUSTED
        result_part = first_tool_result(runtime)
        assert result_part.tool_id == "call_1"
        assert result_part.tool_name == "echo"
        assert result_part.arguments is None
        assert result_part.success is False
        assert result_part.error is not None
        assert result_part.error

    def test_04_unit_11_runtime_11_static_04_informative_failure_reaches_observations(
        self,
        make_stub_runtime: Callable[..., Runtime],
    ) -> None:
        # Arrange — iteration 1 reads a nonexistent file (a well-specified
        # call that fails informatively); iteration 2 must surface that
        # failure as an observation.
        user_contents: list[str] = []

        def fake_complete(request: ModelRequest) -> ModelResponse:
            for message in request.messages:
                if message.role == "user":
                    user_contents.append(message.content)
            if len(user_contents) == 1:
                return make_fake_response(
                    tool_calls=(
                        tool_call(
                            "read", json.dumps({"path": "nonexistent_gap_file.py"})
                        ),
                    ),
                )
            return make_fake_response(text="Done.")

        runtime = make_stub_runtime(
            fake_complete,
            changes_repo=False,
            max_turns=1,
        )
        runtime.ordinary_tools.append(ReadFileTool())

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert — the turn budget ended the session, but the informative
        # failure was journaled with the informative flag.
        assert outcome.status == SessionStatus.BUDGET_EXHAUSTED
        result_part = first_tool_result(runtime)
        assert result_part.success is False
        assert result_part.informative is True
        assert str(result_part.output).startswith(
            "Attempt to read file 'nonexistent_gap_file.py' failed due to"
        )

        # The iteration-2 user message lists the informative failure as an
        # observation, so the stateless model sees it.
        assert len(user_contents) == 2
        assert "Attempt to read file " in user_contents[1]
        assert "nonexistent_gap_file.py" in user_contents[1]
