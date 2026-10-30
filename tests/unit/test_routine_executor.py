#!/usr/bin/env python3
"""Unit tests for the LLM routine executor.

Tests that RoutineExecutor sends one stateless two-message request
through the model adapter with caller-provided sampling settings,
returns the parsed routine result, and records the response usage; and
that parse_bool strictly parses constrained TRUE/FALSE responses.

    Test Classes:
        Test04Unit38RoutineExecutor01Run: RoutineExecutor.run request
            shape, result parsing, and usage recording.
        Test04Unit38RoutineExecutor02Parse: parse_bool constrained-
            response parsing.

    Test Subtypes:
        Morphism, Failure Mode
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
)
from histrel.model.routine_executor import (
    RoutineExecutor,
    parse_bool,
)
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
    Usage,
)


def _fake_complete_capturing(
    requests: list[ModelRequest],
    response_text: str,
) -> Callable[[ModelRequest], ModelResponse]:
    """Build a fake complete that captures requests and replies with text."""

    def fake_complete(request: ModelRequest) -> ModelResponse:
        requests.append(request)
        return ModelResponse(
            text_parts=(response_text,),
            usage=Usage(input=10, output=2, total=12),
            provider_request_id="req-routine",
            finish_reason="stop",
        )

    return fake_complete


class Test04Unit38RoutineExecutor01Run:
    """RoutineExecutor.run sends one request and returns the parsed result."""

    def test_04_unit_38_routine_executor_01_run_01_morphism_01_sends_two_messages_with_sampling_settings(  # noqa: E501
        self,
        make_patched_adapter: Callable[
            [Callable[[ModelRequest], ModelResponse]], OpenAICompatAdapter
        ],
    ) -> None:
        # Arrange
        requests: list[ModelRequest] = []
        recorded: list[ModelResponse] = []
        executor = RoutineExecutor(
            make_patched_adapter(_fake_complete_capturing(requests, "routine text")),
            usage_recorder=recorded.append,
        )

        # Act
        result = executor.run(
            system_prompt="Classify the code.",
            user_message="The code.",
            parse_response=str.upper,
            temperature=0.0,
            reasoning_effort=None,
        )

        # Assert — one two-message request with no tools and the given
        # sampling settings; the parser converts the response text.
        assert result == "ROUTINE TEXT"
        assert len(requests) == 1
        assert [msg.role for msg in requests[0].messages] == ["system", "user"]
        assert requests[0].messages[0].content == "Classify the code."
        assert requests[0].messages[1].content == "The code."
        assert requests[0].tools == ()
        assert requests[0].temperature == 0.0
        assert requests[0].reasoning_effort is None

    def test_04_unit_38_routine_executor_01_run_02_morphism_02_records_usage_and_request_id(  # noqa: E501
        self,
        make_patched_adapter: Callable[
            [Callable[[ModelRequest], ModelResponse]], OpenAICompatAdapter
        ],
    ) -> None:
        # Arrange
        requests: list[ModelRequest] = []
        recorded: list[ModelResponse] = []

        def fake_complete(request: ModelRequest) -> ModelResponse:
            requests.append(request)
            return ModelResponse(
                text_parts=("TRUE",),
                usage=Usage(input=7, output=1, total=8),
                provider_request_id="req-42",
                finish_reason="stop",
            )

        executor = RoutineExecutor(
            make_patched_adapter(fake_complete), usage_recorder=recorded.append
        )

        # Act
        result = executor.run(
            system_prompt="Classify the code.",
            user_message="The code.",
            parse_response=parse_bool,
            temperature=0.0,
            reasoning_effort="low",
        )

        # Assert
        assert result is True
        assert requests[0].reasoning_effort == "low"
        assert executor.last_usage.total == 8
        assert executor.last_provider_request_id == "req-42"
        # The recorder receives the whole response: usage tokens and
        # provider request ID for ledger recording.
        assert len(recorded) == 1
        assert recorded[0].usage == Usage(
            input=7, output=1, reasoning=0, total=8, cached=0
        )
        assert recorded[0].provider_request_id == "req-42"


class Test04Unit38RoutineExecutor02Parse:
    """parse_bool strictly parses TRUE/FALSE routine responses."""

    @pytest.mark.parametrize(
        ("response_text", "expected"),
        [
            ("TRUE", True),
            ("  true \n", True),
            ("True.", True),
            ("FALSE", False),
            (" false\n", False),
            ("False.", False),
        ],
    )
    def test_04_unit_38_routine_executor_02_parse_01_morphism_01_true_false_spellings(
        self,
        response_text: str,
        expected: bool,
    ) -> None:
        # Act
        result = parse_bool(response_text)

        # Assert
        assert result is expected

    @pytest.mark.parametrize("response_text", ["MAYBE", "", "TRUE FALSE", "T"])
    def test_04_unit_38_routine_executor_02_parse_02_failure_01_unparseable_text_raises(  # noqa: E501
        self,
        response_text: str,
    ) -> None:
        # Act / Assert — anything but TRUE/FALSE fails loudly instead
        # of guessing a default classification.
        with pytest.raises(ValueError, match="TRUE or FALSE"):
            parse_bool(response_text)
