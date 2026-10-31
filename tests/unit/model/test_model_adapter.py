#!/usr/bin/env python3
"""Unit tests for the model adapter types and normalization.

Tests provider-independent request/response types, usage normalization,
tool call normalization, and the provider guard. Uses parametrized tests
for broad coverage of usage mapping and guard logic.
"""

from __future__ import annotations

import pytest

from histrel.core.config import AgentConfig, SystemPrompts
from histrel.core.errors import UnsupportedProviderError
from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
    build_client,
    map_usage,
    normalize_tool_calls,
)
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
    NormalizedToolCall,
    ProviderMessage,
    ToolDefinition,
    Usage,
)
from tests.unit.factories import build_agent_config
from tests.unit.model.helpers import FakeCompletionsClient


@pytest.mark.parametrize(
    "prompt_tokens, completion_tokens, cached, reasoning, expected_input, expected_output, expected_cached, expected_reasoning",
    [
        (100, 50, 0, 0, 100, 50, 0, 0),
        (1000, 500, 300, 0, 1000, 500, 300, 0),
        (500, 200, 0, 80, 500, 200, 0, 80),
        (2000, 1000, 500, 200, 2000, 1000, 500, 200),
        (0, 0, 0, 0, 0, 0, 0, 0),
    ],
    ids=[
        "no_cache_no_reasoning",
        "with_cache",
        "with_reasoning",
        "with_both",
        "all_zero",
    ],
)
class Test04Unit01Model01UsageMapping:
    """Parametrized usage normalization from provider fields."""

    def test_04_unit_01_model_01_usage_01_maps_fields(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        cached: int,
        reasoning: int,
        expected_input: int,
        expected_output: int,
        expected_cached: int,
        expected_reasoning: int,
    ) -> None:
        # Arrange — build a usage object as it would come from the provider
        provider_usage = Usage(
            input=prompt_tokens,
            output=completion_tokens,
            reasoning=reasoning,
            total=prompt_tokens + completion_tokens,
            cached=cached,
        )

        # Act — map_usage should produce the normalized form
        result = map_usage(provider_usage)

        # Assert
        assert result.input == expected_input
        assert result.output == expected_output
        assert result.cached == expected_cached
        assert result.reasoning == expected_reasoning
        assert result.total == expected_input + expected_output


class Test04Unit02Model02UsageValidation:
    """Usage normalization edge cases and validation."""

    def test_04_unit_02_model_02_usage_03_preserves_unavailable_flag(
        self,
    ) -> None:
        # Act
        result = map_usage(Usage(available=False))

        # Assert
        assert result.available is False
        assert result.total == 0


class Test04Unit02Model02StreamUsageGuard:
    """A stream that ends without usage must not look measured."""

    def test_04_unit_02_model_02_streamusage_01_marks_usage_unavailable(
        self,
    ) -> None:
        # Arrange — the provider sends no usage chunk at all
        adapter = OpenAICompatAdapter(
            client=FakeCompletionsClient(with_usage=False),  # type: ignore[arg-type]
            model_name="gpt-4o",
            provider="fireworks",
        )
        request = ModelRequest(
            messages=(ProviderMessage(role="user", content="hello"),),
        )

        # Act
        response = adapter.complete(request)

        # Assert
        assert response.usage.available is False
        assert response.usage.input == 0
        assert response.usage.total == 0

    def test_04_unit_02_model_02_streamusage_02_keeps_measured_usage(
        self,
    ) -> None:
        # Arrange
        adapter = OpenAICompatAdapter(
            client=FakeCompletionsClient(),  # type: ignore[arg-type]
            model_name="gpt-4o",
            provider="fireworks",
        )
        request = ModelRequest(
            messages=(ProviderMessage(role="user", content="hello"),),
        )

        # Act
        response = adapter.complete(request)

        # Assert
        assert response.usage.available is True
        assert response.usage.input == 10

    def test_04_unit_02_model_02_usage_01_cached_greater_than_input_fails(
        self,
    ) -> None:
        # Arrange — cached tokens cannot exceed input tokens
        bad_usage = Usage(
            input=100,
            output=50,
            cached=200,  # > input
            total=150,
        )

        # Act & Assert
        with pytest.raises(ValueError, match="cached"):
            map_usage(bad_usage)

    def test_04_unit_02_model_02_usage_02_negative_input_fails(self) -> None:
        # Arrange
        bad_usage = Usage(
            input=-10,
            output=50,
            total=40,
        )

        # Act & Assert
        with pytest.raises(ValueError, match="negative"):
            map_usage(bad_usage)


# ---------------------------------------------------------------------------
# Parametrized tool call normalization tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw_tool_calls, expected",
    [
        # Single tool call
        (
            {
                0: {
                    "id": "call_1",
                    "name": "read_file",
                    "arguments": '{"path": "src/main.py"}',
                }
            },
            (
                NormalizedToolCall(
                    id="call_1", name="read_file", arguments='{"path": "src/main.py"}'
                ),
            ),
        ),
        # Multiple tool calls
        (
            {
                0: {"id": "call_a", "name": "read", "arguments": "{}"},
                1: {"id": "call_b", "name": "write", "arguments": '{"path": "x"}'},
            },
            (
                NormalizedToolCall(id="call_a", name="read", arguments="{}"),
                NormalizedToolCall(
                    id="call_b", name="write", arguments='{"path": "x"}'
                ),
            ),
        ),
        # Empty
        (
            {},
            (),
        ),
        # Arguments accumulated across chunks
        (
            {
                0: {
                    "id": "call_1",
                    "name": "write",
                    "arguments": '{"path": "src',
                }
            },
            (NormalizedToolCall(id="call_1", name="write", arguments='{"path": "src'),),
        ),
    ],
    ids=[
        "single_call",
        "multiple_calls",
        "empty",
        "partial_arguments",
    ],
)
class Test04Unit03Model03ToolCallNormalization:
    """Parametrized tool call normalization from streaming fragments."""

    def test_04_unit_03_model_03_toolcall_01_normalizes(
        self,
        raw_tool_calls: dict,
        expected: tuple,
    ) -> None:
        # Act
        result = normalize_tool_calls(raw_tool_calls)

        # Assert
        assert result == expected


# ---------------------------------------------------------------------------
# Provider guard tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "provider",
    ["openai", "fireworks"],
)
class Test04Unit04Model04ProviderGuard:
    """The adapter accepts each supported provider identifier."""

    def test_04_unit_04_model_04_guard_01_adapter_constructor(
        self,
        provider: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")

        config = build_agent_config(
            provider=provider,
            session_id="s",
            task_id="t",
            model_name="m",
            agent_system_prompt=SystemPrompts(initialize="x", step="x"),
        )
        adapter = OpenAICompatAdapter(
            client=build_client(config),
            model_name="m",
            provider=provider,
        )

        # Assert
        assert adapter.provider == provider

    def test_04_unit_04_model_04_guard_02_unknown_provider_rejected(
        self,
        provider: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — bypass validation to reach the client builder guard
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config = AgentConfig.model_construct(model_provider="claude")

        # Act & Assert
        with pytest.raises(UnsupportedProviderError):
            build_client(config)


# ---------------------------------------------------------------------------
# ModelRequest / ModelResponse model tests
# ---------------------------------------------------------------------------


class Test04Unit05Model05RequestResponse:
    """ModelRequest and ModelResponse model validation."""

    def test_04_unit_05_model_05_reqresp_01_request_defaults(self) -> None:
        # Act
        request = ModelRequest(
            messages=(ProviderMessage(role="user", content="hello"),),
        )

        # Assert
        assert request.tools == ()
        assert request.temperature == 1.0
        assert request.top_p == 1.0
        assert request.reasoning_effort is None
        assert request.max_tokens_output is None

    def test_04_unit_05_model_05_reqresp_02_response_defaults(self) -> None:
        # Act
        response = ModelResponse(
            usage=Usage(),
        )

        # Assert
        assert response.text_parts == ()
        assert response.tool_calls == ()
        assert response.provider_request_id is None
        assert response.finish_reason is None
        assert response.raw_metadata == {}

    def test_04_unit_05_model_05_reqresp_03_tool_definition(self) -> None:
        # Act
        tool = ToolDefinition(
            type="function",
            function={
                "name": "read_file",
                "description": "Read a file",
                "parameters": {"type": "object", "properties": {}},
            },
        )

        # Assert
        assert tool.type == "function"
        assert tool.function["name"] == "read_file"

    def test_04_unit_05_model_05_reqresp_04_normalized_tool_call(self) -> None:
        # Act
        call = NormalizedToolCall(
            id="call_1",
            name="read_file",
            arguments='{"path": "src/main.py"}',
        )

        # Assert
        assert call.id == "call_1"
        assert call.name == "read_file"
        assert call.arguments == '{"path": "src/main.py"}'
