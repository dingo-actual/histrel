#!/usr/bin/env python3
"""Unit tests for adapter request-kwarg forwarding.

Tests that the adapter forwards the prompt cache key, the structured
response format, and the session affinity headers to
chat.completions.create, and omits each when the request does not carry
it. Uses a recording client double (no mocks).
"""

from __future__ import annotations

from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
)
from histrel.model.types import (
    ModelRequest,
    ProviderMessage,
)
from tests.unit.model.helpers import FakeCompletionsClient


class Test04Unit06Model06PromptCacheKey:
    """Adapter forwards prompt_cache_key to chat.completions.create."""

    def test_04_unit_06_model_06_cache_01_forwards_key_in_create_kwargs(
        self, fake_completions_client: FakeCompletionsClient
    ) -> None:
        # Arrange — a fireworks adapter and a request carrying a key
        adapter = OpenAICompatAdapter(
            client=fake_completions_client,  # type: ignore[arg-type]
            model_name="gpt-4o",
            provider="fireworks",
        )
        request = ModelRequest(
            messages=(ProviderMessage(role="user", content="hello"),),
            prompt_cache_key="cache-key-123",
        )

        # Act
        response = adapter.complete(request)

        # Assert — the key reaches the create call verbatim, and the
        # aggregated response is unaffected by its presence.
        captured = fake_completions_client.captured_kwargs
        assert captured["prompt_cache_key"] == "cache-key-123"
        assert response.finish_reason == "stop"

    def test_04_unit_06_model_06_cache_02_omits_key_when_request_has_none(
        self, fake_completions_client: FakeCompletionsClient
    ) -> None:
        # Arrange — a request without a prompt_cache_key
        adapter = OpenAICompatAdapter(
            client=fake_completions_client,  # type: ignore[arg-type]
            model_name="gpt-4o",
            provider="fireworks",
        )
        request = ModelRequest(
            messages=(ProviderMessage(role="user", content="hello"),),
        )

        # Act
        adapter.complete(request)

        # Assert — no empty or null cache key leaks into the request
        assert "prompt_cache_key" not in fake_completions_client.captured_kwargs


# ---------------------------------------------------------------------------
# Response format forwarding tests
# ---------------------------------------------------------------------------


class Test04Unit07Model07ResponseFormat:
    """Adapter forwards response_format to chat.completions.create."""

    def test_04_unit_07_model_07_format_01_morphism_01_forwards_format_in_create_kwargs(
        self, fake_completions_client: FakeCompletionsClient
    ) -> None:
        # Arrange — a structured-output request carrying a json_schema
        # response_format built from a tool definition
        adapter = OpenAICompatAdapter(
            client=fake_completions_client,  # type: ignore[arg-type]
            model_name="gpt-4o",
            provider="fireworks",
        )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "read",
                "schema": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
        request = ModelRequest(
            messages=(ProviderMessage(role="user", content="hello"),),
            response_format=response_format,
        )

        # Act
        adapter.complete(request)

        # Assert — the format constraint reaches the create call verbatim
        assert fake_completions_client.captured_kwargs["response_format"] == (
            response_format
        )

    def test_04_unit_07_model_07_format_02_boundary_01_omits_format_when_request_has_none(
        self, fake_completions_client: FakeCompletionsClient
    ) -> None:
        # Arrange — a request without a response_format
        adapter = OpenAICompatAdapter(
            client=fake_completions_client,  # type: ignore[arg-type]
            model_name="gpt-4o",
            provider="fireworks",
        )
        request = ModelRequest(
            messages=(ProviderMessage(role="user", content="hello"),),
        )

        # Act
        adapter.complete(request)

        # Assert — no null format leaks into the create call
        assert "response_format" not in fake_completions_client.captured_kwargs


# ---------------------------------------------------------------------------
# Session affinity header forwarding tests
# ---------------------------------------------------------------------------


class Test04Unit08Model08SessionAffinity:
    """Adapter forwards extra_headers to chat.completions.create."""

    def test_04_unit_08_model_08_affinity_01_forwards_headers_in_create_kwargs(
        self, fake_completions_client: FakeCompletionsClient
    ) -> None:
        # Arrange — a request carrying a session affinity header
        adapter = OpenAICompatAdapter(
            client=fake_completions_client,  # type: ignore[arg-type]
            model_name="gpt-4o",
            provider="fireworks",
        )
        extra_headers = {"x-session-affinity": "affinity-key-123"}
        request = ModelRequest(
            messages=(ProviderMessage(role="user", content="hello"),),
            extra_headers=extra_headers,
        )

        # Act
        adapter.complete(request)

        # Assert — the headers reach the create call verbatim
        assert fake_completions_client.captured_kwargs["extra_headers"] == extra_headers

    def test_04_unit_08_model_08_affinity_02_omits_headers_when_request_has_none(
        self, fake_completions_client: FakeCompletionsClient
    ) -> None:
        # Arrange — a request without extra headers
        adapter = OpenAICompatAdapter(
            client=fake_completions_client,  # type: ignore[arg-type]
            model_name="gpt-4o",
            provider="fireworks",
        )
        request = ModelRequest(
            messages=(ProviderMessage(role="user", content="hello"),),
        )

        # Act
        adapter.complete(request)

        # Assert — no null or empty headers leak into the create call
        assert "extra_headers" not in fake_completions_client.captured_kwargs
