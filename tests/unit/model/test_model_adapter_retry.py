#!/usr/bin/env python3
"""Unit tests for model adapter retry on transient provider failures.

A single rate limit, server error, timeout, or connection failure must
not end a whole benchmark run: the adapter retries with exponential
backoff before the error reaches the runtime's error handling.
"""

from __future__ import annotations

import time

import httpx2
import pytest
from openai import (
    APITimeoutError,
    BadRequestError,
    RateLimitError,
)

from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
)
from histrel.model.types import ModelRequest, ProviderMessage
from tests.unit.model.helpers import (
    FakeCompletionsClient,
    FlakyCompletionsClient,
    make_api_status_error,
)


@pytest.fixture(autouse=True)
def max_jitter(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the jitter to its upper bound so backoff delays are predictable."""
    monkeypatch.setattr("histrel.model.retry.random.uniform", _jitter_high)


def _jitter_high(low: float, high: float) -> float:
    """Return the upper jitter bound."""
    return high


def _jitter_low(low: float, high: float) -> float:
    """Return the lower jitter bound."""
    return low


# ---------------------------------------------------------------------------
# Transient provider failure retry tests
# ---------------------------------------------------------------------------


class Test04Unit09Model09Retry:
    """Adapter retries transient provider failures with backoff."""

    def _adapter(self, client: object) -> OpenAICompatAdapter:
        """Build a fireworks adapter around a client double."""
        return OpenAICompatAdapter(
            client=client,  # type: ignore[arg-type]
            model_name="gpt-4o",
            provider="fireworks",
        )

    @pytest.mark.parametrize(
        "status_code",
        [429, 500, 503],
        ids=["rate_limit", "internal_server_error", "service_unavailable"],
    )
    def test_04_unit_09_model_09_retry_01_morphism_01_retries_transient_status(
        self,
        status_code: int,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — the provider fails once, then serves the response
        client = FlakyCompletionsClient(make_api_status_error(status_code))
        adapter = self._adapter(client)
        delays: list[float] = []
        monkeypatch.setattr(time, "sleep", delays.append)

        # Act
        response = adapter.complete(
            ModelRequest(messages=(ProviderMessage(role="user", content="hello"),))
        )

        # Assert — one retry after one backoff delay, then success
        assert client.attempts == 2
        assert delays == [1.0]
        assert response.finish_reason == "stop"

    def test_04_unit_09_model_09_retry_02_morphism_02_retries_read_timeout(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — a read timeout is transient
        request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
        client = FlakyCompletionsClient(APITimeoutError(request=request))
        adapter = self._adapter(client)
        delays: list[float] = []
        monkeypatch.setattr(time, "sleep", delays.append)

        # Act
        response = adapter.complete(
            ModelRequest(messages=(ProviderMessage(role="user", content="hello"),))
        )

        # Assert
        assert client.attempts == 2
        assert delays == [1.0]
        assert response.finish_reason == "stop"

    def test_04_unit_09_model_09_retry_03_morphism_03_backoff_doubles_per_attempt(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — two consecutive transient failures
        client = FlakyCompletionsClient(
            make_api_status_error(429), make_api_status_error(429)
        )
        adapter = self._adapter(client)
        delays: list[float] = []
        monkeypatch.setattr(time, "sleep", delays.append)

        # Act
        adapter.complete(
            ModelRequest(messages=(ProviderMessage(role="user", content="hello"),))
        )

        # Assert — the delay doubles between attempts
        assert client.attempts == 3
        assert delays == [1.0, 2.0]

    def test_04_unit_09_model_09_retry_04_morphism_04_jitter_offsets_each_delay(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — jitter pinned to its floor, two consecutive failures
        client = FlakyCompletionsClient(
            make_api_status_error(429), make_api_status_error(429)
        )
        adapter = self._adapter(client)
        delays: list[float] = []
        monkeypatch.setattr(time, "sleep", delays.append)
        monkeypatch.setattr("histrel.model.retry.random.uniform", _jitter_low)

        # Act
        adapter.complete(
            ModelRequest(messages=(ProviderMessage(role="user", content="hello"),))
        )

        # Assert — each delay keeps only its fixed half of the window
        assert client.attempts == 3
        assert delays == [0.5, 1.0]

    def test_04_unit_09_model_09_retry_04_morphism_05_sets_request_timeout(
        self,
        fake_completions_client: FakeCompletionsClient,
    ) -> None:
        # Act
        self._adapter(fake_completions_client).complete(
            ModelRequest(messages=(ProviderMessage(role="user", content="hello"),))
        )

        # Assert — an explicit per-request timeout bounds a stalled stream
        assert fake_completions_client.captured_kwargs["timeout"] > 0


class Test04Unit10Model10RetryLimits:
    """Adapter retry limits: what must not be retried, and the last error."""

    def _adapter(self, client: object) -> OpenAICompatAdapter:
        """Build a fireworks adapter around a client double."""
        return OpenAICompatAdapter(
            client=client,  # type: ignore[arg-type]
            model_name="gpt-4o",
            provider="fireworks",
        )

    def test_04_unit_10_model_10_retry_01_failure_01_does_not_retry_client_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — a 400 is the caller's fault, so a retry cannot help
        client = FlakyCompletionsClient(make_api_status_error(400))
        adapter = self._adapter(client)
        delays: list[float] = []
        monkeypatch.setattr(time, "sleep", delays.append)

        # Act & Assert
        with pytest.raises(BadRequestError):
            adapter.complete(
                ModelRequest(messages=(ProviderMessage(role="user", content="hello"),))
            )
        assert client.attempts == 1
        assert delays == []

    def test_04_unit_10_model_10_retry_02_failure_02_raises_after_max_attempts(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — every attempt fails transiently
        client = FlakyCompletionsClient(
            make_api_status_error(429),
            make_api_status_error(429),
            make_api_status_error(429),
            make_api_status_error(429),
        )
        adapter = self._adapter(client)
        delays: list[float] = []
        monkeypatch.setattr(time, "sleep", delays.append)

        # Act & Assert — the last error propagates to handle_model_error
        with pytest.raises(RateLimitError):
            adapter.complete(
                ModelRequest(messages=(ProviderMessage(role="user", content="hello"),))
            )
        assert client.attempts == 4
        assert delays == [1.0, 2.0, 4.0]
