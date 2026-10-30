#!/usr/bin/env python3
"""Unit tests for the per-session prompt cache key."""

from __future__ import annotations

import uuid
from collections.abc import Callable

import pytest

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
from tests.unit.runtime.helpers import (
    USER_MSG,
    make_fake_response,
)


def _capture_keys(
    monkeypatch: pytest.MonkeyPatch,
    runtime: Runtime,
) -> list[str | None]:
    """Patch the runtime adapter to capture prompt cache keys."""
    captured: list[str | None] = []

    def fake_complete(request: ModelRequest) -> ModelResponse:
        captured.append(request.prompt_cache_key)
        return make_fake_response(text="thinking...")

    monkeypatch.setattr(runtime.model_adapter, "complete", fake_complete)
    return captured


class Test04Unit11Runtime11PromptCacheKey:
    """run_agent sets a per-session prompt cache key for fireworks."""

    def test_04_unit_11_runtime_11_cache_01_fireworks_requests_carry_stable_uuid(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — a fireworks provider session; max_turns=1 gives two
        # model calls to observe the cache key across the session.
        runtime = make_runtime(max_turns=1, provider="fireworks")
        captured_keys = _capture_keys(monkeypatch, runtime)

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert — every request carries the same UUID4 string key
        assert outcome.status == SessionStatus.BUDGET_EXHAUSTED
        assert "max_turns" in outcome.reason
        assert len(captured_keys) == 2
        assert captured_keys[0] == captured_keys[1]
        for key in captured_keys:
            parsed = uuid.UUID(str(key))
            assert parsed.version == 4

    def test_04_unit_11_runtime_11_cache_02_openai_requests_carry_no_key(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — the openai provider must not set a cache key
        runtime = make_runtime(
            max_turns=1,
        )
        captured_keys = _capture_keys(monkeypatch, runtime)

        # Act
        run_agent(USER_MSG, runtime)

        # Assert — the cache key is only set for the fireworks provider
        assert captured_keys == [None, None]
