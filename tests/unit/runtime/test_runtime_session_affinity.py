#!/usr/bin/env python3
"""Unit tests for the session affinity header."""

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


def _capture_headers(
    monkeypatch: pytest.MonkeyPatch,
    runtime: Runtime,
) -> list[dict[str, str] | None]:
    """Patch the runtime adapter to capture request extra headers."""
    captured: list[dict[str, str] | None] = []

    def fake_complete(request: ModelRequest) -> ModelResponse:
        captured.append(request.extra_headers)
        return make_fake_response(text="thinking...")

    monkeypatch.setattr(runtime.model_adapter, "complete", fake_complete)
    return captured


class Test04Unit12Runtime12SessionAffinity:
    """run_agent sends a stable x-session-affinity header on every call."""

    def test_04_unit_12_runtime_12_affinity_01_requests_carry_stable_uuid_header(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — the default provider session; max_turns=1 gives two
        # model calls to observe the affinity header across the session.
        runtime = make_runtime(
            max_turns=1,
        )
        captured_headers = _capture_headers(monkeypatch, runtime)

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert — every request carries the same x-session-affinity
        # UUID4 value under the extra_headers field
        assert outcome.status == SessionStatus.BUDGET_EXHAUSTED
        assert "max_turns" in outcome.reason
        assert len(captured_headers) == 2
        assert captured_headers[0] == captured_headers[1]
        for headers in captured_headers:
            assert headers is not None
            assert set(headers) == {"x-session-affinity"}
            parsed = uuid.UUID(headers["x-session-affinity"])
            assert parsed.version == 4
