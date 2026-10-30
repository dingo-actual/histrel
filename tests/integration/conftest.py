#!/usr/bin/env python3
"""Shared fixtures for the integration tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from openai import OpenAI

from histrel.core.config import AgentConfig
from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
)
from histrel.model.types import ModelRequest, ModelResponse
from tests.integration.environment import (
    FIREWORKS_BASE_URL,
    FIREWORKS_MODEL,
    LIVE_REQUEST_TIMEOUT_SECONDS,
    RecordingAdapter,
    env,
    server_listening,
)
from tests.integration.helpers import (
    PatchedRuntime,
    build_adapter,
    build_patched_runtime,
    make_config,
)
from tests.integration.repo import (
    init_calculator_repo,
)
from tests.integration.scripting import (
    scripted_complete,
)


@pytest.fixture
def patched_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Callable[..., PatchedRuntime]:
    """Return a factory building a runtime over the calculator repo.

    The factory creates the fixture repository, builds the config and
    adapter, initializes the runtime, and attaches the adapter. When
    ``complete`` is given it is patched in as the adapter's complete
    method; otherwise the test patches it once its closure is ready.
    """

    def factory(
        config: AgentConfig | None = None,
        adapter: OpenAICompatAdapter | None = None,
        complete: Callable[[ModelRequest], ModelResponse] | None = None,
        *,
        api_key_env: str | None = "OPENAI_API_KEY",
    ) -> PatchedRuntime:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        init_calculator_repo(workspace)
        # The harness runs the agent with the workspace as CWD; the tools that
        # resolve paths or spawn commands depend on it.
        monkeypatch.chdir(workspace)
        if api_key_env is not None:
            monkeypatch.setenv(api_key_env, "test-key")

        resolved = make_config() if config is None else config
        model_adapter = adapter or build_adapter(resolved)
        if complete is not None:
            monkeypatch.setattr(model_adapter, "complete", complete)
        return build_patched_runtime(
            resolved, workspace, tmp_path / "output", model_adapter
        )

    return factory


@pytest.fixture
def scripted_session(
    patched_runtime: Callable[..., PatchedRuntime],
) -> Callable[..., tuple[PatchedRuntime, list[ModelRequest]]]:
    """Return a factory building a runtime with a scripted adapter.

    The scripted responses are returned in call order, repeating the
    final response for any calls beyond the script. The recorded model
    requests come back alongside the session.
    """

    def factory(
        *responses: ModelResponse,
        config: AgentConfig | None = None,
    ) -> tuple[PatchedRuntime, list[ModelRequest]]:
        complete, recorded = scripted_complete(list(responses))
        return patched_runtime(config, complete=complete), recorded

    return factory


def _require_fireworks() -> str:
    """Return the Fireworks API key, skipping when unavailable."""
    api_key = env("FIREWORKS_API_KEY")
    if not api_key:
        pytest.skip("FIREWORKS_API_KEY not set")
    if not server_listening(FIREWORKS_BASE_URL):
        pytest.skip("Fireworks API unreachable")
    return api_key


def _fireworks_client() -> OpenAI:
    """Build a live Fireworks client."""
    return OpenAI(base_url=FIREWORKS_BASE_URL, api_key=_require_fireworks())


@pytest.fixture
def fireworks_adapter() -> OpenAICompatAdapter:
    """Return a live Fireworks adapter, skipping when unavailable."""
    return OpenAICompatAdapter(
        client=_fireworks_client(),
        model_name=FIREWORKS_MODEL,
        provider="fireworks",
        request_timeout=LIVE_REQUEST_TIMEOUT_SECONDS,
    )


@pytest.fixture
def fireworks_recording_adapter() -> RecordingAdapter:
    """Return a live Fireworks adapter that records its requests."""
    return RecordingAdapter(
        client=_fireworks_client(),
        model_name=FIREWORKS_MODEL,
        provider="fireworks",
        request_timeout=LIVE_REQUEST_TIMEOUT_SECONDS,
    )
