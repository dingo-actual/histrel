#!/usr/bin/env python3
"""Shared fixtures for the context unit tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from histrel.context.compactor import CompactedState
from histrel.core.config import AgentConfig
from histrel.core.ids import SessionId
from histrel.model.adapter_openai_compat import OpenAICompatAdapter
from histrel.model.messages import (
    Message,
    MessageFactory,
    MessageRole,
)
from histrel.model.types import ModelRequest, ModelResponse, Usage
from tests.unit.context.helpers import (
    ArtifactHarness,
    CompactorHarness,
    make_message,
)
from tests.unit.factories import build_agent_config

PatchedAdapterFactory = Callable[
    [Callable[[ModelRequest], ModelResponse]], OpenAICompatAdapter
]


@pytest.fixture
def message_factory() -> MessageFactory:
    """Return a message factory bound to the shared test session id."""
    return MessageFactory(SessionId("test-session"))


@pytest.fixture
def make_messages(
    message_factory: MessageFactory,
) -> Callable[..., list[Message]]:
    """Return a factory building messages from (role, text) pairs."""

    def factory(*specs: tuple[MessageRole, str]) -> list[Message]:
        return [make_message(message_factory, role, text) for role, text in specs]

    return factory


@pytest.fixture
def make_compactor(
    make_patched_adapter: PatchedAdapterFactory,
) -> Callable[..., CompactorHarness]:
    """Return a factory building a CompactorHarness."""

    def factory(
        *,
        summary: str = "Summary.",
        usage: Usage | None = None,
        config: AgentConfig | None = None,
        on_compaction: Callable[[CompactedState], None] | None = None,
    ) -> CompactorHarness:
        return CompactorHarness(
            build_agent_config() if config is None else config,
            make_patched_adapter,
            summary=summary,
            usage=usage,
            on_compaction=on_compaction,
        )

    return factory


@pytest.fixture
def artifact_harness(tmp_path: Path) -> ArtifactHarness:
    """Return an artifact writer bound to a per-test directory."""
    return ArtifactHarness(base_dir=tmp_path / "session-1" / "artifacts")
