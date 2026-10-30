#!/usr/bin/env python3
"""Shared fixtures for benchmark agent unit tests.

Important Classes:
    _NamedTool(Tool): Minimal tool identified only by its name.

Important Functions:
    make_tools (fixture): Factory building tuples of name-only tools.
    make_agent_config (fixture): Factory building AgentConfig instances.
    make_patched_adapter (fixture): Factory building adapters whose
        complete method is patched.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from openai import OpenAI

from histrel.core.config import AgentConfig
from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
)
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
)
from histrel.tools.tool import Tool, ToolContext, ToolExecution
from tests.unit.factories import build_agent_config


class _NamedTool(Tool):
    """Minimal Tool identified only by its name.

    Tool projection filters by ``tool.name`` and never executes the
    tools, so name-only instances are sufficient for projection tests.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.definition: dict[str, Any] = {
            "type": "function",
            "function": {"name": name, "parameters": {}},
        }

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        return ToolExecution(status="ok", output="")


@pytest.fixture
def make_tools() -> Callable[..., tuple[Tool, ...]]:
    """Return a factory building a tuple of name-only Tool instances."""

    def factory(*names: str) -> tuple[Tool, ...]:
        return tuple(_NamedTool(name) for name in names)

    return factory


@pytest.fixture
def make_agent_config() -> Callable[..., AgentConfig]:
    """Return a factory building baseline AgentConfig instances."""

    def factory(**overrides: Any) -> AgentConfig:
        return build_agent_config(**overrides)

    return factory


@pytest.fixture
def make_patched_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> Callable[[Callable[[ModelRequest], ModelResponse]], OpenAICompatAdapter]:
    """Return a factory building an adapter with a patched complete."""

    def factory(
        fake_complete: Callable[[ModelRequest], ModelResponse],
    ) -> OpenAICompatAdapter:
        adapter = OpenAICompatAdapter(
            client=OpenAI(api_key="test-key"),
            model_name="gpt-4o",
            provider="openai",
        )
        monkeypatch.setattr(adapter, "complete", fake_complete)
        return adapter

    return factory
