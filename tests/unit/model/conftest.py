#!/usr/bin/env python3
"""Shared fixtures for the model-layer unit tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from histrel.core.config import AgentConfig
from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
)
from histrel.model.routines import (
    fix_json_arguments,
)
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
    Usage,
)
from tests.unit.model.helpers import (
    READ_TOOL_DEFINITION,
    FakeCompletionsClient,
)


@pytest.fixture
def fake_completions_client() -> FakeCompletionsClient:
    """Provide a recording client double for adapter streaming tests."""
    return FakeCompletionsClient()


@pytest.fixture
def make_json_fixer(
    make_agent_config: Callable[..., AgentConfig],
    make_patched_adapter: Callable[
        [Callable[[ModelRequest], ModelResponse]], OpenAICompatAdapter
    ],
) -> Callable[..., tuple[Callable[[str], str], list[ModelRequest]]]:
    """Return a factory for fix_json_arguments with a captured adapter.

    The routine is closed over a read-tool definition, the patched
    adapter, and a baseline config; each call returns the fixer and the
    captured routine requests.
    """

    def factory(
        response_text: str = '{"path": "main.py"}',
    ) -> tuple[Callable[[str], str], list[ModelRequest]]:
        requests: list[ModelRequest] = []

        def fake_complete(request: ModelRequest) -> ModelResponse:
            requests.append(request)
            return ModelResponse(
                text_parts=(response_text,),
                usage=Usage(input=25, output=9, total=34),
                provider_request_id="req-fix-json",
                finish_reason="stop",
            )

        adapter = make_patched_adapter(fake_complete)
        config = make_agent_config()
        recorded: list[ModelResponse] = []

        def fix_json(malformed_json: str) -> str:
            return fix_json_arguments(
                malformed_json,
                READ_TOOL_DEFINITION,
                adapter,
                config,
                recorded.append,
            )

        return fix_json, requests

    return factory
