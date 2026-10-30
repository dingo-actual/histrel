#!/usr/bin/env python3
"""Unit tests for the malformed-JSON repair LLM routine.

Tests the request shape and result of fix_json_arguments: the
schema-constrained response format, the rendered user message, the
provider-default reasoning effort, the config sampling settings, the
stripped response text, and error propagation.

    Test Classes:
        Test04Unit41FixJson01Request: fix_json_arguments request shape.
        Test04Unit41FixJson02Response: fix_json_arguments result and
            error propagation.

    Test Subtypes:
        Morphism, Contract, Failure Mode
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from histrel.core.config import AgentConfig
from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
)
from histrel.model.routines import fix_json_arguments
from histrel.model.types import ModelRequest, ModelResponse
from tests.unit.model.helpers import READ_TOOL_DEFINITION


class Test04Unit41FixJson01Request:
    """fix_json_arguments builds a schema-constrained routine request.

    The routine sends a system message instructing the model to infer
    the intended JSON, a user message rendering the malformed JSON and
    the tool's JSON schema, and a response_format built from the tool
    definition (json_schema with name and parameters schema, no strict
    flag) so Fireworks enforces the schema during generation.
    """

    def test_04_unit_41_fix_json_01_request_01_morphism_01_builds_json_schema_response_format(
        self,
        make_json_fixer: Callable[..., tuple[Callable[[str], str], list[ModelRequest]]],
    ) -> None:
        # Arrange — malformed JSON that no local repair can fix
        fix_json, requests = make_json_fixer()
        malformed = '{"path": "main.py"'

        # Act
        result = fix_json(malformed)

        # Assert — the schema constraint is built from the tool
        # definition: name from function name, schema from parameters.
        assert len(requests) == 1
        assert requests[0].response_format == {
            "type": "json_schema",
            "json_schema": {
                "name": "read",
                "schema": READ_TOOL_DEFINITION["function"]["parameters"],
            },
        }
        assert result == '{"path": "main.py"}'

    def test_04_unit_41_fix_json_01_request_02_morphism_02_user_message_renders_json_and_schema(
        self,
        make_json_fixer: Callable[..., tuple[Callable[[str], str], list[ModelRequest]]],
    ) -> None:
        # Arrange
        fix_json, requests = make_json_fixer()
        malformed = '{"path": "main.py"'

        # Act
        fix_json(malformed)

        # Assert — the rendered user message carries the malformed JSON
        # and the tool's JSON schema, with a system message above it.
        user_message = requests[0].messages[1].content
        assert malformed in user_message
        assert '"type": "object"' in user_message
        assert '"path"' in user_message
        assert requests[0].messages[0].role == "system"

    def test_04_unit_41_fix_json_01_request_03_contract_01_uses_provider_default_reasoning_effort(
        self,
        make_json_fixer: Callable[..., tuple[Callable[[str], str], list[ModelRequest]]],
    ) -> None:
        # Arrange — json_schema response format disables reasoning, so
        # the routine must not override the provider default effort.
        fix_json, requests = make_json_fixer()

        # Act
        fix_json('{"path": "main.py"')

        # Assert — the setting is omitted from the request entirely.
        assert requests[0].reasoning_effort is None

    def test_04_unit_41_fix_json_01_request_04_contract_02_uses_config_sampling_settings(
        self,
        make_json_fixer: Callable[..., tuple[Callable[[str], str], list[ModelRequest]]],
    ) -> None:
        # Arrange
        fix_json, requests = make_json_fixer()

        # Act
        fix_json('{"path": "main.py"')

        # Assert — sampling comes from the agent config like every
        # other routine call.
        assert requests[0].temperature == 0.0
        assert requests[0].max_tokens_output == 16384


class Test04Unit41FixJson02Response:
    """fix_json_arguments returns the response text and propagates errors."""

    def test_04_unit_41_fix_json_02_response_01_morphism_01_returns_stripped_response_text(
        self,
        make_json_fixer: Callable[..., tuple[Callable[[str], str], list[ModelRequest]]],
    ) -> None:
        # Arrange — a schema-constrained response wrapped in whitespace
        fix_json, _ = make_json_fixer(response_text='  {"path": "main.py"}  ')

        # Act
        result = fix_json('{"path": "main.py"')

        # Assert — the routine output is the bare JSON string.
        assert result == '{"path": "main.py"}'

    def test_04_unit_41_fix_json_02_response_02_failure_01_adapter_error_propagates(
        self,
        make_agent_config: Callable[..., AgentConfig],
        make_patched_adapter: Callable[
            [Callable[[ModelRequest], ModelResponse]], OpenAICompatAdapter
        ],
    ) -> None:
        # Arrange — the routine is a thin call: failures propagate to
        # the fallback caller (tools.json), which keeps the prior error.
        def failing_complete(request: ModelRequest) -> ModelResponse:
            raise RuntimeError("adapter offline")

        adapter = make_patched_adapter(failing_complete)
        config = make_agent_config()
        recorded: list[ModelResponse] = []

        # Act / Assert
        with pytest.raises(RuntimeError, match="adapter offline"):
            fix_json_arguments(
                '{"path": "main.py"',
                READ_TOOL_DEFINITION,
                adapter,
                config,
                recorded.append,
            )
