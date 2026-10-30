#!/usr/bin/env python3
"""Live LLM integration tests against Fireworks AI.

These tests make real API calls using the model
"accounts/fireworks/models/deepseek-v4p1-flash-1017". They skip unless --llm (or
--all) is provided and FIREWORKS_API_KEY is set.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from histrel.core.journal import load_journal
from histrel.core.runtime import run_agent
from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
)
from histrel.model.messages import (
    TextPart,
    ToolRequestPart,
)
from histrel.model.types import (
    ModelRequest,
    ProviderMessage,
)
from histrel.terminal import SessionStatus
from tests.integration.environment import (
    FIREWORKS_MODEL,
)
from tests.integration.helpers import (
    USER_MSG,
    PatchedRuntime,
    make_config,
)


@pytest.mark.requires_live_llm
class Test01Integration04LiveLlm01Fireworks:
    """Live LLM integration tests using Fireworks AI."""

    def test_01_integration_04_live_llm_01_fireworks_01_adapter_returns_response(
        self,
        fireworks_adapter: OpenAICompatAdapter,
    ) -> None:
        # Arrange
        request = ModelRequest(
            messages=(
                ProviderMessage(
                    role="user", content="What is 2 + 3? Answer with just the number."
                ),
            ),
            temperature=0.0,
            top_p=1.0,
            max_tokens_output=100,
        )

        # Act — make a real model call
        response = fireworks_adapter.complete(request)

        # Assert — response has text and usage
        assert len(response.text_parts) > 0
        text = " ".join(response.text_parts)
        assert "5" in text
        assert response.usage.input > 0
        assert response.usage.output > 0
        assert response.usage.total > 0

    def test_01_integration_04_live_llm_01_fireworks_02_agent_loop_e2e(
        self,
        fireworks_adapter: OpenAICompatAdapter,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        # Arrange
        config = make_config(
            max_turns=5,
            model_provider="fireworks",
            model_name=FIREWORKS_MODEL,
            api_key_env="FIREWORKS_API_KEY",
        )
        session = patched_runtime(
            config,
            adapter=fireworks_adapter,
            api_key_env=None,
        )
        runtime = session.runtime

        # Act — run the full agent loop with real LLM
        outcome = run_agent(USER_MSG, runtime)

        # Assert — session completed without infrastructure failure
        assert outcome.status in (
            SessionStatus.SUCCEEDED,
            SessionStatus.BUDGET_EXHAUSTED,
        )

        # Journal has messages from real LLM
        messages = list(load_journal(runtime.journal._path))
        assert len(messages) >= 2  # task + at least one assistant

        # Artifacts were written
        artifacts_dir = session.output / config.session_id / "artifacts"
        assert (artifacts_dir / "messages.jsonl").exists()
        assert (artifacts_dir / "outcome.json").exists()

        # Token ledger recorded usage from real calls
        assert runtime.token_ledger.total_input > 0
        assert runtime.token_ledger.total_output > 0

    def test_01_integration_04_live_llm_01_fireworks_03_tool_call_e2e(
        self,
        fireworks_adapter: OpenAICompatAdapter,
        patched_runtime: Callable[..., PatchedRuntime],
    ) -> None:
        # Arrange
        config = make_config(
            max_turns=5,
            model_provider="fireworks",
            model_name=FIREWORKS_MODEL,
            api_key_env="FIREWORKS_API_KEY",
        )
        session = patched_runtime(
            config,
            adapter=fireworks_adapter,
            api_key_env=None,
        )
        runtime = session.runtime

        # Act
        run_agent(USER_MSG, runtime)

        # Assert — at least one assistant message was committed
        messages = list(load_journal(runtime.journal._path))
        assistant_msgs = [m for m in messages if m.role == "assistant"]
        assert len(assistant_msgs) >= 1

        # At least one assistant message has content (text or tool calls)
        content_parts = [
            p
            for m in assistant_msgs
            for p in m.content
            if isinstance(p, (TextPart, ToolRequestPart))
        ]
        assert len(content_parts) >= 1
