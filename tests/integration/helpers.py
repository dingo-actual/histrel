#!/usr/bin/env python3
"""Core helpers for the integration tests.

Important Classes:
    PatchedRuntime: Runtime over the calculator fixture repo together
        with its model adapter and output directory.

Important Functions:
    make_config(...) -> AgentConfig: Build a minimal AgentConfig.
    with_distinct_prompts(config) -> AgentConfig: Distinct iteration
        system prompts, so prompt selection is observable.
    build_adapter(config) -> OpenAICompatAdapter: Build an adapter over a
        dummy client.
    build_patched_runtime(config, workspace, output, adapter) ->
        PatchedRuntime: Initialize a runtime and attach the adapter.

Important Variables:
    USER_MSG (str): User request message used across integration tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from histrel.core.config import (
    AgentConfig,
    Budgets,
    SystemPrompts,
)
from histrel.core.runtime import Runtime
from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
)

# User request message used across integration tests.
USER_MSG = "Fix the failing test."


def make_config(
    max_turns: int = 25,
    max_tool_calls: int = 0,
    model_provider: str = "openai",
    model_name: str = "gpt-4o",
    api_key_env: str = "OPENAI_API_KEY",
) -> AgentConfig:
    """Build a minimal AgentConfig for integration tests.

    The turn budget is bounded by default so scripted sessions that stop
    without a repository change cannot loop forever.
    """
    prompt = "You are a coding agent. Fix the bug in the repository so all tests pass."
    config = AgentConfig(
        schema_version=1,
        session_id="integration-test-session",
        task_id="fixture-calculator-bug",
        agent_system_prompt=SystemPrompts(initialize=prompt, step=prompt),
        model_provider=model_provider,  # type: ignore[arg-type]
        model_name=model_name,
        model_temperature=0.0,
        model_top_p=1.0,
        model_reasoning_effort=None,
        model_context_limit_tokens=128000,
        model_max_tokens_output=16384,
        model_api_key_env=api_key_env,
        model_base_url=None,
        budgets=Budgets(
            max_wall_seconds=0,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
        ),
    )
    return config


def with_distinct_prompts(config: AgentConfig) -> AgentConfig:
    """Give a config distinct initialize/step system prompts.

    Makes the first-iteration vs later-iteration prompt selection
    observable in recorded model requests.
    """
    return config.model_copy(
        update={
            "agent_system_prompt": SystemPrompts(
                initialize="INITIALIZE-PROMPT",
                step="STEP-PROMPT",
            )
        }
    )


def build_adapter(config: AgentConfig) -> OpenAICompatAdapter:
    """Build an adapter over a dummy client (never called by these tests)."""
    return OpenAICompatAdapter(
        client=OpenAI(api_key="test-key"),
        model_name=config.model_name,
        provider=config.model_provider,
    )


def build_patched_runtime(
    config: AgentConfig,
    workspace: Path,
    output: Path,
    adapter: OpenAICompatAdapter,
) -> PatchedRuntime:
    """Initialize a runtime over the fixture repo and attach the adapter."""
    runtime = Runtime(config=config, workspace_root=workspace, output_dir=output)
    runtime.initialize(USER_MSG)
    runtime.model_adapter = adapter
    return PatchedRuntime(
        runtime=runtime, adapter=adapter, workspace=workspace, output=output
    )


@dataclass(frozen=True)
class PatchedRuntime:
    """A runtime over the calculator fixture repo with a model adapter.

    Instance Attributes:
        runtime (Runtime): Initialized runtime under test.
        adapter (OpenAICompatAdapter): Adapter attached to the runtime.
        workspace (Path): Fixture repository root.
        output (Path): Session output directory.
    """

    runtime: Runtime
    adapter: OpenAICompatAdapter
    workspace: Path
    output: Path
