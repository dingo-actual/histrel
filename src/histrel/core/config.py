#!/usr/bin/env python3
"""
Configuration model for the benchmark agent.

Defines the Pydantic v2 AgentConfig and its sub-models, loaded from a flat
TOML file. The model uses extra="forbid" to reject unknown fields.

    Important Classes:
        Budgets: Wall-clock, turn, and tool-call budgets.
        Compaction: Context compaction thresholds.
        Tests: Test runner registry configuration.
        SystemPrompts: Initialize and step system prompts.
        AgentConfig: Top-level agent configuration.

    Important Functions:
        load_config(path: str | Path) -> AgentConfig: Load config from TOML.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class Budgets(BaseModel):
    """Budget limits for the agent session.

    A value of 0 means unlimited.

    Instance Attributes:
        max_wall_seconds (int): Maximum wall-clock time in seconds.
        max_turns (int): Maximum number of model turns.
        max_tool_calls (int): Maximum number of tool calls.
    """

    max_wall_seconds: int = 0
    max_turns: int = 0
    max_tool_calls: int = 0


class Compaction(BaseModel):
    """Context compaction thresholds.

    When the context reaches compact_at_fraction of the context limit,
    compaction removes oldest messages until the context is at
    target_fraction of the limit.

    Instance Attributes:
        compact_at_fraction (float): Fraction of context limit that triggers compaction.
        target_fraction (float): Fraction of context limit to compact down to.
        compact_retain_tail (int): Number of recent messages to retain verbatim during compaction.
    """

    compact_at_fraction: float = 0.75
    target_fraction: float = 0.35
    compact_retain_tail: int = 10


class SystemPrompts(BaseModel):
    """System prompts for the agent loop.

    Instance Attributes:
        initialize (str): System message for the first loop iteration.
        step (str): System message for all subsequent loop iterations.
    """

    initialize: str
    step: str


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

Mode = Literal["stateless", "react"]
Provider = Literal["openai", "fireworks"]
ReasoningEffort = Literal["none", "low", "medium", "high", "max"] | None


class AgentConfig(BaseModel):
    """Top-level benchmark agent configuration.

    Uses extra="forbid" to reject unknown fields.

    Instance Attributes:
        schema_version (int): Config schema version.
        session_id (str): Session identifier.
        task_id (str): SWE-bench instance identifier.
        mode (str): Agent loop mode. "stateless" replays a per-iteration
            stateless user message; "react" sends the real tool
            definitions and replays conversation history.
        agent_system_prompt (SystemPrompts): Initialize and step system
            prompts for the agent loop.
        model_provider (str): Model provider (openai or fireworks).
        model_name (str): Model name.
        model_temperature (float): Sampling temperature.
        model_top_p (float): Top-p sampling parameter.
        model_reasoning_effort (str | None): Reasoning effort level.
        model_context_limit_tokens (int): Context window size in tokens.
        model_max_tokens_output (int): Maximum output tokens.
        model_api_key_env (str): Environment variable for API key.
        model_base_url (str | None): Custom API base URL.
        duplicate_call_window (int): LLM calls within which repeat
            read, list_dir, search, and glob calls are duplicates.
            0 disables those rules.
        budgets (Budgets): Budget limits.
        compaction (Compaction): Compaction thresholds.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    session_id: str
    task_id: str
    mode: Mode = "stateless"
    agent_system_prompt: SystemPrompts

    # Model
    model_provider: Provider
    model_name: str
    model_temperature: float = 1.0
    model_top_p: float = 1.0
    model_reasoning_effort: ReasoningEffort = None
    model_context_limit_tokens: int = 128000
    model_max_tokens_output: int = 16384
    model_api_key_env: str = "OPENAI_API_KEY"
    model_base_url: str | None = None

    # Duplicate call prevention
    duplicate_call_window: int = Field(default=5, ge=0)

    # Sub-models with defaults
    budgets: Budgets = Field(default_factory=Budgets)
    compaction: Compaction = Field(default_factory=Compaction)


# ---------------------------------------------------------------------------
# TOML loading
# ---------------------------------------------------------------------------


def load_config(path: str | Path) -> AgentConfig:
    """Load an AgentConfig from a flat TOML file.

    Top-level keys map to AgentConfig fields. Nested tables ([budgets],
    [compaction]) map to sub-models.

    Args:
        path (str | Path): Path to the TOML file.

    Returns:
        (AgentConfig) Parsed and validated configuration.

    Raises:
        FileNotFoundError: If the file does not exist.
        pydantic.ValidationError: If the TOML content fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("rb") as fp:
        data = tomllib.load(fp)

    return AgentConfig.model_validate(data)
