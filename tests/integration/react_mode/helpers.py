#!/usr/bin/env python3
"""Shared helpers for the react-mode integration tests.

Important Functions:
    make_react_config(config) -> AgentConfig: Switch a config to react
        mode with a shared system prompt.

Important Variables:
    REACT_SYSTEM_PROMPT (str): Shared initialize/step system prompt, so
        the react prompt selection is observable in requests.
    SESSION_TOOL_NAMES (set[str]): Every tool offered to the model.
"""

from __future__ import annotations

from histrel.core.config import AgentConfig, SystemPrompts

# Shared initialize/step prompt makes the react prompt selection
# observable in recorded model requests.
REACT_SYSTEM_PROMPT = "REACT-SYSTEM."

# The static tool set: every tool is offered on every model call.
SESSION_TOOL_NAMES = {
    "read",
    "edit",
    "glob",
    "search",
    "list_dir",
    "bash",
}


def make_react_config(config: AgentConfig) -> AgentConfig:
    """Switch a config to react mode with a shared system prompt."""
    return config.model_copy(
        update={
            "mode": "react",
            "agent_system_prompt": SystemPrompts(
                initialize=REACT_SYSTEM_PROMPT,
                step=REACT_SYSTEM_PROMPT,
            ),
        }
    )
