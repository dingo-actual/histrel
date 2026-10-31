#!/usr/bin/env python3
"""
Shared factories for benchmark agent unit tests.

Centralizes the baseline AgentConfig construction, the fixture git
repository, and the Runtime construction that were previously duplicated
across the unit modules. Tests use these directly from module-level
helpers and through the `make_agent_config` / `make_runtime` fixtures in
`conftest.py`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from dulwich.porcelain import add, commit
from dulwich.repo import Repo

from histrel.core.config import (
    AgentConfig,
    Budgets,
    SystemPrompts,
)
from histrel.core.runtime import Runtime

# The user message every runtime test initializes its session with.
USER_MSG = "Fix the failing test."

# Default workspace content committed by `init_fixture_repo`.
MAIN_CONTENT = "print('hello')\n"

_AUTHOR = b"Test <test@test.com>"


# Baseline AgentConfig fields shared by every unit test configuration.
_BASELINE_CONFIG_KWARGS: dict[str, Any] = {
    "schema_version": 1,
    "session_id": "test-session",
    "task_id": "test-task",
    "agent_system_prompt": SystemPrompts(
        initialize="You are a coding agent.",
        step="You are a coding agent.",
    ),
    "model_name": "gpt-4o",
    "model_temperature": 0.0,
    "model_top_p": 1.0,
    "model_reasoning_effort": None,
    "model_api_key_env": "OPENAI_API_KEY",
    "model_base_url": None,
}


def build_agent_config(
    *,
    provider: str | None = None,
    max_wall_seconds: int = 0,
    max_turns: int = 0,
    max_tool_calls: int = 0,
    token_budget: int = 0,
    token_budget_buffer: int = 0,
    context_limit: int = 128000,
    max_output: int = 16384,
    duplicate_call_window: int | None = None,
    mode: str | None = None,
    **overrides: Any,
) -> AgentConfig:
    """Build a baseline AgentConfig for unit tests.

    Supplies a baseline configuration and accepts the union of overrides
    used across the unit suite, so no module rebuilds the baseline by
    hand.

    Args:
        provider (str | None): Model provider; defaults to OpenAI.
        max_wall_seconds (int): Wall-clock budget.
        max_turns (int): Turn budget.
        max_tool_calls (int): Tool-call budget.
        token_budget (int): Total token budget.
        token_budget_buffer (int): Tokens reserved for finalization.
        context_limit (int): Model context limit in tokens.
        max_output (int): Model maximum output tokens.
        duplicate_call_window (int | None): Duplicate-call window;
            omitted from the config when None.
        mode (str | None): Agent loop mode; omitted when None.
        **overrides (Any): Extra AgentConfig fields, which win over every
            value above.

    Returns:
        (AgentConfig) The constructed configuration.
    """
    if provider is None:
        provider = "openai"

    config_kwargs: dict[str, Any] = {
        **_BASELINE_CONFIG_KWARGS,
        "model_provider": provider,
        "model_context_limit_tokens": context_limit,
        "model_max_tokens_output": max_output,
        "budgets": Budgets(
            max_wall_seconds=max_wall_seconds,
            max_turns=max_turns,
            max_tool_calls=max_tool_calls,
            token_budget=token_budget,
            token_budget_buffer=token_budget_buffer,
        ),
    }
    if duplicate_call_window is not None:
        config_kwargs["duplicate_call_window"] = duplicate_call_window
    if mode is not None:
        config_kwargs["mode"] = mode
    config_kwargs.update(overrides)
    return AgentConfig(**config_kwargs)


def init_fixture_repo(
    path: Path,
    files: dict[str, str] | None = None,
) -> Repo:
    """Initialize a git repository with an initial commit.

    Args:
        path (Path): Repository root to initialize.
        files (dict[str, str] | None): Repo-relative path to content.
            Parent directories are created as needed. Defaults to
            `{"main.py": MAIN_CONTENT}`.

    Returns:
        (Repo) The initialized repository.
    """
    contents = files if files is not None else {"main.py": MAIN_CONTENT}
    repo = Repo.init(str(path))
    for name, content in contents.items():
        target = path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    add(str(path), [name.encode() for name in contents])
    commit(
        str(path),
        message=b"initial",
        author=_AUTHOR,
        committer=_AUTHOR,
    )
    return repo


def build_runtime(
    config: AgentConfig,
    tmp_path: Path,
    *,
    monkeypatch: pytest.MonkeyPatch | None = None,
    initialize: str | None = None,
    output_dir: Path | None = None,
) -> Runtime:
    """Build a Runtime rooted at a temporary workspace.

    Args:
        config (AgentConfig): Runtime configuration.
        tmp_path (Path): Workspace root and output parent.
        monkeypatch (pytest.MonkeyPatch | None): When given, the provider
            API key env vars are set so initialization can resolve them.
        initialize (str | None): User message to initialize the session
            with; the runtime is left uninitialized when None.
        output_dir (Path | None): Output directory; defaults to an
            ``output`` directory inside the workspace.

    Returns:
        (Runtime) The constructed runtime.
    """
    if monkeypatch is not None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        monkeypatch.setenv("FIREWORKS_API_KEY", "test-key")
    runtime = Runtime(
        config=config,
        workspace_root=tmp_path,
        output_dir=output_dir if output_dir is not None else tmp_path / "output",
    )
    if initialize is not None:
        runtime.initialize(initialize)
    return runtime
