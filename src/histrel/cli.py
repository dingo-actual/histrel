#!/usr/bin/env python3
"""
CLI entry point for the benchmark agent.

Reads environment variables injected by the harness, loads a TOML
configuration file, constructs the runtime, and runs the agent loop.

    Important Classes:
        BenchmarkEnv: Environment variables from the harness.

    Important Functions:
        read_env() -> BenchmarkEnv: Read all harness environment variables.
        build_config_from_env(toml_path, env) -> AgentConfig: Build config from TOML + env.
        main(config_path, workspace_root, output_dir) -> int: Run the agent.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from histrel.core.config import AgentConfig, Budgets, Mode, SystemPrompts, load_config
from histrel.core.runtime import Runtime, run_agent
from histrel.logging.agent_logger import setup_file_sink
from histrel.prompts import get_histrel_sys_prompts
from histrel.terminal import SessionOutcome, SessionStatus

# Bound logger for CLI-level lifecycle events.
log = logger.bind(component="bench_agent.cli")


class BenchmarkEnv(BaseModel):
    """Environment variables injected by the harness.

    Instance Attributes:
        run_name (str): Run identifier for this benchmark batch.
        agent_type (str): Agent type identifier.
        instance_id (str): SWE-bench instance identifier.
        session_id (str): Session identifier.
        timeout_seconds (int): Wall-clock timeout in seconds.
        fireworks_api_key (str): Fireworks API key.
    """

    run_name: str = ""
    agent_type: str = ""
    instance_id: str = ""
    session_id: str = ""
    timeout_seconds: int = 0
    fireworks_api_key: str = ""


def read_env() -> BenchmarkEnv:
    """Read all harness environment variables.

    Returns:
        (BenchmarkEnv) Parsed environment variables.
    """
    timeout_str = os.environ.get("TIMEOUT_SECONDS", "")
    try:
        timeout_seconds = int(timeout_str) if timeout_str else 0
    except ValueError:
        timeout_seconds = 0

    return BenchmarkEnv(
        run_name=os.environ.get("RUN_NAME", ""),
        agent_type=os.environ.get("AGENT_TYPE", ""),
        instance_id=os.environ.get("INSTANCE_ID", ""),
        session_id=os.environ.get("SESSION_ID", ""),
        timeout_seconds=timeout_seconds,
        fireworks_api_key=os.environ.get("FIREWORKS_API_KEY", ""),
    )


def build_config_from_env(
    toml_path: Path,
    env: BenchmarkEnv,
) -> AgentConfig:
    """Build an AgentConfig from a TOML file, overridden by env vars.

    Environment variables override TOML values for session_id, task_id,
    and timeout (which maps to max_wall_seconds budget).

    Args:
        toml_path (Path): Path to the TOML configuration file.
        env (BenchmarkEnv): Environment variables from the harness.

    Returns:
        (AgentConfig) Merged configuration.
    """
    config = load_config(toml_path)

    # Environment overrides
    overrides: dict[str, object] = {}

    if env.session_id:
        overrides["session_id"] = env.session_id
    if env.instance_id:
        overrides["task_id"] = env.instance_id
    if env.timeout_seconds > 0:
        overrides["budgets"] = Budgets(
            max_wall_seconds=env.timeout_seconds,
            max_turns=config.budgets.max_turns,
            max_tool_calls=config.budgets.max_tool_calls,
        )

    if overrides:
        config = config.model_copy(update=overrides)

    return config


def _log_environment(env: BenchmarkEnv) -> None:
    """Log the harness environment at debug level.

    The API key env var is deliberately never logged.

    Args:
        env (BenchmarkEnv): Environment variables from the harness.
    """
    log.debug(
        "harness_environment_read",
        run_name=env.run_name,
        agent_type=env.agent_type,
        instance_id=env.instance_id,
        session_id=env.session_id,
        timeout_seconds=env.timeout_seconds,
    )


def _log_configuration(config: AgentConfig, config_path: Path) -> None:
    """Log the loaded agent configuration at debug level.

    Args:
        config (AgentConfig): Merged agent configuration.
        config_path (Path): Path to the TOML configuration file.
    """
    log.debug(
        "agent_configuration_loaded",
        config_path=str(config_path),
        session_id=config.session_id,
        task_id=config.task_id,
        mode=config.mode,
        model_provider=config.model_provider,
        model_name=config.model_name,
        max_wall_seconds=config.budgets.max_wall_seconds,
    )


def _log_starting(config: AgentConfig, workspace_root: Path, output_dir: Path) -> None:
    """Log the start of the agent session.

    Args:
        config (AgentConfig): Merged agent configuration.
        workspace_root (Path): Repository workspace root.
        output_dir (Path): Output directory for artifacts.
    """
    log.info(
        "benchmark_agent_starting",
        session_id=config.session_id,
        task_id=config.task_id,
        model=config.model_name,
        workspace_root=str(workspace_root),
        output_dir=str(output_dir),
    )


def _log_finished(
    config: AgentConfig, runtime: Runtime, outcome: SessionOutcome
) -> None:
    """Log the end of the agent session.

    Args:
        config (AgentConfig): Merged agent configuration.
        runtime (Runtime): The runtime that ran the session.
        outcome (SessionOutcome): Session outcome.
    """
    log.info(
        "benchmark_agent_finished",
        session_id=config.session_id,
        status=outcome.status.value,
        reason=outcome.reason,
        turns=runtime.turn_count,
        tool_calls=runtime.tool_call_count,
    )


def _exit_code(outcome: SessionOutcome) -> int:
    """Map a session outcome to a process exit code.

    Args:
        outcome (SessionOutcome): Session outcome.

    Returns:
        (int) Exit code: 0 for success, 1 for failure.
    """
    if outcome.status == SessionStatus.SUCCEEDED:
        return 0
    return 1


def main(
    *,
    user_msg: str,
    config_path: Path,
    workspace_root: Path,
    output_dir: Path,
    mode: Mode = "stateless",
) -> int:
    """Run the benchmark agent.

    Reads environment variables, loads config, constructs the runtime,
    and runs the agent loop. Returns an exit code.

    Args:
        user_msg (str): User request message for the session.
        config_path (Path): Path to the TOML configuration file.
        workspace_root (Path): Repository workspace root.
        output_dir (Path): Output directory for artifacts.
        mode (Mode): Agent loop mode from the command line. Defaults to
            "stateless". Selects which prompts are rendered.

    Returns:
        (int) Exit code: 0 for success, 1 for failure.
    """
    env = read_env()
    config = build_config_from_env(config_path, env)
    prompts = get_histrel_sys_prompts(workspace_root, mode)
    config = config.model_copy(
        update={
            "mode": mode,
            "agent_system_prompt": SystemPrompts(**prompts),
        }
    )

    # Catch-all JSON-lines file sink for the agent's own log. The file is
    # written to /tmp so it never pollutes the repository working tree; the
    # harness exports it into the session artifacts directory after the run.
    setup_file_sink()

    _log_environment(env)
    _log_configuration(config, config_path)
    _log_starting(config, workspace_root, output_dir)

    runtime = Runtime(
        config=config,
        workspace_root=workspace_root,
        output_dir=output_dir,
    )

    outcome = run_agent(user_msg, runtime)

    _log_finished(config, runtime, outcome)
    return _exit_code(outcome)
