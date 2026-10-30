#!/usr/bin/env python3
"""
Benchmark agent wrapper for SWE-bench evaluation.

Provides a unified interface for running the benchmark agent (bench-agent)
against SWE-bench test cases, handling initialization, prompt construction,
patch generation via git diff, and token usage tracking.

The benchmark agent is a self-contained Python agent that runs inside the
SWE-bench Docker container and is invoked directly as a Python module:
``python -m histrel <config> <workspace> <output>``.

    Important Classes:
        BenchAgentWrapper: Wrapper for running the benchmark agent on SWE-bench instances.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

from benchmark.predict.agents.prompts import get_bench_agent_sys_prompts
from benchmark.predict.agents.pyc_preflight import (
    DEFAULT_AGENT_ENV_ROOT,
    agent_died_before_init,
    repair_pyc_cache,
    wipe_pycache_dirs,
)
from benchmark.predict.agents.repo import GitRepoManager
from benchmark.predict.agents.templates import MESSAGE_TEMPLATE, render_template
from benchmark.predict.config import Mode
from benchmark.predict.types import AGENT_TYPE
import histrel
from histrel.logging.agent_logger import (
    AGENT_LOG_EXPORT_NAME,
    AGENT_LOG_FILE_PATH,
)
from benchmark.common.logging import safe_log
from loguru import logger

# Default model when bench_settings.toml does not specify one.
_DEFAULT_MODEL = "accounts/fireworks/models/glm-5p3-flash"


def _agent_config_template_path() -> Path:
    """Locate the agent config template.

    Returns:
        (Path) The repository-root ``histrel.toml`` when running from a
        checkout, otherwise the packaged copy beside the ``histrel``
        package.
    """
    repo_root_template = Path(__file__).resolve().parents[3] / "histrel.toml"
    if repo_root_template.exists():
        return repo_root_template
    return Path(histrel.__file__).resolve().parent / "histrel.toml"


# Path to the agent config template: the repository root in a checkout, or the
# copy shipped inside the installed ``histrel`` package (see the wheel's
# force-include in pyproject.toml).
_AGENT_CONFIG_TEMPLATE_PATH = _agent_config_template_path()


class BenchAgentWrapper:
    """Wrapper for running the benchmark agent on SWE-bench instances.

    This class handles benchmark agent execution for benchmarking, including
    configuration generation, prompt construction, command execution, and
    patch extraction via git diff.

    Instance Attributes:
        config_path (Path): Path to bench_settings.toml configuration file.

    Example Usage:
        ```python
            >>> wrapper = BenchAgentWrapper(Path("bench_settings.toml"))
            >>> patch, usage = wrapper.generate_patch(instance, Path("/testbed"), session_id)
            >>> print(f"Tokens used: {usage['total']['tokens_total']}")
        ```
    """

    def __init__(self, config_path: Path):
        """Initialize BenchAgentWrapper.

        Args:
            config_path (Path): Path to bench_settings.toml configuration file.
        """
        self.config_path = config_path
        self.repo_manager = GitRepoManager()

        with contextlib.suppress(Exception):
            logger.bind(
                agent_type=AGENT_TYPE,
                component="benchmark.agent",
                origin="bench_harness_predict_container",
            ).info(
                "Initialized BenchAgentWrapper",
                config_path=str(config_path),
            )

    def _parse_model_from_config(self, config: dict) -> str:
        """Parse model identifier from the harness bench_settings.toml config.

        Args:
            config (dict): Configuration dictionary from bench_settings.toml.

        Returns:
            (str) Model identifier.
        """
        histrel_cfg = config.get("histrel", {})
        model = histrel_cfg.get("models", _DEFAULT_MODEL)
        if isinstance(model, dict):
            model = model.get("default", _DEFAULT_MODEL)
        return str(model)

    def _load_config(self) -> dict:
        """Load configuration from the harness bench_settings.toml file.

        Returns:
            (dict) Configuration dictionary.
        """
        if not self.config_path.exists():
            raise RuntimeError(f"Config file not found: {self.config_path}")
        return tomllib.loads(self.config_path.read_text(encoding="utf-8"))

    def _write_agent_config(self, repo_dir: Path, mode: Mode = "stateless") -> Path:
        """Write the agent TOML config to the repo directory.

        Read the template from the repository-root ``histrel.toml``, then
        replaces the model name and agent system prompt based on the
        harness config and the agent type.
        The agent system prompt is the rendered prompt from
        ``get_bench_agent_sys_prompts`` for the active agent type
        (bench-agent or bench-agent-no-mt) and mode.

        Args:
            repo_dir (Path): Repository directory where the config is written.
            mode (Mode): Agent loop mode. Defaults to "stateless".

        Returns:
            (Path) Path to the written agent_config.toml.
        """
        if not _AGENT_CONFIG_TEMPLATE_PATH.exists():
            raise RuntimeError(
                f"Agent config template not found: {_AGENT_CONFIG_TEMPLATE_PATH}"
            )

        config_text = _AGENT_CONFIG_TEMPLATE_PATH.read_text(encoding="utf-8")

        if mode not in ("stateless", "react"):
            raise RuntimeError(
                f"Invalid mode '{mode}': expected 'stateless' or 'react'"
            )

        # Parse model from harness config
        harness_config = self._load_config()
        model = self._parse_model_from_config(harness_config)

        # Replace placeholders in the template
        config_text = config_text.replace(
            f'model_name = "{_DEFAULT_MODEL}"',
            f'model_name = "{model}"',
        )

        # Replace the placeholder system prompt with the rendered prompt for
        # the agent type. json.dumps output is a valid TOML basic string
        # (same escape set), so the rendered prompt round-trips through
        # tomllib regardless of quotes, backslashes, or newlines it contains.
        sys_prompts = get_bench_agent_sys_prompts(repo_dir, mode=mode)
        prompt_toml_init = json.dumps(sys_prompts["initialize"], ensure_ascii=False)
        prompt_toml_step = json.dumps(sys_prompts["step"], ensure_ascii=False)
        prompt_placeholder = (
            'agent_system_prompt = { initialize = "placeholder", step = "placeholder" }'
        )
        if prompt_placeholder not in config_text:
            raise RuntimeError(
                "agent_system_prompt placeholder not found in agent config "
                f"template: {_AGENT_CONFIG_TEMPLATE_PATH}"
            )
        config_text = config_text.replace(
            prompt_placeholder,
            "agent_system_prompt = { "
            f"initialize = {prompt_toml_init}, step = {prompt_toml_step} "
            "}",
        )

        config_path = repo_dir / "agent_config.toml"
        config_path.write_text(config_text, encoding="utf-8")
        return config_path

    def _load_usage_data(self, usage_file: Path) -> dict:
        """Load usage data from the agent's costs.json file.

        The benchmark agent writes token usage to costs.json,
        token_usage_calls.jsonl, and histrel_calls.jsonl in its artifact
        directory. This method reads all three and converts them to the
        harness-expected usage.json format.

        Per-tool entries from histrel_calls.jsonl are keyed by their
        run_id at the top level of the returned dict, so tool-specific
        records sit alongside 'none' and 'total'.

        maximum_context is the largest input (context) token count among
        all LLM calls. For model/compaction calls it is the largest
        per-call ``input`` (which already includes cached tokens). For
        Histrel runs it is the per-run ``maximum_context`` reported by
        the tool_api usage metadata. Token totals (input + output) are
        never used as context values.

        Args:
            usage_file (Path): Path to costs.json file.

        Returns:
            (dict) Usage data structure with 'total' and 'none' keys
                matching the harness UsageData format, plus per-tool
                entries keyed by run_id.
        """
        error_data = {
            "total": {
                "session_id": "error",
                "tokens_input": -1,
                "tokens_output": -1,
                "tokens_cache_read": -1,
                "tokens_cache_write": -1,
                "tokens_total": -1,
                "cost_total": -1.0,
                "maximum_context": -1,
            }
        }

        if not usage_file.exists():
            return error_data

        try:
            data = json.loads(usage_file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return error_data

            total_cost = float(data.get("total_cost", 0.0))
            # "none" cost covers bench-agent LLM calls (model, compaction,
            # and routine) only; Histrel tool call costs are reported
            # per-tool and in the "total" record.
            model_cost = float(data.get("model_cost", 0.0))
            compaction_cost = float(data.get("compaction_cost", 0.0))
            none_cost = model_cost + compaction_cost

            # Read token_usage_calls.jsonl for bench-agent LLM call token
            # counts (model, compaction, and routine kinds)
            calls_file = usage_file.parent / "token_usage_calls.jsonl"
            model_input = 0
            model_output = 0
            model_cached = 0
            model_max_context = 0

            if calls_file.exists():
                for line in calls_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        call = json.loads(line)
                        usage = call.get("usage", {})
                        call_input = int(usage.get("input", 0))
                        call_output = int(usage.get("output", 0))
                        model_input += call_input
                        model_output += call_output
                        model_cached += int(usage.get("cached", 0))
                        model_max_context = max(model_max_context, call_input)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue

            # Read histrel_calls.jsonl for Histrel tool call usage
            mt_file = usage_file.parent / "histrel_calls.jsonl"
            histrel_entries: dict[str, dict[str, int | float | str]] = {}
            mt_input = 0
            mt_output = 0
            mt_cached = 0
            mt_total = 0
            mt_max_context = 0

            if mt_file.exists():
                for line in mt_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                        run_id = entry.get("run_id", "")
                        mt_usage = entry.get("usage", {})
                        call_input = int(mt_usage.get("input", 0))
                        call_output = int(mt_usage.get("output", 0))
                        call_cached = int(mt_usage.get("cached", 0))
                        call_total = int(
                            mt_usage.get("total", call_input + call_output)
                        )
                        call_max_context = int(mt_usage.get("maximum_context", 0))
                        call_cost = float(entry.get("cost_usd", 0.0))
                        session_id = os.environ.get("SESSION_ID", "")

                        histrel_entries[run_id] = {
                            "session_id": session_id,
                            "tokens_input": call_input,
                            "tokens_output": call_output,
                            "tokens_cache_read": call_cached,
                            "tokens_cache_write": 0,
                            "tokens_total": call_total,
                            "cost_total": call_cost,
                            "maximum_context": call_max_context,
                        }

                        mt_input += call_input
                        mt_output += call_output
                        mt_cached += call_cached
                        mt_total += call_total
                        mt_max_context = max(mt_max_context, call_max_context)
                    except (json.JSONDecodeError, TypeError, ValueError):
                        continue

            total_input = model_input + mt_input
            total_output = model_output + mt_output
            total_cached = model_cached + mt_cached
            max_context = max(model_max_context, mt_max_context)
            # Sum per-record totals: Histrel's tokens_total can exceed
            # input + output for total-only accounted calls, which the
            # session total must not drop.
            total_tokens = (model_input + model_output) + mt_total
            session_id = os.environ.get("SESSION_ID", "")

            result: dict[str, dict[str, int | float | str]] = {
                "total": {
                    "session_id": session_id,
                    "tokens_input": total_input,
                    "tokens_output": total_output,
                    "tokens_cache_read": total_cached,
                    "tokens_cache_write": 0,
                    "tokens_total": total_tokens,
                    "cost_total": total_cost,
                    "maximum_context": max_context,
                },
                # "none" reflects bench-agent-only usage (excludes Histrel)
                "none": {
                    "session_id": session_id,
                    "tokens_input": model_input,
                    "tokens_output": model_output,
                    "tokens_cache_read": model_cached,
                    "tokens_cache_write": 0,
                    "tokens_total": model_input + model_output,
                    "cost_total": none_cost,
                    "maximum_context": model_max_context,
                },
            }
            result.update(histrel_entries)
            return result
        except (json.JSONDecodeError, OSError, TypeError):
            return error_data

    def generate_patch(
        self,
        instance: dict,
        repo_dir: Path,
        session_id: str,
        has_test_runner: bool = False,
        timeout: int = 7200,
        mode: Mode = "stateless",
    ) -> tuple[str, dict]:
        """Generate patch for SWE-bench instance using the benchmark agent.

        Args:
            instance (dict): SWE-bench instance data.
            repo_dir (Path): Path to repository (e.g. /testbed).
            session_id (str): Session identifier for tracking.
            has_test_runner (bool): Whether test runners are available.
            timeout (int): Timeout in seconds. Defaults to 7200.
            mode (Mode): Agent loop mode passed to ``python -m histrel
                --mode``. Defaults to "stateless".

        Returns:
            (tuple[str, dict]) Patch text and usage statistics dictionary.
        """
        agent_type = AGENT_TYPE

        if not has_test_runner:
            logger.bind(
                component="benchmark.agent",
                agent_type=agent_type,
                instance_id=instance.get("instance_id", "unknown"),
                session_id=session_id,
                origin="bench_harness_predict_container",
            ).warning("has_test_runner is False - test runners may not be available")

        logger.bind(
            component="benchmark.agent",
            agent_type=agent_type,
            instance_id=instance.get("instance_id", "unknown"),
            repo_dir=str(repo_dir),
            has_test_runner=has_test_runner,
            session_id=session_id,
            origin="bench_harness_predict_container",
        ).info("Starting benchmark agent patch generation")

        self.repo_manager.validate_repo(repo_dir)

        # Write agent-specific TOML config from template + harness config
        agent_config_path = self._write_agent_config(repo_dir, mode)

        # Output directory for agent artifacts
        output_dir = Path("/workspace/output")

        user_msg = render_template(
            MESSAGE_TEMPLATE,
            repo=instance.get("repo", ""),
            problem_statement=instance.get("problem_statement", ""),
            hints_text=instance.get("hints_text", "") or "--None--",
        )
        user_msg = user_msg.replace("`", "\\`")

        # Build command to run the benchmark agent as a Python module
        agent_cmd = [
            "python",
            "-m",
            "histrel",
            "--mode",
            mode,
            str(agent_config_path),
            str(repo_dir),
            str(output_dir),
            "--",
            user_msg,
        ]

        # Build shell command that activates the agent conda env
        activate_cmd = [
            "source /opt/miniconda3/etc/profile.d/conda.sh &&",
            "conda activate agent &&",
        ]
        shell_cmd = (
            " ".join(activate_cmd)
            + " "
            + " ".join(shlex.quote(arg) for arg in agent_cmd)
        )

        safe_log(
            "Executing benchmark agent",
            "info",
            bind_values={
                "component": "benchmark.agent",
                "command": shell_cmd,
                "command_length": len(shell_cmd),
                "cwd": str(repo_dir),
                "agent_type": agent_type,
                "timeout_seconds": timeout,
                "session_id": session_id,
            },
        )

        # Preflight: this host has produced .pyc files with valid headers
        # but corrupt payloads; a single bad file aborts the agent at
        # import time. Delete unloadable ones so CPython recompiles them.
        repair_pyc_cache(DEFAULT_AGENT_ENV_ROOT)

        instance_id = instance.get("instance_id", "unknown")
        result = self._run_agent_process(
            shell_cmd, repo_dir, timeout, session_id, instance_id, agent_type
        )

        # A crash before the agent logs its initialization event is the
        # signature of corrupt bytecode that still loads (garbage
        # instructions). Wipe the bytecode cache once and retry; if the
        # second run fails the same way, it is a genuine error and the
        # normal failure handling below reports it.
        if agent_died_before_init(result):
            logger.bind(
                component="benchmark.agent",
                agent_type=agent_type,
                session_id=session_id,
                instance_id=instance_id,
                returncode=result.returncode,
                origin="bench_harness_predict_container",
            ).warning(
                "Agent crashed before initialization after preflight repair; "
                "wiping bytecode cache and retrying once"
            )
            wipe_pycache_dirs(DEFAULT_AGENT_ENV_ROOT)
            result = self._run_agent_process(
                shell_cmd, repo_dir, timeout, session_id, instance_id, agent_type
            )

        # Log output
        self._log_agent_output(result, session_id, instance_id, agent_type)

        # Extract patch from git diff
        patch_text = self.repo_manager.extract_patch(repo_dir)

        # Load usage data from agent artifacts
        # The agent writes to {output_dir}/{session_id}/artifacts/costs.json
        agent_artifact_dir = output_dir / session_id / "artifacts"
        usage_file = agent_artifact_dir / "costs.json"
        usage_data = self._load_usage_data(usage_file)

        # Export the agent's /tmp log file into the session artifacts
        # directory (mounted to the host artifacts location).
        self._export_agent_log(
            Path(AGENT_LOG_FILE_PATH), agent_artifact_dir, session_id, agent_type
        )

        # Write usage.json in the location the harness expects
        usage_output_path = output_dir / f"{agent_type}_usage.json"
        usage_output_path.write_text(json.dumps(usage_data, indent=2), encoding="utf-8")

        logger.bind(
            component="benchmark.agent",
            agent_type=agent_type,
            patch_length=len(patch_text),
            session_id=session_id,
            tokens_total=usage_data.get("total", {}).get("tokens_total", -1),
            origin="bench_harness_predict_container",
        ).info("Benchmark agent patch generation complete")

        return patch_text, usage_data

    def _run_agent_process(
        self,
        shell_cmd: str,
        repo_dir: Path,
        timeout: int,
        session_id: str,
        instance_id: str,
        agent_type: str,
    ) -> subprocess.CompletedProcess:
        """Run the benchmark agent subprocess once.

        Args:
            shell_cmd (str): Full shell command (already shell-quoted).
            repo_dir (Path): Working directory for the subprocess.
            timeout (int): Timeout in seconds.
            session_id (str): Session identifier for logging.
            instance_id (str): Instance identifier for logging.
            agent_type (str): Agent type identifier for logging.

        Returns:
            (subprocess.CompletedProcess) Finished process result. A
                timeout is returned as a completed process with
                returncode 124.
        """
        try:
            return subprocess.run(
                ["bash", "-c", shell_cmd],
                cwd=str(repo_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            logger.bind(
                component="benchmark.agent",
                agent_type=agent_type,
                session_id=session_id,
                instance_id=instance_id,
                origin="bench_harness_predict_container",
            ).error("Benchmark agent timed out", timeout=timeout)
            return subprocess.CompletedProcess(
                args=shell_cmd,
                returncode=124,
                stdout="",
                stderr=f"Timed out after {timeout}s",
            )

    def _export_agent_log(
        self,
        log_source: Path,
        agent_artifact_dir: Path,
        session_id: str,
        agent_type: str,
    ) -> None:
        """Copy the agent's log file into the artifacts directory.

        The agent writes its catch-all log to /tmp inside the container.
        This copies the file into the session artifacts directory, which
        is mounted to the host artifacts location, so the log is exported
        with the rest of the run output.

        Args:
            log_source (Path): Agent log file path (typically /tmp).
            agent_artifact_dir (Path): Session artifacts directory.
            session_id (str): Session identifier.
            agent_type (str): Agent type identifier.
        """
        if not log_source.is_file():
            safe_log(
                "Agent log file not found for export",
                "warning",
                bind_values={
                    "component": "benchmark.agent",
                    "agent_type": agent_type,
                    "session_id": session_id,
                    "log_path": str(log_source),
                },
            )
            return

        agent_artifact_dir.mkdir(parents=True, exist_ok=True)
        log_destination = agent_artifact_dir / AGENT_LOG_EXPORT_NAME
        shutil.copy2(log_source, log_destination)

        safe_log(
            "Agent log file exported to artifacts",
            "info",
            bind_values={
                "component": "benchmark.agent",
                "agent_type": agent_type,
                "session_id": session_id,
                "log_source": str(log_source),
                "log_destination": str(log_destination),
                "log_size_bytes": log_destination.stat().st_size,
            },
        )

    def _log_agent_output(
        self,
        result: subprocess.CompletedProcess,
        session_id: str,
        instance_id: str,
        agent_type: str,
    ) -> None:
        """Log benchmark agent command output.

        Args:
            result (subprocess.CompletedProcess): Completed subprocess result.
            session_id (str): Session identifier.
            instance_id (str): Instance identifier.
            agent_type (str): Agent type identifier.
        """
        if result.stdout:
            safe_log(
                "Benchmark agent stdout",
                "info",
                bind_values={
                    "session_id": session_id,
                    "instance_id": instance_id,
                    "agent_type": agent_type,
                    "output": result.stdout,
                    "output_length": len(result.stdout),
                    "component": "benchmark.agent",
                    "origin": "bench_harness_predict_container",
                },
            )
        if result.stderr:
            safe_log(
                "Benchmark agent stderr",
                "warning",
                bind_values={
                    "session_id": session_id,
                    "instance_id": instance_id,
                    "agent_type": agent_type,
                    "output": result.stderr,
                    "output_length": len(result.stderr),
                    "component": "benchmark.agent",
                    "origin": "bench_harness_predict_container",
                },
            )

        if result.returncode != 0:
            sys.stderr.write(f"\n{'=' * 60}\n")
            sys.stderr.write(
                f"BENCHMARK AGENT FAILED (exit code: {result.returncode})\n"
            )
            sys.stderr.write(f"{'=' * 60}\n")
            if result.stderr:
                sys.stderr.write(f"\nSTDERR:\n{result.stderr}\n")
            if result.stdout:
                sys.stderr.write(f"\nSTDOUT:\n{result.stdout}\n")
            sys.stderr.write(f"{'=' * 60}\n")
            sys.stderr.flush()

            safe_log(
                "Benchmark agent failed with non-zero exit code",
                "error",
                bind_values={
                    "session_id": session_id,
                    "instance_id": instance_id,
                    "agent_type": agent_type,
                    "returncode": result.returncode,
                    "component": "benchmark.agent",
                    "origin": "bench_harness_predict_container",
                },
            )
