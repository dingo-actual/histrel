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
from benchmark.predict.usage_helpers import known_cost, known_measurement
from benchmark.predict.types import AGENT_TYPE, build_attempt_identity
import histrel
from histrel.logging.agent_logger import (
    AGENT_LOG_EXPORT_NAME,
    AGENT_LOG_FILE_PATH,
)
from benchmark.common.logging import safe_log
from loguru import logger

# Default model when bench_settings.toml does not specify one.
_DEFAULT_MODEL = "accounts/fireworks/models/glm-5p3-flash"


def _repetition_from_env() -> int:
    """Return the repetition number passed by the harness.

    Returns:
        (int) 1-based repetition number; 1 when unset or malformed.
    """
    try:
        return int(os.environ.get("REPETITION") or "1")
    except ValueError:
        return 1


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


class _ModelCallTotals:
    """Token totals across the bench-agent's LLM calls.

    Instance Attributes:
        input (int): Cumulative input tokens.
        output (int): Cumulative output tokens.
        cached (int): Cumulative cached tokens.
        max_context (int): Largest single-call input token count.
    """

    def __init__(self) -> None:
        """Initialize zeroed totals."""
        self.input = 0
        self.output = 0
        self.cached = 0
        self.max_context = 0


class _HistrelCallTotals:
    """Token totals across Histrel tool calls.

    Instance Attributes:
        entries (dict[str, dict[str, int | float | str | bool | None]]):
            Per-run usage records keyed by run ID.
        input (int): Cumulative input tokens.
        output (int): Cumulative output tokens.
        cached (int): Cumulative cached tokens.
        total (int): Cumulative reported token totals.
        max_context (int): Largest per-run maximum context.
    """

    def __init__(self) -> None:
        """Initialize zeroed totals."""
        self.entries: dict[str, dict[str, int | float | str | bool | None]] = {}
        self.input = 0
        self.output = 0
        self.cached = 0
        self.total = 0
        self.max_context = 0


def _token_count(value: object) -> int:
    """Return a non-negative token count from a usage field.

    Args:
        value (object): Candidate token count.

    Returns:
        (int) The token count, or 0 when the field is unusable.
    """
    measurement = known_measurement(value)
    return 0 if measurement is None else int(measurement)


def _read_jsonl_records(path: Path) -> list[dict]:
    """Read JSONL records, skipping blank and malformed lines.

    Args:
        path (Path): JSONL file to read.

    Returns:
        (list[dict]) Parsed records; empty when the file is unreadable.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    records: list[dict] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def _load_costs(
    usage_file: Path,
) -> tuple[float | None, float | None, float | None, bool]:
    """Load the final cost totals from the agent's costs.json.

    Args:
        usage_file (Path): Path to costs.json file.

    Returns:
        (tuple[float | None, float | None, float | None, bool]) Total,
            model, and compaction cost, plus whether cost accounting is
            complete. Every cost is None when costs.json is missing or
            unreadable: an unknown cost must not read as free.
    """
    try:
        data = json.loads(usage_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None, None, False
    if not isinstance(data, dict):
        return None, None, None, False
    return (
        known_cost(data.get("total_cost")),
        known_cost(data.get("model_cost")),
        known_cost(data.get("compaction_cost")),
        data.get("is_complete") is True,
    )


def _sum_model_calls(calls_file: Path) -> _ModelCallTotals:
    """Sum token usage across the bench-agent's LLM calls.

    Args:
        calls_file (Path): Path to token_usage_calls.jsonl.

    Returns:
        (_ModelCallTotals) Token totals and largest call context.
    """
    totals = _ModelCallTotals()
    for record in _read_jsonl_records(calls_file):
        usage = record.get("usage")
        if not isinstance(usage, dict):
            continue
        call_input = _token_count(usage.get("input"))
        totals.input += call_input
        totals.output += _token_count(usage.get("output"))
        totals.cached += _token_count(usage.get("cached"))
        totals.max_context = max(totals.max_context, call_input)
    return totals


def _sum_histrel_calls(histrel_file: Path) -> _HistrelCallTotals:
    """Sum token usage across Histrel tool calls.

    Args:
        histrel_file (Path): Path to histrel_calls.jsonl.

    Returns:
        (_HistrelCallTotals) Per-run records and token totals.
    """
    totals = _HistrelCallTotals()
    session_id = os.environ.get("SESSION_ID", "")
    for record in _read_jsonl_records(histrel_file):
        usage = record.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        call_input = _token_count(usage.get("input"))
        call_output = _token_count(usage.get("output"))
        call_cached = _token_count(usage.get("cached"))
        call_total = (
            call_input + call_output
            if usage.get("total") is None
            else _token_count(usage.get("total"))
        )
        call_max_context = _token_count(usage.get("maximum_context"))
        call_cost = known_cost(record.get("cost_usd"))
        run_id = str(record.get("run_id", ""))

        totals.entries[run_id] = {
            "session_id": session_id,
            "tokens_input": call_input,
            "tokens_output": call_output,
            "tokens_cache_read": call_cached,
            "tokens_cache_write": 0,
            "tokens_total": call_total,
            "cost_total": call_cost,
            "cost_complete": call_cost is not None,
            "maximum_context": call_max_context,
        }
        totals.input += call_input
        totals.output += call_output
        totals.cached += call_cached
        totals.total += call_total
        totals.max_context = max(totals.max_context, call_max_context)
    return totals


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
        """Load usage data from the agent's artifact directory.

        Token counts are read from token_usage_calls.jsonl and
        histrel_calls.jsonl, which are appended as each call completes, so
        they survive a session that never wrote costs.json. Costs are read
        from costs.json only; an absent or incomplete cost report is
        reported as an unknown cost, never as zero.

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
        model = _sum_model_calls(usage_file.parent / "token_usage_calls.jsonl")
        histrel = _sum_histrel_calls(usage_file.parent / "histrel_calls.jsonl")
        total_cost, model_cost, compaction_cost, costs_complete = _load_costs(
            usage_file
        )
        # "none" cost covers bench-agent LLM calls (model, compaction, and
        # routine) only; Histrel tool call costs are reported per-tool and
        # in the "total" record.
        none_cost = (
            None
            if model_cost is None or compaction_cost is None
            else model_cost + compaction_cost
        )

        session_id = os.environ.get("SESSION_ID", "")
        # Sum per-record totals: Histrel's tokens_total can exceed
        # input + output for total-only accounted calls, which the
        # session total must not drop.
        total_tokens = (model.input + model.output) + histrel.total
        result: dict[str, dict[str, int | float | str | bool | None]] = {
            "total": {
                "session_id": session_id,
                "tokens_input": model.input + histrel.input,
                "tokens_output": model.output + histrel.output,
                "tokens_cache_read": model.cached + histrel.cached,
                "tokens_cache_write": 0,
                "tokens_total": total_tokens,
                "cost_total": total_cost,
                "cost_complete": costs_complete,
                "maximum_context": max(model.max_context, histrel.max_context),
            },
            # "none" reflects bench-agent-only usage (excludes Histrel)
            "none": {
                "session_id": session_id,
                "tokens_input": model.input,
                "tokens_output": model.output,
                "tokens_cache_read": model.cached,
                "tokens_cache_write": 0,
                "tokens_total": model.input + model.output,
                "cost_total": none_cost,
                "cost_complete": costs_complete,
                "maximum_context": model.max_context,
            },
        }
        result.update(histrel.entries)
        return result

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

        # Extract the final workspace state relative to the base commit
        base_commit = instance.get("base_commit")
        patch_text = self.repo_manager.extract_patch(
            repo_dir, base_commit if isinstance(base_commit, str) else None
        )

        # Load usage data from agent artifacts
        # The agent writes to {output_dir}/{session_id}/artifacts/costs.json
        agent_artifact_dir = output_dir / session_id / "artifacts"
        usage_file = agent_artifact_dir / "costs.json"
        # Identity of this attempt travels with the usage record so the
        # host can reject a record it did not ask for.
        identity = build_attempt_identity(
            run_name=os.environ.get("RUN_NAME", ""),
            repetition=_repetition_from_env(),
            mode=mode,
            instance_id=instance.get("instance_id", "unknown"),
            session_id=session_id,
        )
        usage_data = {"identity": identity, **self._load_usage_data(usage_file)}

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
