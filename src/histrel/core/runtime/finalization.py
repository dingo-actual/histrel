#!/usr/bin/env python3
"""
Session finalization for the benchmark agent runtime.

Finalizes a session after a terminal outcome: writes session
artifacts (config, timings, metrics, tokens, costs, diff, journal).

    Important Functions:
        finalize_session(runtime, outcome) -> SessionOutcome: Write.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from histrel.context.cost_calculator import (
    CostCalculator,
    ModelPricing,
    fetch_fireworks_model_pricing,
)
from histrel.context.session_artifacts import SessionArtifacts
from histrel.context.session_events import build_source_context
from histrel.core.clock import monotonic_now
from histrel.terminal import SessionOutcome

if TYPE_CHECKING:
    from histrel.core.runtime.runtime import Runtime

log = logger.bind(component="bench_agent.runtime")


def finalize_session(runtime: Runtime, outcome: SessionOutcome) -> SessionOutcome:
    """Finalize the session: write artifacts.

    Args:
        runtime (Runtime): The runtime session state.
        outcome (SessionOutcome): Terminal outcome.

    Returns:
        (SessionOutcome) The same outcome.
    """
    runtime.timings.add(
        "session_wall_seconds",
        monotonic_now() - runtime.session_start,
    )
    cost_calculator = CostCalculator(pricing=_fetch_pricing(runtime))
    diff_text = _git_diff_text(runtime.workspace_root)
    messages_jsonl = _journal_text(runtime)
    _write_artifacts(runtime, outcome, cost_calculator, messages_jsonl, diff_text)

    log.info(
        "session_finalized",
        session_id=runtime.config.session_id,
        status=outcome.status.value,
        reason=outcome.reason,
        turns=runtime.turn_count,
        tool_calls=runtime.tool_call_count,
    )
    log.debug(
        "session_token_usage",
        session_id=runtime.config.session_id,
        total_input=runtime.token_ledger.total_input,
        total_output=runtime.token_ledger.total_output,
        total_cached=runtime.token_ledger.total_cached,
    )

    return outcome


def _fetch_pricing(runtime: Runtime) -> ModelPricing | None:
    """Fetch model pricing for cost calculation, if Fireworks.

    Args:
        runtime (Runtime): The runtime session state.

    Returns:
        (ModelPricing | None) Resolved pricing, or None.
    """
    if runtime.config.model_provider != "fireworks":
        return None
    return fetch_fireworks_model_pricing(runtime.config.model_name)


def _git_diff_text(workspace_root: Path) -> str:
    """Generate the repository diff text for artifacts.

    Args:
        workspace_root (Path): Path to the repository worktree.

    Returns:
        (str) The ``git diff`` output, or an empty string on failure.
    """
    diff_text = ""
    try:
        result = subprocess.run(
            ["git", "diff"],
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            timeout=10,
        )
        diff_text = result.stdout
    except Exception:
        diff_text = ""
    return diff_text


def _journal_text(runtime: Runtime) -> str:
    """Read the raw journal file for the messages.jsonl artifact.

    The journal file IS the messages.jsonl artifact: read it raw to
    preserve the envelope format (checksums, schema version).

    Args:
        runtime (Runtime): The runtime session state.

    Returns:
        (str) Raw journal content, or an empty string if missing.
    """
    journal_path = runtime.journal._path
    if journal_path.exists():
        return journal_path.read_text(encoding="utf-8")
    return ""


def _write_artifacts(
    runtime: Runtime,
    outcome: SessionOutcome,
    cost_calculator: CostCalculator,
    messages_jsonl: str,
    diff_text: str,
) -> None:
    """Write session artifacts, but not over the journal file.

    Args:
        runtime (Runtime): The runtime session state.
        outcome (SessionOutcome): Terminal outcome.
        cost_calculator (CostCalculator): Cost calculator for tokens.
        messages_jsonl (str): Raw journal content.
        diff_text (str): Repository diff text.
    """
    artifacts = SessionArtifacts(
        base_dir=runtime.output_dir / runtime.config.session_id / "artifacts"
    )
    artifacts.write_artifacts(
        config=runtime.config,
        timings=runtime.timings,
        tool_metrics=runtime.tool_metrics,
        token_ledger=runtime.token_ledger,
        cost_calculator=cost_calculator,
        outcome={
            "status": outcome.status.value,
            "reason": outcome.reason,
        },
        manifest_resolved={
            "model_provider": runtime.config.model_provider,
            "model_name": runtime.config.model_name,
        },
        messages_jsonl=messages_jsonl,
        diff_text=diff_text,
        session_events=runtime.session_events,
        source_context=build_source_context(
            config=runtime.config,
            workspace_root=runtime.workspace_root,
            tool_names=[tool.name for tool in runtime.ordinary_tools],
        ),
    )
