#!/usr/bin/env python3
"""
Session event recording for the benchmark agent.

Records the events the artifact directory reports: context compactions, plus
the fixed inputs that shaped every model call. Events are appended while the
session runs, so ``compactions.jsonl`` and ``source_context.json`` contain what
actually happened rather than a placeholder.

    Important Classes:
        CompactionEvent: One context compaction event.
        SessionEvents: Recorder for compaction events.

    Important Functions:
        build_source_context(...) -> dict[str, Any]: Fixed context inputs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from histrel.core.config import AgentConfig

if TYPE_CHECKING:
    from histrel.context.compactor import CompactedState

# Wall-clock budget for the source-context git queries.
_GIT_TIMEOUT_SECONDS: float = 5.0


class CompactionEvent(BaseModel):
    """One context compaction event.

    Instance Attributes:
        source_message_index_start (int): First compacted message index.
        source_message_index_end (int): Last compacted message index.
        compacted_message_count (int): Number of compacted messages.
        retained_message_count (int): Number of retained tail messages.
        summary (str): The generated working summary.
    """

    source_message_index_start: int
    source_message_index_end: int
    compacted_message_count: int
    retained_message_count: int
    summary: str


class SessionEvents:
    """Recorder for the session's compaction events.

    Instance Attributes:
        compactions (list[CompactionEvent]): Compaction events in order.
    """

    def __init__(self) -> None:
        """Initialize an empty recorder."""
        self.compactions: list[CompactionEvent] = []

    def record_compaction(self, state: CompactedState) -> None:
        """Append one compaction event.

        Args:
            state (CompactedState): State snapshot returned by the compactor.
        """
        start = state.source_message_index_start
        end = state.source_message_index_end
        compacted_count = end - start + 1 if start >= 0 and end >= start else 0
        self.compactions.append(
            CompactionEvent(
                source_message_index_start=start,
                source_message_index_end=end,
                compacted_message_count=compacted_count,
                retained_message_count=len(state.recent_message_ids),
                summary=state.working_summary,
            )
        )


def build_source_context(
    *,
    config: AgentConfig,
    workspace_root: Path,
    tool_names: list[str],
) -> dict[str, Any]:
    """Build the source-context snapshot for the artifact directory.

    Records the fixed inputs that shape every model call: the session and
    task identifiers, the loop mode, the model, the budgets, the workspace
    revision, and the tool set.

    Args:
        config (AgentConfig): Agent configuration.
        workspace_root (Path): Repository workspace root.
        tool_names (list[str]): Names of the session's tools.

    Returns:
        (dict[str, Any]) JSON-serializable source context.
    """
    return {
        "session_id": config.session_id,
        "task_id": config.task_id,
        "mode": config.mode,
        "model": {
            "provider": config.model_provider,
            "name": config.model_name,
            "context_limit_tokens": config.model_context_limit_tokens,
            "max_tokens_output": config.model_max_tokens_output,
        },
        "budgets": config.budgets.model_dump(),
        "workspace": {
            "root": str(workspace_root),
            "head_commit": _git_output(workspace_root, "rev-parse", "HEAD"),
            "branch": _git_output(workspace_root, "rev-parse", "--abbrev-ref", "HEAD"),
        },
        "tools": sorted(tool_names),
    }


def _git_output(workspace_root: Path, *args: str) -> str | None:
    """Run one git query, returning its output or None on failure.

    Args:
        workspace_root (Path): Repository workspace root.
        *args (str): Arguments passed to git.

    Returns:
        (str | None) Stripped output, or None when git is unavailable.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(workspace_root),
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None
