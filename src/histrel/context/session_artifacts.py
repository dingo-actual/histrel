#!/usr/bin/env python3
"""
Session artifact directory writer for the benchmark agent.

Writes a self-contained artifact directory with all session outputs:
messages, costs, timings, tool metrics, outcome, diff, compactions,
source context, and a result manifest with artifact hashes written last.

    Important Classes:
        ArtifactPaths: Paths for all artifact files.
        SessionArtifacts: Writes the complete artifact directory.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from histrel.context.cost_calculator import CostCalculator, CostReport
from histrel.context.session_events import SessionEvents
from histrel.context.timings import Timings
from histrel.context.token_ledger import TokenLedger, entry_payload
from histrel.context.tool_metrics import ToolMetrics
from histrel.core.config import AgentConfig


class ArtifactPaths(BaseModel):
    """Paths for all artifact files within the artifact directory.

    Instance Attributes:
        base_dir (Path): Base artifact directory.
        manifest_resolved (Path): Resolved manifest path.
        messages (Path): Messages JSONL path.
        token_usage_calls (Path): Token usage calls JSONL path.
        costs (Path): Costs JSON path.
        timings (Path): Timings JSON path.
        tool_metrics (Path): Tool metrics JSON path.
        source_context (Path): Source context JSON path.
        outcome (Path): Outcome JSON path.
        repository_diff (Path): Repository diff path.
        compactions (Path): Compactions JSONL path.
        result_manifest (Path): Result manifest JSON path.
    """

    base_dir: Path
    manifest_resolved: Path = None  # type: ignore[assignment]
    messages: Path = None  # type: ignore[assignment]
    token_usage_calls: Path = None  # type: ignore[assignment]
    costs: Path = None  # type: ignore[assignment]
    timings: Path = None  # type: ignore[assignment]
    tool_metrics: Path = None  # type: ignore[assignment]
    source_context: Path = None  # type: ignore[assignment]
    outcome: Path = None  # type: ignore[assignment]
    repository_diff: Path = None  # type: ignore[assignment]
    compactions: Path = None  # type: ignore[assignment]
    result_manifest: Path = None  # type: ignore[assignment]

    def __init__(self, base_dir: Path, **data: Any) -> None:
        """Initialize paths derived from the base directory.

        Args:
            base_dir (Path): Base artifact directory.
        """
        super().__init__(base_dir=base_dir, **data)
        self.manifest_resolved = base_dir / "manifest.resolved.json"
        self.messages = base_dir / "messages.jsonl"
        self.token_usage_calls = base_dir / "token_usage_calls.jsonl"
        self.costs = base_dir / "costs.json"
        self.timings = base_dir / "timings.json"
        self.tool_metrics = base_dir / "tool_metrics.json"
        self.source_context = base_dir / "source_context.json"
        self.outcome = base_dir / "outcome.json"
        self.repository_diff = base_dir / "repository.diff"
        self.compactions = base_dir / "compactions.jsonl"
        self.result_manifest = base_dir / "result_manifest.json"

    def all_paths(self) -> list[Path]:
        """Return all artifact file paths except result_manifest.

        Returns:
            (list[Path]) All artifact paths except the result manifest.
        """
        return [
            self.manifest_resolved,
            self.messages,
            self.token_usage_calls,
            self.costs,
            self.timings,
            self.tool_metrics,
            self.source_context,
            self.outcome,
            self.repository_diff,
            self.compactions,
        ]


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file.

    Args:
        path (Path): File to hash.

    Returns:
        (str) Hex digest string.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    """Write data as JSON to a file.

    Args:
        path (Path): Output file path.
        data (Any): Data to serialize.
    """
    path.write_text(
        json.dumps(data, indent=2, default=str),
        encoding="utf-8",
    )


def _write_jsonl(path: Path, lines: list[str]) -> None:
    """Write JSON lines with a trailing newline when non-empty.

    Args:
        path (Path): Output file path.
        lines (list[str]): Serialized JSON lines.
    """
    path.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )


def _write_token_usage_calls(paths: ArtifactPaths, token_ledger: TokenLedger) -> None:
    """Write the token usage calls JSONL.

    Covers model, compaction, and routine calls.

    Args:
        paths (ArtifactPaths): Artifact file paths.
        token_ledger (TokenLedger): Token usage ledger.
    """
    lines = [json.dumps(entry_payload(entry)) for entry in token_ledger.entries]
    _write_jsonl(paths.token_usage_calls, lines)


def _write_costs(
    paths: ArtifactPaths,
    cost_calculator: CostCalculator,
    token_ledger: TokenLedger,
) -> CostReport:
    """Write the cost report JSON.

    Args:
        paths (ArtifactPaths): Artifact file paths.
        cost_calculator (CostCalculator): Cost calculator.
        token_ledger (TokenLedger): Token usage ledger.

    Returns:
        (CostReport) The computed report.
    """
    cost_report = cost_calculator.compute_report(token_ledger)
    _write_json(
        paths.costs,
        {
            "total_cost": cost_report.total_cost,
            "model_cost": cost_report.model_cost,
            "compaction_cost": cost_report.compaction_cost,
            "is_complete": cost_report.is_complete,
            "call_costs": [c.model_dump() for c in cost_report.call_costs],
        },
    )
    return cost_report


def _write_timings(paths: ArtifactPaths, timings: Timings) -> None:
    """Write the timings JSON, including the metadata block.

    Args:
        paths (ArtifactPaths): Artifact file paths.
        timings (Timings): Wall-clock timing data.
    """
    timings_data = timings.as_dict()
    timings_data.update(timings.metadata_block())
    _write_json(paths.timings, timings_data)


def _write_compactions(paths: ArtifactPaths, session_events: SessionEvents) -> None:
    """Write the compaction events JSONL.

    Args:
        paths (ArtifactPaths): Artifact file paths.
        session_events (SessionEvents): Recorded session events.
    """
    lines = [event.model_dump_json() for event in session_events.compactions]
    _write_jsonl(paths.compactions, lines)


def _write_source_context(paths: ArtifactPaths, source_context: dict[str, Any]) -> None:
    """Write the source context JSON.

    Args:
        paths (ArtifactPaths): Artifact file paths.
        source_context (dict[str, Any]): Fixed session inputs.
    """
    _write_json(paths.source_context, source_context)


def _write_result_manifest(paths: ArtifactPaths) -> None:
    """Write the result manifest with artifact hashes.

    Written last so every other artifact is hashed.

    Args:
        paths (ArtifactPaths): Artifact file paths.
    """
    artifact_hashes: dict[str, str] = {}
    for path in paths.all_paths():
        if path.exists():
            artifact_hashes[path.name] = _sha256_file(path)

    _write_json(
        paths.result_manifest,
        {
            "finalization_status": "complete",
            "artifact_hashes": artifact_hashes,
        },
    )


class SessionArtifacts:
    """Writes the complete artifact directory.

    Instance Attributes:
        _paths (ArtifactPaths): Artifact file paths.
    """

    def __init__(self, base_dir: Path) -> None:
        """Initialize the artifact writer.

        Args:
            base_dir (Path): Base artifact directory.
        """
        self._paths = ArtifactPaths(base_dir=base_dir)

    @property
    def paths(self) -> ArtifactPaths:
        """Return the artifact paths."""
        return self._paths

    def write_artifacts(
        self,
        *,
        config: AgentConfig,
        timings: Timings,
        tool_metrics: ToolMetrics,
        token_ledger: TokenLedger,
        cost_calculator: CostCalculator,
        outcome: dict[str, Any],
        manifest_resolved: dict[str, Any],
        messages_jsonl: str,
        diff_text: str,
        session_events: SessionEvents,
        source_context: dict[str, Any],
    ) -> None:
        """Write all session artifacts to the artifact directory.

        Writes all files first, then writes the result manifest last
        with artifact hashes and finalization status.

        Args:
            config (AgentConfig): Agent configuration.
            timings (Timings): Wall-clock timing data.
            tool_metrics (ToolMetrics): Per-tool invocation metrics.
            token_ledger (TokenLedger): Token usage ledger.
            cost_calculator (CostCalculator): Cost calculator.
            outcome (dict[str, Any]): Session outcome.
            manifest_resolved (dict[str, Any]): Resolved manifest.
            messages_jsonl (str): Messages as JSONL text.
            diff_text (str): Repository diff text.
            session_events (SessionEvents): Recorded session events.
            source_context (dict[str, Any]): Fixed session inputs.
        """
        paths = self._paths
        paths.base_dir.mkdir(parents=True, exist_ok=True)

        # manifest.resolved.json
        _write_json(paths.manifest_resolved, manifest_resolved)

        # messages.jsonl
        paths.messages.write_text(messages_jsonl, encoding="utf-8")

        # token_usage_calls.jsonl
        _write_token_usage_calls(paths, token_ledger)

        # costs.json
        _write_costs(paths, cost_calculator, token_ledger)

        # timings.json
        _write_timings(paths, timings)

        # tool_metrics.json
        _write_json(paths.tool_metrics, tool_metrics.summary())

        # outcome.json
        _write_json(paths.outcome, outcome)

        # repository.diff
        paths.repository_diff.write_text(diff_text, encoding="utf-8")

        # compactions.jsonl and source_context.json
        _write_compactions(paths, session_events)
        _write_source_context(paths, source_context)

        # Result manifest last, so every other artifact is hashed
        _write_result_manifest(paths)
