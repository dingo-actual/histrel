#!/usr/bin/env python3
"""Unit tests for session artifact writing.

Tests the artifact files written by SessionArtifacts using real
filesystem operations in a per-test temporary directory (no mocks).
"""

from __future__ import annotations

import json

from histrel.context.compactor import CompactedState
from histrel.context.session_events import SessionEvents
from histrel.context.timings import Timings
from histrel.context.token_ledger import TokenLedger
from histrel.model.types import Usage
from tests.unit.context.helpers import ArtifactHarness
from tests.unit.factories import build_agent_config


class Test04Unit04Artifacts02WriteArtifacts:
    """Artifact directory creation and core artifact files."""

    def test_04_unit_04_artifacts_02_write_01_creates_directory(
        self, artifact_harness: ArtifactHarness
    ) -> None:
        # Act
        artifact_harness.write()

        # Assert
        assert artifact_harness.base_dir.exists()
        assert artifact_harness.base_dir.is_dir()

    def test_04_unit_04_artifacts_02_write_02_writes_costs_json(
        self, artifact_harness: ArtifactHarness
    ) -> None:
        # Arrange
        ledger = TokenLedger()
        ledger.record(
            message_index=0,
            usage=Usage(input=10000, output=5000, total=15000),
            provider_request_id="r1",
            call_kind="model",
        )

        # Act
        artifact_harness.write(ledger=ledger)

        # Assert
        assert artifact_harness.exists("costs.json")
        costs = artifact_harness.read_json("costs.json")
        assert costs["total_cost"] > 0
        assert costs["is_complete"] is True

    def test_04_unit_04_artifacts_02_write_03_writes_timings_json(
        self, artifact_harness: ArtifactHarness
    ) -> None:
        # Arrange
        timings = Timings()
        timings.add("model_wall_seconds", 5.0)
        timings.add("session_wall_seconds", 30.0)

        # Act
        artifact_harness.write(timings=timings)

        # Assert
        assert artifact_harness.exists("timings.json")
        data = artifact_harness.read_json("timings.json")
        assert data["model_wall_seconds"] == 5.0
        assert data["session_wall_seconds"] == 30.0

    def test_04_unit_04_artifacts_02_write_04_writes_outcome_json(
        self, artifact_harness: ArtifactHarness
    ) -> None:
        # Arrange
        outcome = {"status": "completed", "reason": "change_applied"}

        # Act
        artifact_harness.write(outcome=outcome)

        # Assert
        assert artifact_harness.exists("outcome.json")
        data = artifact_harness.read_json("outcome.json")
        assert data["status"] == "completed"
        assert data["reason"] == "change_applied"

    def test_04_unit_04_artifacts_02_write_05_writes_result_manifest_last(
        self, artifact_harness: ArtifactHarness
    ) -> None:
        # Act
        artifact_harness.write(
            manifest_resolved={"resolved": True},
            messages_jsonl='{"test": true}\n',
            diff_text="diff --git a/file.py\n",
        )

        # Assert — result manifest exists and contains hashes
        assert artifact_harness.exists("result_manifest.json")
        manifest = artifact_harness.read_json("result_manifest.json")
        assert "finalization_status" in manifest
        assert manifest["finalization_status"] == "complete"
        assert "artifact_hashes" in manifest
        assert len(manifest["artifact_hashes"]) > 0


class Test04Unit04Artifacts02WriteArtifactsMessages:
    """Messages and repository diff artifacts."""

    def test_04_unit_04_artifacts_02_write_06_writes_messages_jsonl(
        self, artifact_harness: ArtifactHarness
    ) -> None:
        # Act
        artifact_harness.write(
            messages_jsonl='{"message_id":"m1"}\n{"message_id":"m2"}\n'
        )

        # Assert
        assert artifact_harness.exists("messages.jsonl")
        lines = artifact_harness.read_text("messages.jsonl").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["message_id"] == "m1"

    def test_04_unit_04_artifacts_02_write_07_writes_repository_diff(
        self, artifact_harness: ArtifactHarness
    ) -> None:
        # Arrange
        diff = "diff --git a/src/main.py\n+print('hello')\n"

        # Act
        artifact_harness.write(diff_text=diff)

        # Assert
        assert artifact_harness.exists("repository.diff")
        assert artifact_harness.read_text("repository.diff") == diff


class Test04Unit04Artifacts02WriteArtifactsTokenUsage:
    """Token usage call artifacts."""

    def test_04_unit_04_artifacts_02_write_08_writes_all_ledger_entries(
        self, artifact_harness: ArtifactHarness
    ) -> None:
        # Arrange — a ledger with a model entry and a routine entry
        ledger = TokenLedger()
        ledger.record(
            message_index=0,
            usage=Usage(input=1000, output=500, total=1500),
            provider_request_id="req-1",
            call_kind="model",
        )
        ledger.record(
            message_index=1,
            usage=Usage(input=2000, output=800, total=2800),
            provider_request_id="req-2",
            call_kind="routine",
        )

        # Act
        artifact_harness.write(ledger=ledger, config=build_agent_config())

        # Assert — every ledger entry is written in order
        assert artifact_harness.exists("token_usage_calls.jsonl")
        lines = (
            artifact_harness.read_text("token_usage_calls.jsonl").strip().split("\n")
        )
        assert len(lines) == 2
        assert [json.loads(line)["call_kind"] for line in lines] == [
            "model",
            "routine",
        ]


class Test04Unit04Artifacts02WriteArtifactsSessionEvents:
    """Compaction events and source context artifacts."""

    def test_04_unit_04_artifacts_02_write_09_writes_compaction_events(
        self, artifact_harness: ArtifactHarness
    ) -> None:
        # Arrange
        events = SessionEvents()
        events.record_compaction(
            CompactedState(
                system_prompt="system",
                original_task="task",
                recent_message_ids=("m2",),
                source_message_index_start=1,
                source_message_index_end=3,
                working_summary="Earlier work.",
            )
        )

        # Act
        artifact_harness.write(session_events=events)

        # Assert
        lines = artifact_harness.read_text("compactions.jsonl").strip().split("\n")
        assert len(lines) == 1
        event = json.loads(lines[0])
        assert event["compacted_message_count"] == 3
        assert event["retained_message_count"] == 1
        assert event["summary"] == "Earlier work."

    def test_04_unit_04_artifacts_02_write_10_writes_empty_compactions_without_events(
        self, artifact_harness: ArtifactHarness
    ) -> None:
        # Act
        artifact_harness.write()

        # Assert — no compaction happened, so the file exists and is empty
        assert artifact_harness.read_text("compactions.jsonl") == ""

    def test_04_unit_04_artifacts_02_write_11_writes_source_context(
        self, artifact_harness: ArtifactHarness
    ) -> None:
        # Arrange
        source_context = {"mode": "stateless", "tools": ["bash", "read"]}

        # Act
        artifact_harness.write(source_context=source_context)

        # Assert
        assert artifact_harness.read_json("source_context.json") == source_context

    def test_04_unit_04_artifacts_02_write_12_hashes_session_event_artifacts(
        self, artifact_harness: ArtifactHarness
    ) -> None:
        # Act
        artifact_harness.write(source_context={"mode": "stateless"})

        # Assert
        hashes = artifact_harness.read_json("result_manifest.json")["artifact_hashes"]
        assert "compactions.jsonl" in hashes
        assert "source_context.json" in hashes
        assert "test_results.jsonl" not in hashes
