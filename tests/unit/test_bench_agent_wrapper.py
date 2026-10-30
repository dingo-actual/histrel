#!/usr/bin/env python3
"""Unit tests for the benchmark agent wrapper log export.

Tests that the agent's /tmp log file is copied into the session
artifacts directory and that a missing log file is handled gracefully.
"""

from __future__ import annotations

from benchmark.predict.agents.bench_agent_wrapper import BenchAgentWrapper


class Test04Unit06Wrapper01LogExport:
    """Export of the agent log file into the artifacts directory."""

    def test_04_unit_06_wrapper_01_export_01_copies_log_to_artifacts(
        self, tmp_path
    ) -> None:
        # Arrange
        wrapper = BenchAgentWrapper(tmp_path / "histrel.toml")
        log_source = tmp_path / "bench_agent.log"
        log_source.write_text('{"record": "one"}\n', encoding="utf-8")
        artifact_dir = tmp_path / "artifacts"

        # Act
        wrapper._export_agent_log(
            log_source, artifact_dir, session_id="sess-1", agent_type="bench-agent"
        )

        # Assert
        exported = artifact_dir / "bench_agent.log"
        assert exported.is_file()
        assert exported.read_text(encoding="utf-8") == '{"record": "one"}\n'
        # The source file is copied, not moved
        assert log_source.is_file()

    def test_04_unit_06_wrapper_01_export_02_missing_log_is_not_an_error(
        self, tmp_path
    ) -> None:
        # Arrange
        wrapper = BenchAgentWrapper(tmp_path / "histrel.toml")
        missing_source = tmp_path / "nope.log"
        artifact_dir = tmp_path / "artifacts"

        # Act — must not raise
        wrapper._export_agent_log(
            missing_source, artifact_dir, session_id="sess-1", agent_type="bench-agent"
        )

        # Assert — nothing was exported
        assert not (artifact_dir / "bench_agent.log").exists()
