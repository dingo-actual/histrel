#!/usr/bin/env python3
"""
Unit tests for environment build context manager.

Tests for EnvBuildContextManager that stages the Histrel repository into
SWE-bench build contexts.

    Test Classes:
        Test04Unit03Builder01Staging: File staging tests
        Test04Unit03Builder02Dockerfile: Dockerfile override tests
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path

import pytest

# Skip entire module on Windows before any imports that require swebench
if sys.platform == "win32":
    pytest.skip("Scorer requires Unix", allow_module_level=True)

from benchmark.predict.swebench.env_builder import EnvBuildContextManager


class Test04Unit03Builder01Staging:
    """Tests for file staging functionality."""

    def test_04_unit_03_builder_01_staging_01_initialization(self) -> None:
        """Test that EnvBuildContextManager initializes correctly."""

        with tempfile.TemporaryDirectory() as tmpdir:
            histrel_path = Path(tmpdir) / "histrel"

            manager = EnvBuildContextManager(histrel_path)

            assert manager.histrel_repo_path == histrel_path

    def test_04_unit_03_builder_01_staging_02_stages_histrel_repo(self) -> None:
        """Test that Histrel repository is staged to build directory."""

        with tempfile.TemporaryDirectory() as tmpdir:
            histrel_path = Path(tmpdir) / "histrel"

            histrel_path.mkdir()
            (histrel_path / "test_file.py").write_text("# test")

            manager = EnvBuildContextManager(histrel_path)

            with pytest.MonkeyPatch().context() as mp:
                mp.setattr(
                    "swebench.harness.constants.ENV_IMAGE_BUILD_DIR",
                    Path(tmpdir) / "builds",
                )
                build_dir = manager.stage_build_context("test-env-key")

                assert (build_dir / "histrel" / "test_file.py").exists()

    def test_04_unit_03_builder_01_staging_04_excludes_results_dir(
        self,
    ) -> None:
        """Benchmark prediction results are excluded from the build context.

        ``benchmark/results`` holds run artifacts the image never needs;
        staging it adds hundreds of megabytes to every env build context.
        """

        with tempfile.TemporaryDirectory() as tmpdir:
            histrel_path = Path(tmpdir) / "histrel"

            histrel_path.mkdir()
            results_dir = histrel_path / "benchmark" / "results"
            results_dir.mkdir(parents=True)
            (results_dir / "instances.jsonl").write_text("{}")
            (histrel_path / "test_file.py").write_text("# test")

            manager = EnvBuildContextManager(histrel_path)

            with pytest.MonkeyPatch().context() as mp:
                mp.setattr(
                    "swebench.harness.constants.ENV_IMAGE_BUILD_DIR",
                    Path(tmpdir) / "builds",
                )
                build_dir = manager.stage_build_context("test-env-key")

                assert not (build_dir / "histrel" / "benchmark" / "results").exists()
                assert (build_dir / "histrel" / "test_file.py").exists()


class Test04Unit03Builder02Dockerfile:
    """Tests for Dockerfile override functionality."""

    def test_04_unit_03_builder_02_dockerfile_01_adds_copy_commands(self) -> None:
        """Test that Dockerfile override adds required COPY commands."""

        with tempfile.TemporaryDirectory() as tmpdir:
            histrel_path = Path(tmpdir) / "histrel"
            histrel_path.mkdir()

            manager = EnvBuildContextManager(histrel_path)

            base_dockerfile = "FROM python:3.11-slim\nCOPY ./setup_env.sh /root/"
            result = manager.override_env_dockerfile(base_dockerfile)

            assert "COPY ./histrel /root/histrel" in result
            assert "COPY ./bench_settings.toml /root/bench_settings.toml" in result
            assert "COPY ./setup_env.sh /root/" in result

    def test_04_unit_03_builder_02_dockerfile_02_preserves_original_content(
        self,
    ) -> None:
        """Test that original Dockerfile content is preserved."""

        with tempfile.TemporaryDirectory() as tmpdir:
            histrel_path = Path(tmpdir) / "histrel"
            histrel_path.mkdir()

            manager = EnvBuildContextManager(histrel_path)

            base_dockerfile = (
                "FROM python:3.11-slim\nRUN apt-get update\nCOPY ./setup_env.sh /root/"
            )
            result = manager.override_env_dockerfile(base_dockerfile)

            assert "FROM python:3.11-slim" in result
            assert "RUN apt-get update" in result
            assert "COPY ./setup_env.sh /root/" in result
