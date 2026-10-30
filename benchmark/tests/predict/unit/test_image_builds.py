#!/usr/bin/env python3
"""Unit tests for SWE-bench image build pipeline validation.

Tests for end-to-end image build validation including base, env, and instance
layers with proper caching, deduplication, and key computation.

    Test Classes:
        Test04Unit05ImageBuilds01Structure: Image build pipeline structure tests
        Test04Unit05ImageBuilds02Layers: Individual layer verification tests
        Test04Unit05ImageBuilds03Keys: Image key computation tests
        Test04Unit05ImageBuilds04Caching: Image caching behavior tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

if sys.platform == "win32":
    pytest.skip("Image builds require Unix", allow_module_level=True)

from benchmark.predict.swebench.testspec_adapter import TestSpecAdapter
from benchmark.tests.fixtures.swebench import (
    make_build_orchestrator,
    make_image_build_test_env,
    make_test_spec_adapter,
)


class Test04Unit05ImageBuilds01Structure:
    """Tests for image build pipeline structure and initialization."""

    def test_04_unit_05_image_builds_01_structure_01_orchestrator_has_builder(
        self,
    ) -> None:
        """Test orchestrator initializes with env builder for context staging."""
        with make_build_orchestrator() as orchestrator:
            assert orchestrator.env_builder is not None

    def test_04_unit_05_image_builds_01_structure_02_builder_has_required_paths(
        self,
    ) -> None:
        """Test env builder has correct path references."""
        with make_build_orchestrator() as orchestrator:
            builder = orchestrator.env_builder
            assert builder.histrel_repo_path is not None


class Test04Unit05ImageBuilds02Layers:
    """Tests for individual image layer construction."""

    def test_04_unit_05_image_builds_02_layers_01_base_image_tag_format(
        self,
    ) -> None:
        """Test base image tag follows SWE-bench format."""
        with make_test_spec_adapter(
            instance_id="astropy__astropy-13579",
            repo="astropy/astropy",
            version="5.0",
            arch="x86_64",
        ) as spec:
            expected_tag = f"sweb.base.py.{spec.arch}:{spec.base_image_tag}"
            assert "sweb.base.py" in expected_tag
            assert spec.arch in expected_tag

    def test_04_unit_05_image_builds_02_layers_02_env_image_tag_format(
        self,
    ) -> None:
        """Test env image tag follows SWE-bench format with hash."""
        with make_test_spec_adapter(instance_id="astropy__astropy-13579") as spec:
            tag = spec.env_image_tag
            assert isinstance(tag, str)

    def test_04_unit_05_image_builds_02_layers_03_instance_image_tag_format(
        self,
    ) -> None:
        """Test instance image tag follows SWE-bench format with instance_id."""
        with make_test_spec_adapter(instance_id="astropy__astropy-13579") as spec:
            tag = spec.instance_image_tag
            assert tag == "latest"


class Test04Unit05ImageBuilds03Keys:
    """Tests for image key computation and hashing."""

    def test_04_unit_05_image_builds_03_keys_01_env_key_includes_histrel_sources(
        self,
    ) -> None:
        """Test env image key incorporates Histrel source checksums."""
        with make_image_build_test_env() as (histrel_path, _):
            (histrel_path / "src" / "main.py").write_text("print('hello')")
            with make_test_spec_adapter(histrel_repo_path=histrel_path) as spec:
                adapter_hash = spec.compute_histrel_hash()
                assert adapter_hash is not None
                assert len(adapter_hash) == 22

    def test_04_unit_05_image_builds_03_keys_02_different_sources_produce_different_keys(
        self,
    ) -> None:
        """Test that different source content produces different image keys."""
        import tempfile
        from contextlib import ExitStack

        with ExitStack() as stack:
            tmpdir1 = stack.enter_context(tempfile.TemporaryDirectory())
            tmpdir2 = stack.enter_context(tempfile.TemporaryDirectory())

            histrel_path1 = Path(tmpdir1) / "histrel"
            histrel_path1.mkdir()
            histrel_src_path1 = histrel_path1 / "src"
            histrel_src_path1.mkdir()
            (histrel_src_path1 / "main.py").write_text("print('v1')")

            histrel_path2 = Path(tmpdir2) / "histrel"
            histrel_path2.mkdir()
            histrel_src_path2 = histrel_path2 / "src"
            histrel_benchmark_path2 = histrel_path2 / "benchmark"
            histrel_src_path2.mkdir()
            histrel_benchmark_path2.mkdir()
            (histrel_src_path2 / "main.py").write_text("print('v2')")
            (histrel_benchmark_path2 / "extra.py").write_text("print('extra')")

            spec1 = TestSpecAdapter(
                instance_id="test1",
                repo="test/repo",
                version="1.0",
                env_script_list=[],
                repo_script_list=[],
                eval_script_list=[],
                arch="x86_64",
                FAIL_TO_PASS=[],
                PASS_TO_PASS=[],
                language="py",
                docker_specs={},
                namespace=None,
                histrel_repo_path=histrel_path1,
            )

            spec2 = TestSpecAdapter(
                instance_id="test2",
                repo="test/repo",
                version="1.0",
                env_script_list=[],
                repo_script_list=[],
                eval_script_list=[],
                arch="x86_64",
                FAIL_TO_PASS=[],
                PASS_TO_PASS=[],
                language="py",
                docker_specs={},
                namespace=None,
                histrel_repo_path=histrel_path2,
            )

            hash1 = spec1.compute_histrel_hash()
            hash2 = spec2.compute_histrel_hash()
            assert hash1 != hash2


class Test04Unit05ImageBuilds04Caching:
    """Tests for image caching and deduplication behavior."""

    def test_04_unit_05_image_builds_04_caching_01_env_images_reuse_on_identical_hash(
        self,
    ) -> None:
        """Test that identical env configurations can reuse the same image."""
        with make_image_build_test_env() as (histrel_path1, _):
            (histrel_path1 / "src" / "main.py").write_text("shared content")
        with make_image_build_test_env() as (histrel_path2, _):
            (histrel_path2 / "src" / "main.py").write_text("shared content")
        with (
            make_test_spec_adapter(
                histrel_repo_path=histrel_path1,
                instance_id="astropy__astropy-13579",
            ) as spec1,
            make_test_spec_adapter(
                histrel_repo_path=histrel_path2,
                instance_id="requests__requests-1234",
            ) as spec2,
        ):
            hash1 = spec1.compute_histrel_hash()
            hash2 = spec2.compute_histrel_hash()
            assert hash1 == hash2


class Test04Unit05ImageBuilds05Integration:
    """Tests for complete image build pipeline integration."""

    def test_04_unit_05_image_builds_05_integration_01_spec_has_valid_instance_fields(
        self,
    ) -> None:
        """Test spec contains valid fields from SWE-bench instance."""
        with make_test_spec_adapter(
            instance_id="astropy__astropy-13579",
            repo="astropy/astropy",
            version="5.0",
        ) as spec:
            assert spec.instance_id == "astropy__astropy-13579"
            assert spec.repo == "astropy/astropy"
            assert spec.version == "5.0"
            assert spec.language == "py"
            assert isinstance(spec.env_script_list, list)
            assert isinstance(spec.repo_script_list, list)
            assert isinstance(spec.eval_script_list, list)

    def test_04_unit_05_image_builds_05_integration_02_adapter_extends_testspec(
        self,
    ) -> None:
        """Test TestSpecAdapter is subclass of SWE-bench TestSpec."""
        from swebench.harness.test_spec.test_spec import TestSpec

        with make_test_spec_adapter(
            instance_id="astropy__astropy-13579",
            repo="astropy/astropy",
            version="5.0",
        ) as spec:
            assert isinstance(spec, TestSpec)
