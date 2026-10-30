#!/usr/bin/env python3
"""
Unit tests for SWE-bench BuildOrchestrator.

Tests for BuildOrchestrator class that coordinates SWE-bench image builds
with agent runtime injection across base → env → instance pipeline.

    Test Classes:
        Test04Unit03Orchestrator01Structure: Orchestrator initialization tests
        Test04Unit03Orchestrator02BaseImages: Base image building tests
        Test04Unit03Orchestrator03EnvImages: Environment image building tests
        Test04Unit03Orchestrator04InstanceImages: Instance image building tests
        Test04Unit03Orchestrator05Caching: Image caching behavior tests
        Test04Unit03Orchestrator09BuildCleanup: Partial-image cleanup tests
        Test04Unit03Orchestrator10DanglingPrune: Dangling-image prune tests
"""

from __future__ import annotations

import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace

import docker
import docker.errors
import pytest
from typing import cast

# Skip entire module on Windows before any imports that require swebench
if sys.platform == "win32":
    pytest.skip("Scorer requires Unix", allow_module_level=True)

from benchmark.predict.swebench.build_orchestrator import (
    MAX_BUILD_RETRIES,
    BuildImageError,
    BuildOrchestrator,
)
from benchmark.tests.fixtures.docker_stubs import DockerClientStub
from benchmark.tests.fixtures.swebench import (
    make_build_orchestrator,
    make_test_spec_adapter,
)


def _make_stub_logger() -> SimpleNamespace:
    """Create a stub logger with the attributes BuildImageError requires."""
    return SimpleNamespace(log_file="build_image.log")


class Test04Unit03Orchestrator01Structure:
    """Tests for BuildOrchestrator initialization and structure."""

    def test_04_unit_03_orchestrator_01_structure_01_initialization(
        self,
    ) -> None:
        """Test orchestrator initializes with correct paths and components."""

        with make_build_orchestrator() as orchestrator:
            assert orchestrator.client is not None
            assert orchestrator.histrel_repo_path is not None
            assert orchestrator.env_builder is not None

    def test_04_unit_03_orchestrator_01_structure_02_path_resolution(
        self,
    ) -> None:
        """Test orchestrator resolves paths correctly."""

        with tempfile.TemporaryDirectory() as temp_dir:
            histrel_path = Path(temp_dir) / "histrel"
            histrel_path.mkdir()

            with make_build_orchestrator(
                histrel_repo_path=histrel_path
            ) as orchestrator:
                assert orchestrator.histrel_repo_path == histrel_path


class Test04Unit03Orchestrator02BaseImages:
    """Tests for base image building delegation."""

    def test_04_unit_03_orchestrator_02_base_images_01_build_delegation(
        self,
    ) -> None:
        """Test that base image building delegates to SWE-bench function."""

        with (
            make_test_spec_adapter(instance_id="base-test-1") as spec,
            make_build_orchestrator() as orchestrator,
        ):
            orchestrator._build_base_images([spec])


class Test04Unit03Orchestrator03EnvImages:
    """Tests for environment image building with context staging."""

    def test_04_unit_03_orchestrator_03_env_images_01_stages_context(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that env images trigger context staging."""

        with (
            make_test_spec_adapter(instance_id="env-test-1") as spec,
            make_build_orchestrator() as orchestrator,
        ):
            monkeypatch.setattr(
                "benchmark.predict.swebench.build_orchestrator.build_image",
                lambda *args, **kwargs: None,
            )

            monkeypatch.setattr(
                orchestrator.client.images,
                "get",
                lambda x: (_ for _ in ()).throw(Exception("not found")),
            )

            orchestrator._build_env_images([spec], force_rebuild=False)


class Test04Unit03Orchestrator04InstanceImages:
    """Tests for instance image building."""

    def test_04_unit_03_orchestrator_04_instance_images_01_build_delegation(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that instance image building delegates to SWE-bench function."""

        with (
            make_test_spec_adapter(instance_id="instance-test-1") as spec,
            make_build_orchestrator() as orchestrator,
        ):
            monkeypatch.setattr(
                "benchmark.predict.swebench.build_orchestrator.build_instance_image",
                lambda *args, **kwargs: None,
            )

            mock_image = type("MockImage", (), {"id": "sha256:abc123"})()
            monkeypatch.setattr(
                orchestrator.client.images,
                "get",
                lambda x: mock_image,
            )

            orchestrator._build_instance_images(
                [spec], force_rebuild=False, rebuilt_env_keys=set()
            )


class Test04Unit03Orchestrator05Caching:
    """Tests for image caching behavior."""

    def test_04_unit_03_orchestrator_05_caching_01_image_exists_check(
        self,
    ) -> None:
        """Test that image existence check returns correct boolean."""

        with make_build_orchestrator() as orchestrator:
            assert isinstance(orchestrator._image_exists("nonexistent:latest"), bool)

    def test_04_unit_03_orchestrator_05_caching_02_skips_cached_images(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that cached images are skipped during build."""

        with (
            make_test_spec_adapter(instance_id="cache-test-1") as spec,
            make_build_orchestrator() as orchestrator,
        ):
            monkeypatch.setattr(
                "benchmark.predict.swebench.build_orchestrator.build_image",
                lambda *args, **kwargs: None,
            )

            monkeypatch.setattr(orchestrator, "_image_exists", lambda x: True)

            orchestrator._build_env_images([spec], force_rebuild=False)


class Test04Unit03Orchestrator06Integration:
    """Tests for full pipeline integration."""

    def test_04_unit_03_orchestrator_06_integration_01_build_images_entrypoint(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that build_images is callable with test specs."""

        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.build_image",
            lambda *args, **kwargs: None,
        )

        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.build_instance_image",
            lambda *args, **kwargs: None,
        )

        with (
            make_test_spec_adapter(instance_id="integration-test-1") as spec,
            make_build_orchestrator() as orchestrator,
        ):
            # No image exists, so both build stages run end to end
            orchestrator.build_images(
                [spec], force_rebuild_env=True, force_rebuild_instances=True
            )


class Test04Unit03Orchestrator07ForceRebuildInstances:
    """Tests for force_rebuild_instances wiring.

    When ``force_rebuild_instances=True``, the orchestrator must call
    ``remove_image`` and rebuild even if the instance image already
    exists.  This verifies the ``--rebuild`` flag reaches the instance
    build path (previously only ``force_rebuild_env`` was wired).
    """

    def test_04_unit_03_orchestrator_07_force_rebuild_01_removes_existing(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """force_rebuild_instances=True removes an existing instance image."""

        removed_keys: list[str] = []
        image_exists_state: dict[str, bool] = {"exists": True}

        def _fake_remove_image(client, key, log) -> None:
            removed_keys.append(key)
            image_exists_state["exists"] = False

        def _fake_image_exists(key: str) -> bool:
            return image_exists_state["exists"]

        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.remove_image",
            _fake_remove_image,
        )
        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.build_image",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.build_instance_image",
            lambda *args, **kwargs: None,
        )

        with (
            make_test_spec_adapter(instance_id="force-rebuild-test-1") as spec,
            make_build_orchestrator() as orchestrator,
        ):
            # Simulate the image existing; remove_image flips it to False
            monkeypatch.setattr(orchestrator, "_image_exists", _fake_image_exists)

            orchestrator._build_instance_images(
                [spec], force_rebuild=True, rebuilt_env_keys=set()
            )

        assert len(removed_keys) == 1
        assert "force-rebuild-test-1" in removed_keys[0]

    def test_04_unit_03_orchestrator_07_force_rebuild_02_skips_when_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """force_rebuild_instances=False skips existing instance images."""

        removed_keys: list[str] = []

        def _fake_remove_image(client, key, log) -> None:
            removed_keys.append(key)

        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.remove_image",
            _fake_remove_image,
        )
        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.build_image",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.build_instance_image",
            lambda *args, **kwargs: None,
        )

        with (
            make_test_spec_adapter(instance_id="skip-rebuild-test-1") as spec,
            make_build_orchestrator() as orchestrator,
        ):
            # Simulate the image already existing
            monkeypatch.setattr(orchestrator, "_image_exists", lambda key: True)

            orchestrator._build_instance_images(
                [spec], force_rebuild=False, rebuilt_env_keys=set()
            )

        assert len(removed_keys) == 0


class Test04Unit03Orchestrator08RetryBuild:
    """Tests for transient build failure retry logic.

    The ``_retry_build`` helper retries on ``BuildImageError`` with
    exponential backoff to handle transient failures such as git clone
    network errors (exit code 128) inside ``setup_repo.sh``.
    """

    def test_04_unit_03_orchestrator_08_retry_01_succeeds_on_first_attempt(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_retry_build does not retry when the build succeeds immediately."""

        call_count: dict[str, int] = {"count": 0}

        def _succeeding_build(*args, **kwargs) -> None:
            call_count["count"] += 1

        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.time.sleep",
            lambda _: None,
        )

        with make_build_orchestrator() as orchestrator:
            orchestrator._retry_build(
                build_fn=_succeeding_build,
                build_fn_args=(),
                build_fn_kwargs={},
                image_key="test-image:latest",
            )

        assert call_count["count"] == 1

    def test_04_unit_03_orchestrator_08_retry_02_retries_on_build_image_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_retry_build retries on BuildImageError and succeeds on later attempt."""

        call_count: dict[str, int] = {"count": 0}

        def _flaky_build(*args, **kwargs) -> None:
            call_count["count"] += 1
            if call_count["count"] < 3:
                raise BuildImageError(
                    "test-image:latest",
                    "The command '/bin/sh -c /bin/bash /root/setup_repo.sh' "
                    "returned a non-zero code: 128",
                    _make_stub_logger(),
                )

        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.time.sleep",
            lambda _: None,
        )

        with make_build_orchestrator() as orchestrator:
            orchestrator._retry_build(
                build_fn=_flaky_build,
                build_fn_args=(),
                build_fn_kwargs={},
                image_key="test-image:latest",
            )

        assert call_count["count"] == 3

    def test_04_unit_03_orchestrator_08_retry_03_raises_after_max_retries(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_retry_build raises the last BuildImageError after exhausting retries."""

        call_count: dict[str, int] = {"count": 0}

        def _always_failing_build(*args, **kwargs) -> None:
            call_count["count"] += 1
            raise BuildImageError(
                "test-image:latest",
                "The command '/bin/sh -c /bin/bash /root/setup_repo.sh' "
                "returned a non-zero code: 128",
                _make_stub_logger(),
            )

        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.time.sleep",
            lambda _: None,
        )

        with (
            make_build_orchestrator() as orchestrator,
            pytest.raises(BuildImageError),
        ):
            orchestrator._retry_build(
                build_fn=_always_failing_build,
                build_fn_args=(),
                build_fn_kwargs={},
                image_key="test-image:latest",
            )

        assert call_count["count"] == MAX_BUILD_RETRIES

    def test_04_unit_03_orchestrator_08_retry_04_does_not_retry_non_build_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_retry_build does not retry exceptions other than BuildImageError."""

        call_count: dict[str, int] = {"count": 0}

        def _runtime_error_build(*args, **kwargs) -> None:
            call_count["count"] += 1
            raise RuntimeError("non-transient failure")

        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.time.sleep",
            lambda _: None,
        )

        with (
            make_build_orchestrator() as orchestrator,
            pytest.raises(RuntimeError, match="non-transient failure"),
        ):
            orchestrator._retry_build(
                build_fn=_runtime_error_build,
                build_fn_args=(),
                build_fn_kwargs={},
                image_key="test-image:latest",
            )

        assert call_count["count"] == 1

    def test_04_unit_03_orchestrator_08_retry_05_instance_uses_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """_build_instance_images wraps build_instance_image with retry."""

        call_count: dict[str, int] = {"count": 0}

        def _flaky_instance_build(*args, **kwargs) -> None:
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise BuildImageError(
                    "sweb.eval.x86_64.retry-test-1:latest",
                    "exit code 128",
                    _make_stub_logger(),
                )

        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.build_instance_image",
            _flaky_instance_build,
        )
        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.time.sleep",
            lambda _: None,
        )

        with (
            make_test_spec_adapter(instance_id="retry-test-1") as spec,
            make_build_orchestrator() as orchestrator,
        ):
            monkeypatch.setattr(orchestrator, "_image_exists", lambda key: False)

            orchestrator._build_instance_images(
                [spec], force_rebuild=False, rebuilt_env_keys=set()
            )

        assert call_count["count"] == 2


class Test04Unit03Orchestrator09BuildCleanup:
    """Tests for partial-image cleanup inside the retry loop.

    The classic (legacy) Docker builder leaves a partially tagged image
    behind when a build fails. Retrying against the same tag without
    removing the partial image leaves the daemon in a dirty state, so
    ``_retry_build`` must remove the failed image after every failed
    attempt.
    """

    def test_04_unit_03_orchestrator_09_cleanup_01_removes_after_failed_attempt(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failed attempt removes the partial image before retrying."""

        removed_keys: list[str] = []
        call_count: dict[str, int] = {"count": 0}

        def _recording_remove_image(client, key, log) -> None:
            removed_keys.append(key)

        def _fail_once_then_succeed(*args, **kwargs) -> None:
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise BuildImageError(
                    "cleanup-test-1:latest", "exit code 1", _make_stub_logger()
                )

        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.remove_image",
            _recording_remove_image,
        )
        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.time.sleep",
            lambda _: None,
        )

        with make_build_orchestrator() as orchestrator:
            orchestrator._retry_build(
                build_fn=_fail_once_then_succeed,
                build_fn_args=(),
                build_fn_kwargs={},
                image_key="cleanup-test-1:latest",
            )

        assert removed_keys == ["cleanup-test-1:latest"]

    def test_04_unit_03_orchestrator_09_cleanup_02_no_removal_on_success(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A successful first attempt performs no partial-image removal."""

        removed_keys: list[str] = []

        def _recording_remove_image(client, key, log) -> None:
            removed_keys.append(key)

        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.remove_image",
            _recording_remove_image,
        )

        with make_build_orchestrator() as orchestrator:
            orchestrator._retry_build(
                build_fn=lambda *args, **kwargs: None,
                build_fn_args=(),
                build_fn_kwargs={},
                image_key="cleanup-test-2:latest",
            )

        assert removed_keys == []

    def test_04_unit_03_orchestrator_09_cleanup_03_removes_on_every_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every failed attempt removes the partial image, including the last."""

        removed_keys: list[str] = []

        def _recording_remove_image(client, key, log) -> None:
            removed_keys.append(key)

        def _always_failing_build(*args, **kwargs) -> None:
            raise BuildImageError(
                "cleanup-test-3:latest", "exit code 1", _make_stub_logger()
            )

        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.remove_image",
            _recording_remove_image,
        )
        monkeypatch.setattr(
            "benchmark.predict.swebench.build_orchestrator.time.sleep",
            lambda _: None,
        )

        with (
            make_build_orchestrator() as orchestrator,
            pytest.raises(BuildImageError),
        ):
            orchestrator._retry_build(
                build_fn=_always_failing_build,
                build_fn_args=(),
                build_fn_kwargs={},
                image_key="cleanup-test-3:latest",
            )

        assert removed_keys == ["cleanup-test-3:latest"] * MAX_BUILD_RETRIES


class Test04Unit03Orchestrator10DanglingPrune:
    """Tests for dangling-image pruning around the build pipeline.

    The classic builder leaves untagged intermediate images on every
    build. The pipeline prunes dangling-only images at start (self-heal
    from crashed runs) and on completion, best-effort so cleanup failures
    never break the pipeline.
    """

    def test_04_unit_03_orchestrator_10_prune_01_prunes_start_and_end(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """build_images prunes dangling images at pipeline start and end."""

        stub = DockerClientStub()
        monkeypatch.setattr(
            BuildOrchestrator, "_build_base_images", lambda self, specs: None
        )
        monkeypatch.setattr(
            BuildOrchestrator,
            "_build_env_images",
            lambda self, specs, force_rebuild: set(),
        )
        monkeypatch.setattr(
            BuildOrchestrator,
            "_build_instance_images",
            lambda self, specs, force_rebuild, rebuilt_env_keys: None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            histrel_path = Path(temp_dir) / "histrel"
            histrel_path.mkdir()

            orchestrator = BuildOrchestrator(
                cast(docker.DockerClient, stub), histrel_path
            )
            orchestrator.build_images([])

        assert stub.api.prune_images_calls == [
            {"dangling": True},
            {"dangling": True},
        ]

    def test_04_unit_03_orchestrator_10_prune_02_prune_error_is_nonfatal(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A failing prune never fails the build pipeline."""

        stub = DockerClientStub()
        stub.api.set_prune_error(docker.errors.DockerException("daemon offline"))
        monkeypatch.setattr(
            BuildOrchestrator, "_build_base_images", lambda self, specs: None
        )
        monkeypatch.setattr(
            BuildOrchestrator,
            "_build_env_images",
            lambda self, specs, force_rebuild: set(),
        )
        monkeypatch.setattr(
            BuildOrchestrator,
            "_build_instance_images",
            lambda self, specs, force_rebuild, rebuilt_env_keys: None,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            histrel_path = Path(temp_dir) / "histrel"
            histrel_path.mkdir()

            orchestrator = BuildOrchestrator(
                cast(docker.DockerClient, stub), histrel_path
            )
            orchestrator.build_images([])

        assert len(stub.api.prune_images_calls) == 2

    def test_04_unit_03_orchestrator_10_prune_03_prunes_after_stage_failure(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A stage failure still triggers the end-of-pipeline prune."""

        stub = DockerClientStub()
        monkeypatch.setattr(
            BuildOrchestrator, "_build_base_images", lambda self, specs: None
        )

        def _failing_env_stage(self, specs, force_rebuild: bool) -> set[str]:
            raise BuildImageError(
                "prune-test-3:latest", "exit code 1", _make_stub_logger()
            )

        monkeypatch.setattr(BuildOrchestrator, "_build_env_images", _failing_env_stage)

        with tempfile.TemporaryDirectory() as temp_dir:
            histrel_path = Path(temp_dir) / "histrel"
            histrel_path.mkdir()

            orchestrator = BuildOrchestrator(
                cast(docker.DockerClient, stub), histrel_path
            )
            with pytest.raises(BuildImageError):
                orchestrator.build_images([])

        assert stub.api.prune_images_calls == [
            {"dangling": True},
            {"dangling": True},
        ]
