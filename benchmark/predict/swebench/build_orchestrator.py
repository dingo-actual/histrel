#!/usr/bin/env python3
"""Build orchestrator for SWE-bench image build pipeline integration.

Coordinates SWE-bench image builds with agent runtime injection, managing
base → env → instance build sequence with proper caching and context staging.

    Important Classes:
        BuildOrchestrator: Coordinates SWE-bench image builds with agent runtime injection
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
import time

import docker
import docker.errors
from loguru import logger

from swebench.harness.docker_build import (
    BuildImageError,
    build_base_images,
    build_image,
    build_instance_image,
)
from swebench.harness.docker_utils import remove_image
from swebench.harness.test_spec.test_spec import TestSpec

from benchmark.predict.swebench.env_builder import EnvBuildContextManager

MAX_BUILD_RETRIES: int = 3
BUILD_RETRY_BASE_DELAY: float = 5.0


class BuildOrchestrator:
    """Coordinates SWE-bench image builds with agent runtime injection.

    This orchestrator manages the three-stage build pipeline:
    1. Base images - Pure delegation to SWE-bench
    2. Environment images - Build context staging and Dockerfile override
    3. Instance images - Pure delegation to SWE-bench

    Attributes:
        client: Docker client for image operations
        histrel_repo_path: Path to Histrel repository root
    """

    def __init__(
        self,
        client: docker.DockerClient,
        histrel_repo_path: Path,
    ) -> None:
        """Initialize BuildOrchestrator with Docker client and paths.

        Args:
            client: Docker client for image operations
            histrel_repo_path: Path to Histrel repository root
        """
        self.client = client
        self.histrel_repo_path = histrel_repo_path

        self.env_builder: EnvBuildContextManager = EnvBuildContextManager(
            histrel_repo_path
        )

        logger.bind(
            component="benchmark.predict.swebench.build_orchestrator",
            origin="bench_harness_predict",
        ).info(
            "Initialized BuildOrchestrator",
            histrel_repo=str(histrel_repo_path),
        )

    def _retry_build(
        self,
        build_fn: Callable,
        build_fn_args: tuple,
        build_fn_kwargs: dict,
        image_key: str,
    ) -> None:
        """Retry a Docker image build on transient BuildImageError.

        Args:
            build_fn: The build function to call (build_image or build_instance_image)
            build_fn_args: Positional arguments for build_fn
            build_fn_kwargs: Keyword arguments for build_fn
            image_key: Image name for logging

        Raises:
            BuildImageError: If all retries are exhausted.
        """
        last_exc: BuildImageError | None = None
        for attempt in range(MAX_BUILD_RETRIES):
            try:
                build_fn(*build_fn_args, **build_fn_kwargs)
                return
            except BuildImageError as exc:
                last_exc = exc
                # Remove the partially tagged image the legacy builder
                # leaves behind so the retry starts clean
                remove_image(self.client, image_key, logger)
                if attempt < MAX_BUILD_RETRIES - 1:
                    delay = BUILD_RETRY_BASE_DELAY * (2**attempt)
                    logger.bind(
                        component="benchmark.predict.swebench.build_orchestrator",
                        image_key=image_key,
                        attempt=attempt + 1,
                        max_retries=MAX_BUILD_RETRIES,
                        delay=delay,
                        origin="bench_harness_predict",
                    ).warning(
                        "Build failed, retrying",
                        error=str(exc),
                    )
                    time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def build_images(
        self,
        test_specs: Sequence[TestSpec],
        force_rebuild_env: bool = False,
        force_rebuild_instances: bool = False,
    ) -> None:
        """Build all required images for test specs.

        Executes the three-stage build pipeline: base → env → instance.
        Images are built in order, with caching checks to skip unchanged builds.
        Instance images are rebuilt when their parent env image changes.

        Args:
            test_specs: List of TestSpec objects with injected agent runtime
            force_rebuild_env: Force rebuild environment images even if cached
            force_rebuild_instances: Force rebuild instance images even if cached

        Raises:
            RuntimeError: If any critical build failure occurs
        """
        logger.bind(
            component="benchmark.predict.swebench.build_orchestrator",
            num_specs=len(test_specs),
            origin="bench_harness_predict",
        ).info("Starting multi-stage image build pipeline")

        self._prune_dangling_images("pipeline_start")

        try:
            self._build_base_images(test_specs)
            rebuilt_env_keys = self._build_env_images(test_specs, force_rebuild_env)
            self._build_instance_images(
                test_specs, force_rebuild_instances, rebuilt_env_keys
            )
        except Exception as exc:
            logger.bind(
                component="benchmark.predict.swebench.build_orchestrator",
                origin="bench_harness_predict",
            ).error(
                "Build pipeline failed",
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise
        finally:
            self._prune_dangling_images("pipeline_end")

        logger.bind(
            component="benchmark.predict.swebench.build_orchestrator",
            origin="bench_harness_predict",
        ).info("Build pipeline completed successfully")

    def _prune_dangling_images(self, phase: str) -> None:
        """Prune untagged images left behind by the legacy builder.

        Best-effort: cleanup failures are logged, never raised, so they
        cannot break the build pipeline.

        Args:
            phase: Pipeline phase marker for log context
        """
        try:
            result: dict[str, Any] = self.client.api.prune_images({"dangling": True})
        except docker.errors.DockerException as exc:
            logger.bind(
                component="benchmark.predict.swebench.build_orchestrator",
                phase=phase,
                origin="bench_harness_predict",
            ).warning("Dangling image prune failed", error=str(exc))
            return None

        logger.bind(
            component="benchmark.predict.swebench.build_orchestrator",
            phase=phase,
            num_deleted=len(result.get("ImagesDeleted") or []),
            space_reclaimed_bytes=result.get("SpaceReclaimed") or 0,
            origin="bench_harness_predict",
        ).info("Pruned dangling images")
        return None

    def _build_base_images(self, test_specs: Sequence[TestSpec]) -> None:
        """Build base images via pure delegation to SWE-bench.

        Base images are standard SWE-bench infrastructure and require
        no customization. This method delegates entirely to SWE-bench's
        build_base_images function.

        Args:
            test_specs: List of TestSpec objects (base images use raw specs)
        """

        logger.bind(
            component="benchmark.predict.swebench.build_orchestrator",
            num_specs=len(test_specs),
            origin="bench_harness_predict",
        ).info("Building base images")

        build_base_images(self.client, list(test_specs), force_rebuild=False)

        logger.bind(
            component="benchmark.predict.swebench.build_orchestrator",
            origin="bench_harness_predict",
        ).info("Base images build completed")

    def _build_env_images(
        self, test_specs: Sequence[TestSpec], force_rebuild: bool
    ) -> set[str]:
        """Build environment images with agent runtime injection.

        Environment images contain heavy dependencies that are shared
        across instances. This method handles build context
        staging and Dockerfile override to inject Histrel requirements.

        Args:
            test_specs: List of TestSpecAdapter objects with env_script_list
            force_rebuild: Force rebuild even if cached

        Returns:
            Set of env image keys that were actually built (not found in cache).
            This is used to determine which instance images need rebuilding.
        """

        logger.bind(
            component="benchmark.predict.swebench.build_orchestrator",
            num_specs=len(test_specs),
            origin="bench_harness_predict",
        ).info("Building environment images")

        unique_envs: dict[str, TestSpec] = {}
        for spec in test_specs:
            env_key = getattr(spec, "env_image_key", None)
            if env_key is None:
                env_key = f"sweb.env.{spec.arch}.{spec.version}:latest"
            if env_key not in unique_envs:
                unique_envs[env_key] = spec

        rebuilt_env_keys: set[str] = set()

        for env_key, spec in unique_envs.items():
            logger.bind(
                component="benchmark.predict.swebench.build_orchestrator",
                env_key=env_key,
                origin="bench_harness_predict",
            ).info("Processing environment image", force_rebuild=force_rebuild)

            if not force_rebuild and self._image_exists(env_key):
                logger.bind(
                    component="benchmark.predict.swebench.build_orchestrator",
                    env_key=env_key,
                    origin="bench_harness_predict",
                ).info("Environment image already exists, skipping build")
                continue

            if force_rebuild:
                logger.bind(
                    component="benchmark.predict.swebench.build_orchestrator",
                    env_key=env_key,
                    origin="bench_harness_predict",
                ).info("Force rebuilding environment image")

                # Remove the old env image - don't suppress errors
                remove_image(self.client, env_key, logger)

                # Wait for Docker daemon to sync - retry up to 10 times (1 second total)
                # Docker has eventual consistency; image may still appear to exist briefly
                max_retries = 10
                retry_delay = 0.1  # 100ms
                for attempt in range(max_retries):
                    if not self._image_exists(env_key):
                        logger.bind(
                            component="benchmark.predict.swebench.build_orchestrator",
                            env_key=env_key,
                            attempts=attempt + 1,
                            origin="bench_harness_predict",
                        ).info("Environment image confirmed removed")
                        break
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                else:
                    # All retries exhausted
                    raise RuntimeError(
                        f"Failed to remove environment image {env_key}. "
                        f"The image may be in use by a running container. "
                        f"Please stop any containers using this image and retry."
                    )

            build_dir = self.env_builder.stage_build_context(env_key)
            dockerfile = self.env_builder.override_env_dockerfile(
                getattr(spec, "env_dockerfile", "FROM base\nCOPY ./setup_env.sh /root/")
            )

            self._retry_build(
                build_fn=build_image,
                build_fn_args=(),
                build_fn_kwargs={
                    "image_name": env_key,
                    "setup_scripts": {
                        "setup_env.sh": getattr(spec, "setup_env_script", "")
                    },
                    "dockerfile": dockerfile,
                    "platform": spec.arch,
                    "client": self.client,
                    "build_dir": build_dir,
                },
                image_key=env_key,
            )

            rebuilt_env_keys.add(env_key)
            logger.bind(
                component="benchmark.predict.swebench.build_orchestrator",
                env_key=env_key,
                origin="bench_harness_predict",
            ).info("Environment image built successfully")

        logger.bind(
            component="benchmark.predict.swebench.build_orchestrator",
            num_images=len(unique_envs),
            num_rebuilt=len(rebuilt_env_keys),
            origin="bench_harness_predict",
        ).info("Environment images build completed")

        return rebuilt_env_keys

    def _build_instance_images(
        self,
        test_specs: Sequence[TestSpec],
        force_rebuild: bool,
        rebuilt_env_keys: set[str],
    ) -> None:
        """Build instance images via pure delegation to SWE-bench.

        Instance images contain repository checkouts and test runners.
        They inherit from environment images and are rebuilt when their
        parent env image changes to ensure consistency.

        Args:
            test_specs: List of TestSpecAdapter objects with repo_script_list
            force_rebuild: Force rebuild even if cached
            rebuilt_env_keys: Set of env image keys that were rebuilt.
                Instance images depending on these env images will be rebuilt.
        """

        logger.bind(
            component="benchmark.predict.swebench.build_orchestrator",
            num_specs=len(test_specs),
            rebuilt_env_count=len(rebuilt_env_keys),
            origin="bench_harness_predict",
        ).info("Building instance images")

        for spec in test_specs:
            instance_key = getattr(spec, "instance_image_key", None)
            if instance_key is None:
                instance_key = f"sweb.eval.{spec.arch}.{spec.instance_id}:latest"

            env_key = getattr(spec, "env_image_key", None)
            if env_key is None:
                env_key = f"sweb.env.{spec.arch}.{spec.version}:latest"

            # Check if parent env image was rebuilt - if so, must rebuild instance
            env_rebuilt = env_key in rebuilt_env_keys

            logger.bind(
                component="benchmark.predict.swebench.build_orchestrator",
                instance_id=spec.instance_id,
                instance_key=instance_key,
                env_key=env_key,
                env_rebuilt=env_rebuilt,
                origin="bench_harness_predict",
            ).info("Processing instance image", force_rebuild=force_rebuild)

            if (
                not force_rebuild
                and not env_rebuilt
                and self._image_exists(instance_key)
            ):
                logger.bind(
                    component="benchmark.predict.swebench.build_orchestrator",
                    instance_id=spec.instance_id,
                    origin="bench_harness_predict",
                ).info("Instance image already exists, skipping build")
                continue

            if force_rebuild or env_rebuilt:
                if env_rebuilt:
                    logger.bind(
                        component="benchmark.predict.swebench.build_orchestrator",
                        instance_id=spec.instance_id,
                        env_key=env_key,
                        origin="bench_harness_predict",
                    ).info("Rebuilding instance image due to env image change")
                if force_rebuild:
                    logger.bind(
                        component="benchmark.predict.swebench.build_orchestrator",
                        instance_id=spec.instance_id,
                        origin="bench_harness_predict",
                    ).info("Force rebuilding instance image")

                # Remove the old instance image - don't suppress errors
                remove_image(self.client, instance_key, logger)

                # Wait for Docker daemon to sync - retry up to 10 times (1 second total)
                # Docker has eventual consistency; image may still appear to exist briefly
                max_retries = 10
                retry_delay = 0.1  # 100ms
                for attempt in range(max_retries):
                    if not self._image_exists(instance_key):
                        logger.bind(
                            component="benchmark.predict.swebench.build_orchestrator",
                            instance_id=spec.instance_id,
                            attempts=attempt + 1,
                            origin="bench_harness_predict",
                        ).info("Instance image confirmed removed")
                        break
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                else:
                    # All retries exhausted
                    raise RuntimeError(
                        f"Failed to remove instance image {instance_key}. "
                        f"The image may be in use by a running container. "
                        f"Please stop any containers using this image and retry."
                    )

            self._retry_build(
                build_fn=build_instance_image,
                build_fn_args=(),
                build_fn_kwargs={
                    "test_spec": spec,
                    "client": self.client,
                    "logger": None,
                    "nocache": True,
                },
                image_key=instance_key,
            )

            logger.bind(
                component="benchmark.predict.swebench.build_orchestrator",
                instance_id=spec.instance_id,
                origin="bench_harness_predict",
            ).info("Instance image built successfully")

        logger.bind(
            component="benchmark.predict.swebench.build_orchestrator",
            num_images=len(test_specs),
            origin="bench_harness_predict",
        ).info("Instance images build completed")

    def _image_exists(self, image_name: str) -> bool:
        """Check if a Docker image exists.

        Args:
            image_name: Full image name with tag

        Returns:
            True if image exists, False otherwise
        """
        try:
            self.client.images.get(image_name)
            return True
        except Exception:
            return False
