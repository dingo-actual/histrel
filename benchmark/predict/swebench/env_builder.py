#!/usr/bin/env python3
"""Environment build context manager for SWE-bench integration.

Stages Histrel repository into SWE-bench build
contexts and overrides Dockerfiles to include required files.

    Important Classes:
        EnvBuildContextManager: Stages files and overrides Dockerfiles
"""

from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger


class EnvBuildContextManager:
    """Manages staging files into SWE-bench env build contexts.

    SWE-bench's default env Dockerfile only copies setup_env.sh, but we
    need to include Histrel repo. This manager
    handles staging and Dockerfile modification.

    Attributes:
        histrel_repo_path: Path to Histrel repository root
    """

    def __init__(
        self,
        histrel_repo_path: Path,
    ) -> None:
        """Initialize context manager with required paths.

        Args:
            histrel_repo_path: Path to Histrel repository root
        """
        self.histrel_repo_path = histrel_repo_path

        logger.bind(
            component="benchmark.predict.swebench.env_builder",
            origin="bench_harness_predict",
        ).info(
            "Initialized EnvBuildContextManager",
            histrel_repo=str(histrel_repo_path),
        )

    def stage_build_context(self, env_image_key: str) -> Path:
        """Stage all required files into env build context.

        Creates a build directory following SWE-bench conventions and
        copies all required files for env image build.

        Args:
            env_image_key: SWE-bench env image key (e.g., "sweb.env.py.x86_64.abc123")

        Returns:
            Path to build directory ready for docker build
        """
        from swebench.harness.constants import ENV_IMAGE_BUILD_DIR

        build_dir = ENV_IMAGE_BUILD_DIR / env_image_key.replace(":", "__")
        build_dir.mkdir(parents=True, exist_ok=True)

        logger.bind(
            component="benchmark.predict.swebench.env_builder",
            env_image_key=env_image_key,
            build_dir=str(build_dir),
            origin="bench_harness_predict",
        ).info("Staging build context")

        self._stage_histrel_repo(build_dir)
        self._stage_bench_settings(build_dir)

        logger.bind(
            component="benchmark.predict.swebench.env_builder",
            origin="bench_harness_predict",
        ).info("Build context staged successfully")

        return build_dir  # type: ignore[no-any-return]

    def _stage_histrel_repo(self, build_dir: Path) -> None:
        """Copy Histrel repo to build context.

        Args:
            build_dir: Build directory to copy into
        """
        dest = build_dir / "histrel"
        if dest.exists():
            shutil.rmtree(dest)

        logger.bind(
            component="benchmark.predict.swebench.env_builder",
            source=str(self.histrel_repo_path),
            dest=str(dest),
            origin="bench_harness_predict",
        ).info("Staging Histrel repository")

        shutil.copytree(
            self.histrel_repo_path,
            dest,
            ignore=shutil.ignore_patterns(
                ".git",
                "__pycache__",
                "*.pyc",
                ".pytest_cache",
                "vendor",
                "_build",
                ".hypothesis",
                ".mypy_cache",
                ".pytest_cache",
                ".ruff_cache",
                ".venv",
                ".venv_wsl",
                ".vscode",
                "assets",
                "data",
                "debug",
                "docs",
                "handbook",
                "logs",
                "node_modules",
                "reports",
                "results",
                "scratch",
                ".coveragerc",
                ".dockerignore",
                ".gitignore",
                ".mcp.json",
                ".prettierrc.json",
                "AGENTS.md",
                "*.lock",
                "CLAUDE.md",
                "eslint.config.mjs",
                "forge.yml",
                "*.txt",
                "mypy.ini",
                "*.tmp*",
                "nul",
                "*.bak*",
                "package-lock.json",
                "package.json",
                "pyrightconfig.json",
                "pytest.ini",
                "ty.toml",
                "usage*.json",
                "WARP.md",
            ),
        )

    def _stage_bench_settings(self, build_dir: Path) -> None:
        """Copy bench_settings.toml to build context.

        Args:
            build_dir: Build directory to copy into
        """
        toml_src = self.histrel_repo_path / "benchmark/bench_settings.toml"
        dest = build_dir / "bench_settings.toml"

        if toml_src.exists():
            logger.bind(
                component="benchmark.predict.swebench.env_builder",
                source=str(toml_src),
                dest=str(dest),
                origin="bench_harness_predict",
            ).info("Staging bench_settings.toml")

            shutil.copy2(toml_src, dest)
        else:
            logger.bind(
                component="benchmark.predict.swebench.env_builder",
                source=str(toml_src),
                origin="bench_harness_predict",
            ).warning("bench_settings.toml not found, skipping")

    def override_env_dockerfile(self, base_dockerfile: str) -> str:
        """Override env Dockerfile to copy staged files.

        Inserts COPY commands for Histrel repo before setup_env.sh execution.

        Args:
            base_dockerfile: Original SWE-bench env Dockerfile content

        Returns:
            Modified Dockerfile with COPY commands for Histrel files
        """
        logger.bind(
            component="benchmark.predict.swebench.env_builder",
            origin="bench_harness_predict",
        ).info("Overriding env Dockerfile")

        copy_commands = (
            "COPY ./histrel /root/histrel\n"
            "COPY ./bench_settings.toml /root/bench_settings.toml\n"
        )

        result = base_dockerfile.replace(
            "COPY ./setup_env.sh /root/",
            copy_commands + "COPY ./setup_env.sh /root/",
        )

        logger.bind(
            component="benchmark.predict.swebench.env_builder",
            origin="bench_harness_predict",
        ).info("Dockerfile override complete")

        return result
