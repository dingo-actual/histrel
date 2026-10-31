#!/usr/bin/env python3
"""
Git repository operations for benchmark agents.

    Important Classes:
        GitRepoManager: Clone, checkout, and diff extraction helpers.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger


#: Files the harness writes into the workspace after the base commit. They are
#: harness state, not model changes, so patch extraction always skips them.
HARNESS_GENERATED_PATHS = (
    "agent_config.toml",
    "bench_settings.toml",
    "benchmark/bench_settings.toml",
)


class GitRepoManager:
    """Git operations for repository validation and diff extraction."""

    def run(
        self, cmd: list[str], cwd: Path | None = None
    ) -> subprocess.CompletedProcess:
        """Run a git command and return the completed process."""
        return subprocess.run(
            cmd,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )

    def validate_repo(self, repo_dir: Path) -> None:
        """Validate that repository exists at expected path.

        Args:
            repo_dir: Path to repository (e.g. /testbed/django)

        Raises:
            RuntimeError: If repo_dir doesn't exist or isn't a git repo
        """
        logger.bind(
            component="benchmark.predict.agents.repo.git_repo",
            repo_dir=str(repo_dir),
        ).info("Validating pre-cloned repository")

        if not repo_dir.exists():
            raise RuntimeError(f"Repository directory not found: {repo_dir}")

        git_dir = repo_dir / ".git"
        if not git_dir.exists():
            raise RuntimeError(f"Directory is not a git repository: {repo_dir}")

        if not git_dir.is_dir():
            raise RuntimeError(f"Directory is not a git repository: {repo_dir}")

        logger.bind(
            component="benchmark.predict.agents.repo.git_repo",
            repo_dir=str(repo_dir),
        ).info("Repository validation successful")

    def extract_patch(self, repo_dir: Path, base_commit: str | None = None) -> str:
        """Extract the workspace changes relative to the base commit.

        The patch must describe the final workspace, not the difference
        between the working tree and the index: staged changes, new files,
        and commits made by the agent are all included. Everything is staged
        (ignored files stay out), then diffed against ``base_commit``, so the
        result applies cleanly to a fresh checkout of that commit with
        ``git apply``, the way SWE-bench evaluation applies a prediction.

        Args:
            repo_dir (Path): Path to the repository.
            base_commit (str | None): Commit the changes are measured from.
                Defaults to None, which measures from HEAD.

        Returns:
            (str) Unified patch, or "" when the workspace has no changes.
        """
        base = base_commit or "HEAD"
        excluded = [f":(exclude){path}" for path in HARNESS_GENERATED_PATHS]

        staged = self.run(["git", "add", "-A", "--", ".", *excluded], cwd=repo_dir)
        if staged.returncode != 0:
            logger.bind(
                component="benchmark.predict.agents.repo.git_repo",
                repo_dir=str(repo_dir),
                returncode=staged.returncode,
                stderr=staged.stderr,
                stderr_length=len(staged.stderr),
                stdout=staged.stdout,
                stdout_length=len(staged.stdout),
            ).error("Git add -A failed")
            return ""

        result = self.run(
            [
                "git",
                "diff",
                "--cached",
                "--unified=3",
                "--no-color",
                base,
                "--",
                ".",
                *excluded,
            ],
            cwd=repo_dir,
        )
        if result.returncode != 0:
            logger.bind(
                component="benchmark.predict.agents.repo.git_repo",
                repo_dir=str(repo_dir),
                base_commit=base,
                returncode=result.returncode,
                stderr=result.stderr,
                stderr_length=len(result.stderr),
                stdout=result.stdout,
                stdout_length=len(result.stdout),
            ).error("Git diff failed")
            return ""

        patch_text = str(result.stdout)
        if not patch_text.strip():
            # No changes from the base commit - a legitimate empty patch
            logger.bind(
                component="benchmark.predict.agents.repo.git_repo",
                repo_dir=str(repo_dir),
                base_commit=base,
            ).info("No git changes detected - empty patch")
            return ""

        patch_lines = [
            line for line in patch_text.splitlines() if not line.startswith("index ")
        ]
        if not patch_lines:
            logger.bind(
                component="benchmark.predict.agents.repo.git_repo",
                repo_dir=str(repo_dir),
                patch_text_length=len(patch_text),
            ).warning("Patch text contained only index lines")
            return ""

        trailing_newline = "\n" if patch_text.endswith("\n") else ""
        return "\n".join(patch_lines) + trailing_newline
