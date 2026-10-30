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

    def extract_patch(self, repo_dir: Path) -> str:
        """Extract patch from git working tree changes."""
        # Check for modified files first
        result = self.run(["git", "diff", "--name-only"], cwd=repo_dir)
        if result.returncode != 0:
            logger.bind(
                component="benchmark.predict.agents.repo.git_repo",
                repo_dir=str(repo_dir),
                returncode=result.returncode,
                stderr=result.stderr,
                stderr_length=len(result.stderr),
                stdout=result.stdout,
                stdout_length=len(result.stdout),
            ).error("Git diff --name-only failed")
            return ""

        if not result.stdout.strip():
            # No changes made - this is a legitimate empty patch
            logger.bind(
                component="benchmark.predict.agents.repo.git_repo",
                repo_dir=str(repo_dir),
            ).info("No git changes detected - empty patch")
            return ""

        # Get the actual diff
        result = self.run(["git", "diff", "--unified=3", "--no-color"], cwd=repo_dir)
        if result.returncode != 0:
            logger.bind(
                component="benchmark.predict.agents.repo.git_repo",
                repo_dir=str(repo_dir),
                returncode=result.returncode,
                stderr=result.stderr,
                stderr_length=len(result.stderr),
                stdout=result.stdout,
                stdout_length=len(result.stdout),
            ).error("Git diff failed")
            return ""

        patch_text = str(result.stdout)
        if not patch_text:
            logger.bind(
                component="benchmark.predict.agents.repo.git_repo",
                repo_dir=str(repo_dir),
            ).warning("Git diff returned empty output despite detecting changes")
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
