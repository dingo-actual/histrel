#!/usr/bin/env python3
"""Shared factory for real git repositories used by patch extraction tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

BASE_FILE_CONTENT = "base line\n"


def git(repo_dir: Path, *args: str) -> str:
    """Run a git command in ``repo_dir`` with a fixed test identity."""
    result = subprocess.run(
        [
            "git",
            "-c",
            "user.name=Histrel Test",
            "-c",
            "user.email=test@histrel.invalid",
            "-c",
            "commit.gpgsign=false",
            *args,
        ],
        cwd=repo_dir,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def make_git_repo(repo_dir: Path) -> str:
    """Create a repository with one committed file.

    Returns:
        (str) The base commit hash the tests measure changes from.
    """
    repo_dir.mkdir(parents=True, exist_ok=True)
    git(repo_dir, "init", "-q")
    (repo_dir / "module.py").write_text(BASE_FILE_CONTENT, encoding="utf-8")
    git(repo_dir, "add", "module.py")
    git(repo_dir, "commit", "-q", "-m", "base")
    return git(repo_dir, "rev-parse", "HEAD").strip()


def clone_at_commit(repo_dir: Path, base_commit: str, clone_dir: Path) -> Path:
    """Clone ``repo_dir`` and check out ``base_commit`` in the clone."""
    subprocess.run(
        ["git", "clone", "-q", str(repo_dir), str(clone_dir)],
        capture_output=True,
        text=True,
        check=True,
    )
    git(clone_dir, "checkout", "-q", base_commit)
    return clone_dir
