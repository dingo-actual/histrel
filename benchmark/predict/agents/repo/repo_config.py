#!/usr/bin/env python3
"""
Repository configuration helpers for benchmark agents.

    Important Classes:
        RepoConfigWriter: Writes repo-local bench settings.
"""

from __future__ import annotations

from pathlib import Path


class RepoConfigWriter:
    """Write repo-local configuration files for agent execution."""

    def write_bench_settings(self, repo_dir: Path, source_path: Path) -> None:
        """Copy bench_settings.toml into the repo benchmark directory."""
        dest_dir = repo_dir / "benchmark"
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / "bench_settings.toml"
        dest_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
