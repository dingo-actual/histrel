#!/usr/bin/env python3
"""Shared fixtures for the tool unit tests."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
from dulwich import porcelain
from dulwich.repo import Repo

from histrel.tools.repository import EditFileTool
from histrel.tools.tool import ToolExecution
from tests.unit.tools.helpers import make_context


@pytest.fixture
def isolated_workspace() -> Iterator[Path]:
    """Create an isolated workspace directory and chdir the test into it."""
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp) / "workspace"
        workspace.mkdir()
        old_cwd = Path.cwd()
        os.chdir(workspace)
        try:
            yield workspace
        finally:
            os.chdir(old_cwd)


@pytest.fixture
def set_bash_timeout(monkeypatch: pytest.MonkeyPatch) -> Callable[[float], None]:
    """Return a callable that shortens the bash tool's timeout for a test."""

    def factory(seconds: float) -> None:
        monkeypatch.setattr(
            "histrel.tools.repository.bash.BASH_TIMEOUT_SECONDS", seconds
        )

    return factory


@pytest.fixture
def run_edit() -> Callable[[str, list[Any]], ToolExecution]:
    """Return a callable that runs edit against a path in the CWD."""

    def run(file_path: str, file_edits: list[Any]) -> ToolExecution:
        return EditFileTool().execute(
            {"file_path": file_path, "file_edits": file_edits}, make_context()
        )

    return run


@pytest.fixture
def make_git_repo(
    isolated_workspace: Path,
) -> Callable[..., Path]:
    """Return a factory building a git repo with one commit per snapshot.

    Each snapshot maps file names to new or updated content; files from
    earlier snapshots stay unchanged in later ones. The repo is created
    with dulwich inside the isolated workspace (real objects, no mocks).
    """

    def factory(*snapshots: dict[str, str | bytes], repo_name: str = "repo") -> Path:
        repo_path = isolated_workspace / repo_name
        repo_path.mkdir()
        Repo.init(str(repo_path))
        for snapshot in snapshots:
            for name, content in snapshot.items():
                target = repo_path / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(content, bytes):
                    target.write_bytes(content)
                else:
                    target.write_text(content, encoding="utf-8")
            porcelain.add(str(repo_path), [name.encode() for name in snapshot])
            porcelain.commit(
                str(repo_path),
                message=b"snapshot",
                author=b"Test <test@test.com>",
                committer=b"Test <test@test.com>",
            )
        return repo_path

    return factory
