#!/usr/bin/env python3
"""Unit tests for repository state checking via dulwich.

Tests detection of changes against the committed HEAD state using
real git repositories created with dulwich (no mocks).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from dulwich.porcelain import add, commit
from dulwich.repo import Repo

from histrel.repo_state import RepoStateChecker


def _init_repo(path: Path) -> Repo:
    """Initialize a git repo and return the Repo object."""
    return Repo.init(str(path))


def _commit_file(
    repo_path: Path, filename: str, content: str, message: str = "init"
) -> None:
    """Write, add, and commit a file."""
    (repo_path / filename).write_text(content)
    add(str(repo_path), [filename.encode()])
    commit(
        str(repo_path),
        message=message.encode(),
        author=b"Test <test@test.com>",
        committer=b"Test <test@test.com>",
    )


class Test04Unit01RepoState02Committed:
    """Repository changes against the committed HEAD state."""

    def test_04_unit_01_repo_state_02_committed_01_clean_repo_reports_false(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp)
            _init_repo(repo_path)
            _commit_file(repo_path, "main.py", "print('hello')")

            # Act
            checker = RepoStateChecker(repo_path)
            has_changes = checker.has_changes()

            # Assert — clean repo reports no changes
            assert has_changes is False

    def test_04_unit_01_repo_state_02_committed_02_modified_tracked_file_true(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp)
            _init_repo(repo_path)
            _commit_file(repo_path, "main.py", "print('hello')")

            checker = RepoStateChecker(repo_path)

            # Modify a tracked file without staging it
            (repo_path / "main.py").write_text("print('world')")

            # Act
            has_changes = checker.has_changes()

            # Assert
            assert has_changes is True

    def test_04_unit_01_repo_state_02_committed_03_top_level_untracked_file_false(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp)
            _init_repo(repo_path)
            _commit_file(repo_path, "main.py", "print('hello')")

            checker = RepoStateChecker(repo_path)

            # Create a top-level untracked file
            (repo_path / "fix.txt").write_text("print('fixed')")

            # Act
            has_changes = checker.has_changes()

            # Assert — untracked files are harness scaffolding written into
            # the workspace, and the harness patch is produced with
            # ``git diff``, which ignores them
            assert has_changes is False

    def test_04_unit_01_repo_state_02_committed_04_untracked_file_in_new_dir_false(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp)
            _init_repo(repo_path)
            _commit_file(repo_path, "main.py", "print('hello')")

            checker = RepoStateChecker(repo_path)

            # Create an untracked file inside a new untracked directory
            nested = repo_path / "pkg" / "src"
            nested.mkdir(parents=True)
            (nested / "mod.py").write_text("pass")

            # Act
            has_changes = checker.has_changes()

            # Assert — no tracked file was touched
            assert has_changes is False

    def test_04_unit_01_repo_state_02_committed_05_deleted_tracked_file_true(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp)
            _init_repo(repo_path)
            _commit_file(repo_path, "main.py", "print('hello')")
            _commit_file(repo_path, "util.py", "pass", message="add util")

            checker = RepoStateChecker(repo_path)

            # Delete a tracked file
            (repo_path / "util.py").unlink()

            # Act
            has_changes = checker.has_changes()

            # Assert
            assert has_changes is True

    def test_04_unit_01_repo_state_02_committed_06_staged_file_without_commit_true(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp)
            _init_repo(repo_path)
            _commit_file(repo_path, "main.py", "print('hello')")

            checker = RepoStateChecker(repo_path)

            # Stage a new file without committing it
            (repo_path / "new.py").write_text("print('new')")
            add(str(repo_path), [b"new.py"])

            # Act
            has_changes = checker.has_changes()

            # Assert
            assert has_changes is True

    def test_04_unit_01_repo_state_02_committed_07_unborn_repo_without_files_false(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp)
            _init_repo(repo_path)

            # Act — empty repo with no commits
            checker = RepoStateChecker(repo_path)
            has_changes = checker.has_changes()

            # Assert — unborn HEAD holds no committed or staged state
            assert has_changes is False

    def test_04_unit_01_repo_state_02_committed_08_ignored_path_reports_false(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp)
            _init_repo(repo_path)
            _commit_file(repo_path, "main.py", "print('hello')")
            _commit_file(repo_path, ".gitignore", "ignored.txt")

            checker = RepoStateChecker(repo_path)

            # Create an ignored untracked file
            (repo_path / "ignored.txt").write_text("junk")

            # Act
            has_changes = checker.has_changes()

            # Assert — ignored paths do not count as changes
            assert has_changes is False

    def test_04_unit_01_repo_state_02_committed_09_result_is_repeatable(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo_path = Path(tmp)
            _init_repo(repo_path)
            _commit_file(repo_path, "main.py", "print('hello')")

            checker = RepoStateChecker(repo_path)
            (repo_path / "main.py").write_text("print('world')")

            # Act — repeated checks on the same state
            first = checker.has_changes()
            second = checker.has_changes()

            # Assert — same state produces same result
            assert first is True
            assert second is True
