#!/usr/bin/env python3
"""Unit tests for GitRepoManager SWE-bench integration.

Tests the validate_repo() method for pre-cloned repository validation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from benchmark.predict.agents.repo.git_repo import GitRepoManager


class Test04Unit01GitRepo01ValidateRepo01Success:
    """Tests for successful repository validation."""

    def test_04_unit_01_git_repo_01_validate_repo_01_success_01_valid_repo(self):
        """Repository validation succeeds when .git directory exists."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "django"
            repo_dir.mkdir()
            git_dir = repo_dir / ".git"
            git_dir.mkdir()
            manager = GitRepoManager()

            # Act & Assert
            manager.validate_repo(repo_dir)  # Should not raise

    def test_04_unit_01_git_repo_01_validate_repo_01_success_02_path_as_path_object(
        self,
    ):
        """Repository validation accepts Path object."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "test_repo"
            repo_dir.mkdir()
            (repo_dir / ".git").mkdir()
            manager = GitRepoManager()

            # Act & Assert
            manager.validate_repo(repo_dir)  # Should not raise


class Test04Unit01GitRepo01ValidateRepo02Failure:
    """Tests for repository validation failures."""

    def test_04_unit_01_git_repo_01_validate_repo_02_failure_01_nonexistent_directory(
        self,
    ):
        """Repository validation fails when directory does not exist."""
        # Arrange
        manager = GitRepoManager()
        nonexistent_path = Path("/nonexistent/path/to/repo")

        # Act & Assert
        with pytest.raises(RuntimeError) as exc_info:
            manager.validate_repo(nonexistent_path)
        assert "not found" in str(exc_info.value).lower()

    def test_04_unit_01_git_repo_01_validate_repo_02_failure_02_no_git_directory(self):
        """Repository validation fails when .git directory is missing."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "django"
            repo_dir.mkdir()
            # No .git directory
            manager = GitRepoManager()

            # Act & Assert
            with pytest.raises(RuntimeError) as exc_info:
                manager.validate_repo(repo_dir)
            assert "not a git repository" in str(exc_info.value).lower()

    def test_04_unit_01_git_repo_01_validate_repo_02_failure_03_git_file_not_directory(
        self,
    ):
        """Repository validation fails when .git exists as file not directory."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "django"
            repo_dir.mkdir()
            git_file = repo_dir / ".git"
            git_file.write_text("gitdir: /path/to/actual/git")
            manager = GitRepoManager()

            # Act & Assert
            with pytest.raises(RuntimeError) as exc_info:
                manager.validate_repo(repo_dir)
            assert "not a git repository" in str(exc_info.value).lower()


class Test04Unit01GitRepo02ExtractPatch01Success:
    """Tests for successful patch extraction (unchanged from original)."""

    def test_04_unit_01_git_repo_02_extract_patch_01_success_01_no_changes(self):
        """Extract patch returns empty string when no changes exist."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "repo"
            repo_dir.mkdir()
            (repo_dir / ".git").mkdir()
            manager = GitRepoManager()

            # Act
            result = manager.extract_patch(repo_dir)

            # Assert
            assert result == ""


class Test04Unit01GitRepo03Integration:
    """Integration tests for GitRepoManager with real git operations."""

    def test_04_unit_01_git_repo_03_integration_01_full_workflow(self):
        """Complete workflow: validate then extract patch."""
        # Arrange
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "django"
            repo_dir.mkdir()
            git_dir = repo_dir / ".git"
            git_dir.mkdir()
            # Create a fake git index file to make it look like a real repo
            (git_dir / "index").touch()
            manager = GitRepoManager()

            # Act & Assert - Should validate successfully
            manager.validate_repo(repo_dir)

            # Act - Extract patch from clean repo
            patch = manager.extract_patch(repo_dir)

            # Assert
            assert patch == ""  # No changes in clean repo
