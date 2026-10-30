#!/usr/bin/env python3
"""Repository state checking via dulwich.

Composes direct dulwich comparisons to detect changes against the
committed HEAD state: staged entries (index versus the HEAD tree) and
unstaged files (working tree versus the index).

Untracked files are deliberately excluded. The benchmark harness writes
its own scaffolding (``benchmark/bench_settings.toml``, the agent config)
into the workspace, so an untracked file says nothing about whether the
agent edited the repository; the SWE-bench patch is extracted with
``git diff``, which ignores untracked files as well.

    Important Classes:
        RepoStateChecker: Checks the repository against its committed
        state.
"""

from __future__ import annotations

from pathlib import Path

from dulwich.index import get_unstaged_changes
from dulwich.objects import Commit, ObjectID
from dulwich.repo import Repo


def _head_tree_id(repo: Repo) -> bytes | None:
    """Resolve the HEAD tree id, or None when HEAD is unborn.

    Args:
        repo (Repo): Open dulwich repository.

    Returns:
        (bytes | None) HEAD commit's tree SHA, or None when the
        repository has no commits.
    """
    try:
        head = repo[b"HEAD"]
    except KeyError:
        return None
    if not isinstance(head, Commit):
        raise TypeError("HEAD must point to a commit")
    return bytes(head.tree)


class RepoStateChecker:
    """Checks repository changes against the committed HEAD state.

    Instance Attributes:
        _repo_path (Path): Path to the git repository.
    """

    def __init__(self, repo_path: Path) -> None:
        """Initialize the state checker.

        Args:
            repo_path (Path): Path to the git repository.
        """
        self._repo_path = repo_path

    def has_changes(self) -> bool:
        """Check whether the repository differs from the committed state.

        Composes direct dulwich comparisons: staged entries (index versus
        the HEAD tree; an unborn HEAD holds no committed state, making
        every index entry a change) and unstaged files (working tree
        versus the index). Untracked files are ignored, since the harness
        writes its own scaffolding into the workspace and ``git diff``
        excludes untracked files from the extracted patch.

        Args:
            (none)

        Returns:
            (bool) True when any staged or unstaged change exists against
            HEAD; False when the committed, staged, and working tree
            states all match.

        Raises:
            TypeError: When HEAD does not point to a commit.

        Example Usage:
            ```python
                >>> import tempfile
                >>> from dulwich.repo import Repo as DulwichRepo
                >>> with tempfile.TemporaryDirectory() as tmp:
                ...     DulwichRepo.init(tmp).close()
                ...     RepoStateChecker(Path(tmp)).has_changes()
                False
            ```
        """
        with Repo(str(self._repo_path)) as repo:
            index = repo.open_index()

            # Staged: index entries versus the HEAD tree.
            tree_id = _head_tree_id(repo)
            if tree_id is None:
                if len(index) > 0:
                    return True
            else:
                for _change in index.changes_from_tree(
                    repo.object_store, ObjectID(tree_id)
                ):
                    return True

            # Unstaged: working tree files versus the index.
            normalizer = repo.get_blob_normalizer()
            for _path in get_unstaged_changes(
                index, str(self._repo_path), normalizer.checkin_normalize
            ):
                return True

            return False
