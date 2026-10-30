#!/usr/bin/env python3
"""Unit tests for runtime finalization artifacts."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from histrel.core.runtime import (
    Runtime,
    run_agent,
)
from histrel.terminal import (
    SessionStatus,
)
from tests.unit.runtime.helpers import (
    USER_MSG,
    change_repo,
    make_completer,
    patched_adapter,
)


class Test04Unit05Runtime05Finalization:
    """Runtime finalization and artifact writing."""

    def test_04_unit_05_runtime_05_final_01_writes_artifacts(
        self,
        make_runtime: Callable[..., Runtime],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        # Arrange
        runtime = make_runtime()
        runtime.model_adapter = patched_adapter(
            monkeypatch, make_completer(text="Done.")
        )
        # A repo change plus a text-only response satisfies the terminal
        # condition so the session succeeds.
        change_repo(tmp_path)
        output_dir = tmp_path / "output"

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert
        assert outcome.status == SessionStatus.SUCCEEDED
        assert (output_dir / "test-session" / "artifacts" / "outcome.json").exists()
        assert (
            output_dir / "test-session" / "artifacts" / "result_manifest.json"
        ).exists()
