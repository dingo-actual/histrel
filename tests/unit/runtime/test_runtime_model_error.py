#!/usr/bin/env python3
"""Unit tests for runtime model error handling."""

from __future__ import annotations

from openai import OpenAI

from histrel.core.runtime import (
    Runtime,
    run_agent,
)
from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
)
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
)
from histrel.terminal import (
    SessionStatus,
)
from tests.unit.factories import (
    build_agent_config,
    init_fixture_repo,
)
from tests.unit.runtime.helpers import (
    USER_MSG,
)


class Test04Unit04Runtime04ModelError:
    """Runtime model error handling."""

    def test_04_unit_04_runtime_04_error_01_model_exception_terminates(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        config = build_agent_config()
        init_fixture_repo(tmp_path)

        adapter = OpenAICompatAdapter(
            client=OpenAI(api_key="test-key"),
            model_name="gpt-4o",
            provider="openai",
        )

        def failing_complete(request: ModelRequest) -> ModelResponse:
            raise RuntimeError("API unreachable")

        monkeypatch.setattr(adapter, "complete", failing_complete)

        runtime = Runtime(
            config=config, workspace_root=tmp_path, output_dir=tmp_path / "output"
        )
        runtime.initialize(USER_MSG)
        runtime.model_adapter = adapter

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert
        assert outcome.status == SessionStatus.INFRASTRUCTURE_ERROR
        assert "API unreachable" in outcome.reason
        # A handled model error still finalizes, so the failed attempt's
        # artifacts and incurred usage are preserved.
        artifacts_dir = tmp_path / "output" / config.session_id / "artifacts"
        assert (artifacts_dir / "outcome.json").exists()
        assert (artifacts_dir / "costs.json").exists()
        assert (artifacts_dir / "result_manifest.json").exists()
