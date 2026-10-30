#!/usr/bin/env python3
"""Shared fixtures for the runtime unit tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from histrel.core.runtime import Runtime
from histrel.model.types import ModelRequest, ModelResponse
from tests.unit.factories import (
    build_agent_config,
    build_runtime,
    init_fixture_repo,
)
from tests.unit.runtime.helpers import (
    USER_MSG,
    patched_adapter,
)
from tests.unit.runtime.stubs import EchoTool


@pytest.fixture
def make_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Callable[..., Runtime]:
    """Return a factory building a Runtime rooted at tmp_path.

    By default the workspace is a fixture git repository and the session
    is initialized with USER_MSG. Pass repo=False or initialize=False to
    skip those steps, or files={...} to commit specific file contents.
    """

    def factory(
        *,
        repo: bool = True,
        initialize: bool = True,
        output_dir: Path | None = None,
        files: dict[str, str] | None = None,
        **config_kwargs: Any,
    ) -> Runtime:
        if repo:
            init_fixture_repo(tmp_path, files=files)
        # A bounded turn budget ends sessions that stop without a change.
        config_kwargs.setdefault("max_turns", 50)
        return build_runtime(
            build_agent_config(**config_kwargs),
            tmp_path,
            monkeypatch=monkeypatch,
            initialize=USER_MSG if initialize else None,
            output_dir=output_dir,
        )

    return factory


@pytest.fixture
def make_stub_runtime(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Callable[..., Runtime]:
    """Return a factory building a runtime whose calls run an echo stub.

    The model adapter's complete method is the given fake_complete. The
    tool list is replaced by an echo stub, so calls to ``echo`` execute
    and are advertised to the model; pass changes_repo=False for a stub
    that leaves the repository untouched.
    """

    def factory(
        fake_complete: Any,
        *,
        changes_repo: bool = True,
        **config_kwargs: Any,
    ) -> Runtime:
        init_fixture_repo(tmp_path)
        runtime = build_runtime(
            build_agent_config(**config_kwargs),
            tmp_path,
            monkeypatch=monkeypatch,
            initialize=USER_MSG,
        )
        runtime.model_adapter = patched_adapter(monkeypatch, fake_complete)
        runtime.ordinary_tools = [
            EchoTool(runtime, tmp_path, changes_repo=changes_repo)
        ]
        return runtime

    return factory


@pytest.fixture
def make_json_runtime(
    make_runtime: Callable[..., Runtime],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> Callable[..., Runtime]:
    """Return a factory building a runtime whose JSON repairs fall back.

    The runtime runs in the given agent loop mode, edits resolve against
    the workspace, and an echo stub that leaves the repository untouched
    is appended to the session tools.
    """

    def factory(
        mode: str,
        fake_complete: Callable[[ModelRequest], ModelResponse],
    ) -> Runtime:
        monkeypatch.chdir(tmp_path)
        # A bounded turn budget ends sessions that stop without a change.
        runtime = make_runtime(mode=mode, max_turns=2)
        runtime.model_adapter = patched_adapter(monkeypatch, fake_complete)
        runtime.ordinary_tools.append(EchoTool(runtime, tmp_path, changes_repo=False))
        return runtime

    return factory
