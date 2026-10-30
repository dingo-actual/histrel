#!/usr/bin/env python3
"""Shared fixtures for the react-mode integration tests."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from histrel.core.runtime import run_agent
from histrel.model.types import ModelRequest
from tests.integration.helpers import (
    USER_MSG,
    PatchedRuntime,
    make_config,
)
from tests.integration.react_mode.helpers import (
    make_react_config,
)
from tests.integration.scripting import (
    make_fake_response,
    tool_call,
)


@pytest.fixture
def history_session(
    tmp_path: Path,
    scripted_session: Callable[..., tuple[PatchedRuntime, list[ModelRequest]]],
) -> tuple[str, list[ModelRequest]]:
    """Run a session whose second request replays one read tool call.

    The first turn reads calculator.py with a direct tool call; the
    second turn is text-only, so its request replays turn 1's history.
    Returns the calculator path and every recorded model request.
    """
    calculator = str(tmp_path / "workspace" / "calculator.py")
    config = make_react_config(make_config())
    session, recorded = scripted_session(
        make_fake_response(
            text="Reading the module.",
            tool_calls=(tool_call("read", {"path": calculator}, "call_1"),),
        ),
        make_fake_response(text="Second reasoning state."),
        make_fake_response(text="Second reasoning state."),
        config=config,
    )
    run_agent(USER_MSG, session.runtime)
    return calculator, recorded
