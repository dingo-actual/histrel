#!/usr/bin/env python3
"""
Agent runtime package for the benchmark agent.

Facade re-exporting every previously importable name of the former
``core/runtime.py`` module. The Runtime class and session wiring live
in ``runtime``, the main loop in ``react_loop``, tool execution in
``tool_execution``, session finalization in ``finalization``,
assistant-message committing in ``turn_commits``, and provider-request
construction in ``provider_context``.

    Important Classes:
        Runtime: Orchestrates the full agent session lifecycle.

    Important Functions:
        run_agent(user_msg, runtime) -> SessionOutcome: Run the main loop.
"""

from histrel.core.runtime.provider_context import (
    _latest_assistant_text,
    _react_provider_messages,
    _tool_definitions,
)
from histrel.core.runtime.react_loop import run_agent
from histrel.core.runtime.runtime import Runtime, log
from histrel.core.runtime.turn_commits import (
    _repaired_call,
    _safe_json_dict,
)

__all__ = [
    "Runtime",
    "log",
    "run_agent",
    "_latest_assistant_text",
    "_react_provider_messages",
    "_repaired_call",
    "_safe_json_dict",
    "_tool_definitions",
]
