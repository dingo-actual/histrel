#!/usr/bin/env python3
"""Tool-call headers for stateless observation rendering.

Each observation names the call the agent made and the arguments it
passed, so the agent can see exactly what its own tool calls did.

    Important Functions:
        format_tool_call(tool_name, arguments) -> str: Render the header.
"""

from __future__ import annotations

import json
from typing import Any

# Placeholder name for a call whose tool could not be resolved.
_UNKNOWN_TOOL = "unknown"


def format_tool_call(
    tool_name: str | None,
    arguments: dict[str, Any] | None,
) -> str:
    """Render one call as ``tool_name(key=value, ...)``.

    Values render as JSON, so strings stay quoted and every argument the
    model passed is visible to it on the next iteration.

    Args:
        tool_name (str | None): The executed call's tool name.
        arguments (dict[str, Any] | None): The call's parsed arguments.

    Returns:
        (str) The call header, e.g. ``read(path="main.py", limit=20)``.
    """
    rendered = ", ".join(
        f"{key}={json.dumps(value, ensure_ascii=False)}"
        for key, value in (arguments or {}).items()
    )
    return f"{tool_name or _UNKNOWN_TOOL}({rendered})"
