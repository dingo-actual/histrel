#!/usr/bin/env python3
"""
Prompt rendering for the benchmark agent.

Renders messages sent to the LLM from the Jinja2 templates in
``prompt_templates/``.

    Important Functions:
        render_user_message_init(instance_problem, available_tools) -> str:
            Render the first-iteration user message (template
            ``user_first.j2``).
        render_user_message_step(instance_problem, observations,
            reasoning_state, available_tools, iter_crnt) -> str:
            Render a later-iteration user message (template
            ``user_step.j2``).
        render_fix_json(malformed_json, tool_json_schema) -> str:
            Render the JSON-repair routine's user message (template
            ``fix_json.j2``).

    Important Variables:
        _TEMPLATE_DIR (Path): Directory holding the Jinja2 templates.
        _env (Environment): Shared template environment.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from histrel.tools.tool import Tool

_TEMPLATE_DIR = Path(__file__).resolve().parent / "prompt_templates"

# Plain-text prompt templates: no autoescaping and only the file-system
# loader. StrictUndefined makes a missing context value fail at render
# time instead of silently rendering an empty section.
_env = Environment(
    loader=FileSystemLoader(_TEMPLATE_DIR),
    undefined=StrictUndefined,
)


def render_user_message_init(
    instance_problem: str,
    available_tools: tuple[Tool, ...],
) -> str:
    """Render the first-iteration user message.

    Args:
        instance_problem (str): The instance problem statement.
        available_tools (tuple[Tool, ...]): Projected tools for the
            first iteration.

    Returns:
        (str) Rendered user message.

    Raises:
        UndefinedError: If a template-referenced value is missing.
        TemplateNotFound: If ``user_first.j2`` is missing.

    Example Usage:
        ```python
            >>> message = render_user_message_init("Fix the bug.", ())
            >>> "Fix the bug." in message
            True
        ```
    """
    return _env.get_template("user_first.j2").render(
        instance_problem=instance_problem,
        available_tools=available_tools,
    )


def render_user_message_step(
    instance_problem: str,
    observations: list[str],
    reasoning_state: str,
    available_tools: list[Tool],
    iter_crnt: int,
    reminder: str = "",
) -> str:
    """Render a later-iteration user message.

    Args:
        instance_problem (str): The instance problem statement.
        observations (list[str]): Raw outputs of successful or
            informatively-failing tool calls, each prefixed with
            ``[Iteration N]``.
        reasoning_state (str): The most recent assistant output text
            (LLM-internal reasoning excluded).
        available_tools (list[Tool]): Projected tools for this iteration.
        iter_crnt (int): Current loop iteration number.
        reminder (str): Optional reminder; omitted when empty.

    Returns:
        (str) Rendered user message.

    Raises:
        UndefinedError: If a template-referenced value is missing.
        TemplateNotFound: If ``user_step.j2`` is missing.

    Example Usage:
        ```python
            >>> message = render_user_message_step(
            ...     "Fix the bug.", ["[Iteration 1] ok"], "Read it.", [], 2
            ...     )
            >>> "Current iteration: 2" in message
            True
        ```
    """
    return _env.get_template("user_step.j2").render(
        instance_problem=instance_problem,
        observations=observations,
        reasoning_state=reasoning_state,
        available_tools=available_tools,
        iter_crnt=iter_crnt,
        reminder=reminder,
    )


def render_fix_json(
    malformed_json: str,
    tool_json_schema: str,
) -> str:
    """Render the JSON-repair routine's user message.

    Args:
        malformed_json (str): The malformed JSON arguments string from
            the tool call.
        tool_json_schema (str): The tool's JSON schema, serialized as
            text, describing the intended argument structure.

    Returns:
        (str) Rendered user message.

    Raises:
        UndefinedError: If a template-referenced value is missing.
        TemplateNotFound: If ``fix_json.j2`` is missing.

    Example Usage:
        ```python
            >>> import json
            >>> message = render_fix_json(
            ...     '{"path": "main.py"',
            ...     json.dumps({"type": "object", "properties": {}}),
            ...     )
            >>> "## Malformed JSON" in message
            True
        ```
    """
    return _env.get_template("fix_json.j2").render(
        malformed_json=malformed_json,
        tool_json_schema=tool_json_schema,
    )
