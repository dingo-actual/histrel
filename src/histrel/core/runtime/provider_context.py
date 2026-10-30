#!/usr/bin/env python3
"""
Provider-request context construction for the benchmark agent loop.

Builds the per-iteration tool definitions and provider messages for
both stateless mode (rendered user message) and react mode (replayed
conversation history). The tool definitions are the session's static
tool set in both modes.

    Important Functions:
        _tool_definitions(tools) -> tuple[ToolDefinition, ...]: ...
        _react_provider_messages(...) -> tuple[ProviderMessage, ...]: ...
        build_provider_context(...) -> tuple: Tool defs and messages.
        build_model_request(...) -> ModelRequest: The model request.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from histrel.context.builder import message_to_provider
from histrel.core.journal import load_journal
from histrel.model.messages import Message, TextPart, ToolResultPart
from histrel.model.render import (
    render_user_message_init,
    render_user_message_step,
)
from histrel.model.tool_call_lines import format_tool_call
from histrel.model.types import (
    ModelRequest,
    ProviderMessage,
    ToolDefinition,
)
from histrel.tools.tool import Tool

if TYPE_CHECKING:
    from histrel.core.runtime.runtime import Runtime

# Reminder shown after an iteration that made no tool calls. The
# stateless prompt renders it into the next step message; react mode
# commits it as a user message instead.
TOOL_CALL_REQUIRED_REMINDER = (
    "No tool calls were made during the last iteration. You must respond "
    "with at least one tool call."
)


def _tool_definitions(tools: tuple[Tool, ...]) -> tuple[ToolDefinition, ...]:
    """Convert Tool objects into provider tool definitions.

    Args:
        tools (tuple[Tool, ...]): The session's tools.

    Returns:
        (tuple[ToolDefinition, ...]) Converted tool definitions.
    """
    return tuple(
        ToolDefinition(type="function", function=tool.definition["function"])
        for tool in tools
    )


def _react_provider_messages(
    *,
    system_prompt: str,
    user_msg: str,
    journal_messages: list[Message],
) -> tuple[ProviderMessage, ...]:
    """Build react-mode provider messages from conversation history.

    Lays out the request as: the system prompt, the task user message,
    then every assistant message, tool call, and tool call result in
    commit order. The journal's first user message (the committed task
    message) is skipped because it is the task user message; later
    user messages (revision feedback) are included.

    Args:
        system_prompt (str): System prompt for this iteration.
        user_msg (str): Task user message passed to the agent loop.
        journal_messages (list[Message]): Committed journal messages.

    Returns:
        (tuple[ProviderMessage, ...]) Provider messages for the request.
    """
    history: list[ProviderMessage] = []
    task_message_seen = False
    for msg in journal_messages:
        if msg.role == "user" and not task_message_seen:
            task_message_seen = True
            continue
        converted = message_to_provider(msg)
        if converted is not None:
            history.append(converted)
    return (
        ProviderMessage(role="system", content=system_prompt),
        ProviderMessage(role="user", content=user_msg),
        *history,
    )


def _latest_assistant_text(messages: list[Message]) -> str:
    """Return the text of the most recent assistant output message.

    Scans the journal in reverse for the last assistant message with
    non-empty text. Only text parts are collected -- LLM-internal
    reasoning parts are excluded.

    Args:
        messages (list[Message]): Journal messages in commit order.

    Returns:
        (str) The most recent assistant text, or "" if none exists.
    """
    for msg in reversed(messages):
        if msg.role != "assistant":
            continue
        text = "\n".join(
            part.text for part in msg.content if isinstance(part, TextPart)
        )
        if text:
            return text
    return ""


def _stateless_user_content(
    runtime: Runtime,
    user_msg: str,
    tools: tuple[Tool, ...],
    journal_messages: list[Message],
) -> str:
    """Render the per-iteration stateless user message.

    The first iteration renders the initialize prompt; every later
    iteration renders the step prompt with the turn's observations and
    latest assistant reasoning.

    Args:
        runtime (Runtime): The runtime session state.
        user_msg (str): Task user message passed to the agent loop.
        tools (tuple[Tool, ...]): The session's tools.
        journal_messages (list[Message]): Committed journal messages.

    Returns:
        (str) The rendered user message content.
    """
    if runtime.iteration_count == 1:
        return render_user_message_init(
            instance_problem=user_msg,
            available_tools=tools,
        )
    observations = [
        f"[Iteration {part.iter_num}] {format_tool_call(part.tool_name, part.arguments)}\n"
        f"---\n"
        f"{part.output or ''}"
        for msg in journal_messages
        if msg.role == "tool"
        for part in msg.content
        # Informative failures come from correctly-specified calls:
        # the model must see them to adjust its next action.
        if isinstance(part, ToolResultPart) and (part.success or part.informative)
    ]
    return render_user_message_step(
        instance_problem=user_msg,
        observations=observations,
        reasoning_state=_latest_assistant_text(journal_messages),
        available_tools=list(tools),
        iter_crnt=runtime.iteration_count,
        reminder=(TOOL_CALL_REQUIRED_REMINDER if runtime.tool_call_required else ""),
    )


def build_provider_context(
    runtime: Runtime,
    *,
    system_prompt: str,
    user_msg: str,
    tools: tuple[Tool, ...],
) -> tuple[tuple[ToolDefinition, ...], tuple[ProviderMessage, ...]]:
    """Build one iteration's tool definitions and provider messages.

    Both modes send the session's static tool definitions. Stateless
    mode renders a per-iteration user message with no conversation
    history; react mode replays the full conversation history: the task
    user message plus every assistant message, tool call, and tool
    call result committed so far.

    Args:
        runtime (Runtime): The runtime session state.
        system_prompt (str): System prompt for this iteration.
        user_msg (str): Task user message passed to the agent loop.
        tools (tuple[Tool, ...]): The session's tools.

    Returns:
        (tuple) The (tool definitions, provider messages) for the
        iteration's model request.
    """
    journal_messages = list(load_journal(runtime.journal._path))
    tool_defs = _tool_definitions(tools)
    if runtime.config.mode == "react":
        provider_messages = _react_provider_messages(
            system_prompt=system_prompt,
            user_msg=user_msg,
            journal_messages=journal_messages,
        )
        return tool_defs, provider_messages
    user_content = _stateless_user_content(runtime, user_msg, tools, journal_messages)
    provider_messages = (
        ProviderMessage(role="system", content=system_prompt),
        ProviderMessage(role="user", content=user_content),
    )
    return tool_defs, provider_messages


def build_model_request(
    runtime: Runtime,
    *,
    tool_defs: tuple[ToolDefinition, ...],
    provider_messages: tuple[ProviderMessage, ...],
    prompt_cache_key: str,
    session_affinity_key: str,
) -> ModelRequest:
    """Build one iteration's model request from the provider context.

    Args:
        runtime (Runtime): The runtime session state.
        tool_defs (tuple[ToolDefinition, ...]): Tool definitions for
            the request.
        provider_messages (tuple[ProviderMessage, ...]): Provider
            messages for the request.
        prompt_cache_key (str): Session prompt-cache key ("" to skip).
        session_affinity_key (str): Session affinity key.

    Returns:
        (ModelRequest) The request for the iteration's model call.
    """
    return ModelRequest(
        messages=provider_messages,
        tools=tool_defs,
        temperature=runtime.config.model_temperature,
        top_p=runtime.config.model_top_p,
        reasoning_effort=runtime.config.model_reasoning_effort or "max",
        max_tokens_output=runtime.config.model_max_tokens_output,
        prompt_cache_key=prompt_cache_key or None,
        extra_headers={"x-session-affinity": session_affinity_key},
    )
