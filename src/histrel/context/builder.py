#!/usr/bin/env python3
"""
Journal-message conversion for the benchmark agent.

Converts a single committed journal message into the provider message
format. React mode replays the conversation through this conversion;
stateless mode renders a per-iteration prompt instead (see
``core/runtime/provider_context.py``).

    Important Functions:
        message_to_provider(msg) -> ProviderMessage | None: One message.
"""

from __future__ import annotations

import json
from typing import Any

from histrel.model.messages import (
    Message,
    MessagePart,
    ReasoningPart,
    TextPart,
    ToolRequestPart,
    ToolResultPart,
)
from histrel.model.types import ProviderMessage


def _extract_text(parts: list[MessagePart]) -> str:
    """Extract text content from message parts.

    Args:
        parts (list[MessagePart]): Message content parts.

    Returns:
        (str) Concatenated text from text and reasoning parts.
    """
    texts: list[str] = []
    for part in parts:
        if isinstance(part, (TextPart, ReasoningPart)):
            texts.append(part.text)
    return "\n".join(texts)


def _text_message(msg: Message, role: str) -> ProviderMessage | None:
    """Build a text-only provider message for a role.

    Args:
        msg (Message): Journal message.
        role (str): Provider role ("system" or "user").

    Returns:
        (ProviderMessage | None) Provider message, or None when the
        message has no text content.
    """
    text = _extract_text(msg.content)
    if text:
        return ProviderMessage(role=role, content=text)
    return None


def _tool_calls_from_content(
    content: list[MessagePart],
) -> list[dict[str, Any]] | None:
    """Build provider tool-call dicts from tool request parts.

    Args:
        content (list[MessagePart]): Assistant message content.

    Returns:
        (list[dict[str, Any]] | None) Tool-call dicts, or None when the
        message has no tool requests.
    """
    tool_calls: list[dict[str, Any]] | None = None
    for part in content:
        if isinstance(part, ToolRequestPart):
            if tool_calls is None:
                tool_calls = []
            tool_calls.append(
                {
                    "id": part.tool_id,
                    "type": "function",
                    "function": {
                        "name": part.tool_name,
                        "arguments": json.dumps(part.arguments or {}, sort_keys=True),
                    },
                }
            )
    return tool_calls


def _assistant_message(msg: Message) -> ProviderMessage | None:
    """Build a provider message for an assistant journal message.

    Args:
        msg (Message): Assistant journal message.

    Returns:
        (ProviderMessage | None) Provider message, or None when the
        message has neither text nor tool calls.
    """
    text = _extract_text(msg.content)
    tool_calls = _tool_calls_from_content(msg.content)
    if text or tool_calls:
        return ProviderMessage(
            role="assistant",
            content=text,
            tool_calls=tool_calls,
        )
    return None


def _tool_result_content(part: ToolResultPart) -> str:
    """Build the provider content for a tool result part.

    Prepends only the tool name; call kwargs are not replayed.

    Args:
        part (ToolResultPart): Tool result part.

    Returns:
        (str) Provider tool content.
    """
    result_output = part.output or part.error or ""
    if not part.tool_name:
        return result_output
    return f"Tool: {part.tool_name}\n{result_output}"


def _tool_message(msg: Message) -> ProviderMessage | None:
    """Build a provider message for a tool journal message.

    Uses the first tool result part, matching the historical behavior.

    Args:
        msg (Message): Tool journal message.

    Returns:
        (ProviderMessage | None) Provider message, or None when the
        message has no tool result part.
    """
    for part in msg.content:
        if isinstance(part, ToolResultPart):
            return ProviderMessage(
                role="tool",
                content=_tool_result_content(part),
                tool_call_id=part.tool_id,
            )
    return None


def message_to_provider(msg: Message) -> ProviderMessage | None:
    """Convert a journal message to a provider message.

    Args:
        msg (Message): Journal message.

    Returns:
        (ProviderMessage | None) Provider message, or None if the message
        has no convertible content.
    """
    if msg.role == "system":
        return _text_message(msg, "system")
    if msg.role == "user":
        return _text_message(msg, "user")
    if msg.role == "assistant":
        return _assistant_message(msg)
    if msg.role == "tool":
        return _tool_message(msg)
    return None
