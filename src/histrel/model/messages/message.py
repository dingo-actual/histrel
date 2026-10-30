#!/usr/bin/env python3
"""
Canonical message model and secret redaction.

    Important Classes:
        Message: Complete canonical message.

    Important Functions:
        redact_message(message) -> Message: Redact secrets in a message.
"""

from __future__ import annotations

import uuid as uuid_mod
from typing import Any, Literal

from pydantic import BaseModel, Field

from histrel.model.messages.metadata import (
    ErrorInfo,
    MessageMetadata,
)
from histrel.model.messages.parts import (
    MessagePart,
    ToolRequestPart,
)

# Internal roles include system and tool; these map to the wire roles
# (system -> application, tool -> assistant) when rendered as messages.
MessageRole = Literal["user", "assistant", "system", "tool"]


class Message(BaseModel):
    """Canonical internal message.

    Every interaction that forms part of the agent trajectory is represented
    by a Message object. Messages are append-only after journal commitment.

    Instance Attributes:
        uuid (str): Auto-generated UUID hex for log correlation.
        message_id (str): Unique message identifier within a session.
        session_id (str): Session identifier.
        source (Literal["bench-agent"]): Always "bench-agent".
        role (str): Message role (user, assistant, system, or tool).
        created (int): Unix timestamp when message was created.
        completed (int | None): Unix timestamp when message was completed.
        content (list[MessagePart]): Ordered content items.
        error (ErrorInfo | None): Error information if processing failed.
        metadata (MessageMetadata): Additional message metadata.
        message_index (int): Sequential index within the session; -1 when
            unknown or not yet assigned.
    """

    uuid: str = Field(default_factory=lambda: uuid_mod.uuid4().hex)
    message_id: str
    session_id: str
    source: Literal["bench-agent"] = "bench-agent"
    role: MessageRole
    created: int
    completed: int | None = None
    content: list[MessagePart] = Field(default_factory=list)
    error: ErrorInfo | None = None
    metadata: MessageMetadata = Field(default_factory=MessageMetadata)
    message_index: int = -1


# ---------------------------------------------------------------------------
# Secret redaction
# ---------------------------------------------------------------------------

# Keys whose values should be redacted from tool arguments before journal
# commitment.
_SECRET_ARG_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "token",
        "secret",
        "password",
        "passwd",
        "credential",
        "authorization",
        "auth",
    }
)

_REDACTED = "***REDACTED***"


def _redact_arguments(arguments: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a copy of arguments with secret values redacted.

    Args:
        arguments (dict[str, Any] | None): Tool call arguments.

    Returns:
        (dict[str, Any] | None) Redacted copy, or None if input is None.
    """
    if arguments is None:
        return None
    redacted: dict[str, Any] = {}
    for key, value in arguments.items():
        if key.lower() in _SECRET_ARG_KEYS:
            redacted[key] = _REDACTED
        else:
            redacted[key] = value
    return redacted


def redact_message(message: Message) -> Message:
    """Return a copy of the message with secrets redacted from tool arguments.

    Does not mutate the original message. Scans ToolRequestPart arguments
    for known secret keys and replaces their values with a redaction marker.

    Args:
        message (Message): The message to redact.

    Returns:
        (Message) A new message with secrets redacted.
    """
    if not message.content:
        return message.model_copy(deep=True)

    has_tool_request = any(
        isinstance(part, ToolRequestPart) for part in message.content
    )
    if not has_tool_request:
        return message.model_copy(deep=True)

    redacted = message.model_copy(deep=True)
    new_content: list[MessagePart] = []
    for part in redacted.content:
        if isinstance(part, ToolRequestPart):
            redacted_args = _redact_arguments(part.arguments)
            new_content.append(part.model_copy(update={"arguments": redacted_args}))
        else:
            new_content.append(part)
    redacted.content = new_content
    return redacted
