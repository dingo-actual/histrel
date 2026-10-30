#!/usr/bin/env python3
"""
Canonical message model for the benchmark agent.

Defines Pydantic v2 BaseModel subclasses. Every interaction that forms part
of the agent trajectory is represented by a Message object.

This facade re-exports the model names. Redaction helpers live in
``message.py`` (not ``metadata.py``) because they operate on Message and
ToolRequestPart; placing them in ``metadata.py`` would create an import
cycle.

    Important Classes:
        Timing: Timing information for content items.
        TokenUsage: Token usage statistics (cached is subset of input).
        ErrorInfo: Error information for failed messages.
        MessageMetadata: Additional metadata about messages.
        TextPart: Plain text content.
        ReasoningPart: Extended reasoning content.
        FilePart: File or image attachment reference.
        ToolRequestPart: Tool execution request.
        ToolResultPart: Tool execution result.
        CompactionPart: Conversation history compaction event.
        BeginSessionPart: Session start event.
        EndSessionPart: Session end event.
        MessagePart: Discriminated union of all content types.
        Message: Complete canonical message.
        MessageFactory: Allocates monotonically increasing message indices.

    Important Functions:
        redact_message(message: Message) -> Message: Redact secrets in a message.
"""

from __future__ import annotations

from histrel.model.messages.factory import MessageFactory
from histrel.model.messages.message import (
    _REDACTED,
    _SECRET_ARG_KEYS,
    Message,
    MessageRole,
    _redact_arguments,
    redact_message,
)
from histrel.model.messages.metadata import (
    ErrorInfo,
    MessageMetadata,
    Timing,
    TokenUsage,
)
from histrel.model.messages.parts import (
    BeginSessionPart,
    CompactionPart,
    EndSessionPart,
    FilePart,
    MessagePart,
    ReasoningPart,
    TextPart,
    ToolRequestPart,
    ToolResultPart,
)

__all__ = [
    "BeginSessionPart",
    "CompactionPart",
    "EndSessionPart",
    "ErrorInfo",
    "FilePart",
    "Message",
    "MessageFactory",
    "MessageMetadata",
    "MessagePart",
    "MessageRole",
    "ReasoningPart",
    "TextPart",
    "Timing",
    "TokenUsage",
    "ToolRequestPart",
    "ToolResultPart",
    "_REDACTED",
    "_SECRET_ARG_KEYS",
    "_redact_arguments",
    "redact_message",
]
