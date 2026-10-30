#!/usr/bin/env python3
"""
Message factory for the canonical message model.

    Important Classes:
        MessageFactory: Allocates monotonically increasing message indices.
"""

from __future__ import annotations

import uuid as uuid_mod

from histrel.core.ids import SessionId
from histrel.model.messages.message import Message, MessageRole
from histrel.model.messages.metadata import (
    ErrorInfo,
    MessageMetadata,
)
from histrel.model.messages.parts import MessagePart


class MessageFactory:
    """Allocate monotonically increasing message indices and create messages.

    Only one component should own a MessageFactory to ensure sequence numbers
    are gap-tolerant and monotonically increasing within a session.

    Instance Attributes:
        _session_id (SessionId): The session ID for all messages.
        _next_message_index (int): The next message index to assign.
    """

    def __init__(
        self,
        session_id: SessionId,
        next_message_index: int = 0,
    ) -> None:
        """Initialize the factory.

        Args:
            session_id (SessionId): Session ID for all created messages.
            next_message_index (int): Starting message index. Set to
                max(existing indices) + 1 when replaying a session.
        """
        self._session_id = session_id
        self._next_message_index = next_message_index

    def create(
        self,
        *,
        role: MessageRole,
        content: list[MessagePart],
        metadata: MessageMetadata,
        message_id: str | None = None,
        created: int | None = None,
        completed: int | None = None,
        error: ErrorInfo | None = None,
    ) -> Message:
        """Create a new message with an auto-assigned message index.

        Args:
            role (str): Message role (user, assistant, system, or tool).
            content (list[MessagePart]): Content items in source order.
            metadata (MessageMetadata): Message metadata.
            message_id (str | None): Explicit message ID. Auto-generated
                if not provided.
            created (int | None): Unix timestamp. Defaults to current time.
            completed (int | None): Completion timestamp.
            error (ErrorInfo | None): Error information.

        Returns:
            (Message) A new message with the next sequential index.
        """
        from histrel.core.clock import utc_timestamp

        message_index = self._next_message_index
        self._next_message_index += 1
        return Message(
            message_id=message_id or f"msg-{uuid_mod.uuid4().hex}",
            session_id=str(self._session_id),
            source="bench-agent",
            role=role,
            created=created if created is not None else utc_timestamp(),
            completed=completed,
            content=content,
            error=error,
            metadata=metadata,
            message_index=message_index,
        )

    @property
    def next_message_index(self) -> int:
        """Return the next message index that will be assigned."""
        return self._next_message_index
