#!/usr/bin/env python3
"""
Identifier types and generators for the benchmark agent.

Provides NewType aliases and generator functions for session and message
identifiers. All identifiers are strings, conforming to the
UnifiedMessage schema in debug/server/schemas/messages.py.

    Important Types:
        SessionId (NewType[str]): Session identifier.
        MessageId (NewType[str]): Message identifier.

    Important Functions:
        new_message_id() -> MessageId: Generate a unique message ID.
"""

from __future__ import annotations

import uuid
from typing import NewType

SessionId = NewType("SessionId", str)
MessageId = NewType("MessageId", str)


def new_message_id() -> MessageId:
    """Generate a unique message identifier.

    Returns:
        (MessageId) A UUID-based string unique within a session.
    """
    return MessageId(f"msg-{uuid.uuid4().hex}")
