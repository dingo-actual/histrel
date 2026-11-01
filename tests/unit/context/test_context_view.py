#!/usr/bin/env python3
"""Unit tests for the presented-history view.

The view is the history the model is shown: the whole journal before the
first compaction, then the retained tail plus everything committed since.
"""

from __future__ import annotations

from collections.abc import Callable

from histrel.context.compactor import CompactedState
from histrel.context.context_view import active_messages
from histrel.model.messages import Message, MessageFactory, MessageMetadata, TextPart


def _state(
    *,
    retained: tuple[str, ...] = (),
    middle_end: int = -1,
) -> CompactedState:
    """Build a compaction state for the given retained IDs."""
    return CompactedState(
        system_prompt="system",
        original_task="task",
        working_summary="Summary.",
        recent_message_ids=retained,
        source_message_index_start=1,
        source_message_index_end=middle_end,
    )


def _ids(messages: list[Message]) -> list[str]:
    """Return the message IDs of a message list."""
    return [message.message_id for message in messages]


class Test04Unit03Context05PresentedView:
    """The history presented to the model."""

    def test_04_unit_03_context_05_view_01_all_messages_without_compaction(
        self,
        make_messages: Callable[..., list[Message]],
    ) -> None:
        # Arrange — no compaction yet
        messages = make_messages(("user", "task"), ("assistant", "step"))

        # Act
        presented = active_messages(messages, None)

        # Assert — the whole journal is presented
        assert _ids(presented) == _ids(messages)

    def test_04_unit_03_context_05_view_02_starts_at_first_retained(
        self,
        make_messages: Callable[..., list[Message]],
    ) -> None:
        # Arrange — the tail from index 3 on is retained
        messages = make_messages(
            ("user", "task"),
            ("assistant", "compacted1"),
            ("assistant", "compacted2"),
            ("assistant", "retained1"),
            ("assistant", "retained2"),
        )
        state = _state(retained=(messages[3].message_id, messages[4].message_id))

        # Act
        presented = active_messages(messages, state)

        # Assert — compacted messages are replaced by the summary
        assert _ids(presented) == _ids(messages[3:])

    def test_04_unit_03_context_05_view_03_keeps_tool_call_with_result(
        self,
        message_factory: MessageFactory,
    ) -> None:
        # Arrange — the retained tail starts at the tool result
        factory = message_factory
        task = factory.create(
            role="user",
            content=[TextPart(text="task")],
            metadata=MessageMetadata(),
        )
        request = factory.create(
            role="assistant",
            content=[TextPart(text="calling")],
            metadata=MessageMetadata(),
        )
        result = factory.create(
            role="tool",
            content=[TextPart(text="output")],
            metadata=MessageMetadata(),
        )
        messages = [task, request, result]
        state = _state(retained=(result.message_id,))

        # Act
        presented = active_messages(messages, state)

        # Assert — the tool call stays with the result it answers
        assert _ids(presented) == _ids([request, result])

    def test_04_unit_03_context_05_view_04_missing_retained_ids_shows_all(
        self,
        make_messages: Callable[..., list[Message]],
    ) -> None:
        # Arrange — a state whose retained IDs are not in this history
        messages = make_messages(
            ("user", "task"),
            ("assistant", "compacted"),
            ("assistant", "newer"),
        )
        state = _state(retained=("unknown",), middle_end=1)

        # Act
        presented = active_messages(messages, state)

        # Assert — nothing is hidden when the cut cannot be located
        assert _ids(presented) == _ids(messages)
