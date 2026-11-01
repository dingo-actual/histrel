#!/usr/bin/env python3
"""
Presented-history view for the benchmark agent.

The model is not shown the whole journal once a compaction exists: the
compaction summary stands in for the compacted middle, and the presented
history becomes the retained tail plus every message committed since.
Both prompt-building modes and the compactor read that view here.

    Important Functions:
        active_messages(messages, state) -> list[Message]: Presented view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from histrel.context.compactor import CompactedState
    from histrel.model.messages import Message


def active_messages(
    messages: list[Message],
    state: CompactedState | None,
) -> list[Message]:
    """Return the history currently presented to the model.

    Before the first compaction this is every journal message. After a
    compaction it is the retained tail plus the messages committed since.
    The cut moves back over tool results so a result never loses the tool
    call it answers.

    Args:
        messages (list[Message]): Journal messages in commit order.
        state (CompactedState | None): Latest compaction, if any.

    Returns:
        (list[Message]) The messages the model is shown.
    """
    if state is None:
        return messages
    cut = _retained_cut(messages, state)
    while cut > 0 and messages[cut].role == "tool":
        cut -= 1
    return messages[cut:]


def _retained_cut(messages: list[Message], state: CompactedState) -> int:
    """Find the index of the first message the compaction retained.

    Args:
        messages (list[Message]): Journal messages in commit order.
        state (CompactedState): Latest compaction.

    Returns:
        (int) Index of the first retained message, or 0 when none of the
        retained IDs are in this history. Showing everything is safe:
        the prompt may exceed the threshold again, but nothing is lost.
    """
    retained = set(state.recent_message_ids)
    for index, message in enumerate(messages):
        if message.message_id in retained:
            return index
    return 0
