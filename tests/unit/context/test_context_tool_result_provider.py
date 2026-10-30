#!/usr/bin/env python3
"""Unit tests for tool-result provider messages."""

from __future__ import annotations

from histrel.context.builder import message_to_provider
from histrel.model.messages import (
    Message,
    MessageFactory,
    MessageMetadata,
    ToolRequestPart,
)
from histrel.model.types import ProviderMessage
from tests.unit.context.helpers import (
    make_message,
    make_tool_result_message,
)


def _convert(messages: list[Message]) -> list[ProviderMessage]:
    """Convert journal messages, dropping the inconvertible ones."""
    converted = (message_to_provider(msg) for msg in messages)
    return [pm for pm in converted if pm is not None]


class Test04Unit01Context04ToolResultProvider:
    """Tool result provider message includes the tool name, not the arguments."""

    def test_04_unit_01_context_04_tool_result_01_includes_tool_name_in_content(
        self,
        message_factory: MessageFactory,
    ) -> None:
        # Arrange — a tool result message with tool_name and arguments
        messages = [
            make_message(message_factory, "user", "Fix the bug"),
            message_factory.create(
                role="assistant",
                content=[
                    ToolRequestPart(
                        tool_id="call_1",
                        tool_name="read_file",
                        arguments={"path": "src/main.py"},
                    )
                ],
                metadata=MessageMetadata(),
            ),
            make_tool_result_message(
                message_factory,
                tool_id="call_1",
                tool_name="read_file",
                arguments={"path": "src/main.py"},
                output="file contents",
            ),
        ]

        # Act
        provider_messages = _convert(messages)

        # Assert — the tool-role provider message content contains the
        # tool name header and the tool output, but not the call
        # arguments (kwargs are no longer echoed).
        tool_msgs = [m for m in provider_messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        content = tool_msgs[0].content
        assert "Tool: read_file" in content
        assert content.endswith("file contents")
        assert "src/main.py" not in content
        assert "Arguments:" not in content

    def test_04_unit_01_context_04_tool_result_02_excludes_arguments_from_content(
        self,
        message_factory: MessageFactory,
    ) -> None:
        """Tool call arguments are not replayed in the tool message content."""
        # Arrange — arguments with multiple keys must not appear in the content
        messages = [
            make_message(message_factory, "user", "Fix the bug"),
            make_tool_result_message(
                message_factory,
                tool_id="call_2",
                tool_name="write_file",
                arguments={"path": "out.py", "content": "print(1)"},
                output="ok",
            ),
        ]

        # Act
        provider_messages = _convert(messages)

        # Assert — the tool name and output appear; the argument values do not
        tool_msgs = [m for m in provider_messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        content = tool_msgs[0].content
        assert "Tool: write_file" in content
        assert content.endswith("ok")
        assert "out.py" not in content
        assert "print(1)" not in content

    def test_04_unit_01_context_04_tool_result_03_preserves_tool_call_id(
        self,
        message_factory: MessageFactory,
    ) -> None:
        # Arrange — the tool_call_id must still be set for API compliance
        messages = [
            make_message(message_factory, "user", "Fix the bug"),
            make_tool_result_message(
                message_factory,
                tool_id="call_xyz",
                tool_name="grep",
                arguments={"pattern": "TODO"},
                output="3 matches",
            ),
        ]

        # Act
        provider_messages = _convert(messages)

        # Assert — tool_call_id is still set correctly
        tool_msgs = [m for m in provider_messages if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "call_xyz"
