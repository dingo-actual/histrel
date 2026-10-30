#!/usr/bin/env python3
"""
Content part models for the canonical message model.

    Important Classes:
        TextPart: Plain text content.
        ReasoningPart: Extended reasoning content.
        FilePart: File or image attachment reference.
        ToolRequestPart: Tool execution request.
        ToolResultPart: Tool execution result.
        CompactionPart: Conversation history compaction event.
        BeginSessionPart: Session start event.
        EndSessionPart: Session end event.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from histrel.model.messages.metadata import Timing, TokenUsage


class TextPart(BaseModel):
    """Plain text content.

    Instance Attributes:
        type (Literal["text"]): Content type discriminator.
        text (str): The text content.
        timing (Timing | None): When this content was generated.
    """

    type: Literal["text"] = "text"
    text: str = ""
    timing: Timing | None = None


class ReasoningPart(BaseModel):
    """Extended reasoning or thinking content.

    Instance Attributes:
        type (Literal["reasoning"]): Content type discriminator.
        text (str): The reasoning content.
        timing (Timing | None): When this reasoning was generated.
    """

    type: Literal["reasoning"] = "reasoning"
    text: str = ""
    timing: Timing | None = None


class FilePart(BaseModel):
    """File or image attachment reference.

    Instance Attributes:
        type (Literal["file"]): Content type discriminator.
        reference (str): File identifier, path, URL, or content hash.
        filename (str | None): Original filename if available.
        mime_type (str | None): MIME type.
        source (str | None): Source context.
    """

    type: Literal["file"] = "file"
    reference: str
    filename: str | None = None
    mime_type: str | None = None
    source: str | None = None


class ToolRequestPart(BaseModel):
    """Tool execution request.

    Instance Attributes:
        type (Literal["tool_request"]): Content type discriminator.
        tool_id (str): Unique tool call identifier.
        tool_name (str): Name of the tool being invoked.
        arguments (dict[str, Any] | None): Tool arguments as key-value pairs.
        status (str): Execution status.
        timing (Timing | None): Tool execution timing.
    """

    type: Literal["tool_request"] = "tool_request"
    tool_id: str
    tool_name: str
    arguments: dict[str, Any] | None = None
    status: Literal["pending", "running", "completed", "error"] = "pending"
    timing: Timing | None = None


class ToolResultPart(BaseModel):
    """Tool execution result.

    Uses ``success: bool`` and ``error: str | None`` instead of a status
    enum. Internal ``denied`` and ``timeout`` outcomes map to
    ``success=false`` with the reason in ``error``.

    The ``tool_name`` and ``arguments`` fields echo the original tool
    call, so the LLM can correlate a tool response with the request that
    produced it without scanning prior assistant messages.

    Instance Attributes:
        type (Literal["tool_response"]): Content type discriminator.
        tool_id (str): Tool call identifier matching the request.
        tool_name (str | None): Name of the tool that was invoked.
        arguments (dict[str, Any] | None): Arguments passed to the tool.
        success (bool): Whether tool execution succeeded.
        output (str | None): Tool result output as string.
        error (str | None): Error message if execution failed.
        informative (bool): True when the call was correctly specified
            but failed with informative output. Such failures carry a
            prose reason in ``output`` and are included as observations
            in later iterations.
        timing (Timing | None): Tool execution timing.
    """

    type: Literal["tool_response"] = "tool_response"
    tool_id: str
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    success: bool
    output: str | None = None
    error: str | None = None
    informative: bool = False
    timing: Timing | None = None
    iter_num: int


class CompactionPart(BaseModel):
    """Conversation history compaction event.

    Instance Attributes:
        type (Literal["compaction"]): Content type discriminator.
        original_message_count (int): Number of messages before compaction.
        compacted_message_count (int): Number of messages after compaction.
        summary (str | None): Optional summary of what was compacted.
        tokens (TokenUsage | None): Token usage for the compaction operation.
    """

    type: Literal["compaction"] = "compaction"
    original_message_count: int
    compacted_message_count: int
    summary: str | None = None
    tokens: TokenUsage | None = None


class BeginSessionPart(BaseModel):
    """Session start event.

    Instance Attributes:
        type (Literal["begin_session"]): Content type discriminator.
        session_name (str | None): Optional name for the session.
        agent (str | None): Initial agent/mode for the session.
        model (str | None): Model being used for the session.
    """

    type: Literal["begin_session"] = "begin_session"
    session_name: str | None = None
    agent: str | None = None
    model: str | None = None


class EndSessionPart(BaseModel):
    """Session end event.

    Instance Attributes:
        type (Literal["end_session"]): Content type discriminator.
        reason (str | None): Reason for session ending.
        summary (str | None): Optional summary of the session.
    """

    type: Literal["end_session"] = "end_session"
    reason: Literal["completed", "error", "user_terminated", "timeout"] | None = None
    summary: str | None = None


MessagePart = Annotated[
    TextPart
    | ReasoningPart
    | FilePart
    | ToolRequestPart
    | ToolResultPart
    | CompactionPart
    | BeginSessionPart
    | EndSessionPart,
    Field(discriminator="type"),
]
