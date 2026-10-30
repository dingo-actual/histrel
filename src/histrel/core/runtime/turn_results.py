#!/usr/bin/env python3
"""
Turn-level tool-call execution and result committing.

Executes one response's tool calls: dispatches the calls, commits
duplicate-audit messages for duplicates, commits tool-result messages
for executed calls, and collects the executions for turn effects.

    Important Functions:
        execute_turn_tool_calls(...) -> list[tuple]: Turn executions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from histrel.core.runtime.turn_commits import _safe_json_dict
from histrel.model.messages import (
    Message,
    MessageMetadata,
    MessagePart,
    ToolResultPart,
)
from histrel.model.types import NormalizedToolCall
from histrel.tools.tool import ToolExecution

if TYPE_CHECKING:
    from histrel.core.runtime.runtime import Runtime
    from histrel.model.types import ModelResponse

log = logger.bind(component="bench_agent.runtime")


def execute_turn_tool_calls(
    runtime: Runtime,
    response: ModelResponse,
    assistant_message: Message,
    repaired_calls: dict[int, NormalizedToolCall],
) -> list[tuple[NormalizedToolCall, ToolExecution]]:
    """Execute one response's tool calls and commit their results.

    Commits an audit message for duplicates (with the tool-call id
    so the record stays linked) and a tool-result message for every
    other call, then collects the executions for turn effects.

    Args:
        runtime (Runtime): The runtime session state.
        response (ModelResponse): The model response with calls.
        assistant_message (Message): The committed assistant message.
        repaired_calls (dict[int, NormalizedToolCall]): Repaired
            calls by position, from the assistant commit.

    Returns:
        (list[tuple[NormalizedToolCall, ToolExecution]]) This turn's
        executions.
    """
    turn_executions: list[tuple[NormalizedToolCall, ToolExecution]] = []
    for idx, raw_call in enumerate(response.tool_calls):
        # Dispatch with the repaired arguments when the commit
        # recovered them; the call keeps its id so messages stay linked.
        tool_call = repaired_calls.get(idx, raw_call)
        execution = runtime._execute_tool_call(tool_call)

        if execution.status == "duplicate":
            _commit_duplicate_audit(runtime, tool_call, assistant_message)
            continue

        log.debug(
            "tool_call_executed",
            tool=tool_call.name,
            status=execution.status,
            tool_call_count=runtime.tool_call_count,
        )
        runtime._commit_message(
            _tool_result_message(runtime, tool_call, execution, assistant_message)
        )
        turn_executions.append((tool_call, execution))
    return turn_executions


def _commit_duplicate_audit(
    runtime: Runtime,
    tool_call: NormalizedToolCall,
    assistant_message: Message,
) -> None:
    """Commit the audit message for a duplicate tool call.

    Args:
        runtime (Runtime): The runtime session state.
        tool_call (NormalizedToolCall): The duplicate call.
        assistant_message (Message): The committed assistant message.
    """
    audit_part = ToolResultPart(
        tool_id=tool_call.id,
        tool_name=tool_call.name,
        arguments=_safe_json_dict(tool_call.arguments),
        success=False,
        output="Duplicate tool call: skipped; no state changed.",
        error="Duplicate tool call: skipped; no state changed.",
        informative=False,
        iter_num=runtime.iteration_count,
    )
    runtime._commit_message(
        runtime.message_factory.create(
            role="tool",
            content=[audit_part],
            metadata=MessageMetadata(parent_id=str(assistant_message.message_id)),
        )
    )


def _tool_result_message(
    runtime: Runtime,
    tool_call: NormalizedToolCall,
    execution: ToolExecution,
    assistant_message: Message,
) -> Message:
    """Build the tool-result message for one execution.

    Args:
        runtime (Runtime): The runtime session state.
        tool_call (NormalizedToolCall): The executed call.
        execution (ToolExecution): The execution result.
        assistant_message (Message): The committed assistant message.

    Returns:
        (Message) The tool result message to commit.
    """
    result_content: list[MessagePart] = [
        ToolResultPart(
            tool_id=tool_call.id,
            tool_name=tool_call.name,
            # Malformed arguments must not crash the journal.
            arguments=_safe_json_dict(tool_call.arguments),
            success=(execution.status == "ok"),
            output=str(execution.output) if execution.output is not None else None,
            error=str(execution.output) if execution.status != "ok" else None,
            informative=bool(execution.metadata.get("informative", False)),
            iter_num=runtime.iteration_count,
        )
    ]
    return runtime.message_factory.create(
        role="tool",
        content=result_content,
        metadata=MessageMetadata(parent_id=str(assistant_message.message_id)),
    )
