#!/usr/bin/env python3
"""
Assistant-message committing for the benchmark agent runtime.

Repairs tool-call argument JSON so malformed calls cannot crash the
loop, and commits the assistant message with the repaired arguments.

    Important Functions:
        _repaired_call(runtime, tool_call, index, repaired) -> dict:
            Repair one call's arguments.
        _safe_json_dict(raw: str) -> dict | None: Parse args for the journal.
        commit_assistant_message(runtime, response) -> tuple: Commit.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

from histrel.model.messages import (
    Message,
    MessageMetadata,
    MessagePart,
    TextPart,
    TokenUsage,
    ToolRequestPart,
)
from histrel.model.routines import fix_json_arguments
from histrel.model.types import NormalizedToolCall
from histrel.tools.json import repair_and_parse

if TYPE_CHECKING:
    from histrel.core.runtime.runtime import Runtime
    from histrel.model.types import ModelResponse


def _repaired_call(
    runtime: Runtime,
    tool_call: NormalizedToolCall,
    index: int,
    repaired_calls: dict[int, NormalizedToolCall],
) -> dict[str, Any] | None:
    """Repair one tool call's arguments with its tool's fallback.

    The called tool's schema (from the registry lookup) constrains the
    fix_json_arguments LLM fallback, which runs once here: the repaired
    arguments are recorded in ``repaired_calls`` for dispatch and the
    journal commits, so malformed JSON cannot crash the loop. Calls
    whose arguments stay unrepairable return None; the tool-call loop's
    existing error path reports the failure.

    Args:
        runtime (Runtime): The session runtime supplying the adapter,
            config, and tool registry.
        tool_call (NormalizedToolCall): The model's tool call.
        index (int): Position of the call in the model response.
        repaired_calls (dict[int, NormalizedToolCall]): Output map of
            position to repaired, serialized calls.

    Returns:
        (dict[str, Any] | None) The parsed arguments dict, or None when
        no JSON object can be recovered.
    """
    tool = runtime._get_tool_by_name(tool_call.name)
    llm_fix: Callable[[str], str] | None = None
    if tool is not None:
        llm_fix = partial(
            fix_json_arguments,
            tool_definition=tool.definition,
            adapter=runtime.model_adapter,
            config=runtime.config,
            usage_recorder=runtime._record_routine_usage,
        )
    parsed = repair_and_parse(tool_call.arguments, llm_fix=llm_fix)
    if not isinstance(parsed, dict):
        return None
    repaired_calls[index] = NormalizedToolCall(
        id=tool_call.id,
        name=tool_call.name,
        arguments=json.dumps(parsed),
    )
    return parsed


def _safe_json_dict(raw: str) -> dict[str, Any] | None:
    """Parse a tool-arguments string for the journal; None when unusable.

    Args:
        raw (str): Tool-call arguments string.

    Returns:
        (dict[str, Any] | None) The parsed dict, or None when the string
        is empty, unparseable, or not a JSON object.
    """
    if not raw:
        return None
    try:
        parsed: Any = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def commit_assistant_message(
    runtime: Runtime,
    response: ModelResponse,
) -> tuple[Message, dict[int, NormalizedToolCall]]:
    """Commit one model response's assistant message to the journal.

    Builds the assistant message with its text parts and tool-request
    parts, and returns the committed message plus the repaired calls by
    position. Calls are repaired once here with their tool's fix_json
    fallback: malformed JSON must not crash the journal commit, and the
    repair is reused for dispatch.

    Args:
        runtime (Runtime): The runtime session state.
        response (ModelResponse): The model response to commit.

    Returns:
        (tuple[Message, dict[int, NormalizedToolCall]]) The committed
        assistant message and repaired calls by position.
    """
    assistant_content: list[MessagePart] = []
    if response.text_parts:
        assistant_content.append(TextPart(text="".join(response.text_parts)))
    repaired_calls: dict[int, NormalizedToolCall] = {}
    assistant_content += _repaired_tool_request_parts(runtime, response, repaired_calls)

    assistant_message = runtime.message_factory.create(
        role="assistant",
        content=assistant_content,
        metadata=MessageMetadata(
            model_id=runtime.config.model_name,
            provider_id=runtime.config.model_provider,
            tokens=TokenUsage(
                input=response.usage.input,
                output=response.usage.output,
                reasoning=response.usage.reasoning,
                total=response.usage.total,
                cached=response.usage.cached,
            ),
            has_reasoning=response.usage.reasoning > 0,
            tool_count=len(response.tool_calls),
        ),
    )
    runtime._commit_message(assistant_message)
    return assistant_message, repaired_calls


def _repaired_tool_request_parts(
    runtime: Runtime,
    response: ModelResponse,
    repaired_calls: dict[int, NormalizedToolCall],
) -> list[MessagePart]:
    """Build the tool-request parts with repaired, journalable arguments.

    Args:
        runtime (Runtime): The runtime session state.
        response (ModelResponse): The model response to commit.
        repaired_calls (dict[int, NormalizedToolCall]): Output map of
            position to repaired, serialized calls.

    Returns:
        (list[MessagePart]) The tool-request parts for the message.
    """
    return [
        ToolRequestPart(
            tool_id=tc.id,
            tool_name=tc.name,
            arguments=_repaired_call(runtime, tc, idx, repaired_calls),
        )
        for idx, tc in enumerate(response.tool_calls)
    ]
