#!/usr/bin/env python3
"""
Tool-call execution for the benchmark agent runtime.

Executes single tool calls through the resolution, guard, duplicate,
and dispatch gates. Turn-level execution and result committing live in
``turn_results.py``.

    Important Functions:
        execute_tool_call(runtime, tool_call) -> ToolExecution: One call.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from loguru import logger

from histrel.core.clock import monotonic_now
from histrel.core.runtime.edit_arguments import repair_edit_arguments
from histrel.model.types import NormalizedToolCall
from histrel.tools.repository import EditFileTool
from histrel.tools.tool import Tool, ToolContext, ToolExecution

if TYPE_CHECKING:
    from histrel.core.runtime.runtime import Runtime

log = logger.bind(component="bench_agent.runtime")


def execute_tool_call(
    runtime: Runtime,
    tool_call: NormalizedToolCall,
) -> ToolExecution:
    """Execute a single tool call.

    Duplicate calls (read, list_dir, search, glob) return
    duplicate-status executions that the caller leaves unrecorded.

    Args:
        runtime (Runtime): The runtime session state.
        tool_call (NormalizedToolCall): The tool call to execute.

    Returns:
        (ToolExecution) Execution result.
    """
    runtime.tool_call_count += 1
    started = monotonic_now()

    tool = runtime._get_tool_by_name(tool_call.name)
    if tool is None:
        return ToolExecution(
            status="error",
            output="That action is not available.",
            metadata={"tool_name": tool_call.name},
        )

    arguments, resolution_error = _resolve_arguments(runtime, tool, tool_call, started)
    if resolution_error is not None:
        return resolution_error

    duplicate = _duplicate_gate(runtime, tool, tool_call, arguments)
    if duplicate is not None:
        return duplicate

    return _dispatch_timed(runtime, tool, tool_call, arguments, started)


def _dispatch_timed(
    runtime: Runtime,
    tool: Tool,
    tool_call: NormalizedToolCall,
    arguments: dict[str, Any],
    started: float,
) -> ToolExecution:
    """Dispatch one tool call, attributing its wall time to the tool category."""
    dispatch_started = monotonic_now()
    execution = _dispatch(runtime, tool, tool_call, arguments, started)
    runtime.timings.add(
        "repository_tool_wall_seconds", monotonic_now() - dispatch_started
    )
    return execution


def _resolve_arguments(
    runtime: Runtime,
    tool: Tool,
    tool_call: NormalizedToolCall,
    started: float,
) -> tuple[dict[str, Any], ToolExecution | None]:
    """Resolve tool-call arguments into the invocation dictionary.

    The edit tool's arguments go through the repair routines in
    ``tools/json.py``, which fall back to the fix_json_arguments LLM
    routine (constrained by the edit tool's schema) for defects no
    local repair can fix. An error ToolExecution is returned only when
    the routine cannot recover the arguments either.

    Args:
        runtime (Runtime): The runtime session state.
        tool (Tool): The resolved tool; EditFileTool for edit calls.
        tool_call (NormalizedToolCall): The tool call to resolve.
        started (float): Execution start time for metrics.

    Returns:
        (tuple[dict[str, Any], ToolExecution | None]): The resolved
        arguments, plus an error execution when resolution failed.
    """
    try:
        if isinstance(tool, EditFileTool):
            result = repair_edit_arguments(runtime, tool_call.arguments)
            if isinstance(result, str):
                elapsed = int((monotonic_now() - started) * 1000)
                runtime.tool_metrics.record(
                    tool.name, elapsed_ms=elapsed, success=False
                )
                return {}, ToolExecution(status="error", output=result, metadata={})
            return result, None
        return (json.loads(tool_call.arguments) if tool_call.arguments else {}), None
    except json.JSONDecodeError:
        return {}, ToolExecution(
            status="error",
            output=f"Invalid JSON arguments: {tool_call.arguments}",
            metadata={"tool_name": tool_call.name},
        )


def _duplicate_gate(
    runtime: Runtime,
    tool: Tool,
    tool_call: NormalizedToolCall,
    arguments: dict[str, Any],
) -> ToolExecution | None:
    """Deny duplicate calls via the duplicate guard; else None.

    Args:
        runtime (Runtime): The runtime session state.
        tool (Tool): The resolved tool.
        tool_call (NormalizedToolCall): The tool call.
        arguments (dict[str, Any]): Resolved arguments.

    Returns:
        (ToolExecution | None) Duplicate execution when the call is
        a duplicate; None when it may proceed.
    """
    duplicate = runtime.duplicate_guard.check(
        tool_name=tool_call.name,
        arguments=arguments,
        llm_call_index=runtime.turn_count,
    )
    if duplicate is None:
        return None
    log.debug(
        "tool_call_duplicate_detected",
        tool=tool.name,
        reason=duplicate.reason,
        tool_call_count=runtime.tool_call_count,
    )
    return ToolExecution(status="duplicate", output="", metadata={})


def _dispatch(
    runtime: Runtime,
    tool: Tool,
    tool_call: NormalizedToolCall,
    arguments: dict[str, Any],
    started: float,
) -> ToolExecution:
    """Execute the tool with resolved arguments, containing crashes.

    Args:
        runtime (Runtime): The runtime session state.
        tool (Tool): The resolved tool.
        tool_call (NormalizedToolCall): The tool call.
        arguments (dict[str, Any]): Resolved arguments.
        started (float): Execution start time for metrics.

    Returns:
        (ToolExecution) The execution result.
    """
    ctx = ToolContext(
        session=runtime.session,
        message_index=runtime.message_factory.next_message_index,
    )
    try:
        execution = tool.execute(arguments, ctx)
    except Exception as exc:
        # Contain unexpected tool crashes as error results so one bad
        # call cannot destroy an expensive live session; the model can
        # recover from the error output on its next turn.
        elapsed = int((monotonic_now() - started) * 1000)
        runtime.tool_metrics.record(tool.name, elapsed_ms=elapsed, success=False)
        return ToolExecution(
            status="error",
            output=f"tool crashed: {type(exc).__name__}: {exc}",
            metadata={"tool_name": tool.name},
        )
    elapsed = int((monotonic_now() - started) * 1000)
    if execution.status != "duplicate":
        _record_success(runtime, tool, tool_call, arguments, execution, elapsed)
    return execution


def _record_success(
    runtime: Runtime,
    tool: Tool,
    tool_call: NormalizedToolCall,
    arguments: dict[str, Any],
    execution: ToolExecution,
    elapsed: int,
) -> None:
    """Record metrics and duplicate-guard entries for an execution.

    Args:
        runtime (Runtime): The runtime session state.
        tool (Tool): The resolved tool.
        tool_call (NormalizedToolCall): The tool call.
        arguments (dict[str, Any]): Resolved arguments.
        execution (ToolExecution): The execution result.
        elapsed (int): Execution elapsed milliseconds.
    """
    runtime.tool_metrics.record(
        tool.name,
        elapsed_ms=elapsed,
        success=(execution.status == "ok"),
    )
    if execution.status == "ok":
        runtime.duplicate_guard.record(
            tool_name=tool_call.name,
            arguments=arguments,
            llm_call_index=runtime.turn_count,
        )
