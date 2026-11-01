#!/usr/bin/env python3
"""
The main agent loop for the benchmark agent runtime.

Orchestrates the session loop: builds the provider context, invokes
the model, records the response, commits the assistant message, and
executes tool calls until a terminal condition is reached.

    Important Functions:
        run_agent(user_msg, runtime) -> SessionOutcome: Run the main loop.
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from histrel.core.clock import monotonic_now
from histrel.core.ids import new_message_id
from histrel.core.runtime.context_preparation import prepare_provider_context
from histrel.core.runtime.provider_context import (
    build_model_request,
)
from histrel.core.runtime.runtime import Runtime, log
from histrel.core.runtime.session_setup import (
    arm_tool_call_reminder,
)
from histrel.core.runtime.turn_commits import (
    commit_assistant_message,
)
from histrel.core.runtime.turn_results import (
    execute_turn_tool_calls,
)
from histrel.model.types import ModelResponse
from histrel.terminal import SessionOutcome, SessionStatus

if TYPE_CHECKING:
    from histrel.model.messages import Message
    from histrel.model.types import (
        NormalizedToolCall,
        ProviderMessage,
        ToolDefinition,
    )


def run_agent(user_msg: str, runtime: Runtime) -> SessionOutcome:
    """Run the main agent loop.

    Args:
        user_msg (str): User request message for the session.
        runtime (Runtime): Initialized runtime.

    Returns:
        (SessionOutcome) Terminal session outcome.
    """
    runtime.initialize(user_msg)

    # Fireworks prompt caching and provider session affinity keys.
    session_affinity_key, prompt_cache_key = _session_keys(runtime)

    while True:
        # Evaluate terminal conditions
        if outcome := runtime.evaluate_terminal_conditions():
            return runtime.finalize(outcome)

        # The first loop iteration initializes the agent; every later
        # iteration steps it.
        runtime.iteration_count += 1

        # Invoke the model. Context preparation can itself call the model
        # to summarize a compaction, so it shares this error path.
        call_id = new_message_id()
        started = monotonic_now()
        try:
            tool_defs, provider_messages = _iteration_context(runtime, user_msg)
            response = runtime.model_adapter.complete(
                build_model_request(
                    runtime,
                    tool_defs=tool_defs,
                    provider_messages=provider_messages,
                    prompt_cache_key=prompt_cache_key,
                    session_affinity_key=session_affinity_key,
                )
            )
        except Exception as error:
            return runtime.finalize(runtime.handle_model_error(call_id, error, started))

        _record_model_response(runtime, response, started)

        # Commit the assistant message before executing tools
        assistant_message, repaired_calls = commit_assistant_message(runtime, response)

        # A response without tool calls cannot change the session state.
        # It ends the session when the repository already differs from
        # its committed state; otherwise arm the reminder and continue.
        if not response.tool_calls:
            stall_outcome = _handle_no_tool_call_response(runtime, response)
            if stall_outcome is not None:
                return runtime.finalize(stall_outcome)
            continue

        runtime.tool_call_required = False

        # Execute tool calls and commit results
        _execute_turn(runtime, response, assistant_message, repaired_calls)


def _iteration_context(
    runtime: Runtime,
    user_msg: str,
) -> tuple[tuple[ToolDefinition, ...], tuple[ProviderMessage, ...]]:
    """Prepare one iteration's toolset and provider context.

    Passes the session's static tool set and builds the provider context
    (stateless or react mode), compacting the history first when the
    estimated prompt exceeds the input budget.

    Args:
        runtime (Runtime): The runtime session state.
        user_msg (str): User request message for the session.

    Returns:
        (tuple) The (tool definitions, provider messages) for the
        iteration's model request.
    """
    system_prompt = (
        runtime.config.agent_system_prompt.initialize
        if runtime.iteration_count == 1
        else runtime.config.agent_system_prompt.step
    )

    return prepare_provider_context(
        runtime,
        system_prompt=system_prompt,
        user_msg=user_msg,
        tools=tuple(runtime.ordinary_tools),
    )


def _session_keys(runtime: Runtime) -> tuple[str, str]:
    """Build the session's provider affinity and prompt-cache keys.

    Session affinity: one stable key per session lets the provider
    (or gateway) route all model calls to the same backend. Fireworks
    prompt caching: one stable key per session lets the provider
    reuse cached prompt segments across all model calls.

    Args:
        runtime (Runtime): The runtime session state.

    Returns:
        (tuple[str, str]) The session affinity key and prompt-cache
        key (empty for non-Fireworks providers).
    """
    session_affinity_key: str = str(uuid.uuid4())
    prompt_cache_key: str = (
        str(uuid.uuid4()) if runtime.config.model_provider == "fireworks" else ""
    )
    return session_affinity_key, prompt_cache_key


def _execute_turn(
    runtime: Runtime,
    response: ModelResponse,
    assistant_message: Message,
    repaired_calls: dict[int, NormalizedToolCall],
) -> None:
    """Execute one response's tool calls.

    Args:
        runtime (Runtime): The runtime session state.
        response (ModelResponse): The model response with calls.
        assistant_message (Message): The committed assistant message.
        repaired_calls (dict[int, NormalizedToolCall]): Repaired
            calls by position, from the assistant commit.
    """
    execute_turn_tool_calls(runtime, response, assistant_message, repaired_calls)


def _record_model_response(
    runtime: Runtime,
    response: ModelResponse,
    started: float,
) -> None:
    """Record timing and token usage for one model response.

    Args:
        runtime (Runtime): The runtime session state.
        response (ModelResponse): The model response to record.
        started (float): Monotonic start time of the model call.
    """
    elapsed = monotonic_now() - started
    runtime.timings.add("model_wall_seconds", elapsed)
    runtime.token_ledger.record(
        message_index=runtime.message_factory.next_message_index,
        usage=response.usage,
        provider_request_id=response.provider_request_id,
        call_kind="model",
    )

    runtime.turn_count += 1

    log.debug(
        "model_response_received",
        turn=runtime.turn_count,
        tool_calls=len(response.tool_calls),
        input_tokens=response.usage.input,
        output_tokens=response.usage.output,
        reasoning_tokens=response.usage.reasoning,
    )


def _handle_no_tool_call_response(
    runtime: Runtime,
    response: ModelResponse,
) -> SessionOutcome | None:
    """End the session on a text-only response once the repo has changes.

    A response without tool calls ends the session as succeeded when the
    repository differs from its committed state. Otherwise nothing has
    been produced yet, so the tool-call reminder is armed to steer the
    model back to tool use and the loop continues.

    Args:
        runtime (Runtime): The runtime session state.
        response (ModelResponse): The model response without tool calls.

    Returns:
        (SessionOutcome | None) The success outcome when the repo has
        changes; None to continue the loop.
    """
    has_changes = runtime.repo_checker.has_changes()
    log.debug(
        "no_tool_call_response",
        turn=runtime.turn_count,
        repo_has_changes=has_changes,
        has_text=bool(response.text_parts),
    )
    if has_changes:
        return SessionOutcome(
            status=SessionStatus.SUCCEEDED,
            reason="no_tool_calls_with_repo_changes",
        )
    arm_tool_call_reminder(runtime)
    return None
