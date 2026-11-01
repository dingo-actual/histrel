#!/usr/bin/env python3
"""
Context preparation for the benchmark agent loop.

Builds one iteration's provider context and keeps it within the model's
input budget. The presented history is the active view: the whole journal
until the first compaction, then the retained tail plus every message
committed since. When the estimated outgoing prompt reaches the
compaction threshold, the compactor summarizes the compacted middle, the
resulting state is stored on the runtime, and the prompt is rebuilt from
it.

    Important Functions:
        usable_input_budget(config) -> int: Input budget for history.
        estimate_request_tokens(tool_defs, provider_messages) -> int: ...
        compact_context(runtime, *, system_prompt, original_task) -> None.
        prepare_provider_context(runtime, *, system_prompt, user_msg,
            tools) -> tuple: Tool definitions and provider messages.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

from histrel.context.compactor import estimate_tokens
from histrel.context.context_view import active_messages
from histrel.core.journal import load_journal
from histrel.core.runtime.provider_context import build_provider_context
from histrel.model.types import ProviderMessage, ToolDefinition

if TYPE_CHECKING:
    from histrel.core.config import AgentConfig
    from histrel.core.runtime.runtime import Runtime
    from histrel.tools.tool import Tool

log = logger.bind(component="bench_agent.runtime")


def usable_input_budget(config: AgentConfig) -> int:
    """Return the input budget left for history.

    The context window minus the output allowance, because the model may
    fill its reserved output tokens in the same request.

    Args:
        config (AgentConfig): Agent configuration.

    Returns:
        (int) Usable input budget in tokens.
    """
    return config.model_context_limit_tokens - config.model_max_tokens_output


def estimate_request_tokens(
    tool_defs: tuple[ToolDefinition, ...],
    provider_messages: tuple[ProviderMessage, ...],
) -> int:
    """Estimate the outgoing request size, tool definitions included.

    Args:
        tool_defs (tuple[ToolDefinition, ...]): Tool definitions sent.
        provider_messages (tuple[ProviderMessage, ...]): Messages sent.

    Returns:
        (int) Estimated input tokens.
    """
    parts = [_message_text(message) for message in provider_messages]
    parts.append(json.dumps([tool.model_dump() for tool in tool_defs], sort_keys=True))
    return estimate_tokens("\n".join(parts))


def _message_text(message: ProviderMessage) -> str:
    """Return the text of one provider message, tool calls included.

    Args:
        message (ProviderMessage): Provider message.

    Returns:
        (str) Content plus serialized tool calls.
    """
    if not message.tool_calls:
        return message.content
    return f"{message.content}\n{json.dumps(message.tool_calls, sort_keys=True)}"


def compact_context(
    runtime: Runtime,
    *,
    system_prompt: str,
    original_task: str,
) -> None:
    """Compact the presented history and store the resulting state.

    The state replaces any earlier one; the earlier summary is folded
    into the new summary so no compacted work is lost. When the presented
    history has no compactable middle the earlier state is kept and no
    summary call is made.

    Args:
        runtime (Runtime): The runtime session state.
        system_prompt (str): System prompt for this iteration.
        original_task (str): Original task text.
    """
    previous = runtime.compacted_state
    state = runtime.compactor.compact(
        messages=active_messages(
            list(load_journal(runtime.journal._path)),
            previous,
        ),
        system_prompt=system_prompt,
        original_task=original_task,
        recent_tail_count=runtime.config.compaction.compact_retain_tail,
        model_adapter=runtime.model_adapter,
        token_ledger=runtime.token_ledger,
        prior_summary="" if previous is None else previous.working_summary,
    )
    if state.source_message_index_start < 0:
        log.debug("compaction_skipped", reason="no_compactable_middle")
        return
    runtime.compacted_state = state


def prepare_provider_context(
    runtime: Runtime,
    *,
    system_prompt: str,
    user_msg: str,
    tools: tuple[Tool, ...],
) -> tuple[tuple[ToolDefinition, ...], tuple[ProviderMessage, ...]]:
    """Build one iteration's provider context, compacting when needed.

    The estimated prompt size is checked against the usable input budget.
    A compaction rebuilds the context from the new state, so the request
    that follows carries the summary instead of the compacted middle.

    Args:
        runtime (Runtime): The runtime session state.
        system_prompt (str): System prompt for this iteration.
        user_msg (str): Task user message passed to the agent loop.
        tools (tuple[Tool, ...]): The session's tools.

    Returns:
        (tuple) The (tool definitions, provider messages) for the
        iteration's model request.
    """
    tool_defs, provider_messages = build_provider_context(
        runtime,
        system_prompt=system_prompt,
        user_msg=user_msg,
        tools=tools,
    )
    estimated = estimate_request_tokens(tool_defs, provider_messages)
    budget = usable_input_budget(runtime.config)
    if not runtime.compactor.should_compact(estimated, budget):
        return tool_defs, provider_messages

    log.debug(
        "compaction_triggered",
        estimated_tokens=estimated,
        usable_context=budget,
        iteration=runtime.iteration_count,
    )
    compact_context(runtime, system_prompt=system_prompt, original_task=user_msg)
    return build_provider_context(
        runtime,
        system_prompt=system_prompt,
        user_msg=user_msg,
        tools=tools,
    )
