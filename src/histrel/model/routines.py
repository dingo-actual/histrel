#!/usr/bin/env python3
"""
General LLM routines for the benchmark agent.

A routine here is a stateless single-shot LLM call (see
``routine_executor``): the decision inputs are rendered into one user
message, and the constrained response is parsed to the result type.

    Important Classes:
        None

    Important Functions:
        fix_json_arguments(malformed_json, tool_definition, adapter,
            config, usage_recorder) -> str: Reconstruct malformed
            tool-call arguments JSON with one schema-constrained LLM
            call, recording the routine call's token usage.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from loguru import logger

from histrel.core.config import AgentConfig
from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
)
from histrel.model.render import render_fix_json
from histrel.model.routine_executor import RoutineExecutor
from histrel.model.types import ModelResponse

# Bound logger for LLM routine events.
log = logger.bind(component="bench_agent.model.routines")

# System instructions for the JSON-repair routine: infer the intended
# tool-call arguments from the malformed JSON and the schema, and
# return one bare JSON object conforming to the schema.
_JSON_FIX_SYSTEM_PROMPT = (
    "You are a JSON repair system that fixes malformed JSON arguments "
    "from tool calls.\n"
    "\n"
    "Infer what the intended JSON was for the malformed JSON, based on "
    "the tool JSON schema provided in the user message, and return the "
    "intended JSON correctly structured: one bare JSON object conforming "
    "to the schema. Keep every value from the malformed JSON that the "
    "schema accepts. Return no other text."
)


def fix_json_arguments(
    malformed_json: str,
    tool_definition: dict[str, Any],
    adapter: OpenAICompatAdapter,
    config: AgentConfig,
    usage_recorder: Callable[[ModelResponse], None],
) -> str:
    """Reconstruct malformed tool-call arguments JSON with one LLM call.

    Renders the malformed JSON and the tool's JSON schema (from
    ``tool_definition``'s ``function.parameters``) into one routine
    user message and makes a single schema-constrained call: the
    request carries ``response_format={"type": "json_schema",
    "json_schema": {"name", "schema"}}`` so the provider enforces the
    tool's schema during generation. The reasoning-effort setting is
    omitted (json_schema disables reasoning output, so the provider
    default applies). Adapter and parser errors propagate to the
    caller; the JSON-repair fallback in ``tools.json`` contains them.

    Args:
        malformed_json (str): The malformed JSON arguments string from
            the tool call.
        tool_definition (dict[str, Any]): The tool's OpenAI-compatible
            definition (``Tool.definition``); the function name and
            parameters schema shape the constrained call.
        adapter (OpenAICompatAdapter): Adapter for the routine call.
        config (AgentConfig): Config supplying the routine call's
            sampling settings.
        usage_recorder (Callable[[ModelResponse], None]): Hook that
            receives the routine call's model response before
            parsing; the runtime wires it to the token ledger.

    Returns:
        (str) The model's response text: the intended arguments as a
        JSON object string.

    Raises:
        Exception: Whatever the routine's adapter call or parser
            raises, propagated unchanged.

    Example Usage:
        Needs a live adapter for the repair call, so the example is not
        executed:

        ```python
            >>> fix_json_arguments(  # xdoctest: +SKIP
            ...     '{"path": "main.py"', read_tool.definition,
            ...     adapter, config, recorder)
            '{"path": "main.py"}'
        ```
    """
    function = tool_definition["function"]
    tool_name = function["name"]
    parameters_schema = function["parameters"]
    user_message = render_fix_json(
        malformed_json=malformed_json,
        tool_json_schema=json.dumps(parameters_schema, indent=2),
    )
    executor = RoutineExecutor(adapter, usage_recorder)
    # str.strip: the constrained response is one JSON object, with no
    # surrounding text to parse away.
    result = executor.run(
        system_prompt=_JSON_FIX_SYSTEM_PROMPT,
        user_message=user_message,
        parse_response=str.strip,
        temperature=config.model_temperature,
        reasoning_effort=None,
        max_tokens_output=config.model_max_tokens_output,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": tool_name, "schema": parameters_schema},
        },
    )
    log.info(
        "Reconstructed malformed JSON with LLM routine",
        tool_name=tool_name,
        routine_input_tokens=executor.last_usage.input,
        routine_output_tokens=executor.last_usage.output,
    )
    return result
