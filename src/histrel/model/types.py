#!/usr/bin/env python3
"""
Provider-independent model request and response types.

Defines the normalized types that the agent loop uses to interact with
the model adapter. Provider-specific formats are converted to and from
these types at the adapter edge.

    Important Classes:
        ProviderMessage: A single message in the provider format.
        ToolDefinition: A tool definition in the provider format.
        ModelRequest: Provider-independent model completion request.
        NormalizedToolCall: A normalized tool call from the model response.
        Usage: Token usage statistics for a single model call.
        ModelResponse: Provider-independent model completion response.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class ProviderMessage(BaseModel):
    """A single message in the OpenAI-compatible provider format.

    Instance Attributes:
        role (str): Message role (system, user, assistant, tool).
        content (str): Message content.
        tool_calls (list[dict[str, Any]] | None): Tool calls from assistant messages.
        tool_call_id (str | None): Tool call ID for tool-role messages.
    """

    role: str
    content: str = ""
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None


class ToolDefinition(BaseModel):
    """A tool definition in the OpenAI-compatible provider format.

    Instance Attributes:
        type (str): Tool type (always "function").
        function (dict[str, Any]): Function definition with name, description, parameters.
    """

    type: str = "function"
    function: dict[str, Any]


class ModelRequest(BaseModel):
    """Provider-independent model completion request.

    Instance Attributes:
        messages (tuple[ProviderMessage, ...]): Conversation messages.
        tools (tuple[ToolDefinition, ...]): Available tools.
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling parameter.
        reasoning_effort (str | None): Reasoning effort level; None
            omits the setting from the provider request (single-shot
            routines use this to skip the setting). Defaults to None.
        max_tokens_output (int | None): Maximum output tokens.
        prompt_cache_key (str | None): Provider prompt cache key set by
            the runtime once per session (fireworks prompt caching).
        response_format (dict[str, Any] | None): Provider response
            format constraint, e.g. a ``json_schema`` structured-output
            payload; None omits the setting from the provider request.
        extra_headers (dict[str, str] | None): Extra HTTP headers set by
            the runtime once per session (e.g. the x-session-affinity
            header); None omits them from the provider request.
    """

    messages: tuple[ProviderMessage, ...]
    tools: tuple[ToolDefinition, ...] = ()
    temperature: float = 1.0
    top_p: float = 1.0
    reasoning_effort: str | None = None
    max_tokens_output: int | None = None
    prompt_cache_key: str | None = None
    response_format: dict[str, Any] | None = None
    extra_headers: dict[str, str] | None = None


class NormalizedToolCall(BaseModel):
    """A normalized tool call extracted from a model response.

    Instance Attributes:
        id (str): Tool call identifier.
        name (str): Tool function name.
        arguments (str): JSON-encoded arguments string.
    """

    id: str
    name: str
    arguments: str = ""


class Usage(BaseModel):
    """Token usage statistics for a single model call.

    The ``cached`` field is a subset of ``input``, not additive.

    Instance Attributes:
        input (int): Input tokens (includes cached tokens).
        output (int): Output tokens.
        reasoning (int): Reasoning tokens.
        total (int): Total tokens (input + output).
        cached (int): Cached tokens read (subset of input).
        maximum_context (int): Largest input token count among the LLM
            calls covered by this record; 0 when unknown. For per-call
            records the call's ``input`` is the context, so this stays 0.
    """

    input: int = 0
    output: int = 0
    reasoning: int = 0
    total: int = 0
    cached: int = 0
    maximum_context: int = 0


class ModelResponse(BaseModel):
    """Provider-independent model completion response.

    Instance Attributes:
        text_parts (tuple[str, ...]): Text content fragments in order.
        tool_calls (tuple[NormalizedToolCall, ...]): Tool calls from the response.
        usage (Usage): Token usage statistics.
        provider_request_id (str | None): Provider request ID for audit.
        finish_reason (str | None): Provider finish reason.
        raw_metadata (dict[str, Any]): Additional provider-specific metadata.
    """

    text_parts: tuple[str, ...] = ()
    tool_calls: tuple[NormalizedToolCall, ...] = ()
    usage: Usage
    provider_request_id: str | None = None
    finish_reason: str | None = None
    raw_metadata: dict[str, Any] = {}
