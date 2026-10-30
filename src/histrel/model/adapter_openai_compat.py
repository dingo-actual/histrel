#!/usr/bin/env python3
"""
OpenAI-compatible model adapter for the benchmark agent.

Supports both the Fireworks AI and native OpenAI providers through the
openai Python SDK. The provider guard is hard-coded: the openai provider
is only valid for baseline sessions. The fireworks provider is required
for histrel sessions.

Streaming aggregation:
    - Concatenates delta.content fragments in order.
    - Accumulates tool-call fragments by index, concatenating arguments.
    - Captures finish_reason from the final chunk.
    - Fireworks includes usage in the final streaming chunk by default.
    - OpenAI requires stream_options={"include_usage": True} for usage.

Usage normalization:
    - prompt_tokens -> input (includes cached tokens)
    - completion_tokens -> output
    - prompt_tokens_details.cached_tokens -> cached (subset of input)
    - completion_tokens_details.reasoning_tokens -> reasoning

    Important Classes:
        OpenAICompatAdapter: Streaming model adapter for OpenAI-compatible providers.

    Important Functions:
        build_client(config) -> OpenAI: Build an OpenAI client for the configured provider.
        map_usage(usage) -> Usage: Normalize provider usage to internal Usage.
        normalize_tool_calls(raw) -> tuple[NormalizedToolCall, ...]: Normalize streaming tool call fragments.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Any, NamedTuple

from loguru import logger
from openai import OpenAI

from histrel.core.config import AgentConfig
from histrel.core.errors import UnsupportedProviderError
from histrel.model.retry import (
    REQUEST_TIMEOUT_SECONDS,
    create_stream_with_retry,
)
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
    NormalizedToolCall,
    Usage,
)


def build_client(config: AgentConfig) -> OpenAI:
    """Build an OpenAI SDK client for the configured provider.

    Args:
        config (AgentConfig): Agent configuration.

    Returns:
        (OpenAI) Configured OpenAI SDK client.

    Raises:
        UnsupportedProviderError: If the provider is unknown.
    """
    if config.model_provider == "fireworks":
        return OpenAI(
            base_url="https://api.fireworks.ai/inference/v1",
            api_key=os.environ["FIREWORKS_API_KEY"],
        )
    if config.model_provider == "openai":
        return OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    raise UnsupportedProviderError(f"unknown provider: {config.model_provider}")


def map_usage(usage: Usage) -> Usage:
    """Normalize and validate provider usage data.

    Validates that token counts are non-negative and that cached tokens
    do not exceed input tokens.

    Args:
        usage (Usage): Raw usage data from the provider.

    Returns:
        (Usage) Normalized usage with total computed as input + output.

    Raises:
        ValueError: If counts are negative or cached > input.
    """
    if usage.input < 0 or usage.output < 0 or usage.cached < 0 or usage.reasoning < 0:
        raise ValueError(
            f"negative token counts are not valid: "
            f"input={usage.input}, output={usage.output}, "
            f"cached={usage.cached}, reasoning={usage.reasoning}"
        )
    if usage.cached > usage.input:
        raise ValueError(
            f"cached tokens ({usage.cached}) cannot exceed input tokens ({usage.input})"
        )
    return Usage(
        input=usage.input,
        output=usage.output,
        reasoning=usage.reasoning,
        total=usage.input + usage.output,
        cached=usage.cached,
    )


def normalize_tool_calls(
    raw: dict[int, dict[str, Any]],
) -> tuple[NormalizedToolCall, ...]:
    """Normalize streaming tool call fragments into a tuple of calls.

    Fragments are accumulated by index during streaming. This function
    converts the accumulated dict into an ordered tuple.

    Args:
        raw (dict[int, dict[str, Any]]): Accumulated tool call fragments keyed by index.

    Returns:
        (tuple[NormalizedToolCall, ...]) Normalized tool calls in index order.
    """
    return tuple(
        NormalizedToolCall(
            id=raw[idx]["id"],
            name=raw[idx]["name"],
            arguments=raw[idx]["arguments"],
        )
        for idx in sorted(raw)
    )


class _StreamAccumulation(NamedTuple):
    """Accumulated state from a streaming completion."""

    content: list[str]
    tool_calls: dict[int, dict[str, Any]]
    finish_reason: str | None
    usage: Usage | None
    provider_request_id: str | None


def _accumulate_tool_call_fragment(
    tool_calls: dict[int, dict[str, Any]],
    fragment: Any,
) -> None:
    """Merge one streaming tool-call fragment into the accumulator.

    Fragments arrive keyed by index; the first fragment for an index
    creates the slot, and later fragments extend its argument string.

    Args:
        tool_calls (dict[int, dict[str, Any]]): Accumulator keyed by index.
        fragment: One tool-call delta fragment from a streaming chunk.
    """
    slot = tool_calls.setdefault(
        fragment.index, {"id": "", "name": "", "arguments": ""}
    )
    if fragment.id:
        slot["id"] = fragment.id
    if fragment.function and fragment.function.name:
        slot["name"] = fragment.function.name
    if fragment.function and fragment.function.arguments:
        slot["arguments"] += fragment.function.arguments


def _accumulate_choice(
    choice: Any,
    content: list[str],
    tool_calls: dict[int, dict[str, Any]],
) -> str | None:
    """Accumulate one chunk choice's delta and return its finish reason.

    Args:
        choice: A streaming chunk choice.
        content (list[str]): Text fragment accumulator.
        tool_calls (dict[int, dict[str, Any]]): Tool-call accumulator.

    Returns:
        (str | None) The choice's finish reason when present.
    """
    if choice.delta and choice.delta.content:
        content.append(choice.delta.content)
    if choice.delta and choice.delta.tool_calls:
        for fragment in choice.delta.tool_calls:
            _accumulate_tool_call_fragment(tool_calls, fragment)
    if choice.finish_reason:
        return str(choice.finish_reason)
    return None


def _accumulate_stream(stream: Iterable[Any]) -> _StreamAccumulation:
    """Aggregate streaming chunks into accumulated response state.

    Args:
        stream (Iterable[Any]): The streaming chunk iterator.

    Returns:
        (_StreamAccumulation) Accumulated content, tool calls, finish
        reason, usage, and provider request id.
    """
    content: list[str] = []
    tool_calls: dict[int, dict[str, Any]] = {}
    finish_reason: str | None = None
    usage: Usage | None = None
    provider_request_id: str | None = None

    for chunk in stream:
        if provider_request_id is None and chunk.id:
            provider_request_id = chunk.id
        if chunk.usage:
            usage = _extract_usage_from_chunk(chunk.usage)
        if not chunk.choices:
            continue
        reason = _accumulate_choice(chunk.choices[0], content, tool_calls)
        if reason:
            finish_reason = reason

    return _StreamAccumulation(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
        provider_request_id=provider_request_id,
    )


class OpenAICompatAdapter:
    """Streaming model adapter for OpenAI-compatible providers.

    Enforces the provider guard in the constructor. Uses streaming for
    all completions and aggregates chunks into a single ModelResponse.

    Instance Attributes:
        _client (OpenAI): OpenAI SDK client.
        _model_name (str): Model name.
        _provider (str): Provider identifier.
        _log: Bound logger.
    """

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        provider: str,
        request_timeout: float = REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the adapter.

        Args:
            client (OpenAI): OpenAI SDK client.
            model_name (str): Model name to pass to the API.
            provider (str): Provider identifier (openai or fireworks).
            request_timeout (float): Per-request timeout bounding a stalled
                stream. Defaults to the production bound; tests pass a
                shorter value so a stalled provider fails fast.
        """
        self._client = client
        self._model_name = model_name
        self._provider = provider
        self._request_timeout = request_timeout
        self._log = logger.bind(component="bench_agent.model")

    @property
    def provider(self) -> str:
        """Return the provider identifier."""
        return self._provider

    def _build_create_kwargs(self, request: ModelRequest) -> dict[str, Any]:
        """Build provider-specific chat completion keyword arguments.

        Args:
            request (ModelRequest): The completion request.

        Returns:
            (dict[str, Any]) Keyword arguments for the SDK create call.
        """
        create_kwargs: dict[str, Any] = {
            "model": self._model_name,
            "messages": [msg.model_dump(exclude_none=True) for msg in request.messages],
            "temperature": request.temperature,
            "top_p": request.top_p,
            "stream": True,
        }

        if request.tools:
            create_kwargs["tools"] = [tool.model_dump() for tool in request.tools]

        if request.response_format is not None:
            create_kwargs["response_format"] = request.response_format

        if request.reasoning_effort:
            create_kwargs["reasoning_effort"] = request.reasoning_effort

        # Output budget: fireworks uses max_tokens, openai uses max_completion_tokens.
        if request.max_tokens_output:
            if self._provider == "openai":
                create_kwargs["max_completion_tokens"] = request.max_tokens_output
            else:
                create_kwargs["max_tokens"] = request.max_tokens_output

        # Prompt cache key for providers that support prompt caching
        # (fireworks). The runtime decides when to set it; the adapter
        # only forwards it verbatim.
        if request.prompt_cache_key is not None:
            create_kwargs["prompt_cache_key"] = request.prompt_cache_key

        # Extra HTTP headers set by the runtime (e.g. session affinity);
        # the adapter only forwards them verbatim.
        if request.extra_headers is not None:
            create_kwargs["extra_headers"] = request.extra_headers

        # OpenAI requires stream_options for usage in streaming mode.
        if self._provider == "openai":
            create_kwargs["stream_options"] = {"include_usage": True}

        return create_kwargs

    def _finalize_response(self, accumulated: _StreamAccumulation) -> ModelResponse:
        """Finalize accumulated stream state into a ModelResponse.

        Substitutes an interrupted finish reason and empty usage when the
        stream ended without them, logging a warning in each case.

        Args:
            accumulated (_StreamAccumulation): Accumulated streaming state.

        Returns:
            (ModelResponse) Aggregated response with text, tool calls, and usage.
        """
        finish_reason = accumulated.finish_reason
        if finish_reason is None:
            finish_reason = "interrupted"
            self._log.warning("stream_ended_without_finish_reason")

        usage = accumulated.usage
        if usage is None:
            usage = Usage()
            self._log.warning("stream_ended_without_usage")

        return ModelResponse(
            text_parts=tuple(accumulated.content),
            tool_calls=normalize_tool_calls(accumulated.tool_calls),
            usage=map_usage(usage),
            provider_request_id=accumulated.provider_request_id,
            finish_reason=finish_reason,
            raw_metadata={},
        )

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Execute a streaming model completion request.

        Aggregates streaming chunks into a single ModelResponse. The agent
        loop never sees individual chunks.

        Args:
            request (ModelRequest): The completion request.

        Returns:
            (ModelResponse) Aggregated response with text, tool calls, and usage.

        Raises:
            Exception: The last provider error once the retries are
            exhausted, or any non-transient provider error immediately.
        """
        stream = create_stream_with_retry(
            self._client,
            self._build_create_kwargs(request),
            self._log,
            timeout=self._request_timeout,
        )
        return self._finalize_response(_accumulate_stream(stream))


def _extract_usage_from_chunk(chunk_usage: Any) -> Usage:
    """Extract normalized Usage from a provider chunk usage object.

    Handles both Fireworks and OpenAI usage formats, which share the same
    field names via the openai SDK.

    Args:
        chunk_usage: The usage object from a streaming chunk.

    Returns:
        (Usage) Normalized usage (not yet validated).
    """
    prompt_tokens = getattr(chunk_usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(chunk_usage, "completion_tokens", 0) or 0

    cached = 0
    prompt_details = getattr(chunk_usage, "prompt_tokens_details", None)
    if prompt_details:
        cached = getattr(prompt_details, "cached_tokens", 0) or 0

    reasoning = 0
    completion_details = getattr(chunk_usage, "completion_tokens_details", None)
    if completion_details:
        reasoning = getattr(completion_details, "reasoning_tokens", 0) or 0

    return Usage(
        input=prompt_tokens,
        output=completion_tokens,
        reasoning=reasoning,
        total=prompt_tokens + completion_tokens,
        cached=cached,
    )
