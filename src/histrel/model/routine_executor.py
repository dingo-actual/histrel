#!/usr/bin/env python3
"""
Stateless single-shot LLM routine execution.

A routine sends exactly one two-message request (system + user) through
the model adapter with caller-provided sampling settings, parses the
response text with a caller-provided parser, and records the response
usage through a caller-provided hook.

    Important Classes:
        RoutineExecutor: Runs one routine call and keeps its usage.

    Important Functions:
        parse_bool(response_text) -> bool: Parse a constrained TRUE/FALSE
            response.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from histrel.model.adapter_openai_compat import OpenAICompatAdapter
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
    ProviderMessage,
    Usage,
)

_T = TypeVar("_T")


def parse_bool(response_text: str) -> bool:
    """Parse a constrained TRUE/FALSE routine response.

    Args:
        response_text (str): The raw response text.

    Returns:
        (bool) True for TRUE, False for FALSE.

    Raises:
        ValueError: When the text is neither TRUE nor FALSE.

    Examples:
        >>> parse_bool(" TRUE. ")
        True
    """
    token = response_text.strip().upper().rstrip(".")
    if token == "TRUE":
        return True
    if token == "FALSE":
        return False
    raise ValueError(f"response is neither TRUE or FALSE: {response_text!r}")


class RoutineExecutor:
    """Run one routine call, exposing its usage and request ID.

    Instance Attributes:
        last_usage (Usage): Usage of the most recent call.
        last_provider_request_id (str | None): Request ID of that call.
    """

    def __init__(
        self,
        adapter: OpenAICompatAdapter,
        usage_recorder: Callable[[ModelResponse], None],
    ) -> None:
        """Initialize the executor.

        Args:
            adapter (OpenAICompatAdapter): Adapter used for the call.
            usage_recorder (Callable[[ModelResponse], None]): Hook that
                receives the full response for ledger recording.
        """
        self._adapter = adapter
        self._usage_recorder = usage_recorder
        self.last_usage = Usage()
        self.last_provider_request_id: str | None = None

    def run(
        self,
        *,
        system_prompt: str,
        user_message: str,
        parse_response: Callable[[str], _T],
        temperature: float,
        reasoning_effort: str | None,
        max_tokens_output: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> _T:
        """Send one two-message request and parse its response.

        Args:
            system_prompt (str): The routine's system instructions.
            user_message (str): The rendered routine input.
            parse_response (Callable[[str], _T]): Text-to-result parser.
            temperature (float): Sampling temperature.
            reasoning_effort (str | None): Reasoning effort, if supported.
            max_tokens_output (int | None): Output token cap, if any.
            response_format (dict[str, Any] | None): Provider response
                format constraint, e.g. a ``json_schema`` payload.

        Returns:
            (_T) The parsed routine result.

        Raises:
            ValueError: When the response cannot be parsed.
        """
        request = ModelRequest(
            messages=self._request_messages(system_prompt, user_message),
            tools=(),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_tokens_output=max_tokens_output,
            response_format=response_format,
        )
        response = self._adapter.complete(request)
        self.last_usage = response.usage
        self.last_provider_request_id = response.provider_request_id
        self._usage_recorder(response)
        return parse_response("".join(response.text_parts))

    @staticmethod
    def _request_messages(
        system_prompt: str, user_message: str
    ) -> tuple[ProviderMessage, ...]:
        """Build the routine's two provider messages.

        Args:
            system_prompt (str): The routine's system instructions.
            user_message (str): The rendered routine input.

        Returns:
            (tuple[ProviderMessage, ...]) The system and user messages.
        """
        return (
            ProviderMessage(role="system", content=system_prompt),
            ProviderMessage(role="user", content=user_message),
        )
