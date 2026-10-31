#!/usr/bin/env python3
"""Shared helpers for the model-layer unit tests.

Important Classes:
    FakeCompletionsClient: OpenAI client double that records the kwargs
        of the create call.

Important Functions:
    edit_block(search, replace) -> str: Build one SEARCH/REPLACE edit
        block string.
    make_stream_chunk() -> Any: Build one streaming chunk sufficient
        for complete() aggregation.

Important Variables:
    READ_TOOL_DEFINITION (dict[str, Any]): Read-tool definition whose
        parameters schema is used across the fix-json routine tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import httpx2
from openai import (
    APIStatusError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

# Read-tool definition whose parameters schema is used across
# fix-json routine tests.
READ_TOOL_DEFINITION: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read the contents of a file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read.",
                },
            },
            "required": ["path"],
        },
    },
}


def make_api_status_error(status_code: int) -> APIStatusError:
    """Build the SDK status error the provider returns for a status code."""
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx2.Response(status_code, request=request)
    if status_code == 429:
        return RateLimitError("rate limited", response=response, body=None)
    if status_code >= 500:
        return InternalServerError("server error", response=response, body=None)
    return BadRequestError("bad request", response=response, body=None)


def edit_block(search: str, replace: str) -> str:
    """Build one SEARCH/REPLACE edit block string."""
    return (
        "/***SEARCH***/\n"
        f"{search}"
        "/***END_SEARCH***/\n"
        "/***REPLACE***/\n"
        f"{replace}"
        "/***END_REPLACE***/"
    )


def make_stream_chunk(*, with_usage: bool = True) -> Any:
    """Build one streaming chunk sufficient for complete() aggregation."""
    return SimpleNamespace(
        id="resp-1",
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        if with_usage
        else None,
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(content="ok", tool_calls=None),
                finish_reason="stop",
            )
        ],
    )


class FakeCompletionsClient:
    """OpenAI client double that records create() kwargs.

    Serves a single chunk so ``complete()`` can aggregate a full
    response without any network access.
    """

    def __init__(self, *, with_usage: bool = True) -> None:
        self.captured_kwargs: dict[str, Any] = {}
        self._chunk = make_stream_chunk(with_usage=with_usage)
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.captured_kwargs.update(kwargs)
        return iter((self._chunk,))


class FlakyCompletionsClient:
    """OpenAI client double that fails a scripted number of times.

    Each entry of ``errors`` is raised by one ``create`` call, in order;
    every later call serves one chunk. ``attempts`` counts the calls.
    """

    def __init__(self, *errors: Exception) -> None:
        self._errors = list(errors)
        self.attempts = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_kwargs: Any) -> Any:
        self.attempts += 1
        if self._errors:
            raise self._errors.pop(0)
        return iter((make_stream_chunk(),))
