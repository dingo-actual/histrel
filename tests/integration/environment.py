#!/usr/bin/env python3
"""Environment helpers and the live-provider recording adapter.

Important Classes:
    RecordingAdapter: Adapter that records requests and delegates to the
        live provider.

Important Functions:
    env(name) -> str | None: Read a non-blank environment variable.
    server_listening(url, timeout) -> bool: Probe a TCP endpoint.

Important Variables:
    FIREWORKS_MODEL (str): Model id for live LLM tests.
    FIREWORKS_BASE_URL (str): Base URL for live LLM tests.
    LIVE_REQUEST_TIMEOUT_SECONDS (float): Per-request bound for live calls.
"""

from __future__ import annotations

import contextlib
import os
import socket
from urllib.parse import urlparse

from openai import OpenAI

from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
)
from histrel.model.types import ModelRequest, ModelResponse

# Live LLM settings for Fireworks AI.
FIREWORKS_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash-1017"
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"

# Per-request bound for live calls. The production bound is 120s; a stalled
# proxy would otherwise block the test suite for many minutes.
LIVE_REQUEST_TIMEOUT_SECONDS = 60.0


def env(name: str) -> str | None:
    """Read a non-blank environment variable."""
    value = os.getenv(name)
    return value if value and value.strip() else None


def server_listening(url: str, *, timeout: float = 2.0) -> bool:
    """Report whether a TCP connection to ``url`` succeeds.

    Egress may go through an HTTP proxy, so the proxy endpoint is probed
    as a fallback when the direct connection fails.
    """
    parsed = urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    targets = [(host, port)]
    proxy = urlparse(os.getenv("https_proxy") or os.getenv("HTTPS_PROXY") or "")
    if proxy.hostname:
        targets.append((proxy.hostname, proxy.port or 3128))
    for target in targets:
        with (
            contextlib.suppress(Exception),
            socket.create_connection(target, timeout=timeout),
        ):
            return True
    return False


class RecordingAdapter(OpenAICompatAdapter):
    """Adapter that records requests and delegates to the live provider."""

    def __init__(
        self,
        client: OpenAI,
        model_name: str,
        provider: str,
        request_timeout: float = LIVE_REQUEST_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the recording adapter."""
        super().__init__(
            client=client,
            model_name=model_name,
            provider=provider,
            request_timeout=request_timeout,
        )
        self.recorded: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Record the request, then delegate to the live provider."""
        self.recorded.append(request)
        return super().complete(request)
