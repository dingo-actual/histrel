#!/usr/bin/env python3
"""
Retry policy for transient provider failures.

A single rate limit, server error, timeout, or connection failure would
otherwise lose a whole benchmark run of 40-301 model calls, so the
completion stream is created through ``create_stream_with_retry`` with
exponential backoff. Each attempt is bounded by
``REQUEST_TIMEOUT_SECONDS`` so a hung connection cannot stall a run.

    Important Functions:
        create_stream_with_retry(client, create_kwargs, log): Stream.
        is_transient(error) -> bool: Whether an error is worth retrying.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterable
from typing import Any

from openai import APIConnectionError, APITimeoutError, OpenAI

# Attempts include the first call, so 4 means three retries.
_MAX_ATTEMPTS = 4
_BASE_RETRY_DELAY_SECONDS = 1.0
# Equal jitter: the lower half of each delay is fixed and the upper half is
# random, so concurrent runs do not retry in lockstep. See _retry_delay.
_JITTER_LOW_FACTOR = 0.5

# Per-request timeout bounding a stalled stream. The SDK applies it per
# read, so a long streaming answer is unaffected; only a hung connection
# hits it. Worst case per model call is 4 attempts x 120s plus 7s of backoff.
REQUEST_TIMEOUT_SECONDS = 120.0


def create_stream_with_retry(
    client: OpenAI,
    create_kwargs: dict[str, Any],
    log: Any,
    *,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> Iterable[Any]:
    """Create the completion stream, retrying transient failures.

    Args:
        client (OpenAI): The SDK client.
        create_kwargs (dict[str, Any]): Keyword arguments for the call.
        log (Any): Bound logger for retry warnings.
        timeout (float): Per-request timeout bounding a stalled stream.

    Returns:
        (Iterable[Any]) The provider's streaming chunks.

    Raises:
        Exception: The last provider error once the retries are
        exhausted, or any non-transient provider error immediately.
    """
    attempt = 0
    while True:
        attempt += 1
        try:
            stream: Iterable[Any] = client.chat.completions.create(
                **create_kwargs, timeout=timeout
            )
            return stream
        except Exception as error:
            if attempt >= _MAX_ATTEMPTS or not is_transient(error):
                raise
            delay = _retry_delay(attempt)
            _log_retry(log, attempt, delay, error)
            time.sleep(delay)


def is_transient(error: Exception) -> bool:
    """Report whether a provider error is worth retrying.

    Args:
        error (Exception): The error raised by the SDK call.

    Returns:
        (bool) True for rate limits, server errors, timeouts, and
        connection failures; False for every other error.
    """
    if isinstance(error, (APITimeoutError, APIConnectionError)):
        return True
    status_code = getattr(error, "status_code", None)
    return isinstance(status_code, int) and (status_code == 429 or status_code >= 500)


def _log_retry(log: Any, attempt: int, delay: float, error: Exception) -> None:
    """Log a transient provider failure and the delay before the retry.

    Args:
        log (Any): Bound logger for retry warnings.
        attempt (int): The 1-based attempt number that just failed.
        delay (float): Seconds to wait before the next attempt.
        error (Exception): The transient provider error.
    """
    log.warning(
        "model_call_retrying",
        attempt=attempt,
        max_attempts=_MAX_ATTEMPTS,
        delay_seconds=delay,
        error_type=type(error).__name__,
    )


def _retry_delay(attempt: int) -> float:
    """Return the jittered exponential backoff delay before the next attempt.

    Args:
        attempt (int): The 1-based attempt number that just failed.

    Returns:
        (float) Seconds to wait before retrying, somewhere between half
        the exponential window and the full window.
    """
    base = _BASE_RETRY_DELAY_SECONDS * (2.0 ** (attempt - 1))
    return base * random.uniform(_JITTER_LOW_FACTOR, 1.0)
