#!/usr/bin/env python3
"""
Shared metadata sub-models for the canonical message model.

    Important Classes:
        Timing: Timing information for content items.
        TokenUsage: Token usage statistics (cached is subset of input).
        ErrorInfo: Error information for failed messages.
        MessageMetadata: Additional metadata about messages.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Timing(BaseModel):
    """Timing information for content items.

    Instance Attributes:
        start (int | None): Unix timestamp when started.
        end (int | None): Unix timestamp when completed.
        duration_ms (int | None): Duration in milliseconds.
    """

    start: int | None = None
    end: int | None = None
    duration_ms: int | None = None


class TokenUsage(BaseModel):
    """Token usage statistics.

    The ``cached`` field is a subset of ``input``, not additive. Never
    calculate total as input + output + cached; that double-counts.

    Instance Attributes:
        input (int): Input tokens (includes cached tokens).
        output (int): Output tokens.
        reasoning (int): Reasoning tokens from extended thinking.
        total (int): Total tokens used (input + output).
        cached (int): Cached tokens read (subset of input).
    """

    input: int = 0
    output: int = 0
    reasoning: int = 0
    total: int = 0
    cached: int = 0


class ErrorInfo(BaseModel):
    """Error information for failed messages.

    Instance Attributes:
        type (str): Error type or category.
        message (str): Human-readable error message.
        details (dict[str, Any] | None): Additional error details.
    """

    type: str
    message: str
    details: dict[str, Any] | None = None


class MessageMetadata(BaseModel):
    """Additional metadata about messages.

    Includes run_id for model-call correlation, tokens for usage, and cost
    for billing.

    Instance Attributes:
        agent (str): Agent name used.
        model_id (str | None): Model identifier.
        provider_id (str | None): Provider identifier.
        parent_id (str | None): Parent message ID (causal link).
        cost (float): Generation cost in USD.
        tokens (TokenUsage): Token usage statistics.
        working_directory (str | None): Working directory context.
        has_reasoning (bool): Whether message contains reasoning.
        tool_count (int): Number of tool sessions.
        run_id (str): Run identifier for model-call correlation.
    """

    agent: str = ""
    model_id: str | None = None
    provider_id: str | None = None
    parent_id: str | None = None
    cost: float = 0.0
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    working_directory: str | None = None
    has_reasoning: bool = False
    tool_count: int = 0
    run_id: str = ""
