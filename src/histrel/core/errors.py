#!/usr/bin/env python3
"""
Error types for the benchmark agent.

Defines the exception hierarchy used across the agent runtime. Errors are
categorized to support reliability and recovery requirements.

    Important Classes:
        AgentError: Base exception for all benchmark agent errors.
        UnsupportedProviderError: Raised when a provider is not supported.
        ToolBackendError: Raised when a required tool backend version is not met.
        ConfigValidationError: Raised when configuration validation fails.
        JournalError: Raised when a journal write or read fails.
"""

from __future__ import annotations


class AgentError(Exception):
    """Base exception for all benchmark agent errors."""


class UnsupportedProviderError(AgentError):
    """Raised when the configured model provider is not supported."""


class ToolBackendError(AgentError):
    """Raised when a required tool backend version is not met.

    ripgrep >= 14 and fd == 10.3.0 are checked at startup.
    """


class ConfigValidationError(AgentError):
    """Raised when configuration validation fails."""


class JournalError(AgentError):
    """Raised when a journal write or read fails."""
