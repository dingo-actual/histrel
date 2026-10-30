#!/usr/bin/env python3
"""Histrel logging package.

Owns console and file log configuration for the agent and for harnesses that
embed it.

    Important Functions:
        setup_logging(level, diagnose, intercept_stdlib, redirect_stdio) -> None:
            Configure Loguru to write through Rich.

    Important Variables:
        console (Console): Shared Rich console instance.
"""

from __future__ import annotations

from histrel.logging.console import console, setup_logging

__all__ = ["console", "setup_logging"]
