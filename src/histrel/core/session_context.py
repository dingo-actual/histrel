#!/usr/bin/env python3
"""
Session context for the benchmark agent.

Carries immutable session-level metadata through the agent runtime.

    Important Classes:
        ModelInfo: Model provider and name.
        SessionContext: Immutable session-level context.
"""

from __future__ import annotations

from pydantic import BaseModel

from histrel.core.config import Provider


class ModelInfo(BaseModel):
    """Model provider and name.

    Instance Attributes:
        provider (str): Provider identifier (openai or fireworks).
        name (str): Model name.
    """

    provider: Provider
    name: str


class SessionContext(BaseModel):
    """Immutable session-level context.

    Instance Attributes:
        session_id (str): Session identifier.
        task_id (str): SWE-bench instance identifier.
        model (ModelInfo): Model provider and name.
    """

    session_id: str
    task_id: str
    model: ModelInfo
