#!/usr/bin/env python3
"""
System prompt templates for benchmark agents.

    Important Functions:
        get_bench_agent_sys_prompts(repo_dir, mode) -> dict[str, str]:
            Fetches and renders the bench-agent system prompts for the mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from histrel.prompts import get_histrel_sys_prompts


def get_bench_agent_sys_prompts(
    repo_dir: Path,
    mode: Literal["stateless", "react"] = "stateless",
) -> dict[str, str]:
    """Fetch and render the bench-agent system prompts.

    Args:
        repo_dir (Path): Repository directory, rendered into the prompt.
        mode (Literal["stateless", "react"]): Agent loop mode. "react"
            renders the react template as both the initialize and step
            prompts. Defaults to "stateless".

    Returns:
        (dict[str, str]) Rendered prompts with "initialize" and step"
            keys.

    Raises:
        ValueError: If mode is invalid.
    """
    return get_histrel_sys_prompts(repo_dir, mode)
