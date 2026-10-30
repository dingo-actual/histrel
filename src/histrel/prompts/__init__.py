#!/usr/bin/env python3
"""Rendered system prompts for the Histrel agent loop.

Important Functions:
    get_histrel_sys_prompts(repo_dir, mode) -> dict[str, str]:
        Fetches and renders the system prompts for the loop mode.
"""

from __future__ import annotations

from pathlib import Path

from histrel.core.config import Mode
from histrel.prompts.templates import (
    _SYS_TEMPLATE_BASELINE_INIT,
    _SYS_TEMPLATE_BASELINE_STEP,
    _SYS_TEMPLATE_REACT,
)

__all__ = ["get_histrel_sys_prompts"]


def get_histrel_sys_prompts(
    repo_dir: Path,
    mode: Mode = "stateless",
) -> dict[str, str]:
    """Fetch and render the Histrel system prompts.

    Args:
        repo_dir (Path): Repository directory, rendered into the prompt.
        mode (Mode): Agent loop mode. "react" renders the react template as
            both the initialize and step prompts. Defaults to "stateless".

    Returns:
        (dict[str, str]) Rendered prompts with "initialize" and "step" keys.

    Raises:
        ValueError: If mode is invalid.
    """
    repo = str(repo_dir)
    cwd = Path.cwd().as_posix()

    if mode == "react":
        rendered = _SYS_TEMPLATE_REACT.format(repo=repo, cwd=cwd)
        return {"initialize": rendered, "step": rendered}
    if mode != "stateless":
        raise ValueError("mode must be 'stateless' or 'react'")

    return {
        "initialize": _SYS_TEMPLATE_BASELINE_INIT.format(repo=repo, cwd=cwd),
        "step": _SYS_TEMPLATE_BASELINE_STEP.format(repo=repo, cwd=cwd),
    }
