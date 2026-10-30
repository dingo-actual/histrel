#!/usr/bin/env python3
"""
Fallback logging utility for benchmark execution.

Provides robust logging with multi-level fallback when loguru fails.
"""

from __future__ import annotations

import contextlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from loguru import logger


def is_running_in_docker() -> bool:
    """Check whether this process runs inside a Docker container.

    Returns:
        (bool) True if running inside Docker, False otherwise.
    """
    if Path("/.dockerenv").exists():
        return True
    try:
        cgroup_content = Path("/proc/1/cgroup").read_text()
        return "docker" in cgroup_content
    except Exception:
        return False


def safe_log(
    message: str,
    level: str = "info",
    bind_values: Mapping[str, Any] | None = None,
) -> None:
    """Log with multi-level fallback safety.

    Args:
        bind_values: Dictionary of values to pass to logger.bind()
        level: Log level ("info", "warning", "error", "debug")
        message: Log message text

    Behavior:
        1. In container: always emit JSON to stderr for parseability
        2. Outside container: attempt loguru with fallback to stderr/stdout

    Examples:
        >>> safe_log({"instance_id": "test-1"}, "info", "Starting execution")
        >>> safe_log({"error": "file not found"}, "error", "Failed to load file")
    """
    in_container = is_running_in_docker()
    full_bind_values = dict(bind_values) if bind_values else {}
    full_bind_values["_from_container"] = in_container
    full_bind_values["origin"] = (
        "bench_harness_predict_container" if in_container else "bench_harness_predict"
    )

    if in_container:
        log_data = {
            "level_container": level.upper(),
            "message_container": message,
            "_from_container": True,
            "origin": "bench_harness_predict_container",
        }
        for k, v in full_bind_values.items():
            if k != "_from_container":
                log_data[k] = v
        sys.stderr.write(json.dumps(log_data) + "\n")
        sys.stderr.flush()
        return

    try:
        if full_bind_values:
            logger.bind(**full_bind_values).__getattribute__(level)(message)
        else:
            logger.__getattribute__(level)(message)
    except Exception as _exc:
        prefix = f"[{level.upper()}] " if level != "info" else ""
        sys.stderr.write(f"{prefix}{message}\n")
        sys.stderr.flush()
        return

    with contextlib.suppress(Exception):
        prefix = f"[{level.upper()}] " if level != "info" else ""
        sys.stdout.write(f"{prefix}{message}\n")
        sys.stdout.flush()
        return
