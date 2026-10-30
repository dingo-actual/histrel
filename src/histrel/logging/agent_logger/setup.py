#!/usr/bin/env python3
"""
Setup helper for the agent log file sink.

The agent writes its own event log to a single JSON-lines file in /tmp,
which the harness exports into the session artifacts directory after the
run. The file never pollutes the repository working tree the agent edits.

    Important Functions:
        setup_file_sink(log_path) -> int | None: Add the JSON-lines file sink.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from histrel.logging.agent_logger.constants import AGENT_LOG_FILE_PATH


def setup_file_sink(log_path: str | Path = AGENT_LOG_FILE_PATH) -> int | None:
    """Add a JSON-lines file sink for every log event.

    Args:
        log_path (str | Path): Log file path.

    Returns:
        (int | None) The sink ID, or None if the sink could not be added.
    """
    try:
        return logger.add(
            str(log_path),
            level="DEBUG",
            serialize=True,
            delay=True,
            enqueue=True,
            catch=True,
        )
    except Exception:
        # Logging must never break the agent; report to stderr directly.
        sys.stderr.write(f"Failed to add agent log file sink at {log_path}\n")
        sys.stderr.flush()
        return None
