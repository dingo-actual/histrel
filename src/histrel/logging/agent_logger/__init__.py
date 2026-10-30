#!/usr/bin/env python3
"""
Agent log file package.

Writes the agent's own event log to a JSON-lines file in /tmp, which the
harness exports into the session artifacts directory after the run.

    Important Variables:
        AGENT_LOG_FILE_PATH (str): Absolute path of the agent's own log file.
        AGENT_LOG_EXPORT_NAME (str): File name used when exporting the log.

    Important Functions:
        setup_file_sink(log_path) -> int | None: Add the JSON-lines file sink.
"""

from __future__ import annotations

from histrel.logging.agent_logger.constants import (
    AGENT_LOG_EXPORT_NAME,
    AGENT_LOG_FILE_PATH,
)
from histrel.logging.agent_logger.setup import setup_file_sink

__all__ = [
    "AGENT_LOG_EXPORT_NAME",
    "AGENT_LOG_FILE_PATH",
    "setup_file_sink",
]
