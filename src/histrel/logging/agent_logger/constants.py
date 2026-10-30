#!/usr/bin/env python3
"""
Shared constants for the agent log file.

    Important Variables:
        AGENT_LOG_FILE_PATH (str): Absolute path of the agent's own log file.
        AGENT_LOG_EXPORT_NAME (str): File name used when exporting the log.
"""

from __future__ import annotations

from pathlib import Path

# Absolute path of the agent's own log file. The file lives in /tmp so
# it never pollutes the repository working tree the agent edits; the
# harness copies it into the session artifacts directory after the run.
AGENT_LOG_FILE_PATH = "/tmp/bench_agent.log"

# Name used for the log file when the harness exports it into the
# session artifacts directory.
AGENT_LOG_EXPORT_NAME = Path(AGENT_LOG_FILE_PATH).name
