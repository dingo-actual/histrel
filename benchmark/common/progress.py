#!/usr/bin/env python3
"""
Shared progress bar and logging utilities.

Provides consistent Rich progress bar setup and Loguru logging configuration
for the predictor. Log configuration itself lives in `histrel.logging.console`,
which Histrel owns.

    Important Functions:
        create_progress(verbose: bool) -> Progress
        setup_benchmark_logging(verbose: bool) -> None
        setup_benchmark_file_sink() -> None
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from loguru import logger
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from histrel.logging.console import console, setup_logging


def create_progress(verbose: bool = False) -> Progress:
    """Create a configured Progress instance for benchmark operations.

    Args:
        verbose (bool): If True, don't attach to shared console. Defaults to
            False.

    Returns:
        (Progress) Configured Rich Progress instance with standard columns.
    """
    progress_console = console if not verbose else None
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=progress_console,
        redirect_stdout=True,
        redirect_stderr=True,
    )


def setup_benchmark_logging(verbose: bool = False, level: str = "INFO") -> None:
    """Configure Loguru logging for benchmark operations.

    Sets up Rich console integration and appropriate log level.

    Args:
        verbose (bool): If True, log container output at DEBUG to stderr instead
            of using the Rich console. Defaults to False.
        level (str): Logging level (DEBUG, INFO, WARNING, ERROR). Defaults to "INFO".
    """
    import sys

    def _container_log_filter(record: Any) -> bool:
        return bool(record.get("extra", {}).get("_from_container"))

    if verbose:
        # Verbose mode: skip Rich console (causes Windows unicode errors)
        logger.remove()  # Remove default handler

        # Add stderr sink with filter to exclude host-side logs
        logger.add(
            sys.stderr,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            colorize=True,
            backtrace=True,
            diagnose=False,
            enqueue=True,
            filter=_container_log_filter,
        )
    else:
        setup_logging(level=level)


def ensure_task_complete(progress: Progress, task_id) -> None:
    """Ensure a progress task shows 100% completion.

    Updates both the completed count and total to guarantee the task shows
    as fully complete, avoiding partial progress display.

    Args:
        progress (Progress): Progress instance managing the task.
        task_id: Task ID returned from progress.add_task().
    """
    task = progress._tasks[task_id]
    if task.total is not None:
        progress.update(task_id, completed=task.total)


def setup_benchmark_file_sink() -> None:
    """Configure a file sink for benchmark logging at TRACE level.

    Writes structured JSON logs to logs/benchmark/ directory. This sink is
    always active regardless of verbose mode or CLI flags, providing a complete
    audit trail of all benchmark operations.

    Hard-coded configuration:
        - Path: logs/benchmark/benchmark_{time}.jsonl
        - Level: TRACE (captures all log levels)
        - Format: JSON serialized
        - Rotation: 1 week
        - Compression: zip
        - Delay: True (create file on first log)
        - Enqueue: True (thread-safe)
    """
    log_dir = Path("logs/benchmark")
    with contextlib.suppress(Exception):
        log_dir.mkdir(parents=True, exist_ok=True)

    log_path = str(log_dir / "benchmark_{time}.jsonl")

    with contextlib.suppress(Exception):
        logger.add(
            log_path,
            level="TRACE",
            serialize=True,
            rotation="1 week",
            compression="zip",
            delay=True,
            enqueue=True,
            catch=True,
        )
