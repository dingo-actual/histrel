#!/usr/bin/env python3
"""Rich console and Loguru integration for Histrel.

Provides one shared Rich ``Console`` and configures Loguru to write through it,
so progress displays stay pinned to the bottom while log lines render above.

    Important Classes:
        InterceptHandler(logging.Handler): Redirect stdlib logging to Loguru.
        StdoutInterceptor: Redirect stdout writes to Loguru.
        StderrInterceptor: Redirect stderr writes to Loguru.

    Important Functions:
        setup_logging(level, diagnose, intercept_stdlib, redirect_stdio) -> None:
            Configure Loguru to write through Rich.
        rich_sink(message) -> None: Loguru sink that renders via Rich.

    Important Variables:
        console (Console): Shared Rich console instance.
        LEVEL_STYLES (dict[str, str]): Rich style per Loguru level name.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import Any

from loguru import logger
from rich.console import Console
from rich.traceback import install as install_rich_traceback

console = Console()
install_rich_traceback(console=console, show_locals=False)

LEVEL_STYLES: dict[str, str] = {
    "TRACE": "dim",
    "DEBUG": "cyan",
    "INFO": "green",
    "SUCCESS": "bold green",
    "WARNING": "yellow",
    "ERROR": "bold red",
    "CRITICAL": "bold white on red",
}

_logging_configured = False


class InterceptHandler(logging.Handler):
    """Redirect stdlib logging records to Loguru.

    Keeps third-party library logs in the same Rich-rendered stream as
    Histrel's own Loguru logs.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Forward a stdlib log record to Loguru.

        Args:
            record (logging.LogRecord): Record emitted by the stdlib logging.
        """
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Skip the stdlib logging frames so the reported location is the caller.
        frame: Any = sys._getframe(6)
        depth = 6
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        with contextlib.suppress(Exception):
            logger.opt(depth=depth, exception=record.exc_info).log(
                level, record.getMessage()
            )


class StdoutInterceptor:
    """Redirect stdout writes to Loguru at INFO level."""

    def write(self, message: str) -> None:
        """Log a stdout write.

        Args:
            message (str): Text written to stdout.
        """
        with contextlib.suppress(Exception):
            if message.strip():
                logger.opt(depth=1).info(message.rstrip())

    def flush(self) -> None:
        """No-op flush required by the file-like protocol."""


class StderrInterceptor:
    """Redirect stderr writes to Loguru at ERROR level."""

    def write(self, message: str) -> None:
        """Log a stderr write.

        Args:
            message (str): Text written to stderr.
        """
        with contextlib.suppress(Exception):
            if message.strip():
                logger.opt(depth=1).error(message.rstrip())

    def flush(self) -> None:
        """No-op flush required by the file-like protocol."""


def rich_sink(message: Any) -> None:
    """Render a Loguru record through the shared Rich console.

    Args:
        message (Any): Loguru message object carrying the record.
    """
    record = message.record
    text = record["message"]  # Already formatted with Loguru's braces.
    style = LEVEL_STYLES.get(record["level"].name, "")
    console.log(text, style=style, markup=False, highlight=False)


def setup_logging(
    level: str = "INFO",
    diagnose: bool = False,
    intercept_stdlib: bool = True,
    redirect_stdio: bool = False,
) -> None:
    """Configure Loguru to write through the Rich console.

    Safe to call more than once; calls after the first are ignored.

    Args:
        level (str): Minimum level to display. Defaults to "INFO".
        diagnose (bool): Enable diagnostic detail in tracebacks. Defaults to False.
        intercept_stdlib (bool): Route stdlib logging into Loguru.
            Defaults to True.
        redirect_stdio (bool): Route stdout and stderr into Loguru.
            Defaults to False.
    """
    global _logging_configured

    if _logging_configured:
        return

    # Replace default handlers; Rich supplies the styling, so no color codes.
    logger.remove()
    logger.add(
        rich_sink,
        level=level,
        colorize=False,
        enqueue=True,  # Thread-safe, keeps progress rendering smooth.
        backtrace=True,
        diagnose=diagnose,
    )

    if intercept_stdlib:
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        for log_name in logging.root.manager.loggerDict:
            logging.getLogger(log_name).handlers = []
            logging.getLogger(log_name).propagate = True

    if redirect_stdio:
        sys.stdout = StdoutInterceptor()  # type: ignore[assignment]
        sys.stderr = StderrInterceptor()  # type: ignore[assignment]

    _logging_configured = True
