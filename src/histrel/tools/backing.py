#!/usr/bin/env python3
"""
Backing binary verification for search and glob tools.

Verifies that ripgrep and fd are installed with the required versions
at startup, before the first model call. Fails fast on any mismatch.

    Important Constants:
        MINIMUM_RIPGREP_VERSION: Minimum required ripgrep version.
        REQUIRED_FD_VERSION: Exact required fd version.

    Important Functions:
        parse_version(output) -> tuple[int, ...]: Parse a version string.
        verify_search_backends() -> None: Verify rg and fd versions.
"""

from __future__ import annotations

import re
import subprocess

from loguru import logger

from histrel.core.errors import ToolBackendError

MINIMUM_RIPGREP_VERSION: tuple[int, ...] = (14, 0, 0)
REQUIRED_FD_VERSION: tuple[int, ...] = (10, 3, 0)


def parse_version(output: str) -> tuple[int, ...]:
    """Parse a version string from command output.

    Extracts the first dotted numeric version string from the output.

    Args:
        output (str): Command output (e.g. "ripgrep 14.1.0 (rev abc)").

    Returns:
        (tuple[int, ...]) Parsed version as a tuple of integers.

    Raises:
        ToolBackendError: If no version string is found.
    """
    match = re.search(r"(\d+(?:\.\d+)+)", output)
    if not match:
        raise ToolBackendError(f"unparseable version output: {output!r}")
    return tuple(int(part) for part in match.group(1).split("."))


def verify_search_backends() -> None:
    """Verify that ripgrep and fd meet version requirements.

    Runs both binaries and checks their versions. Fails fast with
    ToolBackendError if either is missing or does not meet the
    required version.

    Raises:
        ToolBackendError: If a binary is missing or has the wrong version.
    """
    log = logger.bind(component="bench_agent.tools.backing")

    try:
        rg_output = subprocess.run(
            ["rg", "--version"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ToolBackendError("ripgrep not found or failed to run") from exc

    try:
        fd_output = subprocess.run(
            ["fd", "--version"], capture_output=True, text=True, check=True
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        raise ToolBackendError("fd not found or failed to run") from exc

    rg_version = parse_version(rg_output)
    fd_version = parse_version(fd_output)

    log.debug(
        "backing_versions_verified",
        ripgrep_version=".".join(map(str, rg_version)),
        fd_version=".".join(map(str, fd_version)),
    )

    if rg_version < MINIMUM_RIPGREP_VERSION:
        raise ToolBackendError(
            f"ripgrep >= 14 required, found {'.'.join(map(str, rg_version))}"
        )
    if fd_version != REQUIRED_FD_VERSION:
        raise ToolBackendError(
            f"fd 10.3.0 required, found {'.'.join(map(str, fd_version))}"
        )
