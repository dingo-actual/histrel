#!/usr/bin/env python3
"""Pytest configuration for benchmark tests.

Ensures the project root is on sys.path so absolute imports like
`from benchmark.config import BenchmarkConfig` work during test collection.
Also handles opt-in test markers (--llm, --slow, --tools, --all) for
benchmark tests, mirroring the root tests/conftest.py behavior.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure shared fixtures are imported so pytest registers them
# Suppress noisy vendor warnings (pydantic v2 deprecations in SWE-agent)
import warnings  # noqa: E402

from benchmark.tests.fixtures.docker_stubs import docker_client_stub  # noqa: E402, F401

try:  # Target the specific pydantic warning class when available
    from pydantic.warnings import PydanticDeprecatedSince211  # type: ignore

    warnings.filterwarnings(
        "ignore",
        category=PydanticDeprecatedSince211,
        message=r"Accessing the 'model_fields' attribute.*",
    )
except Exception:
    warnings.filterwarnings(
        "ignore",
        message=r"Accessing the 'model_fields' attribute.*PydanticDeprecatedSince211",
    )


def pytest_addoption(parser):
    """Add custom command-line options for opt-in tests.

    Guards against duplicate registration when the root tests/conftest.py
    has already added these options.
    """
    for opts, kwargs in [
        (
            ("--llm",),
            {
                "action": "store_true",
                "default": False,
                "help": "Run integration tests that make real LLM API calls",
            },
        ),
        (
            ("--slow",),
            {
                "action": "store_true",
                "default": False,
                "help": "Run tests marked as slow_test; otherwise they are skipped",
            },
        ),
        (
            ("--tools",),
            {
                "action": "store_true",
                "default": False,
                "help": (
                    "Run tests that require external tools (e.g., ripgrep, tree, "
                    "Exa/Tavily clients); otherwise they are skipped"
                ),
            },
        ),
        (
            ("--all",),
            {
                "action": "store_true",
                "default": False,
                "help": (
                    "Enable all opt-in tests and behaviors: equivalent to "
                    "--slow --llm --tools"
                ),
            },
        ),
    ]:
        with contextlib.suppress(ValueError):
            parser.addoption(*opts, **kwargs)


def pytest_configure(config):
    """Register custom markers for opt-in tests."""
    config.addinivalue_line(
        "markers",
        (
            "requires_live_llm: mark test as requiring live LLM API calls"
            "(skipped unless --llm or --all)"
        ),
    )
    config.addinivalue_line(
        "markers",
        (
            "requires_tools: mark test as requiring external tools (e.g., "
            "ripgrep, tree, Exa/Tavily clients); skipped unless --tools or --all"
        ),
    )
    config.addinivalue_line(
        "markers",
        "slow_test: mark test as slow (skipped unless --slow or --all)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip opt-in tests unless their flags are provided."""

    def _env_truthy(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

    with_llm = config.getoption("--llm") or _env_truthy("HISTREL_WITH_LLM")
    run_slow = config.getoption("--slow")
    with_tools = config.getoption("--tools") or _env_truthy("HISTREL_WITH_TOOLS")

    if config.getoption("--all"):
        with_llm = True
        run_slow = True
        with_tools = True

    skip_live_llm = pytest.mark.skip(
        reason="Need --llm flag to run tests with live LLM API calls"
    )
    skip_tools = pytest.mark.skip(
        reason="Need --tools flag to run tests that require external tools"
    )
    skip_slow = pytest.mark.skip(
        reason="Need --slow flag to run tests marked as slow_test"
    )

    for item in items:
        if not with_llm and item.get_closest_marker("requires_live_llm"):
            item.add_marker(skip_live_llm)
        if not with_tools and item.get_closest_marker("requires_tools"):
            item.add_marker(skip_tools)
        if not run_slow and item.get_closest_marker("slow_test"):
            item.add_marker(skip_slow)
