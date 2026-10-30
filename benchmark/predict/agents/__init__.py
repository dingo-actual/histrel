#!/usr/bin/env python3
"""
Benchmark agent wrappers for SWE-bench evaluation.

This package provides the wrapper for running the
benchmark agent against SWE-bench test cases. The wrapper handles
initialization, prompt construction, patch generation via git diff, and
token tracking.

    Important Classes:
        BenchAgentWrapper: Wrapper for running the benchmark agent on SWE-bench instances.
"""

from __future__ import annotations

from benchmark.predict.agents.bench_agent_wrapper import BenchAgentWrapper

__all__ = [
    "BenchAgentWrapper",
]
