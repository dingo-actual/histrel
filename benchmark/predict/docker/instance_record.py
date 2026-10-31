#!/usr/bin/env python3
"""Agent-facing projection of a SWE-bench instance record.

Dataset records carry fields the agent must never see: the reference ``patch``,
``test_patch``, and evaluation metadata such as ``FAIL_TO_PASS`` /
``PASS_TO_PASS``. Containers receive only the allowlisted fields below, so the
task stays solvable while the ground truth stays on the host.

    Important Functions:
        build_agent_instance(instance) -> dict[str, Any]
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

#: Instance fields that may be written to the container-visible record.
AGENT_INSTANCE_FIELDS = (
    "instance_id",
    "repo",
    "base_commit",
    "problem_statement",
    "hints_text",
    "has_test_runner",
)


def build_agent_instance(instance: Mapping[str, Any]) -> dict[str, Any]:
    """Build the allowlisted agent-facing record for an instance.

    Args:
        instance (Mapping[str, Any]): Full SWE-bench dataset record.

    Returns:
        (dict[str, Any]) Task fields only; absent fields are omitted.
    """
    return {
        field: instance[field] for field in AGENT_INSTANCE_FIELDS if field in instance
    }
