"""Type definitions for benchmark prediction modules.

Provides TypedDict and type aliases for usage data structures
to ensure proper type checking throughout the codebase.
"""

from __future__ import annotations

from typing import TypedDict, cast

#: The only benchmark agent type the harness runs.
AGENT_TYPE = "bench-agent"


class UsageRecordData(TypedDict, total=False):
    """Type for a single usage record (e.g., 'none', 'total', or tool-specific).

    Attributes:
        session_id: Unique session identifier
        tokens_input: Number of input tokens used
        tokens_output: Number of output tokens generated
        tokens_cache_read: Number of cache read tokens
        tokens_cache_write: Number of cache write tokens
        tokens_total: Total token count
        cost_total: Total cost in USD, or None when the cost is unknown
        cost_complete: Whether the cost accounting for the record is complete
        maximum_context: Maximum context size encountered
    """

    session_id: str
    tokens_input: int
    tokens_output: int
    tokens_cache_read: int
    tokens_cache_write: int
    tokens_total: int
    cost_total: float | None
    cost_complete: bool
    maximum_context: int


class UsageData(TypedDict, total=False):
    """Type for complete usage.json structure from the agent harness.

    Contains usage records keyed by record type (e.g., 'none', 'total')
    or tool-specific identifiers.
    """

    none: UsageRecordData
    total: UsageRecordData


# Type alias for token usage returned by DockerRunner.run_agent()
TokenUsage = dict[str, float | UsageData | None]

# Type for session usage accumulator: "session_id", "none"/"total" accumulators,
# and per-instance UsageData keyed by instance id.
SessionUsage = dict[str, str | None | dict[str, int | float] | UsageData]


def get_accumulator_from_agent_data(
    session_usage: SessionUsage, key: str
) -> dict[str, int | float] | None:
    """Get an accumulator dict from agent session data if it exists.

    Args:
        session_usage: The session usage dict
        key: The key to look up (e.g., "none", "total")

    Returns:
        The accumulator dict if found and is an accumulator, None otherwise
    """
    value = session_usage.get(key)
    # Check if it's an accumulator dict (all values are numeric) vs UsageData
    # (has nested dicts). Returned as-is so callers can mutate it in place.
    if isinstance(value, dict) and all(
        isinstance(v, (int, float)) for v in value.values()
    ):
        return cast(dict[str, int | float], value)
    return None
