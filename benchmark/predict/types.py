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
    or tool-specific identifiers, plus the identity of the attempt that
    produced them.
    """

    identity: AttemptIdentity
    none: UsageRecordData
    total: UsageRecordData


class AttemptIdentity(TypedDict):
    """Identity of one benchmark attempt (instance x repetition x mode).

    Attributes:
        run_name: Run directory name; a repeated run uses ``repeat_NN``.
        repetition: 1-based repetition number.
        mode: Agent mode the attempt ran with.
        agent_type: Benchmark agent type.
        instance_id: SWE-bench instance identifier.
        session_id: Harness session identifier for this attempt.
    """

    run_name: str
    repetition: int
    mode: str
    agent_type: str
    instance_id: str
    session_id: str


def build_attempt_identity(
    *,
    run_name: str,
    repetition: int,
    mode: str,
    instance_id: str,
    session_id: str,
) -> AttemptIdentity:
    """Build the identity of one benchmark attempt.

    Args:
        run_name (str): Run directory name.
        repetition (int): 1-based repetition number.
        mode (str): Agent mode.
        instance_id (str): SWE-bench instance identifier.
        session_id (str): Harness session identifier.

    Returns:
        (AttemptIdentity) Identity recorded by the container and checked
            by the host before ingesting a usage record.
    """
    return AttemptIdentity(
        run_name=run_name,
        repetition=int(repetition),
        mode=mode,
        agent_type=AGENT_TYPE,
        instance_id=instance_id,
        session_id=session_id,
    )


def identity_matches(identity: object, expected: AttemptIdentity) -> bool:
    """Return whether a recorded identity matches the expected attempt.

    Args:
        identity (object): Identity read from a usage record.
        expected (AttemptIdentity): Identity of the attempt being ingested.

    Returns:
        (bool) True only when every field matches. A missing or malformed
            identity never matches.
    """
    if not isinstance(identity, dict):
        return False
    recorded = cast(dict[str, object], identity)
    return all(recorded.get(key) == value for key, value in expected.items())


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
