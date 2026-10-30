"""Helper functions for processing benchmark agent usage data."""

from __future__ import annotations

from benchmark.predict.types import (
    SessionUsage,
    UsageData,
    get_accumulator_from_agent_data,
)


def accumulate_usage(session_usage: SessionUsage, instance_usage: UsageData) -> None:
    """Accumulate usage data for an instance into session totals.

    Updates the "none" and "total" records in session_usage by adding
    values from instance_usage. For maximum_context, uses max() instead
    of addition.

    Args:
        session_usage (SessionUsage): The session usage structure to update.
        instance_usage (UsageData): Complete usage data from a single instance run.
    """
    for record_type in ("none", "total"):
        record_raw = instance_usage.get(record_type)
        if not isinstance(record_raw, dict):
            continue
        target_record = get_accumulator_from_agent_data(session_usage, record_type)
        if target_record is None:
            continue
        for field, value in record_raw.items():
            if field == "session_id" or not isinstance(value, (int, float)):
                continue
            if field == "maximum_context":
                target_record[field] = max(target_record.get(field, 0), value)
            else:
                target_record[field] = target_record.get(field, 0) + value
