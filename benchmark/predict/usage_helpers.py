"""Helper functions for processing benchmark agent usage data."""

from __future__ import annotations

import math

from benchmark.predict.types import (
    SessionUsage,
    UsageData,
    get_accumulator_from_agent_data,
)


def known_measurement(value: object) -> int | float | None:
    """Return a usable measurement from a usage record.

    Missing values, booleans, non-numbers, non-finite numbers, and the
    legacy ``-1`` "unknown" sentinel are not measurements.

    Args:
        value (object): Candidate value from a usage record.

    Returns:
        (int | float | None) The measurement, or None when unavailable.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        numeric = float(value)
    except OverflowError:
        return None
    if not math.isfinite(numeric) or numeric < 0:
        return None
    return value


def known_cost(value: object) -> float | None:
    """Return a reported cost in USD, or None when the cost is unknown.

    Args:
        value (object): Candidate cost from a usage record.

    Returns:
        (float | None) The cost, or None when the cost is unknown.
    """
    measurement = known_measurement(value)
    return None if measurement is None else float(measurement)


def accumulate_usage(session_usage: SessionUsage, instance_usage: UsageData) -> None:
    """Accumulate usage data for an instance into session totals.

    Updates the "none" and "total" records in session_usage by adding
    values from instance_usage. For maximum_context, uses max() instead
    of addition. Instances with an unknown cost are counted in
    ``cost_unknown_instances`` instead of being folded into ``cost_total``.
    A record marked ``cost_complete: False`` keeps its known charges in
    ``cost_total`` but flips the accumulator's own ``cost_complete`` to
    False, so the total is reported as partial.

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
        cost = known_cost(record_raw.get("cost_total"))
        if cost is None:
            target_record["cost_unknown_instances"] = (
                target_record.get("cost_unknown_instances", 0) + 1
            )
            target_record["cost_complete"] = False
        else:
            target_record["cost_total"] = target_record.get("cost_total", 0.0) + cost
            # A partial cost keeps its known charges, but the session total
            # must not look complete.
            if record_raw.get("cost_complete") is not True:
                target_record["cost_complete"] = False
        for field, value in record_raw.items():
            if field in {"session_id", "cost_complete", "cost_total"}:
                continue
            measurement = known_measurement(value)
            if measurement is None:
                continue
            if field == "maximum_context":
                target_record[field] = max(target_record.get(field, 0), measurement)
            else:
                target_record[field] = target_record.get(field, 0) + measurement
