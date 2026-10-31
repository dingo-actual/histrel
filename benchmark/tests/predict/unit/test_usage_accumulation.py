#!/usr/bin/env python3
"""Unit tests for benchmark usage accumulation, including cost tracking.

Verifies that :func:`benchmark.predict.usage_helpers.accumulate_usage` folds
per-instance usage records into session totals:

- Integer token fields accumulate by addition
- Float ``cost_total`` accumulates too, and stays at its 0.0 placeholder when
  records omit cost
- ``maximum_context`` uses max() instead of addition
"""

from __future__ import annotations

import pytest

from benchmark.predict.runner import _empty_usage_accumulator
from benchmark.predict.types import (
    SessionUsage,
    UsageData,
    get_accumulator_from_agent_data,
)
from benchmark.predict.usage_helpers import accumulate_usage


def _make_session_usage() -> SessionUsage:
    return {
        "session_id": None,
        "none": _empty_usage_accumulator(),
        "total": _empty_usage_accumulator(),
    }


def _read_accumulator(session_usage: SessionUsage, key: str) -> dict[str, int | float]:
    record = get_accumulator_from_agent_data(session_usage, key)
    assert record is not None
    return record


class Test04Unit02UsageAccumulation01Tokens:
    def test_token_fields_accumulate_and_context_uses_max(self):
        session_usage = _make_session_usage()
        instance_usage: UsageData = {
            "none": {
                "session_id": "s1",
                "tokens_input": 100,
                "tokens_output": 40,
                "tokens_cache_read": 60,
                "tokens_cache_write": 5,
                "tokens_total": 140,
                "maximum_context": 1000,
            },
            "total": {
                "session_id": "s1",
                "tokens_input": 200,
                "tokens_output": 80,
                "tokens_cache_read": 120,
                "tokens_cache_write": 10,
                "tokens_total": 280,
                "maximum_context": 900,
            },
        }

        accumulate_usage(session_usage, instance_usage)

        total = _read_accumulator(session_usage, "total")
        none_record = _read_accumulator(session_usage, "none")
        assert total["tokens_input"] == 200
        assert total["tokens_output"] == 80
        assert total["tokens_cache_read"] == 120
        assert total["tokens_cache_write"] == 10
        assert total["tokens_total"] == 280
        assert total["maximum_context"] == 900
        assert none_record["maximum_context"] == 1000


class Test04Unit02UsageAccumulation02Cost:
    def test_cost_accumulates_when_reported(self):
        session_usage = _make_session_usage()
        instance_usage: UsageData = {
            "none": {
                "session_id": "s1",
                "tokens_input": 100,
                "tokens_output": 40,
                "tokens_total": 140,
                "cost_total": 0.0125,
                "maximum_context": 100,
            },
            "total": {
                "session_id": "s1",
                "tokens_input": 200,
                "tokens_output": 80,
                "tokens_total": 280,
                "cost_total": 0.0375,
                "maximum_context": 100,
            },
        }

        accumulate_usage(session_usage, instance_usage)

        total = _read_accumulator(session_usage, "total")
        none_record = _read_accumulator(session_usage, "none")
        assert total["cost_total"] == pytest.approx(0.0375)
        assert none_record["cost_total"] == pytest.approx(0.0125)

    def test_cost_placeholder_stays_zero_when_records_omit_cost(self):
        # Usage records without cost fields must leave the zero placeholder
        # intact, keep token accumulation unaffected, and count the unknown
        # cost instead of treating it as a free run.
        session_usage = _make_session_usage()
        instance_usage: UsageData = {
            "none": {
                "session_id": "s1",
                "tokens_input": 100,
                "tokens_output": 40,
                "tokens_cache_read": 0,
                "tokens_cache_write": 0,
                "tokens_total": 140,
                "maximum_context": 100,
            },
            "total": {
                "session_id": "s1",
                "tokens_input": 200,
                "tokens_output": 80,
                "tokens_cache_read": 0,
                "tokens_cache_write": 0,
                "tokens_total": 280,
                "maximum_context": 100,
            },
        }

        accumulate_usage(session_usage, instance_usage)

        total = _read_accumulator(session_usage, "total")
        none_record = _read_accumulator(session_usage, "none")
        assert total["cost_total"] == 0.0
        assert none_record["cost_total"] == 0.0
        assert total["cost_unknown_instances"] == 1
        assert none_record["cost_unknown_instances"] == 1
        assert total["tokens_total"] == 280

    def test_unknown_cost_is_counted_not_summed(self):
        # A failed task reports an unknown cost; it must not be summed as 0
        # or -1, and the incompleteness must stay visible.
        session_usage = _make_session_usage()
        failed: UsageData = {
            "total": {
                "session_id": "s1",
                "tokens_input": 100,
                "tokens_output": 40,
                "tokens_total": 140,
                "cost_total": None,
                "cost_complete": False,
                "maximum_context": 100,
            }
        }
        succeeded: UsageData = {
            "total": {
                "session_id": "s2",
                "tokens_input": 300,
                "tokens_output": 60,
                "tokens_total": 360,
                "cost_total": 0.02,
                "cost_complete": True,
                "maximum_context": 200,
            }
        }

        accumulate_usage(session_usage, failed)
        accumulate_usage(session_usage, succeeded)

        total = _read_accumulator(session_usage, "total")
        assert total["cost_total"] == pytest.approx(0.02)
        assert total["cost_unknown_instances"] == 1
        assert total["cost_complete"] is False
        assert total["tokens_total"] == 500
        assert total["maximum_context"] == 200

    def test_mixed_instances_sum_cost(self):
        session_usage = _make_session_usage()
        first: UsageData = {
            "total": {
                "session_id": "s1",
                "tokens_input": 100,
                "tokens_output": 40,
                "tokens_total": 140,
                "cost_total": 0.01,
                "maximum_context": 100,
            }
        }
        second: UsageData = {
            "total": {
                "session_id": "s2",
                "tokens_input": 300,
                "tokens_output": 60,
                "tokens_total": 360,
                "cost_total": 0.0025,
                "maximum_context": 200,
            }
        }

        accumulate_usage(session_usage, first)
        accumulate_usage(session_usage, second)

        total = _read_accumulator(session_usage, "total")
        assert total["cost_total"] == pytest.approx(0.0125)
        assert total["tokens_total"] == 500
        assert total["maximum_context"] == 200

    def test_partial_cost_keeps_charges_but_stays_incomplete(self):
        # An explicitly incomplete record still reported charges for the
        # calls it could price. Keep the money, lose the completeness.
        session_usage = _make_session_usage()
        partial: UsageData = {
            "total": {
                "session_id": "s1",
                "tokens_input": 100,
                "tokens_output": 40,
                "tokens_total": 140,
                "cost_total": 0.01,
                "cost_complete": False,
                "maximum_context": 100,
            }
        }

        accumulate_usage(session_usage, partial)

        total = _read_accumulator(session_usage, "total")
        assert total["cost_total"] == pytest.approx(0.01)
        assert total["cost_complete"] is False
        assert total["cost_unknown_instances"] == 0

    def test_complete_costs_leave_total_complete(self):
        session_usage = _make_session_usage()
        complete: UsageData = {
            "total": {
                "session_id": "s1",
                "tokens_input": 100,
                "tokens_output": 40,
                "tokens_total": 140,
                "cost_total": 0.01,
                "cost_complete": True,
                "maximum_context": 100,
            }
        }

        accumulate_usage(session_usage, complete)

        total = _read_accumulator(session_usage, "total")
        assert total["cost_total"] == pytest.approx(0.01)
        assert total["cost_complete"] is True

    def test_incomplete_record_taints_later_complete_total(self):
        # Completeness is sticky: a later complete record must not relabel an
        # earlier partial total as complete.
        session_usage = _make_session_usage()
        partial: UsageData = {
            "total": {"tokens_total": 140, "cost_total": 0.01, "cost_complete": False}
        }
        complete: UsageData = {
            "total": {"tokens_total": 360, "cost_total": 0.02, "cost_complete": True}
        }

        accumulate_usage(session_usage, complete)
        accumulate_usage(session_usage, partial)

        total = _read_accumulator(session_usage, "total")
        assert total["cost_total"] == pytest.approx(0.03)
        assert total["cost_complete"] is False


class Test04Unit03UsageAccumulation03PerToolKeys:
    """Per-tool usage keys from Histrel calls survive into session_usage.json.

    The bench agent produces usage data with per-tool keys (run IDs like
    'SESSION-123-analyze-query') in addition to 'none' and 'total'. The
    runner stores the full usage dict under session_usage[instance_id];
    accumulate_usage only processes 'none'/'total' for the top-level
    accumulators. The per-tool keys must survive at the instance level so
    they appear in session_usage.json.
    """

    def test_per_tool_keys_preserved_at_instance_level(self):
        session_usage = _make_session_usage()
        instance_usage: dict = {
            "none": {
                "session_id": "s1",
                "tokens_input": 100,
                "tokens_output": 40,
                "tokens_total": 140,
                "maximum_context": 100,
            },
            "total": {
                "session_id": "s1",
                "tokens_input": 6000,
                "tokens_output": 2500,
                "tokens_total": 8500,
                "maximum_context": 7000,
            },
            "s1-123-analyze-query": {
                "session_id": "s1",
                "tokens_input": 5000,
                "tokens_output": 2000,
                "tokens_total": 7000,
                "maximum_context": 7000,
            },
        }

        # Simulate what runner.py does: store under session_usage[instance_id]
        session_usage["instance-1"] = instance_usage  # type: ignore[assignment]

        # Also accumulate 'none'/'total' into the top-level
        accumulate_usage(session_usage, instance_usage)  # type: ignore[arg-type]

        # The per-tool key must be accessible at the instance level
        instance_data = session_usage["instance-1"]
        assert isinstance(instance_data, dict)
        assert "s1-123-analyze-query" in instance_data
        tool_entry = instance_data["s1-123-analyze-query"]  # type: ignore[typeddict-item]
        assert isinstance(tool_entry, dict)
        assert tool_entry["tokens_input"] == 5000  # type: ignore[typeddict-item]
        assert tool_entry["tokens_total"] == 7000  # type: ignore[typeddict-item]
        # Top-level total still accumulates correctly
        total = _read_accumulator(session_usage, "total")
        assert total["tokens_total"] == 8500
