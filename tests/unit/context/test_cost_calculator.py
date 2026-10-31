#!/usr/bin/env python3
"""Unit tests for cost calculation and report generation.

Tests CostCalculator with real ModelPricing data and TokenLedger
entries. Uses parametrized tests for broad coverage. No mocks.
"""

from __future__ import annotations

import pytest

from histrel.context.cost_calculator import CostCalculator
from histrel.context.token_ledger import TokenLedger
from histrel.model.types import Usage
from tests.unit.context.helpers import TEST_PRICING


def _record_model_call(
    ledger: TokenLedger,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    cached: int = 0,
    reasoning: int = 0,
    request_id: str = "req-1",
) -> None:
    """Record a model call in the token ledger."""
    ledger.record(
        message_index=0,
        usage=Usage(
            input=input_tokens,
            output=output_tokens,
            reasoning=reasoning,
            total=input_tokens + output_tokens,
            cached=cached,
        ),
        provider_request_id=request_id,
        call_kind="model",
    )


def _record_compaction_call(
    ledger: TokenLedger,
    input_tokens: int = 2000,
    output_tokens: int = 500,
    request_id: str = "req-compaction",
) -> None:
    """Record a compaction call in the token ledger."""
    ledger.record(
        message_index=5,
        usage=Usage(
            input=input_tokens,
            output=output_tokens,
            total=input_tokens + output_tokens,
        ),
        provider_request_id=request_id,
        call_kind="compaction",
    )


def _record_routine_call(
    ledger: TokenLedger,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    request_id: str = "req-routine",
) -> None:
    """Record an LLM routine call in the token ledger."""
    ledger.record(
        message_index=3,
        usage=Usage(
            input=input_tokens,
            output=output_tokens,
            total=input_tokens + output_tokens,
        ),
        provider_request_id=request_id,
        call_kind="routine",
    )


class Test04Unit01Cost01Calculator:
    """CostCalculator single-call cost computation."""

    @pytest.mark.parametrize(
        "input_tokens, output_tokens, cached, expected_cost",
        [
            # No cached tokens: all input at uncached rate
            (100000, 50000, 0, (3 * 100000 + 8 * 50000) / 1e6),
            # 30% cached: 70k uncached + 30k cached + 50k output
            (100000, 50000, 30000, (3 * 70000 + 0.3 * 30000 + 8 * 50000) / 1e6),
            # All cached input
            (100000, 0, 100000, (0.3 * 100000) / 1e6),
            # Zero tokens
            (0, 0, 0, 0.0),
        ],
        ids=[
            "no_cache",
            "partial_cache",
            "all_cached",
            "zero_tokens",
        ],
    )
    def test_04_unit_01_cost_01_calc_01_single_call_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached: int,
        expected_cost: float,
    ) -> None:
        # Arrange
        calculator = CostCalculator(pricing=TEST_PRICING)
        usage = Usage(
            input=input_tokens,
            output=output_tokens,
            total=input_tokens + output_tokens,
            cached=cached,
        )

        # Act
        call_cost = calculator.compute_call_cost(usage)

        # Assert
        assert call_cost is not None
        assert abs(call_cost.cost_usd - expected_cost) < 1e-9

    def test_04_unit_01_cost_01_calc_02_none_pricing_returns_none(self) -> None:
        # Arrange
        calculator = CostCalculator(pricing=None)
        usage = Usage(input=1000, output=500, total=1500)

        # Act
        call_cost = calculator.compute_call_cost(usage)

        # Assert — when pricing is unavailable, cost is None
        assert call_cost is None


class Test04Unit02Cost02Report:
    """CostCalculator report generation from TokenLedger."""

    def test_04_unit_02_cost_02_report_01_aggregates_model_calls(self) -> None:
        # Arrange
        ledger = TokenLedger()
        _record_model_call(
            ledger, input_tokens=10000, output_tokens=5000, request_id="r1"
        )
        _record_model_call(
            ledger, input_tokens=20000, output_tokens=8000, request_id="r2"
        )
        _record_routine_call(
            ledger, input_tokens=1000, output_tokens=100, request_id="r-routine"
        )
        calculator = CostCalculator(pricing=TEST_PRICING)

        # Act
        report = calculator.compute_report(ledger)

        # Assert — routine calls cost into model_cost and appear in
        # the per-call breakdown with their own call kind.
        total_cost = report.total_cost
        model_cost = report.model_cost
        compaction_cost = report.compaction_cost
        assert total_cost is not None
        assert model_cost is not None
        assert compaction_cost is not None
        assert total_cost > 0
        assert model_cost > 0
        assert len(report.call_costs) == 3
        routine_call = [c for c in report.call_costs if c.call_kind == "routine"]
        assert len(routine_call) == 1
        assert routine_call[0].provider_request_id == "r-routine"
        routine_cost = (3 * 1000 + 8 * 100) / 1e6
        assert abs(routine_call[0].cost_usd - routine_cost) < 1e-9
        model_cost_expected = (3 * 30000 + 8 * 13000) / 1e6 + routine_cost
        assert abs(model_cost - model_cost_expected) < 1e-9
        assert abs(total_cost - model_cost - compaction_cost) < 1e-9

    def test_04_unit_02_cost_02_report_02_separates_compaction_cost(self) -> None:
        # Arrange
        ledger = TokenLedger()
        _record_model_call(
            ledger, input_tokens=10000, output_tokens=5000, request_id="r1"
        )
        _record_compaction_call(ledger, input_tokens=5000, output_tokens=1000)
        calculator = CostCalculator(pricing=TEST_PRICING)

        # Act
        report = calculator.compute_report(ledger)

        # Assert
        total_cost = report.total_cost
        model_cost = report.model_cost
        compaction_cost = report.compaction_cost
        assert total_cost is not None
        assert model_cost is not None
        assert compaction_cost is not None
        assert model_cost > 0
        assert compaction_cost > 0
        assert total_cost == model_cost + compaction_cost

    def test_04_unit_02_cost_02_report_03_no_pricing_marks_incomplete(self) -> None:
        # Arrange
        ledger = TokenLedger()
        _record_model_call(ledger, request_id="r1")
        calculator = CostCalculator(pricing=None)

        # Act
        report = calculator.compute_report(ledger)

        # Assert — missing pricing makes the cost unknown, not free.
        assert report.is_complete is False
        assert report.total_cost is None
        assert report.model_cost is None
        assert report.compaction_cost is None

    def test_04_unit_02_cost_02_report_04_complete_when_pricing_available(
        self,
    ) -> None:
        # Arrange
        ledger = TokenLedger()
        _record_model_call(ledger, request_id="r1")
        calculator = CostCalculator(pricing=TEST_PRICING)

        # Act
        report = calculator.compute_report(ledger)

        # Assert
        assert report.is_complete is True

    def test_04_unit_02_cost_02_report_05_empty_ledger(self) -> None:
        # Arrange
        ledger = TokenLedger()
        calculator = CostCalculator(pricing=TEST_PRICING)

        # Act
        report = calculator.compute_report(ledger)

        # Assert
        assert report.total_cost == 0.0
        assert report.model_cost == 0.0
        assert report.compaction_cost == 0.0
        assert len(report.call_costs) == 0
        assert report.is_complete is True
