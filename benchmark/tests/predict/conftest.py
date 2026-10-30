#!/usr/bin/env python3
"""Predictor-specific test fixtures."""

from __future__ import annotations

import pytest

from benchmark.predict.config import PredictorConfig

# Re-export fixtures so pytest can discover them
__all__ = ["predictor_config"]


@pytest.fixture
def predictor_config() -> PredictorConfig:
    """Provide default predictor config for tests."""
    return PredictorConfig(
        output_dir="test_output",
        run_name="test-run",
        verbose=False,
    )
