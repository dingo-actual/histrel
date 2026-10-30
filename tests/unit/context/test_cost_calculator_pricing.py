#!/usr/bin/env python3
"""Unit tests for resilient Fireworks pricing resolution.

Tests the retry, persistent-cache, catalog-fallback, and failure-TTL
behavior of the cost calculator's pricing lookup. Uses a patched
urlopen and real temporary cache files (no network, no mocks).
"""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from histrel.context import cost_calculator as cc

_PAGE_HTML = (
    '<html><script>self.__next_f.push([1,"...skuInfos\\":[{\\"$typeName\\":'
    '\\"gateway.SKUInfo\\",\\"sku\\":\\"LLM input tokens (cached)\\",'
    '\\"unit\\":\\"1M tokens\\",\\"amount\\":{\\"$typeName\\":'
    '\\"google.type.Money\\",\\"currencyCode\\":\\"USD\\",\\"units\\":\\"$n0\\",'
    '\\"nanos\\":30000000}},{\\"$typeName\\":\\"gateway.SKUInfo\\",'
    '\\"sku\\":\\"LLM output tokens\\",\\"unit\\":\\"1M tokens\\",'
    '\\"amount\\":{\\"$typeName\\":\\"google.type.Money\\",'
    '\\"currencyCode\\":\\"USD\\",\\"units\\":\\"$n2\\",\\"nanos\\":500000000}},'
    '{\\"$typeName\\":\\"gateway.SKUInfo\\",\\"sku\\":\\"LLM input tokens '
    '(uncached)\\",\\"unit\\":\\"1M tokens\\",\\"amount\\":{\\"$typeName\\":'
    '\\"google.type.Money\\",\\"currencyCode\\":\\"USD\\",\\"units\\":\\"$n4\\",'
    '\\"nanos\\":150000000}}]"})</script></html>'
)

_CATALOG_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash-1017"
_UNKNOWN_MODEL = "accounts/fireworks/models/absent-from-catalog"


class _FakeResponse:
    """Context-manager HTTP response returning fixed HTML (no network)."""

    def __init__(self, html: str) -> None:
        self._html = html

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def read(self) -> bytes:
        return self._html.encode("utf-8")


@pytest.fixture(autouse=True)
def _clean_pricing_cache() -> Iterator[None]:
    """Isolate the process-level pricing cache between tests."""
    cc.clear_pricing_cache()
    yield
    cc.clear_pricing_cache()


@pytest.fixture(autouse=True)
def _isolated_persistent_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect the persistent pricing cache to a per-test file."""
    cache_file = tmp_path / "pricing_cache.json"
    monkeypatch.setenv(cc.PRICING_CACHE_ENV, str(cache_file))
    return cache_file


@pytest.fixture(autouse=True)
def _instant_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove real retry sleeps so failed-fetch tests stay fast."""
    monkeypatch.setattr(cc, "_sleep", lambda seconds: None)


@pytest.fixture()
def fake_clock(monkeypatch: pytest.MonkeyPatch) -> dict[str, float]:
    """Provide a controllable monotonic clock for the failure TTL."""
    state: dict[str, float] = {"now": 1000.0}
    monkeypatch.setattr(cc, "_now", lambda: state["now"])
    return state


@pytest.fixture()
def capture_sink() -> Iterator[list[dict[str, Any]]]:
    """Provide a capturing sink that collects message.record dicts."""
    captured: list[dict[str, Any]] = []

    def _capture(message: Any) -> None:
        try:
            captured.append(dict(message.record))
        except Exception:
            captured.append({"message": str(message)})

    sink_id = logger.add(_capture, level="DEBUG")
    try:
        yield captured
    finally:
        logger.remove(sink_id)


def _patch_urlopen_always_fail(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Patch urlopen to always fail; return the call log."""
    calls: list[str] = []

    def _fake(request: Any, timeout: float) -> Any:
        calls.append(str(getattr(request, "full_url", "")))
        raise OSError("network down")

    monkeypatch.setattr(urllib.request, "urlopen", _fake)
    return calls


def _patch_urlopen_flaky_then_succeed(
    monkeypatch: pytest.MonkeyPatch, *, failures: int, html: str
) -> list[str]:
    """Patch urlopen to fail ``failures`` times then succeed."""
    calls: list[str] = []
    attempts = {"count": 0}

    def _fake(request: Any, timeout: float) -> Any:
        calls.append(str(getattr(request, "full_url", "")))
        attempts["count"] += 1
        if attempts["count"] <= failures:
            raise OSError("network down")
        return _FakeResponse(html)

    monkeypatch.setattr(urllib.request, "urlopen", _fake)
    return calls


def _patch_urlopen_succeed(monkeypatch: pytest.MonkeyPatch, *, html: str) -> list[str]:
    """Patch urlopen to always succeed with ``html``."""
    calls: list[str] = []

    def _fake(request: Any, timeout: float) -> Any:
        calls.append(str(getattr(request, "full_url", "")))
        return _FakeResponse(html)

    monkeypatch.setattr(urllib.request, "urlopen", _fake)
    return calls


class Test04Unit03Cost03PricingResolution:
    """Resilient Fireworks pricing resolution for the benchmark agent."""

    def test_04_unit_03_cost_03_pricing_01_transient_failure_is_retried(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — one failure, then success
        calls = _patch_urlopen_flaky_then_succeed(
            monkeypatch, failures=1, html=_PAGE_HTML
        )

        # Act
        pricing = cc.fetch_fireworks_model_pricing(
            "accounts/fireworks/models/test-model"
        )

        # Assert
        assert pricing is not None
        assert pricing.uncached_input_per_1m == pytest.approx(4.15)
        assert pricing.cached_input_per_1m == pytest.approx(0.03)
        assert pricing.output_per_1m == pytest.approx(2.5)
        assert len(calls) == 2

    def test_04_unit_03_cost_03_pricing_02_exhausted_retries_return_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        calls = _patch_urlopen_always_fail(monkeypatch)

        # Act
        pricing = cc.fetch_fireworks_model_pricing(_UNKNOWN_MODEL)

        # Assert
        assert pricing is None
        assert len(calls) == cc._FETCH_ATTEMPTS

    def test_04_unit_03_cost_03_pricing_03_success_writes_persistent_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _isolated_persistent_cache: Path,
    ) -> None:
        # Arrange
        _patch_urlopen_succeed(monkeypatch, html=_PAGE_HTML)

        # Act
        cc.fetch_fireworks_model_pricing("accounts/fireworks/models/test-model")

        # Assert
        data = json.loads(_isolated_persistent_cache.read_text(encoding="utf-8"))
        entry = data["accounts/fireworks/models/test-model"]
        assert entry["uncached_input_per_1m"] == pytest.approx(4.15)
        assert entry["cached_input_per_1m"] == pytest.approx(0.03)
        assert entry["output_per_1m"] == pytest.approx(2.5)

    def test_04_unit_03_cost_03_pricing_04_persistent_cache_used_when_fetch_fails(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _isolated_persistent_cache: Path,
    ) -> None:
        # Arrange — a previously persisted entry for the unknown model
        _isolated_persistent_cache.write_text(
            json.dumps(
                {
                    _UNKNOWN_MODEL: {
                        "uncached_input_per_1m": 1.25,
                        "cached_input_per_1m": 0.25,
                        "output_per_1m": 2.5,
                    }
                }
            ),
            encoding="utf-8",
        )
        _patch_urlopen_always_fail(monkeypatch)

        # Act
        pricing = cc.fetch_fireworks_model_pricing(_UNKNOWN_MODEL)

        # Assert — values come from the file, not the network
        assert pricing is not None
        assert pricing.uncached_input_per_1m == pytest.approx(1.25)
        assert pricing.cached_input_per_1m == pytest.approx(0.25)
        assert pricing.output_per_1m == pytest.approx(2.5)

    def test_04_unit_03_cost_03_pricing_05_corrupt_cache_is_ignored(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _isolated_persistent_cache: Path,
    ) -> None:
        # Arrange — corrupt file, network down, model in no catalog
        _isolated_persistent_cache.write_text("{not-json", encoding="utf-8")
        _patch_urlopen_always_fail(monkeypatch)

        # Act / Assert — must not raise
        assert cc.fetch_fireworks_model_pricing(_UNKNOWN_MODEL) is None

    def test_04_unit_03_cost_03_pricing_06_catalog_fallback_logs_error(
        self, monkeypatch: pytest.MonkeyPatch, capture_sink: list[dict[str, Any]]
    ) -> None:
        # Arrange — network down, no persistent cache entry
        _patch_urlopen_always_fail(monkeypatch)

        # Act
        pricing = cc.fetch_fireworks_model_pricing(_CATALOG_MODEL)

        # Assert — catalog prices served
        assert pricing is not None
        assert pricing.uncached_input_per_1m == pytest.approx(0.15)
        assert pricing.cached_input_per_1m == pytest.approx(0.03)
        assert pricing.output_per_1m == pytest.approx(0.5)
        errors = [
            record
            for record in capture_sink
            if record.get("extra", {}).get("event")
            == "fireworks_pricing_catalog_fallback"
        ]
        assert errors, "expected an error log when falling back to the catalog"
        assert errors[0]["extra"]["model_id"] == _CATALOG_MODEL

    def test_04_unit_03_cost_03_pricing_07_unresolvable_model_warns(
        self, monkeypatch: pytest.MonkeyPatch, capture_sink: list[dict[str, Any]]
    ) -> None:
        # Arrange
        _patch_urlopen_always_fail(monkeypatch)

        # Act
        assert cc.fetch_fireworks_model_pricing(_UNKNOWN_MODEL) is None

        # Assert — structured warning recorded
        warnings = [
            record
            for record in capture_sink
            if record.get("extra", {}).get("event") == "fireworks_pricing_unresolved"
        ]
        assert warnings
        assert warnings[0]["extra"]["model_id"] == _UNKNOWN_MODEL

    def test_04_unit_03_cost_03_pricing_08_failure_ttl_suppresses_immediate_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_clock: dict[str, float],
    ) -> None:
        # Arrange
        calls = _patch_urlopen_always_fail(monkeypatch)

        # Act — resolve twice in quick succession
        assert cc.fetch_fireworks_model_pricing(_UNKNOWN_MODEL) is None
        attempts_after_first = len(calls)
        assert cc.fetch_fireworks_model_pricing(_UNKNOWN_MODEL) is None

        # Assert — second resolution was served from the failure cache
        assert len(calls) == attempts_after_first

    def test_04_unit_03_cost_03_pricing_09_failure_ttl_retries_after_expiry(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_clock: dict[str, float],
    ) -> None:
        # Arrange
        calls = _patch_urlopen_always_fail(monkeypatch)
        assert cc.fetch_fireworks_model_pricing(_UNKNOWN_MODEL) is None

        # Act — move past the failure TTL, then resolve again
        fake_clock["now"] += cc._FAILURE_TTL_SECONDS + 1.0
        assert cc.fetch_fireworks_model_pricing(_UNKNOWN_MODEL) is None

        # Assert — the fetch was retried after the TTL expired
        assert len(calls) == 2 * cc._FETCH_ATTEMPTS
