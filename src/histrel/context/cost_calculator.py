#!/usr/bin/env python3
"""
Cost calculation for the benchmark agent.

Computes USD costs from token ledger entries and provider pricing
data. When pricing is unavailable, costs are reported as zero and
the report is marked incomplete.

The resolution chain is a retried page fetch, a failure TTL, a
persistent on-disk cache, and a checked-in catalog.

    Important Classes:
        ModelPricing: Per-1M-token prices for a single model, in USD.
        CallCost: Cost for a single model or compaction call.
        CostReport: Aggregated cost report with per-call breakdowns.
        CostCalculator: Computes costs from token ledger entries.

    Important Functions:
        fetch_fireworks_model_pricing(model_id) -> ModelPricing | None:
            Resolve (and cache) pricing for a Fireworks model from the
            page fetch, the persistent cache, or the checked-in catalog.
        parse_fireworks_model_pricing(html) -> ModelPricing | None:
            Parse pricing from a model documentation page's HTML.
        persistent_cache_path() -> Path:
            On-disk pricing cache location (override via
            ``PRICING_CACHE_ENV``).
        clear_pricing_cache() -> None:
            Clear the process-level pricing cache.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from histrel.context.token_ledger import LedgerEntry, TokenLedger
from histrel.model.types import Usage

_FIREWORKS_MODEL_PAGE_BASE_URL = "https://app.fireworks.ai/models/fireworks"
_FETCH_TIMEOUT_SECONDS = 5.0

# One resolution retries the page fetch this many times, with a linear
# backoff between attempts.
_FETCH_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = 0.5

# A failed resolution is remembered only this long, so a transient outage
# can never silence pricing (and cost accounting) for a whole session.
_FAILURE_TTL_SECONDS = 60.0

# Environment variable overriding the persistent pricing cache location.
PRICING_CACHE_ENV = "HISTREL_FIREWORKS_PRICING_CACHE"

_CATALOG_PATH = Path(__file__).parent / "fireworks_pricing_catalog.json"

# Test seams: retry sleep and the failure-TTL clock.
_sleep = time.sleep
_now = time.monotonic

_SKU_UNCACHED_INPUT = "LLM input tokens (uncached)"
_SKU_CACHED_INPUT = "LLM input tokens (cached)"
_SKU_OUTPUT = "LLM output tokens"

_SKU_PRICE_RE = re.compile(
    r'"sku"\s*:\s*"(?P<sku>[^"]*)"[^}]*?'
    r'"units"\s*:\s*"(?P<units>\$n\d+|\d+)"'
    r'\s*,\s*"nanos"\s*:\s*(?P<nanos>\d+)'
)


@dataclass(frozen=True)
class ModelPricing:
    """Per-1M-token prices for a single model, in USD.

    Instance Attributes:
        uncached_input_per_1m (float): USD per 1M uncached input tokens.
        cached_input_per_1m (float): USD per 1M cached input tokens.
        output_per_1m (float): USD per 1M output tokens.
    """

    uncached_input_per_1m: float
    cached_input_per_1m: float
    output_per_1m: float


def fireworks_model_page_url(model_id: str) -> str:
    """Build the documentation page URL for a Fireworks model id.

    Args:
        model_id (str): Model identifier such as
            ``accounts/fireworks/models/deepseek-v4p1-flash-1017`` or a bare suffix.

    Returns:
        (str) The model page URL.
    """
    marker = "/models/"
    if marker in model_id:
        suffix = model_id.rsplit(marker, 1)[-1]
    else:
        suffix = model_id.rsplit("/", 1)[-1]
    return f"{_FIREWORKS_MODEL_PAGE_BASE_URL}/{suffix}"


def parse_fireworks_model_pricing(html: str) -> ModelPricing | None:
    """Parse per-1M-token prices from a Fireworks model page.

    Args:
        html (str): Raw HTML of a model documentation page.

    Returns:
        (ModelPricing | None) Parsed prices, or None when the page does
        not expose all three required SKUs.
    """
    start = html.find("skuInfos")
    if start < 0:
        return None
    end = html.find("]", start)
    if end < 0:
        return None
    segment = html[start:end].replace('\\"', '"')

    prices: dict[str, float] = {}
    for match in _SKU_PRICE_RE.finditer(segment):
        units = match.group("units")
        units_value = float(units.removeprefix("$n"))
        nanos = int(match.group("nanos"))
        prices[match.group("sku")] = units_value + nanos / 1e9

    uncached = prices.get(_SKU_UNCACHED_INPUT)
    cached = prices.get(_SKU_CACHED_INPUT)
    output = prices.get(_SKU_OUTPUT)
    if uncached is None or cached is None or output is None:
        return None
    return ModelPricing(
        uncached_input_per_1m=uncached,
        cached_input_per_1m=cached,
        output_per_1m=output,
    )


_cache_lock = threading.Lock()
# model_id -> (resolved pricing or None, monotonic time of resolution)
_pricing_cache: dict[str, tuple[ModelPricing | None, float]] = {}

_catalog_cache: dict[str, ModelPricing] | None = None


def _entry_to_pricing(entry: object) -> ModelPricing | None:
    """Build a ModelPricing from a decoded JSON cache or catalog entry.

    Args:
        entry (object): Decoded JSON value for one model.

    Returns:
        (ModelPricing | None) Prices, or None when the entry is malformed.
    """
    if not isinstance(entry, dict):
        return None
    try:
        return ModelPricing(
            uncached_input_per_1m=float(entry["uncached_input_per_1m"]),
            cached_input_per_1m=float(entry["cached_input_per_1m"]),
            output_per_1m=float(entry["output_per_1m"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def persistent_cache_path() -> Path:
    """Return the persistent pricing cache file location.

    The ``PRICING_CACHE_ENV`` environment variable overrides the default
    user cache path, which lets hosts and containers share a cache file.

    Returns:
        (Path) Cache file path.
    """
    env_path = os.environ.get(PRICING_CACHE_ENV)
    if env_path:
        return Path(env_path)
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "histrel" / "fireworks_pricing.json"


def _read_persistent_cache(model_id: str) -> ModelPricing | None:
    """Read one model's pricing from the persistent cache file.

    Args:
        model_id (str): Fireworks model identifier.

    Returns:
        (ModelPricing | None) Cached prices, or None when the file is
            missing, corrupt, or lacks the model.
    """
    try:
        data = json.loads(persistent_cache_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return _entry_to_pricing(data.get(model_id))


def _write_persistent_cache(model_id: str, pricing: ModelPricing) -> None:
    """Write one model's network-observed pricing to the cache file.

    Failures are ignored: the persistent cache is a durability
    optimization, never a prerequisite for pricing resolution.

    Args:
        model_id (str): Fireworks model identifier.
        pricing (ModelPricing): Prices observed from the model page.
    """
    path = persistent_cache_path()
    try:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            existing = {}
        data = existing if isinstance(existing, dict) else {}
        data[model_id] = {
            "uncached_input_per_1m": pricing.uncached_input_per_1m,
            "cached_input_per_1m": pricing.cached_input_per_1m,
            "output_per_1m": pricing.output_per_1m,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fp:
                json.dump(data, fp, indent=2)
            # Atomic replace so concurrent readers never see a torn file.
            Path(tmp_name).replace(path)
        except OSError:
            with contextlib.suppress(OSError):
                Path(tmp_name).unlink()
            raise
    except OSError:
        return None


def _load_catalog() -> dict[str, ModelPricing]:
    """Load the checked-in catalog of last-resort prices.

    Returns:
        (dict[str, ModelPricing]) Prices keyed by model id; empty when
            the catalog file is missing or corrupt.
    """
    global _catalog_cache
    if _catalog_cache is None:
        catalog: dict[str, ModelPricing] = {}
        try:
            data = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            data = {}
        if isinstance(data, dict):
            for model_id, entry in data.items():
                pricing = _entry_to_pricing(entry)
                if pricing is not None:
                    catalog[model_id] = pricing
        _catalog_cache = catalog
    return _catalog_cache


def _fetch_page_pricing(model_id: str) -> ModelPricing | None:
    """Fetch and parse the model page, retrying transient failures.

    Args:
        model_id (str): Fireworks model identifier.

    Returns:
        (ModelPricing | None) Parsed prices, or None when every attempt
            fails or the page does not expose the required SKUs.
    """
    for attempt in range(_FETCH_ATTEMPTS):
        try:
            request = urllib.request.Request(
                fireworks_model_page_url(model_id),
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(
                request, timeout=_FETCH_TIMEOUT_SECONDS
            ) as response:
                html = response.read().decode("utf-8", errors="replace")
            pricing = parse_fireworks_model_pricing(html)
            if pricing is not None:
                return pricing
        except Exception:
            # Transient failure: retried below; the last one falls through.
            pass
        if attempt + 1 < _FETCH_ATTEMPTS:
            _sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
    return None


def fetch_fireworks_model_pricing(model_id: str) -> ModelPricing | None:
    """Resolve pricing for a Fireworks model, caching per-process results.

    Resolution order: process cache, then the (retried) model page fetch
    with write-through to the persistent on-disk cache, then that
    persistent cache, then the checked-in catalog. Successful
    resolutions are cached for the process lifetime; failed ones are
    cached only for ``_FAILURE_TTL_SECONDS`` so a transient outage
    cannot zero a whole session's cost accounting. Total failure logs a
    structured warning; a catalog fallback logs a structured error
    because the checked-in prices may be stale.

    Args:
        model_id (str): Fireworks model identifier.

    Returns:
        (ModelPricing | None) Prices, or None when unavailable from any
            source.
    """
    key = model_id.strip()
    with _cache_lock:
        cached = _pricing_cache.get(key)
        if cached is not None:
            pricing, resolved_at = cached
            if pricing is not None or _now() - resolved_at < _FAILURE_TTL_SECONDS:
                return pricing

    pricing = _fetch_page_pricing(key)
    if pricing is not None:
        _write_persistent_cache(key, pricing)
    else:
        pricing = _read_persistent_cache(key)
    if pricing is None:
        catalog_pricing = _load_catalog().get(key)
        if catalog_pricing is not None:
            logger.bind(
                component="bench_agent.cost",
                event="fireworks_pricing_catalog_fallback",
            ).error(
                "Fireworks model pricing fell back to the checked-in catalog "
                "because the model page fetch and the persistent cache "
                "failed; catalog prices may be stale",
                model_id=key,
            )
            pricing = catalog_pricing

    with _cache_lock:
        _pricing_cache[key] = (pricing, _now())

    if pricing is None:
        logger.bind(
            component="bench_agent.cost",
            event="fireworks_pricing_unresolved",
        ).warning(
            "Fireworks model pricing could not be resolved from the model "
            "page, the persistent cache, or the local catalog; cost "
            "accounting for this model will be incomplete",
            model_id=key,
        )
    return pricing


def clear_pricing_cache() -> None:
    """Clear the process-level pricing cache.

    Tests use this to force resolution between runs; production callers
    normally rely on the per-process cache.
    """
    with _cache_lock:
        _pricing_cache.clear()


class CallCost(BaseModel):
    """Cost for a single model or compaction call.

    Instance Attributes:
        provider_request_id (str | None): Provider request ID.
        call_kind (str): Call type ("model", "compaction", or "routine").
        input_tokens (int): Input tokens.
        output_tokens (int): Output tokens.
        cached_tokens (int): Cached tokens.
        reasoning_tokens (int): Reasoning tokens.
        cost_usd (float): Cost in USD.
    """

    provider_request_id: str | None
    call_kind: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    cost_usd: float


class CostReport(BaseModel):
    """Aggregated cost report.

    Instance Attributes:
        total_cost (float): Total cost across all calls.
        model_cost (float): Cost from model and routine calls only.
        compaction_cost (float): Cost from compaction calls only.
        call_costs (list[CallCost]): Per-call cost breakdowns.
        is_complete (bool): Whether cost accounting is complete.
    """

    total_cost: float = 0.0
    model_cost: float = 0.0
    compaction_cost: float = 0.0
    call_costs: list[CallCost] = []
    is_complete: bool = True


def _usage_cost_usd(usage: Usage, pricing: ModelPricing) -> float:
    """Compute the USD cost of a usage record at the given pricing.

    Args:
        usage (Usage): Token usage from a model response.
        pricing (ModelPricing): Per-1M-token prices.

    Returns:
        (float) Cost in USD.
    """
    uncached_input = usage.input - usage.cached
    return (
        uncached_input * pricing.uncached_input_per_1m
        + usage.cached * pricing.cached_input_per_1m
        + usage.output * pricing.output_per_1m
    ) / 1_000_000


def _make_call_cost(entry: LedgerEntry, pricing: ModelPricing) -> CallCost:
    """Compute the cost row for a single ledger entry.

    Args:
        entry (LedgerEntry): Token ledger entry.
        pricing (ModelPricing): Per-1M-token prices.

    Returns:
        (CallCost) Per-call cost breakdown.
    """
    usage = entry.usage
    return CallCost(
        provider_request_id=entry.provider_request_id,
        call_kind=entry.call_kind,
        input_tokens=usage.input,
        output_tokens=usage.output,
        cached_tokens=usage.cached,
        reasoning_tokens=usage.reasoning,
        cost_usd=_usage_cost_usd(usage, pricing),
    )


class CostCalculator:
    """Computes costs from token ledger entries and provider pricing.

    Instance Attributes:
        _pricing (ModelPricing | None): Per-1M-token prices, or None.
    """

    def __init__(self, pricing: ModelPricing | None) -> None:
        """Initialize the cost calculator.

        Args:
            pricing (ModelPricing | None): Parsed provider pricing, or
                None when pricing is unavailable.
        """
        self._pricing = pricing

    def compute_call_cost(self, usage: Usage) -> CallCost | None:
        """Compute the cost for a single usage record.

        Args:
            usage (Usage): Token usage from a model response.

        Returns:
            (CallCost | None) Call cost, or None when pricing is
            unavailable.
        """
        if self._pricing is None:
            return None

        return CallCost(
            provider_request_id=None,
            call_kind="model",
            input_tokens=usage.input,
            output_tokens=usage.output,
            cached_tokens=usage.cached,
            reasoning_tokens=usage.reasoning,
            cost_usd=_usage_cost_usd(usage, self._pricing),
        )

    def compute_report(self, ledger: TokenLedger) -> CostReport:
        """Compute a full cost report from the token ledger.

        Args:
            ledger (TokenLedger): Token ledger with recorded usage.

        Returns:
            (CostReport) Aggregated cost report.
        """
        if self._pricing is None:
            return CostReport(
                total_cost=0.0,
                model_cost=0.0,
                compaction_cost=0.0,
                call_costs=[],
                is_complete=False,
            )

        pricing = self._pricing
        call_costs = [_make_call_cost(entry, pricing) for entry in ledger.entries]

        # Model and routine calls cost into model_cost; compaction
        # calls are tracked separately.
        model_cost = sum(c.cost_usd for c in call_costs if c.call_kind != "compaction")
        compaction_cost = sum(
            c.cost_usd for c in call_costs if c.call_kind == "compaction"
        )

        return CostReport(
            total_cost=model_cost + compaction_cost,
            model_cost=model_cost,
            compaction_cost=compaction_cost,
            call_costs=call_costs,
            is_complete=True,
        )
