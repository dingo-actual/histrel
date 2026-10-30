#!/usr/bin/env python3
"""
Predictor configuration models.

Pydantic v2 models for configuring the prediction generation phase
of the benchmark suite.

    Important Classes:
        SWEbenchConfig(BaseModel): SWE-bench specific settings for prediction.
        PredictorConfig(BaseModel): Top-level predictor configuration.

    Important Functions:
        load_predictor_config(path: str | None) -> PredictorConfig
"""

from __future__ import annotations

import secrets
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Mode = Literal["stateless", "react"]


class SWEbenchConfig(BaseModel):
    """Configuration for SWE-bench prediction generation."""

    enabled: bool = True
    split: str = "test"
    dataset_id: str = "princeton-nlp/SWE-bench_Lite"
    max_instances: int = Field(default=25, ge=1)
    timeout_seconds: int = Field(default=7200, ge=1)
    docker: bool = True
    docker_base_image: str = "python:3.11-slim"
    tools: list[str] = Field(default_factory=lambda: ["ripgrep", "fd", "tokei"])

    @field_validator("tools")
    @classmethod
    def _validate_tools(cls, v: list[str]) -> list[str]:
        if not isinstance(v, list):
            raise ValueError("tools must be a list of strings")
        if not all(isinstance(tool, str) for tool in v):
            raise ValueError("all tools must be strings")
        return v


class CancellationConfig(BaseModel):
    """Configuration for HTTP cancellation server."""

    enabled: bool = Field(default=True, description="Enable HTTP cancellation")
    host: str = Field(default="127.0.0.1", description="Bind host for server")
    port: int = Field(default=8765, ge=1024, le=65535, description="Server port")


class PredictorConfig(BaseModel):
    """Top-level predictor configuration."""

    model_config = ConfigDict(extra="ignore")

    output_dir: str = "benchmark/results"
    run_name: str = Field(
        default_factory=lambda: f"run{secrets.token_hex(4)}",
        description="Unique identifier for this benchmark run",
    )
    mode: Mode = "stateless"
    seed: int = 12321
    allow_network: bool = True
    allow_shell_observation: bool = True
    verbose: bool = False
    rebuild: bool = False
    show_container_streams: bool = False

    swe_bench: SWEbenchConfig = Field(default_factory=SWEbenchConfig)
    cancellation: CancellationConfig = Field(default_factory=CancellationConfig)


def _read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file and return its dictionary representation."""
    try:
        import tomllib
    except Exception as exc:
        raise RuntimeError(
            "Python 3.11+ is required to load TOML (tomllib not available)."
        ) from exc

    with path.open("rb") as fp:
        return tomllib.load(fp)


def _coerce_benchmark_dict(doc: dict[str, Any]) -> dict[str, Any]:
    """Extract and normalize the [benchmark] section from a TOML document."""
    section = doc.get("benchmark", {})
    if not isinstance(section, dict):
        return {}

    swe_bench = section.get("swe_bench", {})
    verbose_tbl = section.get("verbose", {})
    cancellation_tbl = section.get("cancellation", {})

    result: dict[str, Any] = {
        key: val
        for key, val in section.items()
        if key not in {"swe_bench", "verbose", "cancellation"}
    }
    if isinstance(swe_bench, dict):
        result["swe_bench"] = swe_bench

    if isinstance(verbose_tbl, dict):
        enabled = verbose_tbl.get("enabled", False)
        result["verbose"] = enabled

    if isinstance(cancellation_tbl, dict):
        result["cancellation"] = cancellation_tbl

    return result


def load_predictor_config(path: str | None = None) -> PredictorConfig:
    """Load PredictorConfig from TOML file.

    Behavior:
    - If `path` is provided and exists, load that file (authoritative).
    - Else, search for bench_settings.toml in the following order:
      1) <this package>/bench_settings.toml (benchmark-local; highest priority)
      2) ./bench_settings.toml
      3) ~/.config/histrel/bench_settings.toml
      4) ~/.histrel/bench_settings.toml
      Use the first file that exists and contains a [benchmark] section.
    - Otherwise, return defaults.

    Args:
        path (str | None): Explicit config file path. Defaults to None.

    Returns:
        (PredictorConfig) Loaded configuration.

    Raises:
        FileNotFoundError: If explicit path is provided but doesn't exist.
    """
    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"Config file not found: {candidate}")
        raw = _read_toml(candidate)
        data = _coerce_benchmark_dict(raw)
        return PredictorConfig.model_validate(data)

    bench_local = Path(__file__).resolve().parent.parent / "bench_settings.toml"
    candidates = [
        bench_local,
        Path.cwd() / "bench_settings.toml",
        Path.home() / ".config" / "histrel" / "bench_settings.toml",
        Path.home() / ".histrel" / "bench_settings.toml",
    ]
    for p in candidates:
        try:
            if not p.exists():
                continue
            raw = _read_toml(p)
            data = _coerce_benchmark_dict(raw)
            if data:
                return PredictorConfig.model_validate(data)
        except Exception:
            continue

    return PredictorConfig()
