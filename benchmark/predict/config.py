#!/usr/bin/env python3
"""
Predictor configuration models.

Pydantic v2 models for configuring the prediction generation phase
of the benchmark suite.

    Important Classes:
        HistrelConfig(BaseModel): Agent model and token-budget settings.
        SWEbenchConfig(BaseModel): SWE-bench specific settings for prediction.
        PredictorConfig(BaseModel): Top-level predictor configuration.

    Important Functions:
        load_predictor_config(path: str | None) -> PredictorConfig
"""

from __future__ import annotations

import json
import secrets
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Mode = Literal["stateless", "react"]

#: Model used when a settings file names none.
DEFAULT_MODEL = "accounts/fireworks/models/glm-5p3-flash"


class HistrelConfig(BaseModel):
    """Agent model and token-budget settings passed to the container.

    Instance Attributes:
        models (str | dict[str, str]): Model id, or a table of template
            name to model id with an optional "default" entry.
        token_budget (int): Total prompt plus completion tokens for one
            instance run; 0 means unlimited.
        token_budget_buffer (int): Portion of ``token_budget`` reserved
            for finalization; 0 reserves nothing.
    """

    model_config = ConfigDict(extra="ignore")

    models: str | dict[str, str] = DEFAULT_MODEL
    token_budget: int = Field(default=0, ge=0)
    token_budget_buffer: int = Field(default=0, ge=0)

    def model_id(self) -> str:
        """Return the model id of the default template entry.

        Returns:
            (str) Model id used for the agent.
        """
        if isinstance(self.models, dict):
            return str(self.models.get("default", DEFAULT_MODEL))
        return str(self.models)


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

    histrel: HistrelConfig = Field(default_factory=HistrelConfig)
    swe_bench: SWEbenchConfig = Field(default_factory=SWEbenchConfig)

    def render_bench_settings(self) -> str:
        """Render the resolved settings in ``bench_settings.toml`` format.

        The container reads this file to pick the agent model and token
        budgets, so it keeps the ``[histrel]`` and ``[benchmark]`` layout
        of the harness settings file.

        Returns:
            (str) TOML text of the resolved settings.
        """
        lines = [
            "# Resolved benchmark configuration.",
            "# Written by the predictor and mounted into each instance container.",
            "",
            "[histrel]",
            f"models = {json.dumps(self.histrel.model_id())}",
            f"token_budget = {self.histrel.token_budget}",
            f"token_budget_buffer = {self.histrel.token_budget_buffer}",
            "",
            "[benchmark]",
            f"output_dir = {json.dumps(self.output_dir)}",
            f"run_name = {json.dumps(self.run_name)}",
            f"mode = {json.dumps(self.mode)}",
            f"seed = {self.seed}",
            f"allow_network = {_toml_bool(self.allow_network)}",
            f"allow_shell_observation = {_toml_bool(self.allow_shell_observation)}",
            f"rebuild = {_toml_bool(self.rebuild)}",
            f"show_container_streams = {_toml_bool(self.show_container_streams)}",
            "",
            "[benchmark.swe_bench]",
            f"enabled = {_toml_bool(self.swe_bench.enabled)}",
            f"split = {json.dumps(self.swe_bench.split)}",
            f"dataset_id = {json.dumps(self.swe_bench.dataset_id)}",
            f"max_instances = {self.swe_bench.max_instances}",
            f"timeout_seconds = {self.swe_bench.timeout_seconds}",
            f"docker = {_toml_bool(self.swe_bench.docker)}",
            f"docker_base_image = {json.dumps(self.swe_bench.docker_base_image)}",
            "tools = ["
            + ", ".join(json.dumps(tool) for tool in self.swe_bench.tools)
            + "]",
            "",
            "[benchmark.verbose]",
            f"enabled = {_toml_bool(self.verbose)}",
        ]
        return "\n".join(lines) + "\n"

    def write_bench_settings(self, directory: Path) -> Path:
        """Write the resolved settings into a run directory.

        Args:
            directory (Path): Directory that receives the settings file.

        Returns:
            (Path) Path to the written ``bench_settings.toml``.
        """
        path = Path(directory) / "bench_settings.toml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render_bench_settings(), encoding="utf-8")
        return path


def _toml_bool(value: bool) -> str:
    """Render a Python bool as a TOML bool.

    Args:
        value (bool): Value to render.

    Returns:
        (str) ``true`` or ``false``.
    """
    return "true" if value else "false"


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
    """Extract and normalize the benchmark sections from a TOML document."""
    section = doc.get("benchmark", {})
    if not isinstance(section, dict):
        return {}

    histrel_tbl = doc.get("histrel", {})
    swe_bench = section.get("swe_bench", {})
    verbose_tbl = section.get("verbose", {})

    result: dict[str, Any] = {
        key: val for key, val in section.items() if key not in {"swe_bench", "verbose"}
    }
    if isinstance(histrel_tbl, dict):
        result["histrel"] = histrel_tbl
    if isinstance(swe_bench, dict):
        result["swe_bench"] = swe_bench

    if isinstance(verbose_tbl, dict):
        enabled = verbose_tbl.get("enabled", False)
        result["verbose"] = enabled

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
