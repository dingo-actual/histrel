#!/usr/bin/env python3
"""
Shared schema definitions for benchmark metadata and configuration.

Provides Pydantic models for run metadata that coordinate between the
predictor (which writes metadata) and downstream consumers (which read it).

    Important Classes:
        RunMetadata(BaseModel): Metadata about a benchmark run for coordination.
        PredictionRecord(BaseModel): Single prediction entry format.
        RunSpec(BaseModel): Run specification for resumability.
        UsageRecord(BaseModel): Token usage tracking record.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel


class RunMetadata(BaseModel):
    """Metadata written by the predictor and read by downstream consumers.

    The predictor writes this file after generating predictions to provide
    context about what was run and where to find artifacts.

    Instance Attributes:
        run_name (str): Unique identifier for this benchmark run.
        dataset_id (str): HuggingFace dataset identifier.
        split (str): Dataset split used (train/test/dev).
        instance_ids (list[str]): List of instance IDs processed.
        predictions_file (str): Relative path to predictions JSONL file.
        timestamp (str): ISO 8601 timestamp when run started.
        output_dir (str): Absolute path to output directory.
    """

    run_name: str
    dataset_id: str
    split: str
    instance_ids: list[str]
    predictions_files: list[str]
    timestamp: str
    output_dir: str


class PredictionRecord(BaseModel):
    """Single prediction record format for SWE-bench harness.

    Instance Attributes:
        instance_id (str): SWE-bench instance identifier.
        model_name_or_path (str): Agent identifier.
        model_patch (str): Generated patch text.
    """

    instance_id: str
    model_name_or_path: str
    model_patch: str


class RunSpec(BaseModel):
    """Run specification for resumability.

    Persisted to disk to enable resuming interrupted prediction runs.

    Instance Attributes:
        config (dict[str, Any]): Serialized predictor configuration.
        instances (list[dict[str, Any]]): List of test instances to process.
    """

    config: dict[str, Any]
    instances: list[dict[str, Any]]


class UsageRecord(BaseModel):
    """Token usage tracking record.

    Instance Attributes:
        instance_id (str): SWE-bench instance identifier.
        execution_time (float): Execution time in seconds.
        tokens_input (int): Total input tokens used.
        tokens_output (int): Total output tokens generated.
        tokens_cache_read (int): Cache read tokens.
        tokens_cache_write (int): Cache write tokens.
        tokens_total (int): Total tokens used.
        cost_total (float): Total cost in USD (0.0 when the agent does not
            report cost).
        maximum_context (int): Maximum context size.
        raw_usage_data (dict | None): Complete agent usage.json content.
        mcp_tools (dict[str, dict[str, int]] | None): Per-tool MCP usage statistics.
        mcp_totals (dict[str, int] | None): Aggregated MCP usage totals.
    """

    instance_id: str
    execution_time: float
    tokens_input: int
    tokens_output: int
    tokens_cache_read: int
    tokens_cache_write: int
    tokens_total: int
    cost_total: float = 0.0
    maximum_context: int
    raw_usage_data: dict | None = None
    mcp_tools: dict[str, dict[str, int]] | None = None
    mcp_totals: dict[str, int] | None = None


def read_run_metadata(metadata_path: Path) -> RunMetadata:
    """Read run metadata from JSON file.

    Args:
        metadata_path (Path): Path to metadata file.

    Returns:
        (RunMetadata) Parsed metadata.

    Raises:
        FileNotFoundError: If metadata file doesn't exist.
        ValueError: If metadata file is malformed.
    """
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as fp:
        return RunMetadata.model_validate_json(fp.read())


def write_run_spec(output_dir: Path, run_name: str, spec: RunSpec) -> Path:
    """Write run specification to JSON file.

    Args:
        output_dir (Path): Output directory for spec file.
        run_name (str): Run identifier for file naming.
        spec (RunSpec): Specification to write.

    Returns:
        (Path) Path to written spec file.
    """
    spec_path = output_dir / f"{run_name}_spec.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)

    with spec_path.open("w", encoding="utf-8") as fp:
        fp.write(spec.model_dump_json(indent=2))

    return spec_path


def read_run_spec(spec_path: Path) -> RunSpec:
    """Read run specification from JSON file.

    Args:
        spec_path (Path): Path to spec file.

    Returns:
        (RunSpec) Parsed specification.

    Raises:
        FileNotFoundError: If spec file doesn't exist.
        ValueError: If spec file is malformed.
    """
    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    with spec_path.open("r", encoding="utf-8") as fp:
        return RunSpec.model_validate_json(fp.read())
