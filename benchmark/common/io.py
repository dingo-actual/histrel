#!/usr/bin/env python3
"""
I/O utilities for the benchmark suite.

Includes helpers for reading and writing JSON/JSONL, path management, and
stable hashing for caching keys.

    Important Functions:
        to_posix_path(path: Path) -> str
        from_posix_path(path_str: str) -> Path
        write_run_metadata(output_dir: Path, run_name: str, dataset_id: str, split: str, instance_ids: list[str], predictions_files: list[str]) -> Path
        write_json(path: str | os.PathLike[str], data: Mapping[str, Any]) -> None
        write_jsonl(path: str | os.PathLike[str], records: Iterable[Mapping[str, Any]]) -> None
        append_jsonl(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None
        read_json(path: str | os.PathLike[str]) -> Any
        read_jsonl(path: str | os.PathLike[str]) -> list[Any]
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmark.common.schemas import RunMetadata


def to_posix_path(path: Path) -> str:
    """Convert a Path to a POSIX-style string for cross-platform storage.

    On Windows, converts paths like 'C:\\Users\\...' to '/c/Users/...'.
    On Unix-like systems, returns the path as-is.

    Args:
        path (Path): Path to convert.

    Returns:
        (str) POSIX-style path string suitable for cross-platform storage.
    """
    if sys.platform == "win32":
        parts = path.resolve().parts
        if len(parts) > 0 and re.match(r"^[A-Za-z]:$", parts[0]):
            drive_letter = parts[0][0].lower()
            return "/" + drive_letter + "/" + "/".join(parts[1:])
        return path.as_posix()
    return path.resolve().as_posix()


def from_posix_path(path_str: str) -> Path:
    """Convert a POSIX-style path string to a platform-appropriate Path.

    On Linux/WSL, converts paths like '/c/Users/...' to '/mnt/c/Users/...'.
    On Windows, converts paths like '/c/Users/...' to 'C:\\Users\\...'.
    Regular POSIX paths are returned as-is.

    Args:
        path_str (str): POSIX-style path string.

    Returns:
        (Path) Platform-appropriate Path object.
    """
    if sys.platform == "win32":
        match = re.match(r"^/([a-z])/(.*)$", path_str)
        if match:
            drive_letter = match.group(1).upper()
            rest = match.group(2)
            return Path(f"{drive_letter}:/" + rest)
        return Path(path_str)

    if sys.platform.startswith("linux"):
        match = re.match(r"^/([a-z])/(.*)$", path_str)
        if match:
            drive_letter = match.group(1)
            rest = match.group(2)
            wsl_path = Path(f"/mnt/{drive_letter}/{rest}")
            if wsl_path.exists():
                return wsl_path
            return Path(f"/{drive_letter}/{rest}")

    return Path(path_str)


def ensure_dir(path: str | os.PathLike[str]) -> None:
    """Ensure a directory exists (mkdir -p semantics).

    If a file exists at the path, it will be removed first to allow
    directory creation. This handles edge cases where a file was
    created instead of a directory in a previous run.
    """
    p = Path(path)
    if p.exists() and p.is_file():
        # Remove file if it exists where we need a directory
        p.unlink()
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: str | os.PathLike[str], data: Mapping[str, Any]) -> None:
    """Write JSON to disk with UTF-8 encoding and stable formatting."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(
    path: str | os.PathLike[str], records: Iterable[Mapping[str, Any]]
) -> None:
    """Write newline-delimited JSON (JSONL) to disk.

    Each record is serialized with sorted keys for reproducibility.
    Uses LF line endings for compatibility with Linux containers.
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8", newline="\n") as fp:
        for rec in records:
            fp.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")


def append_jsonl(path: str | os.PathLike[str], record: Mapping[str, Any]) -> None:
    """Append a single record to a JSONL file.

    Creates the file and parent directories if they don't exist.
    Serializes with sorted keys for reproducibility.
    Uses LF line endings for compatibility with Linux containers.

    Args:
        path (str | os.PathLike[str]): Path to JSONL file.
        record (Mapping[str, Any]): Record to append.
    """
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("a", encoding="utf-8", newline="\n") as fp:
        fp.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def read_json(path: str | os.PathLike[str]) -> Any:
    """Read JSON from disk."""
    p = Path(path)
    return json.loads(p.read_text(encoding="utf-8"))


def read_jsonl(path: str | os.PathLike[str]) -> list[Any]:
    """Read newline-delimited JSON (JSONL) and return a list of objects."""
    p = Path(path)
    rows: list[Any] = []
    with p.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def run_output_dir(base_output_dir: str, run_name: str, suite: str) -> Path:
    """Compute the output directory for a given run and suite.

    Example: benchmark/results/local-dev/lm-eval/
    """
    return Path(base_output_dir) / run_name / suite


def stable_hash(parts: Sequence[Any]) -> str:
    """Compute a stable SHA-256 hash for a sequence of values.

    Values should be JSON-serializable; for non-serializable values we fall
    back to their repr(). This function is intended for cache keys and artifact
    identity, not for cryptographic security.
    """
    normalized: list[Any] = []
    for v in parts:
        try:
            json.dumps(v)
            normalized.append(v)
        except TypeError:
            normalized.append({"repr": repr(v)})

    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest


def write_run_metadata(
    output_dir: Path,
    run_name: str,
    dataset_id: str,
    split: str,
    instance_ids: list[str],
    predictions_files: list[str],
) -> Path:
    """Assemble and write run metadata to JSON file.

    Creates a RunMetadata object with current timestamp and writes it to
    run_metadata.json in the output directory.

    Args:
        output_dir (Path): Output directory for metadata file.
        run_name (str): Unique identifier for this benchmark run.
        dataset_id (str): HuggingFace dataset identifier.
        split (str): Dataset split used (train/test/dev).
        instance_ids (list[str]): List of instance IDs processed.
        predictions_files (list[str]): List of prediction file names.

    Returns:
        (Path) Path to written metadata file.
    """
    metadata_path = output_dir / "run_metadata.json"

    metadata = RunMetadata(
        run_name=run_name,
        dataset_id=dataset_id,
        split=split,
        instance_ids=instance_ids,
        predictions_files=predictions_files,
        timestamp=datetime.now(UTC).isoformat(),
        output_dir=to_posix_path(output_dir),
    )

    write_json(metadata_path, metadata.model_dump())
    return metadata_path
