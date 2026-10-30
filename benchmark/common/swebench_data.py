#!/usr/bin/env python3
"""
Utilities to fetch SWE-bench instances from Hugging Face and write to JSONL.

Provides a single function:
- materialize_instances_jsonl(dataset_id, split, dest_path, limit=None) -> Path

Requires the 'datasets' package. If unavailable, raises a clear RuntimeError
with installation instructions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _write_jsonl(path: Path, items: Iterable[Mapping[str, object]]) -> None:
    import json

    _ensure_dir(path)
    with path.open("w", encoding="utf-8") as fp:
        for row in items:
            fp.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def materialize_instances_jsonl(
    dataset_id: str,
    split: str,
    dest_path: str | Path,
    *,
    limit: int | None = None,
) -> Path:
    """Download SWE-bench instances and write them to a JSONL file.

    Args:
        dataset_id: Hugging Face dataset id (e.g., 'princeton-nlp/SWE-bench_Lite').
        split: HF split to use (typically 'test').
        dest_path: Path to write the JSONL file.
        limit: Optional maximum number of instances to write.

    Returns:
        Path to the written JSONL file.

    Raises:
        RuntimeError if the 'datasets' package is not available.
    """
    try:
        from datasets import load_dataset  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The 'datasets' package is required to fetch SWE-bench instances. "
            "Install it with: pip install datasets"
        ) from exc

    ds = load_dataset(dataset_id, split=split)

    # Convert to a generator of Python dicts
    def _iter_rows():
        for count, row in enumerate(ds):  # row is a dict-like mapping
            yield dict(row)
            if limit is not None and count >= max(1, int(limit)):
                break

    path = Path(dest_path)
    _write_jsonl(path, _iter_rows())
    return path
