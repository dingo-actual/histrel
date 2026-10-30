#!/usr/bin/env python3
"""Architecture guard: the core package must not import the benchmark harness.

Histrel owns the runtime; the benchmark harness depends on Histrel, never the
other way round. Installed imports, relative imports, and dynamic imports that
name the harness are all rejected.
"""

from __future__ import annotations

import ast
from pathlib import Path

_SRC = Path(__file__).resolve().parents[2] / "src" / "histrel"
_BENCHMARK_ROOT = "benchmark"


def _iter_src_modules() -> list[Path]:
    """Collect every Python module in the Histrel source tree.

    Returns:
        (list[Path]) Module paths, sorted for deterministic failures.
    """
    return sorted(_SRC.rglob("*.py"))


def _imports_benchmark(tree: ast.AST) -> list[str]:
    """Find imports of the benchmark package inside one module AST.

    Args:
        tree (ast.AST): Parsed module.

    Returns:
        (list[str]) Human-readable descriptions of offending imports.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] == _BENCHMARK_ROOT:
                    found.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".", 1)[0] == _BENCHMARK_ROOT:
                found.append(f"line {node.lineno}: from {module} import ...")
        elif isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name not in {"import_module", "__import__"}:
                continue
            for arg in node.args:
                if (
                    isinstance(arg, ast.Constant)
                    and isinstance(arg.value, str)
                    and arg.value.split(".", 1)[0] == _BENCHMARK_ROOT
                ):
                    found.append(f"line {node.lineno}: {name}({arg.value!r})")
    return found


def test_04_unit_00_multi_07_histrel_imports_no_benchmark() -> None:
    """No Histrel module may import the benchmark harness."""
    # Arrange
    modules = _iter_src_modules()
    assert modules, f"no modules found under {_SRC}"

    # Act
    offenders = {
        str(path.relative_to(_SRC)): _imports_benchmark(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
        for path in modules
    }
    violations = {name: hits for name, hits in offenders.items() if hits}

    # Assert
    assert not violations, (
        f"Histrel must not import the benchmark harness: {violations}"
    )
