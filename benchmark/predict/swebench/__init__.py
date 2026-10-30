#!/usr/bin/env python3
"""SWE-bench integration module for benchmark predict refactor.

Provides adapters and utilities for integrating SWE-bench's three-layer
image architecture (base → env → instance) with the Histrel benchmark.
Importing this package first also makes the vendored SWE-bench snapshot in
``benchmark/vendor/SWE-bench`` the import source for ``swebench`` (see
``_prefer_vendored_swebench``).

    Important Classes:
        TestSpecAdapter: Extended TestSpec with Histrel requirements
        EnvBuildContextManager: Stages files into SWE-bench build contexts
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
from pathlib import Path

from loguru import logger


def _warn_on_version_mismatch(vendor_root: Path) -> None:
    """Warn when the vendored SWE-bench version differs from the installed one.

    Args:
        vendor_root: Root of the vendored SWE-bench repository
    """
    try:
        installed = importlib.metadata.version("swebench")
        init_text = (vendor_root / "swebench" / "__init__.py").read_text(
            encoding="utf-8"
        )
        match = re.search(r'__version__\s*=\s*"([^"]+)"', init_text)
    except (importlib.metadata.PackageNotFoundError, OSError):
        return
    if match is not None and match.group(1) != installed:
        logger.bind(
            component="benchmark.predict.swebench",
            origin="bench_harness_predict",
        ).warning(
            "Vendored SWE-bench version differs from installed package",
            vendored_version=match.group(1),
            installed_version=installed,
        )


def _prefer_vendored_swebench() -> None:
    """Make the vendored SWE-bench snapshot the import source for ``swebench``.

    The PyPI wheel ships without the pinned conda lockfiles under
    ``swebench/resources/swebench-og``, which downgrades environment builds
    to unpinned dev ``environment.yml`` files whose conda solve can hang or
    exhaust memory (matplotlib 3.5). Prepending the vendored clone restores
    the lockfiles and pins the harness to the exact vendored version.

    ``swebench`` modules already imported from elsewhere are dropped from
    ``sys.modules``: the ``sys.path`` entry alone cannot redirect an already
    imported package because submodules resolve via the package
    ``__path__`` set on first import.

    No-op when the vendored clone is not present.
    """
    vendor_root = Path(__file__).resolve().parents[2] / "vendor" / "SWE-bench"
    if not (vendor_root / "swebench" / "__init__.py").exists():
        return

    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))

    swebench_module = sys.modules.get("swebench")
    if swebench_module is not None and "vendor/SWE-bench" not in str(
        swebench_module.__file__
    ):
        for module_name in [
            name
            for name in sys.modules
            if name == "swebench" or name.startswith("swebench.")
        ]:
            del sys.modules[module_name]

    _warn_on_version_mismatch(vendor_root)


_prefer_vendored_swebench()
