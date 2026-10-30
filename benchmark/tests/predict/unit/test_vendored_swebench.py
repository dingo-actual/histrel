#!/usr/bin/env python3
"""
Unit tests for vendored SWE-bench import preference.

The predict harness must import SWE-bench from the vendored snapshot in
``benchmark/vendor/SWE-bench`` so that builds use the pinned conda
lockfiles under ``swebench/resources/swebench-og``. The PyPI wheel is
built without those resources, which previously forced matplotlib-family
environment builds onto unpinned dev ``environment.yml`` files whose conda
solve hangs or exhausts memory.

    Test Classes:
        Test04Unit07Vendored01Resolution: swebench package resolution tests
        Test04Unit07Vendored02Lockfiles: pinned lockfile availability tests
        Test04Unit07Vendored03EnvScript: generated env script content tests
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

# Import the harness package FIRST so its vendored-SWE-bench path shim runs
# before any direct swebench imports in this module (import order is the
# behavior under test here).
import benchmark.predict.swebench  # isort: skip  # noqa: F401

import swebench
from swebench.harness.utils import load_cached_environment_yml

from benchmark.predict.swebench.testspec_adapter import TestSpecAdapter

_VENDOR_ROOT = Path(__file__).resolve().parents[3] / "vendor" / "SWE-bench" / "swebench"
_VENDOR_AVAILABLE = (_VENDOR_ROOT / "__init__.py").exists()

pytestmark = pytest.mark.skipif(
    not _VENDOR_AVAILABLE, reason="vendored SWE-bench clone not present"
)

# Real identifiers for the instance whose env build previously hung: the
# unpinned-dev-yml conda solve for matplotlib 3.5 never terminates.
_MPL_22719_INSTANCE: dict = {
    "instance_id": "matplotlib__matplotlib-22719",
    "repo": "matplotlib/matplotlib",
    "version": "3.5",
    "base_commit": "a2a1b0a11b993fe5f8fab64b6161e99243a6393c",
    "environment_setup_commit": "de98877e3dc45de8dd441d008f23d88738dc015d",
    "patch": (
        "diff --git a/lib/matplotlib/dummy.py b/lib/matplotlib/dummy.py\n"
        "--- a/lib/matplotlib/dummy.py\n"
        "+++ b/lib/matplotlib/dummy.py\n"
        "@@ -1,2 +1,3 @@\n"
        " existing line one\n"
        " existing line two\n"
        "+placeholder line\n"
    ),
    "test_patch": (
        "diff --git a/lib/tests/test_dummy.py b/lib/tests/test_dummy.py\n"
        "--- a/lib/tests/test_dummy.py\n"
        "+++ b/lib/tests/test_dummy.py\n"
        "@@ -1,2 +1,3 @@\n"
        " existing test line one\n"
        " existing test line two\n"
        "+placeholder test line\n"
    ),
    "FAIL_TO_PASS": ["lib.tests.test_dummy::test_placeholder"],
    "PASS_TO_PASS": [],
}


class Test04Unit07Vendored01Resolution:
    """Tests that swebench resolves to the vendored snapshot."""

    def test_04_unit_07_vendored_01_resolution_01_imports_from_vendor(
        self,
    ) -> None:
        """swebench package resolves inside benchmark/vendor/SWE-bench."""

        assert "vendor/SWE-bench" in str(Path(swebench.__file__))


class Test04Unit07Vendored02Lockfiles:
    """Tests for pinned conda lockfile availability from the vendor."""

    def test_04_unit_07_vendored_02_lockfiles_01_matplotlib_22719_cached(
        self,
    ) -> None:
        """matplotlib-22719 has a pinned environment.yml lockfile."""

        yml = load_cached_environment_yml("matplotlib__matplotlib-22719")

        assert yml is not None
        assert "_libgcc_mutex=" in yml
        assert "channels:" in yml


class Test04Unit07Vendored03EnvScript:
    """Tests that generated env scripts use pinned lockfiles."""

    def test_04_unit_07_vendored_03_env_script_01_uses_pinned_yml(self) -> None:
        """matplotlib-22719 setup_env embeds the pinned lockfile.

        The previously broken path fetched the unpinned dev yml and solved
        it from scratch (hang/OOM). The pinned path embeds exact
        package=version=build pins.
        """

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = TestSpecAdapter.from_instance(
                _MPL_22719_INSTANCE,
                histrel_repo_path=root,
            )

        script = spec.setup_env_script

        assert "conda env create -f /root/environment.yml" in script
        assert "_libgcc_mutex=" in script

    def test_04_unit_07_vendored_03_env_script_02_no_unpinned_solve(
        self,
    ) -> None:
        """matplotlib-22719 setup_env does not solve an unpinned yml.

        The unpinned fallback runs ``conda install python=...`` after
        ``conda env create``; the pinned lockfile path never does.
        """

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            spec = TestSpecAdapter.from_instance(
                _MPL_22719_INSTANCE,
                histrel_repo_path=root,
            )

        assert "conda install python=" not in spec.setup_env_script
