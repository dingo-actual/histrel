#!/usr/bin/env python3
"""Unit tests for search backing binary version verification."""

from __future__ import annotations

import subprocess

import pytest

from histrel.core.errors import ToolBackendError
from histrel.tools.backing import (
    MINIMUM_RIPGREP_VERSION,
    REQUIRED_FD_VERSION,
    parse_version,
    verify_search_backends,
)


class Test04Unit03Tools03Backing:
    """Search backing binary version verification."""

    @pytest.mark.parametrize(
        "version_str, expected",
        [
            ("ripgrep 14.1.0 (rev c33c7c6)", (14, 1, 0)),
            ("fd 10.3.0", (10, 3, 0)),
            ("ripgrep 14.0.0", (14, 0, 0)),
            ("ripgrep 15.2.1 (rev abc123)", (15, 2, 1)),
        ],
        ids=["rg_14_1_0", "fd_10_3_0", "rg_14_0_0", "rg_15_2_1"],
    )
    def test_04_unit_03_tools_03_backing_01_parse_version(
        self, version_str: str, expected: tuple[int, ...]
    ) -> None:
        # Act
        result = parse_version(version_str)

        # Assert
        assert result == expected

    def test_04_unit_03_tools_03_backing_02_unparseable_fails(self) -> None:
        # Act & Assert
        with pytest.raises(ToolBackendError, match="unparseable"):
            parse_version("no version here")

    def test_04_unit_03_tools_03_backing_03_constants(self) -> None:
        # Assert — verify the pinned version constants
        assert MINIMUM_RIPGREP_VERSION == (14, 0, 0)
        assert REQUIRED_FD_VERSION == (10, 3, 0)

    def test_04_unit_03_tools_03_backing_04_verify_succeeds_if_installed(
        self,
    ) -> None:
        # Act & Assert — verify_search_backends should succeed if rg and fd
        # are installed with the correct versions. If they are not installed,
        # this test should be skipped, not failed.
        try:
            rg_out = subprocess.run(
                ["rg", "--version"], capture_output=True, text=True, check=True
            ).stdout
            fd_out = subprocess.run(
                ["fd", "--version"], capture_output=True, text=True, check=True
            ).stdout
        except (FileNotFoundError, subprocess.CalledProcessError):
            pytest.skip("ripgrep or fd not installed")

        rg_ver = parse_version(rg_out)
        fd_ver = parse_version(fd_out)

        if rg_ver < MINIMUM_RIPGREP_VERSION or fd_ver != REQUIRED_FD_VERSION:
            pytest.skip(f"Installed rg={rg_ver}, fd={fd_ver} do not match requirements")

        # Act — should not raise
        verify_search_backends()
