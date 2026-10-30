#!/usr/bin/env python3
"""Unit tests for tool-call observation headers.

Tests that a call renders as ``tool_name(key=value, ...)`` with JSON
values, so the agent sees the arguments it passed.

    Test Classes:
        Test04Unit05ModelToolCallLines: Header rendering.
    Test Subtypes:
        Morphism, Boundary
"""

from __future__ import annotations

from histrel.model.tool_call_lines import format_tool_call


class Test04Unit05ModelToolCallLines:
    """format_tool_call: one call rendered as a header line."""

    def test_04_unit_05_model_tool_call_lines_01_morphism_01_renders_named_arguments(
        self,
    ) -> None:
        # Act
        header = format_tool_call("read", {"path": "src/main.py", "limit": 20})

        # Assert
        assert header == 'read(path="src/main.py", limit=20)'

    def test_04_unit_05_model_tool_call_lines_02_boundary_01_renders_no_arguments(
        self,
    ) -> None:
        # Act
        header = format_tool_call("list_dir", None)

        # Assert
        assert header == "list_dir()"

    def test_04_unit_05_model_tool_call_lines_03_boundary_02_renders_nested_values(
        self,
    ) -> None:
        # Act
        header = format_tool_call("edit", {"file_path": "main.py", "file_edits": ["x"]})

        # Assert
        assert header == 'edit(file_path="main.py", file_edits=["x"])'

    def test_04_unit_05_model_tool_call_lines_04_boundary_03_renders_unknown_tool(
        self,
    ) -> None:
        # Act — a call whose tool could not be resolved still names a
        # header, so the failure output below it is attributable.
        header = format_tool_call(None, None)

        # Assert
        assert header == "unknown()"
