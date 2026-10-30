#!/usr/bin/env python3
"""
Tool registry for the benchmark agent.

Consolidates tool construction into one registry keyed by tool name.

    Important Classes:
        ToolRegistry: Holds all tools for a session.

    Important Functions:
        build_all_tools() -> ToolRegistry: Build the full tool set.
"""

from __future__ import annotations

from histrel.tools.repository import (
    BashTool,
    EditFileTool,
    GlobTool,
    ListDirTool,
    ReadFileTool,
    SearchTool,
)
from histrel.tools.tool import Tool, build_tool_examples_section


def _build_ordinary_tools() -> list[Tool]:
    """Build the ordinary repository tools.

    Returns:
        (list[Tool]) Ordinary tool instances.
    """
    return [
        ReadFileTool(),
        EditFileTool(),
        SearchTool(),
        ListDirTool(),
        GlobTool(),
        BashTool(),
    ]


def build_all_tools() -> ToolRegistry:
    """Build the full tool set.

    Returns:
        (ToolRegistry) Complete tool registry.
    """
    return ToolRegistry(
        all_tools={tool.name: tool for tool in _build_ordinary_tools()},
    )


class ToolRegistry:
    """Holds all tools for a session.

    Instance Attributes:
        all_tools (dict[str, Tool]): All tools keyed by name.
    """

    def __init__(self, *, all_tools: dict[str, Tool]) -> None:
        """Initialize the registry.

        Args:
            all_tools (dict[str, Tool]): All tools keyed by name.
        """
        self.all_tools = all_tools

    def system_prompt_tool_section(self) -> str:
        """Build the tool examples section for the system prompt.

        Returns:
            (str) Formatted section listing all tools with examples.
        """
        return build_tool_examples_section(list(self.all_tools.values()))
