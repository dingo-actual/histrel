#!/usr/bin/env python3
"""Unit tests for user-message template rendering.

Tests that the per-iteration user-message render functions project the
instance problem, tools, observations, reasoning state, and iteration
number into the rendered message.

    Test Classes:
        Test04Unit04RenderUser01Init: First-iteration user-message rendering.
        Test04Unit04RenderUser02Step: Later-iteration user-message rendering.
    Test Subtypes:
        Morphism, Boundary
"""

from __future__ import annotations

from typing import Any

from histrel.model.render import (
    render_user_message_init,
    render_user_message_step,
)
from histrel.tools.repository import ReadFileTool
from histrel.tools.tool import (
    Tool,
    ToolContext,
    ToolExecution,
)


class NoParamTool(Tool):
    """A parameterless tool used to cover the empty-parameters branch."""

    name = "noop"
    definition: dict[str, Any] = {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "Do nothing.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }

    examples: list[str] = []

    def execute(self, arguments: dict[str, Any], context: ToolContext) -> ToolExecution:
        """Return an empty successful execution."""
        return ToolExecution(status="ok", output="")


class Test04Unit04RenderUser01Init:
    """Tests for rendering the first-iteration user message."""

    def test_04_unit_04_render_user_01_init_01_morphism_01_projects_problem_and_tool(
        self,
    ) -> None:
        """The init message contains the problem."""
        message = render_user_message_init(
            instance_problem="Fix the parser regression in src/.",
            available_tools=(ReadFileTool(),),
        )

        assert "Fix the parser regression in src/." in message
        assert "# Inputs" in message


class Test04Unit04RenderUser02Step:
    """Tests for rendering later-iteration user messages."""

    def test_04_unit_04_render_user_02_step_01_morphism_01_projects_all_sections(
        self,
    ) -> None:
        """The step message contains every projected section."""
        message = render_user_message_step(
            instance_problem="Fix the parser regression in src/.",
            observations=["[Iteration 1] Contents of file src/main.py"],
            reasoning_state="Reading the failing test to locate the bug.",
            available_tools=[ReadFileTool()],
            iter_crnt=2,
        )

        assert "Fix the parser regression in src/." in message
        assert "Current iteration: 2" in message
        assert "[Iteration 1] Contents of file src/main.py" in message
        assert "Reading the failing test to locate the bug." in message
        assert "Name: read" in message

    def test_04_unit_04_render_user_02_step_02_boundary_01_tool_without_parameters(
        self,
    ) -> None:
        """A tool without parameters renders ``- None`` under Parameters."""
        message = render_user_message_step(
            instance_problem="Fix the parser regression in src/.",
            observations=[],
            reasoning_state="Reading the failing test to locate the bug.",
            available_tools=[NoParamTool()],
            iter_crnt=2,
        )

        assert "Name: noop" in message
        assert "- None" in message
        assert "path (string" not in message
