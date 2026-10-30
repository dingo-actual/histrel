#!/usr/bin/env python3
"""Unit tests for bench-agent system prompt selection by mode.

Tests get_bench_agent_sys_prompts mode handling: stateless (default)
selects the initialize/step prompt pair, react uses the single react
template for both, and invalid modes are rejected.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmark.predict.agents.prompts import get_bench_agent_sys_prompts


class TestUnitPrompts01ModeSelection:
    """get_bench_agent_sys_prompts mode selection."""

    def test_unit_prompts_01_mode_01_default_is_stateless_behavior(self) -> None:
        # Act — no mode argument
        prompts = get_bench_agent_sys_prompts(Path("/testbed"))

        # Assert — initialize/step pair, distinct prompts
        assert "Code Reasoning Initialization System" in prompts["initialize"]
        assert "Single-Step Code Reasoning System" in prompts["step"]
        assert "/testbed" in prompts["initialize"]

    def test_unit_prompts_01_mode_02_stateless_pair(self) -> None:
        # Act
        prompts = get_bench_agent_sys_prompts(Path("/testbed"), mode="stateless")

        # Assert
        assert "Code Reasoning Initialization System" in prompts["initialize"]
        assert "Single-Step Code Reasoning System" in prompts["step"]

    def test_unit_prompts_01_mode_04_react_uses_react_template_for_both(self) -> None:
        # Act
        prompts = get_bench_agent_sys_prompts(Path("/testbed"), mode="react")

        # Assert — the react template is rendered for both slots
        assert prompts["initialize"] == prompts["step"]
        assert "# Code Agent" in prompts["initialize"]
        assert "/testbed" in prompts["initialize"]

    def test_unit_prompts_01_mode_05_react_renders_repo_and_cwd(self) -> None:
        # Act
        prompts = get_bench_agent_sys_prompts(Path("/testbed"), mode="react")

        # Assert — repo placeholder is rendered; cwd is rendered and the
        # literal-brace escapes survive (unrendered {{ would indicate
        # str.format was not applied)
        assert "/testbed" in prompts["step"]
        assert "{{" not in prompts["step"]
        assert "{repo}" not in prompts["step"]

    def test_unit_prompts_01_mode_06_invalid_mode_rejected(self) -> None:
        # Act & Assert — the invalid mode value is deliberately mistyped
        with pytest.raises(ValueError, match="mode"):
            get_bench_agent_sys_prompts(
                Path("/testbed"),
                mode="agentic",  # type: ignore[arg-type]
            )
