#!/usr/bin/env python3
"""Unit tests for BenchAgentWrapper SWE-bench integration.

Tests config template handling, model parsing from histrel.toml,
agent config generation, usage data loading, and patch generation
flow with patched subprocess execution.
"""

from __future__ import annotations

import json
import sys
import tomllib
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

# Skip entire module on Windows before any imports that require swebench
if sys.platform == "win32":
    pytest.skip("Scorer requires Unix", allow_module_level=True)

from benchmark.predict.agents import bench_agent_wrapper as bench_agent_wrapper_module
from benchmark.predict.config import Mode
from benchmark.tests.fixtures.swebench import make_bench_agent_wrapper


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

_MINIMAL_TOML = (
    '[histrel]\nmodels = "accounts/fireworks/models/deepseek-v4p1-flash-1017"\n'
)

_MINIMAL_TOML_DICT_MODEL = '[histrel]\nmodels = {"default" = "accounts/fireworks/models/deepseek-v4p1-flash-1017"}\n'

_MODES: list[Mode] = ["stateless", "react"]


# ---------------------------------------------------------------------------
# Init and config parsing tests
# ---------------------------------------------------------------------------


class Test04Unit03BenchAgentWrapper01Init:
    """BenchAgentWrapper initialization and config parsing."""

    def test_04_unit_03_bench_agent_wrapper_01_init_01_defaults(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)

        # Act
        with make_bench_agent_wrapper(config_path) as wrapper:
            # Assert
            assert wrapper.config_path == config_path

    def test_04_unit_03_bench_agent_wrapper_01_init_03_has_repo_manager(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)

        # Act
        with make_bench_agent_wrapper(config_path) as wrapper:
            # Assert
            assert wrapper.repo_manager is not None


class Test04Unit03BenchAgentWrapper02ConfigParsing:
    """_parse_model_from_config and _load_config tests."""

    def test_04_unit_03_bench_agent_wrapper_02_config_01_parses_string_model(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)

        # Act
        with make_bench_agent_wrapper(config_path) as wrapper:
            config = wrapper._load_config()
            model = wrapper._parse_model_from_config(config)

            # Assert
            assert model == "accounts/fireworks/models/deepseek-v4p1-flash-1017"

    def test_04_unit_03_bench_agent_wrapper_02_config_02_parses_dict_model(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML_DICT_MODEL)

        # Act
        with make_bench_agent_wrapper(config_path) as wrapper:
            config = wrapper._load_config()
            model = wrapper._parse_model_from_config(config)

            # Assert
            assert model == "accounts/fireworks/models/deepseek-v4p1-flash-1017"

    def test_04_unit_03_bench_agent_wrapper_02_config_03_missing_config_raises(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "nonexistent.toml"

        # Act / Assert
        with (
            make_bench_agent_wrapper(config_path) as wrapper,
            pytest.raises(RuntimeError, match="Config file not found"),
        ):
            wrapper._load_config()


# ---------------------------------------------------------------------------
# Agent config writing tests
# ---------------------------------------------------------------------------


class Test04Unit03BenchAgentWrapper03WriteAgentConfig:
    """_write_agent_config tests."""

    def test_04_unit_03_bench_agent_wrapper_03_write_config_01_writes_toml(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        # Act
        with make_bench_agent_wrapper(config_path) as wrapper:
            agent_config = wrapper._write_agent_config(repo_dir)

            # Assert
            assert agent_config.exists()
            assert agent_config == repo_dir / "agent_config.toml"

    def test_04_unit_03_bench_agent_wrapper_03_write_config_04_provider_is_fireworks(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        # Act
        with make_bench_agent_wrapper(config_path) as wrapper:
            agent_config = wrapper._write_agent_config(repo_dir)

            # Assert — provider is always fireworks for the bench agent
            content = agent_config.read_text(encoding="utf-8")
            assert 'model_provider = "fireworks"' in content
            assert 'model_api_key_env = "FIREWORKS_API_KEY"' in content

    @pytest.mark.parametrize("mode", _MODES)
    def test_04_unit_03_bench_agent_wrapper_03_write_config_05_system_prompt_histrel(
        self, tmp_path: Path, mode: Mode
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        # Act
        with make_bench_agent_wrapper(config_path) as wrapper:
            agent_config = wrapper._write_agent_config(repo_dir, mode)

            # Assert — prompts render with repo path; react reuses one
            # prompt for both slots, stateless uses distinct prompts
            written = tomllib.loads(agent_config.read_text(encoding="utf-8"))
            assert written["agent_system_prompt"]["initialize"]
            assert written["agent_system_prompt"]["step"]
            assert str(repo_dir) in written["agent_system_prompt"]["initialize"]
            assert str(repo_dir) in written["agent_system_prompt"]["step"]
            if mode == "react":
                assert (
                    written["agent_system_prompt"]["initialize"]
                    == written["agent_system_prompt"]["step"]
                )
            else:
                assert (
                    written["agent_system_prompt"]["initialize"]
                    != written["agent_system_prompt"]["step"]
                )

    @pytest.mark.parametrize("mode", _MODES)
    def test_04_unit_03_bench_agent_wrapper_07_mode_01_mode_not_written(
        self, tmp_path: Path, mode: Mode
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        # Act
        with make_bench_agent_wrapper(config_path) as wrapper:
            agent_config = wrapper._write_agent_config(repo_dir, mode)

            # Assert — mode travels on the command line, not in the config
            written = tomllib.loads(agent_config.read_text(encoding="utf-8"))
            assert "mode" not in written

    @pytest.mark.parametrize("mode", _MODES)
    def test_04_unit_03_bench_agent_wrapper_07_mode_02_command_passes_mode(
        self, tmp_path: Path, mode: Mode
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fake_result = MagicMock(returncode=0, stdout="", stderr="")
        captured_cmd: list[str] = []

        def capture_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return fake_result

        with (
            make_bench_agent_wrapper(config_path) as wrapper,
            patch.object(wrapper.repo_manager, "validate_repo"),
            patch.object(wrapper.repo_manager, "extract_patch", return_value=""),
            patch("subprocess.run", side_effect=capture_run),
            patch.object(
                wrapper, "_load_usage_data", return_value={"total": {"tokens_total": 0}}
            ),
            patch("benchmark.predict.agents.bench_agent_wrapper.Path") as mock_path,
        ):
            mock_path.return_value = output_dir

            # Act
            wrapper.generate_patch(
                instance={"instance_id": "test-123"},
                repo_dir=repo_dir,
                session_id="test-session",
                mode=mode,
            )

            # Assert — the agent is told the mode explicitly
            shell_cmd = captured_cmd[2]
            assert f"--mode {mode}" in shell_cmd

    def test_04_unit_03_bench_agent_wrapper_07_mode_04_invalid_mode_rejected(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()

        # Act & Assert
        with (
            make_bench_agent_wrapper(config_path) as wrapper,
            pytest.raises(RuntimeError, match="agentic"),
        ):
            wrapper._write_agent_config(repo_dir, cast(Mode, "agentic"))


# ---------------------------------------------------------------------------
# Usage data loading tests
# ---------------------------------------------------------------------------


class Test04Unit03BenchAgentWrapper04LoadUsageData:
    """_load_usage_data tests."""

    def test_04_unit_03_bench_agent_wrapper_04_usage_01_valid_costs_and_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        monkeypatch.setenv("SESSION_ID", "test-session-1")
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        costs_file = artifact_dir / "costs.json"
        costs_file.write_text(
            json.dumps(
                {"total_cost": 0.05, "model_cost": 0.04, "compaction_cost": 0.01}
            )
        )
        calls_file = artifact_dir / "token_usage_calls.jsonl"
        calls_file.write_text(
            json.dumps({"usage": {"input": 1000, "output": 500, "cached": 200}})
            + "\n"
            + json.dumps({"usage": {"input": 2000, "output": 1000, "cached": 400}})
            + "\n"
        )

        # Act
        with make_bench_agent_wrapper() as wrapper:
            result = wrapper._load_usage_data(costs_file)

            # Assert
            assert result["total"]["tokens_input"] == 3000
            assert result["total"]["tokens_output"] == 1500
            assert result["total"]["tokens_cache_read"] == 600
            assert result["total"]["tokens_total"] == 4500
            assert result["total"]["cost_total"] == 0.05
            assert result["total"]["session_id"] == "test-session-1"
            assert result["total"]["maximum_context"] == 2000  # max(1000, 2000)
            # "none" mirrors "total"
            assert result["none"]["tokens_total"] == 4500
            assert result["none"]["maximum_context"] == 2000

    def test_04_unit_03_bench_agent_wrapper_04_usage_02_missing_file(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        missing_file = tmp_path / "nonexistent.json"

        # Act
        with make_bench_agent_wrapper() as wrapper:
            result = wrapper._load_usage_data(missing_file)

            # Assert — no cost report means an unknown cost, not -1 or 0.
            assert result["total"]["tokens_total"] == 0
            assert result["total"]["cost_total"] is None
            assert result["total"]["cost_complete"] is False

    def test_04_unit_03_bench_agent_wrapper_04_usage_03_invalid_json(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        costs_file = tmp_path / "costs.json"
        costs_file.write_text("not valid json")

        # Act
        with make_bench_agent_wrapper() as wrapper:
            result = wrapper._load_usage_data(costs_file)

            # Assert
            assert result["total"]["tokens_total"] == 0
            assert result["total"]["cost_total"] is None

    def test_04_unit_03_bench_agent_wrapper_04_usage_03b_missing_costs_keeps_tokens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — the session died before finalizing costs.json, but the
        # incrementally appended call logs are on disk.
        monkeypatch.setenv("SESSION_ID", "sess-crashed")
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()
        (artifacts_dir / "token_usage_calls.jsonl").write_text(
            json.dumps({"usage": {"input": 700, "output": 300, "cached": 100}}) + "\n"
        )
        (artifacts_dir / "histrel_calls.jsonl").write_text(
            json.dumps(
                {
                    "run_id": "run-1",
                    "usage": {"input": 200, "output": 50, "cached": 0},
                    "cost_usd": 0.02,
                }
            )
            + "\n"
        )

        # Act
        with make_bench_agent_wrapper() as wrapper:
            result = wrapper._load_usage_data(artifacts_dir / "costs.json")

            # Assert — token usage is preserved; the cost is unknown, not 0
            # and not -1.
            assert result["total"]["tokens_input"] == 900
            assert result["total"]["tokens_total"] == 1250
            assert result["total"]["cost_total"] is None
            assert result["total"]["cost_complete"] is False
            assert result["run-1"]["cost_total"] == 0.02

    def test_04_unit_03_bench_agent_wrapper_04_usage_04_no_calls_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — costs.json exists but token_usage_calls.jsonl does not
        monkeypatch.setenv("SESSION_ID", "sess-no-calls")
        costs_file = tmp_path / "costs.json"
        costs_file.write_text(json.dumps({"total_cost": 0.0}))

        # Act
        with make_bench_agent_wrapper() as wrapper:
            result = wrapper._load_usage_data(costs_file)

            # Assert
            assert result["total"]["tokens_input"] == 0
            assert result["total"]["tokens_output"] == 0
            assert result["total"]["tokens_total"] == 0
            assert result["total"]["cost_total"] == 0.0

    def test_04_unit_03_bench_agent_wrapper_04_usage_05_partial_calls_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — some lines are valid, some are garbage
        monkeypatch.setenv("SESSION_ID", "sess-partial")
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        costs_file = artifact_dir / "costs.json"
        costs_file.write_text(json.dumps({"total_cost": 0.02}))
        calls_file = artifact_dir / "token_usage_calls.jsonl"
        calls_file.write_text(
            json.dumps({"usage": {"input": 500, "output": 200}})
            + "\n"
            + "garbage line\n"
            + json.dumps({"usage": {"input": 100, "output": 50}})
            + "\n"
        )

        # Act
        with make_bench_agent_wrapper() as wrapper:
            result = wrapper._load_usage_data(costs_file)

            # Assert — only valid lines are counted
            assert result["total"]["tokens_input"] == 600
            assert result["total"]["tokens_output"] == 250

    def test_04_unit_03_bench_agent_wrapper_04_usage_06_includes_histrel_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — costs.json, token_usage_calls.jsonl, and histrel_calls.jsonl
        monkeypatch.setenv("SESSION_ID", "test-session-mt")
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        costs_file = artifact_dir / "costs.json"
        costs_file.write_text(
            json.dumps({"total_cost": 0.06, "model_cost": 0.04, "compaction_cost": 0.0})
        )
        calls_file = artifact_dir / "token_usage_calls.jsonl"
        calls_file.write_text(
            json.dumps({"usage": {"input": 1000, "output": 500, "cached": 200}}) + "\n"
        )
        histrel_file = artifact_dir / "histrel_calls.jsonl"
        histrel_file.write_text(
            json.dumps(
                {
                    "run_id": "test-session-mt-123-analyze-query",
                    "usage": {
                        "input": 5000,
                        "output": 2000,
                        "cached": 1000,
                        "reasoning": 500,
                        "total": 7000,
                        "maximum_context": 4200,
                    },
                    "cost_usd": 0.031,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "run_id": "test-session-mt-456-compose-code",
                    "usage": {
                        "input": 8000,
                        "output": 3000,
                        "cached": 2000,
                        "reasoning": 0,
                        "total": 11000,
                        "maximum_context": 7500,
                    },
                    "cost_usd": 0.052,
                }
            )
            + "\n"
        )

        # Act
        with make_bench_agent_wrapper() as wrapper:
            result = wrapper._load_usage_data(costs_file)

            # Assert — total includes model + histrel tokens
            assert result["total"]["tokens_input"] == 1000 + 5000 + 8000
            assert result["total"]["tokens_output"] == 500 + 2000 + 3000
            assert result["total"]["tokens_cache_read"] == 200 + 1000 + 2000
            assert result["total"]["tokens_total"] == 1500 + 7000 + 11000
            assert result["total"]["cost_total"] == 0.06
            # maximum_context is the largest single LLM-call input, never a
            # per-run or per-call token total (input + output)
            assert result["total"]["maximum_context"] == max(1000, 4200, 7500)
            # "none" reflects bench-agent-only usage (excludes Histrel)
            assert result["none"]["tokens_input"] == 1000
            assert result["none"]["tokens_output"] == 500
            assert result["none"]["tokens_total"] == 1500
            assert result["none"]["maximum_context"] == 1000
            assert result["none"]["cost_total"] == pytest.approx(0.04)
            # Per-tool keys exist at top level keyed by run_id
            assert "test-session-mt-123-analyze-query" in result
            assert result["test-session-mt-123-analyze-query"]["tokens_input"] == 5000
            assert result["test-session-mt-123-analyze-query"]["cost_total"] == 0.031
            assert (
                result["test-session-mt-123-analyze-query"]["maximum_context"] == 4200
            )
            assert "test-session-mt-456-compose-code" in result
            assert result["test-session-mt-456-compose-code"]["tokens_input"] == 8000
            assert result["test-session-mt-456-compose-code"]["cost_total"] == 0.052
            assert result["test-session-mt-456-compose-code"]["maximum_context"] == 7500

    def test_04_unit_03_bench_agent_wrapper_04_usage_07_histrel_calls_missing_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — only costs.json and token_usage_calls.jsonl, no histrel
        monkeypatch.setenv("SESSION_ID", "sess-no-mt")
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        costs_file = artifact_dir / "costs.json"
        costs_file.write_text(json.dumps({"total_cost": 0.01}))
        calls_file = artifact_dir / "token_usage_calls.jsonl"
        calls_file.write_text(
            json.dumps({"usage": {"input": 500, "output": 200}}) + "\n"
        )

        # Act
        with make_bench_agent_wrapper() as wrapper:
            result = wrapper._load_usage_data(costs_file)

            # Assert — works fine without histrel_calls.jsonl
            assert result["total"]["tokens_input"] == 500
            assert result["total"]["tokens_output"] == 200

    def test_04_unit_03_bench_agent_wrapper_04_usage_08_histrel_partial_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange — histrel_calls.jsonl with some invalid lines
        monkeypatch.setenv("SESSION_ID", "sess-partial-mt")
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        costs_file = artifact_dir / "costs.json"
        costs_file.write_text(json.dumps({"total_cost": 0.03}))
        calls_file = artifact_dir / "token_usage_calls.jsonl"
        calls_file.write_text(
            json.dumps({"usage": {"input": 1000, "output": 500}}) + "\n"
        )
        histrel_file = artifact_dir / "histrel_calls.jsonl"
        histrel_file.write_text(
            json.dumps(
                {
                    "run_id": "sess-partial-mt-1-analyze-query",
                    "usage": {"input": 5000, "output": 2000, "total": 7000},
                }
            )
            + "\n"
            + "garbage line\n"
            + json.dumps(
                {
                    "run_id": "sess-partial-mt-2-compose-code",
                    "usage": {"input": 3000, "output": 1000, "total": 4000},
                }
            )
            + "\n"
        )

        # Act
        with make_bench_agent_wrapper() as wrapper:
            result = wrapper._load_usage_data(costs_file)

            # Assert — only valid histrel lines are counted
            assert result["total"]["tokens_input"] == 1000 + 5000 + 3000
            assert result["total"]["tokens_output"] == 500 + 2000 + 1000
            assert "sess-partial-mt-1-analyze-query" in result
            assert "sess-partial-mt-2-compose-code" in result

    def test_04_unit_03_bench_agent_wrapper_04_usage_09_histrel_maximum_context_not_tokens_total(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regress the session_usage.json bug where each Histrel run
        reported maximum_context equal to tokens_total.

        maximum_context must stay the largest single LLM-call input of the
        run (as reported by tool_api), never the run's token total.
        """
        # Arrange — a session with one model call and one Histrel run whose
        # cumulative totals dwarf its largest single-call input
        monkeypatch.setenv("SESSION_ID", "test-session-histrel-b")
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        costs_file = artifact_dir / "costs.json"
        costs_file.write_text(json.dumps({"total_cost": 0.10}))
        calls_file = artifact_dir / "token_usage_calls.jsonl"
        calls_file.write_text(
            json.dumps({"usage": {"input": 3000, "output": 800, "cached": 100}}) + "\n"
        )
        histrel_file = artifact_dir / "histrel_calls.jsonl"
        histrel_file.write_text(
            json.dumps(
                {
                    "run_id": "test-session-histrel-b-1789374251-analyze-query",
                    "usage": {
                        "input": 181890,
                        "output": 8862,
                        "cached": 20480,
                        "total": 190752,
                        "maximum_context": 145000,
                    },
                    "cost_usd": 0.03,
                }
            )
            + "\n"
        )

        # Act
        with make_bench_agent_wrapper() as wrapper:
            result = wrapper._load_usage_data(costs_file)

        # Assert — the per-run entry keeps maximum_context distinct from
        # the run's token total
        run_key = "test-session-histrel-b-1789374251-analyze-query"
        assert result[run_key]["tokens_total"] == 190752
        assert result[run_key]["maximum_context"] == 145000
        assert result[run_key]["maximum_context"] != result[run_key]["tokens_total"]
        # Session totals report the largest single-call input
        assert result["total"]["tokens_total"] == 190752 + 3800
        assert result["total"]["maximum_context"] == max(3000, 145000)
        assert result["none"]["maximum_context"] == 3000

    def test_04_unit_03_bench_agent_wrapper_04_usage_10_none_cost_excludes_histrel(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regress the usage.json bug where the "none" record copied the
        "total" cost, which includes Histrel tool call costs.

        "none" must report the bench-agent-only cost (model + compaction
        calls), so it stays smaller than "total" whenever Histrel calls
        occurred.
        """
        # Arrange — costs.json with model/compaction/histrel costs that
        # sum exactly to total_cost
        monkeypatch.setenv("SESSION_ID", "test-session-none-cost")
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        costs_file = artifact_dir / "costs.json"
        costs_file.write_text(
            json.dumps(
                {
                    "total_cost": 0.105,
                    "model_cost": 0.02,
                    "compaction_cost": 0.005,
                }
            )
        )
        calls_file = artifact_dir / "token_usage_calls.jsonl"
        calls_file.write_text(
            json.dumps({"usage": {"input": 1000, "output": 500, "cached": 200}}) + "\n"
        )
        histrel_file = artifact_dir / "histrel_calls.jsonl"
        histrel_file.write_text(
            json.dumps(
                {
                    "run_id": "test-session-none-cost-1-analyze-query",
                    "usage": {"input": 5000, "output": 2000, "total": 7000},
                    "cost_usd": 0.03,
                }
            )
            + "\n"
            + json.dumps(
                {
                    "run_id": "test-session-none-cost-2-compose-code",
                    "usage": {"input": 3000, "output": 1000, "total": 4000},
                    "cost_usd": 0.05,
                }
            )
            + "\n"
        )

        # Act
        with make_bench_agent_wrapper() as wrapper:
            result = wrapper._load_usage_data(costs_file)

        # Assert — "none" cost excludes the Histrel call costs
        assert result["none"]["cost_total"] == pytest.approx(0.025)
        assert result["none"]["cost_total"] != result["total"]["cost_total"]
        # "none" plus every Histrel call cost reconstructs "total"
        assert result["none"]["cost_total"] + 0.03 + 0.05 == pytest.approx(
            result["total"]["cost_total"]
        )

    def test_04_unit_03_bench_agent_wrapper_04_usage_11_none_cost_includes_compaction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The "none" cost must include compaction call costs, since
        compaction calls are bench-agent LLM calls recorded in
        token_usage_calls.jsonl.
        """
        # Arrange — no Histrel calls; model + compaction costs differ from
        # total only by rounding-free fixture values
        monkeypatch.setenv("SESSION_ID", "test-session-compaction-cost")
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        costs_file = artifact_dir / "costs.json"
        costs_file.write_text(
            json.dumps(
                {
                    "total_cost": 0.05,
                    "model_cost": 0.03,
                    "compaction_cost": 0.01,
                }
            )
        )
        calls_file = artifact_dir / "token_usage_calls.jsonl"
        calls_file.write_text(
            json.dumps({"usage": {"input": 1000, "output": 500, "cached": 200}}) + "\n"
        )

        # Act
        with make_bench_agent_wrapper() as wrapper:
            result = wrapper._load_usage_data(costs_file)

        # Assert — "none" cost is model cost plus compaction cost
        assert result["none"]["cost_total"] == pytest.approx(0.04)

    def test_04_unit_03_bench_agent_wrapper_04_usage_12_total_tokens_total_matches_run_totals(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The "total" record's tokens_total must sum the per-record token
        totals, not input + output per source.

        Histrel accounts total-only LLM calls through TokenManager.add(),
        where tokens_total grows faster than tokens_input + tokens_output.
        The per-tool record reports the run's tokens_total; the session
        "total" record must not drop that difference.
        """
        # Arrange — one model call and one Histrel run whose tokens_total
        # (9000) exceeds input + output (7000) due to total-only accounting
        monkeypatch.setenv("SESSION_ID", "test-session-run-totals")
        artifact_dir = tmp_path / "artifacts"
        artifact_dir.mkdir()
        costs_file = artifact_dir / "costs.json"
        costs_file.write_text(
            json.dumps({"total_cost": 0.05, "model_cost": 0.04, "compaction_cost": 0.0})
        )
        calls_file = artifact_dir / "token_usage_calls.jsonl"
        calls_file.write_text(
            json.dumps(
                {"usage": {"input": 1000, "output": 400, "cached": 100, "total": 1400}}
            )
            + "\n"
        )
        histrel_file = artifact_dir / "histrel_calls.jsonl"
        histrel_file.write_text(
            json.dumps(
                {
                    "run_id": "test-session-run-totals-1-analyze-query",
                    "usage": {
                        "input": 5000,
                        "output": 2000,
                        "cached": 1000,
                        "total": 9000,
                        "maximum_context": 4500,
                    },
                    "cost_usd": 0.01,
                }
            )
            + "\n"
        )

        # Act
        with make_bench_agent_wrapper() as wrapper:
            result = wrapper._load_usage_data(costs_file)

        # Assert — per-tool entry keeps the run's reported tokens_total
        run_key = "test-session-run-totals-1-analyze-query"
        assert result[run_key]["tokens_total"] == 9000
        # Session "total" sums reported per-record totals (1400 + 9000),
        # not per-source input + output (1400 + 7000)
        assert result["total"]["tokens_total"] == 10400
        assert result["total"]["tokens_input"] == 6000
        assert result["total"]["tokens_output"] == 2400
        assert result["none"]["tokens_total"] == 1400


# ---------------------------------------------------------------------------
# generate_patch tests
# ---------------------------------------------------------------------------


class Test04Unit03BenchAgentWrapper05GeneratePatch:
    """generate_patch contract and boundary tests."""

    @pytest.fixture
    def _patched_generate(self, tmp_path: Path):
        """Return a context manager that patches subprocess and output paths."""
        fake_result = MagicMock(returncode=0, stdout="", stderr="")
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        def _make_patches(wrapper):
            return (
                patch.object(wrapper.repo_manager, "validate_repo"),
                patch.object(
                    wrapper.repo_manager, "extract_patch", return_value="diff --git"
                ),
                patch("subprocess.run", return_value=fake_result),
                patch.object(
                    wrapper,
                    "_load_usage_data",
                    return_value={"total": {"tokens_total": 100}},
                ),
                patch(
                    "benchmark.predict.agents.bench_agent_wrapper.Path",
                    return_value=output_dir,
                ),
            )

        return _make_patches

    def test_04_unit_03_bench_agent_wrapper_05_patch_01_contract_01_returns_tuple(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()
        instance = {"instance_id": "test-123", "repo": "test/repo"}
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fake_result = MagicMock(returncode=0, stdout="", stderr="")

        with (
            make_bench_agent_wrapper(config_path) as wrapper,
            patch.object(wrapper.repo_manager, "validate_repo"),
            patch.object(
                wrapper.repo_manager, "extract_patch", return_value="diff --git"
            ),
            patch("subprocess.run", return_value=fake_result),
            patch.object(
                wrapper,
                "_load_usage_data",
                return_value={"total": {"tokens_total": 100}},
            ),
            patch("benchmark.predict.agents.bench_agent_wrapper.Path") as mock_path,
        ):
            # Make Path() return our tmp output_dir for the /workspace/output case
            mock_path.return_value = output_dir
            # Act
            result = wrapper.generate_patch(
                instance=instance,
                repo_dir=repo_dir,
                session_id="test-session",
            )

            # Assert
            assert isinstance(result, tuple)
            assert len(result) == 2
            assert result[0] == "diff --git"

    def test_04_unit_03_bench_agent_wrapper_05_patch_02_boundary_01_empty_instance(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()
        instance: dict[str, Any] = {}
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fake_result = MagicMock(returncode=0, stdout="", stderr="")

        with (
            make_bench_agent_wrapper(config_path) as wrapper,
            patch.object(wrapper.repo_manager, "validate_repo"),
            patch.object(wrapper.repo_manager, "extract_patch", return_value=""),
            patch("subprocess.run", return_value=fake_result),
            patch.object(
                wrapper, "_load_usage_data", return_value={"total": {"tokens_total": 0}}
            ),
            patch("benchmark.predict.agents.bench_agent_wrapper.Path") as mock_path,
        ):
            mock_path.return_value = output_dir
            # Act — should not raise on empty instance
            result = wrapper.generate_patch(
                instance=instance,
                repo_dir=repo_dir,
                session_id="test",
            )

            # Assert
            assert isinstance(result, tuple)

    def test_04_unit_03_bench_agent_wrapper_05_patch_03_boundary_02_has_test_runner_false(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()
        instance = {"instance_id": "test-123"}
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fake_result = MagicMock(returncode=0, stdout="", stderr="")

        with (
            make_bench_agent_wrapper(config_path) as wrapper,
            patch.object(wrapper.repo_manager, "validate_repo"),
            patch.object(wrapper.repo_manager, "extract_patch", return_value=""),
            patch("subprocess.run", return_value=fake_result),
            patch.object(
                wrapper, "_load_usage_data", return_value={"total": {"tokens_total": 0}}
            ),
            patch("benchmark.predict.agents.bench_agent_wrapper.Path") as mock_path,
        ):
            mock_path.return_value = output_dir
            # Act — should handle has_test_runner=False
            result = wrapper.generate_patch(
                instance=instance,
                repo_dir=repo_dir,
                session_id="test",
                has_test_runner=False,
            )

            # Assert
            assert isinstance(result, tuple)

    def test_04_unit_03_bench_agent_wrapper_05_patch_04_isolation_01_writes_agent_config(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()
        instance = {"instance_id": "test-123"}
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fake_result = MagicMock(returncode=0, stdout="", stderr="")

        with (
            make_bench_agent_wrapper(config_path) as wrapper,
            patch.object(wrapper.repo_manager, "validate_repo"),
            patch.object(wrapper.repo_manager, "extract_patch", return_value=""),
            patch("subprocess.run", return_value=fake_result),
            patch.object(
                wrapper, "_load_usage_data", return_value={"total": {"tokens_total": 0}}
            ),
            patch("benchmark.predict.agents.bench_agent_wrapper.Path") as mock_path,
        ):
            mock_path.return_value = output_dir
            # Act
            wrapper.generate_patch(
                instance=instance,
                repo_dir=repo_dir,
                session_id="test-session",
            )

            # Assert — agent_config.toml was written to repo_dir
            agent_config = repo_dir / "agent_config.toml"
            assert agent_config.exists()

    def test_04_unit_03_bench_agent_wrapper_05_patch_05_isolation_02_command_uses_python_module(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()
        instance = {"instance_id": "test-123"}
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fake_result = MagicMock(returncode=0, stdout="", stderr="")
        captured_cmd: list[str] = []

        def capture_run(cmd, **kwargs):
            captured_cmd.extend(cmd)
            return fake_result

        with (
            make_bench_agent_wrapper(config_path) as wrapper,
            patch.object(wrapper.repo_manager, "validate_repo"),
            patch.object(wrapper.repo_manager, "extract_patch", return_value=""),
            patch("subprocess.run", side_effect=capture_run),
            patch.object(
                wrapper, "_load_usage_data", return_value={"total": {"tokens_total": 0}}
            ),
            patch("benchmark.predict.agents.bench_agent_wrapper.Path") as mock_path,
        ):
            mock_path.return_value = output_dir
            # Act
            wrapper.generate_patch(
                instance=instance,
                repo_dir=repo_dir,
                session_id="test-session",
            )

            # Assert — the command runs via bash -c and contains
            # "python -m histrel"
            assert len(captured_cmd) >= 1
            shell_cmd = captured_cmd[2] if len(captured_cmd) >= 3 else ""
            assert "python -m histrel" in shell_cmd

    def test_04_unit_03_bench_agent_wrapper_05_patch_06_failure_01_nonzero_exit_logs_error(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()
        instance = {"instance_id": "test-123"}
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        fake_result = MagicMock(returncode=1, stdout="output", stderr="error")

        with (
            make_bench_agent_wrapper(config_path) as wrapper,
            patch.object(wrapper.repo_manager, "validate_repo"),
            patch.object(wrapper.repo_manager, "extract_patch", return_value=""),
            patch("subprocess.run", return_value=fake_result),
            patch.object(
                wrapper, "_load_usage_data", return_value={"total": {"tokens_total": 0}}
            ),
            patch("benchmark.predict.agents.bench_agent_wrapper.Path") as mock_path,
            patch("sys.stderr.write") as mock_stderr,
        ):
            mock_path.return_value = output_dir
            # Act — should not raise, but log error
            result = wrapper.generate_patch(
                instance=instance,
                repo_dir=repo_dir,
                session_id="test-session",
            )

            # Assert
            assert isinstance(result, tuple)
            assert mock_stderr.called

    def test_04_unit_03_bench_agent_wrapper_05_patch_07_usage_identity(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Arrange
        config_path = tmp_path / "histrel.toml"
        config_path.write_text(_MINIMAL_TOML)
        repo_dir = tmp_path / "repo"
        repo_dir.mkdir()
        (repo_dir / ".git").mkdir()
        instance = {"instance_id": "test-123"}
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        monkeypatch.setenv("RUN_NAME", "repeat_02")
        monkeypatch.setenv("REPETITION", "2")

        with (
            make_bench_agent_wrapper(config_path) as wrapper,
            patch.object(wrapper.repo_manager, "validate_repo"),
            patch.object(wrapper.repo_manager, "extract_patch", return_value=""),
            patch("subprocess.run", return_value=MagicMock(returncode=0)),
            patch("benchmark.predict.agents.bench_agent_wrapper.Path") as mock_path,
        ):
            mock_path.return_value = output_dir
            # Act
            _, usage = wrapper.generate_patch(
                instance=instance,
                repo_dir=repo_dir,
                session_id="repeat-02-bench-agent-test-123",
                mode="react",
            )

        # Assert — the record states which attempt produced it
        assert usage["identity"] == {
            "run_name": "repeat_02",
            "repetition": 2,
            "mode": "react",
            "agent_type": "bench-agent",
            "instance_id": "test-123",
            "session_id": "repeat-02-bench-agent-test-123",
        }
        written = json.loads(
            (output_dir / "bench-agent_usage.json").read_text(encoding="utf-8")
        )
        assert written["identity"] == usage["identity"]


# ---------------------------------------------------------------------------
# Docker runner dispatch tests
# ---------------------------------------------------------------------------


class Test04Unit03BenchAgentWrapper06DockerRunner:
    """DockerRunner.run_agent configures the bench-agent container."""

    def test_04_unit_03_bench_agent_wrapper_06_docker_01_dispatches_bench_agent(
        self, docker_client_stub, tmp_path: Path
    ) -> None:
        # Arrange
        from benchmark.predict.config import PredictorConfig
        from benchmark.predict.docker.runner import DockerRunner

        cfg = PredictorConfig(
            output_dir=str(tmp_path / "results"),
            run_name="bench-test",
        )
        runner = DockerRunner(docker_client_stub)
        instance = {"instance_id": "dummy", "repo": "owner/repo"}
        out_dir = Path(cfg.output_dir) / cfg.run_name / "artifacts"

        # Act
        patch_path, token_usage, exit_code = runner.run_agent(
            instance,
            image="swebench/test:latest",
            config=cfg,
            output_dir=out_dir,
            timeout=5,
        )

        # Assert
        assert patch_path.name == "bench-agent_dummy.jsonl"
        assert docker_client_stub.containers.run_calls
        call = docker_client_stub.containers.run_calls[0]
        # AGENT_TYPE is set to the single supported agent type
        assert call["environment"]["AGENT_TYPE"] == "bench-agent"

    def test_04_unit_03_bench_agent_wrapper_06_docker_03_session_id_format(
        self, docker_client_stub, tmp_path: Path
    ) -> None:
        # Arrange
        from benchmark.predict.config import PredictorConfig
        from benchmark.predict.docker.runner import DockerRunner

        cfg = PredictorConfig(
            output_dir=str(tmp_path / "results"),
            run_name="bench_run_001",
        )
        runner = DockerRunner(docker_client_stub)
        instance = {"instance_id": "django__django-12345", "repo": "django/django"}
        out_dir = Path(cfg.output_dir) / cfg.run_name / "artifacts"

        # Act
        runner.run_agent(
            instance,
            image="swebench/test:latest",
            config=cfg,
            output_dir=out_dir,
            timeout=5,
        )

        # Assert — SESSION_ID uses hyphens, not underscores
        call = docker_client_stub.containers.run_calls[0]
        session_id = call["environment"]["SESSION_ID"]
        # All underscores are replaced with hyphens in all parts
        assert session_id == "bench-run-001-bench-agent-django--django-12345"
        assert "_" not in session_id


class Test04Unit03BenchAgentWrapper07TemplatePath:
    """Agent config template resolution across checkout and installed layouts."""

    def test_04_unit_03_bench_agent_wrapper_07_template_01_uses_repo_root(
        self,
    ) -> None:
        # Act
        template_path = bench_agent_wrapper_module._agent_config_template_path()

        # Assert — running from a checkout, the repository-root file is used
        assert template_path.name == "histrel.toml"
        assert template_path.exists()
        assert (
            template_path.parent
            == Path(bench_agent_wrapper_module.__file__).resolve().parents[3]
        )

    def test_04_unit_03_bench_agent_wrapper_07_template_02_falls_back_to_package(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Arrange — an installed layout: no histrel.toml beside site-packages,
        # one inside the histrel package
        site_packages = tmp_path / "site-packages"
        package_dir = site_packages / "histrel"
        package_dir.mkdir(parents=True)
        package_init = package_dir / "__init__.py"
        package_init.write_text("", encoding="utf-8")
        packaged_template = package_dir / "histrel.toml"
        packaged_template.write_text('mode = "stateless"\n', encoding="utf-8")
        wrapper_path = site_packages / "benchmark" / "predict" / "agents" / "wrapper.py"
        wrapper_path.parent.mkdir(parents=True)
        wrapper_path.write_text("", encoding="utf-8")

        monkeypatch.setattr(bench_agent_wrapper_module, "__file__", str(wrapper_path))
        monkeypatch.setattr(
            "benchmark.predict.agents.bench_agent_wrapper.histrel.__file__",
            str(package_init),
        )

        # Act
        template_path = bench_agent_wrapper_module._agent_config_template_path()

        # Assert
        assert template_path == packaged_template
