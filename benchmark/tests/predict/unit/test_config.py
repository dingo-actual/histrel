#!/usr/bin/env python3
"""
Unit tests for benchmark configuration models and loader.

Naming follows the main suite taxonomy.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from benchmark.predict.config import (
    HistrelConfig,
    PredictorConfig,
    SWEbenchConfig,
    _coerce_benchmark_dict,
    load_predictor_config,
)


class Test04Unit01Config01Schema01Defaults:
    def test_04_unit_01_config_01_schema_01_defaults_01_models_defaults(self) -> None:
        # Arrange & Act
        swe = SWEbenchConfig()
        cfg = PredictorConfig()

        # Assert
        assert swe.enabled is True
        assert swe.docker is True
        assert swe.docker_base_image == "python:3.11-slim"
        assert isinstance(swe.tools, list) and {"ripgrep", "fd", "tokei"}.issubset(
            set(swe.tools)
        )

        assert cfg.output_dir == "benchmark/results"
        assert isinstance(cfg.swe_bench, SWEbenchConfig)
        assert cfg.verbose is False


class Test04Unit01Config01Schema02TomlCoercion:
    def test_04_unit_01_config_01_schema_02_toml_coercion_01_maps_verbose_table(
        self,
    ) -> None:
        # Arrange
        doc = {
            "benchmark": {
                "run_name": "coercion-test",
                "verbose": {"enabled": True},
                "swe_bench": {"dataset_id": "princeton-nlp/SWE-bench_Lite"},
            }
        }

        # Act
        data = _coerce_benchmark_dict(doc)

        # Assert
        assert data["verbose"] is True
        assert "swe_bench" in data and isinstance(data["swe_bench"], dict)
        assert data["swe_bench"]["dataset_id"] == "princeton-nlp/SWE-bench_Lite"


class Test04Unit01Config01Schema03Loader:
    def test_04_unit_01_config_01_schema_03_loader_01_load_from_path(self) -> None:
        # Arrange: create a temporary TOML file
        content = (
            "[benchmark]\n"
            'run_name = "bench-local-tests"\n'
            "[benchmark.swe_bench]\n"
            "enabled = true\n"
            'dataset_id = "princeton-nlp/SWE-bench_Lite"\n'
            "[benchmark.verbose]\n"
            "enabled = true\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / "histrel.toml"
            p.write_text(content, encoding="utf-8")

            # Act
            cfg = load_predictor_config(str(p))

            # Assert
            assert cfg.run_name == "bench-local-tests"
            assert cfg.verbose is True
            assert cfg.swe_bench.dataset_id == "princeton-nlp/SWE-bench_Lite"


class Test04Unit01Config02ResolvedSettings:
    """Resolved settings: histrel table, rendering, and round-trip."""

    def test_04_unit_01_config_02_resolved_01_coercion_reads_histrel_table(
        self,
    ) -> None:
        # Arrange
        doc = {
            "histrel": {
                "models": "accounts/fireworks/models/test-model",
                "token_budget": 5000,
                "token_budget_buffer": 500,
            },
            "benchmark": {"run_name": "resolved"},
        }

        # Act
        data = _coerce_benchmark_dict(doc)
        cfg = PredictorConfig.model_validate(data)

        # Assert
        assert cfg.histrel.model_id() == "accounts/fireworks/models/test-model"
        assert cfg.histrel.token_budget == 5000
        assert cfg.histrel.token_budget_buffer == 500

    def test_04_unit_01_config_02_resolved_02_table_model_resolves_default(
        self,
    ) -> None:
        # Act
        cfg = PredictorConfig.model_validate(
            {"histrel": {"models": {"default": "model-a", "other": "model-b"}}}
        )

        # Assert
        assert cfg.histrel.model_id() == "model-a"

    def test_04_unit_01_config_02_resolved_03_rendering_round_trips(
        self, tmp_path: Path
    ) -> None:
        # Arrange — settings a container must see, distinct from the defaults
        cfg = PredictorConfig(
            output_dir="results-out",
            run_name="round-trip",
            mode="react",
            verbose=True,
            histrel=HistrelConfig(
                models="accounts/fireworks/models/custom-model",
                token_budget=750_000,
                token_budget_buffer=10_000,
            ),
            swe_bench=SWEbenchConfig(
                dataset_id="princeton-nlp/SWE-bench_Lite",
                split="dev",
                max_instances=3,
                timeout_seconds=120,
            ),
        )
        path = tmp_path / "bench_settings.toml"
        path.write_text(cfg.render_bench_settings(), encoding="utf-8")

        # Act
        loaded = load_predictor_config(str(path))

        # Assert
        assert loaded.histrel == cfg.histrel
        assert loaded.swe_bench == cfg.swe_bench
        assert loaded.run_name == "round-trip"
        assert loaded.output_dir == "results-out"
        assert loaded.mode == "react"
        assert loaded.verbose is True

    def test_04_unit_01_config_02_resolved_04_write_into_run_dir(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        cfg = PredictorConfig(histrel=HistrelConfig(models="model-a"))

        # Act
        path = cfg.write_bench_settings(tmp_path)

        # Assert — saved with the results under the mounted file name
        assert path == tmp_path / "bench_settings.toml"
        assert load_predictor_config(str(path)).histrel.model_id() == "model-a"


class Test04Unit01Config03CliOverrides:
    """SWE-bench CLI overrides leave TOML values in place when omitted."""

    def test_04_unit_01_config_03_overrides_01_defaults_are_none(self) -> None:
        # Arrange
        from benchmark.predict.cli import build_parser

        # Act
        args = build_parser().parse_args([])

        # Assert
        assert args.split is None
        assert args.max_instances is None
        assert args.timeout_seconds is None

    def test_04_unit_01_config_03_overrides_02_omitted_flags_keep_toml(
        self,
    ) -> None:
        # Arrange
        from benchmark.predict.cli import _apply_overrides, build_parser

        cfg = PredictorConfig(
            swe_bench=SWEbenchConfig(split="dev", max_instances=7, timeout_seconds=120)
        )
        args = build_parser().parse_args([])

        # Act
        updated = _apply_overrides(cfg, args)

        # Assert
        assert updated.swe_bench.split == "dev"
        assert updated.swe_bench.max_instances == 7
        assert updated.swe_bench.timeout_seconds == 120

    def test_04_unit_01_config_03_overrides_03_flags_win(self) -> None:
        # Arrange
        from benchmark.predict.cli import _apply_overrides, build_parser

        cfg = PredictorConfig(
            swe_bench=SWEbenchConfig(split="dev", max_instances=7, timeout_seconds=120)
        )
        args = build_parser().parse_args(
            ["--split", "test", "--max-instances", "2", "--timeout-seconds", "60"]
        )

        # Act
        updated = _apply_overrides(cfg, args)

        # Assert
        assert updated.swe_bench.split == "test"
        assert updated.swe_bench.max_instances == 2
        assert updated.swe_bench.timeout_seconds == 60


class Test04Unit01Config01Schema04Mode:
    """Agent loop mode: default, loader, and CLI override."""

    def test_04_unit_01_config_01_schema_04_mode_01_default_is_stateless(self) -> None:
        # Act
        cfg = PredictorConfig()

        # Assert
        assert cfg.mode == "stateless"

    def test_04_unit_01_config_01_schema_04_mode_02_cli_flag_overrides_config(
        self, tmp_path: Path
    ) -> None:
        # Arrange
        from benchmark.predict.cli import _apply_overrides, build_parser

        cfg = PredictorConfig(mode="stateless")
        args = build_parser().parse_args(["--mode", "react"])

        # Act
        updated = _apply_overrides(cfg, args)

        # Assert
        assert updated.mode == "react"

    def test_04_unit_01_config_01_schema_04_mode_03_cli_default_leaves_config(
        self,
    ) -> None:
        # Arrange
        from benchmark.predict.cli import _apply_overrides, build_parser

        cfg = PredictorConfig()
        args = build_parser().parse_args([])

        # Act
        updated = _apply_overrides(cfg, args)

        # Assert — no flag means the config default (stateless) stands
        assert updated.mode == "stateless"

    def test_04_unit_01_config_01_schema_04_mode_04_unknown_mode_rejected(
        self,
    ) -> None:
        # Arrange
        from benchmark.predict.cli import build_parser

        # Act & Assert
        with pytest.raises(SystemExit):
            build_parser().parse_args(["--mode", "agentic"])
