#!/usr/bin/env python3
"""
Unit tests for benchmark configuration models and loader.

Naming follows the main suite taxonomy.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from benchmark.predict.config import (
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
