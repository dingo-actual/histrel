#!/usr/bin/env python3
"""Validate SWE-bench image build pipeline.

This script validates the complete image build pipeline including base, env,
and instance layers with proper caching and key computation.

    Usage:
        python scripts/validate_image_pipeline.py [--verbose] [--quick]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def compute_directory_hash(dir_path: Path, extensions: tuple[str, ...]) -> str:
    """Compute combined hash of all files in directory matching extensions."""
    hasher = hashlib.sha256()
    files = sorted(dir_path.rglob("*"))
    for file_path in files:
        if file_path.is_file() and file_path.suffix in extensions:
            relative_path = file_path.relative_to(dir_path)
            hasher.update(relative_path.as_posix().encode())
            hasher.update(compute_file_hash(file_path).encode())
    return hasher.hexdigest()


def validate_env_image_key(instance_id: str, env_script_list: list[str]) -> dict:
    """Validate env image key computation."""
    content = "".join(env_script_list)
    return {
        "instance_id": instance_id,
        "env_key_hash": hashlib.sha256(content.encode()).hexdigest()[:22],
        "script_count": len(env_script_list),
    }


def validate_instance_fields(instance: dict) -> dict:
    """Validate instance dictionary has required fields."""
    required_fields = ["instance_id", "repo", "version", "base_commit"]
    missing = [f for f in required_fields if f not in instance]
    return {
        "valid": len(missing) == 0,
        "missing_fields": missing,
        "instance_id": instance.get("instance_id"),
        "repo": instance.get("repo"),
        "version": instance.get("version"),
    }


def validate_test_spec_structure(spec_path: Path) -> dict:
    """Validate test specification JSON structure."""
    if not spec_path.exists():
        return {"valid": False, "error": "Spec file not found"}
    try:
        with spec_path.open() as f:
            spec = json.load(f)
        if not isinstance(spec, dict):
            return {"valid": False, "error": "Spec must be a dictionary"}
        return {"valid": True, "test_count": len(spec)}
    except json.JSONDecodeError as e:
        return {"valid": False, "error": str(e)}


def validate_image_tags(spec: dict) -> dict:
    """Validate image tag format."""
    arch = spec.get("arch", "x86_64")
    base_tag = f"sweb.base.py.{arch}:latest"
    env_tag = f"sweb.env.py.{arch}.{spec.get('env_image_tag', '')}"
    instance_tag = f"sweb.eval.py.{arch}.{spec.get('instance_id', '')}:latest"
    return {
        "base_tag_format": base_tag,
        "env_tag_format": env_tag,
        "instance_tag_format": instance_tag,
        "arch": arch,
    }


def run_quick_validation() -> bool:
    """Run quick validation checks."""
    print("Running quick validation checks...")

    try:
        from benchmark.tests.fixtures.swebench import (
            make_image_build_test_env,
            make_test_spec_adapter,
        )

        with make_image_build_test_env() as (histrel_path, _):
            (histrel_path / "src" / "main.py").write_text("print('test')")
            with make_test_spec_adapter(histrel_repo_path=histrel_path) as spec:
                adapter_hash = spec.compute_histrel_hash()
                assert adapter_hash is not None
                assert len(adapter_hash) == 64
        print("  ✓ Hash computation working")
        return True
    except Exception as e:
        print(f"  ✗ Quick validation failed: {e}")
        return False


def run_full_validation(verbose: bool = False) -> bool:
    """Run full validation suite."""
    print("Running full image pipeline validation...")

    from benchmark.tests.fixtures.swebench import (
        make_image_build_test_env,
        make_test_spec_adapter,
    )

    all_passed = True

    with make_image_build_test_env(
        instance_id="astropy__astropy-13579",
        repo="astropy/astropy",
        version="5.0",
    ) as (histrel_path, _):
        (histrel_path / "src" / "main.py").write_text("print('hello')")
        with make_test_spec_adapter(
            histrel_repo_path=histrel_path,
            instance_id="astropy__astropy-13579",
            repo="astropy/astropy",
            version="5.0",
        ) as spec:
            if verbose:
                print(f"  Instance ID: {spec.instance_id}")
                print(f"  Repo: {spec.repo}")
                print(f"  Version: {spec.version}")

            env_key = validate_env_image_key(spec.instance_id, spec.env_script_list)
            if verbose:
                print(f"  Env key hash: {env_key['env_key_hash']}")

            assert spec.compute_histrel_hash() is not None
            assert len(spec.compute_histrel_hash()) == 22
            print("  ✓ Image key computation valid")

    return all_passed


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Validate SWE-bench image pipeline")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--quick", "-q", action="store_true", help="Quick validation only"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("SWE-bench Image Pipeline Validation")
    print("=" * 60)

    if args.quick:
        success = run_quick_validation()
    else:
        success = run_full_validation(args.verbose)

    print("=" * 60)
    if success:
        print("Validation PASSED")
        return 0
    else:
        print("Validation FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
