#!/usr/bin/env python3
"""
Allow running the benchmark agent as a module: python -m histrel
"""

import argparse
import sys
from pathlib import Path
from typing import cast

from histrel.cli import main
from histrel.core.config import Mode


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for ``python -m histrel``.

    Returns:
        (argparse.ArgumentParser) Parser for the agent entry point.
    """
    parser = argparse.ArgumentParser(
        prog="python -m histrel",
        description="Run the Histrel code agent on a workspace.",
    )
    parser.add_argument(
        "--mode",
        choices=("stateless", "react"),
        default="stateless",
        help="Agent loop mode (default: stateless).",
    )
    parser.add_argument("config_path", type=Path, help="Path to the agent TOML.")
    parser.add_argument("workspace_root", type=Path, help="Workspace root.")
    parser.add_argument("output_dir", type=Path, help="Artifact output directory.")
    parser.add_argument("user_msg", help="User request message for the session.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    sys.exit(
        main(
            user_msg=args.user_msg,
            config_path=args.config_path,
            workspace_root=args.workspace_root,
            output_dir=args.output_dir,
            mode=cast(Mode, args.mode),
        )
    )
