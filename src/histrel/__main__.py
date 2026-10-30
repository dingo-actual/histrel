#!/usr/bin/env python3
"""
Allow running the benchmark agent as a module: python -m histrel
"""

import sys
from pathlib import Path

from histrel.cli import main

if __name__ == "__main__":
    if len(sys.argv) < 5:
        sys.stderr.write(
            "Usage: python -m histrel "
            "<config_path> <workspace_root> <output_dir> <user_msg>"
        )
        sys.stderr.flush()
        sys.exit(2)

    sys.exit(
        main(
            user_msg=sys.argv[4],
            config_path=Path(sys.argv[1]),
            workspace_root=Path(sys.argv[2]),
            output_dir=Path(sys.argv[3]),
        )
    )
