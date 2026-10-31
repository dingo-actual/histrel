#!/usr/bin/env python3
"""Run the agent with the same arguments as the ``histrel`` command."""

import sys

from histrel.cli import run

if __name__ == "__main__":
    sys.exit(run(prog="python -m histrel"))
