#!/usr/bin/env python3
"""Calculator fixture repository for the integration tests.

Important Functions:
    init_calculator_repo(path) -> None: Build the fixture repository with
        a defect in the add function.

Important Variables:
    CALCULATOR_FILES (dict[str, str]): Fixture repo file contents.
    FIXED_CALCULATOR_SOURCE (str): Corrected module, used to simulate the
        agent's fix.
"""

from __future__ import annotations

from pathlib import Path

from dulwich.porcelain import add, commit
from dulwich.repo import Repo

_CALCULATOR_SOURCE = (
    '"""Simple calculator module with a bug."""\n'
    "\n"
    "\n"
    "def add(a: int, b: int) -> int:\n"
    '    """Add two numbers. BUG: returns a - b instead of a + b."""\n'
    "    return a - b\n"
    "\n"
    "\n"
    "def multiply(a: int, b: int) -> int:\n"
    '    """Multiply two numbers (correct)."""\n'
    "    return a * b\n"
)

_CALCULATOR_TEST_SOURCE = (
    '"""Failing test for the calculator module."""\n'
    "\n"
    "from calculator import add, multiply\n"
    "\n"
    "\n"
    "def test_add():\n"
    '    """add(2, 3) should return 5, but the bug returns -1."""\n'
    "    assert add(2, 3) == 5\n"
    "\n"
    "\n"
    "def test_multiply():\n"
    '    """Adjacent behavior: multiply is correct."""\n'
    "    assert multiply(2, 3) == 6\n"
)

_CALCULATOR_README = (
    "# Calculator\n\nA simple calculator with a bug in the add function.\n"
)

# Fixture repository contents: a defective module, a focused failing
# test, and a README (adjacent behavior / blast radius).
CALCULATOR_FILES: dict[str, str] = {
    "calculator.py": _CALCULATOR_SOURCE,
    "test_calculator.py": _CALCULATOR_TEST_SOURCE,
    "README.md": _CALCULATOR_README,
}

# Corrected calculator module, used to simulate the agent's fix.
FIXED_CALCULATOR_SOURCE = (
    '"""Simple calculator module."""\n'
    "\n"
    "\n"
    "def add(a: int, b: int) -> int:\n"
    '    """Add two numbers."""\n'
    "    return a + b\n"
    "\n"
    "\n"
    "def multiply(a: int, b: int) -> int:\n"
    '    """Multiply two numbers."""\n'
    "    return a * b\n"
)


def init_calculator_repo(path: Path) -> None:
    """Initialize the calculator fixture repository with a defect.

    The defect requires repository discovery: the agent must find the
    bug in calculator.py and fix it so the test passes.

    Args:
        path (Path): Repository root to initialize.
    """
    Repo.init(str(path))
    for name, content in CALCULATOR_FILES.items():
        (path / name).write_text(content, encoding="utf-8")

    add(str(path), [name.encode() for name in CALCULATOR_FILES])
    commit(
        str(path),
        message=b"initial: calculator with add bug",
        author=b"Test <test@test.com>",
        committer=b"Test <test@test.com>",
    )
