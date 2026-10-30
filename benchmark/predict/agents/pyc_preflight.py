#!/usr/bin/env python3
"""
Preflight integrity checks for the agent's cached Python bytecode.

The benchmark host has produced ``.pyc`` files whose 16-byte headers are
valid but whose payloads are corrupt (invalid marshal data, or garbage
bytecode that executes wrong instructions). CPython trusts the header, so
a single corrupt file aborts the agent at import time or during its first
actions. These helpers detect and remove such files before the agent
runs; CPython then recompiles the modules from source.

    Important Functions:
        find_invalid_pycs(env_root): List bytecode files that fail to
            load, i.e. files CPython cannot safely execute.
        repair_pyc_cache(env_root): Delete unloadable bytecode files
            that have a matching source file, so CPython recompiles
            them; returns the list of deleted files.
        agent_died_before_init(result): Detect an agent subprocess crash
            that happened before session initialization completed
            (the signature of bytecode corruption at startup).
        wipe_pycache_dirs(env_root): Remove all ``__pycache__``
            directories under the environment root; used for the one
            recovery retry, covering corrupt-but-loadable bytecode.

    Important Variables:
        DEFAULT_AGENT_ENV_ROOT (Path): Default conda env root used by
            the benchmark agent inside the SWE-bench containers.
        AGENT_INIT_LOG_MARKER (str): Log event the agent emits once its
            runtime is initialized.
"""

from __future__ import annotations

import marshal
import os
import shutil
import subprocess
from importlib.util import MAGIC_NUMBER
from pathlib import Path

from loguru import logger

# Agent conda env root inside the SWE-bench environment image.
DEFAULT_AGENT_ENV_ROOT = Path("/opt/miniconda3/envs/agent")

# The agent runtime logs this event name once initialization succeeds
# (see Runtime.initialize in histrel.core.runtime).
AGENT_INIT_LOG_MARKER = "agent_session_initialized"

# Fixed size of the .pyc header: magic (4) + flags (4) + mtime (4) + size (4).
_PYC_HEADER_SIZE = 16

_COMPONENT = "benchmark.predict.agents.pyc_preflight"


def _pyc_source_path(pyc_path: Path) -> Path:
    """Return the source (.py) path a .pyc file was compiled from.

    Args:
        pyc_path (Path): Path of the .pyc file.

    Returns:
        (Path) Expected source path: sibling ``mod.py`` for legacy pycs
            stored next to sources, or ``pkg/mod.py`` for pycs stored in
            a ``__pycache__`` directory.
    """
    stem = pyc_path.name.split(".")[0]
    if pyc_path.parent.name == "__pycache__":
        return pyc_path.parent.parent / f"{stem}.py"
    return pyc_path.parent / f"{stem}.py"


def _pyc_is_loadable(pyc_path: Path) -> bool:
    """Check that a .pyc file parses as bytecode for this interpreter.

    A file fails when it is truncated, when its magic number does not
    match, or when its marshal payload cannot be decoded. These are
    exactly the failure modes observed on this benchmark host.

    Args:
        pyc_path (Path): Path of the .pyc file.

    Returns:
        (bool) True if the file loads; False otherwise.
    """
    try:
        data = pyc_path.read_bytes()
    except OSError:
        return False
    if len(data) < _PYC_HEADER_SIZE or data[:4] != MAGIC_NUMBER:
        return False
    try:
        marshal.loads(data[_PYC_HEADER_SIZE:])
    except (ValueError, EOFError, TypeError):
        return False
    return True


def find_invalid_pycs(env_root: Path) -> list[Path]:
    """List bytecode files under env_root that fail to load.

    Args:
        env_root (Path): Root directory to scan (a conda env root).

    Returns:
        (list[Path]) Unloadable .pyc files, in walk order.
    """
    if not env_root.is_dir():
        return []
    invalid: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(env_root):
        for name in filenames:
            if not name.endswith(".pyc"):
                continue
            pyc_path = Path(dirpath) / name
            if not _pyc_is_loadable(pyc_path):
                invalid.append(pyc_path)
    return invalid


def repair_pyc_cache(env_root: Path = DEFAULT_AGENT_ENV_ROOT) -> list[Path]:
    """Delete unloadable bytecode files so CPython recompiles them.

    Only files with a matching ``.py`` source are deleted; deleting an
    unloadable .pyc that has no source would break its import
    permanently, so those are reported (warning) and left alone. The
    environment root missing entirely is also reported (warning), which
    happens when the wrapper runs outside the benchmark container.

    Args:
        env_root (Path): Conda env root to repair. Defaults to the
            agent env inside the SWE-bench container.

    Returns:
        (list[Path]) The .pyc files that were deleted.
    """
    if not env_root.is_dir():
        logger.bind(
            component=_COMPONENT,
            event="pyc_cache_repair_skipped",
        ).warning(
            "Pyc cache root does not exist; no repair performed",
            env_root=str(env_root),
        )
        return []

    deleted: list[Path] = []
    for pyc_path in find_invalid_pycs(env_root):
        if _pyc_source_path(pyc_path).exists():
            pyc_path.unlink()
            deleted.append(pyc_path)
            logger.bind(
                component=_COMPONENT,
                event="pyc_cache_repaired",
            ).warning(
                "Deleted corrupt bytecode file; CPython will recompile from source",
                pyc_path=str(pyc_path),
            )
        else:
            logger.bind(
                component=_COMPONENT,
                event="pyc_cache_unrepairable",
            ).warning(
                "Corrupt bytecode file has no source; left untouched",
                pyc_path=str(pyc_path),
            )

    if deleted:
        logger.bind(
            component=_COMPONENT,
            event="pyc_cache_repair_complete",
        ).info(
            "Pyc cache repair removed corrupt bytecode files",
            env_root=str(env_root),
            repaired_count=len(deleted),
            repaired_files=[str(path) for path in deleted],
        )
    return deleted


def agent_died_before_init(result: subprocess.CompletedProcess) -> bool:
    """Check whether the agent crashed before initialization completed.

    Corruption crashes (invalid bytecode at import time or during the
    first repository snapshot) happen before the agent logs its
    initialization event. Timeouts (returncode 124 by convention in the
    wrapper) are excluded: a session that ran long either initialized or
    failed for reasons that a bytecode wipe cannot address.

    Args:
        result (subprocess.CompletedProcess): Finished agent process.

    Returns:
        (bool) True if the process failed without reaching
            initialization.
    """
    if result.returncode in (0, 124):
        return False
    combined = f"{result.stdout or ''}{result.stderr or ''}"
    return AGENT_INIT_LOG_MARKER not in combined


def wipe_pycache_dirs(env_root: Path = DEFAULT_AGENT_ENV_ROOT) -> int:
    """Remove every ``__pycache__`` directory under env_root.

    Used by the recovery retry after a pre-initialization crash. This
    also discards corrupt-but-loadable bytecode, which
    ``repair_pyc_cache`` cannot detect; CPython recompiles everything it
    imports afterwards. In the benchmark container the deletions land in
    the container's writable layer, so the image itself is unaffected.

    Args:
        env_root (Path): Conda env root to clear. Defaults to the agent
            env inside the SWE-bench container.

    Returns:
        (int) Number of ``__pycache__`` directories removed.
    """
    if not env_root.is_dir():
        logger.bind(
            component=_COMPONENT,
            event="pyc_cache_wipe_skipped",
        ).warning(
            "Pyc cache root does not exist; no wipe performed",
            env_root=str(env_root),
        )
        return 0

    removed = 0
    for cache_dir in sorted(env_root.rglob("__pycache__")):
        shutil.rmtree(cache_dir, ignore_errors=True)
        removed += 1

    logger.bind(
        component=_COMPONENT,
        event="pyc_cache_wipe_complete",
    ).warning(
        "Wiped bytecode cache directories for recovery retry",
        env_root=str(env_root),
        wiped_count=removed,
    )
    return removed
