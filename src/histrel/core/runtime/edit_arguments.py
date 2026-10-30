#!/usr/bin/env python3
"""
Edit-tool argument repair policy for the benchmark agent runtime.

``tools/json.py`` fixes the JSON malformations it can, with no knowledge
of the runtime. A structure it cannot fix is offered to the
``fix_json_arguments`` LLM routine, constrained by the edit tool's
schema — that wiring needs the runtime, so it lives here.

    Important Functions:
        repair_edit_arguments(runtime, raw) -> dict[str, Any] | str:
            Repair edit arguments, falling back to the LLM routine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from loguru import logger

from histrel.model.routines import fix_json_arguments
from histrel.tools.json import (
    repair_edit_arguments as _local_repair,
)
from histrel.tools.repository import EditFileTool

if TYPE_CHECKING:
    from histrel.core.runtime.runtime import Runtime

log = logger.bind(component="bench_agent.runtime")


def repair_edit_arguments(runtime: Runtime, raw: str) -> dict[str, Any] | str:
    """Repair edit-tool arguments, falling back to the LLM routine.

    Valid outer JSON can still carry an inner ``file_edits`` string
    mangled beyond local repair, and that case never reaches the LLM
    fallback inside ``repair_and_parse``, so a structure the local
    repairs reject is offered to the routine here.

    Args:
        runtime (Runtime): The runtime session state.
        raw (str): The model's raw edit arguments string.

    Returns:
        (dict[str, Any] | str) The repaired arguments dict, or a
        descriptive error string for the unrepairable defects.
    """
    result = _local_repair(raw)
    if isinstance(result, dict):
        return result

    fixed = _fix_with_routine(runtime, raw)
    retried = _local_repair(fixed) if fixed else result
    # Unusable routine output keeps the original structural error, so the
    # committed result names the defect the model actually produced.
    return retried if isinstance(retried, dict) else result


def _fix_with_routine(runtime: Runtime, raw: str) -> str:
    """Run fix_json_arguments for the edit tool, containing its failures.

    Args:
        runtime (Runtime): The runtime session state.
        raw (str): The model's raw edit arguments string.

    Returns:
        (str) The routine's reconstructed JSON, or "" when it failed.
    """
    try:
        return fix_json_arguments(
            raw,
            tool_definition=EditFileTool.definition,
            adapter=runtime.model_adapter,
            config=runtime.config,
            usage_recorder=runtime._record_routine_usage,
        )
    except Exception as error:
        log.warning("Edit-argument repair routine failed", error=str(error))
        return ""
