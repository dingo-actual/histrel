#!/usr/bin/env python3
"""
Session setup and message committing for the benchmark runtime.

Wires a Runtime's subsystems at construction time, performs one-time
pre-loop session initialization (tools, task message), and commits
messages to the journal.

    Important Functions:
        wire_session_context(runtime) -> None: Session context.
        wire_subsystems(runtime) -> None: Subsystems and guards.
        wire_adapters(runtime) -> None: Model adapter.
        initialize_session(runtime, user_msg) -> None: Pre-loop init.
        commit_message(runtime, message) -> None: Commit one message.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from histrel.context.compactor import Compactor
from histrel.context.session_events import SessionEvents
from histrel.context.timings import Timings
from histrel.context.token_ledger import TokenLedger
from histrel.context.tool_metrics import ToolMetrics
from histrel.core.clock import monotonic_now
from histrel.core.duplicate_guard import DuplicateCallGuard
from histrel.core.ids import SessionId
from histrel.core.journal import MessageJournal
from histrel.core.runtime.provider_context import (
    TOOL_CALL_REQUIRED_REMINDER,
)
from histrel.core.session_context import ModelInfo, SessionContext
from histrel.model.adapter_openai_compat import (
    OpenAICompatAdapter,
    build_client,
)
from histrel.model.messages import (
    Message,
    MessageFactory,
    MessageMetadata,
    TextPart,
)
from histrel.repo_state import RepoStateChecker
from histrel.terminal import TerminalEvaluator
from histrel.tools.registry import build_all_tools

if TYPE_CHECKING:
    from histrel.core.runtime.runtime import Runtime

log = logger.bind(component="bench_agent.runtime")


def wire_session_context(runtime: Runtime) -> None:
    """Create the session context, message factory, and journal.

    Args:
        runtime (Runtime): The runtime being constructed.
    """
    runtime.session = SessionContext(
        session_id=runtime.config.session_id,
        task_id=runtime.config.task_id,
        model=ModelInfo(
            provider=runtime.config.model_provider,
            name=runtime.config.model_name,
        ),
    )
    runtime.message_factory = MessageFactory(SessionId(runtime.config.session_id))
    artifacts_dir = runtime.output_dir / runtime.config.session_id / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    runtime.journal = MessageJournal(artifacts_dir / "messages.jsonl")


def wire_subsystems(runtime: Runtime) -> None:
    """Create the subsystems: ledger, timings, metrics, guards.

    Args:
        runtime (Runtime): The runtime being constructed.
    """
    runtime.token_ledger = TokenLedger()
    runtime.timings = Timings()
    runtime.tool_metrics = ToolMetrics()
    runtime.duplicate_guard = DuplicateCallGuard(
        window=runtime.config.duplicate_call_window
    )
    runtime.repo_checker = RepoStateChecker(runtime.workspace_root)
    runtime.session_events = SessionEvents()
    runtime.compactor = Compactor(
        runtime.config,
        on_compaction=runtime.session_events.record_compaction,
    )
    runtime.terminal_evaluator = TerminalEvaluator(runtime.config)


def wire_adapters(runtime: Runtime) -> None:
    """Create the model adapter.

    Args:
        runtime (Runtime): The runtime being constructed.
    """
    runtime.model_adapter = OpenAICompatAdapter(
        client=build_client(runtime.config),
        model_name=runtime.config.model_name,
        provider=runtime.config.model_provider,
    )


def initialize_session(runtime: Runtime, user_msg: str) -> None:
    """Perform pre-loop initialization.

    Builds tools and commits the task message. Idempotent — calling it more
    than once has no effect.

    Args:
        runtime (Runtime): The runtime session state.
        user_msg (str): User request message for the session.
    """
    if runtime._initialized:
        return
    runtime._initialized = True
    runtime.session_start = monotonic_now()

    # Build tools
    _build_tools(runtime)

    # Commit the task message (first user message)
    _commit_task_message(runtime, user_msg)

    log.info(
        "agent_session_initialized",
        session_id=runtime.config.session_id,
        task_id=runtime.config.task_id,
        model=runtime.config.model_name,
        workspace_root=str(runtime.workspace_root),
    )


def commit_message(runtime: Runtime, message: Message) -> None:
    """Commit a message to the journal.

    Args:
        runtime (Runtime): The runtime session state.
        message (Message): The message to commit.
    """
    runtime.journal.append(message)


def arm_tool_call_reminder(runtime: Runtime) -> None:
    """Remind the next iteration that a tool call is required.

    React mode replays the conversation, so the reminder is committed as
    a user message; stateless mode renders it into the next step prompt
    from the runtime flag.

    Args:
        runtime (Runtime): The runtime session state.
    """
    runtime.tool_call_required = True
    if runtime.config.mode != "react":
        return
    commit_message(
        runtime,
        runtime.message_factory.create(
            role="user",
            content=[TextPart(text=TOOL_CALL_REQUIRED_REMINDER)],
            metadata=MessageMetadata(),
        ),
    )


def _commit_task_message(runtime: Runtime, user_msg: str) -> None:
    """Commit the initial task message to the journal.

    Args:
        runtime (Runtime): The runtime session state.
        user_msg (str): User request message for the session.
    """
    # The task is the instance ID — in a real deployment it would
    # come from the harness via environment variables.
    msg = runtime.message_factory.create(
        role="user",
        content=[TextPart(text=user_msg)],
        metadata=MessageMetadata(),
    )
    commit_message(runtime, msg)


def _build_tools(runtime: Runtime) -> None:
    """Build the tool registry.

    Args:
        runtime (Runtime): The runtime session state.
    """
    runtime.tool_registry = build_all_tools()
    runtime.ordinary_tools = list(runtime.tool_registry.all_tools.values())

    log.debug(
        "tool_registry_built",
        tools=sorted(runtime.tool_registry.all_tools),
    )
