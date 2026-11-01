#!/usr/bin/env python3
"""
Agent runtime session state for the benchmark agent.

The Runtime class wires the session's subsystems together and carries
per-session state, delegating behavior to the runtime package's
collaborators: the main loop (react_loop), tool execution
(tool_execution), state effects (effects), session setup and message
committing (session_setup), terminal evaluation (termination),
provider-request construction (provider_context), assistant-message
committing (turn_commits), and session finalization (finalization).

    Important Classes:
        Runtime: Orchestrates the full agent session lifecycle.

    Important Variables:
        log (logger): Bound logger for runtime lifecycle events.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from histrel.context.compactor import CompactedState, Compactor
from histrel.context.session_events import SessionEvents
from histrel.context.timings import Timings
from histrel.context.token_ledger import TokenLedger
from histrel.context.tool_metrics import ToolMetrics
from histrel.core.clock import monotonic_now
from histrel.core.config import AgentConfig
from histrel.core.duplicate_guard import DuplicateCallGuard
from histrel.core.journal import MessageJournal
from histrel.core.runtime.finalization import finalize_session
from histrel.core.runtime.session_setup import (
    commit_message,
    initialize_session,
    wire_adapters,
    wire_session_context,
    wire_subsystems,
)
from histrel.core.runtime.termination import (
    evaluate_terminal_conditions,
)
from histrel.core.runtime.tool_execution import execute_tool_call
from histrel.core.session_context import SessionContext
from histrel.model.adapter_openai_compat import OpenAICompatAdapter
from histrel.model.messages import (
    ErrorInfo,
    Message,
    MessageFactory,
    MessageMetadata,
)
from histrel.model.types import ModelResponse, NormalizedToolCall
from histrel.repo_state import RepoStateChecker
from histrel.terminal import (
    SessionOutcome,
    SessionStatus,
    TerminalEvaluator,
)
from histrel.tools.registry import ToolRegistry
from histrel.tools.tool import Tool, ToolExecution

# Bound logger for runtime lifecycle events.
log = logger.bind(component="bench_agent.runtime")


class Runtime:
    """Orchestrates the full agent session lifecycle.

    Instance Attributes:
        config (AgentConfig): Agent configuration.
        workspace_root (Path): Repository workspace root.
        output_dir (Path): Output directory for artifacts.
        session (SessionContext): Session context.
        message_factory (MessageFactory): Message factory.
        journal (MessageJournal): Append-only journal.
        token_ledger (TokenLedger): Token usage ledger.
        timings (Timings): Wall-clock timing tracker.
        tool_metrics (ToolMetrics): Per-tool metrics.
        duplicate_guard (DuplicateCallGuard): Duplicate tool call guard.
        repo_checker (RepoStateChecker): Repository state checker.
        session_events (SessionEvents): Compaction and source event recorder.
        compactor (Compactor): Context compactor.
        compacted_state (CompactedState | None): Latest compaction, whose
            summary stands in for the journal messages before its
            retained tail.
        terminal_evaluator (TerminalEvaluator): Terminal condition evaluator.
        model_adapter (OpenAICompatAdapter): Model adapter.
        ordinary_tools (list[Tool]): Ordinary tools.
        tool_registry (ToolRegistry | None): Tool registry.
        turn_count (int): Number of model turns completed.
        iteration_count (int): Number of agent loop iterations started.
        tool_call_required (bool): Whether the next turn must carry a tool call.
        tool_call_count (int): Total tool calls executed.
        session_start (float): Monotonic session start time.
    """

    session: SessionContext
    message_factory: MessageFactory
    journal: MessageJournal
    token_ledger: TokenLedger
    timings: Timings
    tool_metrics: ToolMetrics
    duplicate_guard: DuplicateCallGuard
    repo_checker: RepoStateChecker
    session_events: SessionEvents
    compactor: Compactor
    compacted_state: CompactedState | None
    terminal_evaluator: TerminalEvaluator
    model_adapter: OpenAICompatAdapter
    tool_registry: ToolRegistry | None

    def __init__(
        self,
        *,
        config: AgentConfig,
        workspace_root: Path,
        output_dir: Path,
    ) -> None:
        """Initialize the runtime.

        Args:
            config (AgentConfig): Agent configuration.
            workspace_root (Path): Repository workspace root.
            output_dir (Path): Output directory for artifacts.
        """
        self.config = config
        self.workspace_root = workspace_root
        self.output_dir = output_dir
        wire_session_context(self)
        wire_subsystems(self)
        wire_adapters(self)
        self.ordinary_tools: list[Tool] = []
        self.tool_registry: ToolRegistry | None = None
        self.compacted_state = None
        self.turn_count = 0
        self.iteration_count = 0
        self.tool_call_required = False
        self.tool_call_count = 0
        self.session_start = 0.0
        self._initialized = False

    def initialize(self, user_msg: str) -> None:
        """Perform pre-loop initialization (idempotent).

        Initializes the session, builds tools, and commits the task message.
        """
        initialize_session(self, user_msg)

    def _commit_message(self, message: Message) -> None:
        """Commit a message to the journal.

        Args:
            message (Message): The message to commit.
        """
        commit_message(self, message)

    def evaluate_terminal_conditions(self) -> SessionOutcome | None:
        """Evaluate whether the session should terminate.

        Checks budget exhaustion.

        Returns:
            (SessionOutcome | None) Outcome if terminal, None otherwise.
        """
        return evaluate_terminal_conditions(self)

    def _get_tool_by_name(self, name: str) -> Tool | None:
        """Find a tool by name from the session's tools.

        Args:
            name (str): Tool name.

        Returns:
            (Tool | None) The tool, or None if not found.
        """
        return next((tool for tool in self.ordinary_tools if tool.name == name), None)

    def _record_routine_usage(self, response: ModelResponse) -> None:
        """Record one LLM routine call's token usage in the ledger.

        Wired as the ``usage_recorder`` hook of every routine call, so
        input, output, reasoning, and cached tokens flow into the
        session's token ledger and cost report.

        Args:
            response (ModelResponse): The routine call's model response.
        """
        self.token_ledger.record(
            message_index=self.message_factory.next_message_index,
            usage=response.usage,
            provider_request_id=response.provider_request_id,
            call_kind="routine",
        )

    def _execute_tool_call(
        self,
        tool_call: NormalizedToolCall,
    ) -> ToolExecution:
        """Execute a single tool call (delegates to tool_execution)."""
        return execute_tool_call(self, tool_call)

    def handle_model_error(
        self,
        call_id: str,
        error: Exception,
        started: float,
    ) -> SessionOutcome:
        """Handle a model API error.

        Args:
            call_id (str): Model call ID.
            error (Exception): The error that occurred.
            started (float): Monotonic start time.

        Returns:
            (SessionOutcome) Infrastructure error outcome.
        """
        elapsed = monotonic_now() - started
        self.timings.add("model_wall_seconds", elapsed)

        # Commit an error message
        msg = self.message_factory.create(
            role="system",
            content=[],
            error=ErrorInfo(
                type="infrastructure_error",
                message=str(error),
            ),
            metadata=MessageMetadata(),
        )
        self._commit_message(msg)

        return SessionOutcome(
            status=SessionStatus.INFRASTRUCTURE_ERROR,
            reason=str(error),
        )

    def finalize(self, outcome: SessionOutcome) -> SessionOutcome:
        """Finalize the session: write artifacts (delegates to finalization)."""
        return finalize_session(self, outcome)

    @property
    def original_task(self) -> str:
        """Return the original task text."""
        return f"Task: {self.config.task_id}"
