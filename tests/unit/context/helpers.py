#!/usr/bin/env python3
"""Shared helpers for the context unit tests.

Important Classes:
    CompactorHarness: Compactor wired to a stubbed adapter and a ledger.
    ArtifactHarness: SessionArtifacts writer bound to one directory.

Important Functions:
    make_message(factory, role, text) -> Message: Build a single-part
        message.
    make_tool_result_message(factory, ...) -> Message: Build a tool-role
        message holding one tool result part.

Important Variables:
    SYSTEM_PROMPT (str): Protected-head system prompt used by the tests.
    TASK_TEXT (str): Protected-head task text used by the tests.
    TEST_PRICING (ModelPricing): Pricing used by the cost and artifact
        tests.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from histrel.context.compactor import CompactedState, Compactor
from histrel.context.cost_calculator import (
    CostCalculator,
    ModelPricing,
)
from histrel.context.session_artifacts import SessionArtifacts
from histrel.context.session_events import SessionEvents
from histrel.context.timings import Timings
from histrel.context.token_ledger import TokenLedger
from histrel.context.tool_metrics import ToolMetrics
from histrel.core.config import AgentConfig
from histrel.model.adapter_openai_compat import OpenAICompatAdapter
from histrel.model.messages import (
    Message,
    MessageFactory,
    MessageMetadata,
    MessageRole,
    TextPart,
    ToolResultPart,
)
from histrel.model.types import ModelRequest, ModelResponse, Usage
from tests.unit.factories import build_agent_config

SYSTEM_PROMPT = "You are a coding agent."
TASK_TEXT = "task"

TEST_PRICING = ModelPricing(
    uncached_input_per_1m=3.0,
    cached_input_per_1m=0.3,
    output_per_1m=8.0,
)

_COMPACTION_REQUEST_ID = "req-compaction"
_DEFAULT_USAGE = Usage(input=100, output=50, total=150)


def make_message(factory: MessageFactory, role: MessageRole, text: str) -> Message:
    """Build a message holding one text part.

    Args:
        factory (MessageFactory): Factory creating the message.
        role (MessageRole): Message role.
        text (str): Text content of the single part.

    Returns:
        (Message) Message with a single text part.
    """
    return factory.create(
        role=role,
        content=[TextPart(text=text)],
        metadata=MessageMetadata(),
    )


def make_tool_result_message(
    factory: MessageFactory,
    *,
    tool_id: str,
    tool_name: str,
    arguments: dict[str, Any],
    output: str,
) -> Message:
    """Build a tool-role message holding one successful tool result part.

    Args:
        factory (MessageFactory): Factory creating the message.
        tool_id (str): Tool call identifier.
        tool_name (str): Name of the tool that produced the result.
        arguments (dict[str, Any]): Arguments the tool was called with.
        output (str): Tool output text.

    Returns:
        (Message) Tool-role message with a single tool result part.
    """
    return factory.create(
        role="tool",
        content=[
            ToolResultPart(
                tool_id=tool_id,
                tool_name=tool_name,
                arguments=arguments,
                success=True,
                output=output,
                iter_num=1,
            )
        ],
        metadata=MessageMetadata(),
    )


class CompactorHarness:
    """Compactor wired to a stubbed adapter and an empty token ledger.

    Instance Attributes:
        compactor (Compactor): Compactor under test.
        adapter (OpenAICompatAdapter): Adapter whose complete is patched.
        ledger (TokenLedger): Ledger receiving compaction usage.
        requests (list[ModelRequest]): Requests seen by the adapter.
    """

    def __init__(
        self,
        config: AgentConfig,
        patch_complete: Callable[
            [Callable[[ModelRequest], ModelResponse]], OpenAICompatAdapter
        ],
        *,
        summary: str = "Summary.",
        usage: Usage | None = None,
        on_compaction: Callable[[CompactedState], None] | None = None,
    ) -> None:
        """Initialize the harness.

        Args:
            config (AgentConfig): Compactor configuration.
            patch_complete (Callable): Factory that patches an adapter's
                complete method with this harness's stub.
            summary (str): Summary text the stub returns.
            usage (Usage | None): Usage the stub returns.
            on_compaction (Callable | None): Compaction event callback.
        """
        self.compactor = Compactor(config, on_compaction=on_compaction)
        self.ledger = TokenLedger()
        self.requests: list[ModelRequest] = []
        self._summary = summary
        self._usage = _DEFAULT_USAGE if usage is None else usage
        self.adapter = patch_complete(self.complete)

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Record the request and return the fixed summary response."""
        self.requests.append(request)
        return ModelResponse(
            text_parts=(self._summary,),
            tool_calls=(),
            usage=self._usage,
            provider_request_id=_COMPACTION_REQUEST_ID,
            finish_reason="stop",
        )

    @property
    def call_count(self) -> int:
        """Number of model calls the stub adapter received."""
        return len(self.requests)

    def compact(
        self,
        messages: list[Message],
        *,
        system_prompt: str = SYSTEM_PROMPT,
        original_task: str = TASK_TEXT,
        recent_tail_count: int = 1,
    ) -> CompactedState:
        """Compact through the stubbed adapter and the harness ledger."""
        return self.compactor.compact(
            messages=messages,
            system_prompt=system_prompt,
            original_task=original_task,
            recent_tail_count=recent_tail_count,
            model_adapter=self.adapter,
            token_ledger=self.ledger,
        )


@dataclass
class ArtifactHarness:
    """SessionArtifacts writer bound to one artifact directory.

    Instance Attributes:
        base_dir (Path): Artifact directory.
        artifacts (SessionArtifacts): Writer under test.
    """

    base_dir: Path
    artifacts: SessionArtifacts = field(init=False)

    def __post_init__(self) -> None:
        """Create the writer for ``base_dir``."""
        self.artifacts = SessionArtifacts(base_dir=self.base_dir)

    def write(
        self,
        *,
        ledger: TokenLedger | None = None,
        config: AgentConfig | None = None,
        timings: Timings | None = None,
        tool_metrics: ToolMetrics | None = None,
        outcome: dict[str, Any] | None = None,
        manifest_resolved: dict[str, Any] | None = None,
        messages_jsonl: str = "",
        diff_text: str = "",
        session_events: SessionEvents | None = None,
        source_context: dict[str, Any] | None = None,
    ) -> None:
        """Write all artifacts, defaulting the unspecified inputs."""
        self.artifacts.write_artifacts(
            config=build_agent_config() if config is None else config,
            timings=Timings() if timings is None else timings,
            tool_metrics=ToolMetrics() if tool_metrics is None else tool_metrics,
            token_ledger=TokenLedger() if ledger is None else ledger,
            cost_calculator=CostCalculator(pricing=TEST_PRICING),
            outcome={"status": "completed"} if outcome is None else outcome,
            manifest_resolved={} if manifest_resolved is None else manifest_resolved,
            messages_jsonl=messages_jsonl,
            diff_text=diff_text,
            session_events=SessionEvents()
            if session_events is None
            else session_events,
            source_context={} if source_context is None else source_context,
        )

    def exists(self, name: str) -> bool:
        """Report whether one artifact file was written."""
        return (self.base_dir / name).exists()

    def read_text(self, name: str) -> str:
        """Return the text of one written artifact."""
        return (self.base_dir / name).read_text(encoding="utf-8")

    def read_json(self, name: str) -> Any:
        """Return the parsed JSON of one written artifact."""
        return json.loads(self.read_text(name))
