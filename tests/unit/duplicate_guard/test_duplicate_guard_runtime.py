#!/usr/bin/env python3
"""Runtime integration tests: duplicate calls are skipped,
not executed, and the guard window comes from the config."""

from __future__ import annotations

from histrel.core.runtime import Runtime, run_agent
from histrel.model.types import (
    ModelRequest,
    ModelResponse,
)
from histrel.terminal import SessionStatus
from tests.unit.duplicate_guard.helpers import (
    USER_MSG,
    assert_duplicate_skipped,
    make_fake_response,
    make_runtime,
    read_call,
    tool_results,
)
from tests.unit.factories import (
    build_agent_config,
    init_fixture_repo,
)


class Test04Unit37Runtime37DuplicateSkips:
    """Runtime integration: duplicate calls are skipped, not executed."""

    def test_04_unit_37_dupskips_01_duplicate_read_not_executed(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange — the model reads the same range in two consecutive
        # turns, then produces a plain-text wrap-up that leaves the repo
        # unchanged, so the turn budget ends the session.
        config = build_agent_config(max_turns=3)
        runtime, adapter = make_runtime(monkeypatch, tmp_path, config)

        call_count = 0

        def fake_complete(request: ModelRequest) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return make_fake_response(tool_calls=(read_call("call_1"),))
            if call_count == 2:
                return make_fake_response(tool_calls=(read_call("call_2"),))
            return make_fake_response(text="I reused the earlier read result.")

        monkeypatch.setattr(adapter, "complete", fake_complete)

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert — the session ends when the turn budget is exhausted;
        # only the first read executed.
        assert outcome.status == SessionStatus.BUDGET_EXHAUSTED
        assert "max_turns (3) exceeded" in outcome.reason

        # The first read executed; the duplicate was skipped and
        # committed only as a non-informative audit result
        assert_duplicate_skipped(tool_results(runtime, "read"))

        read_metrics = runtime.tool_metrics.summary().get("read")
        assert read_metrics is not None
        assert read_metrics["count"] == 1

    def test_04_unit_37_dupskips_03_duplicate_audit_not_observed(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange — turn 2 mixes a duplicate read with a productive
        # read of a different range, so a later request renders turn
        # 2's observations.
        config = build_agent_config(max_turns=3)
        runtime, adapter = make_runtime(monkeypatch, tmp_path, config)
        # An untracked file keeps the repository unchanged, so the
        # session cannot end on the text-only response.
        (tmp_path / "notes.py").write_text("print('hello')\nprint('world')\n")

        observer = []
        call_count = 0

        def fake_complete(request: ModelRequest) -> ModelResponse:
            nonlocal observer
            nonlocal call_count
            call_count += 1
            observer = list(request.messages)
            if call_count == 1:
                return make_fake_response(tool_calls=(read_call("call_1"),))
            if call_count == 2:
                return make_fake_response(
                    tool_calls=(
                        read_call("call_2"),
                        read_call("call_3", 2, path="notes.py"),
                    ),
                )
            return make_fake_response(text="Done reading.")

        monkeypatch.setattr(adapter, "complete", fake_complete)

        # Act
        outcome = run_agent(USER_MSG, runtime)

        # Assert — the turn budget ends the session; the last request
        # renders turn 2's observations without the audit result for
        # the duplicate: only the productive read is shown.
        assert outcome.status == SessionStatus.BUDGET_EXHAUSTED
        user_content = "".join(str(m.content) for m in observer if m.role == "user")
        assert "Duplicate tool call" not in user_content

        results = tool_results(runtime, "read")
        assert_duplicate_skipped(results, count=3)
        assert results[2].success is True


class Test04Unit37Runtime37DuplicateSkips03Window:
    """Runtime passes duplicate_call_window from the config to the guard."""

    def test_04_unit_37_dupskips_03_window_01_guard_uses_config_window(
        self, monkeypatch, tmp_path
    ) -> None:
        # Arrange
        config = build_agent_config(duplicate_call_window=0)
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")
        init_fixture_repo(tmp_path)
        monkeypatch.chdir(tmp_path)
        runtime = Runtime(
            config=config, workspace_root=tmp_path, output_dir=tmp_path / "output"
        )

        # Act
        window = runtime.duplicate_guard.window

        # Assert
        assert window == 0
