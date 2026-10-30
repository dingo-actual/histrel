#!/usr/bin/env python3
"""Unit tests for HTTP cancellation functionality.

Tests verify that prediction runs can be cancelled mid-execution via HTTP POST
requests, with graceful container shutdown and proper cleanup actions.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
from aiohttp.test_utils import make_mocked_request
from aiohttp.web import Application

# Skip entire module on Windows before any imports that require swebench
if sys.platform == "win32":
    pytest.skip("Scorer requires Unix", allow_module_level=True)

from benchmark.common.docker_utils import run_container
from benchmark.predict.config import PredictorConfig
from benchmark.predict.docker.runner import DockerRunner
from benchmark.predict.runner import (
    CANCELLATION_EVENT_KEY,
    RUN_NAME_KEY,
    STOP_CONTAINER_CALLBACK_KEY,
    _cancellation_handler,
    _start_cancellation_server,
)
from benchmark.tests.fixtures.docker_stubs import DockerClientStub


class Test04Unit05Cancellation01Handler01Contract:
    """Contract tests for cancellation HTTP handler."""

    @pytest.mark.asyncio
    async def test_04_unit_05_cancellation_01_handler_01_contract_01_sets_event(
        self,
    ) -> None:
        """Test that handler sets cancellation event when called."""
        # Arrange
        cancellation_event = asyncio.Event()

        # Create real aiohttp app and request
        app = Application()
        app[CANCELLATION_EVENT_KEY] = cancellation_event
        app[RUN_NAME_KEY] = "test-run-id"

        # Create a test request with the app
        request = make_mocked_request("POST", "/cancel", app=app)

        # Act
        response = await _cancellation_handler(request)

        # Assert
        assert cancellation_event.is_set()
        assert response.status == 200

    @pytest.mark.asyncio
    async def test_04_unit_05_cancellation_01_handler_01_contract_02_returns_json(
        self,
    ) -> None:
        """Test that handler returns JSON response with run_name."""
        # Arrange
        cancellation_event = asyncio.Event()

        app = Application()
        app[CANCELLATION_EVENT_KEY] = cancellation_event
        app[RUN_NAME_KEY] = "test-run-123"

        request = make_mocked_request("POST", "/cancel", app=app)

        # Act
        response = await _cancellation_handler(request)

        # Assert
        assert response.status == 200
        assert response.content_type == "application/json"
        # Verify the body contains expected data
        response_text = response.text
        assert response_text is not None
        body = json.loads(response_text)
        assert body["status"] == "cancelled"
        assert body["run_name"] == "test-run-123"

    @pytest.mark.asyncio
    async def test_04_unit_05_cancellation_01_handler_01_contract_03_invokes_callback(
        self,
    ) -> None:
        """Test that handler invokes stop_container_callback if present."""
        # Arrange
        cancellation_event = asyncio.Event()
        callback_invoked = False

        async def real_callback() -> None:
            nonlocal callback_invoked
            callback_invoked = True

        app = Application()
        app[CANCELLATION_EVENT_KEY] = cancellation_event
        app[RUN_NAME_KEY] = "test-run"
        app[STOP_CONTAINER_CALLBACK_KEY] = real_callback

        request = make_mocked_request("POST", "/cancel", app=app)

        # Act
        response = await _cancellation_handler(request)

        # Assert
        assert cancellation_event.is_set()
        assert callback_invoked
        assert response.status == 200


class Test04Unit05Cancellation02Server01Isolation:
    """Isolation tests for cancellation server startup."""

    @pytest.mark.asyncio
    async def test_04_unit_05_cancellation_02_server_01_isolation_01_starts_cleanly(
        self,
    ) -> None:
        """Test that cancellation server starts without external dependencies."""
        # Arrange
        cancellation_event = asyncio.Event()
        host = "127.0.0.1"
        port = 0  # Use port 0 to get random available port
        run_name = "test-run"

        # Act
        runner = await _start_cancellation_server(
            host, port, cancellation_event, run_name
        )

        try:
            # Assert
            assert runner is not None
            assert runner.app[CANCELLATION_EVENT_KEY] is cancellation_event
            assert runner.app[RUN_NAME_KEY] == run_name
        finally:
            # Cleanup
            await runner.cleanup()

    @pytest.mark.asyncio
    async def test_04_unit_05_cancellation_02_server_01_isolation_02_passes_callback(
        self,
    ) -> None:
        """Test that server accepts and stores stop_container_callback."""
        # Arrange
        cancellation_event = asyncio.Event()

        async def real_callback() -> None:
            pass

        host = "127.0.0.1"
        port = 0
        run_name = "test-run"

        # Act
        runner = await _start_cancellation_server(
            host, port, cancellation_event, run_name, real_callback
        )

        try:
            # Assert
            assert runner.app[STOP_CONTAINER_CALLBACK_KEY] is real_callback
        finally:
            await runner.cleanup()


class Test04Unit05Cancellation03Event01Temporal:
    """Temporal tests for cancellation event state transitions."""

    @pytest.mark.asyncio
    async def test_04_unit_05_cancellation_03_event_01_temporal_01_initial_state(
        self,
    ) -> None:
        """Test that cancellation event starts unset."""
        # Arrange & Act
        cancellation_event = asyncio.Event()

        # Assert
        assert not cancellation_event.is_set()

    @pytest.mark.asyncio
    async def test_04_unit_05_cancellation_03_event_01_temporal_02_state_transition(
        self,
    ) -> None:
        """Test that event transitions from unset to set."""
        # Arrange
        cancellation_event = asyncio.Event()
        assert not cancellation_event.is_set()

        # Act
        cancellation_event.set()

        # Assert
        assert cancellation_event.is_set()

    @pytest.mark.asyncio
    async def test_04_unit_05_cancellation_03_event_01_temporal_03_persistent_state(
        self,
    ) -> None:
        """Test that event remains set after being triggered."""
        # Arrange
        cancellation_event = asyncio.Event()

        # Act
        cancellation_event.set()
        first_check = cancellation_event.is_set()
        second_check = cancellation_event.is_set()

        # Assert
        assert first_check
        assert second_check


class Test04Unit05Cancellation04Callback01Morphism:
    """Morphism tests for cancellation callback behavior."""

    @pytest.mark.asyncio
    async def test_04_unit_05_cancellation_04_callback_01_morphism_01_callable(
        self,
    ) -> None:
        """Test that cancellation callback is callable."""
        # Arrange
        cancellation_event = asyncio.Event()
        callback = cancellation_event.is_set

        # Act
        result = callback()

        # Assert
        assert callable(callback)
        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_04_unit_05_cancellation_04_callback_01_morphism_02_returns_false(
        self,
    ) -> None:
        """Test that callback returns False when event is not set."""
        # Arrange
        cancellation_event = asyncio.Event()
        callback = cancellation_event.is_set

        # Act
        result = callback()

        # Assert
        assert result is False

    @pytest.mark.asyncio
    async def test_04_unit_05_cancellation_04_callback_01_morphism_03_returns_true(
        self,
    ) -> None:
        """Test that callback returns True when event is set."""
        # Arrange
        cancellation_event = asyncio.Event()
        cancellation_event.set()
        callback = cancellation_event.is_set

        # Act
        result = callback()

        # Assert
        assert result is True


class Test04Unit05Cancellation05Container01FailureMode:
    """Failure mode tests for container cancellation."""

    def test_04_unit_05_cancellation_05_container_01_failure_mode_01_stops_container(
        self, docker_client_stub: DockerClientStub
    ) -> None:
        """Test that run_container stops container when callback returns True."""

        # Arrange - callback that immediately returns True
        def callback() -> bool:
            return True

        # Act
        exit_code, container_id = run_container(
            docker_client_stub,
            image="test-image:latest",
            command=["sleep", "1000"],
            cancellation_callback=callback,
            timeout=10,
        )

        # Assert - exit code 130 indicates cancellation
        assert exit_code == 130

    def test_04_unit_05_cancellation_05_container_01_failure_mode_02_no_callback(
        self, docker_client_stub: DockerClientStub
    ) -> None:
        """Test that run_container works normally without cancellation callback."""
        # Arrange
        docker_client_stub.containers.set_next(status=0, output="success")

        # Act
        exit_code, container_id = run_container(
            docker_client_stub,
            image="test-image:latest",
            command=["echo", "hello"],
            cancellation_callback=None,
        )

        # Assert - should work normally with exit code 0
        assert exit_code == 0

    def test_04_unit_05_cancellation_05_container_01_failure_mode_03_invokes_started(
        self, docker_client_stub: DockerClientStub
    ) -> None:
        """Test that run_container invokes container_started_callback."""
        # Arrange
        callback_invoked_with = None

        def started_callback(cid: str) -> None:
            nonlocal callback_invoked_with
            callback_invoked_with = cid

        docker_client_stub.containers.set_next(status=0, output="success")

        # Act
        exit_code, container_id = run_container(
            docker_client_stub,
            image="test-image:latest",
            command=["echo", "hello"],
            container_started_callback=started_callback,
        )

        # Assert - callback should be invoked with container ID
        assert callback_invoked_with is not None
        assert callback_invoked_with == container_id

    def test_04_unit_05_cancellation_05_container_01_failure_mode_04_preserves_nonzero_exit(
        self, docker_client_stub: DockerClientStub
    ) -> None:
        """Test that run_container does not mask exit code 1 as 124 (timeout).

        When a container exits with a non-zero code (e.g., the entrypoint
        returns 1) while a timeout is configured, the real exit code must
        be preserved.  Previously the code treated ``timeout and exit_code
        == 1`` as a timeout, masking application failures as timeouts.
        """
        docker_client_stub.containers.set_next(status=1, output="entrypoint error")

        def callback() -> bool:
            return False

        exit_code, _ = run_container(
            docker_client_stub,
            image="test-image:latest",
            command=["python", "/workspace/entrypoint.py"],
            cancellation_callback=callback,
            timeout=7200,
        )

        assert exit_code == 1


class Test04Unit05Cancellation06DockerRunner01Contract:
    """Contract tests for DockerRunner cancellation support."""

    def test_04_unit_05_cancellation_06_docker_runner_01_contract_01_accepts_callback(
        self, docker_client_stub: DockerClientStub, tmp_path: Path
    ) -> None:
        """Test that run_agent accepts cancellation_callback parameter."""
        # Arrange
        cfg = PredictorConfig(output_dir=str(tmp_path / "results"), run_name="test-run")
        runner = DockerRunner(docker_client_stub)

        instance = {
            "instance_id": "test-instance",
            "repo": "test/repo",
            "problem_statement": "test problem",
        }

        def callback() -> bool:
            return False

        docker_client_stub.containers.set_next(status=0, output="patch generated")

        # Create necessary directories
        output_dir = tmp_path / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        tmp_instances = output_dir / "tmp_instances"
        tmp_instances.mkdir(parents=True, exist_ok=True)

        # Write usage file
        (output_dir / "bench-agent_usage.json").write_text(
            json.dumps({"tokens_input_total": 100, "tokens_output_total": 50})
        )

        # Act
        patch_path, token_usage, exit_code = runner.run_agent(
            instance,
            image="test-image:latest",
            config=cfg,
            output_dir=output_dir,
            cancellation_callback=callback,
        )

        # Assert - should execute without error
        assert patch_path is not None
        assert exit_code == 0

    def test_04_unit_05_cancellation_06_docker_runner_01_contract_02_accepts_started(
        self, docker_client_stub: DockerClientStub, tmp_path: Path
    ) -> None:
        """Test that run_agent accepts container_started_callback parameter."""
        # Arrange
        cfg = PredictorConfig(output_dir=str(tmp_path / "results"), run_name="test-run")
        runner = DockerRunner(docker_client_stub)

        instance = {
            "instance_id": "test-instance",
            "repo": "test/repo",
            "problem_statement": "test problem",
        }

        callback_invoked = False

        def started_callback(cid: str) -> None:
            nonlocal callback_invoked
            callback_invoked = True

        docker_client_stub.containers.set_next(status=0, output="patch generated")

        # Create necessary directories
        output_dir = tmp_path / "artifacts"
        output_dir.mkdir(parents=True, exist_ok=True)
        tmp_instances = output_dir / "tmp_instances"
        tmp_instances.mkdir(parents=True, exist_ok=True)

        # Write usage file
        (output_dir / "bench-agent_usage.json").write_text(
            json.dumps({"tokens_input_total": 100, "tokens_output_total": 50})
        )

        # Act
        patch_path, token_usage, exit_code = runner.run_agent(
            instance,
            image="test-image:latest",
            config=cfg,
            output_dir=output_dir,
            container_started_callback=started_callback,
        )

        # Assert - callback should have been invoked
        assert callback_invoked


class Test04Unit05Cancellation07Integration01Boundary:
    """Boundary tests for cancellation edge cases."""

    @pytest.mark.asyncio
    async def test_04_unit_05_cancellation_07_integration_01_boundary_01_immediate(
        self,
    ) -> None:
        """Test cancellation that occurs immediately after event creation."""
        # Arrange
        cancellation_event = asyncio.Event()

        # Act - set event immediately
        cancellation_event.set()

        # Assert - callback should immediately return True
        callback = cancellation_event.is_set
        assert callback() is True

    @pytest.mark.asyncio
    async def test_04_unit_05_cancellation_07_integration_01_boundary_02_none_callback(
        self, docker_client_stub: DockerClientStub
    ) -> None:
        """Test that None callback is handled gracefully."""
        # Arrange
        docker_client_stub.containers.set_next(status=0, output="success")

        # Act - pass None as callback
        exit_code, container_id = run_container(
            docker_client_stub,
            image="test-image:latest",
            command=["echo", "test"],
            cancellation_callback=None,
            container_started_callback=None,
        )

        # Assert - should work normally
        assert exit_code == 0

    @pytest.mark.asyncio
    async def test_04_unit_05_cancellation_07_integration_01_boundary_03_empty_app(
        self,
    ) -> None:
        """Test handler with missing stop_container_callback."""
        # Arrange
        cancellation_event = asyncio.Event()

        app = Application()
        app[CANCELLATION_EVENT_KEY] = cancellation_event
        app[RUN_NAME_KEY] = "test-run"
        # Deliberately omit stop_container_callback

        request = make_mocked_request("POST", "/cancel", app=app)

        # Act - should not raise
        response = await _cancellation_handler(request)

        # Assert
        assert cancellation_event.is_set()
        assert response.status == 200
