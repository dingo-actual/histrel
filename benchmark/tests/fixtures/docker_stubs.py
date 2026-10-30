#!/usr/bin/env python3
"""Shared Docker client stubs for tests.

These are shared fixtures/fixture factories for tests that need to exercise
benchmark.utils.docker_utils and benchmark.docker.runner without requiring a
real Docker daemon.

They simulate the subset of the docker SDK API that our code uses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

from docker.errors import ImageNotFound

from benchmark.common.docker_utils import ImageLike

# Images API stubs ------------------------------------------------------------


@dataclass
class _ImageStub:
    id: str | None
    tags: list[str] | None


class _ImagesAPI:
    def __init__(self) -> None:
        self._images: list[_ImageStub] = []
        self.build_calls: list[dict[str, Any]] = []

    def list(self) -> list[ImageLike]:
        return cast(list[ImageLike], list(self._images))

    def add(
        self, tag: str, image_id: str | None = "sha256:deadbeefcafebabe"
    ) -> _ImageStub:
        img = _ImageStub(id=image_id, tags=[tag])
        self._images.append(img)
        return img

    def build(
        self,
        *,
        path: str,
        dockerfile: str,
        tag: str,
        buildargs: dict[str, str],
        rm: bool,
    ) -> tuple[ImageLike, Any]:
        self.build_calls.append(
            {
                "path": path,
                "dockerfile": dockerfile,
                "tag": tag,
                "buildargs": buildargs,
                "rm": rm,
            }
        )
        img = self.add(tag, image_id="sha256:builtimageid")
        logs = [
            {"stream": "Step 1/3"},
            {"stream": "Step 2/3"},
            {"stream": "Step 3/3 Done"},
        ]
        return cast(ImageLike, img), logs

    def remove(self, image_tag: str, force: bool = True) -> None:
        # Remove matching images by tag if present
        self._images = [
            img for img in self._images if image_tag not in (img.tags or [])
        ]

    def get(self, name: str) -> _ImageStub:
        for img in self._images:
            if name in (img.tags or []) or name == img.id:
                return img
        raise ImageNotFound(name)


# Containers API stubs --------------------------------------------------------


class _ContainerStub:
    def __init__(self, *, status_code: int = 0, output: str = "") -> None:
        self._status_code = status_code
        self._output = output or "container output"
        self.removed = False
        self.id = "test-container-id"
        self.status = "exited"

    def wait(self, timeout: int | None = None) -> dict[str, int]:
        return {"StatusCode": self._status_code}

    def logs(self, stdout: bool = True, stderr: bool = True) -> bytes:
        return self._output.encode("utf-8")

    def attach(self, *args: Any, **kwargs: Any) -> Any:
        return iter([])

    def remove(self, force: bool = True) -> None:
        self.removed = True

    def reload(self) -> None:
        """Reload container state (no-op in stub)."""


class _ContainersAPI:
    def __init__(self) -> None:
        self.run_calls: list[dict[str, Any]] = []
        self._next_status: int = 0
        self._next_output: str = ""
        self._last: _ContainerStub | None = None

    def set_next(self, *, status: int, output: str) -> None:
        self._next_status = status
        self._next_output = output

    def run(
        self,
        image: str,
        command: list[str] | str,
        volumes: dict[str, dict[str, str]],
        environment: dict[str, str],
        network_mode: str,
        extra_hosts: dict[str, str] | None,
        detach: bool,
        remove: bool,
    ) -> _ContainerStub:
        self.run_calls.append(
            {
                "image": image,
                "command": command,
                "volumes": volumes,
                "environment": environment,
                "network_mode": network_mode,
                "extra_hosts": extra_hosts,
                "detach": detach,
                "remove": remove,
            }
        )
        self._last = _ContainerStub(
            status_code=self._next_status, output=self._next_output
        )
        return self._last

    def get(self, container_id: str) -> _ContainerStub:
        # Return the last created container stub for simplicity
        return self._last or _ContainerStub(status_code=0, output="")


# Low-level API stubs -------------------------------------------------------


class _APIStub:
    """Recording stub for the low-level Docker API used by the build pipeline."""

    def __init__(self) -> None:
        self.prune_images_calls: list[dict[str, Any] | None] = []
        self.build_calls: list[dict[str, Any]] = []
        self._prune_error: Exception | None = None

    def build(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Record a low-level build request and stream a success chunk."""
        self.build_calls.append(kwargs)
        return [{"stream": "Step 1/1: built"}]

    def set_prune_error(self, exc: Exception) -> None:
        """Configure the next prune_images call to raise ``exc``."""
        self._prune_error = exc

    def prune_images(self, filters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record a dangling-image prune request.

        Args:
            filters: Filter dict passed by the caller (e.g. dangling-only)

        Returns:
            A minimal prune response payload.
        """
        self.prune_images_calls.append(filters)
        if self._prune_error is not None:
            raise self._prune_error
        return {"ImagesDeleted": [], "SpaceReclaimed": 0}


# Docker client stub ----------------------------------------------------------


class DockerClientStub:
    """In-memory stand-in for ``docker.DockerClient`` in build pipeline tests."""

    def __init__(self) -> None:
        self.images = _ImagesAPI()
        self.containers = _ContainersAPI()
        self.api = _APIStub()


# Pytest fixtures -------------------------------------------------------------

try:
    import pytest  # type: ignore
except Exception:  # pragma: no cover - tests will import pytest
    pytest = None  # type: ignore

if pytest is not None:

    @pytest.fixture()
    def docker_client_stub() -> DockerClientStub:
        """Provide a fresh DockerClientStub per test."""
        return DockerClientStub()
