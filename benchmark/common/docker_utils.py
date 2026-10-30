#!/usr/bin/env python3
"""Docker management utilities for the benchmark suite.

Functions:
- image_exists: check if an image tag exists locally
- build_image: build an image from a Dockerfile
- run_container: run a one-shot container with stdout/stderr to files
- cleanup_container: remove a container by ID (force)
- cleanup_image: remove an image by tag (force)

Notes:
- Requires the docker SDK for Python (docker)
- Uses the benchmark logger utilities for consistent logging
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import IO, Any, Protocol

from loguru import logger

# Protocols ------------------------------------------------------------------


class ImageLike(Protocol):
    id: str | None
    tags: Sequence[str] | None


class ImagesAPI(Protocol):
    def list(self) -> list[ImageLike]: ...
    def build(
        self,
        *,
        path: str,
        dockerfile: str,
        tag: str,
        buildargs: dict[str, str],
        rm: bool,
    ) -> tuple[ImageLike, Any]: ...
    def remove(self, image_tag: str, force: bool = True) -> None: ...


class ContainerLike(Protocol):
    def wait(self, timeout: int | None = None) -> dict[str, Any]: ...
    def logs(self, *args: Any, **kwargs: Any) -> Any: ...
    def attach(self, *args: Any, **kwargs: Any) -> Any: ...
    def remove(self, force: bool = True) -> None: ...
    def reload(self) -> None: ...


class ContainersAPI(Protocol):
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
    ) -> ContainerLike: ...
    def get(self, container_id: str) -> ContainerLike: ...


class DockerClientLike(Protocol):
    @property
    def images(self) -> ImagesAPI: ...

    @property
    def containers(self) -> ContainersAPI: ...


# Helper ---------------------------------------------------------------------


def _log_build_output(build_logs: list[dict[str, Any]] | Any) -> None:
    """Emit docker build logs at INFO level for errors, DEBUG for normal output.

    Accepts the iterator from client.images.build and logs any stream chunks.
    """
    for chunk in build_logs or []:
        with contextlib.suppress(Exception):
            # Check for error in chunk
            if "error" in chunk or "errorDetail" in chunk:
                error_msg = chunk.get("error") or chunk.get("errorDetail", {}).get(
                    "message", "Unknown error"
                )
                logger.bind(component="histrel.benchmark.docker").error(
                    f"Docker build error: {error_msg}"
                )

            stream = chunk.get("stream")
            if stream:
                msg = stream.strip()
                if msg:
                    # Log errors and warnings at INFO, everything else at DEBUG
                    if "error" in msg.lower() or "warning" in msg.lower():
                        logger.bind(component="histrel.benchmark.docker").info(msg)
                    else:
                        logger.bind(component="histrel.benchmark.docker").debug(msg)


# Image operations -----------------------------------------------------------


def image_exists(client: DockerClientLike, tag: str) -> bool:
    """Return True if an image with the given tag exists locally."""
    return any(tag in (img.tags or []) for img in client.images.list())


def build_image(
    client: DockerClientLike,
    dockerfile_path: Path,
    tag: str,
    *,
    build_args: dict[str, str] | None = None,
    rebuild: bool = False,
) -> str:
    """Build a Docker image and return its ID.

    If rebuild is False and an image with the tag already exists, it will be
    reused and the existing image ID is returned.
    """
    if not rebuild and image_exists(client, tag):
        logger.bind(component="histrel.benchmark.docker").info(
            f"Reusing existing Docker image: {tag}"
        )
        # Return the ID of the first matching image
        for img in client.images.list():
            if tag in (img.tags or []):
                return img.id or tag

    # Use project root as build context (find pyproject.toml upwards from Dockerfile)
    context_dir = dockerfile_path.parent
    while context_dir.parent != context_dir:
        if (context_dir / "pyproject.toml").exists():
            break
        context_dir = context_dir.parent

    logger.bind(component="histrel.benchmark.docker").info(
        f"Building Docker image: {tag} (dockerfile={dockerfile_path}, context={context_dir})"
    )
    # Convert dockerfile path to use forward slashes (Docker requires this even on Windows)
    relative_dockerfile = dockerfile_path.relative_to(context_dir).as_posix()
    image, build_logs = client.images.build(
        path=str(context_dir),
        dockerfile=relative_dockerfile,
        tag=tag,
        buildargs=build_args or {},
        rm=True,
    )
    _log_build_output(build_logs)
    _id = image.id or "<no-id>"
    logger.bind(component="histrel.benchmark.docker").info(
        f"Built image {tag} ({_id[:12]})"
    )
    return image.id or tag


# Container operations -------------------------------------------------------


def run_container(
    client: DockerClientLike,
    *,
    image: str,
    command: list[str] | str,
    volumes: dict[str, dict[str, str]] | None = None,
    environment: dict[str, str] | None = None,
    network_mode: str = "bridge",
    extra_hosts: dict[str, str] | None = None,
    timeout: int | None = None,
    stdout_file: IO[bytes] | None = None,
    stderr_file: IO[bytes] | None = None,
) -> tuple[int, str]:
    """Run a container to completion, streaming stdout/stderr to files.

    Args:
        client: Docker client instance.
        image: Docker image to run.
        command: Command to execute in container.
        volumes: Volume mounts for container.
        environment: Environment variables.
        network_mode: Docker network mode.
        extra_hosts: Extra host entries.
        timeout: Max runtime in seconds.
        stdout_file: File handle to write container stdout (binary mode).
        stderr_file: File handle to write container stderr (binary mode).

    Returns:
        (exit_code, container_id) Tuple of exit code and container ID.
    """
    logger.bind(component="histrel.benchmark.docker").info(
        f"Running container (image={image}, network={network_mode})"
    )
    container = client.containers.run(
        image,
        command=command,
        volumes=volumes or {},
        environment=environment or {},
        network_mode=network_mode,
        extra_hosts=extra_hosts or {},
        detach=True,
        remove=False,
    )

    # Notify callback that container has started
    container_id_str = getattr(container, "id", "<unknown>")

    # Start a background thread to stream logs to files
    import threading

    def _stream_logs() -> None:
        with contextlib.suppress(Exception):
            for stdout_chunk, stderr_chunk in container.attach(stream=True, demux=True):
                if stdout_chunk is not None and stdout_file is not None:
                    with contextlib.suppress(Exception):
                        stdout_file.write(stdout_chunk)
                        stdout_file.flush()
                if stderr_chunk is not None and stderr_file is not None:
                    with contextlib.suppress(Exception):
                        stderr_file.write(stderr_chunk)
                        stderr_file.flush()

    t = threading.Thread(
        target=_stream_logs,
        name=f"docker-log-{getattr(container, 'id', '')}",
        daemon=True,
    )
    t.start()

    # Wait for container completion with timeout support
    exit_code = 1
    try:
        result = container.wait(timeout=timeout)
        exit_code = int(result.get("StatusCode", 1))
        logger.bind(
            component="histrel.benchmark.docker",
            container_id=container_id_str,
            exit_code=exit_code,
            status=result.get("StatusCode"),
        ).info("Container wait completed")
    except Exception:
        # Timeout or unexpected error: attempt to stop the container (best-effort)
        stop = getattr(container, "stop", None)
        if callable(stop):
            with contextlib.suppress(Exception):
                stop(timeout=5)
        exit_code = 124

    finally:
        if exit_code != 0:
            logger.bind(
                component="histrel.benchmark.docker",
                container_id=container_id_str,
                exit_code=exit_code,
            ).error(f"Container finished with non-zero exit code {exit_code}")
        else:
            logger.bind(
                component="histrel.benchmark.docker",
                container_id=container_id_str,
            ).info("Container finished successfully")

    # Ensure log thread finishes quickly
    t.join(timeout=2.0)

    with contextlib.suppress(Exception):
        container.remove(force=True)

    logger.bind(component="histrel.benchmark.docker").info(
        f"Container finished (exit={exit_code})"
    )
    return exit_code, getattr(container, "id", "<unknown>")


def cleanup_container(client: DockerClientLike, container_id: str) -> None:
    """Remove a container by ID (force)."""
    with contextlib.suppress(Exception):
        container = client.containers.get(container_id)
        container.remove(force=True)
        logger.bind(component="histrel.benchmark.docker").info(
            f"Removed container {container_id[:12]}"
        )


def cleanup_image(client: DockerClientLike, image_tag: str) -> None:
    """Remove an image by tag (force)."""
    with contextlib.suppress(Exception):
        client.images.remove(image_tag, force=True)
        logger.bind(component="histrel.benchmark.docker").info(
            f"Removed image {image_tag}"
        )
