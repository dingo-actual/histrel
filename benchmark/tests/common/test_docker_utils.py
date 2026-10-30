#!/usr/bin/env python3
"""Unit tests for benchmark.common.docker_utils.

Follows unit test taxonomy and naming conventions.
"""

from __future__ import annotations

from pathlib import Path

from benchmark.common import docker_utils as du
from benchmark.tests.fixtures.docker_stubs import DockerClientStub


class Test04Unit02Docker01Utils01ImageExists:
    def test_04_unit_02_docker_01_utils_01_image_exists_01_false_when_empty(
        self, docker_client_stub: DockerClientStub
    ) -> None:
        assert du.image_exists(docker_client_stub, "histrel-solver:latest") is False

    def test_04_unit_02_docker_01_utils_01_image_exists_02_true_when_tag_present(
        self, docker_client_stub: DockerClientStub
    ) -> None:
        docker_client_stub.images.add("histrel-solver:latest")
        assert du.image_exists(docker_client_stub, "histrel-solver:latest") is True


class Test04Unit02Docker01Utils02BuildImage:
    def test_04_unit_02_docker_01_utils_02_build_image_01_reuse_existing(
        self, tmp_path: Path, docker_client_stub: DockerClientStub
    ) -> None:
        tag = "histrel-solver:latest"
        docker_client_stub.images.add(tag, image_id="sha256:existing")
        # Create mock pyproject.toml so build context discovery stops at tmp_path
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname='test'\n", encoding="utf-8"
        )
        dockerfile = tmp_path / "Dockerfile.histrel"
        dockerfile.write_text("FROM scratch\n", encoding="utf-8")

        img_id = du.build_image(docker_client_stub, dockerfile, tag, rebuild=False)
        assert img_id == "sha256:existing"
        assert docker_client_stub.images.build_calls == []  # no build performed

    def test_04_unit_02_docker_01_utils_02_build_image_02_builds_when_rebuild_true(
        self, tmp_path: Path, docker_client_stub: DockerClientStub
    ) -> None:
        tag = "histrel-solver:latest"
        # Create mock pyproject.toml so build context discovery stops at tmp_path
        (tmp_path / "pyproject.toml").write_text(
            "[project]\nname='test'\n", encoding="utf-8"
        )
        dockerfile = tmp_path / "Dockerfile.histrel"
        dockerfile.write_text("FROM scratch\n", encoding="utf-8")

        img_id = du.build_image(docker_client_stub, dockerfile, tag, rebuild=True)
        assert img_id == "sha256:builtimageid"
        # build_image now uses project root as context and makes dockerfile path relative
        assert docker_client_stub.images.build_calls
        call = docker_client_stub.images.build_calls[0]
        assert call["dockerfile"] == "Dockerfile.histrel"
        assert call["path"] == str(tmp_path)


class Test04Unit02Docker01Utils03RunContainer:
    def test_04_unit_02_docker_01_utils_03_run_container_01_returns_exit_and_logs(
        self, tmp_path: Path, docker_client_stub: DockerClientStub
    ) -> None:
        docker_client_stub.containers.set_next(status=0, output="ok")

        stdout_file = (tmp_path / "stdout.log").open("wb")
        stderr_file = (tmp_path / "stderr.log").open("wb")

        exit_code, _container_id = du.run_container(
            docker_client_stub,
            image="histrel-solver:latest",
            command=["python", "-c", "print('hi')"],
            volumes={str(tmp_path): {"bind": "/workspace/output", "mode": "rw"}},
            environment={"PYTHONUNBUFFERED": "1"},
            network_mode="bridge",
            extra_hosts=None,
            timeout=5,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
        )

        stdout_file.close()
        stderr_file.close()

        assert exit_code == 0
        # ensure container was invoked with expected image and volumes
        assert docker_client_stub.containers.run_calls
        call = docker_client_stub.containers.run_calls[0]
        assert call["image"] == "histrel-solver:latest"
        assert "/workspace/output" in call["volumes"][str(tmp_path)]["bind"]
