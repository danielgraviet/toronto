"""Shared Daytona client used by GPU and CPU runners."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class SandboxSpec:
    """How Daytona should provision one sandbox.

    ``snapshot`` and ``image`` are mutually exclusive. A snapshot is the
    preferred option for the GPU because it contains the preinstalled stack.
    """

    snapshot: str | None = None
    image: str | None = None
    name: str | None = None
    cpu: int | None = None
    memory: int | None = None
    disk: int | None = None
    gpu: int | None = None
    gpu_type: str | None = None
    ephemeral: bool = False
    timeout_seconds: int = 120

    def __post_init__(self) -> None:
        if bool(self.snapshot) == bool(self.image):
            raise ValueError("Set exactly one of snapshot or image")


class DaytonaRunner:
    """Small adapter around the current Daytona async SDK.

    Imports are delayed until construction so local core tests do not need a
    live API key or a remote Daytona connection.
    """

    def __init__(self, api_key: str | None = None, api_url: str | None = None) -> None:
        load_dotenv()
        key = api_key or os.getenv("DAYTONA_API_KEY")
        if not key:
            raise RuntimeError("DAYTONA_API_KEY is required")
        import daytona

        config = daytona.DaytonaConfig(
            api_key=key,
            api_url=api_url or os.getenv("DAYTONA_API_URL"),
        )
        self._client = daytona.AsyncDaytona(config)

    async def create(self, spec: SandboxSpec) -> Any:
        import daytona

        common = {
            "name": spec.name,
            "language": "python",
            "timeout": spec.timeout_seconds,
        }
        if spec.snapshot:
            params = daytona.CreateSandboxFromSnapshotParams(
                snapshot=spec.snapshot,
                name=common["name"],
                language=common["language"],
                ephemeral=spec.ephemeral,
                auto_delete_interval=0 if spec.ephemeral else None,
            )
        else:
            resources = daytona.Resources(
                cpu=spec.cpu,
                memory=spec.memory,
                disk=spec.disk,
                gpu=spec.gpu,
                gpu_type=spec.gpu_type,
            )
            params = daytona.CreateSandboxFromImageParams(
                image=spec.image,
                name=common["name"],
                language=common["language"],
                resources=resources,
                ephemeral=spec.ephemeral,
                auto_delete_interval=0 if spec.ephemeral else None,
            )
        return await self._client.create(params, timeout=spec.timeout_seconds)

    async def delete(self, sandbox: Any, timeout_seconds: int = 60) -> None:
        await sandbox.delete(timeout=timeout_seconds)

    async def run_code(self, sandbox: Any, code: str, timeout_seconds: int = 30) -> Any:
        return await sandbox.code_interpreter.run_code(code, timeout=timeout_seconds)

    async def upload(self, sandbox: Any, local_path: str | Path, remote_path: str, timeout_seconds: int = 60) -> None:
        await sandbox.fs.upload_file(str(local_path), remote_path, timeout=timeout_seconds)

    async def upload_tree(
        self,
        sandbox: Any,
        local_root: str | Path,
        remote_root: str,
        timeout_seconds: int = 60,
    ) -> None:
        """Upload source/task files without uploading secrets or caches."""
        root = Path(local_root)
        paths = [
            path for path in root.rglob("*")
            if path.is_file()
            and "__pycache__" not in path.parts
            and path.suffix in {".py", ".yaml"}
            and path.name != ".env"
        ]
        await asyncio.gather(
            *(
                self.upload(
                    sandbox,
                    path,
                    f"{remote_root}/{path.relative_to(root)}",
                    timeout_seconds,
                )
                for path in paths
            )
        )

    async def exec(
        self,
        sandbox: Any,
        command: str,
        timeout_seconds: int = 60,
        env: dict[str, str] | None = None,
    ) -> Any:
        return await sandbox.process.exec(command, env=env, timeout=timeout_seconds)

    async def close(self) -> None:
        await self._client.close()
