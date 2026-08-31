from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    title: str
    func_name: str
    prompt: str
    banned_patterns: tuple[str, ...]
    tests: tuple[dict[str, Any], ...]
    reference: str
    audience_blurb: str
    error_tests: tuple[dict[str, Any], ...] = ()

    @property
    def total_tests(self) -> int:
        return len(self.tests) + len(self.error_tests)


class TaskLoader:
    def __init__(self, directory: str | Path = "tasks") -> None:
        self.directory = Path(directory)

    def load(self, task_id: str) -> Task:
        path = self.directory / f"{task_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Unknown task: {task_id}")
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return self._from_mapping(raw, path)

    def list(self) -> list[Task]:
        return [self.load(path.stem) for path in sorted(self.directory.glob("*.yaml"))]

    @staticmethod
    def _from_mapping(raw: Any, path: Path) -> Task:
        required = ("id", "title", "func_name", "prompt", "banned_patterns", "tests", "reference", "audience_blurb")
        if not isinstance(raw, dict) or any(key not in raw for key in required):
            raise ValueError(f"Task {path} is missing a required field")
        if raw["id"] != path.stem:
            raise ValueError(f"Task id {raw['id']!r} does not match {path.name}")
        return Task(
            id=str(raw["id"]), title=str(raw["title"]), func_name=str(raw["func_name"]),
            prompt=str(raw["prompt"]), banned_patterns=tuple(map(str, raw["banned_patterns"])),
            tests=tuple(raw["tests"]), reference=str(raw["reference"]),
            audience_blurb=str(raw["audience_blurb"]), error_tests=tuple(raw.get("error_tests", ())),
        )
