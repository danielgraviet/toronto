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
    holdout_tests: tuple[dict[str, Any], ...] = ()
    holdout_error_tests: tuple[dict[str, Any], ...] = ()
    prompt_variants: tuple[str, ...] = ()
    validation_tests: tuple[dict[str, Any], ...] = ()
    validation_error_tests: tuple[dict[str, Any], ...] = ()
    reward_weights: tuple[float, ...] = ()
    test_groups: tuple[str, ...] = ()

    @property
    def total_tests(self) -> int:
        return len(self.tests) + len(self.error_tests)

    @property
    def prompts(self) -> tuple[str, ...]:
        return (self.prompt, *self.prompt_variants)

    def __post_init__(self) -> None:
        if self.reward_weights and len(self.reward_weights) != self.total_tests:
            raise ValueError(
                f"Task {self.id} reward_weights must contain {self.total_tests} values"
            )
        if len(set(self.prompts)) != len(self.prompts):
            raise ValueError(f"Task {self.id} prompt variants must be unique")


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
        required = ("id", "title", "func_name", "prompt", "banned_patterns", "reference", "audience_blurb")
        if not isinstance(raw, dict) or any(key not in raw for key in required):
            raise ValueError(f"Task {path} is missing a required field")
        if raw["id"] != path.stem:
            raise ValueError(f"Task id {raw['id']!r} does not match {path.name}")
        tests, error_tests, reward_weights, group_names = _training_cases(raw, path)
        return Task(
            id=str(raw["id"]), title=str(raw["title"]), func_name=str(raw["func_name"]),
            prompt=str(raw["prompt"]), banned_patterns=tuple(map(str, raw["banned_patterns"])),
            tests=tests, reference=str(raw["reference"]),
            audience_blurb=str(raw["audience_blurb"]),
            error_tests=error_tests,
            holdout_tests=tuple(raw.get("holdout_tests", ())),
            holdout_error_tests=tuple(raw.get("holdout_error_tests", ())),
            prompt_variants=tuple(map(str, raw.get("prompt_variants", ()))),
            validation_tests=tuple(raw.get("validation_tests", ())),
            validation_error_tests=tuple(raw.get("validation_error_tests", ())),
            reward_weights=reward_weights,
            test_groups=group_names,
        )


def _training_cases(
    raw: dict[str, Any], path: Path
) -> tuple[tuple[dict[str, Any], ...], tuple[dict[str, Any], ...], tuple[float, ...], tuple[str, ...]]:
    """Expand optional named groups into the flat grader representation."""
    groups = raw.get("test_groups")
    if groups is None:
        if "tests" not in raw:
            raise ValueError(f"Task {path} must define tests or test_groups")
        return (
            tuple(raw["tests"]),
            tuple(raw.get("error_tests", ())),
            tuple(map(float, raw.get("reward_weights", ()))),
            (),
        )
    if not isinstance(groups, list) or not groups:
        raise ValueError(f"Task {path} test_groups must be a non-empty list")
    tests: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    weights: list[float] = []
    names: list[str] = []
    for group in groups:
        if not isinstance(group, dict) or not group.get("name"):
            raise ValueError(f"Task {path} has an invalid test group")
        name = str(group["name"])
        if name in names:
            raise ValueError(f"Task {path} repeats test group {name!r}")
        weight = float(group.get("weight", 1.0))
        if weight <= 0:
            raise ValueError(f"Task {path} test group weights must be positive")
        group_tests = tuple(group.get("tests", ()))
        group_errors = tuple(group.get("error_tests", ()))
        if not group_tests and not group_errors:
            raise ValueError(f"Task {path} test group {name!r} is empty")
        tests.extend(group_tests)
        errors.extend(group_errors)
        weights.extend([weight] * (len(group_tests) + len(group_errors)))
        names.append(name)
    return tuple(tests), tuple(errors), tuple(weights), tuple(names)
