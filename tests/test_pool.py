from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from graders.pool import DaytonaGraderPool
from tasks import TaskLoader


TASKS_DIR = Path(__file__).parents[1] / "tasks"
TASK = TaskLoader(TASKS_DIR).load("fizzbuzz_plus")
GOOD = """    if n % 3 == 0 and n % 5 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    if n % 7 == 0:
        return "Daytona"
    return str(n)
"""


class FakeSandbox:
    def __init__(self, sandbox_id: str) -> None:
        self.id = sandbox_id


class FakeResponse:
    error = None

    def __init__(self, passed: int = 9, total: int = 9) -> None:
        self.stdout = json.dumps({"results": [True] * passed + [False] * (total - passed)})


class FakeRunner:
    def __init__(self, num_tests: int = 9) -> None:
        self.num_tests = num_tests
        self.created: list[FakeSandbox] = []
        self.deleted: list[str] = []
        self.executed: list[str] = []
        self.executed_code: list[str] = []
        self.active: dict[str, int] = {}
        self.max_active: dict[str, int] = {}

    async def create(self, spec: Any) -> FakeSandbox:
        sandbox = FakeSandbox(f"sandbox-{len(self.created)}")
        self.created.append(sandbox)
        return sandbox

    async def delete(self, sandbox: FakeSandbox) -> None:
        self.deleted.append(sandbox.id)

    async def close(self) -> None:
        return None

    async def run_code(self, sandbox: FakeSandbox, code: str, timeout_seconds: int) -> FakeResponse:
        self.executed.append(sandbox.id)
        self.executed_code.append(code)
        self.active[sandbox.id] = self.active.get(sandbox.id, 0) + 1
        self.max_active[sandbox.id] = max(self.max_active.get(sandbox.id, 0), self.active[sandbox.id])
        await asyncio.sleep(0)
        self.active[sandbox.id] -= 1
        return FakeResponse(passed=self.num_tests, total=self.num_tests)


def test_pool_warms_reuses_round_robin_and_preserves_result_order() -> None:
    asyncio.run(_test_pool_warms_reuses_round_robin_and_preserves_result_order())


async def _test_pool_warms_reuses_round_robin_and_preserves_result_order() -> None:
    runner = FakeRunner()
    pool = DaytonaGraderPool(runner=runner, task_loader=TaskLoader(TASKS_DIR), requested_size=2)

    assert await pool.start() == 2
    results = await pool.evaluate_batch([GOOD] * 4, [TASK.prompt] * 4)

    assert [(result.num_passed, result.sandbox_id) for result in results] == [
        (9, "sandbox-0"),
        (9, "sandbox-1"),
        (9, "sandbox-0"),
        (9, "sandbox-1"),
    ]
    assert runner.max_active == {"sandbox-0": 1, "sandbox-1": 1}
    await pool.close()
    assert runner.deleted == ["sandbox-0", "sandbox-1"]


def test_pool_short_circuits_banned_completion_without_running_it() -> None:
    asyncio.run(_test_pool_short_circuits_banned_completion_without_running_it())


async def _test_pool_short_circuits_banned_completion_without_running_it() -> None:
    runner = FakeRunner()
    pool = DaytonaGraderPool(runner=runner, task_loader=TaskLoader(TASKS_DIR), requested_size=1)

    results = await pool.evaluate_batch(["    return eval(n)", GOOD], [TASK.prompt] * 2)

    assert results[0].banned
    assert results[0].sandbox_id == "sandbox-0"
    assert results[1].no_error
    assert runner.executed == ["sandbox-0"]
    await pool.close()


def test_pool_rejects_mismatched_inputs() -> None:
    asyncio.run(_test_pool_rejects_mismatched_inputs())


async def _test_pool_rejects_mismatched_inputs() -> None:
    runner = FakeRunner()
    pool = DaytonaGraderPool(runner=runner, task_loader=TaskLoader(TASKS_DIR), requested_size=1)

    with pytest.raises(ValueError, match="equal length"):
        await pool.evaluate_batch([GOOD], [])
    await pool.close()


def test_pool_accepts_explicit_task_variants_for_holdout_evaluation() -> None:
    asyncio.run(_test_pool_accepts_explicit_task_variants_for_holdout_evaluation())


async def _test_pool_accepts_explicit_task_variants_for_holdout_evaluation() -> None:
    runner = FakeRunner()
    pool = DaytonaGraderPool(runner=runner, task_loader=TaskLoader(TASKS_DIR), requested_size=1)
    holdout = replace(TASK, prompt="# holdout\n" + TASK.prompt)

    results = await pool.evaluate_batch([GOOD], ["holdout prompt"], tasks=[holdout])

    assert results[0].no_error
    assert "# holdout" in runner.executed_code[0]
    await pool.close()


def test_pool_resolves_a_declared_prompt_variant() -> None:
    asyncio.run(_test_pool_resolves_a_declared_prompt_variant())


async def _test_pool_resolves_a_declared_prompt_variant() -> None:
    runner = FakeRunner(num_tests=7)
    task = TaskLoader(TASKS_DIR).load("safe_parser")
    pool = DaytonaGraderPool(runner=runner, task_loader=TaskLoader(TASKS_DIR), requested_size=1)

    results = await pool.evaluate_batch(["    return [int(x) for x in s.split(',')]"] , [task.prompts[1]])

    assert results[0].error is None
    assert task.prompt in runner.executed_code[0]
    await pool.close()
