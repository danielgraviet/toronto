"""Reusable asynchronous Daytona CPU sandbox pool."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from runners.daytona import DaytonaRunner, SandboxSpec
from tasks import Task, TaskLoader

from .core import build_test_harness, has_banned_pattern, parse_results, sanitize_completion
from .models import EvalResult
from .utils import elapsed_ms, start_timer


@dataclass(slots=True)
class _SandboxSlot:
    sandbox: Any
    gate: asyncio.Semaphore


class DaytonaGraderPool:
    """Warm and reuse CPU sandboxes for concurrent completion evaluation."""

    def __init__(
        self,
        runner: DaytonaRunner | None = None,
        task_loader: TaskLoader | None = None,
        requested_size: int = 32,
        timeout_seconds: int = 1,
        snapshot: str | None = None,
        image: str = "python:3.13",
    ) -> None:
        if requested_size < 1:
            raise ValueError("requested_size must be positive")
        self.runner = runner or DaytonaRunner()
        self.task_loader = task_loader or TaskLoader()
        self.requested_size = requested_size
        self.timeout_seconds = timeout_seconds
        self.snapshot = snapshot
        self.image = image
        self._slots: list[_SandboxSlot] = []
        self._started = False

    @property
    def healthy_size(self) -> int:
        return len(self._slots)

    async def start(self) -> int:
        if self._started:
            return self.healthy_size
        self._started = True
        specs = [self._spec(index) for index in range(self.requested_size)]
        created = await asyncio.gather(
            *(self.runner.create(spec) for spec in specs), return_exceptions=True
        )
        for index, result in enumerate(created):
            if isinstance(result, Exception):
                print(f"Sandbox {index} failed to warm: {type(result).__name__}: {result}")
                continue
            self._slots.append(_SandboxSlot(result, asyncio.Semaphore(1)))
        minimum = max(1, (self.requested_size + 1) // 2)
        if self.healthy_size < minimum:
            await self.stop()
            raise RuntimeError(
                f"Only {self.healthy_size}/{self.requested_size} sandboxes are healthy; "
                f"minimum is {minimum}"
            )
        return self.healthy_size

    async def stop(self) -> None:
        # Detach the active slots before awaiting deletion. This makes the
        # pool visibly empty immediately, even if cleanup takes time.
        slots = self._slots
        self._slots = []
        await asyncio.gather(
            *(self.runner.delete(slot.sandbox) for slot in slots), return_exceptions=True
        )
        self._started = False

    async def close(self) -> None:
        await self.stop()
        await self.runner.close()

    async def evaluate_batch(self, completions: list[str], prompts: list[str]) -> list[EvalResult]:
        if not self._started:
            await self.start()
        if not self._slots:
            raise RuntimeError("Grader pool has no healthy sandboxes")
        if len(completions) != len(prompts):
            raise ValueError("completions and prompts must have equal length")

        async def evaluate_at(index: int, completion: str, prompt: str) -> EvalResult:
            # Modulo wraps indices back to the first slot, producing an even
            # round-robin assignment: 0, 1, ..., N-1, 0, 1, ...
            slot = self._slots[index % len(self._slots)]
            # A slot has one gate, so two evaluations can never use the same
            # sandbox concurrently. Different slots still run in parallel.
            async with slot.gate:
                return await self._evaluate_one(slot.sandbox, completion, prompt)

        return list(await asyncio.gather(
            *(evaluate_at(index, completion, prompt) for index, (completion, prompt) in enumerate(zip(completions, prompts)))
        ))

    async def _evaluate_one(self, sandbox: Any, completion: str, prompt: str) -> EvalResult:
        started = start_timer()
        try:
            task = self._task_for_prompt(prompt)
        except (FileNotFoundError, ValueError) as exc:
            return self._failure(completion, None, "unknown_task", started, exc, sandbox)
        body = sanitize_completion(completion, task.func_name)
        if not body:
            return self._failure(completion, task, "empty_completion", started, sandbox=sandbox)
        if has_banned_pattern(body, task):
            return EvalResult(
                False, 0, task.total_tests, elapsed_ms(started),
                sandbox_id=getattr(sandbox, "id", None), banned=True, completion=completion,
            )
        try:
            response = await self.runner.run_code(
                sandbox, build_test_harness(task, body), timeout_seconds=self.timeout_seconds
            )
            if getattr(response, "error", None) is not None:
                return self._failure(completion, task, "execution_error", started, sandbox=sandbox)
            parsed = parse_results(getattr(response, "stdout", ""), task.total_tests)
            if parsed is None:
                return self._failure(completion, task, "invalid_results", started, sandbox=sandbox)
            passed, total = parsed
            return EvalResult(
                True, passed, total, elapsed_ms(started),
                sandbox_id=getattr(sandbox, "id", None), completion=completion,
            )
        except Exception as exc:
            return self._failure(completion, task, type(exc).__name__, started, exc, sandbox)

    def _task_for_prompt(self, prompt: str) -> Task:
        for task in self.task_loader.list():
            if task.prompt == prompt:
                return task
        raise ValueError("prompt is not registered in the task directory")

    def _spec(self, index: int) -> SandboxSpec:
        if self.snapshot:
            return SandboxSpec(snapshot=self.snapshot, name=f"toronto-grader-{index}", ephemeral=True)
        return SandboxSpec(image=self.image, name=f"toronto-grader-{index}", ephemeral=True)

    def _failure(
        self,
        completion: str,
        task: Task | None,
        error: str,
        started: float,
        exc: Exception | None = None,
        sandbox: Any | None = None,
    ) -> EvalResult:
        if exc:
            print(f"Grader failure: {type(exc).__name__}: {exc}")
        return EvalResult(
            False, 0, task.total_tests if task else 0, elapsed_ms(started),
            error=error, sandbox_id=getattr(sandbox, "id", None), completion=completion,
        )
