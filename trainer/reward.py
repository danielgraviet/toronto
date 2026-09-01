"""Adapt asynchronous Daytona grading to TRL's reward-function interface."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from graders.core import reward_for
from graders.models import EvalResult, RewardKnobs

T = TypeVar("T")
ResultObserver = Callable[[EvalResult, int], None]
BatchObserver = Callable[[list[EvalResult], list[float]], None]


class AsyncLoopBridge:
    """Run coroutines from synchronous code on a dedicated event-loop thread."""

    def __init__(self) -> None:
        self._ready = threading.Event()
        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._run, name="daytona-reward-loop", daemon=True)
        self._thread.start()
        self._ready.wait()

    def run(self, awaitable: Awaitable[T]) -> T:
        if self._closed or self._loop is None:
            raise RuntimeError("AsyncLoopBridge is closed")
        future = asyncio.run_coroutine_threadsafe(_await(awaitable), self._loop)
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        self._ready.set()
        loop.run_forever()
        loop.close()


async def _await(awaitable: Awaitable[T]) -> T:
    return await awaitable


class DaytonaReward:
    """Callable reward function suitable for ``GRPOTrainer.reward_funcs``."""

    def __init__(
        self,
        pool: Any,
        knobs: RewardKnobs | None = None,
        observer: ResultObserver | None = None,
        batch_observer: BatchObserver | None = None,
        bridge: AsyncLoopBridge | None = None,
    ) -> None:
        self.pool = pool
        self.knobs = knobs or RewardKnobs()
        self.observer = observer
        self.batch_observer = batch_observer
        self.bridge = bridge or AsyncLoopBridge()
        self._owns_bridge = bridge is None

    @property
    def __name__(self) -> str:
        """Let TRL name this callable in its reward metrics."""
        return "daytona_reward"

    def __call__(self, prompts: list[str], completions: list[Any], **_: Any) -> list[float]:
        texts = [_completion_text(completion) for completion in completions]
        results = self.bridge.run(self.pool.evaluate_batch(texts, prompts))
        rewards: list[float] = []
        for index, result in enumerate(results):
            if self.observer:
                self.observer(result, index)
            rewards.append(reward_for(result, self.knobs))
        if self.batch_observer:
            self.batch_observer(results, rewards)
        return rewards

    def close(self) -> None:
        if self._owns_bridge:
            self.bridge.close()


def _completion_text(completion: Any) -> str:
    """Normalize TRL standard or conversational completion shapes."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion and isinstance(completion[0], dict):
        return str(completion[0].get("content", ""))
    return str(completion)
