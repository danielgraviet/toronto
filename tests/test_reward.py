from __future__ import annotations

import asyncio

from graders.models import EvalResult, RewardKnobs
from trainer.reward import DaytonaReward


class FakePool:
    def __init__(self, results: list[EvalResult]) -> None:
        self.results = results
        self.received: tuple[list[str], list[str]] | None = None

    async def evaluate_batch(
        self,
        completions: list[str],
        prompts: list[str],
        tasks=None,
        on_result=None,
    ) -> list[EvalResult]:
        self.received = (completions, prompts)
        await asyncio.sleep(0)
        if on_result is not None:
            for index, result in enumerate(self.results):
                on_result(index, result)
        return self.results


class InlineBridge:
    def run(self, awaitable):
        return asyncio.run(awaitable)

    def close(self) -> None:
        return None


def test_daytona_reward_returns_ordered_scalar_rewards() -> None:
    pool = FakePool([
        EvalResult(True, 9, 9, completion="    return good"),
        EvalResult(True, 4, 9, completion="    return partial"),
        EvalResult(False, 0, 9, error="timeout", completion="    while True: pass"),
    ])
    reward = DaytonaReward(pool, bridge=InlineBridge())

    values = reward(["p1", "p2", "p3"], ["a", "b", "c"])

    assert values == [1.0, 4 / 9, -1.0]
    assert pool.received == (["a", "b", "c"], ["p1", "p2", "p3"])


def test_daytona_reward_normalizes_conversational_completions() -> None:
    pool = FakePool([EvalResult(True, 1, 1, completion="body")])
    reward = DaytonaReward(pool, bridge=InlineBridge())

    assert reward(["prompt"], [[{"role": "assistant", "content": "body"}]]) == [1.0]
    assert pool.received == (["body"], ["prompt"])


def test_daytona_reward_applies_current_knobs_and_observes_results() -> None:
    result = EvalResult(True, 10, 10, duration_ms=0, completion="\n".join("    pass" for _ in range(10)))
    pool = FakePool([result])
    observed: list[tuple[EvalResult, int]] = []
    reward = DaytonaReward(
        pool,
        knobs=RewardKnobs(lambda_len=0.1, lambda_speed=0.2),
        observer=lambda item, index: observed.append((item, index)),
        bridge=InlineBridge(),
    )

    assert reward(["prompt"], ["body"]) == [1.0]
    assert observed == [(result, 0)]
