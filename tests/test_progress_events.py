from __future__ import annotations

import asyncio

from graders.models import EvalResult
from trainer.progress import truncate_preview


class StreamingPool:
    async def evaluate_batch(self, completions, prompts, tasks=None, on_result=None):
        results = [
            EvalResult(True, 2, 4, completion=text, sandbox_id=f"sbx-{index}")
            for index, text in enumerate(completions)
        ]
        if on_result is not None:
            for index, result in enumerate(results):
                await asyncio.sleep(0)
                on_result(index, result)
        return results


class InlineBridge:
    def run(self, awaitable):
        return asyncio.run(awaitable)

    def close(self) -> None:
        return None


def test_rollout_reward_payload_shape_from_observer() -> None:
    from trainer.reward import DaytonaReward

    emitted: list[dict] = []
    pool = StreamingPool()

    def observer(result: EvalResult, index: int) -> None:
        emitted.append(
            {
                "event": "rollout_reward",
                "index": index,
                "reward": 0.5,
                "num_passed": result.num_passed,
                "num_tests": result.num_tests,
                "completion_preview": truncate_preview(result.completion),
                "sandbox_id": result.sandbox_id,
            }
        )

    reward = DaytonaReward(pool, observer=observer, bridge=InlineBridge())
    values = reward(["prompt"], ["def two_sum(nums, target):\n    return []"])

    assert len(values) == 1
    assert emitted[0]["index"] == 0
    assert "two_sum" in emitted[0]["completion_preview"]
    assert emitted[0]["sandbox_id"] == "sbx-0"
