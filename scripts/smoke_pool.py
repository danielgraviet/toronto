"""Exercise a small real Daytona grader pool."""

from __future__ import annotations

import argparse
import asyncio

from graders.pool import DaytonaGraderPool
from runners.sandbox import GOOD_FIZZBUZZ_COMPLETION
from tasks import TaskLoader


async def run(pool_size: int, completion_count: int) -> None:
    task = TaskLoader().load("fizzbuzz_plus")
    pool = DaytonaGraderPool(requested_size=pool_size)
    try:
        print(f"healthy: {await pool.start()}")
        results = await pool.evaluate_batch(
            [GOOD_FIZZBUZZ_COMPLETION] * completion_count,
            [task.prompt] * completion_count,
        )
        print(
            "results:",
            [(result.num_passed, result.num_tests, result.sandbox_id, result.error) for result in results],
        )
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test a real Daytona grader pool")
    parser.add_argument("--pool-size", type=int, default=2)
    parser.add_argument("--completions", type=int, default=4)
    args = parser.parse_args()
    asyncio.run(run(args.pool_size, args.completions))


if __name__ == "__main__":
    main()
