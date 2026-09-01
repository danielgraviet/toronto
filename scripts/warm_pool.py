"""Verify that a Daytona CPU grader pool can be created for stage."""

from __future__ import annotations

import argparse
import asyncio

from graders.pool import DaytonaGraderPool
from trainer.config import get_profile


async def warm(pool_size: int) -> None:
    pool = DaytonaGraderPool(requested_size=pool_size)
    try:
        healthy = await pool.start()
        print(f"Daytona grader capacity: {healthy}/{pool_size} healthy")
        if healthy < pool_size:
            raise RuntimeError(
                f"Only {healthy}/{pool_size} graders warmed; stage expects all requested slots"
            )
    finally:
        await pool.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Check and release Daytona CPU stage capacity")
    parser.add_argument("--profile", default="stage", choices=("stage", "full"))
    parser.add_argument("--n", type=int, default=None, help="override the profile pool size")
    args = parser.parse_args()
    profile = get_profile(args.profile)
    asyncio.run(warm(args.n or profile.pool_size))


if __name__ == "__main__":
    main()
