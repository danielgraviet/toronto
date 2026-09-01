"""Run one Toronto task harness in one CPU Daytona sandbox."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from graders.core import build_test_harness, parse_results, sanitize_completion
from runners.daytona import DaytonaRunner, SandboxSpec
from tasks import TaskLoader


GOOD_FIZZBUZZ_COMPLETION = """    if n % 3 == 0 and n % 5 == 0:
        return \"FizzBuzz\"
    if n % 3 == 0:
        return \"Fizz\"
    if n % 5 == 0:
        return \"Buzz\"
    if n % 7 == 0:
        return \"Daytona\"
    return str(n)
"""


async def run_one(task_id: str, completion: str, snapshot: str | None) -> None:
    task = TaskLoader().load(task_id)
    body = sanitize_completion(completion, task.func_name)
    harness = build_test_harness(task, body)
    runner = DaytonaRunner()
    if snapshot:
        spec = SandboxSpec(snapshot=snapshot, name="toronto-cpu-smoke", ephemeral=True)
    else:
        spec = SandboxSpec(image="python:3.13", name="toronto-cpu-smoke", ephemeral=True)
    sandbox = None
    try:
        sandbox = await runner.create(spec)
        print(f"Created sandbox: {getattr(sandbox, 'id', 'unknown')}")
        response = await runner.run_code(sandbox, harness, timeout_seconds=10)
        stdout = getattr(response, "stdout", "")
        parsed = parse_results(stdout, task.total_tests)
        print(f"stdout:\n{stdout}")
        print(f"parsed: {parsed}")
        if parsed is None:
            raise RuntimeError("Sandbox did not return the expected results payload")
    finally:
        if sandbox is not None:
            await runner.delete(sandbox)
            print("Deleted sandbox")
        await runner.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test one CPU Daytona grader")
    parser.add_argument("--task", default="fizzbuzz_plus")
    parser.add_argument("--completion-file", type=Path)
    parser.add_argument("--snapshot", help="optional CPU snapshot; default uses a Python image")
    args = parser.parse_args()
    completion = args.completion_file.read_text(encoding="utf-8") if args.completion_file else GOOD_FIZZBUZZ_COMPLETION
    asyncio.run(run_one(args.task, completion, args.snapshot))


if __name__ == "__main__":
    main()
