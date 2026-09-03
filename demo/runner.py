"""Launch the fixed GRPO demo with the Rich display."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re

from demo.config import DEMO_CONFIG
from demo.display import DemoDisplay
from runners.gpu import run_remote


def build_remote_args() -> argparse.Namespace:
    config = DEMO_CONFIG
    return argparse.Namespace(
        model=config.model_name,
        profile=config.profile_name,
        num_completions=4,
        max_new_tokens=config.max_completion_length,
        remote=True,
        gpu_type=config.gpu_type,
        show_build_logs=False,
        keep=False,
        sandbox_id=None,
        inspect_trl=False,
        synthetic_grpo=False,
        real_grpo_smoke=True,
        sweep=False,
        pool_size=config.pool_size,
        train_steps=config.train_steps,
        eval_samples=config.eval_samples,
        seed=config.seed,
        task_id=config.task_id,
        learning_rate=config.learning_rate,
        train_batch_size=config.train_batch_size,
        max_completion_length=config.max_completion_length,
        baseline_only=False,
        generation_backend=os.getenv("TORONTO_GENERATION_BACKEND"),
    )


def _provisioning_line(line: str) -> str | None:
    match = re.search(r"Created remote sandbox:\s*(\S+)", line)
    if match:
        return json.dumps({"event": "provisioning", "gpu_sandbox_id": match.group(1)})
    return None


async def run_demo() -> None:
    display = DemoDisplay()
    with display.start():
        def handle_line(line: str) -> None:
            provisioning = _provisioning_line(line)
            if provisioning:
                display.handle_line(provisioning)
            display.handle_line(line)

        await run_remote(
            build_remote_args(),
            on_progress_line=handle_line,
            poll_interval=DEMO_CONFIG.poll_interval,
            quiet_progress=True,
        )
        if display.state.complete:
            display.wait_for_exit()


def main() -> None:
    asyncio.run(run_demo())
