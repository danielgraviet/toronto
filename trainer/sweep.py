"""Small independent hyperparameter sweep for the real Daytona GRPO path."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .smoke import DEFAULT_MODEL_NAME, ExperimentResult, run


SWEEP_CONFIGS: tuple[tuple[float, int, int], ...] = (
    (2e-6, 8, 4),
    (5e-6, 8, 4),
    (2e-6, 8, 16),
    (5e-6, 8, 16),
)


def run_sweep(
    model_name: str,
    pool_size: int,
    eval_samples: int,
    seed: int,
    task_id: str,
    output_root: str,
) -> dict[str, object]:
    """Run each configuration from the same base model and rank validation results."""
    root = Path(output_root)
    trials: list[dict[str, object]] = []
    for index, (learning_rate, train_steps, train_batch_size) in enumerate(SWEEP_CONFIGS, start=1):
        output_dir = root / (
            f"trial-{index}-lr-{learning_rate:g}-steps-{train_steps}"
            f"-batch-{train_batch_size}"
        )
        print(
            f"\n=== Sweep trial {index}/{len(SWEEP_CONFIGS)}: "
            f"lr={learning_rate:g}, steps={train_steps}, batch={train_batch_size} ==="
        )
        result = run(
            model_name=model_name,
            pool_size=pool_size,
            output_dir=str(output_dir),
            train_steps=train_steps,
            eval_samples=eval_samples,
            seed=seed,
            task_id=task_id,
            learning_rate=learning_rate,
            evaluate_final=False,
            train_batch_size=train_batch_size,
        )
        trials.append(
            {
                "trial": index,
                "learning_rate": learning_rate,
                "train_steps": train_steps,
                "train_batch_size": train_batch_size,
                **asdict(result),
            }
        )

    winner = max(trials, key=lambda item: (item["best_validation_pass_rate"], item["learning_rate"]))
    report = {"task_id": task_id, "seed": seed, "trials": trials, "winner": winner}
    report_path = root / "sweep-results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== Sweep ranking ===")
    for trial in sorted(trials, key=lambda item: item["best_validation_pass_rate"], reverse=True):
        print(
            f"  trial {trial['trial']}: validation="
            f"{trial['best_validation_pass_rate']:.1%}, "
            f"lr={trial['learning_rate']:g}, steps={trial['train_steps']}, "
            f"batch={trial['train_batch_size']}"
        )
    print(f"Winner: trial {winner['trial']}")
    print(f"Sweep report: {report_path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small GRPO hyperparameter sweep")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--pool-size", type=int, default=16)
    parser.add_argument("--eval-samples", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-id", default="two_sum_plus")
    parser.add_argument("--output-root", default="/tmp/toronto-grpo-sweep")
    args = parser.parse_args()
    run_sweep(
        args.model,
        args.pool_size,
        args.eval_samples,
        args.seed,
        args.task_id,
        args.output_root,
    )


if __name__ == "__main__":
    main()
