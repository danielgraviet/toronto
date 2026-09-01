"""One real GRPO step using the Daytona CPU reward path."""

from __future__ import annotations

import importlib
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from graders.core import reward_for
from graders.models import EvalResult
from graders.pool import DaytonaGraderPool
from runners.gpu import DEFAULT_MODEL_NAME, generate_completions, load_model
from tasks import TaskLoader

from .reward import DaytonaReward
from .config import get_profile


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    baseline_pass_rate: float
    best_validation_pass_rate: float
    best_checkpoint: str
    final_holdout_pass_rate: float | None = None


def run(
    model_name: str,
    pool_size: int,
    output_dir: str,
    train_steps: int = 6,
    eval_samples: int = 64,
    seed: int = 42,
    task_id: str = "fizzbuzz_plus",
    learning_rate: float = 5e-6,
    evaluate_final: bool = True,
    train_batch_size: int = 4,
    max_completion_length: int = 256,
    baseline_only: bool = False,
) -> ExperimentResult:
    torch = importlib.import_module("torch")
    datasets = importlib.import_module("datasets")
    trl = importlib.import_module("trl")

    model, tokenizer = load_model(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    task = TaskLoader().load(task_id)
    training_prompts = tuple(task.prompts)
    if train_batch_size < 1:
        raise ValueError("train_batch_size must be positive")
    training_prompts = tuple(
        training_prompts[index % len(training_prompts)] for index in range(train_batch_size)
    )
    holdout_task = replace(
        task,
        tests=task.holdout_tests,
        error_tests=task.holdout_error_tests,
        reward_weights=_evaluation_weights(task.holdout_tests, task.holdout_error_tests),
    )
    validation_task = replace(
        task,
        tests=task.validation_tests,
        error_tests=task.validation_error_tests,
        reward_weights=_evaluation_weights(
            task.validation_tests, task.validation_error_tests
        ),
    )
    # Four prompt rows let TRL form four-generation groups while still using
    # one task. Each row gets four sampled candidates per fresh step.
    dataset = datasets.Dataset.from_dict({"prompt": list(training_prompts[:4])})
    pool = DaytonaGraderPool(requested_size=pool_size)
    observed: list[EvalResult] = []
    curve: list[dict[str, float | int]] = []

    def observe_batch(results: list[EvalResult], scores: list[float]) -> None:
        pass_rate = sum(item.num_passed for item in results) / max(
            1, sum(item.num_tests for item in results)
        )
        point = {
            "step": len(curve) + 1,
            "mean_reward": statistics.mean(scores) if scores else 0.0,
            "pass_rate": pass_rate,
        }
        curve.append(point)
        print(
            f"Step {point['step']}: mean reward={point['mean_reward']:.3f}, "
            f"pass rate={point['pass_rate']:.1%}"
        )

    reward = DaytonaReward(
        pool,
        observer=lambda result, _: observed.append(result),
        batch_observer=observe_batch,
    )
    try:
        baseline_eval_task = holdout_task if evaluate_final else validation_task
        baseline_label = "holdout" if evaluate_final else "validation"
        print(f"\n=== Baseline: base model ({baseline_label}) ===")
        # Baseline belongs in its own measurement, not in the training curve.
        reward.batch_observer = None
        _seed_torch(torch, seed)
        baseline_completions = generate_completions(
            model, tokenizer, task.prompt, num_completions=eval_samples, max_new_tokens=1000
        )
        baseline_results = reward.bridge.run(
            pool.evaluate_batch(
                baseline_completions,
                [task.prompt] * len(baseline_completions),
                tasks=[baseline_eval_task] * len(baseline_completions),
            )
        )
        baseline_scores = [reward_for(result, reward.knobs) for result in baseline_results]
        _print_evaluation_summary("Baseline", baseline_results, baseline_scores)
        reward.batch_observer = observe_batch
        if baseline_only:
            return ExperimentResult(
                baseline_pass_rate=_pass_rate(baseline_results),
                best_validation_pass_rate=_pass_rate(baseline_results),
                best_checkpoint="baseline",
            )

        config = trl.GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=1,
            num_generations=4,
            steps_per_generation=1,
            max_completion_length=max_completion_length,
            learning_rate=learning_rate,
            max_steps=train_steps,
            save_strategy="steps",
            save_steps=1,
            save_total_limit=train_steps,
            logging_steps=1,
            report_to="none",
            bf16=torch.cuda.is_available(),
            use_vllm=False,
            gradient_checkpointing=True,
        )
        trainer = trl.GRPOTrainer(
            model=model,
            reward_funcs=reward,
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
        )
        train_result = trainer.train()
        trainer.save_model(output_dir)
        _print_smoke_summary(train_result.metrics, trainer.state.log_history, curve, output_dir)

        print("\n=== Checkpoint selection: validation evaluation ===")
        del trainer, model
        torch.cuda.empty_cache()
        checkpoints = sorted(
            Path(output_dir).glob("checkpoint-*"),
            key=lambda path: int(path.name.rsplit("-", 1)[1]),
        )
        best: tuple[Path, list[EvalResult], list[float]] | None = None
        for checkpoint in checkpoints:
            checkpoint_model, checkpoint_tokenizer = load_model(str(checkpoint))
            # Reuse the baseline seed so checkpoint comparisons are comparable.
            _seed_torch(torch, seed)
            completions = generate_completions(
                checkpoint_model,
                checkpoint_tokenizer,
                validation_task.prompt,
                num_completions=eval_samples,
                max_new_tokens=1000,
            )
            results = reward.bridge.run(
                pool.evaluate_batch(
                    completions,
                    [validation_task.prompt] * len(completions),
                    tasks=[validation_task] * len(completions),
                )
            )
            scores = [reward_for(result, reward.knobs) for result in results]
            pass_rate = sum(item.num_passed for item in results) / max(
                1, sum(item.num_tests for item in results)
            )
            print(f"  {checkpoint.name}: pass rate={pass_rate:.1%}")
            candidate = (checkpoint, results, scores)
            if best is None or _evaluation_key(candidate) > _evaluation_key(best):
                best = candidate
            del checkpoint_model, checkpoint_tokenizer
            torch.cuda.empty_cache()

        if best is None:
            raise RuntimeError("No step checkpoints were produced")
        best_checkpoint, _, _ = best
        best_validation_pass_rate = _pass_rate(best[1])
        if not evaluate_final:
            print(f"Best validation pass rate: {best_validation_pass_rate:.1%}")
            return ExperimentResult(
                baseline_pass_rate=_pass_rate(baseline_results),
                best_validation_pass_rate=best_validation_pass_rate,
                best_checkpoint=best_checkpoint.name,
            )

        best_model, best_tokenizer = load_model(str(best_checkpoint))
        _seed_torch(torch, seed)
        final_completions = generate_completions(
            best_model,
            best_tokenizer,
            task.prompt,
            num_completions=eval_samples,
            max_new_tokens=1000,
        )
        final_results = reward.bridge.run(
            pool.evaluate_batch(
                final_completions,
                [task.prompt] * len(final_completions),
                tasks=[holdout_task] * len(final_completions),
            )
        )
        final_scores = [reward_for(result, reward.knobs) for result in final_results]
        del best_model, best_tokenizer
        torch.cuda.empty_cache()
        print(f"\n=== Final evaluation: best checkpoint ({best_checkpoint.name}) ===")
        _print_evaluation_summary("Final holdout", final_results, final_scores)
        return ExperimentResult(
            baseline_pass_rate=_pass_rate(baseline_results),
            best_validation_pass_rate=best_validation_pass_rate,
            best_checkpoint=best_checkpoint.name,
            final_holdout_pass_rate=_pass_rate(final_results),
        )
    finally:
        # Daytona's async client was created and used on the reward bridge's
        # event loop. Close it on that same loop before stopping the bridge;
        # creating a second loop here can make socket cleanup fail with
        # ``RuntimeError: Event loop is closed``.
        reward.bridge.run(pool.close())
        reward.close()


def _print_smoke_summary(
    metrics: dict[str, object],
    log_history: list[dict[str, object]],
    curve: list[dict[str, float | int]],
    output_dir: str,
) -> None:
    """Turn TRL's dense metric dictionary into a talk-friendly summary."""
    def latest_number(name: str, default: float = 0.0) -> float:
        for entry in reversed(log_history):
            value = entry.get(name)
            if value is not None:
                return float(value)
        value = metrics.get(name)
        return float(value) if value is not None else default

    print("\n=== Real Daytona GRPO smoke summary ===")
    print(f"Training steps: {len(curve)}")
    print(f"Policy loss: {latest_number('loss', latest_number('train_loss')):.4f}")
    print(f"Gradient norm: {latest_number('grad_norm'):.3f}")
    print(f"Training time: {latest_number('step_time', latest_number('train_runtime')):.2f}s")
    print("\nTraining curve:")
    for point in curve:
        print(
            f"  step {point['step']}: reward={point['mean_reward']:.3f}, "
            f"pass rate={point['pass_rate']:.1%}"
        )
    print(f"Saved model: {output_dir}")


def _seed_torch(torch: Any, seed: int) -> None:
    """Make sampled baseline/final evaluations comparable."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_evaluation_summary(
    label: str, results: list[EvalResult], scores: list[float]
) -> None:
    pass_rate = sum(item.num_passed for item in results) / max(
        1, sum(item.num_tests for item in results)
    )
    print(f"{label} completions: {len(results)}")
    print(f"{label} average reward: {statistics.mean(scores) if scores else 0.0:.3f}")
    print(f"{label} pass rate: {pass_rate:.1%}")
    for index, (result, score) in enumerate(zip(results[:8], scores[:8]), start=1):
        status = f"{result.num_passed}/{result.num_tests} tests"
        if result.error:
            status += f", error={result.error}"
        print(f"  {index}. reward={score:.3f}, {status}")
    if len(results) > 8:
        print(f"  ... {len(results) - 8} more completions omitted")


def _evaluation_key(
    evaluation: tuple[Path, list[EvalResult], list[float]],
) -> tuple[float, float]:
    """Prefer held-out pass rate, then reward, when selecting a checkpoint."""
    _, results, scores = evaluation
    pass_rate = _pass_rate(results)
    return pass_rate, statistics.mean(scores) if scores else 0.0


def _pass_rate(results: list[EvalResult]) -> float:
    return sum(item.num_passed for item in results) / max(
        1, sum(item.num_tests for item in results)
    )


def _evaluation_weights(
    tests: tuple[dict[str, Any], ...],
    error_tests: tuple[dict[str, Any], ...],
) -> tuple[float, ...]:
    """Give each non-training evaluation case an aligned neutral weight."""
    return (1.0,) * (len(tests) + len(error_tests))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run one GRPO step with real Daytona rewards")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--profile", default=None, choices=("stage", "full"))
    parser.add_argument("--pool-size", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--eval-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task-id", default="fizzbuzz_plus")
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--train-batch-size", type=int, default=None)
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--output-dir", default="/tmp/toronto-real-grpo-smoke")
    args = parser.parse_args()
    profile = get_profile(args.profile)
    run(
        model_name=args.model,
        pool_size=args.pool_size or profile.pool_size,
        output_dir=args.output_dir,
        train_steps=args.train_steps or profile.train_steps,
        eval_samples=args.eval_samples or profile.eval_samples,
        seed=args.seed,
        task_id=args.task_id,
        learning_rate=args.learning_rate,
        evaluate_final=True,
        train_batch_size=args.train_batch_size or profile.train_batch_size,
        max_completion_length=args.max_completion_length or profile.max_completion_length,
        baseline_only=args.baseline_only,
    )


if __name__ == "__main__":
    main()
