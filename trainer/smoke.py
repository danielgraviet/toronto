"""One real GRPO step using the Daytona CPU reward path."""

from __future__ import annotations

import importlib
import statistics
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from graders.core import reward_for
from graders.models import EvalResult
from graders.pool import DaytonaGraderPool
from runners.gpu import DEFAULT_MODEL_NAME, generate_completions, load_model
from tasks import TaskLoader

from .progress import ProgressEmitter, format_completion_for_display, truncate_preview
from .reward import DaytonaReward
from .callbacks import build_step_validation_callback
from .config import get_profile
from .generation import get_generation_config, grpo_generation_kwargs
from .timing import timed_phase


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
    progress_file: str | None = None,
    progress: ProgressEmitter | None = None,
    step_eval_samples: int = 8,
    generation_backend: str | None = None,
) -> ExperimentResult:
    torch = importlib.import_module("torch")
    datasets = importlib.import_module("datasets")
    trl = importlib.import_module("trl")

    emitter = progress or ProgressEmitter(progress_file)
    gen_config = get_generation_config(generation_backend)
    training_step = {"value": 0}
    eval_backend = "hf"

    def emit(payload: dict[str, Any]) -> None:
        emitter.emit(payload)

    emit(
        {
            "event": "run_started",
            "generation_backend": gen_config.backend,
            "train_steps": train_steps,
            "task_id": task_id,
        }
    )

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
        mean_reward = statistics.mean(scores) if scores else 0.0
        if step_callback is not None:
            step_callback.set_latest_mean_reward(mean_reward)
        point = {
            "step": len(curve) + 1,
            "mean_reward": mean_reward,
            "pass_rate": pass_rate,
        }
        curve.append(point)
        emit(
            {
                "event": "step_finished",
                "step": point["step"],
                "mean_reward": point["mean_reward"],
                "pass_rate": point["pass_rate"],
            }
        )
        print(
            f"Step {point['step']}: mean reward={point['mean_reward']:.3f}, "
            f"pass rate={point['pass_rate']:.1%}"
        )

    def observe_result(result: EvalResult, index: int) -> None:
        observed.append(result)
        score = reward_for(result, reward.knobs)
        emit(
            {
                "event": "rollout_reward",
                "index": index,
                "reward": score,
                "num_passed": result.num_passed,
                "num_tests": result.num_tests,
                "duration_ms": result.duration_ms,
                "banned": result.banned,
                "error": result.error,
                "completion_preview": truncate_preview(result.completion),
                "phase": reward.grading_phase,
                "step": training_step["value"] if reward.grading_phase == "training" else None,
                "sandbox_id": result.sandbox_id,
            }
        )

    def on_grading_started(count: int) -> None:
        payload: dict[str, Any] = {
            "event": "grading_started",
            "count": count,
            "phase": reward.grading_phase,
        }
        if reward.grading_phase == "training":
            training_step["value"] += 1
            payload["step"] = training_step["value"]
            emit(
                {
                    "event": "step_started",
                    "step": training_step["value"],
                    "total_steps": train_steps,
                    "count": count,
                }
            )
        emit(payload)

    step_callback: Any | None = None
    reward = DaytonaReward(
        pool,
        observer=observe_result,
        batch_observer=observe_batch,
        on_grading_started=on_grading_started,
    )
    try:
        emit({"event": "model_loading"})
        model, tokenizer = load_model(model_name)
        emit({"event": "model_ready"})
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        emit({"event": "pool_warming", "requested": pool_size})
        reward.bridge.run(pool.start())
        emit({"event": "pool_ready", "healthy": pool.healthy_size, "requested": pool_size})

        baseline_eval_task = holdout_task if evaluate_final else validation_task
        baseline_label = "holdout" if evaluate_final else "validation"
        print(f"\n=== Baseline: base model ({baseline_label}) ===")
        emit({"event": "baseline_started"})
        reward.batch_observer = None
        reward.grading_phase = "baseline"
        _seed_torch(torch, seed)
        emit({"event": "generation_started", "phase": "baseline", "count": eval_samples})
        with timed_phase(emit, "baseline_generate", eval_backend):
            baseline_completions = generate_completions(
                model, tokenizer, task.prompt, num_completions=eval_samples,
                max_new_tokens=max_completion_length,
            )
        emit({"event": "generation_finished", "phase": "baseline", "count": len(baseline_completions)})
        emit({"event": "grading_started", "phase": "baseline", "count": len(baseline_completions)})
        with timed_phase(emit, "baseline_grade", eval_backend):
            baseline_results = reward.bridge.run(
                pool.evaluate_batch(
                    baseline_completions,
                    [task.prompt] * len(baseline_completions),
                    tasks=[baseline_eval_task] * len(baseline_completions),
                    on_result=lambda index, result: observe_result(result, index),
                )
            )
            validation_baseline_results = reward.bridge.run(
                pool.evaluate_batch(
                    baseline_completions,
                    [validation_task.prompt] * len(baseline_completions),
                    tasks=[validation_task] * len(baseline_completions),
                )
            )
        baseline_scores = [reward_for(result, reward.knobs) for result in baseline_results]
        baseline_best_index = max(range(len(baseline_results)), key=lambda index: baseline_scores[index])
        validation_scores = [reward_for(result, reward.knobs) for result in validation_baseline_results]
        validation_baseline_pass_rate = _pass_rate(validation_baseline_results)
        baseline_sample_index = min(
            range(len(validation_scores)), key=lambda index: validation_scores[index]
        )
        _print_evaluation_summary("Baseline", baseline_results, baseline_scores)
        emit(
            {
                "event": "baseline_finished",
                "pass_rate": validation_baseline_pass_rate,
                "sample_completion": format_completion_for_display(
                    task, baseline_completions[baseline_sample_index]
                ),
                "sample_reward": validation_scores[baseline_sample_index],
            }
        )
        reward.batch_observer = observe_batch
        reward.grading_phase = "training"
        if baseline_only:
            emit({"event": "complete"})
            return ExperimentResult(
                baseline_pass_rate=_pass_rate(baseline_results),
                best_validation_pass_rate=_pass_rate(baseline_results),
                best_checkpoint="baseline",
            )

        emit(
            {
                "event": "trainer_initializing",
                "generation_backend": gen_config.backend,
                "train_steps": train_steps,
            }
        )
        config = trl.GRPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=train_batch_size,
            gradient_accumulation_steps=1,
            num_generations=4,
            # Align generation batch with prompt batch so each step grades
            # train_batch_size * num_generations rollouts (e.g. 4×4=16).
            steps_per_generation=train_batch_size,
            max_completion_length=max_completion_length,
            learning_rate=learning_rate,
            max_steps=train_steps,
            save_strategy="steps",
            save_steps=1,
            save_total_limit=train_steps,
            logging_steps=1,
            report_to="none",
            bf16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            **grpo_generation_kwargs(gen_config),
        )
        step_callback = build_step_validation_callback(
            tokenizer=tokenizer,
            validation_task=validation_task,
            pool=pool,
            bridge=reward.bridge,
            emit=emit,
            eval_samples=step_eval_samples,
            max_new_tokens=max_completion_length,
            seed=seed,
            torch=torch,
        )
        with timed_phase(emit, "trainer_init", gen_config.backend):
            trainer = trl.GRPOTrainer(
                model=model,
                reward_funcs=reward,
                args=config,
                train_dataset=dataset,
                processing_class=tokenizer,
                callbacks=[step_callback],
            )
        emit(
            {
                "event": "training_started",
                "steps": train_steps,
                "generation_backend": gen_config.backend,
                "rollouts_per_step": train_batch_size * 4,
            }
        )
        with timed_phase(emit, "training", gen_config.backend):
            train_result = _train_with_progress(emit, trainer, gen_config.backend)
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
        selection_checkpoints = checkpoints[-1:] if train_steps <= 4 else checkpoints
        with timed_phase(emit, "checkpoint_eval", eval_backend):
            for checkpoint in selection_checkpoints:
                checkpoint_model, checkpoint_tokenizer = load_model(str(checkpoint))
                # Reuse the baseline seed so checkpoint comparisons are comparable.
                _seed_torch(torch, seed)
                completions = generate_completions(
                    checkpoint_model,
                    checkpoint_tokenizer,
                    validation_task.prompt,
                    num_completions=eval_samples,
                    max_new_tokens=max_completion_length,
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
            emit({"event": "complete"})
            return ExperimentResult(
                baseline_pass_rate=_pass_rate(baseline_results),
                best_validation_pass_rate=best_validation_pass_rate,
                best_checkpoint=best_checkpoint.name,
            )

        best_model, best_tokenizer = load_model(str(best_checkpoint))
        _seed_torch(torch, seed)
        with timed_phase(emit, "final_eval", eval_backend):
            final_completions = generate_completions(
                best_model,
                best_tokenizer,
                task.prompt,
                num_completions=eval_samples,
                max_new_tokens=max_completion_length,
            )
            final_results = reward.bridge.run(
                pool.evaluate_batch(
                    final_completions,
                    [task.prompt] * len(final_completions),
                    tasks=[holdout_task] * len(final_completions),
                )
            )
        final_scores = [reward_for(result, reward.knobs) for result in final_results]
        final_best_index = max(range(len(final_results)), key=lambda index: final_scores[index])
        baseline_pass_rate = _pass_rate(baseline_results)
        final_pass_rate = _pass_rate(final_results)
        use_baseline = final_pass_rate < baseline_pass_rate
        best_index = baseline_best_index if use_baseline else final_best_index
        selected_results = baseline_results if use_baseline else final_results
        selected_scores = baseline_scores if use_baseline else final_scores
        selected_completions = baseline_completions if use_baseline else final_completions
        checkpoint_step = int(best_checkpoint.name.rsplit("-", 1)[1])
        emit(
            {
                "event": "best_completion",
                "completion": format_completion_for_display(
                    task, selected_completions[best_index]
                ),
                "reward": selected_scores[best_index],
                "pass_rate": best_validation_pass_rate,
                "holdout_pass_rate": final_pass_rate,
                "baseline_pass_rate": validation_baseline_pass_rate,
                "checkpoint": best_checkpoint.name,
                "best_step": checkpoint_step,
                "train_steps": train_steps,
                "learning_rate": learning_rate,
                "source": "baseline" if use_baseline else "checkpoint",
            },
        )
        del best_model, best_tokenizer
        torch.cuda.empty_cache()
        print(f"\n=== Final evaluation: best checkpoint ({best_checkpoint.name}) ===")
        _print_evaluation_summary("Final holdout", final_results, final_scores)
        emit(
            {
                "event": "complete",
                "train_steps": train_steps,
                "pool_size": pool_size,
                "eval_samples": eval_samples,
            }
        )
        return ExperimentResult(
            baseline_pass_rate=_pass_rate(baseline_results),
            best_validation_pass_rate=best_validation_pass_rate,
            best_checkpoint=best_checkpoint.name,
            final_holdout_pass_rate=_pass_rate(final_results),
        )
    except Exception as exc:
        emit({"event": "error", "message": str(exc)})
        raise
    finally:
        # Daytona's async client was created and used on the reward bridge's
        # event loop. Close it on that same loop before stopping the bridge;
        # creating a second loop here can make socket cleanup fail with
        # ``RuntimeError: Event loop is closed``.
        reward.bridge.run(pool.close())
        reward.close()


def _train_with_progress(emit: Any, trainer: Any, generation_backend: str) -> Any:
    """Run GRPO training; emit heartbeats while vLLM warms the first rollout batch."""
    if generation_backend != "vllm":
        return trainer.train()

    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(10.0):
            emit(
                {
                    "event": "training_progress",
                    "message": "vLLM generating first rollout batch (can take 1–2 min)…",
                }
            )

    thread = threading.Thread(target=heartbeat, name="training-heartbeat", daemon=True)
    thread.start()
    try:
        return trainer.train()
    finally:
        stop.set()
        thread.join(timeout=0.1)


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
    parser.add_argument("--progress-file")
    parser.add_argument(
        "--generation-backend",
        choices=("hf", "vllm"),
        default=None,
        help="GRPO rollout backend (default: TORONTO_GENERATION_BACKEND or hf)",
    )
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
        progress_file=args.progress_file,
        generation_backend=args.generation_backend,
    )


if __name__ == "__main__":
    main()
