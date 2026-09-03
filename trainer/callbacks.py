"""TRL training callbacks for live demo metrics."""

from __future__ import annotations

from typing import Any, Callable

from graders.models import EvalResult
from runners.gpu import generate_completions


def build_step_validation_callback(
    *,
    tokenizer: Any,
    validation_task: Any,
    pool: Any,
    bridge: Any,
    emit: Callable[[dict[str, Any]], None],
    eval_samples: int,
    max_new_tokens: int,
    seed: int,
    torch: Any,
) -> Any:
    """Return a ``transformers.TrainerCallback`` for per-step validation metrics."""
    from transformers import TrainerCallback

    class StepValidationCallback(TrainerCallback):
        """After each GRPO step, evaluate on the validation split for comparable success rates."""

        def __init__(self) -> None:
            self._latest_mean_reward: float | None = None

        def set_latest_mean_reward(self, value: float) -> None:
            self._latest_mean_reward = value

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
            model = kwargs.get("model")
            if model is None:
                return
            step = int(state.global_step)
            was_training = model.training
            model.eval()
            try:
                torch.manual_seed(seed + step)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed + step)
                with torch.inference_mode():
                    completions = generate_completions(
                        model,
                        tokenizer,
                        validation_task.prompt,
                        num_completions=eval_samples,
                        max_new_tokens=max_new_tokens,
                    )
                results: list[EvalResult] = bridge.run(
                    pool.evaluate_batch(
                        completions,
                        [validation_task.prompt] * len(completions),
                        tasks=[validation_task] * len(completions),
                    )
                )
            finally:
                if was_training:
                    model.train()
            pass_rate = sum(item.num_passed for item in results) / max(
                1, sum(item.num_tests for item in results)
            )
            emit(
                {
                    "event": "step_evaluated",
                    "step": step,
                    "pass_rate": pass_rate,
                    "mean_reward": self._latest_mean_reward,
                }
            )

    return StepValidationCallback()
