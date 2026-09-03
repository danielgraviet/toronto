"""Frozen demo configuration — no CLI flags on talk day."""

from __future__ import annotations

from dataclasses import dataclass

from runners.gpu import DEFAULT_MODEL_NAME
from trainer.config import get_profile


@dataclass(frozen=True, slots=True)
class DemoConfig:
    task_id: str
    model_name: str
    profile_name: str
    pool_size: int
    train_steps: int
    eval_samples: int
    train_batch_size: int
    max_completion_length: int
    learning_rate: float
    gpu_type: str
    seed: int
    poll_interval: float
    step_eval_samples: int


_PROFILE = get_profile("stage")

DEMO_CONFIG = DemoConfig(
    task_id="two_sum_plus",
    model_name=DEFAULT_MODEL_NAME,
    profile_name="stage",
    pool_size=_PROFILE.pool_size,
    train_steps=_PROFILE.train_steps,
    eval_samples=_PROFILE.eval_samples,
    train_batch_size=4,
    max_completion_length=_PROFILE.max_completion_length,
    learning_rate=5e-6,
    gpu_type="RTX-PRO-6000",
    seed=42,
    poll_interval=0.25,
    step_eval_samples=8,
)
