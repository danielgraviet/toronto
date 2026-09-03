"""Small environment-backed profiles for rehearsal and stage runs."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunProfile:
    name: str
    pool_size: int
    train_steps: int
    eval_samples: int
    train_batch_size: int
    max_completion_length: int


PROFILES = {
    "stage": RunProfile(
        name="stage",
        pool_size=16,
        train_steps=6,
        eval_samples=16,
        train_batch_size=16,
        max_completion_length=192,
    ),
    "full": RunProfile(
        name="full",
        pool_size=32,
        train_steps=8,
        eval_samples=64,
        train_batch_size=16,
        max_completion_length=256,
    ),
}


def get_profile(name: str | None = None) -> RunProfile:
    selected = name or os.getenv("STAGE_PROFILE", "stage")
    try:
        return PROFILES[selected]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"Unknown run profile {selected!r}; choose {choices}") from exc
