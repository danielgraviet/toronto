"""Generation backend configuration for GRPO training rollouts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

GenerationBackend = Literal["hf", "vllm"]


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    backend: GenerationBackend = "hf"
    vllm_gpu_memory_utilization: float = 0.3
    vllm_enable_sleep_mode: bool = True


def get_generation_config(
    backend: str | None = None,
) -> GenerationConfig:
    """Resolve generation backend from CLI override or ``TORONTO_GENERATION_BACKEND``."""
    selected = (backend or os.getenv("TORONTO_GENERATION_BACKEND", "hf")).lower()
    if selected not in {"hf", "vllm"}:
        raise ValueError(
            f"Unknown generation backend {selected!r}; choose 'hf' or 'vllm'"
        )
    memory_raw = os.getenv("TORONTO_VLLM_GPU_MEMORY_UTILIZATION", "0.3")
    sleep_raw = os.getenv("TORONTO_VLLM_ENABLE_SLEEP_MODE", "1").lower()
    return GenerationConfig(
        backend=selected,  # type: ignore[arg-type]
        vllm_gpu_memory_utilization=float(memory_raw),
        vllm_enable_sleep_mode=sleep_raw not in {"0", "false", "no"},
    )


def grpo_generation_kwargs(config: GenerationConfig) -> dict[str, Any]:
    """Return TRL ``GRPOConfig`` keyword overrides for the selected backend."""
    if config.backend == "hf":
        return {"use_vllm": False}
    return {
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
        "vllm_enable_sleep_mode": config.vllm_enable_sleep_mode,
    }
