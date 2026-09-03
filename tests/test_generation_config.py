from __future__ import annotations

import os

import pytest

from trainer.generation import GenerationConfig, get_generation_config, grpo_generation_kwargs


def test_default_backend_is_hf(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TORONTO_GENERATION_BACKEND", raising=False)
    config = get_generation_config()
    assert config.backend == "hf"
    assert grpo_generation_kwargs(config) == {"use_vllm": False}


def test_vllm_backend_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORONTO_GENERATION_BACKEND", "vllm")
    monkeypatch.setenv("TORONTO_VLLM_GPU_MEMORY_UTILIZATION", "0.25")
    config = get_generation_config()
    assert config.backend == "vllm"
    assert config.vllm_gpu_memory_utilization == 0.25
    kwargs = grpo_generation_kwargs(config)
    assert kwargs["use_vllm"] is True
    assert kwargs["vllm_mode"] == "colocate"
    assert kwargs["vllm_gpu_memory_utilization"] == 0.25


def test_cli_override_beats_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORONTO_GENERATION_BACKEND", "vllm")
    config = get_generation_config("hf")
    assert config.backend == "hf"


def test_invalid_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TORONTO_GENERATION_BACKEND", "banana")
    with pytest.raises(ValueError, match="Unknown generation backend"):
        get_generation_config()
