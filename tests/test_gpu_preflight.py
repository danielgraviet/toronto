from __future__ import annotations

import pytest

from runners.daytona import GPU_IMAGE_BASE, GPU_PIP_PACKAGES, validate_gpu_image_spec
from runners.gpu import format_remote_failure, remote_preflight_script


def test_validate_gpu_image_spec_passes() -> None:
    validate_gpu_image_spec()
    assert "cuda13" in GPU_IMAGE_BASE
    assert "vllm==0.21.0" in GPU_PIP_PACKAGES


def test_validate_gpu_image_spec_rejects_wrong_cuda() -> None:
    import runners.daytona as daytona_module

    original = daytona_module.GPU_IMAGE_BASE
    try:
        daytona_module.GPU_IMAGE_BASE = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime"
        with pytest.raises(ValueError, match="CUDA 13"):
            validate_gpu_image_spec()
    finally:
        daytona_module.GPU_IMAGE_BASE = original


def test_remote_preflight_script_hf() -> None:
    script = remote_preflight_script("hf")
    assert 'importlib.import_module("trl")' in script
    assert "vllm" not in script
    assert "preflight_ok" in script


def test_remote_preflight_script_vllm() -> None:
    script = remote_preflight_script("vllm")
    assert 'importlib.import_module("vllm")' in script
    assert "grpo_trainer" in script
    assert "preflight_ok" in script


def test_format_remote_failure_includes_tail() -> None:
    log = "\n".join(f"line {index}" for index in range(50))
    message = format_remote_failure(log, tail_lines=5)
    assert "Remote trainer failed" in message
    assert "line 49" in message
    assert "line 0" not in message
