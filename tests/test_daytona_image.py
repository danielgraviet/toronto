from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from runners.daytona import GPU_IMAGE_BASE, build_gpu_image, validate_gpu_image_spec


def test_build_gpu_image_includes_vllm() -> None:
    image = MagicMock()
    image.pip_install.return_value = image
    image.workdir.return_value = image
    base = MagicMock(return_value=image)
    daytona = SimpleNamespace(Image=SimpleNamespace(base=base))

    with patch.dict("sys.modules", {"daytona": daytona}):
        result = build_gpu_image()

    base.assert_called_once_with(GPU_IMAGE_BASE)
    packages = image.pip_install.call_args.args
    assert "vllm==0.21.0" in packages
    assert "trl==1.5.0" in packages
    image.workdir.assert_called_once_with("/tmp/toronto")
    assert result is image


def test_gpu_image_spec_validates_in_ci() -> None:
    validate_gpu_image_spec()
