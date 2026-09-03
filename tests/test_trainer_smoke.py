from __future__ import annotations

from trainer.smoke import _evaluation_weights
from trainer.config import get_profile


def test_evaluation_weights_match_split_case_count() -> None:
    assert _evaluation_weights(({"input": 1},), ({"input": 2},)) == (1.0, 1.0)


def test_stage_profile_is_short_and_stage_sized() -> None:
    profile = get_profile("stage")

    assert profile.pool_size == 16
    assert profile.train_steps == 6
    assert profile.eval_samples == 16
    assert profile.train_batch_size == 16
    assert profile.max_completion_length == 192


def test_real_smoke_entrypoint_has_all_runtime_dependencies() -> None:
    """Catch missing imports before a remote GPU is provisioned."""
    from runners.gpu import validate_real_smoke_entrypoint

    validate_real_smoke_entrypoint()
