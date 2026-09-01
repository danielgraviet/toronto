from trainer.sweep import SWEEP_CONFIGS


def test_sweep_has_independent_learning_rate_and_step_trials() -> None:
    assert SWEEP_CONFIGS == (
        (2e-6, 8, 4),
        (5e-6, 8, 4),
        (2e-6, 8, 16),
        (5e-6, 8, 16),
    )
