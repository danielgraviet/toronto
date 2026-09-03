from __future__ import annotations

from pathlib import Path

from demo.events import DemoState, replay_jsonl


def test_replay_fixture_advances_training_curve() -> None:
    fixture = Path(__file__).parents[1] / "demo" / "fixtures" / "sample_run.jsonl"
    state = replay_jsonl(fixture.read_text(encoding="utf-8"), DemoState())
    assert state.baseline_success_rate == 0.12
    assert len(state.curve) == 2
    assert state.curve[-1].step == 6
    assert state.curve[-1].pass_rate == 0.41
    assert state.best_step == 6
    assert state.lift == 0.45 - 0.12
    assert state.best_success_rate == 0.45
    assert state.baseline_completion is not None
    assert state.holdout_pass_rate == 0.52
    assert state.best_reward == 0.82
    assert state.complete is True


def test_finale_state_has_checkpoint_metadata() -> None:
    fixture = Path(__file__).parents[1] / "demo" / "fixtures" / "sample_run.jsonl"
    state = replay_jsonl(fixture.read_text(encoding="utf-8"), DemoState())
    assert state.checkpoint == "checkpoint-6"
    assert state.learning_rate == 5e-6


def test_rollout_reward_updates_grid() -> None:
    from demo.events import apply_event

    state = DemoState(rollout_size=4, rollouts=["pending"] * 4)
    state = apply_event(
        state,
        {
            "event": "rollout_reward",
            "index": 0,
            "reward": -1.0,
            "num_passed": 0,
            "num_tests": 11,
            "completion_preview": "def two_sum(nums, target):\n    pass",
        },
    )
    state = apply_event(
        state,
        {
            "event": "rollout_reward",
            "index": 1,
            "reward": 0.45,
            "num_passed": 5,
            "num_tests": 11,
            "completion_preview": "def two_sum(nums, target):\n    seen = {}",
        },
    )
    assert state.rollouts[0] == "negative"
    assert state.rollouts[1] == "positive"


def test_sandbox_result_alias_matches_rollout_reward() -> None:
    from demo.events import apply_event

    state = DemoState(rollout_size=4, rollouts=["pending"] * 4)
    state = apply_event(
        state,
        {
            "event": "sandbox_result",
            "index": 2,
            "reward": 0.5,
            "num_passed": 5,
            "num_tests": 10,
            "completion_preview": "def two_sum(): pass",
        },
    )
    assert state.rollouts[2] == "positive"


def test_run_started_and_phase_timing_events() -> None:
    from demo.events import apply_event

    state = DemoState()
    state = apply_event(
        state,
        {"event": "run_started", "generation_backend": "vllm", "train_steps": 6, "task_id": "two_sum_plus"},
    )
    assert state.generation_backend == "vllm"
    assert state.total_steps == 6

    state = apply_event(
        state,
        {"event": "phase_timing", "phase": "training", "seconds": 42.5, "backend": "vllm"},
    )
    assert state.timings["training"] == 42.5
    assert state.timing_backends["training"] == "vllm"
