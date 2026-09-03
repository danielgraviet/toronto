from __future__ import annotations

from demo.events import DemoState, apply_event
from trainer.smoke import _train_with_progress


def test_trainer_initializing_vllm_message() -> None:
    state = DemoState()
    state = apply_event(
        state,
        {"event": "trainer_initializing", "generation_backend": "vllm", "train_steps": 6},
    )
    assert state.phase == "training"
    assert "vLLM" in state.phase_message


def test_training_started_resets_rollout_grid() -> None:
    state = DemoState(
        rollout_size=16,
        rollouts=["positive"] * 16,
        trajectories=[],
        baseline_success_rate=0.44,
    )
    state = apply_event(
        state,
        {
            "event": "training_started",
            "steps": 6,
            "generation_backend": "vllm",
            "rollouts_per_step": 16,
        },
    )
    assert state.rollout_size == 16
    assert state.rollouts == ["pending"] * 16
    assert state.trajectories == []


def test_training_progress_updates_message() -> None:
    state = DemoState()
    state = apply_event(
        state,
        {"event": "training_progress", "message": "vLLM generating first rollout batch…"},
    )
    assert "vLLM" in state.phase_message


def test_train_with_progress_hf_skips_heartbeat() -> None:
    class Trainer:
        def train(self):
            return "ok"

    events: list[dict] = []
    result = _train_with_progress(events.append, Trainer(), "hf")
    assert result == "ok"
    assert events == []
