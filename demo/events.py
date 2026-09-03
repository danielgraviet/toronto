"""Parse progress JSONL into display state."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

Phase = Literal[
    "starting",
    "provisioning",
    "model_loading",
    "pool_warming",
    "baseline",
    "collecting",
    "computing_returns",
    "training",
    "policy_update",
    "evaluating",
    "complete",
    "error",
]

RolloutState = Literal["pending", "positive", "negative"]


@dataclass
class TrajectoryEntry:
    index: int
    reward: float
    num_passed: int
    num_tests: int
    preview: str
    note: str = ""


@dataclass
class CurvePoint:
    step: int
    mean_reward: float
    pass_rate: float


@dataclass
class DemoState:
    phase: Phase = "starting"
    phase_message: str = "Starting trainer…"
    task_id: str = "two_sum_plus"
    model_name: str = "Qwen/Qwen2.5-0.5B"
    training_step: int = 0
    total_steps: int = 6
    mean_return: float | None = None
    success_rate: float | None = None
    baseline_success_rate: float | None = None
    best_success_rate: float | None = None
    best_step: int | None = None
    lift: float | None = None
    rollout_size: int = 16
    rollouts: list[RolloutState] = field(default_factory=list)
    curve: list[CurvePoint] = field(default_factory=list)
    trajectories: list[TrajectoryEntry] = field(default_factory=list)
    cycle_phase: Literal["collect", "reward", "update"] = "collect"
    error: str | None = None
    complete: bool = False
    best_completion: str | None = None
    best_reward: float | None = None
    baseline_completion: str | None = None
    baseline_sample_reward: float | None = None
    holdout_pass_rate: float | None = None
    checkpoint: str | None = None
    learning_rate: float | None = None
    pool_size: int = 16
    eval_samples: int = 16
    source: str | None = None
    generation_backend: str = "hf"
    timings: dict[str, float] = field(default_factory=dict)
    timing_backends: dict[str, str] = field(default_factory=dict)


def parse_event(line: str) -> dict[str, Any] | None:
    line = line.strip()
    if not line.startswith("{"):
        return None
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "event" not in payload:
        return None
    return payload


def _rollout_state(reward: float) -> RolloutState:
    return "positive" if reward > 0 else "negative"


def _trajectory_note(event: dict[str, Any]) -> str:
    if event.get("banned"):
        return "constraint violation"
    if event.get("error"):
        return str(event["error"]).replace("_", " ")
    passed = int(event.get("num_passed", 0))
    total = int(event.get("num_tests", 0))
    if total:
        return f"{passed}/{total} checks"
    return ""


def apply_event(state: DemoState, event: dict[str, Any]) -> DemoState:
    name = str(event.get("event", ""))
    if name == "run_started":
        state.generation_backend = str(event.get("generation_backend", state.generation_backend))
        if event.get("train_steps") is not None:
            state.total_steps = int(event["train_steps"])
        if event.get("task_id"):
            state.task_id = str(event["task_id"])
    elif name == "phase_timing":
        phase = str(event.get("phase", "unknown"))
        state.timings[phase] = float(event.get("seconds", 0))
        state.timing_backends[phase] = str(event.get("backend", "hf"))
    elif name == "provisioning":
        state.phase = "provisioning"
        state.phase_message = "Starting trainer…"
    elif name == "model_loading":
        state.phase = "model_loading"
        state.phase_message = "Loading policy π…"
    elif name == "model_ready":
        state.phase = "pool_warming"
        state.phase_message = "Warming environment…"
    elif name == "pool_warming":
        state.phase = "pool_warming"
        state.phase_message = "Warming environment…"
    elif name == "pool_ready":
        state.phase_message = "Environment ready"
    elif name == "baseline_started":
        state.phase = "baseline"
        state.phase_message = "Initial policy eval"
        state.cycle_phase = "collect"
    elif name == "generation_started":
        state.phase = "collecting"
        state.phase_message = "Collecting rollouts…"
        state.cycle_phase = "collect"
        count = int(event.get("count", state.rollout_size))
        state.rollout_size = count
        state.rollouts = ["pending"] * count
    elif name == "generation_finished":
        state.phase = "computing_returns"
        state.phase_message = "Computing returns…"
        state.cycle_phase = "reward"
    elif name == "grading_started":
        state.phase = "computing_returns"
        state.phase_message = "Computing returns…"
        state.cycle_phase = "reward"
    elif name in {"rollout_reward", "sandbox_result"}:
        index = int(event["index"])
        reward = float(event.get("reward", 0))
        while len(state.rollouts) <= index:
            state.rollouts.append("pending")
        state.rollouts[index] = _rollout_state(reward)
        entry = TrajectoryEntry(
            index=index,
            reward=reward,
            num_passed=int(event.get("num_passed", 0)),
            num_tests=int(event.get("num_tests", 0)),
            preview=str(event.get("completion_preview", "")),
            note=_trajectory_note(event),
        )
        state.trajectories = [entry, *[item for item in state.trajectories if item.index != index]][
            :12
        ]
    elif name == "step_started":
        state.phase = "training"
        state.training_step = int(event.get("step", state.training_step))
        state.total_steps = int(event.get("total_steps", state.total_steps))
        state.phase_message = f"Training step {state.training_step}/{state.total_steps}"
        state.cycle_phase = "collect"
        count = int(event.get("count", state.rollout_size))
        state.rollout_size = count
        state.rollouts = ["pending"] * count
    elif name == "baseline_finished":
        state.baseline_success_rate = float(event.get("pass_rate", 0))
        state.success_rate = state.baseline_success_rate
        if event.get("sample_completion"):
            state.baseline_completion = str(event["sample_completion"])
        if event.get("sample_reward") is not None:
            state.baseline_sample_reward = float(event["sample_reward"])
        state.phase = "baseline"
        state.phase_message = "Initial policy eval complete"
        state.cycle_phase = "update"
    elif name == "trainer_initializing":
        state.phase = "training"
        backend = str(event.get("generation_backend", state.generation_backend))
        state.generation_backend = backend
        if backend == "vllm":
            state.phase_message = "Initializing vLLM engine…"
        else:
            state.phase_message = "Preparing GRPO trainer…"
        state.cycle_phase = "collect"
    elif name == "training_started":
        state.phase = "training"
        state.total_steps = int(event.get("steps", state.total_steps))
        backend = str(event.get("generation_backend", state.generation_backend))
        state.generation_backend = backend
        rollouts = int(event.get("rollouts_per_step", state.rollout_size))
        state.rollout_size = rollouts
        state.rollouts = ["pending"] * rollouts
        state.trajectories = []
        if backend == "vllm":
            state.phase_message = "GRPO training (vLLM rollouts)"
        else:
            state.phase_message = "GRPO training"
        state.cycle_phase = "collect"
    elif name == "training_progress":
        state.phase = "training"
        state.phase_message = str(event.get("message", state.phase_message))
        state.cycle_phase = "collect"
    elif name == "step_finished":
        state.phase = "training"
        state.training_step = int(event.get("step", state.training_step))
        state.mean_return = float(event.get("mean_reward", 0))
        state.cycle_phase = "update"
    elif name == "step_evaluated":
        state.phase = "training"
        step = int(event.get("step", state.training_step))
        state.training_step = step
        success = float(event.get("pass_rate", 0))
        state.success_rate = success
        if event.get("mean_reward") is not None:
            state.mean_return = float(event["mean_reward"])
        point = CurvePoint(
            step=step,
            mean_reward=state.mean_return or 0.0,
            pass_rate=success,
        )
        state.curve = [item for item in state.curve if item.step != step] + [point]
        state.curve.sort(key=lambda item: item.step)
        if state.baseline_success_rate is not None:
            state.lift = success - state.baseline_success_rate
        if state.best_success_rate is None or success >= state.best_success_rate:
            state.best_success_rate = success
            state.best_step = step
        state.phase_message = f"Training step {state.training_step}/{state.total_steps}"
        state.cycle_phase = "update"
    elif name == "best_completion":
        state.best_completion = str(event.get("completion", ""))
        state.best_reward = float(event.get("reward", 0))
        validation_rate = float(event.get("pass_rate", state.success_rate or 0))
        state.success_rate = validation_rate
        if event.get("holdout_pass_rate") is not None:
            state.holdout_pass_rate = float(event["holdout_pass_rate"])
        if event.get("baseline_pass_rate") is not None:
            state.baseline_success_rate = float(event["baseline_pass_rate"])
        if event.get("checkpoint"):
            state.checkpoint = str(event["checkpoint"])
        if event.get("best_step") is not None:
            state.best_step = int(event["best_step"])
        if event.get("train_steps") is not None:
            state.total_steps = int(event["train_steps"])
        if event.get("learning_rate") is not None:
            state.learning_rate = float(event["learning_rate"])
        if event.get("source"):
            state.source = str(event["source"])
        if state.best_success_rate is None or validation_rate > state.best_success_rate:
            state.best_success_rate = validation_rate
        if state.baseline_success_rate is not None:
            state.lift = (state.best_success_rate or validation_rate) - state.baseline_success_rate
        state.phase = "evaluating"
        state.phase_message = "Best trajectory selected"
    elif name == "complete":
        state.phase = "complete"
        state.phase_message = "Training complete"
        state.complete = True
        state.cycle_phase = "update"
        if event.get("train_steps") is not None:
            state.total_steps = int(event["train_steps"])
        if event.get("pool_size") is not None:
            state.pool_size = int(event["pool_size"])
        if event.get("eval_samples") is not None:
            state.eval_samples = int(event["eval_samples"])
    elif name == "error":
        state.phase = "error"
        state.error = str(event.get("message", "unknown error"))
        state.phase_message = state.error
    return state


def replay_events(events: list[dict[str, Any]], initial: DemoState | None = None) -> DemoState:
    state = initial or DemoState()
    for event in events:
        state = apply_event(state, event)
    return state


def replay_jsonl(text: str, initial: DemoState | None = None) -> DemoState:
    events = []
    for line in text.splitlines():
        event = parse_event(line)
        if event:
            events.append(event)
    return replay_events(events, initial)
