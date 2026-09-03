"""Small local control API for the stage UI.

The browser never receives Daytona or Hugging Face credentials. This process
owns them and launches the existing GPU runner as a child process.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from tasks import TaskLoader

ROOT = Path(__file__).parents[1]
load_dotenv(ROOT / ".env")
app = FastAPI(title="Toronto stage control API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def start_stage_warmup() -> None:
    """Start the default stage baseline while the operator sets up the UI."""
    if os.getenv("TORONTO_AUTO_START", "1").lower() not in {"0", "false", "no"}:
        _start_run(baseline_only=True)

@app.on_event("shutdown")
async def cleanup_stage_sandbox() -> None:
    """Release the persistent stage GPU sandbox when the API exits."""
    global _process
    with _lock:
        process, sandbox_id = _process, _state.get("gpu_sandbox_id")
    if process and process.poll() is None:
        process.terminate()
        process.wait(timeout=15)
    if sandbox_id:
        from runners.daytona import DaytonaRunner

        runner = DaytonaRunner()
        try:
            sandbox = await runner.get(str(sandbox_id))
            await runner.delete(sandbox)
        except Exception as exc:
            with _lock:
                _state["error"] = f"Could not clean up GPU sandbox {sandbox_id}: {exc}"
        finally:
            await runner.close()

_lock = threading.Lock()
_process: subprocess.Popen[str] | None = None
_state: dict[str, Any] = {
    "phase": "idle",
    "running": False,
    "task_id": "two_sum_plus",
    "gpu_sandbox_id": None,
    "profile": os.getenv("TORONTO_PROFILE", "stage"),
    "baseline_pass_rate": None,
    "current_pass_rate": None,
    "best_completion": None,
    "best_reward": None,
    "best_source": None,
    "curve": [],
    "total_steps": 6 if os.getenv("TORONTO_PROFILE", "stage") == "stage" else 8,
    "logs": [],
    "error": None,
}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "running": _state["running"], "phase": _state["phase"]}


@app.get("/state")
def state() -> dict[str, Any]:
    with _lock:
        return {"ok": True, "state": json.loads(json.dumps(_state))}


@app.get("/tasks")
def tasks() -> dict[str, Any]:
    return {
        "ok": True,
        "tasks": [
            {"id": task.id, "title": task.title, "blurb": task.audience_blurb}
            for task in TaskLoader(ROOT / "tasks").list()
        ],
    }


@app.post("/task/lock")
def lock_task(payload: dict[str, Any]) -> dict[str, Any]:
    task_id = str(payload.get("task_id", ""))
    TaskLoader(ROOT / "tasks").load(task_id)
    with _lock:
        _state["task_id"] = task_id
    return state()


@app.post("/reward/knobs")
def reward_knobs(payload: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        _state["reward_knobs"] = {
            "lambda_len": float(payload.get("lambda_len", 0)),
            "lambda_ban": float(payload.get("lambda_ban", 0)),
            "lambda_speed": float(payload.get("lambda_speed", 0)),
        }
    return state()


@app.post("/baseline")
def baseline(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return _start_run(baseline_only=True)


@app.post("/train")
def train(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    steps = int((payload or {}).get("steps", _state["total_steps"]))
    return _start_run(baseline_only=False, steps=max(1, min(steps, 8)))


@app.post("/stop")
def stop() -> dict[str, Any]:
    global _process
    with _lock:
        process = _process
    if process and process.poll() is None:
        process.terminate()
        with _lock:
            _state["phase"] = "stopped"
            _state["running"] = False
    return state()


@app.post("/mode")
def mode(payload: dict[str, Any]) -> dict[str, Any]:
    with _lock:
        _state["mode"] = str(payload.get("mode", "live"))
    return state()


@app.websocket("/ws")
async def websocket(websocket: WebSocket) -> None:
    import asyncio

    await websocket.accept()
    while True:
        await websocket.send_json(state()["state"])
        await asyncio.sleep(0.5)


def _start_run(*, baseline_only: bool, steps: int = 4) -> dict[str, Any]:
    global _process
    with _lock:
        if _state["running"]:
            # Startup warmup may already be doing the run requested by the page.
            already_running = True
        else:
            already_running = False
    if already_running:
        return state()
    with _lock:
        task_id = _state["task_id"]
        command = [
            sys.executable,
            "-u",
            "-m",
            "runners.gpu",
            "--remote",
            "--real-grpo-smoke",
            "--profile",
            str(_state["profile"]),
            "--gpu-type",
            os.getenv("TORONTO_GPU_TYPE", "RTX-PRO-6000"),
            "--task-id",
            task_id,
        ]
        if _state["gpu_sandbox_id"]:
            command.extend(["--sandbox-id", str(_state["gpu_sandbox_id"])])
        # The first baseline leaves the GPU sandbox available for training.
        command.append("--keep")
        if baseline_only:
            command.append("--baseline-only")
        else:
            command.extend(["--train-steps", str(steps)])
        _state.update(
            phase="queued",
            running=True,
            baseline_pass_rate=None if not baseline_only else _state["baseline_pass_rate"],
            current_pass_rate=None,
            best_completion=None,
            best_reward=None,
            best_source=None,
            curve=[],
            logs=[],
            error=None,
        )
        try:
            _process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env={**os.environ, "PYTHONPATH": str(ROOT)},
            )
        except OSError as exc:
            _state.update(phase="error", running=False, error=f"Could not start GPU runner: {exc}")
            return state()
        threading.Thread(target=_watch_process, args=(_process,), daemon=True).start()
    return state()


def _watch_process(process: subprocess.Popen[str]) -> None:
    global _process
    assert process.stdout is not None
    for raw_line in process.stdout:
        line = raw_line.strip()
        if not line:
            continue
        with _lock:
            _state["logs"] = (_state["logs"] + [line])[-30:]
            progress = _parse_progress(line)
            if progress:
                event = progress.get("event")
                if event == "model_loading":
                    _state["phase"] = "model_loading"
                elif event == "model_ready":
                    _state["phase"] = "baseline"
                elif event == "baseline_started":
                    _state["phase"] = "baseline"
                elif event == "baseline_finished":
                    _state["phase"] = "baseline"
                    _state["baseline_pass_rate"] = progress.get("pass_rate")
                    _state["current_pass_rate"] = progress.get("pass_rate")
                elif event == "training_started":
                    _state["phase"] = "training"
                elif event == "step_finished":
                    _state["phase"] = "training"
                    _state["curve"].append(
                        {
                            "step": int(progress["step"]),
                            "mean_reward": float(progress["mean_reward"]),
                            "pass_rate": float(progress["pass_rate"]),
                        }
                    )
                    _state["current_pass_rate"] = float(progress["pass_rate"])
                elif event == "best_completion":
                    _state["best_completion"] = str(progress.get("completion", ""))
                    _state["best_reward"] = float(progress.get("reward", 0))
                    _state["best_source"] = str(progress.get("source", "checkpoint"))
                    _state["current_pass_rate"] = float(progress.get("pass_rate", 0))
            elif "Created remote sandbox" in line:
                match = re.search(r"Created remote sandbox:\s*(\S+)", line)
                if match:
                    _state["gpu_sandbox_id"] = match.group(1)
                _state["phase"] = "provisioning"
            elif "Reusing remote sandbox" in line:
                _state["phase"] = "provisioning"
            elif "Baseline" in line and "pass rate:" in line:
                _state["phase"] = "baseline"
                _state["baseline_pass_rate"] = _percent(line)
                _state["current_pass_rate"] = _percent(line)
            elif match := re.search(r"Step (\d+): mean reward=([-0-9.]+), pass rate=([0-9.]+)%", line):
                _state["phase"] = "training"
                _state["curve"].append(
                    {
                        "step": int(match.group(1)),
                        "mean_reward": float(match.group(2)),
                        "pass_rate": float(match.group(3)) / 100,
                    }
                )
                _state["current_pass_rate"] = float(match.group(3)) / 100
            elif "Final holdout" in line and "pass rate:" in line:
                _state["phase"] = "complete"
                _state["current_pass_rate"] = _percent(line)
            elif "Deleted remote sandbox" in line:
                _state["phase"] = "complete" if process.returncode == 0 else "error"
    code = process.wait()
    with _lock:
        _state["running"] = False
        _state["phase"] = "complete" if code == 0 else "error"
        if code != 0:
            _state["error"] = "GPU runner exited with code " + str(code)
        _process = None


def _percent(line: str) -> float | None:
    match = re.search(r"pass rate:\s*([0-9.]+)%", line)
    return float(match.group(1)) / 100 if match else None


def _parse_progress(line: str) -> dict[str, Any] | None:
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) and "event" in value else None
