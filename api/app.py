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

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from tasks import TaskLoader

ROOT = Path(__file__).parents[1]
app = FastAPI(title="Toronto stage control API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_lock = threading.Lock()
_process: subprocess.Popen[str] | None = None
_state: dict[str, Any] = {
    "phase": "idle",
    "running": False,
    "task_id": "two_sum_plus",
    "profile": "stage",
    "baseline_pass_rate": None,
    "current_pass_rate": None,
    "curve": [],
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
    steps = int((payload or {}).get("steps", 4))
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
            return {"ok": False, "error": "a run is already active"}
        task_id = _state["task_id"]
        command = [
            sys.executable,
            "-m",
            "runners.gpu",
            "--remote",
            "--real-grpo-smoke",
            "--profile",
            "stage",
            "--gpu-type",
            os.getenv("TORONTO_GPU_TYPE", "RTX-PRO-6000"),
            "--task-id",
            task_id,
        ]
        if baseline_only:
            command.append("--baseline-only")
        else:
            command.extend(["--train-steps", str(steps)])
        _state.update(
            phase="queued",
            running=True,
            baseline_pass_rate=None if not baseline_only else _state["baseline_pass_rate"],
            current_pass_rate=None,
            curve=[],
            logs=[],
            error=None,
        )
        _process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=os.environ.copy(),
        )
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
            if "Created remote sandbox" in line:
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
