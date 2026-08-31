from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import time
from typing import Any

from toronto.tasks.loader import Task

from .models import EvalResult, RewardKnobs


def sanitize_completion(raw: str, func_name: str) -> str:
    text = raw.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text).strip()
    marker = re.search(rf"^\s*def\s+{re.escape(func_name)}\b[^:]*:\s*$", text, re.MULTILINE)
    if marker:
        text = text[marker.end():]
    lines = text.splitlines()
    kept: list[str] = []
    for line in lines:
        if not line.strip():
            if kept:
                kept.append(line)
            continue
        if line.startswith((" ", "\t")):
            kept.append(line)
        elif kept:
            break
    return "\n".join(kept).rstrip()


def has_banned_pattern(body: str, task: Task) -> bool:
    return any(pattern in body for pattern in task.banned_patterns)


def _literal(value: Any) -> str:
    return repr(value)


def build_test_harness(task: Task, body: str) -> str:
    normal = []
    for case in task.tests:
        args = case["input"]
        call = f"{task.func_name}(**{_literal(args)})" if isinstance(args, dict) else f"{task.func_name}({_literal(args)})"
        normal.append(f"check(lambda: {call}, {_literal(case['output'])})")
    errors = []
    for case in task.error_tests:
        args = case["input"]
        call = f"{task.func_name}({_literal(args)})"
        expected = case.get("raises", "Exception").__name__ if isinstance(case.get("raises"), type) else str(case.get("raises", "Exception"))
        errors.append(f"check_raises(lambda: {call}, {expected})")
    checks = ",\n        ".join(normal + errors)
    return f"""{task.prompt}\n{body}\n\nimport json\n\ndef check(fn, expected):\n    try:\n        return fn() == expected\n    except Exception:\n        return False\n\ndef check_raises(fn, expected_name):\n    try:\n        fn()\n    except Exception as exc:\n        return type(exc).__name__ == expected_name\n    return False\n\nresults = [\n        {checks}\n    ]\nprint(json.dumps({{'results': results}}))\n"""


def parse_results(stdout: str, expected_tests: int) -> tuple[int, int] | None:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        return None
    try:
        payload = json.loads(lines[-1])
        results = payload["results"]
        if not isinstance(results, list) or len(results) != expected_tests or not all(isinstance(x, bool) for x in results):
            return None
        return sum(results), len(results)
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def evaluate_local(task: Task, completion: str, timeout_seconds: float = 1.0) -> EvalResult:
    started = time.perf_counter()
    body = sanitize_completion(completion, task.func_name)
    if not body or has_banned_pattern(body, task):
        return EvalResult(False, 0, task.total_tests, _elapsed(started), banned=bool(body and has_banned_pattern(body, task)), completion=completion)
    try:
        ast.parse(task.prompt + "\n" + body)
        process = subprocess.run([sys.executable, "-c", build_test_harness(task, body)], capture_output=True, text=True, timeout=timeout_seconds)
    except (subprocess.TimeoutExpired, OSError, SyntaxError) as exc:
        return EvalResult(False, 0, task.total_tests, _elapsed(started), error=type(exc).__name__, completion=completion)
    parsed = parse_results(process.stdout, task.total_tests)
    if process.returncode != 0 or parsed is None:
        return EvalResult(False, 0, task.total_tests, _elapsed(started), error="harness_failed", completion=completion)
    passed, total = parsed
    return EvalResult(True, passed, total, _elapsed(started), completion=completion)


def reward_for(result: EvalResult, knobs: RewardKnobs) -> float:
    if not result.no_error or result.banned:
        return -1.0
    base = result.num_passed / result.num_tests if result.num_tests else 0.0
    lines = max(1, len(result.completion.splitlines()))
    normalized_lines = min(1.0, lines / 20.0)
    speed_term = max(0.0, 1.0 - result.duration_ms / 1000.0)
    reward = base - knobs.lambda_len * normalized_lines + knobs.lambda_speed * speed_term
    return max(-1.0, min(1.0, reward))


def _elapsed(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)
