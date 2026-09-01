"""Small shared grader utilities."""

from __future__ import annotations

import time


def start_timer() -> float:
    """Return a monotonic timestamp suitable for ``elapsed_ms``."""
    return time.perf_counter()


def elapsed_ms(started: float) -> int:
    """Return elapsed wall-clock time since ``started`` in milliseconds."""
    return int((time.perf_counter() - started) * 1000)
