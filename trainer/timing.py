"""Phase timing helpers for progress events."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any


@contextmanager
def timed_phase(
    emit: Callable[[dict[str, Any]], None],
    phase: str,
    backend: str,
) -> Iterator[None]:
    """Emit a ``phase_timing`` event when the wrapped block finishes."""
    started = time.monotonic()
    try:
        yield
    finally:
        emit(
            {
                "event": "phase_timing",
                "phase": phase,
                "seconds": round(time.monotonic() - started, 2),
                "backend": backend,
            }
        )
