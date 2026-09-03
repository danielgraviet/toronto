"""JSONL progress events for live CLI and control API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable


class ProgressEmitter:
    """Write structured progress events to a JSONL file and optional callback."""

    def __init__(
        self,
        path: str | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._path = path
        self._on_event = on_event

    def emit(self, payload: dict[str, Any]) -> None:
        if self._path:
            with Path(self._path).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(payload) + "\n")
                stream.flush()
        if self._on_event:
            self._on_event(payload)


def truncate_preview(text: str, max_lines: int = 4, max_chars: int = 280) -> str:
    """Short trajectory snippet for the terminal log."""
    lines = text.strip().splitlines()[:max_lines]
    preview = "\n".join(lines)
    if len(preview) > max_chars:
        return preview[: max_chars - 3] + "..."
    return preview


def format_completion_for_display(task: Any, completion: str) -> str:
    """Extract the graded function body and render a readable solution for the finale."""
    from graders.core import sanitize_completion

    body = sanitize_completion(completion, task.func_name)
    if not body:
        return truncate_preview(completion, max_lines=16, max_chars=600)
    for line in task.prompt.splitlines():
        if line.strip().startswith(f"def {task.func_name}"):
            return f"{line.rstrip()}\n{body}"
    return f"def {task.func_name}(...):\n{body}"
