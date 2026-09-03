from __future__ import annotations

from tasks import TaskLoader
from trainer.progress import format_completion_for_display


def test_format_completion_for_display_keeps_one_clean_function() -> None:
    task = TaskLoader().load("two_sum_plus")
    raw = """def two_sum(nums, target):
    seen = {}
    for i, n in enumerate(nums):
        if target - n in seen:
            return [seen[target - n], i]
        seen[n] = i
    return []

def two_sum(nums, target):
    for i in range(len(nums)):
        pass

print(two_sum([1, 2], 3))"""
    formatted = format_completion_for_display(task, raw)
    assert formatted.count("def two_sum") == 1
    assert "print(" not in formatted
    assert "seen = {}" in formatted
    assert formatted.startswith("def two_sum(nums: list[int], target: int)")
