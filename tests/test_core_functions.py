from __future__ import annotations

import ast
from pathlib import Path

import pytest

from graders.core import (
    build_test_harness,
    evaluate_local,
    has_banned_pattern,
    parse_results,
    reward_for,
    sanitize_completion,
)
from graders.models import EvalResult, RewardKnobs
from tasks import TaskLoader

TASKS_DIR = Path(__file__).parents[1] / "tasks"
loader = TaskLoader(TASKS_DIR)
fizzbuzz = loader.load("fizzbuzz_plus")
safe_parser = loader.load("safe_parser")

GOOD_COMPLETION = """    if n % 3 == 0 and n % 5 == 0:
        return "FizzBuzz"
    if n % 3 == 0:
        return "Fizz"
    if n % 5 == 0:
        return "Buzz"
    if n % 7 == 0:
        return "Daytona"
    return str(n)
"""


def test_sanitize_completion_preserves_first_line_indentation() -> None:
    body = sanitize_completion(GOOD_COMPLETION, "fizzbuzz")

    assert body.splitlines()[0] == "    if n % 3 == 0 and n % 5 == 0:"
    assert body.splitlines()[-1] == "    return str(n)"


def test_sanitize_completion_extracts_function_body_and_stops_at_top_level_code() -> None:
    raw = '''def fizzbuzz(n: int) -> str:
    """Return a label."""
    return str(n)

print(fizzbuzz(1))
'''

    assert sanitize_completion(raw, "fizzbuzz") == '    """Return a label."""\n    return str(n)'


def test_sanitize_completion_removes_python_fences() -> None:
    raw = "```python\n" + GOOD_COMPLETION + "\n```"

    assert sanitize_completion(raw, "fizzbuzz") == GOOD_COMPLETION.rstrip("\n")


@pytest.mark.parametrize("raw", ["", "not indented code", "```python\n```"])
def test_sanitize_completion_returns_empty_for_no_function_body(raw: str) -> None:
    assert sanitize_completion(raw, "fizzbuzz") == ""


def test_has_banned_pattern_detects_configured_patterns() -> None:
    assert has_banned_pattern("    return eval(value)", fizzbuzz)
    assert has_banned_pattern("    import os\n    return os.getenv('X')", fizzbuzz)


def test_has_banned_pattern_allows_normal_code() -> None:
    assert not has_banned_pattern(GOOD_COMPLETION, fizzbuzz)
    assert not has_banned_pattern("    return str(n)", fizzbuzz)


def test_build_test_harness_contains_fizzbuzz_checks_and_is_valid_python() -> None:
    harness = build_test_harness(fizzbuzz, GOOD_COMPLETION)

    ast.parse(harness)
    assert "def fizzbuzz(n: int) -> str:" in harness
    assert "check(lambda: fizzbuzz(105), 'FizzBuzz')" in harness
    assert "print(json.dumps({'results': results}))" in harness


def test_build_test_harness_supports_keyword_inputs_and_expected_errors() -> None:
    body = '''    if s.strip() == "":
        raise ValueError("empty")
    return [int(part.strip()) for part in s.split(",")]
'''
    harness = build_test_harness(safe_parser, body)

    ast.parse(harness)
    assert "parse_int_csv('1,2,3')" in harness
    assert "check_raises(lambda: parse_int_csv('1,,2'), ValueError)" in harness
    assert "check_raises(lambda: parse_int_csv(''), ValueError)" in harness


@pytest.mark.parametrize(
    ("stdout", "expected", "result"),
    [
        ('{"results": [true, false, true]}', 3, (2, 3)),
        ('debug output\n{"results": [true, true]}\n', 2, (2, 2)),
        ('{"results": []}', 0, (0, 0)),
    ],
)
def test_parse_results_accepts_last_json_line(
    stdout: str, expected: int, result: tuple[int, int]
) -> None:
    assert parse_results(stdout, expected) == result


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("", 1),
        ('{"results": [true]}', 2),
        ('{"results": [1, 0]}', 2),
        ('{"result": [true]}', 1),
        ("not json", 1),
    ],
)
def test_parse_results_rejects_malformed_or_wrong_shape(stdout: str, expected: int) -> None:
    assert parse_results(stdout, expected) is None


def test_evaluate_local_grades_known_good_fizzbuzz() -> None:
    result = evaluate_local(fizzbuzz, GOOD_COMPLETION)

    assert result.no_error
    assert not result.banned
    assert (result.num_passed, result.num_tests) == (9, 9)


def test_evaluate_local_turns_syntax_errors_into_failed_result() -> None:
    result = evaluate_local(fizzbuzz, "    return (")

    assert not result.no_error
    assert result.num_passed == 0
    assert result.num_tests == fizzbuzz.total_tests
    assert result.error == "SyntaxError"


def test_reward_for_uses_pass_rate_as_base() -> None:
    result = EvalResult(True, 5, 10, completion="    return value")

    assert reward_for(result, RewardKnobs()) == pytest.approx(0.5)


def test_reward_for_returns_negative_one_for_failure_or_ban() -> None:
    failed = EvalResult(False, 10, 10, completion="    return value")
    banned = EvalResult(True, 10, 10, banned=True, completion="    eval(value)")

    assert reward_for(failed, RewardKnobs()) == -1.0
    assert reward_for(banned, RewardKnobs()) == -1.0


def test_reward_for_applies_length_penalty() -> None:
    result = EvalResult(True, 10, 10, completion="\n".join("    pass" for _ in range(10)))

    assert reward_for(result, RewardKnobs(lambda_len=0.1)) == pytest.approx(0.95)


def test_reward_for_applies_speed_bonus_and_clips_to_one() -> None:
    fast = EvalResult(True, 10, 10, duration_ms=0, completion="    return value")
    slow = EvalResult(True, 10, 10, duration_ms=2000, completion="    return value")

    assert reward_for(fast, RewardKnobs(lambda_speed=0.25)) == 1.0
    assert reward_for(slow, RewardKnobs(lambda_speed=0.25)) == 1.0


def test_reward_for_clips_large_length_penalty_to_negative_one() -> None:
    result = EvalResult(True, 0, 10, completion="\n".join("    pass" for _ in range(20)))

    assert reward_for(result, RewardKnobs(lambda_len=2.0)) == -1.0
