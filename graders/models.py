from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RewardKnobs:
    lambda_len: float = 0.0
    lambda_ban: float = 0.0
    lambda_speed: float = 0.0


@dataclass(frozen=True, slots=True)
class EvalResult:
    no_error: bool
    num_passed: int
    num_tests: int
    duration_ms: int = 0
    sandbox_id: str | None = None
    banned: bool = False
    error: str | None = None
    completion: str = ""
    test_results: tuple[bool, ...] = ()
    test_weights: tuple[float, ...] = ()
