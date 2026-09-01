from api.app import _percent
import pytest


def test_api_percent_parses_runner_pass_rate() -> None:
    assert _percent("Final holdout pass rate: 36.2%") == pytest.approx(0.362)


def test_api_percent_returns_none_for_unrelated_log() -> None:
    assert _percent("Created remote sandbox: abc") is None
