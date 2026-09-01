from pathlib import Path

from tasks import TaskLoader


def test_safe_parser_has_curriculum_and_separate_evaluation_sets() -> None:
    task = TaskLoader(Path(__file__).parents[1] / "tasks").load("safe_parser")

    assert len(task.prompts) == 4
    assert task.total_tests == 7
    assert len(task.validation_tests) + len(task.validation_error_tests) == 7
    assert len(task.holdout_tests) + len(task.holdout_error_tests) == 7
    assert task.reward_weights == (1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0)


def test_two_sum_plus_expands_named_groups_into_weighted_cases() -> None:
    task = TaskLoader(Path(__file__).parents[1] / "tasks").load("two_sum_plus")

    assert task.test_groups == ("basic", "edge_values", "ordering", "no_solution")
    assert task.total_tests == 10
    assert task.reward_weights == (1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0)
    assert len(task.validation_tests) == 6
    assert len(task.holdout_tests) == 6
    validation_inputs = {repr(case["input"]) for case in task.validation_tests}
    holdout_inputs = {repr(case["input"]) for case in task.holdout_tests}
    assert not validation_inputs & holdout_inputs
