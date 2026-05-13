from __future__ import annotations

from env.parallel_monitoring_utils import resolve_parallel_end_reason


def test_recovered_step_exception_is_not_terminal() -> None:
    info = {
        "error": "Failed to get observation from game server (got None)",
        "recovered": True,
        "no_execution": True,
        "task_transition": True,
    }

    assert resolve_parallel_end_reason(info, terminated=False, truncated=False) is None


def test_unrecovered_step_exception_stays_terminal() -> None:
    info = {
        "error": "Failed to get observation from game server (got None)",
        "runtime_exit_reason": "step_exception",
        "recovered": False,
    }

    assert (
        resolve_parallel_end_reason(info, terminated=False, truncated=True)
        == "step_exception"
    )
