from __future__ import annotations

from typing import Any, Dict, Iterable, List


_NORMAL_TASK_END_REASONS = {
    "completed",
    "truncated",
    "max_steps",
    "max_llm_calls",
    "cortex_no_execution_watchdog",
}

_EXPLICIT_RUNTIME_FAILURE_REASONS = {
    "error",
    "reset_error",
    "set_agent_failed",
    "worker_recv_error",
    "worker_protocol_error",
    "assertion_error",
    "step_exception",
}


def _dedupe_reasons(reasons: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for reason in reasons:
        normalized = str(reason or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _join_reasons(reasons: Iterable[str]) -> str | None:
    ordered = _dedupe_reasons(reasons)
    return "; ".join(ordered) if ordered else None


def annotate_task_result_validity(result: Dict[str, Any]) -> Dict[str, Any]:
    end_reason = str(result.get("end_reason", "") or "").strip().lower()
    runtime_exit_reason = str(result.get("runtime_exit_reason", "") or "").strip().lower()
    budget_exit_reason = str(result.get("budget_exit_reason", "") or "").strip()
    completed = bool(result.get("completed", False))
    evaluation_diagnostics = result.get("evaluation_diagnostics", [])
    if not isinstance(evaluation_diagnostics, list):
        evaluation_diagnostics = []

    invalid_reasons: List[str] = []
    if end_reason == "stopped":
        invalid_reasons.append("end_reason=stopped")
    if end_reason == "interrupted":
        invalid_reasons.append("keyboard_interrupt")
    if end_reason == "error":
        invalid_reasons.append("end_reason=error")
    elif end_reason in _EXPLICIT_RUNTIME_FAILURE_REASONS:
        invalid_reasons.append(f"end_reason={end_reason}")
    if runtime_exit_reason and runtime_exit_reason not in _NORMAL_TASK_END_REASONS and runtime_exit_reason != end_reason:
        invalid_reasons.append(f"runtime_exit_reason={runtime_exit_reason}")

    normal_exit = completed or end_reason in _NORMAL_TASK_END_REASONS
    explicit_runtime_failure = (
        end_reason in _EXPLICIT_RUNTIME_FAILURE_REASONS
        or runtime_exit_reason in _EXPLICIT_RUNTIME_FAILURE_REASONS
    )
    if not normal_exit and not budget_exit_reason and not explicit_runtime_failure:
        invalid_reasons.append("budget_exit_reason_missing_for_non_normal_exit")
    for diagnostic in evaluation_diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        diagnostic_type = str(diagnostic.get("type", "") or "").strip()
        source = str(diagnostic.get("source", "") or "").strip()
        if diagnostic_type in {
            "combat_evaluator_unavailable",
            "combat_evaluator_exception",
            "combat_baseline_unconfirmed",
        }:
            if source:
                invalid_reasons.append(f"{diagnostic_type}:{source}")
            else:
                invalid_reasons.append(diagnostic_type)

    result["run_status"] = end_reason or "unknown"
    result["is_valid_benchmark"] = not invalid_reasons
    result["benchmark_status"] = "valid" if result["is_valid_benchmark"] else "invalid"
    result["invalid_reason"] = _join_reasons(invalid_reasons)
    return result


def annotate_run_summary_validity(
    run_summary: Dict[str, Any],
    *,
    interrupted: bool = False,
) -> Dict[str, Any]:
    tasks = run_summary.get("tasks", [])
    if not isinstance(tasks, list):
        tasks = []

    expected_tasks = run_summary.get("expected_tasks")
    try:
        expected_tasks_int = int(expected_tasks) if expected_tasks is not None else len(tasks)
    except (TypeError, ValueError):
        expected_tasks_int = len(tasks)
    actual_tasks = len(tasks)

    invalid_reasons: List[str] = []
    if interrupted:
        invalid_reasons.append("keyboard_interrupt")
    if actual_tasks != expected_tasks_int:
        invalid_reasons.append(f"expected_tasks_mismatch:{expected_tasks_int}!={actual_tasks}")
    if actual_tasks < expected_tasks_int:
        invalid_reasons.append("partial_run")

    invalid_task_reasons = [
        str(task.get("invalid_reason", "") or "").strip()
        for task in tasks
        if isinstance(task, dict) and task.get("is_valid_benchmark") is False
    ]
    if invalid_task_reasons:
        invalid_reasons.append("invalid_task_results")
        invalid_reasons.extend(invalid_task_reasons)

    if interrupted:
        run_status = "interrupted"
    elif actual_tasks != expected_tasks_int:
        run_status = "partial"
    else:
        run_status = "completed"

    run_summary["actual_tasks"] = actual_tasks
    run_summary["run_status"] = run_status
    run_summary["is_valid_benchmark"] = not invalid_reasons
    run_summary["benchmark_status"] = "valid" if run_summary["is_valid_benchmark"] else "invalid"
    run_summary["invalid_reason"] = _join_reasons(invalid_reasons)
    return run_summary
