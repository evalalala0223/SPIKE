from __future__ import annotations

import datetime as dt
import os
import shutil
from typing import Any, Dict, Optional

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def resolve_screenshot_path(src_path: Any, project_root: str) -> Optional[str]:
    text = str(src_path or "").strip()
    if not text:
        return None

    candidates = []
    if os.path.isabs(text):
        candidates.append(text)
    else:
        candidates.append(os.path.abspath(text))
        candidates.append(os.path.abspath(os.path.join(project_root, text)))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def get_latest_source_screenshot(obs: Any, project_root: str) -> Optional[str]:
    if not isinstance(obs, dict):
        return None

    image_paths = obs.get("image_paths")
    if not image_paths:
        return None

    try:
        latest_src = list(image_paths)[-1]
    except Exception:
        return None

    return resolve_screenshot_path(latest_src, project_root)


def refresh_live_latest_screenshot(
    obs: Any,
    project_root: str,
    task_dir: str,
    target_name: str = "live_latest.jpeg",
) -> Optional[str]:
    resolved_src = get_latest_source_screenshot(obs, project_root)
    if resolved_src is None:
        return None

    os.makedirs(task_dir, exist_ok=True)
    target_path = os.path.join(task_dir, target_name)
    if Image is not None:
        try:
            with Image.open(resolved_src) as image:
                image.convert("RGB").save(target_path, format="JPEG")
            return target_name
        except Exception:
            pass

    shutil.copy2(resolved_src, target_path)
    return target_name


def should_append_result_step(info_obj: Any) -> bool:
    info_obj = info_obj if isinstance(info_obj, dict) else {}
    return not bool(info_obj.get("no_execution", False))


def resolve_parallel_end_reason(
    info_obj: Any,
    *,
    terminated: bool,
    truncated: bool,
) -> Optional[str]:
    info_obj = info_obj if isinstance(info_obj, dict) else {}
    task_eval = info_obj.get("task_eval", {})
    if not isinstance(task_eval, dict):
        task_eval = {}

    error_text = str(info_obj.get("error", "") or "").strip()
    runtime_exit_reason = str(info_obj.get("runtime_exit_reason", "") or "").strip()
    budget_exit_reason = str(info_obj.get("budget_exit_reason", "") or "").strip()
    task_completed = bool(task_eval.get("completed", False))

    if bool(info_obj.get("recovered", False)) and not terminated and not truncated:
        return None
    if error_text:
        return runtime_exit_reason or "error"
    if runtime_exit_reason:
        return runtime_exit_reason
    if budget_exit_reason:
        return budget_exit_reason
    if terminated:
        if task_completed or not truncated:
            return "completed"
        return "truncated"
    if not truncated:
        return None

    return "truncated"


def resolve_parallel_run_status(end_reason: Any) -> str:
    normalized = str(end_reason or "").strip().lower()
    if not normalized:
        return "truncated"
    if normalized in {"completed", "running", "stopped"}:
        return normalized
    if normalized in {
        "error",
        "worker_recv_error",
        "worker_protocol_error",
        "invalid_worker_result",
        "set_agent_failed",
        "assertion_error",
        "step_exception",
    } or normalized.endswith("error"):
        return "error"
    return normalized


def build_runtime_diagnostics(task_meta: Any, info_obj: Any) -> Dict[str, Any]:
    task_meta = task_meta if isinstance(task_meta, dict) else {}
    info_obj = info_obj if isinstance(info_obj, dict) else {}
    return {
        "planning_attempt_count": _coerce_int(task_meta.get("planning_attempt_count")),
        "blocked_replan_count": _coerce_int(task_meta.get("blocked_replan_count")),
        "no_execution_return_count": _coerce_int(task_meta.get("no_execution_return_count")),
        "executed_step_count": _coerce_int(task_meta.get("executed_step_count")),
        "watchdog_triggered": bool(task_meta.get("watchdog_triggered", False)),
        "watchdog_reason": str(task_meta.get("watchdog_reason", "") or ""),
        "last_planning_sec": task_meta.get("last_planning_sec"),
        "planning_sec_median": task_meta.get("planning_sec_median"),
        "last_no_execution_planning_sec": task_meta.get("last_no_execution_planning_sec"),
        "no_execution_planning_sec_median": task_meta.get("no_execution_planning_sec_median"),
        "no_execution_planning_sample_count": _coerce_int(task_meta.get("no_execution_planning_sample_count")),
        "watchdog_dynamic_timeout_sec": task_meta.get("watchdog_dynamic_timeout_sec"),
        "same_step_elapsed_sec": task_meta.get("same_step_elapsed_sec"),
        "last_no_execution": bool(info_obj.get("no_execution", False)),
        "last_warning": str(info_obj.get("warning", "") or ""),
    }


def build_live_status_payload(
    *,
    task_index: int,
    task_name: Any,
    task_description: Any,
    run_status: str,
    info_obj: Any,
    task_meta: Any,
    latest_result_screenshot: Optional[str],
    latest_source_screenshot: Optional[str],
    now_iso: Optional[str] = None,
) -> Dict[str, Any]:
    info_obj = info_obj if isinstance(info_obj, dict) else {}
    task_meta = task_meta if isinstance(task_meta, dict) else {}

    last_step_index = info_obj.get("step_index")
    if not isinstance(last_step_index, int):
        last_step_index = None

    return {
        "task_index": int(task_index),
        "task_name": str(task_name or ""),
        "task_description": str(task_description or ""),
        "run_status": str(run_status or "running"),
        "last_update_time": now_iso or dt.datetime.now().isoformat(),
        "last_step_index": last_step_index,
        "last_action": str(info_obj.get("action", "") or ""),
        "last_no_execution": bool(info_obj.get("no_execution", False)),
        "last_warning": str(info_obj.get("warning", "") or ""),
        "runtime_exit_reason": str(info_obj.get("runtime_exit_reason", "") or task_meta.get("runtime_exit_reason", "") or ""),
        "budget_exit_reason": str(info_obj.get("budget_exit_reason", "") or task_meta.get("budget_exit_reason", "") or ""),
        "planning_attempt_count": _coerce_int(task_meta.get("planning_attempt_count")),
        "blocked_replan_count": _coerce_int(task_meta.get("blocked_replan_count")),
        "no_execution_return_count": _coerce_int(task_meta.get("no_execution_return_count")),
        "executed_step_count": _coerce_int(task_meta.get("executed_step_count")),
        "watchdog_triggered": bool(task_meta.get("watchdog_triggered", False)),
        "watchdog_reason": str(task_meta.get("watchdog_reason", "") or ""),
        "last_planning_sec": task_meta.get("last_planning_sec"),
        "planning_sec_median": task_meta.get("planning_sec_median"),
        "last_no_execution_planning_sec": task_meta.get("last_no_execution_planning_sec"),
        "no_execution_planning_sec_median": task_meta.get("no_execution_planning_sec_median"),
        "no_execution_planning_sample_count": _coerce_int(task_meta.get("no_execution_planning_sample_count")),
        "watchdog_dynamic_timeout_sec": task_meta.get("watchdog_dynamic_timeout_sec"),
        "same_step_elapsed_sec": task_meta.get("same_step_elapsed_sec"),
        "llm_call_count": task_meta.get("llm_call_count"),
        "llm_call_breakdown": task_meta.get("llm_call_breakdown", {}),
        "agent_run_dir_name": str(task_meta.get("agent_run_dir_name", "") or ""),
        "planner_comp_model": str(task_meta.get("planner_comp_model", "") or ""),
        "embedding_model": str(task_meta.get("embedding_model", "") or ""),
        "prompt_profile": str(task_meta.get("prompt_profile", "") or ""),
        "resolved_action_planning_template": str(task_meta.get("resolved_action_planning_template", "") or ""),
        "resolved_task_inference_template": str(task_meta.get("resolved_task_inference_template", "") or ""),
        "latest_result_screenshot": latest_result_screenshot,
        "latest_source_screenshot": latest_source_screenshot,
    }
