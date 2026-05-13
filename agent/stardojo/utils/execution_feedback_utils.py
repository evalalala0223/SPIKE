from __future__ import annotations

import json
import re
from typing import Any, Dict

_POSITION_ISSUE_MARKERS = (
    "not adjacent",
    "move closer",
    "too far",
    "wrong position",
    "become adjacent",
    "not in range",
    "out of range",
)

_EXPLICIT_EXECUTION_FAILURE_MARKERS = (
    "empty_action_nop",
    "path is likely blocked by an obstacle",
    "player position did not change",
    "composite skill reported failure",
    "label id ",
    "circuit-breaker:",
    "axis-circuit-breaker:",
    "this action is refused for this step",
)

_REFUSAL_SIGNAL_MARKERS = (
    "circuit-breaker:",
    "axis-circuit-breaker:",
    "this action is refused for this step",
)

_CONFIRMATION_SNAPSHOT_KEYS = (
    "location",
    "position",
    "current_position",
    "facing_direction",
    "facing_position",
    "selected_position",
    "selected_item_name",
    "toolbar_information",
    "current_menu",
    "crops",
    "buildings",
    "furniture",
    "exits",
)


def detect_position_issue(*texts: Any) -> bool:
    combined = " ".join(str(text or "").strip().lower() for text in texts if text)
    return any(marker in combined for marker in _POSITION_ISSUE_MARKERS)


def _extract_exec_info(record: Any) -> Dict[str, Any]:
    if not isinstance(record, dict):
        return {}

    nested = record.get("exec_info")
    if isinstance(nested, dict):
        return nested

    return record


def execution_errors_info(record: Any) -> str:
    exec_info = _extract_exec_info(record)
    errors_info = exec_info.get("errors_info", "")
    if not errors_info and isinstance(record, dict):
        errors_info = record.get("errors_info", "")
    return str(errors_info or "").strip()


def execution_has_signal(record: Any) -> bool:
    exec_info = _extract_exec_info(record)
    if bool(exec_info.get("done", False)):
        return True

    executed_skills = exec_info.get("executed_skills", [])
    if isinstance(executed_skills, list) and len(executed_skills) > 0:
        return True

    if str(exec_info.get("last_skill", "") or "").strip():
        return True

    if isinstance(record, dict):
        top_level_skills = record.get("executed_skills", [])
        if isinstance(top_level_skills, list) and len(top_level_skills) > 0:
            return True
        if str(record.get("last_skill", "") or "").strip():
            return True

    return False


def execution_has_explicit_failure(record: Any) -> bool:
    if isinstance(record, dict) and record.get("error"):
        return True

    exec_info = _extract_exec_info(record)
    if bool(exec_info.get("errors", False)):
        return True

    lowered_errors = execution_errors_info(record).lower()
    return any(marker in lowered_errors for marker in _EXPLICIT_EXECUTION_FAILURE_MARKERS)


def execution_has_no_confirmation(record: Any) -> bool:
    lowered_errors = execution_errors_info(record).lower()
    return "no confirmation" in lowered_errors or "returned no confirmation" in lowered_errors


def execution_refused_action(record: Any) -> str:
    exec_info = _extract_exec_info(record)
    for key in ("refused_action", "blocked_action"):
        value = exec_info.get(key)
        if value:
            return str(value).strip()

    errors_info = execution_errors_info(record)
    lowered_errors = errors_info.lower()
    if not any(marker in lowered_errors for marker in _REFUSAL_SIGNAL_MARKERS):
        return ""

    match = re.search(r"action\s*`([^`]+)`", errors_info, re.IGNORECASE)
    if not match:
        return ""
    return str(match.group(1) or "").strip()


def execution_refusal_type(record: Any) -> str:
    exec_info = _extract_exec_info(record)
    refusal_type = exec_info.get("refusal_type")
    if refusal_type:
        return str(refusal_type).strip().lower()

    lowered_errors = execution_errors_info(record).lower()
    if "axis-circuit-breaker" in lowered_errors:
        return "axis_circuit_breaker"
    if "circuit-breaker" in lowered_errors:
        return "same_action_circuit_breaker"
    return ""


def infer_execution_success_raw(record: Any) -> bool:
    return execution_has_signal(record) and not execution_has_explicit_failure(record)


def execution_counts_as_recent_success(record: Any) -> bool:
    if not isinstance(record, dict):
        return False

    if record.get("success") is False:
        return False

    if execution_has_explicit_failure(record):
        return False

    if bool(record.get("uncertain_execution", False)):
        return False

    task_kind = str(record.get("task_kind", "") or "").strip().lower()
    if task_kind in {"till", "fertilize", "sow", "water", "harvest"}:
        if record.get("completed") is True:
            return True
        progress_delta = record.get("progress_delta", None)
        return progress_delta not in (None, "", 0, 0.0)

    if record.get("completed") is True:
        return True

    progress_delta = record.get("progress_delta", None)
    if progress_delta not in (None, "", 0, 0.0):
        return True

    if record.get("state_changed") is True:
        return True

    if "success" in record:
        if any(key in record for key in ("state_changed", "progress_delta", "completed")):
            return False
        return bool(record.get("success")) and infer_execution_success_raw(record)

    return infer_execution_success_raw(record)


def _normalize_for_snapshot(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _normalize_for_snapshot(child)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }

    if isinstance(value, list):
        return [_normalize_for_snapshot(child) for child in value]

    if isinstance(value, tuple):
        return [_normalize_for_snapshot(child) for child in value]

    if isinstance(value, set):
        normalized_items = [_normalize_for_snapshot(child) for child in value]
        return sorted(
            normalized_items,
            key=lambda item: json.dumps(
                item,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ),
        )

    return value


def stable_snapshot_text(value: Any) -> str:
    if value in (None, "", [], {}):
        return ""

    normalized = _normalize_for_snapshot(value)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def execution_confirmation_snapshot(record: Any) -> str:
    if not isinstance(record, dict):
        return ""

    snapshot = {
        key: record.get(key)
        for key in _CONFIRMATION_SNAPSHOT_KEYS
        if key in record and record.get(key) not in (None, "", [], {})
    }
    return stable_snapshot_text(snapshot)


def execution_observation_confirms_change(previous_record: Any, current_record: Any) -> bool:
    previous_snapshot = execution_confirmation_snapshot(previous_record)
    current_snapshot = execution_confirmation_snapshot(current_record)
    if not previous_snapshot or not current_snapshot:
        return False
    return previous_snapshot != current_snapshot


__all__ = [
    "detect_position_issue",
    "execution_confirmation_snapshot",
    "execution_counts_as_recent_success",
    "execution_errors_info",
    "execution_has_explicit_failure",
    "execution_has_no_confirmation",
    "execution_refused_action",
    "execution_refusal_type",
    "execution_observation_confirms_change",
    "execution_has_signal",
    "infer_execution_success_raw",
    "stable_snapshot_text",
]
