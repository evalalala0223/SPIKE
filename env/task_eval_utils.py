from __future__ import annotations

import logging
from typing import Any, Dict, Tuple


def _is_combat_kill_evaluator(task_obj: Any) -> bool:
    if task_obj is None:
        return False
    evaluator = str(getattr(task_obj, "evaluator", "") or "").strip().lower()
    if evaluator == "kill":
        return True
    return task_obj.__class__.__name__ == "Combat"


def safe_task_evaluate(task_obj: Any, obs: Any, proxy: Any) -> Tuple[Dict[str, Any], Exception | None]:
    try:
        task_eval = task_obj.evaluate(obs, proxy)
    except Exception as exc:
        if not _is_combat_kill_evaluator(task_obj):
            raise

        logging.exception("Combat evaluator failed during task evaluation: %s", exc)
        diagnostics = []
        existing = getattr(task_obj, "evaluation_diagnostics", [])
        if isinstance(existing, list):
            diagnostics.extend(item for item in existing if isinstance(item, dict))
        diagnostics.append(
            {
                "type": "combat_evaluator_exception",
                "source": "task_evaluate",
                "detail": f"{type(exc).__name__}: {exc}",
            }
        )
        try:
            task_obj.evaluation_diagnostics = diagnostics
        except Exception:
            pass

        current_quantity = getattr(task_obj, "current_quantity", 0)
        if not isinstance(current_quantity, (int, float)):
            current_quantity = 0
        fallback = {
            "completed": False,
            "quantity": current_quantity,
            "evaluation_diagnostics": diagnostics,
            "baseline_known": bool(getattr(task_obj, "baseline_known", False)),
            "evaluation_error": f"{type(exc).__name__}: {exc}",
            "evaluation_fallback": "combat_evaluator_exception",
        }
        return fallback, exc

    if not isinstance(task_eval, dict):
        raise TypeError(f"task.evaluate() must return dict, got {type(task_eval).__name__}")
    return task_eval, None
