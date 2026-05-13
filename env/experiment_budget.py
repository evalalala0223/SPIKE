from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional


DEFAULT_EXPERIMENT_BUDGET_MODE = "benchmark_steps"
VALID_EXPERIMENT_BUDGET_MODES = (
    "benchmark_steps",
    "benchmark_llm_calls",
)

_STEP_LIMITS_BY_MODE = {
    "benchmark_steps": {
        "easy": 30,
        "medium": 50,
        "hard": 150,
        "default": 150,
    },
}

_LLM_CALL_LIMITS = {
    "easy": 120,
    "medium": 200,
    "hard": 600,
    "default": 600,
}


@dataclass(frozen=True)
class ExperimentBudget:
    mode: str
    step_budget: int
    llm_call_budget: Optional[int]

    @property
    def uses_llm_calls(self) -> bool:
        return self.mode == "benchmark_llm_calls"

    @property
    def budget_metric(self) -> str:
        return "llm_calls" if self.uses_llm_calls else "steps"


@dataclass(frozen=True)
class BudgetProgress:
    mode: str
    budget_metric: str
    step_budget: int
    llm_call_budget: Optional[int]
    step_count: int
    llm_call_count: int
    used: int
    limit: int
    exhausted: bool
    end_reason: Optional[str]


def _coerce_positive_int(value: Any, *, field_name: str) -> Optional[int]:
    if value is None:
        return None
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        logging.warning("Invalid %s=%r, ignoring override", field_name, value)
        return None


def normalize_experiment_budget_mode(value: Any) -> str:
    normalized = str(value or DEFAULT_EXPERIMENT_BUDGET_MODE).strip().lower()
    if normalized not in VALID_EXPERIMENT_BUDGET_MODES:
        logging.warning(
            "Invalid experiment_budget_mode=%r, falling back to %s",
            value,
            DEFAULT_EXPERIMENT_BUDGET_MODE,
        )
        return DEFAULT_EXPERIMENT_BUDGET_MODE
    return normalized


def resolve_experiment_budget_mode(
    task_config: Optional[Mapping[str, Any]] = None,
    default_mode: str = DEFAULT_EXPERIMENT_BUDGET_MODE,
) -> str:
    if isinstance(task_config, Mapping) and task_config.get("experiment_budget_mode") is not None:
        return normalize_experiment_budget_mode(task_config.get("experiment_budget_mode"))
    return normalize_experiment_budget_mode(default_mode)


def _resolve_difficulty_key(task: Any = None) -> str:
    difficulty = str(getattr(task, "difficulty", "") or "").strip().lower()
    return difficulty or "default"


def _resolve_step_budget_from_difficulty(task: Any = None, mode: str = DEFAULT_EXPERIMENT_BUDGET_MODE) -> int:
    difficulty = _resolve_difficulty_key(task)
    resolved_mode = "benchmark_steps" if mode == "benchmark_llm_calls" else mode
    limits = _STEP_LIMITS_BY_MODE.get(resolved_mode, _STEP_LIMITS_BY_MODE["benchmark_steps"])
    return int(limits.get(difficulty, limits["default"]))


def _resolve_llm_call_budget(task: Any = None) -> int:
    difficulty = _resolve_difficulty_key(task)
    return int(_LLM_CALL_LIMITS.get(difficulty, _LLM_CALL_LIMITS["default"]))


def resolve_experiment_budget(
    task: Any = None,
    task_config: Optional[Mapping[str, Any]] = None,
    default_mode: str = DEFAULT_EXPERIMENT_BUDGET_MODE,
) -> ExperimentBudget:
    mode = resolve_experiment_budget_mode(task_config=task_config, default_mode=default_mode)
    explicit_step_budget = None
    explicit_llm_call_budget = None
    if isinstance(task_config, Mapping):
        explicit_step_budget = _coerce_positive_int(
            task_config.get("max_turn_count"),
            field_name="max_turn_count",
        )
        explicit_llm_call_budget = _coerce_positive_int(
            task_config.get("max_llm_calls"),
            field_name="max_llm_calls",
        )

    step_budget = explicit_step_budget or _resolve_step_budget_from_difficulty(task=task, mode=mode)
    llm_call_budget = None
    if mode == "benchmark_llm_calls":
        if explicit_llm_call_budget is not None:
            llm_call_budget = explicit_llm_call_budget
        elif explicit_step_budget is not None:
            llm_call_budget = explicit_step_budget * 4
        else:
            llm_call_budget = _resolve_llm_call_budget(task=task)

    return ExperimentBudget(
        mode=mode,
        step_budget=step_budget,
        llm_call_budget=llm_call_budget,
    )


def evaluate_budget_progress(
    *,
    step_count: int,
    llm_call_count: int,
    budget: ExperimentBudget,
) -> BudgetProgress:
    if budget.uses_llm_calls:
        limit = int(budget.llm_call_budget or 0)
        used = int(llm_call_count)
        exhausted = used >= limit
        end_reason = "max_llm_calls" if exhausted else None
    else:
        limit = int(budget.step_budget)
        used = int(step_count)
        exhausted = used >= limit
        end_reason = "max_steps" if exhausted else None

    return BudgetProgress(
        mode=budget.mode,
        budget_metric=budget.budget_metric,
        step_budget=int(budget.step_budget),
        llm_call_budget=budget.llm_call_budget,
        step_count=int(step_count),
        llm_call_count=int(llm_call_count),
        used=used,
        limit=limit,
        exhausted=exhausted,
        end_reason=end_reason,
    )
