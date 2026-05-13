from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.experiment_budget import evaluate_budget_progress, resolve_experiment_budget


class TestExperimentBudget(unittest.TestCase):
    def test_invalid_mode_falls_back_to_benchmark_steps(self) -> None:
        task = SimpleNamespace(difficulty="easy")

        budget = resolve_experiment_budget(
            task=task,
            task_config={"experiment_budget_mode": "legacy_steps"},
        )

        self.assertEqual(budget.mode, "benchmark_steps")
        self.assertEqual(budget.step_budget, 30)
        self.assertIsNone(budget.llm_call_budget)

    def test_benchmark_step_budget_uses_original_benchmark_limits(self) -> None:
        task = SimpleNamespace(difficulty="medium")

        budget = resolve_experiment_budget(task=task, default_mode="benchmark_steps")

        self.assertEqual(budget.mode, "benchmark_steps")
        self.assertEqual(budget.step_budget, 50)
        self.assertIsNone(budget.llm_call_budget)

    def test_benchmark_llm_budget_uses_step_reference_and_llm_budget(self) -> None:
        task = SimpleNamespace(difficulty="hard")

        budget = resolve_experiment_budget(task=task, default_mode="benchmark_llm_calls")

        self.assertEqual(budget.mode, "benchmark_llm_calls")
        self.assertEqual(budget.step_budget, 150)
        self.assertEqual(budget.llm_call_budget, 600)

    def test_task_override_can_drive_cost_budget_from_explicit_step_cap(self) -> None:
        task = SimpleNamespace(difficulty="easy")

        budget = resolve_experiment_budget(
            task=task,
            task_config={"max_turn_count": 25, "experiment_budget_mode": "benchmark_llm_calls"},
            default_mode="benchmark_steps",
        )

        self.assertEqual(budget.step_budget, 25)
        self.assertEqual(budget.llm_call_budget, 100)

    def test_explicit_llm_call_override_wins(self) -> None:
        task = SimpleNamespace(difficulty="easy")

        budget = resolve_experiment_budget(
            task=task,
            task_config={"max_turn_count": 25, "max_llm_calls": 77, "experiment_budget_mode": "benchmark_llm_calls"},
        )

        self.assertEqual(budget.step_budget, 25)
        self.assertEqual(budget.llm_call_budget, 77)

    def test_budget_progress_switches_metric_by_mode(self) -> None:
        step_budget = resolve_experiment_budget(
            task=SimpleNamespace(difficulty="easy"),
            default_mode="benchmark_steps",
        )
        llm_budget = resolve_experiment_budget(
            task=SimpleNamespace(difficulty="easy"),
            default_mode="benchmark_llm_calls",
        )

        step_progress = evaluate_budget_progress(step_count=30, llm_call_count=10, budget=step_budget)
        llm_progress = evaluate_budget_progress(step_count=30, llm_call_count=120, budget=llm_budget)

        self.assertTrue(step_progress.exhausted)
        self.assertEqual(step_progress.end_reason, "max_steps")
        self.assertEqual(step_progress.budget_metric, "steps")

        self.assertTrue(llm_progress.exhausted)
        self.assertEqual(llm_progress.end_reason, "max_llm_calls")
        self.assertEqual(llm_progress.budget_metric, "llm_calls")


if __name__ == "__main__":
    unittest.main()
