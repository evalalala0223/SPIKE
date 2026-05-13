from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _load_result_validity_utils_module():
    module_name = "_test_result_validity_utils_module"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    module_path = Path(__file__).resolve().parents[1] / "env" / "result_validity_utils.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_RESULT_VALIDITY_MODULE = _load_result_validity_utils_module()
annotate_run_summary_validity = _RESULT_VALIDITY_MODULE.annotate_run_summary_validity
annotate_task_result_validity = _RESULT_VALIDITY_MODULE.annotate_task_result_validity


class TestResultValidityUtils(unittest.TestCase):
    def test_task_result_marks_stopped_run_invalid(self) -> None:
        result = annotate_task_result_validity(
            {
                "completed": False,
                "end_reason": "stopped",
                "budget_exit_reason": None,
            }
        )

        self.assertEqual(result["run_status"], "stopped")
        self.assertFalse(result["is_valid_benchmark"])
        self.assertIn("end_reason=stopped", result["invalid_reason"])

    def test_task_result_accepts_normal_budget_exit(self) -> None:
        result = annotate_task_result_validity(
            {
                "completed": False,
                "end_reason": "max_steps",
                "budget_exit_reason": "max_steps",
            }
        )

        self.assertEqual(result["run_status"], "max_steps")
        self.assertTrue(result["is_valid_benchmark"])
        self.assertIsNone(result["invalid_reason"])

    def test_task_result_marks_reset_error_invalid_without_budget_missing_noise(self) -> None:
        result = annotate_task_result_validity(
            {
                "completed": False,
                "end_reason": "reset_error",
                "runtime_exit_reason": "reset_error",
                "budget_exit_reason": None,
            }
        )

        self.assertEqual(result["run_status"], "reset_error")
        self.assertFalse(result["is_valid_benchmark"])
        self.assertIn("end_reason=reset_error", result["invalid_reason"])
        self.assertNotIn("budget_exit_reason_missing_for_non_normal_exit", result["invalid_reason"])

    def test_run_summary_marks_partial_expected_task_mismatch_invalid(self) -> None:
        run_summary = annotate_run_summary_validity(
            {
                "expected_tasks": 4,
                "tasks": [
                    {
                        "task_index": 1,
                        "is_valid_benchmark": True,
                        "invalid_reason": None,
                    },
                    {
                        "task_index": 2,
                        "is_valid_benchmark": True,
                        "invalid_reason": None,
                    },
                ],
            }
        )

        self.assertEqual(run_summary["run_status"], "partial")
        self.assertEqual(run_summary["actual_tasks"], 2)
        self.assertFalse(run_summary["is_valid_benchmark"])
        self.assertIn("expected_tasks_mismatch:4!=2", run_summary["invalid_reason"])
        self.assertIn("partial_run", run_summary["invalid_reason"])

    def test_run_summary_marks_keyboard_interrupt_invalid(self) -> None:
        run_summary = annotate_run_summary_validity(
            {
                "expected_tasks": 1,
                "tasks": [
                    {
                        "task_index": 1,
                        "is_valid_benchmark": True,
                        "invalid_reason": None,
                    }
                ],
            },
            interrupted=True,
        )

        self.assertEqual(run_summary["run_status"], "interrupted")
        self.assertFalse(run_summary["is_valid_benchmark"])
        self.assertIn("keyboard_interrupt", run_summary["invalid_reason"])


if __name__ == "__main__":
    unittest.main()
