from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

if "gymnasium" not in sys.modules:
    gymnasium_stub = types.ModuleType("gymnasium")
    gymnasium_stub.register = lambda *args, **kwargs: None
    sys.modules["gymnasium"] = gymnasium_stub

from env.result_validity_utils import annotate_task_result_validity
from env.task_eval_utils import safe_task_evaluate
from env.tasks.combat import Combat


class _StubProxy:
    def __init__(self, monster_kill_reads: list[int | None]) -> None:
        self.monster_kill_reads = list(monster_kill_reads)
        self.set_monster_stat_calls: list[tuple[str, int]] = []

    def set_monster_stat(self, monster: str, kills: int) -> None:
        self.set_monster_stat_calls.append((monster, kills))

    def get_monster_kills(self, _monster: str, retries: int = 3, retry_sleep_s: float = 0.5) -> int | None:
        if not self.monster_kill_reads:
            return None
        return self.monster_kill_reads.pop(0)


class _ExplodingProxy(_StubProxy):
    def __init__(self) -> None:
        super().__init__([])

    def get_monster_kills(self, _monster: str, retries: int = 3, retry_sleep_s: float = 0.5) -> int | None:
        raise RuntimeError("boom")


class TestCombatTask(unittest.TestCase):
    def _make_task(self) -> Combat:
        return Combat(
            llm_description="kill_1_green_slime_with_rusty_sword",
            object="Green Slime",
            quantity=1,
            tool="Rusty Sword",
            save="dummy",
            init_commands=[],
            evaluator="kill",
            difficulty="easy",
        )

    def test_kill_evaluator_resets_baseline_and_waits_for_first_successful_read(self) -> None:
        task = self._make_task()
        proxy = _StubProxy([None, None, 7, 9])

        with mock.patch("env.tasks.base.load_save.load_save"):
            task.init_task(proxy)

        self.assertEqual(proxy.set_monster_stat_calls, [("Green Slime", 0)])
        self.assertFalse(task.baseline_known)

        first = task.evaluate({"player": {}}, proxy)
        self.assertFalse(first["baseline_known"])
        self.assertEqual(first["quantity"], 0)
        self.assertIn(
            "combat_evaluator_unavailable",
            [item["type"] for item in first["evaluation_diagnostics"]],
        )

        second = task.evaluate({"player": {}}, proxy)
        self.assertTrue(second["baseline_known"])
        self.assertEqual(second["quantity"], 0)

        third = task.evaluate({"player": {}}, proxy)
        self.assertTrue(third["completed"])
        self.assertEqual(third["quantity"], 2)

    def test_combat_evaluator_unavailable_marks_result_invalid(self) -> None:
        result = annotate_task_result_validity(
            {
                "end_reason": "max_steps",
                "completed": False,
                "budget_exit_reason": "max_steps",
                "evaluation_diagnostics": [
                    {
                        "type": "combat_evaluator_unavailable",
                        "source": "get_monster_kills",
                    }
                ],
            }
        )

        self.assertFalse(result["is_valid_benchmark"])
        self.assertIn("combat_evaluator_unavailable:get_monster_kills", result["invalid_reason"])

    def test_safe_task_evaluate_converts_combat_evaluator_exception_into_fallback(self) -> None:
        task = self._make_task()
        proxy = _ExplodingProxy()
        task.last_obs = {"player": {}}
        task.baseline_known = False

        task_eval, caught_error = safe_task_evaluate(task, {"player": {}}, proxy)

        self.assertIsNotNone(caught_error)
        self.assertFalse(task_eval["completed"])
        self.assertEqual(task_eval["quantity"], 0)
        self.assertEqual(
            task_eval["evaluation_diagnostics"][-1]["type"],
            "combat_evaluator_exception",
        )
        self.assertEqual(task_eval["evaluation_fallback"], "combat_evaluator_exception")

    def test_combat_evaluator_exception_marks_result_invalid(self) -> None:
        result = annotate_task_result_validity(
            {
                "end_reason": "max_steps",
                "completed": False,
                "budget_exit_reason": "max_steps",
                "evaluation_diagnostics": [
                    {
                        "type": "combat_evaluator_exception",
                        "source": "task_evaluate",
                    }
                ],
            }
        )

        self.assertFalse(result["is_valid_benchmark"])
        self.assertIn("combat_evaluator_exception:task_evaluate", result["invalid_reason"])


if __name__ == "__main__":
    unittest.main()
