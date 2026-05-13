from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from cradle.runner.big_brain import BigBrain
from cradle.runner.dual_brain import DualBrainController


class TestPlanningGuardrails(unittest.TestCase):
    def test_cultivation_reasoning_salvage_rejects_directional_tool_use(self) -> None:
        self.assertFalse(
            BigBrain._salvaged_action_is_safe(
                salvaged_action='use(direction="down")',
                result_state={"task": "till_5_tile_with_hoe"},
                prior_state={"task_description": "till_5_tile_with_hoe"},
            )
        )

    def test_navigation_reasoning_salvage_keeps_move_action(self) -> None:
        self.assertTrue(
            BigBrain._salvaged_action_is_safe(
                salvaged_action="move(x=3, y=0)",
                result_state={"task": "go_to_bus_stop"},
                prior_state={"task_description": "go_to_bus_stop"},
            )
        )

    def test_big_brain_replan_does_not_count_no_progress_feedback_as_success(self) -> None:
        self.assertFalse(
            DualBrainController._latest_execution_counts_as_success(
                {
                    "has_execution_feedback": True,
                    "last_state_changed": False,
                    "task_progress_delta": 0,
                    "latest_task_eval": {"completed": False},
                }
            )
        )

    def test_big_brain_replan_counts_real_progress_feedback_as_success(self) -> None:
        self.assertTrue(
            DualBrainController._latest_execution_counts_as_success(
                {
                    "has_execution_feedback": True,
                    "last_state_changed": True,
                    "task_progress_delta": 1,
                    "latest_task_eval": {"completed": False},
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
