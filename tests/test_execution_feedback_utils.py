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

from stardojo.utils.execution_feedback_utils import (
    execution_counts_as_recent_success,
    execution_has_explicit_failure,
    execution_observation_confirms_change,
    execution_refusal_type,
    execution_refused_action,
)


class TestExecutionFeedbackUtils(unittest.TestCase):
    def test_toolbar_selection_change_counts_as_observation_confirmation(self) -> None:
        previous = {
            "selected_position": 0,
            "selected_item_name": "Axe",
            "toolbar_information": "Currently selected item: slot_index 0: Axe",
        }
        current = {
            "selected_position": 2,
            "selected_item_name": "Watering Can",
            "toolbar_information": "Currently selected item: slot_index 2: Watering Can",
        }

        self.assertTrue(execution_observation_confirms_change(previous, current))

    def test_prompt_derived_route_context_drift_does_not_confirm_execution(self) -> None:
        previous = {
            "front_tile_summary": "(none)",
            "nearest_grounded_target_summary": "Nearest grounded route context: Bus Stop (relative offset: x=4, y=0).",
            "position": "(62, 18)",
        }
        current = {
            "front_tile_summary": "Front tile (0, 1) toward down: Coop Door. Not an obvious clearable obstacle.",
            "nearest_grounded_target_summary": "Nearest grounded interaction target: Coop Door at (0, 1).",
            "position": "(62, 18)",
        }

        self.assertFalse(execution_observation_confirms_change(previous, current))

    def test_position_change_still_confirms_execution(self) -> None:
        previous = {
            "position": "(62, 18)",
            "selected_item_name": "Hoe",
        }
        current = {
            "position": "(63, 18)",
            "selected_item_name": "Hoe",
        }

        self.assertTrue(execution_observation_confirms_change(previous, current))

    def test_untracked_time_only_change_does_not_confirm_execution(self) -> None:
        previous = {
            "selected_position": 0,
            "selected_item_name": "Axe",
            "toolbar_information": "Currently selected item: slot_index 0: Axe",
            "time": "6:00am",
        }
        current = {
            "selected_position": 0,
            "selected_item_name": "Axe",
            "toolbar_information": "Currently selected item: slot_index 0: Axe",
            "time": "6:10am",
        }

        self.assertFalse(execution_observation_confirms_change(previous, current))

    def test_circuit_breaker_warning_counts_as_explicit_failure(self) -> None:
        record = {
            "errors_info": (
                "CIRCUIT-BREAKER: action `move(x=-2, y=1)` previously produced explicit failure 3 times in a row. "
                "This action is REFUSED for this step."
            )
        }

        self.assertTrue(execution_has_explicit_failure(record))

    def test_execution_refused_action_prefers_structured_exec_info_field(self) -> None:
        record = {
            "exec_info": {
                "refusal_type": "same_action_circuit_breaker",
                "refused_action": 'move(x=-2, y=1)',
                "errors_info": "CIRCUIT-BREAKER: action `move(x=-2, y=1)` ...",
            }
        }

        self.assertEqual(execution_refused_action(record), 'move(x=-2, y=1)')
        self.assertEqual(execution_refusal_type(record), "same_action_circuit_breaker")

    def test_execution_refused_action_can_be_parsed_from_warning_text(self) -> None:
        record = {
            "errors_info": (
                "AXIS-CIRCUIT-BREAKER: 3 consecutive blocked move() calls toward -y. "
                "The path is blocked in this direction. Injecting recovery move `move(x=1, y=0)`. "
                "Next plan MUST try a different direction. CIRCUIT-BREAKER: action `move(x=0, y=-1)` "
                "previously produced explicit failure 3 times in a row."
            )
        }

        self.assertEqual(execution_refused_action(record), 'move(x=0, y=-1)')
        self.assertEqual(execution_refusal_type(record), "axis_circuit_breaker")

    def test_cultivation_move_without_progress_does_not_count_as_recent_success(self) -> None:
        record = {
            "task_kind": "till",
            "success": True,
            "state_changed": True,
            "progress_delta": 0,
            "completed": False,
            "exec_info": {
                "executed_skills": ['move(x=-1, y=0)'],
                "last_skill": 'move(x=-1, y=0)',
                "errors": False,
                "errors_info": "",
            },
        }

        self.assertFalse(execution_counts_as_recent_success(record))

    def test_cultivation_progress_still_counts_as_recent_success(self) -> None:
        record = {
            "task_kind": "till",
            "success": True,
            "state_changed": True,
            "progress_delta": 1,
            "completed": False,
            "exec_info": {
                "executed_skills": ['use(direction="up")'],
                "last_skill": 'use(direction="up")',
                "errors": False,
                "errors_info": "",
            },
        }

        self.assertTrue(execution_counts_as_recent_success(record))


if __name__ == "__main__":
    unittest.main()
