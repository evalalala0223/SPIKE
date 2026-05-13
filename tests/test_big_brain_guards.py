from __future__ import annotations

from pathlib import Path
import unittest

from cradle.runner.big_brain import BigBrain, BrainPlanResult
from cradle.runner.vllm_client import VLLMClient


ROOT = Path(__file__).resolve().parents[1]
ACTION_PLANNING_CORTEX_TEMPLATE = (
    ROOT / "agent" / "res" / "stardew" / "prompts" / "templates" / "action_planning_cortex.prompt"
)


class TestBigBrainGuards(unittest.TestCase):
    def test_non_move_failed_action_is_kept_grounded(self) -> None:
        brain = BigBrain(workflow_app=None)
        brain.record_failed_action('use(direction="left")')

        suggestions = brain._extract_suggestions(
            {
                "planned_actions": ['use(direction="left")', 'move(x=0, y=1)'],
                "planning_reasoning": "",
            }
        )

        self.assertEqual(suggestions[0]["action"], 'use(direction="left")')

    def test_failed_move_still_diversifies(self) -> None:
        brain = BigBrain(workflow_app=None)

        diversified = brain._diversify_action('move(x=3, y=0)')

        self.assertNotEqual(diversified, 'move(x=3, y=0)')
        self.assertTrue(diversified.startswith("move("))

    def test_plan_state_update_clears_external_execution_flags(self) -> None:
        brain = BigBrain(workflow_app=None)

        update = brain.get_plan_as_state_update(
            BrainPlanResult(
                suggestions=[{"action": 'move(x=1, y=0)', "reason": "advance"}],
                context_summary="ctx",
                current_task="task",
            )
        )

        self.assertFalse(update["has_execution_feedback"])
        self.assertFalse(update["execution_pending"])
        self.assertEqual(update["pending_action"], "")
        self.assertIsNone(update["pending_step_index"])
        self.assertEqual(update["pending_suggested_action"], "")
        self.assertFalse(update["force_big_brain_replan"])
        self.assertEqual(update["subtask_description"], "task")

    def test_salvage_ignores_free_form_reasoning_without_actions_block(self) -> None:
        salvaged = BigBrain._salvage_action_from_reasoning(
            "I first considered move(x=-1, y=0), but that is probably wrong, so I need to rethink."
        )

        self.assertEqual(salvaged, "")

    def test_salvage_accepts_explicit_actions_block(self) -> None:
        salvaged = BigBrain._salvage_action_from_reasoning(
            'Reasoning:\n1. The doorway is adjacent.\nActions:\n```python\nmove(x=0, y=-1)\n```'
        )

        self.assertEqual(salvaged, "move(x=0, y=-1)")

    def test_cortex_prompt_uses_main_task_not_subtask_for_current_task(self) -> None:
        client = VLLMClient(api_key="dummy")
        client.template = ACTION_PLANNING_CORTEX_TEMPLATE.read_text(encoding="utf-8")

        prompt = client._build_prompt_from_template(
            game_state={
                "task": "clear_5_stone_with_pickaxe",
                "main_task": "clear_5_stone_with_pickaxe",
                "subtask_description": "step off the porch and line up with the nearest stone",
                "gathered_info": {
                    "location": "Farm",
                    "position": [64, 15],
                    "facing_direction": "down",
                    "inventory": [],
                },
            },
            suggestion={"action": 'move(x=0, y=1)', "reason": "advance"},
            execution_log=[],
            context_summary="ctx",
            mem0_reference="",
            step=0,
            total_steps=4,
            skill_list='["move(x=0, y=1)"]',
        )

        self.assertIn("Current task:\nclear_5_stone_with_pickaxe", prompt)
        self.assertIn(
            "Current subtask:\nstep off the porch and line up with the nearest stone",
            prompt,
        )
        self.assertNotIn(
            "Current task:\nstep off the porch and line up with the nearest stone",
            prompt,
        )

    def test_cortex_prompt_includes_grounded_front_tile_blocker_summary(self) -> None:
        client = VLLMClient(api_key="dummy")
        client.template = ACTION_PLANNING_CORTEX_TEMPLATE.read_text(encoding="utf-8")

        prompt = client._build_prompt_from_template(
            game_state={
                "task": "go_to_pierre_store",
                "main_task": "go_to_pierre_store",
                "subtask_description": "leave the farmhouse porch",
                "gathered_info": {
                    "location": "Farm",
                    "position": [64, 15],
                    "facing_direction": "down",
                    "surroundings": "[0, 1]: Farmhouse",
                    "inventory": [],
                },
            },
            suggestion={"action": 'move(x=0, y=1)', "reason": "leave the porch"},
            execution_log=[
                {
                    "action": "move(x=0, y=1)",
                    "success": False,
                    "errors_info": "path is likely blocked by an obstacle",
                }
            ],
            context_summary="ctx",
            mem0_reference="",
            step=0,
            total_steps=4,
            skill_list='["move(x=0, y=1)"]',
        )

        self.assertIn("Front tile (0, 1) toward down: Farmhouse.", prompt)
        self.assertIn("route around it", prompt.lower())

    def test_cortex_prompt_uses_suggested_move_when_facing_direction_is_missing(self) -> None:
        client = VLLMClient(api_key="dummy")
        client.template = ACTION_PLANNING_CORTEX_TEMPLATE.read_text(encoding="utf-8")

        prompt = client._build_prompt_from_template(
            game_state={
                "task": "clear_10_weeds_with_scythe",
                "main_task": "clear_10_weeds_with_scythe",
                "subtask_description": "move off the porch and clear nearby weeds",
                "gathered_info": {
                    "location": "Farm",
                    "position": [64, 15],
                    "surroundings": "\n".join(
                        [
                            "[0, 1]: Farmhouse",
                            "[-3, 3]: Weeds",
                        ]
                    ),
                    "inventory": [],
                },
            },
            suggestion={"action": 'move(x=0, y=1)', "reason": "leave the porch"},
            execution_log=[],
            context_summary="ctx",
            mem0_reference="",
            step=0,
            total_steps=4,
            skill_list='["move(x=0, y=1)"]',
        )

        self.assertIn("Front tile (0, 1) toward down: Farmhouse.", prompt)

    def test_surroundings_summary_keeps_explicit_nearby_empty_cells(self) -> None:
        summary = VLLMClient._build_prompt_surroundings_summary(
            {
                "gathered_info": {
                    "surroundings": "\n".join(
                        [
                            "[0, -1]: empty",
                            "[1, 0]: Stone",
                            "[1, 1]: empty",
                            "[2, 0]: empty",
                        ]
                    )
                }
            }
        )

        self.assertIn("[1, 1]: empty", summary)
        self.assertIn("[2, 0]: empty", summary)


if __name__ == "__main__":
    unittest.main()
