from __future__ import annotations

from pathlib import Path
import unittest

from stardojo.utils.stardew_prompt_state import extract_stardew_prompt_fact_fields


class TestStardewActionPlanningContext(unittest.TestCase):
    def test_extract_prompt_fact_fields_prefers_state_and_marks_no_menu(self) -> None:
        fields = extract_stardew_prompt_fact_fields(
            state={
                "location": "BusStop",
                "position": [82, 14],
                "facing_direction": "right",
                "facing_position": [83, 14],
                "current_menu": None,
                "inventory": ['slot_index 5: Speed-Gro (quantity: 1)'],
                "chosen_item": {"index": 5, "currentitem": "Speed-Gro"},
                "surroundings": "[1, 0]: empty",
                "exits": "Town at [89, 14] (relative 7, 0)",
                "basic_knowledge": ["Check exits before cross-map travel."],
                "time": "9:10 am",
                "season": "spring",
                "money": 50000,
            },
            gathered_info={
                "location": "Farm",
                "current_menu": {"type": "Map"},
            },
        )

        self.assertEqual(fields["location"], "BusStop")
        self.assertEqual(fields["current_position"], [82, 14])
        self.assertEqual(fields["facing_direction"], "right")
        self.assertEqual(fields["current_menu"], "No Menu")
        self.assertIn("Town", fields["exits"])

    def test_extract_prompt_fact_fields_derives_front_tile_blocker_context(self) -> None:
        fields = extract_stardew_prompt_fact_fields(
            state={
                "facing_direction": "down",
                "surroundings": "[0, 1]: Farmhouse\n[1, 0]: empty\n[0, 3]: HoeDirt",
                "last_blocker_signature": "blocked | shot_0.png | move off the porch",
            },
            gathered_info={},
        )

        self.assertEqual(
            fields["front_tile_summary"],
            "Front tile (0, 1) toward down: Farmhouse. Not an obvious clearable obstacle.",
        )
        self.assertIn("route around it", fields["blocked_recovery_hint"])
        self.assertIn("farmhouse", fields["current_blocker_signature"].lower())
        self.assertIn("hoedirt", fields["nearest_grounded_target_summary"].lower())

    def test_cortex_action_planning_wires_prompt_fact_fields_into_runtime(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cradle_preprocess = (
            repo_root / "agent" / "cradle" / "provider" / "process" / "action_planning.py"
        ).read_text(encoding="utf-8")
        stardojo_preprocess = (
            repo_root / "agent" / "stardojo" / "provider" / "process" / "action_planning.py"
        ).read_text(encoding="utf-8")
        langgraph_nodes = (
            repo_root / "agent" / "cradle" / "runner" / "langgraph_nodes.py"
        ).read_text(encoding="utf-8")

        self.assertIn("extract_stardew_prompt_fact_fields", cradle_preprocess)
        self.assertIn("extract_stardew_prompt_fact_fields", stardojo_preprocess)
        self.assertIn("extract_stardew_prompt_fact_fields", langgraph_nodes)
        self.assertIn("**prompt_fact_fields", cradle_preprocess)
        self.assertIn("**prompt_fact_fields", stardojo_preprocess)
        self.assertIn("**prompt_fact_fields", langgraph_nodes)


if __name__ == "__main__":
    unittest.main()
