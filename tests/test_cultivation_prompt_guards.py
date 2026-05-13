from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = ROOT / "agent" / "res" / "stardew" / "prompts" / "templates"


class TestCultivationPromptGuards(unittest.TestCase):
    def test_action_planning_cultivation_prompt_mentions_outdoor_door_guard(self) -> None:
        prompt = (PROMPT_ROOT / "action_planning_cultivation.prompt").read_text(encoding="utf-8")

        self.assertIn("Farmhouse Door / Farmhouse Entrance", prompt)
        self.assertIn("do NOT interact with it", prompt)

    def test_action_planning_cultivation_prompt_marks_hoedirt_as_done_for_tilling(self) -> None:
        prompt = (PROMPT_ROOT / "action_planning_cultivation.prompt").read_text(encoding="utf-8")

        self.assertIn("visible HoeDirt / tilled soil means that tile is already done", prompt)

    def test_littlebrain_prompt_blocks_door_entry_for_outdoor_cultivation(self) -> None:
        prompt = (PROMPT_ROOT / "action_planning_littlebrain.prompt").read_text(encoding="utf-8")

        self.assertIn("For outdoor cultivation tasks", prompt)
        self.assertIn("Farmhouse Door / house door", prompt)
        self.assertIn("not the next interaction target", prompt)

    def test_littlebrain_prompt_marks_hoedirt_as_already_tilled(self) -> None:
        prompt = (PROMPT_ROOT / "action_planning_littlebrain.prompt").read_text(encoding="utf-8")

        self.assertIn("visible HoeDirt / tilled soil means already tilled", prompt)

    def test_task_inference_cultivation_prompt_avoids_farmhouse_entry_for_outdoor_crop_work(self) -> None:
        prompt = (PROMPT_ROOT / "task_inference_cultivation.prompt").read_text(encoding="utf-8")

        self.assertIn("Farmhouse Door / Farmhouse Entrance is not the next crop target", prompt)
        self.assertIn("instead of entering the door", prompt)


if __name__ == "__main__":
    unittest.main()
