from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROMPT_ROOT = ROOT / "agent" / "res" / "stardew" / "prompts" / "templates"


class TestNavigationFarmOpsPromptGuards(unittest.TestCase):
    def test_farm_ops_prompt_says_hay_task_requires_grass_or_hay(self) -> None:
        prompt = (PROMPT_ROOT / "action_planning_farm_ops.prompt").read_text(encoding="utf-8")

        self.assertIn("forage_10_hay_with_scythe", prompt)
        self.assertIn("Grass or Hay", prompt)
        self.assertIn("if the current local patch only shows nearby scytheable Weeds/Fiber", prompt)

    def test_farm_ops_task_inference_prompt_allows_local_weeds_fallback_for_hay(self) -> None:
        prompt = (PROMPT_ROOT / "task_inference_farm_ops.prompt").read_text(encoding="utf-8")

        self.assertIn("explicit Grass or Hay", prompt)
        self.assertIn("only shows nearby scytheable Weeds/Fiber", prompt)

    def test_cortex_prompts_prefer_short_combat_search_when_enemy_not_visible(self) -> None:
        action_prompt = (PROMPT_ROOT / "action_planning_cortex.prompt").read_text(encoding="utf-8")
        task_prompt = (PROMPT_ROOT / "task_inference_cortex.prompt").read_text(encoding="utf-8")

        self.assertIn("If a kill_* task has no visible enemy", action_prompt)
        self.assertIn("short grounded mine-search moves", action_prompt)
        self.assertIn("if no enemy is currently visible", task_prompt)
        self.assertIn("short grounded mine-floor search", task_prompt)


if __name__ == "__main__":
    unittest.main()
