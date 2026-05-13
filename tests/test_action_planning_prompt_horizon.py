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


PROMPT_FILES = [
    ROOT / "agent" / "res" / "stardew" / "prompts" / "templates" / "action_planning_cultivation.prompt",
    ROOT / "agent" / "res" / "stardew" / "prompts" / "templates" / "action_planning_farm_clearup.prompt",
    ROOT / "agent" / "res" / "stardew" / "prompts" / "templates" / "action_planning_farm_ops.prompt",
    ROOT / "agent" / "res" / "stardew" / "prompts" / "templates" / "action_planning_shopping.prompt",
]


class TestActionPlanningPromptHorizon(unittest.TestCase):
    def test_profile_specific_prompts_require_four_step_plans(self) -> None:
        for prompt_path in PROMPT_FILES:
            text = prompt_path.read_text(encoding="utf-8")
            self.assertIn("exactly 4", text, msg=str(prompt_path))
            self.assertIn("# Step 4", text, msg=str(prompt_path))


if __name__ == "__main__":
    unittest.main()
