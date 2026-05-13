from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from stardojo.environment.skill import Skill
from stardojo.environment.stardew.atomic_skills.basic_skills import move


class TestSkillSerialization(unittest.TestCase):
    def test_importable_skill_serializes_by_module_reference(self) -> None:
        skill = Skill(
            skill_name="move",
            skill_function=move.skill_function,
            skill_embedding=np.array([1.0, 2.0], dtype=np.float64),
            skill_code="def move(x, y):\n    return x, y\n",
            skill_code_base64="",
        )

        data = skill.to_dict()

        self.assertEqual(data["skill_module"], move.skill_function.__module__)
        self.assertEqual(data["skill_function_name"], move.skill_function.__name__)
        self.assertEqual(data["skill_function"], "")

    def test_importable_skill_round_trips_without_dill_payload(self) -> None:
        skill = Skill(
            skill_name="move",
            skill_function=move.skill_function,
            skill_embedding=np.array([1.0, 2.0], dtype=np.float64),
            skill_code="def move(x, y):\n    return x, y\n",
            skill_code_base64="",
        )

        restored = Skill.from_dict(skill.to_dict())

        self.assertIs(restored.skill_function, move.skill_function)
        self.assertEqual(restored.skill_name, "move")
        self.assertEqual(restored.skill_embedding.tolist(), [1.0, 2.0])


if __name__ == "__main__":
    unittest.main()
