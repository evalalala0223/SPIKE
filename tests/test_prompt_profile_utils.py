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


from stardojo.utils.prompt_profile_utils import infer_stardew_prompt_profile


class TestPromptProfileUtils(unittest.TestCase):
    def test_forage_hay_with_scythe_uses_farm_ops_profile(self) -> None:
        self.assertEqual(
            infer_stardew_prompt_profile("forage_10_hay_with_scythe"),
            "farm_ops",
        )

    def test_fill_feeding_bench_with_hay_stays_farm_ops(self) -> None:
        self.assertEqual(
            infer_stardew_prompt_profile("fill_1_feeding_bench_with_hay"),
            "farm_ops",
        )

    def test_craft_scarecrow_uses_crafting_profile(self) -> None:
        self.assertEqual(
            infer_stardew_prompt_profile("craft_1_scarecrow"),
            "crafting",
        )


if __name__ == "__main__":
    unittest.main()
