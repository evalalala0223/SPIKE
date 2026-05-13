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

from stardojo.utils.task_grounding import (
    build_clear_task_profile,
    classify_clearable_target,
    classify_tilling_target,
    clear_target_matches_profile,
    is_safe_empty_till_use_target,
)


class TestTaskGrounding(unittest.TestCase):
    def test_weeds_profile_only_matches_weeds(self) -> None:
        profile = build_clear_task_profile("clear_10_weeds_with_scythe")

        self.assertTrue(clear_target_matches_profile("Weeds", profile))
        self.assertFalse(clear_target_matches_profile("Stone", profile))
        self.assertFalse(clear_target_matches_profile("Grass", profile))

    def test_debris_profile_accepts_weeds_stone_and_twig_but_not_grass(self) -> None:
        profile = build_clear_task_profile("clear_30_debris_with_scythe_and_pickaxe_and_axe")

        self.assertTrue(clear_target_matches_profile("Weeds", profile))
        self.assertTrue(clear_target_matches_profile("Stone", profile))
        self.assertTrue(clear_target_matches_profile("Twig", profile))
        self.assertFalse(clear_target_matches_profile("Grass", profile))

    def test_hay_profile_accepts_grass_or_local_weeds_fallback(self) -> None:
        profile = build_clear_task_profile("forage_10_hay_with_scythe")

        self.assertTrue(clear_target_matches_profile("Grass", profile))
        self.assertTrue(clear_target_matches_profile("Hay", profile))
        self.assertTrue(clear_target_matches_profile("Weeds", profile))

    def test_open_empty_patch_counts_as_till_target(self) -> None:
        surroundings = {
            (0, 0): "empty",
            (1, 0): "empty",
            (1, 1): "empty",
            (0, 1): "empty",
            (2, 0): "empty",
        }

        candidate = classify_tilling_target(surroundings, (1, 0))

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["kind"], "open_patch")
        self.assertTrue(is_safe_empty_till_use_target(surroundings, (1, 0)))

    def test_empty_tile_next_to_mailbox_is_not_safe_till_target(self) -> None:
        surroundings = {
            (0, 0): "empty",
            (-1, 0): "empty",
            (-1, 1): "Mailbox",
            (0, 1): "empty",
            (1, 0): "empty",
            (1, 1): "empty",
            (2, 0): "empty",
            (2, 1): "empty",
        }

        self.assertIsNone(classify_tilling_target(surroundings, (-1, 0)))
        self.assertFalse(is_safe_empty_till_use_target(surroundings, (-1, 0)))

    def test_safe_till_target_is_not_rejected_just_because_player_stands_near_structures(self) -> None:
        surroundings = {
            (0, 0): "empty",
            (0, -1): "Farmhouse",
            (1, 0): "empty",
            (1, 1): "empty",
            (2, 0): "empty",
            (2, 1): "empty",
            (3, 0): "empty",
        }

        candidate = classify_tilling_target(surroundings, (2, 0))

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["kind"], "open_patch")
        self.assertTrue(
            is_safe_empty_till_use_target(
                surroundings,
                (2, 0),
                current_cell=(0, 0),
            )
        )

    def test_monster_target_maps_to_rusty_sword(self) -> None:
        classified = classify_clearable_target("npc: Name: Green Slime Friendship: 0")

        self.assertIsNotNone(classified)
        self.assertEqual(classified["family"], "monster")
        self.assertEqual(classified["tool"], "Rusty Sword")


if __name__ == "__main__":
    unittest.main()
