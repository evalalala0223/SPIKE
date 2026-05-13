from __future__ import annotations

import unittest

from stardojo.utils.stardew_prompt_state import extract_stardew_prompt_fact_fields


class TestStardewPromptState(unittest.TestCase):
    def test_open_gathered_menu_wins_over_stale_state_no_menu(self) -> None:
        fields = extract_stardew_prompt_fact_fields(
            state={
                "current_menu": {"type": "No Menu"},
            },
            gathered_info={
                "current_menu": {
                    "type": "DialogueBox",
                    "dialogues": ["Eat Egg?"],
                },
            },
        )

        self.assertEqual(fields["current_menu"]["type"], "DialogueBox")

    def test_map_description_infers_map_page_when_only_no_menu_is_reported(self) -> None:
        fields = extract_stardew_prompt_fact_fields(
            state={
                "current_menu": {"type": "No Menu"},
            },
            gathered_info={
                "description": (
                    "Overall, the image shows a map of the Stardew Valley area, "
                    "specifically highlighting the StarDojo Farm region."
                ),
            },
        )

        self.assertEqual(fields["current_menu"]["type"], "MapPage")
        self.assertTrue(fields["current_menu"]["inferred_from_description"])

    def test_current_surroundings_override_stale_explicit_front_tile_summary(self) -> None:
        fields = extract_stardew_prompt_fact_fields(
            state={
                "front_tile_summary": "Front tile (0, 1) toward down: Stone. Clearable with Pickaxe.",
                "blocked_recovery_hint": "Prefer clearing the stone.",
                "current_blocker_signature": "Old blocker signature",
                "nearest_grounded_target_summary": "Old nearest target",
                "facing_direction": "down",
            },
            gathered_info={
                "facing_direction": "down",
                "surroundings": "\n".join(
                    [
                        "[0, 1]: Farmhouse",
                        "[1, 2]: empty",
                        "[-3, 3]: Weeds",
                    ]
                ),
            },
        )

        self.assertIn("farmhouse", fields["front_tile_summary"].lower())
        self.assertNotIn("stone", fields["front_tile_summary"].lower())
        self.assertIn("farmhouse", fields["current_blocker_signature"].lower())
        self.assertIn("weeds", fields["nearest_grounded_target_summary"].lower())

    def test_explicit_front_tile_summary_is_kept_when_current_facts_are_missing(self) -> None:
        fields = extract_stardew_prompt_fact_fields(
            state={
                "front_tile_summary": "Front tile (0, 1) toward down: Farmhouse. Not an obvious clearable obstacle.",
                "blocked_recovery_hint": "Route around it.",
                "current_blocker_signature": "Stored blocker signature",
                "nearest_grounded_target_summary": "Stored nearest target summary",
            },
            gathered_info={},
        )

        self.assertIn("farmhouse", fields["front_tile_summary"].lower())
        self.assertIn("route around it", fields["blocked_recovery_hint"].lower())
        self.assertIn("stored blocker signature", fields["current_blocker_signature"].lower())
        self.assertIn("stored nearest target summary", fields["nearest_grounded_target_summary"].lower())

    def test_description_with_structured_relative_lines_feeds_front_tile_and_grounded_target(self) -> None:
        fields = extract_stardew_prompt_fact_fields(
            state={
                "facing_direction": "down",
            },
            gathered_info={
                "facing_direction": "down",
                "description": "\n".join(
                    [
                        "[0, 1]: Farmhouse",
                        "[1, 0]: HoeDirt",
                        "[2, 0]: Weeds",
                    ]
                ),
            },
        )

        self.assertIn("[0, 1]: Farmhouse", fields["surroundings"])
        self.assertIn("farmhouse", fields["front_tile_summary"].lower())
        self.assertIn("weeds", fields["nearest_grounded_target_summary"].lower())

    def test_natural_language_surroundings_are_preserved_when_not_structured(self) -> None:
        fields = extract_stardew_prompt_fact_fields(
            state={
                "surroundings": "The coop hatch is directly beside you.",
            },
            gathered_info={},
        )

        self.assertIn("coop hatch", fields["surroundings"].lower())

    def test_hay_tasks_prefer_grass_in_nearest_grounded_target_summary(self) -> None:
        fields = extract_stardew_prompt_fact_fields(
            state={
                "task": "forage_10_hay_with_scythe",
            },
            gathered_info={
                "surroundings": "\n".join(
                    [
                        "[0, -1]: Weeds",
                        "[1, 0]: Grass",
                    ]
                ),
            },
        )

        self.assertIn("hay target", fields["nearest_grounded_target_summary"].lower())
        self.assertIn("grass", fields["nearest_grounded_target_summary"].lower())
        self.assertNotIn("weeds", fields["nearest_grounded_target_summary"].lower())


if __name__ == "__main__":
    unittest.main()
