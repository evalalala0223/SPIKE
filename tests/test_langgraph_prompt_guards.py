from __future__ import annotations

import unittest

from cradle.runner.langgraph_nodes import LangGraphNodes


class TestLangGraphPromptGuards(unittest.TestCase):
    def setUp(self) -> None:
        self.nodes = LangGraphNodes.__new__(LangGraphNodes)

    @staticmethod
    def _make_state(**overrides):
        state = {
            "gathered_info": {},
            "latest_execution_summary": "",
            "failure_signals": "",
            "recent_execution_feedback": [],
            "task_progress_summary": "",
            "last_menu_changed": False,
            "current_menu": "",
            "surroundings": "",
            "furniture": "",
            "npcs": "",
            "buildings": "",
            "toolbar_information": "",
            "chosen_item": "",
            "selected_item_name": "",
        }
        state.update(overrides)
        return state

    def test_current_task_has_nearby_target_for_farm_ops_positive_cases(self) -> None:
        cases = {
            "pet_3_animal": self._make_state(npcs="A chicken is standing right next to the player."),
            "open_1_deluxe_coop": self._make_state(surroundings="The coop hatch is directly beside you."),
            "fill_1_feeding_bench_with_hay": self._make_state(furniture="A Feeding Bench and Hopper are nearby."),
            "forage_10_hay_with_scythe": self._make_state(surroundings="Tall grass covers the nearby farm tiles."),
        }

        for main_task, state in cases.items():
            with self.subTest(main_task=main_task):
                self.assertTrue(self.nodes._current_task_has_nearby_target(main_task, state))

    def test_hay_task_treats_local_weeds_as_nearby_scytheable_target(self) -> None:
        state = self._make_state(surroundings="[1, 0]: Weeds")

        self.assertTrue(self.nodes._current_task_has_nearby_target("forage_10_hay_with_scythe", state))

    def test_feeding_bench_hay_task_does_not_treat_weeds_as_nearby_target(self) -> None:
        state = self._make_state(surroundings="[1, 0]: Weeds")

        self.assertFalse(self.nodes._current_task_has_nearby_target("fill_1_feeding_bench_with_hay", state))

    def test_current_task_has_nearby_target_does_not_use_buildings_as_farm_ops_truth_source(self) -> None:
        state = self._make_state(buildings="There is a Coop and a Barn somewhere on this map.")

        self.assertFalse(self.nodes._current_task_has_nearby_target("pet_3_animal", state))
        self.assertFalse(self.nodes._current_task_has_nearby_target("open_1_deluxe_coop", state))
        self.assertFalse(self.nodes._current_task_has_nearby_target("fill_1_pet_bowl_with_watering_can", state))

    def test_farm_ops_route_subtasks_become_stale_once_target_is_nearby(self) -> None:
        cases = (
            (
                "pet_3_animal",
                self._make_state(npcs="A chicken is already beside the player."),
                "The current subtask is route to the coop first and look for the animals.",
            ),
            (
                "fill_1_pet_bowl_with_watering_can",
                self._make_state(furniture="A Pet Bowl is right beside the farmhouse porch."),
                "The current subtask is route to the pet bowl by the farmhouse before interacting.",
            ),
            (
                "fill_1_feeding_bench_with_hay",
                self._make_state(furniture="A Feeding Bench and Hopper are visible nearby."),
                "The current subtask is route into the coop or barn and check the feeder.",
            ),
        )

        for main_task, state, previous_subtask in cases:
            with self.subTest(main_task=main_task):
                reason = self.nodes._subtask_conflicts_with_current_facts(previous_subtask, main_task, state)
                self.assertIn("farm_ops nearby target conflicts with route/acquisition subtask", reason)

    def test_reasoning_blob_is_not_treated_as_precise_subtask(self) -> None:
        self.assertFalse(
            self.nodes._looks_like_precise_subtask_text(
                "1. Item Availability: Hay is present.\n2. Tool Selection: choose Hay.\n3. Move to the coop."
            )
        )
        self.assertTrue(
            self.nodes._looks_like_precise_subtask_text(
                "The current subtask is route to the outside of the deluxe coop and open its animal door hatch."
            )
        )


if __name__ == "__main__":
    unittest.main()
