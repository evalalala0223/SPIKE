from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _load_stardew_action_planning_module():
    module_name = "_test_cradle_stardew_action_planning_process"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    provider_stub = types.ModuleType("cradle.provider")

    class _BaseProvider:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    provider_stub.BaseProvider = _BaseProvider
    previous_provider_module = sys.modules.get("cradle.provider")
    sys.modules["cradle.provider"] = provider_stub
    try:
        module_path = (
            Path(__file__).resolve().parents[1]
            / "agent"
            / "cradle"
            / "provider"
            / "process"
            / "action_planning.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module spec from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_provider_module is not None:
            sys.modules["cradle.provider"] = previous_provider_module
        else:
            sys.modules.pop("cradle.provider", None)


_ACTION_PLANNING_MODULE = _load_stardew_action_planning_module()
StardewActionPlanningPreprocessProvider = (
    _ACTION_PLANNING_MODULE.StardewActionPlanningPreprocessProvider
)


class _FakeMemory:
    def __init__(self, history: dict[str, list[object]]) -> None:
        self._history = history
        self.working_area: dict[str, object] = {}

    def get_recent_history(self, key: str, k: int = 1) -> list[object]:
        values = self._history.get(key, [])
        return list(values[:k])


class TestActionPlanningPromptParams(unittest.TestCase):
    def test_nearest_grounded_target_summary_uses_surroundings_even_without_facing_direction(self) -> None:
        fake_memory = _FakeMemory(
            {
                "toolbar_information": [
                    "Currently selected item: slot_index 3: Pickaxe"
                ],
                "selected_position": [3],
                "summarization": ["Current summary"],
                "task_description": ["clear_5_stone_with_pickaxe"],
                "subtask_description": [
                    "The current subtask is clear a nearby stone."
                ],
                "gathered_info": [
                    {
                        "current_menu": {"type": "No Menu"},
                        "buildings": ["Farmhouse"],
                        "inventory": ["Pickaxe"],
                        "surroundings": "\n".join(
                            [
                                "[1, 0]: Stone",
                                "[0, 1]: empty",
                            ]
                        ),
                        "crops": [],
                        "furniture": [],
                        "npcs": [],
                        "exits": ["Farm Exit South"],
                    }
                ],
            }
        )
        provider = StardewActionPlanningPreprocessProvider(
            gm=None,
            toolbar_information="",
        )

        with patch.object(_ACTION_PLANNING_MODULE, "memory", fake_memory):
            params = provider()

        self.assertIn("stone", params["nearest_grounded_target_summary"].lower())

    def test_clear_weeds_prompt_summary_ignores_nearer_non_weeds_debris(self) -> None:
        fake_memory = _FakeMemory(
            {
                "toolbar_information": [
                    "Currently selected item: slot_index 4: Scythe"
                ],
                "selected_position": [4],
                "summarization": ["Current summary"],
                "task_description": ["clear_10_weeds_with_scythe"],
                "subtask_description": [
                    "The current subtask is clear nearby weeds."
                ],
                "gathered_info": [
                    {
                        "current_menu": {"type": "No Menu"},
                        "inventory": ["Scythe"],
                        "surroundings": "\n".join(
                            [
                                "[1, 0]: Stone",
                                "[2, 0]: Weeds",
                                "[0, 1]: empty",
                            ]
                        ),
                        "crops": [],
                        "furniture": [],
                        "npcs": [],
                        "exits": [],
                    }
                ],
            }
        )
        provider = StardewActionPlanningPreprocessProvider(
            gm=None,
            toolbar_information="",
        )

        with patch.object(_ACTION_PLANNING_MODULE, "memory", fake_memory):
            params = provider()

        self.assertIn("weeds", params["nearest_grounded_target_summary"].lower())
        self.assertNotIn("stone", params["nearest_grounded_target_summary"].lower())

    def test_till_prompt_summary_prefers_open_patch_over_weeds(self) -> None:
        fake_memory = _FakeMemory(
            {
                "toolbar_information": [
                    "Currently selected item: slot_index 1: Hoe"
                ],
                "selected_position": [1],
                "summarization": ["Current summary"],
                "task_description": ["till_5_tile_with_hoe"],
                "subtask_description": [
                    "The current subtask is till a nearby patch."
                ],
                "gathered_info": [
                    {
                        "current_menu": {"type": "No Menu"},
                        "inventory": ["Hoe"],
                        "surroundings": "\n".join(
                            [
                                "[1, 0]: Weeds",
                                "[2, 0]: empty",
                                "[2, 1]: empty",
                                "[1, 1]: empty",
                                "[3, 0]: empty",
                            ]
                        ),
                        "crops": [],
                        "furniture": [],
                        "npcs": [],
                        "exits": [],
                    }
                ],
            }
        )
        provider = StardewActionPlanningPreprocessProvider(
            gm=None,
            toolbar_information="",
        )

        with patch.object(_ACTION_PLANNING_MODULE, "memory", fake_memory):
            params = provider()

        self.assertIn("till target", params["nearest_grounded_target_summary"].lower())
        self.assertNotIn("weeds", params["nearest_grounded_target_summary"].lower())

    def test_cultivation_prompt_params_include_expected_interface_fields(self) -> None:
        fake_memory = _FakeMemory(
            {
                "toolbar_information": [
                    "Currently selected item: slot_index 5: Basic Retaining Soil"
                ],
                "selected_position": [5],
                "summarization": ["Current summary"],
                "task_description": ["fertilize_5_dirt_with_basic_retaining_soil"],
                "subtask_description": [
                    "The current subtask is select Basic Retaining Soil and fertilize nearby HoeDirt."
                ],
                "gathered_info": [
                    {
                        "current_menu": {"type": "No Menu"},
                        "buildings": ["Farmhouse"],
                        "inventory": ["Basic Retaining Soil"],
                        "facing_direction": "down",
                        "surroundings": "\n".join(
                            [
                                "[0, 1]: Farmhouse",
                                "[1, 2]: empty",
                                "[0, 3]: HoeDirt",
                            ]
                        ),
                        "crops": [],
                        "furniture": [],
                        "npcs": [],
                        "exits": ["Farm Exit South"],
                    }
                ],
                "latest_execution_summary": [
                    "No progress after the previous fertilize attempt."
                ],
                "action_feedback": [
                    "No progress after the previous fertilize attempt."
                ],
                "date_time": ["Fri. 5 06:10"],
                "task_progress_quantity": [1],
                "zero_progress_streak": [2],
                "repeated_action_streak": [3],
                "last_errors_info": ["interact() returned no confirmation; action may not have taken effect."],
            }
        )
        provider = StardewActionPlanningPreprocessProvider(
            gm=None,
            toolbar_information="",
        )

        with patch.object(_ACTION_PLANNING_MODULE, "memory", fake_memory):
            params = provider()

        self.assertEqual(
            params["action_feedback"],
            "No progress after the previous fertilize attempt.",
        )
        self.assertEqual(params["target_item"], "Basic Retaining Soil")
        self.assertEqual(params["source_type"], "inventory_preloaded")
        self.assertIn("preloaded", params["source_detail"].lower())
        self.assertEqual(params["current_menu"], {"type": "No Menu"})
        self.assertEqual(params["buildings"], ["Farmhouse"])
        self.assertIn("farmhouse", params["front_tile_summary"].lower())
        self.assertIn("farmhouse", params["current_blocker_signature"].lower())
        self.assertIn("hoedirt", params["nearest_grounded_target_summary"].lower())
        self.assertEqual(params["date_time"], "Fri. 5 06:10")
        self.assertIn("No progress", params["task_progress_summary"])
        self.assertIn("zero_progress_streak=2", params["failure_signals"])
        self.assertIn("repeated_action_streak=3", params["failure_signals"])

    def test_pet_task_prompt_params_include_animal_housing_hint(self) -> None:
        fake_memory = _FakeMemory(
            {
                "toolbar_information": [
                    "Currently selected item: slot_index 0: Axe"
                ],
                "selected_position": [0],
                "summarization": ["No animals are currently visible nearby."],
                "task_description": ["pet_3_animal"],
                "subtask_description": [
                    "The current subtask is route to the coop first, enter it, and pet visible animals there; if none are reachable in the coop, check the barn next."
                ],
                "gathered_info": [
                    {
                        "current_menu": {"type": "No Menu"},
                        "buildings": ["Farmhouse"],
                        "inventory": [],
                        "crops": [],
                        "furniture": [],
                        "npcs": [],
                        "exits": ["Farm Exit South"],
                    }
                ],
                "latest_execution_summary": [
                    "No animals were visible outside the farmhouse."
                ],
                "action_feedback": [
                    "No animals were visible outside the farmhouse."
                ],
            }
        )
        provider = StardewActionPlanningPreprocessProvider(
            gm=None,
            toolbar_information="",
        )

        with patch.object(_ACTION_PLANNING_MODULE, "memory", fake_memory):
            params = provider()

        self.assertEqual(params["target_item"], "Animal")
        self.assertEqual(params["source_type"], "animal_housing")
        self.assertIn("coop", params["source_detail"].lower())
        self.assertIn("barn", params["source_detail"].lower())
        self.assertEqual(params["current_menu"], {"type": "No Menu"})
        self.assertEqual(params["exits"], ["Farm Exit South"])

    def test_go_to_coop_prompt_params_include_farm_building_hint(self) -> None:
        fake_memory = _FakeMemory(
            {
                "toolbar_information": [
                    "Currently selected item: slot_index 0: Axe"
                ],
                "selected_position": [0],
                "summarization": ["The player is still outside on the farm."],
                "task_description": ["go_to_coop"],
                "subtask_description": [
                    "The current subtask is route to the coop entrance and enter it."
                ],
                "gathered_info": [
                    {
                        "current_menu": {"type": "No Menu"},
                        "buildings": ["Farmhouse", "Coop"],
                        "inventory": [],
                        "crops": [],
                        "furniture": [],
                        "npcs": [],
                        "exits": ["Farm Exit South"],
                    }
                ],
                "latest_execution_summary": [
                    "The player is still outside the coop."
                ],
                "action_feedback": [
                    "The player is still outside the coop."
                ],
            }
        )
        provider = StardewActionPlanningPreprocessProvider(
            gm=None,
            toolbar_information="",
        )

        with patch.object(_ACTION_PLANNING_MODULE, "memory", fake_memory):
            params = provider()

        self.assertEqual(params["target_item"], "Coop")
        self.assertEqual(params["source_type"], "farm_building")
        self.assertIn("enter", params["source_detail"].lower())

    def test_non_preloaded_seed_task_prompt_params_include_shop_route(self) -> None:
        fake_memory = _FakeMemory(
            {
                "toolbar_information": [
                    "Currently selected item: slot_index 0: Axe"
                ],
                "selected_position": [0],
                "summarization": ["Potato Seeds are not visible in the current toolbar."],
                "task_description": ["sow_1_dirt_with_potato_seeds"],
                "subtask_description": [
                    "The current subtask is check inventory for Potato Seeds; if they are missing, route to Pierre's General Store to buy them, then prepare one dirt for sowing."
                ],
                "gathered_info": [
                    {
                        "current_menu": {"type": "No Menu"},
                        "buildings": ["Farmhouse"],
                        "inventory": [],
                        "crops": [],
                        "furniture": [],
                        "npcs": [],
                        "exits": ["Farm Exit East", "Farm Exit South"],
                    }
                ],
                "latest_execution_summary": [
                    "Potato Seeds are not present in the current facts."
                ],
                "action_feedback": [
                    "Potato Seeds are not present in the current facts."
                ],
            }
        )
        provider = StardewActionPlanningPreprocessProvider(
            gm=None,
            toolbar_information="",
        )

        with patch.object(_ACTION_PLANNING_MODULE, "memory", fake_memory):
            params = provider()

        self.assertEqual(params["target_item"], "Potato Seeds")
        self.assertEqual(params["source_type"], "inventory_or_shop")
        self.assertIn("pierre", params["source_detail"].lower())

    def test_working_area_subtask_overrides_stale_recent_history(self) -> None:
        fake_memory = _FakeMemory(
            {
                "toolbar_information": [
                    "Currently selected item: slot_index 3: Scythe"
                ],
                "selected_position": [3],
                "summarization": ["Old summary from history."],
                "task_description": ["clear_10_weeds_with_scythe"],
                "subtask_description": [
                    "The current subtask is select Scythe from the toolbar."
                ],
                "history_summary": ["Old summary from history."],
                "gathered_info": [
                    {
                        "current_menu": {"type": "No Menu"},
                        "buildings": ["Farmhouse"],
                        "inventory": ["Scythe"],
                        "crops": [],
                        "furniture": [],
                        "npcs": [],
                        "exits": [],
                    }
                ],
            }
        )
        fake_memory.working_area.update(
            {
                "task_description": "clear_10_weeds_with_scythe",
                "subtask_description": "The current subtask is walk down from the porch and clear nearby weeds.",
                "history_summary": "The selected Scythe is already correct, so movement is the next blocker.",
                "action_feedback": "The player is still standing on the porch.",
            }
        )
        provider = StardewActionPlanningPreprocessProvider(
            gm=None,
            toolbar_information="",
        )

        with patch.object(_ACTION_PLANNING_MODULE, "memory", fake_memory):
            params = provider()

        self.assertEqual(
            params["subtask_description"],
            "The current subtask is walk down from the porch and clear nearby weeds.",
        )
        self.assertEqual(
            params["history_summary"],
            "The selected Scythe is already correct, so movement is the next blocker.",
        )
        self.assertEqual(
            params["action_feedback"],
            "The player is still standing on the porch.",
        )

if __name__ == "__main__":
    unittest.main()
