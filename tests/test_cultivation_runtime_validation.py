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


from stardojo.utils.cortex_runtime_utils import validate_runtime_pre_execution_action


class TestCultivationRuntimeValidation(unittest.TestCase):
    def test_underscore_task_names_are_treated_as_cultivation(self) -> None:
        validation = validate_runtime_pre_execution_action(
            state={
                "task": "till_5_tile_with_hoe",
                "toolbar_information": "Currently selected item: slot_index 1: Axe",
                "gathered_info": {
                    "current_menu": {"type": "No Menu"},
                    "surroundings": "[0, -1]: Diggable soil",
                    "selected_item_name": "Axe",
                },
            },
            action_text='use(direction="up")',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(validation.get("invalid_reason"), "cultivation_validation:till_requires_hoe")

    def test_till_repeated_empty_target_requests_local_recovery(self) -> None:
        state = {
            "task": "till_5_tile_with_hoe",
            "zero_progress_streak": 2,
            "repeated_action_streak": 2,
            "position_issue_detected": False,
            "toolbar_information": "\n".join(
                [
                    "slot_index 1: Hoe (quantity: 1)",
                    "Currently selected item: slot_index 1: Hoe",
                ]
            ),
            "inventory": [
                "slot_index 1: Hoe (quantity: 1)",
            ],
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "toolbar_information": "\n".join(
                    [
                        "slot_index 1: Hoe (quantity: 1)",
                        "Currently selected item: slot_index 1: Hoe",
                    ]
                ),
                "inventory": [
                    "slot_index 1: Hoe (quantity: 1)",
                ],
                "surroundings": "\n".join(
                    [
                        "[0, -1]: empty",
                        "[1, 0]: Diggable soil",
                    ]
                ),
                "selected_item_name": "Hoe",
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='use(direction="up")',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(
            validation.get("invalid_reason"),
            "cultivation_validation:till_repeated_empty_target",
        )
        self.assertEqual(validation.get("required_change_type"), "change_position")
        self.assertTrue(str(validation.get("fallback_action", "")).startswith("move("))

    def test_till_empty_target_near_house_is_rejected_even_before_repetition(self) -> None:
        state = {
            "task": "till_5_tile_with_hoe",
            "location": "FarmHouse",
            "zero_progress_streak": 0,
            "repeated_action_streak": 0,
            "position_issue_detected": False,
            "toolbar_information": "\n".join(
                [
                    "slot_index 1: Hoe (quantity: 1)",
                    "Currently selected item: slot_index 1: Hoe",
                ]
            ),
            "inventory": [
                "slot_index 1: Hoe (quantity: 1)",
            ],
            "gathered_info": {
                "location": "FarmHouse",
                "current_menu": {"type": "No Menu"},
                "toolbar_information": "\n".join(
                    [
                        "slot_index 1: Hoe (quantity: 1)",
                        "Currently selected item: slot_index 1: Hoe",
                    ]
                ),
                "inventory": [
                    "slot_index 1: Hoe (quantity: 1)",
                ],
                "surroundings": "\n".join(
                    [
                        "[0, 0]: empty",
                        "[0, -1]: Farmhouse",
                        "[1, -1]: Farmhouse",
                        "[1, 0]: empty",
                    ]
                ),
                "selected_item_name": "Hoe",
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='use(direction="right")',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(
            validation.get("invalid_reason"),
            "cultivation_validation:till_inside_house_requires_exit",
        )
        self.assertEqual(validation.get("required_change_type"), "change_position")
        self.assertTrue(str(validation.get("fallback_action", "")).startswith("move("))

    def test_fertilize_inside_house_rewrites_redundant_setup_to_exit(self) -> None:
        state = {
            "task": "fertilize_5_dirt_with_basic_retaining_soil",
            "location": "FarmHouse",
            "toolbar_information": "\n".join(
                [
                    "slot_index 5: Basic Retaining Soil (quantity: 5)",
                    "Currently selected item: slot_index 5: Basic Retaining Soil",
                ]
            ),
            "inventory": [
                "slot_index 5: Basic Retaining Soil (quantity: 5)",
            ],
            "gathered_info": {
                "location": "FarmHouse",
                "current_menu": {"type": "No Menu"},
                "toolbar_information": "\n".join(
                    [
                        "slot_index 5: Basic Retaining Soil (quantity: 5)",
                        "Currently selected item: slot_index 5: Basic Retaining Soil",
                    ]
                ),
                "inventory": [
                    "slot_index 5: Basic Retaining Soil (quantity: 5)",
                ],
                "surroundings": "\n".join(
                    [
                        "[0, 0]: empty",
                        "[0, 1]: empty",
                        "[1, 0]: Bed",
                    ]
                ),
                "selected_item_name": "Basic Retaining Soil",
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text="choose_item(slot_index=5)",
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(
            validation.get("invalid_reason"),
            "cultivation_validation:fertilize_inside_house_requires_exit",
        )
        self.assertEqual(validation.get("required_change_type"), "change_position")
        self.assertTrue(str(validation.get("fallback_action", "")).startswith("move("))

    def test_sow_inside_house_rewrites_selected_seed_action_to_exit(self) -> None:
        state = {
            "task": "sow_5_dirt_with_cauliflower_seeds",
            "location": "FarmHouse",
            "toolbar_information": "\n".join(
                [
                    "slot_index 6: Cauliflower Seeds (quantity: 5)",
                    "Currently selected item: slot_index 6: Cauliflower Seeds",
                ]
            ),
            "inventory": [
                "slot_index 6: Cauliflower Seeds (quantity: 5)",
            ],
            "gathered_info": {
                "location": "FarmHouse",
                "current_menu": {"type": "No Menu"},
                "toolbar_information": "\n".join(
                    [
                        "slot_index 6: Cauliflower Seeds (quantity: 5)",
                        "Currently selected item: slot_index 6: Cauliflower Seeds",
                    ]
                ),
                "inventory": [
                    "slot_index 6: Cauliflower Seeds (quantity: 5)",
                ],
                "surroundings": "\n".join(
                    [
                        "[0, 0]: empty",
                        "[0, 1]: empty",
                        "[1, 0]: Bed",
                    ]
                ),
                "selected_item_name": "Cauliflower Seeds",
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text="choose_item(slot_index=6)",
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(
            validation.get("invalid_reason"),
            "cultivation_validation:sow_inside_house_requires_exit",
        )
        self.assertEqual(validation.get("required_change_type"), "change_position")
        self.assertTrue(str(validation.get("fallback_action", "")).startswith("move("))

    def test_sow_menu_open_inventory_is_rewritten_to_choose_item(self) -> None:
        state = {
            "task": "sow_1_dirt_with_potato_seeds",
            "target_item": "Potato Seeds",
            "toolbar_information": "\n".join(
                [
                    "slot_index 6: Potato Seeds (quantity: 1)",
                    "Currently selected item: slot_index 1: Hoe",
                ]
            ),
            "inventory": [
                "slot_index 6: Potato Seeds (quantity: 1)",
            ],
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "toolbar_information": "\n".join(
                    [
                        "slot_index 6: Potato Seeds (quantity: 1)",
                        "Currently selected item: slot_index 1: Hoe",
                    ]
                ),
                "inventory": [
                    "slot_index 6: Potato Seeds (quantity: 1)",
                ],
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='menu(option="open", menu_name="inventory")',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(
            validation.get("invalid_reason"),
            "cultivation_validation:sow_menu_open_instead_of_select",
        )
        self.assertEqual(validation.get("fallback_action"), "choose_item(slot_index=6)")

    def test_shop_move_while_aligned_is_rewritten_to_interact(self) -> None:
        state = {
            "task": "sell_5_parsnip_to_pierre",
            "prompt_profile": "shopping",
            "facing_direction": "up",
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "facing_direction": "up",
                "surroundings": "\n".join(
                    [
                        "[0, -2]: npc: Name: Pierre",
                        "[0, -1]: SeedShop Counter",
                        "[0, 0]: empty",
                    ]
                ),
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='move(x=0, y=-1)',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(validation.get("invalid_reason"), "runtime_validation:service_move_while_aligned")
        self.assertEqual(validation.get("fallback_action"), 'interact(direction="up")')

    def test_shop_sell_uses_selected_inventory_slot(self) -> None:
        state = {
            "task": "sell_5_parsnip_to_pierre",
            "prompt_profile": "shopping",
            "selected_position": 6,
            "selected_item_name": "Parsnip",
            "target_item": "Parsnip",
            "gathered_info": {
                "current_menu": {"type": "ShopMenu"},
                "selected_position": 6,
                "selected_item_name": "Parsnip",
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='choose_option(option_index=2, quantity=5, direction="out")',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(validation.get("invalid_reason"), "runtime_validation:shop_sell_wrong_slot")
        self.assertEqual(
            validation.get("fallback_action"),
            'choose_option(option_index=7, quantity=5, direction="out")',
        )

    def test_shop_sell_wrong_slot_is_rewritten_even_when_menu_type_is_not_shopmenu(self) -> None:
        state = {
            "task": "sell_5_parsnip_to_pierre",
            "prompt_profile": "shopping",
            "target_item": "Parsnip",
            "toolbar_information": "\n".join(
                [
                    "slot_index 6: Parsnip (quantity: 5)",
                    "Currently selected item: slot_index 6: Parsnip",
                ]
            ),
            "gathered_info": {
                "current_menu": {"type": "DialogueBox"},
                "toolbar_information": "\n".join(
                    [
                        "slot_index 6: Parsnip (quantity: 5)",
                        "Currently selected item: slot_index 6: Parsnip",
                    ]
                ),
                "inventory": ["slot_index 6: Parsnip (quantity: 5)"],
                "selected_item_name": "Parsnip",
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='choose_option(option_index=1, quantity=5, direction="out")',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(validation.get("invalid_reason"), "runtime_validation:shop_sell_wrong_slot")
        self.assertEqual(
            validation.get("fallback_action"),
            'choose_option(option_index=7, quantity=5, direction="out")',
        )

    def test_shop_choose_option_without_menu_realigns_to_counter_context(self) -> None:
        state = {
            "task": "sell_5_parsnip_to_pierre",
            "prompt_profile": "shopping",
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "surroundings": "\n".join(
                    [
                        "[-1, -2]: npc: Name: Pierre",
                        "[-1, -1]: SeedShop Counter",
                        "[-1, 0]: empty",
                        "[0, 0]: empty",
                    ]
                ),
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='choose_option(option_index=1, quantity=5, direction="out")',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(validation.get("invalid_reason"), "runtime_validation:service_menu_requires_context")
        self.assertEqual(validation.get("fallback_action"), "move(x=-1, y=0)")

    def test_combat_move_is_rewritten_to_attack_when_adjacent_enemy_is_visible(self) -> None:
        state = {
            "task": "kill_10_green_slime_with_rusty_sword",
            "prompt_profile": "combat",
            "toolbar_information": "\n".join(
                [
                    "slot_index 5: Rusty Sword (quantity: 1)",
                    "Currently selected item: slot_index 5: Rusty Sword",
                ]
            ),
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "toolbar_information": "\n".join(
                    [
                        "slot_index 5: Rusty Sword (quantity: 1)",
                        "Currently selected item: slot_index 5: Rusty Sword",
                    ]
                ),
                "selected_item_name": "Rusty Sword",
                "surroundings": "\n".join(
                    [
                        "[-1, 0]: npc: Name: Green Slime Friendship: 0",
                        "[0, -1]: empty",
                    ]
                ),
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='move(x=1, y=0)',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(
            validation.get("invalid_reason"),
            "runtime_validation:combat_adjacent_enemy_requires_attack",
        )
        self.assertEqual(validation.get("fallback_action"), 'use(direction="left")')

    def test_farm_ops_route_conflicting_move_is_rewritten_toward_grounded_waypoint(self) -> None:
        state = {
            "task": "harvest_1_egg",
            "prompt_profile": "farm_ops",
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "buildings": "Deluxe Coop (door: 17 tiles left, relative offset: x=-17, y=-1)",
                "surroundings": "\n".join(
                    [
                        "[-1, 0]: Farmhouse",
                        "[0, -1]: empty",
                        "[1, 0]: empty",
                    ]
                ),
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='move(x=4, y=0)',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(
            validation.get("invalid_reason"),
            "runtime_validation:farm_ops_route_conflicts_with_waypoint",
        )
        self.assertEqual(validation.get("fallback_action"), 'move(x=0, y=-1)')

    def test_navigation_without_waypoint_clamps_long_move_to_short_probe(self) -> None:
        state = {
            "task": "go_to_bus_stop",
            "prompt_profile": "navigation",
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "surroundings": "\n".join(
                    [
                        "[-1, 0]: empty",
                        "[-2, 0]: empty",
                        "[-3, 0]: empty",
                    ]
                ),
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='move(x=-10, y=0)',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(
            validation.get("invalid_reason"),
            "runtime_validation:navigation_requires_short_probe",
        )
        self.assertEqual(validation.get("fallback_action"), 'move(x=-3, y=0)')

    def test_crafting_task_prefers_direct_craft_when_materials_ready(self) -> None:
        state = {
            "task": "craft_1_torch",
            "prompt_profile": "crafting",
            "target_item": "Torch",
            "inventory": [
                "slot_index 6: Wood (quantity: 1)",
                "slot_index 7: Sap (quantity: 2)",
            ],
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "inventory": [
                    "slot_index 6: Wood (quantity: 1)",
                    "slot_index 7: Sap (quantity: 2)",
                ],
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='menu(option="open", menu_name="crafting")',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(validation.get("invalid_reason"), "runtime_validation:crafting_should_direct_craft")
        self.assertEqual(validation.get("fallback_action"), 'craft(item="Torch")')

    def test_crafting_task_infers_target_item_from_task_when_missing(self) -> None:
        state = {
            "task": "craft_1_torch",
            "prompt_profile": "crafting",
            "inventory": [
                "slot_index 6: Wood (quantity: 1)",
                "slot_index 7: Sap (quantity: 2)",
            ],
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "inventory": [
                    "slot_index 6: Wood (quantity: 1)",
                    "slot_index 7: Sap (quantity: 2)",
                ],
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='menu(option="open", menu_name="crafting")',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(validation.get("invalid_reason"), "runtime_validation:crafting_should_direct_craft")
        self.assertEqual(validation.get("fallback_action"), 'craft(item="Torch")')

    def test_crafting_task_does_not_force_direct_craft_when_target_already_present(self) -> None:
        state = {
            "task": "craft_1_torch",
            "prompt_profile": "crafting",
            "target_item": "Torch",
            "inventory": [
                "slot_index 6: Torch (quantity: 1)",
                "slot_index 7: Wood (quantity: 1)",
                "slot_index 8: Sap (quantity: 2)",
            ],
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "inventory": [
                    "slot_index 6: Torch (quantity: 1)",
                    "slot_index 7: Wood (quantity: 1)",
                    "slot_index 8: Sap (quantity: 2)",
                ],
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='menu(option="open", menu_name="crafting")',
        )

        self.assertTrue(validation.get("is_valid", True))

    def test_crafting_world_gather_action_is_allowed_when_materials_missing(self) -> None:
        state = {
            "task": "craft_1_basic_retaining_soil",
            "prompt_profile": "crafting",
            "target_item": "Basic Retaining Soil",
            "toolbar_information": "Currently selected item: slot_index 3: Pickaxe",
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "selected_item_name": "Pickaxe",
                "toolbar_information": "Currently selected item: slot_index 3: Pickaxe",
                "surroundings": "[0, 1]: Stone",
            },
            "inventory": [],
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='use(direction="down")',
        )

        self.assertTrue(validation.get("is_valid", True))

    def test_sleep_dialogue_forces_confirm_option(self) -> None:
        state = {
            "task": "go_to_bed",
            "prompt_profile": "navigation",
            "gathered_info": {
                "current_menu": {
                    "type": "DialogueBox",
                    "dialogues": ["Go to sleep for the night?"],
                    "responses": [{"responseKey": "Yes"}, {"responseKey": "No"}],
                },
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='interact(direction="up")',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(validation.get("invalid_reason"), "runtime_validation:sleep_dialogue_requires_confirm")
        self.assertEqual(validation.get("fallback_action"), "choose_option(option_index=1, quantity=0)")

    def test_navigation_blocked_route_is_rerouted_before_execution(self) -> None:
        state = {
            "task": "go_to_backwoods",
            "prompt_profile": "navigation",
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "surroundings": "\n".join(
                    [
                        "[0, -1]: Farmhouse Door, exit: Farmhouse Entrance",
                        "[0, 1]: empty",
                        "[1, 0]: empty",
                        "[-1, 0]: empty",
                    ]
                ),
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='move(x=0, y=-10)',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(validation.get("invalid_reason"), "runtime_validation:navigation_blocked_route")
        self.assertTrue(str(validation.get("fallback_action", "")).startswith("move("))

    def test_navigation_preserves_adjacent_exit_anchor_move(self) -> None:
        state = {
            "task": "go_to_backwoods",
            "prompt_profile": "navigation",
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "surroundings": "[0, -1]: Pet Bowl Entrance, exit: Backwoods",
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='move(x=0, y=-1)',
        )

        self.assertTrue(validation.get("is_valid", True))

    def test_navigation_exit_route_clamps_large_move_to_unit_step(self) -> None:
        state = {
            "task": "go_to_bus_stop",
            "prompt_profile": "navigation",
            "zero_progress_streak": 1,
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "exits": "Bus Stop Exit (relative offset: x=-3, y=0)",
                "surroundings": "\n".join(
                    [
                        "[-1, 0]: empty",
                        "[-2, 0]: empty",
                        "[-3, 0]: Bus Stop Exit",
                    ]
                ),
            },
        }

        validation = validate_runtime_pre_execution_action(
            state=state,
            action_text='move(x=-5, y=0)',
        )

        self.assertFalse(validation.get("is_valid", True))
        self.assertEqual(validation.get("invalid_reason"), "runtime_validation:navigation_requires_unit_step")
        self.assertEqual(validation.get("fallback_action"), "move(x=-1, y=0)")


if __name__ == "__main__":
    unittest.main()
