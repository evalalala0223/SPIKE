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

from cradle.runner.langgraph_nodes import LangGraphNodes
from stardojo.utils.task_bootstrap import build_initial_subtask


class _StubProvider:
    def __init__(self, result: dict | None = None) -> None:
        self._result = result or {}

    def __call__(self, *_: object, **__: object) -> dict:
        return dict(self._result)


class _FakeMem0Provider:
    def retrieve(self, _: str) -> dict:
        actions = [
            'move(x=0, y=5)',
            'move(x=4, y=0)',
            'move(x=1, y=0)',
            'interact(direction="right")',
        ]
        return {
            "memory_hits": [
                {
                    "state": "task=forage_1_clam | progress=Task is completed: forage_1_clam.",
                    "actions": actions,
                    "success": True,
                    "successes": 1,
                    "metadata": {
                        "progress": "Task is completed: forage_1_clam. final_quantity=1.",
                    },
                }
            ],
            "memory_confidence": 0.42,
            "memory_actions": actions,
            "memory_source": "primary",
        }

    def record_quick_path_decision(self, **_: object) -> None:
        return None


def _make_nodes(task_inference_result: dict | None = None) -> LangGraphNodes:
    providers = {
        "video_clip": _StubProvider({}),
        "self_reflection": _StubProvider({}),
        "task_inference": _StubProvider(task_inference_result),
        "action_planning": _StubProvider({}),
        "skill_execute": _StubProvider({}),
    }
    return LangGraphNodes(providers=providers)


class TestTaskInferenceGuards(unittest.TestCase):
    def test_memory_reference_formats_route_prefix_for_prompt_use(self) -> None:
        node = LangGraphNodes.__new__(LangGraphNodes)
        text = node._format_memory_reference(
            {
                "memory_hits": _FakeMem0Provider().retrieve("")["memory_hits"],
                "memory_confidence": 0.42,
                "memory_actions": _FakeMem0Provider().retrieve("")["memory_actions"],
                "memory_retrieval_mode": "hint",
            }
        )

        self.assertIn("same-task route prefix", text)
        self.assertIn('move(x=0, y=5)', text)
        self.assertIn("current facts already show the target", text)

    def test_memory_retrieve_node_outputs_memory_reference(self) -> None:
        node = LangGraphNodes.__new__(LangGraphNodes)
        node.mem0_enabled = True
        node.mem0_provider = _FakeMem0Provider()
        node.mem0_quick_path_max_consecutive_hits = 2
        node.mem0_quick_path_repeat_action_limit = 2
        node.mem0_quick_path_disable_without_embedding = False
        node.mem0_quick_path_execute_threshold = 0.92
        node.mem0_quick_path_max_retry_for_execute = 0

        result = node.memory_retrieve_node({"task": "forage_1_clam"})

        self.assertIn("memory_reference", result)
        self.assertIn("Historical successful action chain", result["memory_reference"])

    def test_setup_only_mem0_actions_only_match_inventory_setup(self) -> None:
        self.assertTrue(
            LangGraphNodes._is_setup_only_mem0_actions(["choose_item(slot_index=4)"])
        )
        self.assertFalse(
            LangGraphNodes._is_setup_only_mem0_actions(['move(x=0, y=1)'])
        )

    def test_move_only_mem0_actions_are_tracked_separately(self) -> None:
        self.assertTrue(
            LangGraphNodes._is_move_only_mem0_actions(['move(x=0, y=1)'])
        )
        self.assertFalse(
            LangGraphNodes._is_move_only_mem0_actions(['choose_item(slot_index=4)'])
        )

    def test_count_recent_successes_ignores_blocked_move_feedback(self) -> None:
        history = [
            {
                "success": True,
                "state_changed": True,
                "exec_info": {
                    "executed_skills": ['move(x=0, y=1)'],
                    "last_skill": 'move(x=0, y=1)',
                    "errors": False,
                    "errors_info": (
                        "move(x=0, y=1) toward down FAILED - "
                        "player position did not change, path is likely blocked by an obstacle"
                    ),
                },
            }
        ]

        self.assertEqual(LangGraphNodes._count_recent_successes(history), 0)

    def test_count_recent_successes_requires_state_change_for_state_aware_records(self) -> None:
        history = [
            {
                "success": True,
                "state_changed": False,
                "progress_delta": 0,
                "exec_info": {
                    "executed_skills": ['choose_item(slot_index=3)'],
                    "last_skill": 'choose_item(slot_index=3)',
                    "errors": False,
                    "errors_info": "",
                },
            }
        ]

        self.assertEqual(LangGraphNodes._count_recent_successes(history), 0)

    def test_recent_instability_detects_zero_progress_feedback(self) -> None:
        state = {
            "zero_progress_streak": 1,
            "repeated_action_streak": 0,
            "consecutive_failures": 0,
            "position_issue_detected": False,
            "has_execution_feedback": True,
            "last_state_changed": False,
            "last_exec_info": {
                "errors": False,
                "errors_info": "",
            },
        }

        self.assertTrue(LangGraphNodes._has_recent_instability(state))

    def test_empty_subtask_keeps_previous_when_refresh_was_not_forced(self) -> None:
        previous_subtask = "The current subtask is move toward the visible weeds."
        state = {
            "main_task": "clear_10_weeds_with_scythe",
            "task": "clear_10_weeds_with_scythe",
            "subtask_description": previous_subtask,
            "gathered_info": {},
            "previous_results": [],
            "dual_brain_enabled": False,
        }

        result = _make_nodes({}).task_inference_node(state)

        self.assertEqual(result["subtask_description"], previous_subtask)
        self.assertFalse(result["task_changed"])

    def test_tool_selection_subtask_is_replaced_when_item_is_already_correct_and_porch_is_blocked(self) -> None:
        main_task = "clear_10_weeds_with_scythe"
        state = {
            "main_task": main_task,
            "task": main_task,
            "subtask_description": "The current subtask is select the Scythe from the toolbar and clear nearby weeds.",
            "selected_item_already_correct": True,
            "gathered_info": {
                "selected_item_name": "Scythe",
                "facing_direction": "down",
                "surroundings": "\n".join(
                    [
                        "[0, 1]: Farmhouse",
                        "[-3, 3]: Weeds",
                    ]
                ),
            },
            "previous_results": [],
            "dual_brain_enabled": False,
        }

        result = _make_nodes(
            {"subtask_description": "The current subtask is select the Scythe from the toolbar and clear nearby weeds."}
        ).task_inference_node(state)

        self.assertIn("farmhouse porch", result["subtask_description"].lower())
        self.assertIn("weeds", result["subtask_description"].lower())
        self.assertNotIn("select scythe", result["subtask_description"].lower())

    def test_clear_recovery_continues_from_recent_progress_on_nearest_visible_weeds(self) -> None:
        nodes = _make_nodes({})

        subtask, reasoning = nodes._build_current_fact_recovery_subtask(
            "clear_10_weeds_with_scythe",
            {
                "task_progress_delta": 1,
                "gathered_info": {
                    "selected_item_name": "Scythe",
                    "surroundings": "\n".join(
                        [
                            "[1, 0]: Weeds",
                            "[2, 0]: Weeds",
                        ]
                    ),
                },
            },
        )

        self.assertIn("keep the local clear-up going", subtask.lower())
        self.assertIn("weeds", subtask.lower())
        self.assertIn("scythe", subtask.lower())
        self.assertIn("task progress increased", reasoning.lower())

    def test_mixed_clear_recovery_mentions_nearest_target_and_required_tool(self) -> None:
        nodes = _make_nodes({})

        subtask, reasoning = nodes._build_current_fact_recovery_subtask(
            "clear_30_debris_with_scythe_and_pickaxe_and_axe",
            {
                "gathered_info": {
                    "selected_item_name": "Scythe",
                    "surroundings": "[1, 0]: Stone",
                },
            },
        )

        self.assertIn("stone", subtask.lower())
        self.assertIn("pickaxe", subtask.lower())
        self.assertIn("grounded debris target", reasoning.lower())

    def test_weeds_clear_recovery_ignores_nearer_stone_target(self) -> None:
        nodes = _make_nodes({})

        subtask, reasoning = nodes._build_current_fact_recovery_subtask(
            "clear_10_weeds_with_scythe",
            {
                "gathered_info": {
                    "selected_item_name": "Scythe",
                    "surroundings": "\n".join(
                        [
                            "[1, 0]: Stone",
                            "[2, 0]: Weeds",
                        ]
                    ),
                },
            },
        )

        self.assertIn("weeds", subtask.lower())
        self.assertNotIn("stone", subtask.lower())
        self.assertNotIn("pickaxe", subtask.lower())
        self.assertIn("grounded debris target", reasoning.lower())

    def test_selected_item_already_correct_does_not_invalidate_non_tool_choose_language(self) -> None:
        nodes = _make_nodes({})
        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is pick up a visible egg from the coop floor.",
            "harvest_1_egg",
            {
                "selected_item_already_correct": True,
                "gathered_info": {
                    "selected_item_name": "Scythe",
                },
            },
        )

        self.assertEqual(reason, "")

    def test_empty_subtask_does_not_revive_stale_previous_subtask_after_conflict_refresh(self) -> None:
        main_task = "clear_10_weeds_with_scythe"
        previous_subtask = "The current subtask is move to the blocked weeds and keep swinging right."
        state = {
            "main_task": main_task,
            "task": main_task,
            "subtask_description": previous_subtask,
            "gathered_info": {},
            "latest_execution_summary": "The player had no progress and appears blocked by an obstacle.",
            "previous_results": [
                {"last_skill": 'use(direction="right")'},
                {"last_skill": 'move(x=1, y=0)'},
            ],
            "dual_brain_enabled": True,
        }

        result = _make_nodes({}).task_inference_node(state)

        self.assertEqual(result["subtask_description"], build_initial_subtask(main_task))
        self.assertNotEqual(result["subtask_description"], previous_subtask)

    def test_empty_subtask_does_not_keep_stale_menu_subtask_after_no_menu_refresh(self) -> None:
        main_task = "go_to_bed"
        previous_subtask = "The current subtask is close the inventory menu."
        success_record = {
            "success": True,
            "state_changed": True,
            "exec_info": {
                "executed_skills": ['interact(direction="up")'],
                "last_skill": 'interact(direction="up")',
                "errors": False,
                "errors_info": "",
            },
        }
        state = {
            "main_task": main_task,
            "task": main_task,
            "subtask_description": previous_subtask,
            "current_menu": {"type": "No Menu"},
            "previous_results": [success_record, success_record],
            "dual_brain_enabled": True,
            "last_menu_changed": True,
        }

        result = _make_nodes({}).task_inference_node(state)

        self.assertEqual(result["subtask_description"], build_initial_subtask(main_task))
        self.assertNotEqual(result["subtask_description"], previous_subtask)

    def test_stale_menu_subtask_from_provider_is_replaced_with_bootstrap(self) -> None:
        main_task = "go_to_bed"
        state = {
            "main_task": main_task,
            "task": main_task,
            "subtask_description": "",
            "current_menu": {"type": "No Menu"},
            "previous_results": [],
            "dual_brain_enabled": False,
        }

        result = _make_nodes(
            {"subtask_description": "The current subtask is close the inventory menu."}
        ).task_inference_node(state)

        self.assertEqual(result["subtask_description"], build_initial_subtask(main_task))

    def test_go_to_bed_visible_bed_marks_debris_search_subtask_stale(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is walk outside the farmhouse to the nearest patch of ground where weeds, stones, or twigs are visible.",
            "go_to_bed",
            {
                "gathered_info": {
                    "surroundings": "\n".join(
                        [
                            "[0, 2]: empty",
                            "[0, 3]: Bed",
                            "[1, 3]: Bed",
                        ]
                    ),
                    "description": "The character is inside the farmhouse and the bed is visible.",
                }
            },
        )

        self.assertEqual(reason, "navigation nearby target conflicts with stale route/search subtask")

    def test_go_to_bed_new_day_completion_claim_without_progress_is_rejected(self) -> None:
        nodes = _make_nodes({})

        self.assertTrue(
            nodes._subtask_claims_completion_without_progress(
                'The current subtask is complete as the "go_to_bed" task has been successfully finished with the start of a new day.',
                {
                    "main_task": "go_to_bed",
                    "task": "go_to_bed",
                    "completed": False,
                },
            )
        )

    def test_go_to_bus_stop_visible_exit_marks_clearup_subtask_stale(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is clear the weeds and rocks blocking the path directly south of the player to open the route to the farm exit.",
            "go_to_bus_stop",
            {
                "gathered_info": {
                    "exits": "Bus Stop (3 tiles right, relative offset: x=3, y=0)",
                    "surroundings": "\n".join(
                        [
                            "[0, 1]: Weeds",
                            "[1, 0]: empty",
                            "[2, 0]: empty",
                        ]
                    ),
                }
            },
        )

        self.assertEqual(reason, "navigation nearby target conflicts with stale route/search subtask")

    def test_go_to_coop_direction_conflict_with_grounded_waypoint_is_stale(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is move east toward the Coop entrance.",
            "go_to_coop",
            {
                "gathered_info": {
                    "buildings": "Coop (door: 2 tiles left, relative offset: x=-2, y=0)",
                    "surroundings": "[1, 0]: empty",
                }
            },
        )

        self.assertEqual(reason, "navigation direction conflicts with grounded waypoint")

    def test_go_to_coop_recovery_subtask_uses_grounded_building_waypoint(self) -> None:
        nodes = _make_nodes({})

        subtask, reasoning = nodes._build_current_fact_recovery_subtask(
            "go_to_coop",
            {
                "gathered_info": {
                    "buildings": "Coop (door: 2 tiles left, relative offset: x=-2, y=0)",
                    "surroundings": "[1, 0]: empty",
                }
            },
        )

        self.assertIn("coop", subtask.lower())
        self.assertIn("entrance", subtask.lower())
        self.assertIn("enter", subtask.lower())
        self.assertIn("building facts", reasoning.lower())

    def test_crafting_task_clearup_subtask_from_provider_is_replaced_with_bootstrap(self) -> None:
        main_task = "craft_1_scarecrow"
        state = {
            "main_task": main_task,
            "task": main_task,
            "subtask_description": build_initial_subtask(main_task),
            "gathered_info": {},
            "previous_results": [],
            "dual_brain_enabled": False,
        }

        result = _make_nodes(
            {"subtask_description": "The current subtask is move to the left and approach the nearby weeds to begin clearing the debris."}
        ).task_inference_node(state)

        self.assertNotEqual(
            result["subtask_description"],
            "The current subtask is move to the left and approach the nearby weeds to begin clearing the debris.",
        )
        self.assertIn("craft scarecrow", result["subtask_description"].lower())

    def test_crafting_recipe_ready_material_route_subtask_is_replaced_with_craft_ready_recovery(self) -> None:
        main_task = "craft_1_scarecrow"
        state = {
            "main_task": main_task,
            "task": main_task,
            "subtask_description": build_initial_subtask(main_task),
            "gathered_info": {
                "toolbar_information": "\n".join(
                    [
                        "slot_index 6: Wood (quantity: 50)",
                        "slot_index 7: Coal (quantity: 1)",
                        "slot_index 8: Fiber (quantity: 20)",
                    ]
                ),
                "inventory": [
                    "slot_index 6: Wood (quantity: 50)",
                    "slot_index 7: Coal (quantity: 1)",
                    "slot_index 8: Fiber (quantity: 20)",
                ],
                "current_menu": {"type": "No Menu"},
            },
            "previous_results": [],
            "dual_brain_enabled": False,
        }

        result = _make_nodes(
            {"subtask_description": "The current subtask is navigate east to the Mines entrance to obtain Coal."}
        ).task_inference_node(state)

        self.assertIn("craft scarecrow", result["subtask_description"].lower())
        self.assertNotIn("mines entrance", result["subtask_description"].lower())

    def test_crafting_missing_material_menu_subtask_is_replaced_with_local_material_recovery(self) -> None:
        main_task = "craft_1_basic_retaining_soil"
        state = {
            "main_task": main_task,
            "task": main_task,
            "subtask_description": build_initial_subtask(main_task),
            "gathered_info": {
                "selected_item_name": "Pickaxe",
                "inventory": [
                    "slot_index 0: Axe",
                    "slot_index 1: Hoe",
                    "slot_index 2: Watering Can",
                    "slot_index 3: Pickaxe",
                ],
                "toolbar_information": "\n".join(
                    [
                        "slot_index 0: Axe (quantity: 1)",
                        "slot_index 1: Hoe (quantity: 1)",
                        "slot_index 2: Watering Can (quantity: 1)",
                        "slot_index 3: Pickaxe (quantity: 1)",
                        "Currently selected item: slot_index 3: Pickaxe",
                    ]
                ),
                "surroundings": "[0, 1]: Stone",
                "current_menu": {"type": "No Menu"},
            },
            "previous_results": [],
            "dual_brain_enabled": False,
        }

        result = _make_nodes(
            {"subtask_description": "The current subtask is open the crafting menu to craft basic retaining soil."}
        ).task_inference_node(state)

        self.assertIn("stone", result["subtask_description"].lower())
        self.assertIn("pickaxe", result["subtask_description"].lower())
        self.assertIn("crafting basic retaining soil", result["subtask_description"].lower())

    def test_go_to_backwoods_recovery_uses_pet_bowl_entrance_waypoint(self) -> None:
        nodes = _make_nodes({})

        subtask, reasoning = nodes._build_current_fact_recovery_subtask(
            "go_to_backwoods",
            {
                "gathered_info": {
                    "exits": "Pet Bowl Entrance (relative offset: x=-1, y=-2)",
                    "surroundings": "[-1, 0]: empty",
                }
            },
        )

        self.assertIn("backwoods", subtask.lower())
        self.assertIn("exit", subtask.lower())
        self.assertIn("grounded waypoint", reasoning.lower())

    def test_till_recovery_subtask_prefers_grounded_open_patch(self) -> None:
        nodes = _make_nodes({})

        subtask, reasoning = nodes._build_current_fact_recovery_subtask(
            "till_5_tile_with_hoe",
            {
                "gathered_info": {
                    "selected_item_name": "Hoe",
                    "surroundings": "\n".join(
                        [
                            "[-1, 0]: empty",
                            "[-1, 1]: empty",
                            "[0, 1]: empty",
                            "[1, 0]: Farmhouse",
                        ]
                    ),
                }
            },
        )

        self.assertIn("till", subtask.lower())
        self.assertIn("open", subtask.lower())
        self.assertNotIn("weed", subtask.lower())
        self.assertNotIn("grass", subtask.lower())
        self.assertIn("grounded", reasoning.lower())

    def test_clearup_task_cultivation_subtask_from_provider_is_replaced_with_bootstrap(self) -> None:
        main_task = "chop_10_wood_with_axe"
        state = {
            "main_task": main_task,
            "task": main_task,
            "subtask_description": build_initial_subtask(main_task),
            "gathered_info": {},
            "previous_results": [],
            "dual_brain_enabled": False,
        }

        result = _make_nodes(
            {"subtask_description": "The current subtask is move south away from the farmhouse to locate the tilled farmland or search for seeds to begin cultivation."}
        ).task_inference_node(state)

        self.assertEqual(result["subtask_description"], build_initial_subtask(main_task))

    def test_combat_task_clearup_subtask_from_provider_is_replaced_with_bootstrap(self) -> None:
        main_task = "kill_10_green_slime_with_rusty_sword"
        state = {
            "main_task": main_task,
            "task": main_task,
            "subtask_description": build_initial_subtask(main_task),
            "gathered_info": {},
            "previous_results": [],
            "dual_brain_enabled": False,
        }

        result = _make_nodes(
            {"subtask_description": "The current subtask is move to the left and approach the nearby weeds to begin clearing the debris."}
        ).task_inference_node(state)

        self.assertEqual(result["subtask_description"], build_initial_subtask(main_task))

    def test_forage_visible_target_remote_search_subtask_is_replaced_with_local_pickup(self) -> None:
        main_task = "forage_1_wild_horseradish"
        state = {
            "main_task": main_task,
            "task": main_task,
            "subtask_description": build_initial_subtask(main_task),
            "gathered_info": {
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Wild Horseradish",
                        "[0, 1]: empty",
                    ]
                ),
                "description": "A wild horseradish is visible near the player.",
            },
            "previous_results": [],
            "dual_brain_enabled": False,
        }

        result = _make_nodes(
            {"subtask_description": "The current subtask is move towards the bus stop or mountain lake area to search for a visible Wild Horseradish and pick it up."}
        ).task_inference_node(state)

        self.assertEqual(
            result["subtask_description"],
            "The current subtask is face the adjacent Wild Horseradish and interact to pick it up now.",
        )

    def test_go_to_coop_clear_search_subtask_from_provider_is_replaced_with_bootstrap(self) -> None:
        main_task = "go_to_coop"
        state = {
            "main_task": main_task,
            "task": main_task,
            "subtask_description": build_initial_subtask(main_task),
            "gathered_info": {
                "surroundings": "\n".join(
                    [
                        "[0, -1]: Farmhouse Door, exit: Farmhouse Entrance",
                        "[-3, 3]: Weeds",
                    ]
                ),
            },
            "previous_results": [],
            "dual_brain_enabled": False,
        }

        result = _make_nodes(
            {"subtask_description": "The current subtask is move south away from the farmhouse to clear the immediate debris and search for the coop."}
        ).task_inference_node(state)

        self.assertEqual(result["subtask_description"], build_initial_subtask(main_task))

    def test_go_to_bus_stop_landmark_route_subtask_from_provider_is_replaced_with_bootstrap(self) -> None:
        main_task = "go_to_bus_stop"
        state = {
            "main_task": main_task,
            "task": main_task,
            "subtask_description": build_initial_subtask(main_task),
            "gathered_info": {
                "surroundings": "\n".join(
                    [
                        "[0, -1]: empty",
                        "[0, 1]: empty",
                        "[1, 0]: empty",
                    ]
                ),
            },
            "previous_results": [],
            "dual_brain_enabled": False,
        }

        result = _make_nodes(
            {"subtask_description": "The current subtask is move south along the path past the large tree to reach the farm exit."}
        ).task_inference_node(state)

        self.assertEqual(result["subtask_description"], build_initial_subtask(main_task))

    def test_go_to_bus_stop_farm_fallback_recovery_routes_east(self) -> None:
        nodes = _make_nodes({})

        subtask, reasoning = nodes._build_current_fact_recovery_subtask(
            "go_to_bus_stop",
            {
                "gathered_info": {
                    "location": "Farm",
                    "surroundings": "\n".join(
                        [
                            "[1, 0]: empty",
                            "[0, 1]: empty",
                        ]
                    ),
                },
            },
        )

        self.assertIn("move east across the farm", subtask.lower())
        self.assertIn("eastward", reasoning.lower())

    def test_forage_daffodil_town_fallback_routes_south_toward_bus_stop_grass(self) -> None:
        nodes = _make_nodes({})

        subtask, reasoning = nodes._build_current_fact_recovery_subtask(
            "forage_1_daffodil",
            {
                "gathered_info": {
                    "location": "Town",
                    "surroundings": "\n".join(
                        [
                            "[0, 1]: empty",
                            "[1, 0]: empty",
                        ]
                    ),
                    "buildings": "Saloon (door: 3 tiles down, 16 tiles right, relative offset: x=16, y=3)",
                    "description": "The player is standing in the town square with the Stardrop Saloon visible to the right.",
                }
            },
        )

        self.assertIn("south", subtask.lower())
        self.assertIn("bus stop road", subtask.lower())
        self.assertIn("daffodil", subtask.lower())
        self.assertIn("grassy", reasoning.lower())

    def test_combat_visible_enemy_search_subtask_is_replaced_with_bootstrap(self) -> None:
        main_task = "kill_10_green_slime_with_rusty_sword"
        state = {
            "main_task": main_task,
            "task": main_task,
            "subtask_description": build_initial_subtask(main_task),
            "gathered_info": {
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Green Slime",
                        "[0, 1]: empty",
                    ]
                ),
                "description": "A green slime is visible close to the player.",
            },
            "previous_results": [],
            "dual_brain_enabled": False,
        }

        result = _make_nodes(
            {"subtask_description": "The current subtask is move to adjacent unexplored tiles within the Mines to locate a visible green slime, then position yourself adjacent to it and strike with the Rusty Sword."}
        ).task_inference_node(state)

        self.assertEqual(
            result["subtask_description"],
            "The current subtask is attack the adjacent green slime with the Rusty Sword.",
        )

    def test_combat_recovery_subtask_switches_to_local_attack_when_enemy_is_adjacent(self) -> None:
        nodes = _make_nodes({})

        subtask, reasoning = nodes._build_current_fact_recovery_subtask(
            "kill_10_green_slime_with_rusty_sword",
            {
                "gathered_info": {
                    "selected_item_name": "Rusty Sword",
                    "surroundings": "\n".join(
                        [
                            "[-1, 0]: npc: Name: Green Slime Friendship: 0",
                            "[0, -1]: empty",
                        ]
                    ),
                }
            },
        )

        self.assertIn("attack", subtask.lower())
        self.assertIn("green slime", subtask.lower())
        self.assertIn("rusty sword", subtask.lower())
        self.assertIn("adjacent combat target", reasoning.lower())

    def test_shopping_recovery_subtask_stays_in_local_sell_flow_when_menu_is_open(self) -> None:
        nodes = _make_nodes({})

        subtask, reasoning = nodes._build_current_fact_recovery_subtask(
            "sell_5_parsnip_to_pierre",
            {
                "gathered_info": {
                    "current_menu": {"type": "ShopMenu"},
                    "toolbar_information": "\n".join(
                        [
                            "slot_index 6: Parsnip (quantity: 5)",
                            "Currently selected item: slot_index 0: Axe",
                        ]
                    ),
                    "inventory": ["slot_index 6: Parsnip (quantity: 5)"],
                    "surroundings": "\n".join(
                        [
                            "[0, -2]: npc: Name: Pierre",
                            "[0, -1]: SeedShop Counter",
                        ]
                    ),
                }
            },
        )

        self.assertIn("sell", subtask.lower())
        self.assertIn("parsnip", subtask.lower())
        self.assertIn("current shop menu", subtask.lower())
        self.assertIn("active menu", reasoning.lower())

    def test_farm_ops_recovery_subtask_dismisses_blocking_dialogue_first(self) -> None:
        nodes = _make_nodes({})

        subtask, reasoning = nodes._build_current_fact_recovery_subtask(
            "harvest_1_egg",
            {
                "gathered_info": {
                    "current_menu": {"type": "DialogueBox", "dialogues": ["You found a Geode!"]},
                }
            },
        )

        self.assertIn("dismiss", subtask.lower())
        self.assertIn("dialogue", subtask.lower())
        self.assertIn("blocking", reasoning.lower())
        self.assertIn("farm progress", reasoning.lower())

    def test_egg_recovery_subtask_prefers_coop_waypoint_over_barn(self) -> None:
        nodes = _make_nodes({})

        subtask, reasoning = nodes._build_current_fact_recovery_subtask(
            "harvest_1_egg",
            {
                "gathered_info": {
                    "selected_item_name": "Axe",
                    "buildings": (
                        "Deluxe Barn (door: 7 tiles left, relative offset: x=-7, y=0)\n"
                        "Deluxe Coop (door: 1 tiles up, 13 tiles left, relative offset: x=-13, y=-1)"
                    ),
                    "surroundings": "[-3, -3]: Deluxe Barn\n[-3, -2]: Deluxe Barn",
                }
            },
        )

        self.assertIn("deluxe coop", subtask.lower())
        self.assertNotIn("barn", subtask.lower())
        self.assertIn("coop-only", reasoning.lower())

    def test_milk_task_nearby_goat_subtask_is_marked_stale_without_grounded_target(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is use the Milk Pail on the nearby goat to collect milk.",
            "harvest_1_milk_with_milk_pail",
            {
                "location": "Deluxe Barn",
                "gathered_info": {
                    "location": "Deluxe Barn",
                    "description": "A goat is visible deeper inside the barn.",
                    "surroundings": "",
                }
            },
        )

        self.assertEqual(reason, "milk task claims nearby animal without grounded target")

    def test_milk_recovery_subtask_stays_inside_barn_when_animals_only_visible_in_image(self) -> None:
        nodes = _make_nodes({})

        subtask, reasoning = nodes._build_current_fact_recovery_subtask(
            "harvest_1_milk_with_milk_pail",
            {
                "location": "Deluxe Barn",
                "gathered_info": {
                    "location": "Deluxe Barn",
                    "selected_item_name": "Milk Pail",
                    "description": "A goat is visible near the trough deeper inside the barn.",
                    "surroundings": "\n".join(
                        [
                            "[-1, 0]: empty",
                            "[0, -1]: empty",
                            "[0, 1]: empty",
                            "[1, 0]: empty",
                        ]
                    ),
                }
            },
        )

        self.assertIn("inside the barn", subtask.lower())
        self.assertIn("visible goat", subtask.lower())
        self.assertIn("align before using the milk pail", reasoning.lower())

    def test_cultivation_recent_success_skip_requires_objective_progress(self) -> None:
        nodes = _make_nodes({})

        streak = nodes._count_recent_objective_successes(
            "till_5_tile_with_hoe",
            [
                {
                    "action": "move(x=1, y=0)",
                    "success": True,
                    "state_changed": True,
                    "progress_delta": 0,
                    "completed": False,
                },
                {
                    "action": "move(x=1, y=1)",
                    "success": True,
                    "state_changed": True,
                    "progress_delta": 0,
                    "completed": False,
                },
            ],
        )

        self.assertEqual(streak, 0)

    def test_cultivation_recent_success_skip_counts_real_progress(self) -> None:
        nodes = _make_nodes({})

        streak = nodes._count_recent_objective_successes(
            "till_5_tile_with_hoe",
            [
                {
                    "action": 'use(direction="right")',
                    "success": True,
                    "state_changed": True,
                    "progress_delta": 1,
                    "completed": False,
                },
                {
                    "action": 'use(direction="down")',
                    "success": True,
                    "state_changed": True,
                    "progress_delta": 1,
                    "completed": False,
                },
            ],
        )

        self.assertEqual(streak, 2)

    def test_shopping_open_menu_marks_counter_reopen_subtask_stale(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is walk to Pierre's counter, open the shop menu again, and then sell the parsnips.",
            "sell_5_parsnip_to_pierre",
            {
                "gathered_info": {
                    "current_menu": {"type": "ShopMenu"},
                    "toolbar_information": "\n".join(
                        [
                            "slot_index 6: Parsnip (quantity: 5)",
                            "Currently selected item: slot_index 6: Parsnip",
                        ]
                    ),
                    "inventory": ["slot_index 6: Parsnip (quantity: 5)"],
                    "surroundings": "\n".join(
                        [
                            "[0, -2]: npc: Name: Pierre",
                            "[0, -1]: SeedShop Counter",
                        ]
                    ),
                }
            },
        )

        self.assertEqual(
            reason,
            "shopping local menu context conflicts with stale counter/menu subtask",
        )

    def test_self_reflection_uncertain_execution_overrides_provider_success(self) -> None:
        nodes = _make_nodes({})
        nodes.self_reflection_provider = _StubProvider(
            {"success": True, "reasoning": "The model guessed success."}
        )
        state = {
            "uncertain_execution": True,
            "has_execution_feedback": True,
            "last_exec_info": {
                "executed_skills": ["choose_item(slot_index=4)"],
                "last_skill": "choose_item(slot_index=4)",
                "errors": False,
                "errors_info": "choose_item() returned no confirmation; action may not have taken effect.",
            },
        }

        result = nodes.self_reflection_node(state)

        self.assertFalse(result["reflection_result"]["success"])
        self.assertEqual(result["reflection_result"]["status"], "uncertain_execution")

    def test_cultivate_and_harvest_direct_harvest_subtask_is_marked_stale(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is move to one garlic and harvest it.",
            "cultivate_and_harvest_1_garlic",
            {"gathered_info": {}},
        )

        self.assertIn("garlic seeds", reason)

    def test_crafting_torch_prefers_tree_source_for_wood_and_sap(self) -> None:
        nodes = _make_nodes({})

        target = nodes._select_nearest_crafting_material_target(
            ["wood", "sap"],
            {
                (-2, 0): "Tree",
                (1, 0): "Weeds",
            },
        )

        self.assertEqual(target["tool"], "Axe")
        self.assertEqual(target["label"], "tree")
        self.assertEqual(target["material"], "wood and sap")

    def test_crafting_subtask_with_wrong_material_is_marked_stale(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is clear the weeds to collect fiber for crafting torch.",
            "craft_1_torch",
            {
                "toolbar_information": "\n".join(
                    [
                        "slot_index 0: Axe (quantity: 1)",
                        "slot_index 6: Fiber (quantity: 1)",
                        "Currently selected item: slot_index 0: Axe",
                    ]
                ),
                "gathered_info": {
                    "inventory": [
                        "slot_index 0: Axe (quantity: 1)",
                        "slot_index 6: Fiber (quantity: 1)",
                    ],
                    "toolbar_information": "\n".join(
                        [
                            "slot_index 0: Axe (quantity: 1)",
                            "slot_index 6: Fiber (quantity: 1)",
                            "Currently selected item: slot_index 0: Axe",
                        ]
                    ),
                    "surroundings": "[-2, 0]: Tree\n[-3, 3]: Weeds",
                },
            },
        )

        self.assertIn("unsupported material", reason)

    def test_missing_fertilizer_does_not_invalidate_inventory_or_shop_subtask(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            (
                "The current subtask is check inventory for Speed-Gro; if it is missing, "
                "route to Pierre's General Store to buy it, then prepare one dirt for fertilizing."
            ),
            "fertilize_1_dirt_with_speed_gro",
            {
                "gathered_info": {
                    "surroundings": "\n".join(
                        [
                            "[1, 0]: HoeDirt",
                            "[2, 0]: Parsnip Seeds (growing), HoeDirt",
                        ]
                    ),
                    "inventory": [],
                    "toolbar_information": "\n".join(
                        [
                            "slot_index 0: Axe",
                            "slot_index 1: Hoe",
                            "slot_index 2: Watering Can",
                        ]
                    ),
                    "selected_item_name": "Axe",
                }
            },
        )

        self.assertEqual(reason, "")

    def test_seeded_hoedirt_counts_as_grounded_fertilize_target_when_unfertilized(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            (
                "The current subtask is check inventory for Speed-Gro; if it is missing, "
                "route to Pierre's General Store to buy it, then prepare one dirt for fertilizing."
            ),
            "fertilize_1_dirt_with_speed_gro",
            {
                "gathered_info": {
                    "surroundings": "[1, 0]: Parsnip Seeds (growing), HoeDirt",
                    "inventory": ["slot_index 8: Speed-Gro"],
                    "toolbar_information": "slot_index 8: Speed-Gro",
                    "selected_item_name": "Speed-Gro",
                }
            },
        )

        self.assertEqual(reason, "nearby grounded target conflicts with route/acquisition subtask")

    def test_explicitly_fertilized_hoedirt_does_not_count_as_grounded_fertilize_target(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            (
                "The current subtask is check inventory for Speed-Gro; if it is missing, "
                "route to Pierre's General Store to buy it, then prepare one dirt for fertilizing."
            ),
            "fertilize_1_dirt_with_speed_gro",
            {
                "gathered_info": {
                    "surroundings": "[1, 0]: Parsnip Seeds (growing), Basic Retaining Soil, HoeDirt",
                    "inventory": ["slot_index 8: Speed-Gro"],
                    "toolbar_information": "slot_index 8: Speed-Gro",
                    "selected_item_name": "Speed-Gro",
                }
            },
        )

        self.assertEqual(reason, "")

    def test_prefertilized_empty_hoedirt_counts_as_grounded_sow_target(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            (
                "The current subtask is check inventory for Potato Seeds; if they are missing, "
                "route to Pierre's General Store to buy them, then return to sow."
            ),
            "sow_1_dirt_with_potato_seeds",
            {
                "gathered_info": {
                    "surroundings": "[1, 0]: Basic Retaining Soil, HoeDirt",
                    "inventory": ["slot_index 5: Potato Seeds"],
                    "toolbar_information": "slot_index 5: Potato Seeds",
                    "selected_item_name": "Potato Seeds",
                }
            },
        )

        self.assertEqual(reason, "nearby grounded target conflicts with route/acquisition subtask")

    def test_farm_ops_route_subtask_is_not_marked_stale_by_distant_scene_description_alone(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is route to the coop first, enter it, and pet visible animals there; if none are reachable in the coop, check the barn next.",
            "pet_8_animal",
            {
                "gathered_info": {
                    "surroundings": "\n".join(
                        [
                            "[0, 1]: Bed",
                            "[1, 0]: empty",
                        ]
                    ),
                    "description": "A barn and some farm animals are visible elsewhere on the farm.",
                }
            },
        )

        self.assertEqual(reason, "")

    def test_farm_ops_route_subtask_is_marked_stale_when_nearby_animal_is_grounded(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is route to the coop first, enter it, and pet visible animals there; if none are reachable in the coop, check the barn next.",
            "pet_8_animal",
            {
                "gathered_info": {
                    "surroundings": "[1, 0]: Chicken",
                }
            },
        )

        self.assertEqual(reason, "farm_ops nearby target conflicts with route/acquisition subtask")

    def test_egg_task_barn_subtask_is_marked_stale(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is move left toward the Deluxe Barn and enter it to collect an egg.",
            "harvest_1_egg",
            {
                "gathered_info": {
                    "buildings": (
                        "Deluxe Barn (door: 7 tiles left, relative offset: x=-7, y=0)\n"
                        "Deluxe Coop (door: 1 tiles up, 13 tiles left, relative offset: x=-13, y=-1)"
                    ),
                }
            },
        )

        self.assertEqual(reason, "egg task conflicts with barn subtask")

    def test_pet_bowl_subtask_is_marked_stale_when_claimed_immediate_left_bowl_is_not_adjacent(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is use the equipped Watering Can to fill the pet bowl located immediately to the left.",
            "fill_1_pet_bowl_with_watering_can",
            {
                "gathered_info": {
                    "surroundings": "\n".join(
                        [
                            "[-1, -3]: Pet Bowl",
                            "[-1, -2]: Pet Bowl",
                            "[0, -3]: Pet Bowl",
                            "[0, -2]: Pet Bowl",
                            "[-1, 0]: npc: Name: Cat_1 Friendship: 0 isTalked: False GiftsToday:",
                        ]
                    ),
                }
            },
        )

        self.assertEqual(reason, "pet bowl directional mismatch:left")

    def test_pet_bowl_subtask_is_marked_stale_when_move_left_worsens_visible_alignment(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is move left to align directly next to the pet bowl.",
            "fill_1_pet_bowl_with_watering_can",
            {
                "gathered_info": {
                    "surroundings": "\n".join(
                        [
                            "[0, -3]: Pet Bowl",
                            "[0, -2]: Pet Bowl",
                            "[1, -3]: Pet Bowl",
                            "[1, -2]: Pet Bowl",
                            "[0, 0]: empty",
                            "[1, 0]: empty",
                        ]
                    ),
                }
            },
        )

        self.assertEqual(reason, "pet bowl directional mismatch:left")

    def test_pet_bowl_subtask_is_not_stale_when_move_up_improves_visible_alignment(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is move up to align directly next to the pet bowl.",
            "fill_1_pet_bowl_with_watering_can",
            {
                "gathered_info": {
                    "surroundings": "\n".join(
                        [
                            "[0, -3]: Pet Bowl",
                            "[0, -2]: Pet Bowl",
                            "[1, -3]: Pet Bowl",
                            "[1, -2]: Pet Bowl",
                            "[0, 0]: empty",
                            "[1, 0]: empty",
                        ]
                    ),
                }
            },
        )

        self.assertEqual(reason, "")

    def test_pet_bowl_recovery_subtask_aligns_to_visible_bowl_when_not_adjacent(self) -> None:
        nodes = _make_nodes({})

        subtask, reasoning = nodes._build_current_fact_recovery_subtask(
            "fill_1_pet_bowl_with_watering_can",
            {
                "gathered_info": {
                    "selected_item_name": "Watering Can",
                    "surroundings": "\n".join(
                        [
                            "[-1, -3]: Pet Bowl",
                            "[-1, -2]: Pet Bowl",
                            "[0, -3]: Pet Bowl",
                            "[0, -2]: Pet Bowl",
                            "[-1, 0]: npc: Name: Cat_1 Friendship: 0 isTalked: False GiftsToday:",
                        ]
                    ),
                }
            },
        )

        self.assertIn("visible pet bowl", subtask.lower())
        self.assertIn("fill it with the watering can", subtask.lower())
        self.assertNotIn("immediately to the left", subtask.lower())
        self.assertIn("not adjacent", reasoning.lower())

    def test_open_coop_route_subtask_is_not_marked_stale_without_visible_hatch(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is route to the outside of the deluxe coop and open its animal door hatch.",
            "open_1_deluxe_coop",
            {
                "gathered_info": {
                    "surroundings": "[1, 0]: Deluxe Barn",
                    "description": "A coop building is visible on the farm.",
                }
            },
        )

        self.assertEqual(reason, "")

    def test_generic_pet_task_does_not_switch_to_ungrounded_barn_first_subtask(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is enter the barn to look for animals to pet.",
            "pet_3_animal",
            {
                "gathered_info": {
                    "surroundings": "\n".join(
                        [
                            "[0, -1]: Farmhouse Door, exit: Farmhouse Entrance",
                            "[1, 0]: Farmhouse",
                            "[2, 0]: empty",
                        ]
                    ),
                    "description": "A red barn roof is visible elsewhere on the farm.",
                }
            },
        )

        self.assertEqual(reason, "generic animal pet task conflicts with ungrounded barn-first subtask")

    def test_hay_scythe_tool_subtask_is_not_marked_stale_by_target_item_mismatch(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is select the Scythe from the toolbar and cut the weeds to the right to collect hay.",
            "forage_10_hay_with_scythe",
            {
                "gathered_info": {
                    "toolbar_information": "slot_index 4: Scythe",
                    "selected_item_name": "Scythe",
                    "surroundings": "[1, 0]: Weeds",
                }
            },
        )

        self.assertEqual(reason, "")

    def test_dialogue_dismiss_subtask_is_not_marked_stale_when_gathered_menu_is_open(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            'The current subtask is select "No" in the "Eat Egg?" dialogue box to dismiss the menu.',
            "forage_1_clam",
            {
                "current_menu": {"type": "No Menu"},
                "gathered_info": {
                    "current_menu": {
                        "type": "DialogueBox",
                        "dialogues": ["Eat Egg?"],
                        "responses": [{"responseText": "Yes"}, {"responseText": "No"}],
                    },
                },
            },
        )

        self.assertEqual(reason, "")

    def test_crafting_subtask_with_screenshot_grid_coordinates_is_marked_stale(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is use the Pickaxe to break the rock located at grid (3, 2) to collect Stone.",
            "craft_1_basic_retaining_soil",
            {
                "gathered_info": {
                    "toolbar_information": "\n".join(
                        [
                            "slot_index 0: Axe (quantity: 1)",
                            "slot_index 3: Pickaxe (quantity: 1)",
                        ]
                    ),
                    "selected_item_name": "Pickaxe",
                    "surroundings": "\n".join(
                        [
                            "[0, -1]: Farmhouse Door, exit: Farmhouse Entrance",
                            "[-3, 3]: Weeds",
                        ]
                    ),
                }
            },
        )

        self.assertEqual(reason, "subtask uses screenshot grid coordinates")

    def test_go_to_backwoods_outdoor_farmhouse_exit_subtask_is_marked_stale(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is move south toward the FarmHouse exit door to leave the building.",
            "go_to_backwoods",
            {
                "gathered_info": {
                    "location": "Farm",
                    "exits": "Pet Bowl Entrance (relative offset: x=0, y=-8)",
                }
            },
        )

        self.assertIn("farmhouse-exit", reason)

    def test_go_to_backwoods_grounded_exit_route_is_not_marked_stale(self) -> None:
        nodes = _make_nodes({})

        reason = nodes._subtask_conflicts_with_current_facts(
            "The current subtask is move north across the farm toward the pet bowl entrance path that leads to the Backwoods.",
            "go_to_backwoods",
            {
                "gathered_info": {
                    "location": "Farm",
                    "exits": "Pet Bowl Entrance (relative offset: x=0, y=-8)",
                }
            },
        )

        self.assertEqual(reason, "")


if __name__ == "__main__":
    unittest.main()
