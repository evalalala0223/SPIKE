from __future__ import annotations

import contextlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from cradle.runner.little_brain import LittleBrain
from cradle.runner.vllm_client import VLLMClient, VLLMDecision


class _StubVLLMClient:
    def __init__(self, decision: VLLMDecision) -> None:
        self._decision = decision

    def decide(self, **_: object) -> VLLMDecision:
        return self._decision


class TestLittleBrainGuards(unittest.TestCase):
    def test_health_check_reuses_recent_shared_success_cache(self) -> None:
        client = VLLMClient(api_key="dummy", model="Qwen/Qwen3.5-397B-A17B-FP8")

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "health.json"
            lock_path = Path(temp_dir) / "health.lock"
            with mock.patch.object(
                client,
                "_get_shared_health_check_paths",
                return_value=(str(cache_path), str(lock_path)),
            ):
                client._write_shared_health_check_cache(True)
                with mock.patch("cradle.runner.vllm_client.requests.post") as post_mock:
                    self.assertTrue(client.health_check())
                post_mock.assert_not_called()

    def test_health_check_reuses_recent_failure_cache_while_probe_lock_exists(self) -> None:
        client = VLLMClient(api_key="dummy", model="Qwen/Qwen3.5-397B-A17B-FP8")

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "health.json"
            lock_path = Path(temp_dir) / "health.lock"
            lock_path.write_text("locked", encoding="utf-8")

            with mock.patch.object(
                client,
                "_get_shared_health_check_paths",
                return_value=(str(cache_path), str(lock_path)),
            ):
                with mock.patch.object(
                    client,
                    "_read_shared_health_check_cache",
                    side_effect=[None, False],
                ):
                    with mock.patch("cradle.runner.vllm_client.requests.post") as post_mock:
                        self.assertFalse(client.health_check())
                    post_mock.assert_not_called()

    def test_parse_response_normalizes_legacy_directional_syntax(self) -> None:
        client = VLLMClient(api_key="dummy")

        decision = client._parse_response(
            "ACTION: use(down)\nREASON: clear stone",
            {"action": 'move(x=0, y=1)', "reason": "fallback"},
        )

        self.assertEqual(decision.action, 'use(direction="down")')

    def test_parse_response_strips_trailing_status_noise(self) -> None:
        client = VLLMClient(api_key="dummy")

        decision = client._parse_response(
            'move(x=1, y=1) -> success")',
            {"action": 'move(x=0, y=1)', "reason": "fallback"},
        )

        self.assertEqual(decision.action, 'move(x=1, y=1)')

    def test_parse_response_does_not_mine_actions_from_reasoning_sentences(self) -> None:
        client = VLLMClient(api_key="dummy")

        decision = client._parse_response(
            'Reasoning:\nExecuting `use(direction="down")` would be ineffective here.',
            {"action": 'move(x=0, y=1)', "reason": "fallback"},
        )

        self.assertEqual(decision.action, 'move(x=0, y=1)')

    def test_parse_response_does_not_mine_skill_doc_examples(self) -> None:
        client = VLLMClient(api_key="dummy")

        decision = client._parse_response(
            "Reasoning:\nThe valid action description says call use(up) to use against (0,-1).",
            {"action": 'move(x=0, y=1)', "reason": "fallback"},
        )

        self.assertEqual(decision.action, 'move(x=0, y=1)')

    def test_parse_response_marks_invalidated_suggestion_for_validator_side_recovery(self) -> None:
        client = VLLMClient(api_key="dummy")

        decision = client._parse_response(
            (
                "Reasoning:\n"
                "The planning brain suggested move(x=0, y=1), but the inventory is empty and "
                "the critical blocker is the wrong tool. Moving would be useless here."
            ),
            {"action": 'move(x=0, y=1)', "reason": "fallback"},
        )

        self.assertFalse(decision.escalate)
        self.assertEqual(decision.action, 'move(x=0, y=1)')
        self.assertEqual(decision.reason, "parse_fallback_invalidated_suggestion")

    def test_parse_response_marks_new_invalidity_markers_for_validator_side_recovery(self) -> None:
        client = VLLMClient(api_key="dummy")

        decision = client._parse_response(
            (
                "Reasoning:\n"
                "Path not found from the current tile and the required item is missing item "
                "for the requested action, so this suggestion is not helpful."
            ),
            {"action": 'move(x=0, y=1)', "reason": "fallback"},
        )

        self.assertFalse(decision.escalate)
        self.assertEqual(decision.action, 'move(x=0, y=1)')
        self.assertEqual(decision.reason, "parse_fallback_invalidated_suggestion")

    def test_extract_canonical_action_candidate_rejects_placeholder_craft_argument(self) -> None:
        self.assertEqual(VLLMClient._extract_canonical_action_candidate("craft(item)"), "")

    def test_tilling_context_supports_snake_case_task_name(self) -> None:
        self.assertTrue(
            VLLMClient._is_tilling_or_digging_context(
                {
                    "task": "till_5_tile_with_hoe",
                    "gathered_info": {},
                }
            )
        )

    def test_watering_context_excludes_pet_bowl_tasks(self) -> None:
        self.assertFalse(
            VLLMClient._is_watering_context(
                {
                    "task": "fill_1_pet_bowl_with_watering_can",
                    "task_description": "fill_1_pet_bowl_with_watering_can",
                    "subtask_description": "Use the Watering Can to fill the pet bowl.",
                    "gathered_info": {},
                }
            )
        )

    def test_front_obstacle_context_preserves_explicit_route_summaries(self) -> None:
        context = VLLMClient._build_front_obstacle_context(
            {
                "current_blocker_signature": "Immediate blocker on the current facing line: Farmhouse at (0, -1).",
                "nearest_grounded_target_summary": "Nearest grounded route context: Bus Stop (relative offset: x=4, y=0).",
                "gathered_info": {
                    "surroundings": "",
                    "current_blocker_signature": "Immediate blocker on the current facing line: Farmhouse at (0, -1).",
                    "nearest_grounded_target_summary": "Nearest grounded route context: Bus Stop (relative offset: x=4, y=0).",
                },
            },
            [],
            hint_action='move(x=4, y=0)',
        )

        self.assertIn("farmhouse", context["current_blocker_signature"].lower())
        self.assertIn("bus stop", context["nearest_grounded_target_summary"].lower())

    def test_autonomous_local_recovery_prefers_visible_coop_route(self) -> None:
        action = VLLMClient._build_autonomous_local_recovery_action(
            game_state={
                "task": "go_to_coop",
                "target_item": "Coop",
                "source_type": "farm_building",
                "gathered_info": {
                    "surroundings": "[0, 1]: empty",
                    "buildings": "Coop (door: 4 tiles right, relative offset: x=4, y=0)",
                    "inventory": [],
                    "toolbar_information": "Currently selected item: slot_index 0: Axe",
                },
            }
        )

        self.assertEqual(action, "move(x=4, y=0)")

    def test_autonomous_local_recovery_pet_bowl_prefers_direct_interact(self) -> None:
        action = VLLMClient._build_autonomous_local_recovery_action(
            game_state={
                "task": "fill_1_pet_bowl_with_watering_can",
                "target_item": "Pet Bowl",
                "source_type": "pet_area",
                "gathered_info": {
                    "furniture": "Pet Bowl (relative offset: x=1, y=0)",
                    "surroundings": "[1, 0]: Pet Bowl",
                    "inventory": [],
                    "toolbar_information": "Currently selected item: slot_index 2: Watering Can",
                    "selected_item_name": "Watering Can",
                },
            }
        )

        self.assertEqual(action, 'interact(direction="right")')

    def test_local_route_recovery_pet_bowl_outside_prefers_bowl_over_adjacent_farmhouse(self) -> None:
        action = VLLMClient._build_local_route_recovery_action(
            game_state={
                "task": "fill_1_pet_bowl_with_watering_can",
                "source_type": "pet_area",
                "location": "Farm",
                "gathered_info": {
                    "location": "Farm",
                    "buildings": "\n".join(
                        [
                            "Farmhouse (door: 1 tiles up, relative offset: x=0, y=-1)",
                            "Pet Bowl (door: 9 tiles up, 12 tiles left, relative offset: x=-12, y=-9)",
                        ]
                    ),
                    "surroundings": "[0, -1]: Farmhouse Door, exit: Farmhouse Entrance",
                },
            }
        )

        self.assertTrue(action.startswith("move("))
        self.assertIn("x=-4", action)
        self.assertNotIn('interact(direction="up")', action)

    def test_local_route_recovery_go_to_bed_inside_farmhouse_uses_bedward_fallback(self) -> None:
        action = VLLMClient._build_local_route_recovery_action(
            game_state={
                "task": "go_to_bed",
                "location": "FarmHouse",
                "gathered_info": {
                    "location": "FarmHouse",
                    "furniture": "Bed at 10, 9",
                    "surroundings": "\n".join(
                        [
                            "[1, 0]: empty",
                            "[0, 1]: empty",
                        ]
                    ),
                },
            }
        )

        self.assertEqual(action, "move(x=1, y=1)")

    def test_move_axis_swap_is_reverted_without_blocker_evidence(self) -> None:
        suggestion = {"action": 'move(x=0, y=1)'}
        decision = VLLMDecision(action='move(x=1, y=0)', reason="side_move", escalate=False)
        state = {"gathered_info": {"surroundings": ""}}

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertEqual(validated.action, 'move(x=0, y=1)')
        self.assertTrue(validated.reason.startswith("follow_big_brain_move"))

    def test_single_axis_magnitude_shrink_is_reverted(self) -> None:
        suggestion = {"action": 'move(x=3, y=0)'}
        decision = VLLMDecision(action='move(x=1, y=0)', reason="shorter", escalate=False)
        state = {"gathered_info": {"surroundings": ""}}

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertEqual(validated.action, 'move(x=3, y=0)')

    def test_single_axis_magnitude_shrink_is_allowed_under_combat_instability(self) -> None:
        suggestion = {"action": 'move(x=5, y=0)'}
        decision = VLLMDecision(action='move(x=1, y=0)', reason="shorter", escalate=False)
        state = {
            "task": "kill_5_green_slime_with_rusty_sword",
            "zero_progress_streak": 1,
            "repeated_action_streak": 0,
            "position_issue_detected": False,
            "gathered_info": {"surroundings": ""},
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=1, y=0)')
        self.assertEqual(validated.reason, "shorter")

    def test_combat_adjacent_monster_use_override_is_allowed_over_bigbrain_move(self) -> None:
        suggestion = {"action": 'move(x=1, y=0)'}
        decision = VLLMDecision(action='use(direction="right")', reason="attack", escalate=False)
        state = {
            "task": "kill_1_green_slime_with_rusty_sword",
            "toolbar_information": "Currently selected item: slot_index 5: Rusty Sword",
            "gathered_info": {
                "selected_item_name": "Rusty Sword",
                "inventory": ["slot_index 5: Rusty Sword"],
                "surroundings": "[1, 0]: npc: Name: Green Slime Friendship: 0",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'use(direction="right")')
        self.assertEqual(validated.reason, "attack")

    def test_single_axis_magnitude_shrink_is_allowed_when_visible_path_blocker_is_ahead(self) -> None:
        suggestion = {"action": 'move(x=0, y=10)'}
        decision = VLLMDecision(action='move(x=0, y=1)', reason="shorter", escalate=False)
        state = {
            "task": "go_to_bus_stop",
            "gathered_info": {
                "surroundings": "\n".join(
                    [
                        "[0, 1]: empty",
                        "[0, 2]: Weeds",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=0, y=1)')
        self.assertEqual(validated.reason, "shorter")

    def test_tilling_move_off_farmhouse_is_not_rewritten_into_hoe_use(self) -> None:
        suggestion = {"action": 'move(x=0, y=1)'}
        decision = VLLMDecision(action='move(x=0, y=1)', reason="follow", escalate=False)
        state = {
            "task": "till_5_tile_with_hoe",
            "toolbar_information": "\n".join(
                [
                    "slot_index 1: Hoe (quantity: 1)",
                    "Currently selected item: slot_index 1: Hoe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Hoe",
                "inventory": [
                    "slot_index 1: Hoe (quantity: 1)",
                ],
                "surroundings": "\n".join(
                    [
                        "[0, 0]: Farmhouse",
                        "[0, 1]: empty",
                        "[1, 0]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=0, y=1)')

    def test_tilling_move_near_structure_cluster_is_not_rewritten_into_empty_tile_use(self) -> None:
        suggestion = {"action": 'move(x=1, y=0)'}
        decision = VLLMDecision(action='move(x=1, y=0)', reason="follow", escalate=False)
        state = {
            "task": "till_5_tile_with_hoe",
            "toolbar_information": "\n".join(
                [
                    "slot_index 1: Hoe (quantity: 1)",
                    "Currently selected item: slot_index 1: Hoe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Hoe",
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
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=1, y=0)')

    def test_tilling_use_inside_farmhouse_is_rewritten_to_exit_recovery(self) -> None:
        suggestion = {"action": 'use(direction="right")'}
        decision = VLLMDecision(action='use(direction="right")', reason="till", escalate=False)
        state = {
            "task": "till_5_tile_with_hoe",
            "location": "FarmHouse",
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "gathered_info": {
                "location": "FarmHouse",
                "selected_item_name": "Hoe",
                "surroundings": "\n".join(
                    [
                        "[1, 0]: empty",
                        "[0, 1]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertTrue(validated.action.startswith("move("))
        self.assertEqual(validated.reason, "till_inside_house_exit_recovery")

    def test_navigation_move_off_porch_is_not_rewritten_sideways(self) -> None:
        suggestion = {"action": 'move(x=5, y=0)'}
        decision = VLLMDecision(action='move(x=5, y=0)', reason="follow", escalate=False)
        state = {
            "task": "go_to_bus_stop",
            "gathered_info": {
                "description": "The player character is standing on a porch outside the farmhouse.",
                "surroundings": "\n".join(
                    [
                        "[0, 0]: empty",
                        "[0, 1]: empty",
                        "[1, -1]: Farmhouse",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=5, y=0)')

    def test_clearup_move_override_is_allowed_when_it_matches_grounded_local_recovery(self) -> None:
        suggestion = {"action": 'move(x=1, y=0)'}
        decision = VLLMDecision(action='move(x=-3, y=3)', reason="toward weeds", escalate=False)
        state = {
            "task": "clear_10_weeds_with_scythe",
            "toolbar_information": "\n".join(
                [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 4: Scythe (quantity: 1)",
                    "Currently selected item: slot_index 4: Scythe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Scythe",
                "inventory": [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 4: Scythe (quantity: 1)",
                ],
                "surroundings": "\n".join(
                    [
                        "[-3, 3]: Weeds",
                        "[1, 0]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=-3, y=3)')
        self.assertEqual(validated.reason, "toward weeds")

    def test_clear_weeds_use_on_grass_is_rerouted_to_profile_aligned_target(self) -> None:
        suggestion = {"action": 'use(direction="right")'}
        decision = VLLMDecision(action='use(direction="right")', reason="swing", escalate=False)
        state = {
            "task": "clear_10_weeds_with_scythe",
            "toolbar_information": "\n".join(
                [
                    "slot_index 4: Scythe (quantity: 1)",
                    "Currently selected item: slot_index 4: Scythe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Scythe",
                "inventory": [
                    "slot_index 4: Scythe (quantity: 1)",
                ],
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Grass",
                        "[-1, 0]: Weeds",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'use(direction="left")')
        self.assertEqual(validated.reason, "clear_profile_target_recovery")

    def test_clear_weeds_use_on_grass_escalates_without_profile_aligned_fallback(self) -> None:
        suggestion = {"action": 'use(direction="right")'}
        decision = VLLMDecision(action='use(direction="right")', reason="swing", escalate=False)
        state = {
            "task": "clear_10_weeds_with_scythe",
            "toolbar_information": "\n".join(
                [
                    "slot_index 4: Scythe (quantity: 1)",
                    "Currently selected item: slot_index 4: Scythe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Scythe",
                "inventory": [
                    "slot_index 4: Scythe (quantity: 1)",
                ],
                "surroundings": "[1, 0]: Grass",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertTrue(validated.escalate)
        self.assertEqual(validated.reason, "clear_profile_target_mismatch")

    def test_blocked_recovery_progressive_offset_does_not_keep_large_axis_stride(self) -> None:
        recovery = VLLMClient._build_structure_blocked_move_recovery(
            action_text='move(x=10, y=0)',
            game_state={
                "task": "go_to_bus_stop",
                "gathered_info": {
                    "surroundings": "[1, 0]: Farmhouse",
                },
            },
            blocker="Farmhouse",
        )

        self.assertTrue(recovery.startswith("move("))
        components = VLLMClient._parse_move_components(recovery)
        self.assertIsNotNone(components)
        self.assertLessEqual(abs(int(components[0])), 4)
        self.assertLessEqual(abs(int(components[1])), 4)
        self.assertNotEqual(int(components[0]), 10)

    def test_noop_move_cannot_expand_into_ungrounded_long_move(self) -> None:
        suggestion = {"action": 'move(x=0, y=0)'}
        decision = VLLMDecision(action='move(x=5, y=5)', reason="search", escalate=False)
        state = {"gathered_info": {"surroundings": ""}}

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=0, y=0)')
        self.assertTrue(validated.reason.startswith("follow_big_brain_move"))

    def test_noop_big_brain_move_allows_grounded_tilling_use_override(self) -> None:
        suggestion = {"action": 'move(x=0, y=0)'}
        decision = VLLMDecision(action='use(direction="left")', reason="till", escalate=False)
        state = {
            "task": "till_5_tile_with_hoe",
            "toolbar_information": "\n".join(
                [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 1: Hoe (quantity: 1)",
                    "Currently selected item: slot_index 1: Hoe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Hoe",
                "inventory": [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 1: Hoe (quantity: 1)",
                ],
                "surroundings": "\n".join(
                    [
                        "[-1, 0]: HoeDirt",
                        "[0, 1]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertTrue(validated.action.startswith('use(direction="'))
        self.assertEqual(validated.reason, "noop_move_grounded_override")

    def test_grounded_big_brain_use_suggestion_is_not_overridden_by_little_brain_move(self) -> None:
        suggestion = {"action": 'use(direction="left")'}
        decision = VLLMDecision(
            action='move(x=2, y=0)',
            reason="parse_fallback_no_suggestion_local_recovery",
            escalate=False,
        )
        state = {
            "task": "till_5_tile_with_hoe",
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "gathered_info": {
                "selected_item_name": "Hoe",
                "surroundings": "\n".join(
                    [
                        "[-1, 0]: Dirt",
                        "[1, 0]: empty",
                        "[0, 1]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'use(direction="left")')
        self.assertTrue(validated.reason.startswith("follow_big_brain_non_move"))

    def test_ungrounded_big_brain_use_suggestion_can_still_fall_back_to_move(self) -> None:
        suggestion = {"action": 'use(direction="right")'}
        decision = VLLMDecision(
            action='move(x=1, y=0)',
            reason="parse_fallback_no_suggestion_local_recovery",
            escalate=False,
        )
        state = {
            "task": "go_to_bus_stop",
            "toolbar_information": "Currently selected item: slot_index 3: Pickaxe",
            "gathered_info": {
                "selected_item_name": "Pickaxe",
                "surroundings": "\n".join(
                    [
                        "[1, 0]: empty",
                        "[0, 1]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=1, y=0)')

    def test_repeated_refused_action_reverts_to_different_big_brain_suggestion(self) -> None:
        suggestion = {"action": 'move(x=-1, y=0)'}
        decision = VLLMDecision(action='move(x=-2, y=1)', reason="retry", escalate=False)
        state = {
            "last_exec_info": {
                "refusal_type": "axis_circuit_breaker",
                "refused_action": 'move(x=-2, y=1)',
                "errors_info": (
                    "CIRCUIT-BREAKER: action `move(x=-2, y=1)` previously produced explicit failure 3 times in a row. "
                    "This action is REFUSED for this step."
                ),
            },
            "gathered_info": {
                "surroundings": "[1, 0]: empty",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=-1, y=0)')
        self.assertEqual(validated.reason, "avoid_refused_action:follow_suggestion")

    def test_repeated_refused_action_uses_local_recovery_when_suggestion_is_same(self) -> None:
        suggestion = {"action": 'move(x=-2, y=1)'}
        decision = VLLMDecision(action='move(x=-2, y=1)', reason="retry", escalate=False)
        state = {
            "last_exec_info": {
                "refusal_type": "axis_circuit_breaker",
                "refused_action": 'move(x=-2, y=1)',
                "errors_info": (
                    "AXIS-CIRCUIT-BREAKER: 3 consecutive blocked move() calls toward -y. "
                    "The path is blocked in this direction."
                ),
            },
            "gathered_info": {
                "surroundings": "[1, 0]: empty",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertNotEqual(validated.action, 'move(x=-2, y=1)')
        self.assertEqual(validated.reason, "avoid_refused_action:local_recovery")

    def test_blocked_route_recovery_avoids_last_refused_move(self) -> None:
        recovered = VLLMClient._build_structure_blocked_move_recovery(
            action_text='move(x=-5, y=0)',
            game_state={
                "task": "go_to_bus_stop",
                "last_exec_info": {
                    "refusal_type": "same_action_circuit_breaker",
                    "refused_action": 'move(x=-1, y=-3)',
                    "errors_info": (
                        "CIRCUIT-BREAKER: action `move(x=-1, y=-3)` previously produced explicit failure 3 times in a row. "
                        "This action is REFUSED for this step."
                    ),
                },
                "gathered_info": {
                    "surroundings": "\n".join(
                        [
                            "[-1, 0]: Farmhouse",
                            "[0, -1]: Farmhouse",
                            "[0, 1]: Farmhouse",
                            "[1, 0]: Farmhouse",
                            "[-1, -3]: empty",
                            "[-3, 1]: empty",
                            "[-2, 1]: empty",
                            "[-3, 0]: empty",
                            "[-2, -1]: empty",
                        ]
                    ),
                },
            },
            blocker="Farmhouse",
        )

        self.assertNotEqual(recovered, 'move(x=-1, y=-3)')
        self.assertEqual(recovered, 'move(x=-3, y=1)')

    def test_navigation_blocked_route_fallback_prefers_unit_step(self) -> None:
        from stardojo.utils.cortex_runtime_utils import validate_runtime_pre_execution_action

        state = {
            "task": "go_to_bus_stop",
            "last_exec_info": {
                "refusal_type": "same_action_circuit_breaker",
                "refused_action": 'move(x=-1, y=-3)',
                "errors_info": (
                    "CIRCUIT-BREAKER: action `move(x=-1, y=-3)` previously produced explicit failure 3 times in a row. "
                    "This action is REFUSED for this step."
                ),
            },
            "gathered_info": {
                "surroundings": "\n".join(
                    [
                        "[-1, 0]: Farmhouse",
                        "[0, -1]: Farmhouse",
                        "[0, 1]: Farmhouse",
                        "[1, 0]: Farmhouse",
                        "[-1, -3]: empty",
                        "[-3, 1]: empty",
                        "[-2, 1]: empty",
                        "[-3, 0]: empty",
                        "[-2, -1]: empty",
                    ]
                ),
            },
        }

        validated = validate_runtime_pre_execution_action(
            state=state,
            action_text='move(x=-5, y=0)',
        )

        self.assertFalse(validated["is_valid"])
        self.assertEqual(validated["invalid_reason"], "runtime_validation:navigation_blocked_route")
        self.assertEqual(validated["fallback_action"], 'move(x=-1, y=1)')
        self.assertTrue(
            validate_runtime_pre_execution_action(
                state=state,
                action_text=validated["fallback_action"],
            )["is_valid"]
        )

    def test_hay_scythe_forage_move_rewrites_to_adjacent_use(self) -> None:
        from stardojo.utils.cortex_runtime_utils import validate_runtime_pre_execution_action

        state = {
            "task": "forage_10_hay_with_scythe_17",
            "prompt_profile": "navigation",
            "toolbar_information": "\n".join(
                [
                    "slot_index 4: Scythe (quantity: 1)",
                    "Currently selected item: slot_index 4: Scythe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Scythe",
                "current_menu": {"type": "No Menu"},
                "inventory": ["slot_index 4: Scythe (quantity: 1)"],
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Weeds",
                        "[0, 1]: empty",
                    ]
                ),
            },
        }

        validated = validate_runtime_pre_execution_action(
            state=state,
            action_text='move(x=1, y=-4)',
        )

        self.assertFalse(validated["is_valid"])
        self.assertEqual(
            validated["invalid_reason"],
            "runtime_validation:hay_scythe_adjacent_target_requires_use",
        )
        self.assertEqual(validated["fallback_action"], 'use(direction="right")')
        self.assertTrue(
            validate_runtime_pre_execution_action(
                state=state,
                action_text=validated["fallback_action"],
            )["is_valid"]
        )

    def test_hay_scythe_forage_preserves_big_brain_use_over_move_override(self) -> None:
        suggestion = {"action": 'use(direction="right")', "reason": "cut nearby grass"}
        decision = VLLMDecision(action='move(x=1, y=-4)', reason="continue navigation", escalate=False)
        state = {
            "task": "forage_10_hay_with_scythe_17",
            "toolbar_information": "\n".join(
                [
                    "slot_index 4: Scythe (quantity: 1)",
                    "Currently selected item: slot_index 4: Scythe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Scythe",
                "inventory": ["slot_index 4: Scythe (quantity: 1)"],
                "surroundings": "[1, 0]: Weeds",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'use(direction="right")')
        self.assertEqual(validated.reason, "follow_big_brain_hay_scythe_use")

    def test_hay_scythe_forage_allows_move_when_suggested_use_target_is_not_grass(self) -> None:
        suggestion = {"action": 'use(direction="right")', "reason": "cut nearby grass"}
        decision = VLLMDecision(action='move(x=0, y=1)', reason="continue navigation", escalate=False)
        state = {
            "task": "forage_10_hay_with_scythe_17",
            "toolbar_information": "\n".join(
                [
                    "slot_index 4: Scythe (quantity: 1)",
                    "Currently selected item: slot_index 4: Scythe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Scythe",
                "inventory": ["slot_index 4: Scythe (quantity: 1)"],
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Stone",
                        "[0, 1]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=0, y=1)')
        self.assertEqual(validated.reason, "continue navigation")

    def test_animal_product_harvest_interact_bypasses_crop_grounding_gate(self) -> None:
        from stardojo.utils.cortex_runtime_utils import validate_cultivation_pre_execution_action

        for task_name in (
            "harvest_1_milk_with_milk_pail",
            "harvest_3_egg",
            "harvest_1_duck_egg",
            "harvest_1_wool_with_shears",
        ):
            with self.subTest(task_name=task_name):
                validated = validate_cultivation_pre_execution_action(
                    state={"task": task_name, "task_description": task_name},
                    action_text='interact(direction="up")',
                )
                self.assertTrue(validated["is_valid"])

    def test_crop_harvest_with_egg_substring_still_requires_grounded_target(self) -> None:
        from stardojo.utils.cortex_runtime_utils import validate_cultivation_pre_execution_action

        validated = validate_cultivation_pre_execution_action(
            state={"task": "harvest_1_eggplant", "task_description": "harvest 1 eggplant"},
            action_text='interact(direction="up")',
        )

        self.assertFalse(validated["is_valid"])
        self.assertEqual(
            validated["invalid_reason"],
            "cultivation_validation:harvest_requires_grounded_target",
        )

    def test_crop_harvest_grounding_blocker_rewrites_to_stand_tile_move(self) -> None:
        from stardojo.utils.cortex_runtime_utils import validate_runtime_pre_execution_action

        state = {
            "task": "harvest_5_parsnip",
            "target_item": "Parsnip",
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "surroundings": "\n".join(
                    [
                        "[-2, -1]: Kale (ready to harvest), HoeDirt",
                        "[-2, 0]: Parsnip (ready to harvest), HoeDirt",
                        "[-1, -1]: Parsnip (ready to harvest), HoeDirt",
                        "[-1, 0]: HoeDirt",
                        "[0, -1]: HoeDirt",
                        "[0, 0]: Parsnip (ready to harvest), HoeDirt",
                        "[0, 1]: HoeDirt",
                        "[1, -1]: Parsnip (ready to harvest), HoeDirt",
                        "[1, 0]: HoeDirt",
                        "[2, -1]: Parsnip (ready to harvest), HoeDirt",
                        "[2, 0]: Parsnip (ready to harvest), HoeDirt",
                    ]
                ),
            },
        }

        validated = validate_runtime_pre_execution_action(
            state=state,
            action_text='interact(direction="down")',
        )

        self.assertFalse(validated["is_valid"])
        self.assertEqual(
            validated["invalid_reason"],
            "cultivation_validation:harvest_requires_grounded_target",
        )
        self.assertEqual(validated["fallback_action"], "move(x=-1, y=0)")
        self.assertTrue(
            validate_runtime_pre_execution_action(
                state=state,
                action_text=validated["fallback_action"],
            )["is_valid"]
        )

    def test_direct_craft_rewrite_stops_after_nonprogress_streak(self) -> None:
        from stardojo.utils.cortex_runtime_utils import validate_runtime_pre_execution_action

        validated = validate_runtime_pre_execution_action(
            state={
                "task": "craft_1_scarecrow",
                "task_description": "craft_1_scarecrow",
                "prompt_profile": "crafting",
                "last_action": 'craft(item="Scarecrow")',
                "zero_progress_streak": 1,
                "repeated_action_streak": 2,
                "gathered_info": {
                    "current_menu": {"type": "No Menu"},
                    "inventory": [
                        "slot_index 6: Wood (quantity: 50)",
                        "slot_index 7: Coal (quantity: 1)",
                        "slot_index 8: Fiber (quantity: 20)",
                    ],
                },
            },
            action_text='menu(option="open", menu_name="crafting")',
        )

        self.assertTrue(validated["is_valid"])
        self.assertEqual(validated.get("invalid_reason", ""), "")

    def test_oversized_move_remains_clamped_after_override_guard(self) -> None:
        suggestion = {"action": 'move(x=15, y=0)', "reason": "travel"}
        little_brain = LittleBrain(
            vllm_client=_StubVLLMClient(
                VLLMDecision(action='move(x=15, y=0)', reason="travel", escalate=False)
            ),
            execute_internally=False,
            max_relative_move=10,
        )
        little_brain.load_plan([suggestion], "ctx", "task")

        result = little_brain.execute(
            {
                "current_step": 0,
                "suggestions": [suggestion],
                "context_summary": "ctx",
                "task": "task",
                "previous_actions": [],
            }
        )

        self.assertEqual(result["planned_actions"], ['move(x=10, y=0)'])

    def test_diagonal_split_requires_explicit_blocker_evidence(self) -> None:
        suggestion = {"action": 'move(x=3, y=-2)'}
        decision = VLLMDecision(action='move(x=0, y=-2)', reason="split", escalate=False)

        blocked_state = {"gathered_info": {"surroundings": "[1, 0]: Stone"}}
        clear_state = {"gathered_info": {"surroundings": ""}}

        blocked_validated = VLLMClient._validate_decision_against_state(decision, suggestion, blocked_state)
        clear_validated = VLLMClient._validate_decision_against_state(decision, suggestion, clear_state)

        self.assertEqual(blocked_validated.action, 'move(x=0, y=-2)')
        self.assertEqual(clear_validated.action, 'move(x=3, y=-2)')

    def test_placeable_item_override_does_not_reselect_when_already_selected(self) -> None:
        suggestion = {"action": 'interact(direction="right")'}
        decision = VLLMDecision(action="choose_item(slot_index=0)", reason="equip", escalate=False)
        state = {
            "toolbar_information": "Currently selected item: slot_index 5: Basic Retaining Soil",
            "gathered_info": {
                "surroundings": "[1, 0]: HoeDirt",
                "inventory": [],
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertEqual(validated.action, 'interact(direction="right")')
        self.assertEqual(validated.reason, "placeable_item_already_selected")

    def test_farm_ops_choose_item_override_is_not_reverted_to_blocked_big_brain_move(self) -> None:
        suggestion = {"action": "move(x=4, y=0)"}
        decision = VLLMDecision(action="choose_item(slot_index=5)", reason="equip hay", escalate=False)
        state = {
            "task": "fill_1_feeding_bench_with_hay",
            "toolbar_information": "\n".join(
                [
                    "slot_index 4: Scythe (quantity: 1)",
                    "slot_index 5: Hay (quantity: 12)",
                    "Currently selected item: slot_index 4: Scythe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Scythe",
                "inventory": [
                    "slot_index 4: Scythe (quantity: 1)",
                    "slot_index 5: Hay (quantity: 12)",
                ],
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Farmhouse",
                        "[0, 1]: empty",
                        "[0, 2]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, "choose_item(slot_index=5)")
        self.assertEqual(validated.reason, "farm_ops_inventory_setup_fix")

    def test_cultivation_choose_item_override_is_not_reverted_to_big_brain_move(self) -> None:
        suggestion = {"action": "move(x=3, y=-4)"}
        decision = VLLMDecision(action="choose_item(slot_index=5)", reason="equip seeds", escalate=False)
        state = {
            "task": "sow_5_dirt_with_cauliflower_seeds",
            "toolbar_information": "\n".join(
                [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 1: Hoe (quantity: 1)",
                    "slot_index 5: Cauliflower Seeds (quantity: 5)",
                    "Currently selected item: slot_index 0: Axe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Axe",
                "inventory": [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 1: Hoe (quantity: 1)",
                    "slot_index 5: Cauliflower Seeds (quantity: 5)",
                ],
                "surroundings": "\n".join(
                    [
                        "[0, 1]: empty",
                        "[0, 2]: HoeDirt",
                        "[1, 0]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, "choose_item(slot_index=5)")
        self.assertEqual(validated.reason, "inventory_setup_before_move_fix")

    def test_clear_tool_choose_item_override_is_not_reverted_to_big_brain_move(self) -> None:
        suggestion = {"action": "move(x=-1, y=1)"}
        decision = VLLMDecision(action="choose_item(slot_index=4)", reason="equip scythe", escalate=False)
        state = {
            "task": "clear_10_weeds_with_scythe",
            "subtask_description": "Switch to the Scythe in the toolbar before moving to the nearby weeds.",
            "toolbar_information": "\n".join(
                [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 2: Pickaxe (quantity: 1)",
                    "slot_index 4: Scythe (quantity: 1)",
                    "Currently selected item: slot_index 0: Axe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Axe",
                "inventory": [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 2: Pickaxe (quantity: 1)",
                    "slot_index 4: Scythe (quantity: 1)",
                ],
                "surroundings": "\n".join(
                    [
                        "[-1, 1]: Weeds",
                        "[0, 1]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, "choose_item(slot_index=4)")
        self.assertEqual(validated.reason, "tool_setup_before_move_fix")

    def test_clear_tool_choose_item_override_rejects_wrong_tool_before_big_brain_move(self) -> None:
        suggestion = {"action": "move(x=-1, y=1)"}
        decision = VLLMDecision(action="choose_item(slot_index=2)", reason="equip pickaxe", escalate=False)
        state = {
            "task": "clear_10_weeds_with_scythe",
            "subtask_description": "Switch to the Scythe in the toolbar before moving to the nearby weeds.",
            "toolbar_information": "\n".join(
                [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 2: Pickaxe (quantity: 1)",
                    "slot_index 4: Scythe (quantity: 1)",
                    "Currently selected item: slot_index 0: Axe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Axe",
                "inventory": [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 2: Pickaxe (quantity: 1)",
                    "slot_index 4: Scythe (quantity: 1)",
                ],
                "surroundings": "\n".join(
                    [
                        "[-1, 1]: Weeds",
                        "[0, 1]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, "move(x=-1, y=1)")
        self.assertTrue(validated.reason.startswith("follow_big_brain_move"))

    def test_choose_item_unknown_slot_escalates_when_slot_is_not_visible(self) -> None:
        suggestion = {"action": 'choose_item(slot_index=12)'}
        decision = VLLMDecision(action='choose_item(slot_index=12)', reason="equip seeds", escalate=False)
        state = {
            "task": "sow_1_dirt_with_potato_seeds",
            "toolbar_information": "\n".join(
                [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 1: Hoe (quantity: 1)",
                    "Currently selected item: slot_index 1: Hoe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Hoe",
                "inventory": [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 1: Hoe (quantity: 1)",
                ],
                "surroundings": "[1, 0]: HoeDirt",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertTrue(validated.escalate)
        self.assertEqual(validated.reason, "choose_item_unknown_slot")

    def test_choose_item_accepts_visible_slot_from_toolbar_list_shape(self) -> None:
        suggestion = {"action": 'choose_item(slot_index=4)'}
        decision = VLLMDecision(action='choose_item(slot_index=4)', reason="equip scythe", escalate=False)
        state = {
            "task": "clear_10_weeds_with_scythe",
            "toolbar_information": [
                "slot_index 0: Axe (quantity: 1)",
                "slot_index 4: Scythe (quantity: 1)",
                "Currently selected item: slot_index 0: Axe",
            ],
            "gathered_info": {
                "selected_item_name": "Axe",
                "inventory": [
                    "slot_index 0: Axe (quantity: 1)",
                ],
                "surroundings": "[0, -1]: Weeds",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, "choose_item(slot_index=4)")
        self.assertNotEqual(validated.reason, "choose_item_unknown_slot")

    def test_choose_item_allows_toolbar_slot_when_current_slot_facts_are_partial(self) -> None:
        suggestion = {"action": 'choose_item(slot_index=4)'}
        decision = VLLMDecision(action='choose_item(slot_index=4)', reason="equip scythe", escalate=False)
        state = {
            "task": "clear_10_weeds_with_scythe",
            "gathered_info": {
                "selected_item_name": "Axe",
                "inventory": [
                    "slot_index 0: Axe (quantity: 1)",
                ],
                "surroundings": "[0, -1]: Weeds",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, "choose_item(slot_index=4)")
        self.assertNotEqual(validated.reason, "choose_item_unknown_slot")

    def test_choose_item_empty_slot_uses_route_recovery_when_shop_route_is_grounded(self) -> None:
        suggestion = {"action": 'choose_item(slot_index=5)'}
        decision = VLLMDecision(action='choose_item(slot_index=5)', reason="equip fertilizer", escalate=False)
        state = {
            "task": "fertilize_1_dirt_with_speed_gro",
            "target_item": "Speed-Gro",
            "source_type": "inventory_or_shop",
            "source_detail": "Check inventory first; if Speed-Gro is missing, route to Pierre's General Store.",
            "location": "Farm",
            "toolbar_information": "\n".join(
                [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 1: Hoe (quantity: 1)",
                    "slot_index 5: No item",
                    "Currently selected item: slot_index 0: Axe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Axe",
                "inventory": [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 1: Hoe (quantity: 1)",
                    "slot_index 5: No item",
                ],
                "surroundings": "\n".join(
                    [
                        "[1, 0]: empty",
                        "[2, 0]: empty",
                    ]
                ),
                "exits": "Bus Stop (3 tiles right, relative offset: x=3, y=0)",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, "move(x=1, y=0)")
        self.assertEqual(validated.reason, "choose_item_empty_slot_route_recovery")

    def test_decide_recovers_locally_when_no_suggestion_response_is_truncated(self) -> None:
        client = VLLMClient(api_key="dummy")

        class _FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Reasoning:\n"
                                    "1. The current task is fill_1_feeding_bench_with_hay.\n"
                                    "2. Hay is already selected and the feeding bench is not visible yet.\n"
                                    "3. I should move toward the coop interior"
                                )
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 120,
                        "total_tokens": 220,
                    },
                }

        state = {
            "task": "fill_1_feeding_bench_with_hay",
            "toolbar_information": "\n".join(
                [
                    "slot_index 4: Scythe (quantity: 1)",
                    "slot_index 5: Hay (quantity: 12)",
                    "Currently selected item: slot_index 5: Hay",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Hay",
                "inventory": [
                    "slot_index 4: Scythe (quantity: 1)",
                    "slot_index 5: Hay (quantity: 12)",
                ],
                "surroundings": "\n".join(
                    [
                        "[0, 1]: Farmhouse",
                        "[1, 0]: empty",
                        "[1, 1]: empty",
                    ]
                ),
            },
        }

        with mock.patch("cradle.runner.vllm_client.requests.post", return_value=_FakeResponse()):
            with mock.patch(
                "cradle.runner.vllm_client.acquire_llm_endpoint_slot",
                side_effect=lambda **kwargs: contextlib.nullcontext(),
            ):
                with mock.patch("cradle.runner.vllm_client.increment_llm_call_counter", return_value=None):
                    decision = client.decide(
                        context_summary="ctx",
                        suggestion={"action": "", "reason": "big_brain_unavailable_decide_freely"},
                        execution_log=[],
                        mem0_reference="",
                        step=0,
                        total_steps=1,
                        skill_list="move(x, y)",
                        game_state=state,
                    )

        self.assertFalse(decision.escalate)
        self.assertTrue(decision.action.startswith("move("))
        self.assertEqual(decision.reason, "parse_fallback_no_suggestion_local_recovery")

    def test_decide_no_suggestion_uses_tilling_autonomous_recovery(self) -> None:
        client = VLLMClient(api_key="dummy")

        class _FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Reasoning:\n"
                                    "1. The porch is blocked.\n"
                                    "2. I should look for soil elsewhere."
                                )
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 80,
                        "total_tokens": 180,
                    },
                }

        state = {
            "task": "till_5_tile_with_hoe",
            "task_description": "till_5_tile_with_hoe",
            "subtask_description": "The current subtask is select the Hoe and till 5 tile.",
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "gathered_info": {
                "selected_item_name": "Hoe",
                "surroundings": "\n".join(
                    [
                        "[0, 3]: empty",
                        "[1, 0]: Farmhouse",
                    ]
                ),
            },
        }

        with mock.patch("cradle.runner.vllm_client.requests.post", return_value=_FakeResponse()):
            with mock.patch(
                "cradle.runner.vllm_client.acquire_llm_endpoint_slot",
                side_effect=lambda **kwargs: contextlib.nullcontext(),
            ):
                with mock.patch("cradle.runner.vllm_client.increment_llm_call_counter", return_value=None):
                    decision = client.decide(
                        context_summary="ctx",
                        suggestion={"action": "", "reason": "big_brain_unavailable_decide_freely"},
                        execution_log=[],
                        mem0_reference="",
                        step=0,
                        total_steps=1,
                        skill_list="move(x, y)",
                        game_state=state,
                    )

        self.assertFalse(decision.escalate)
        self.assertEqual(decision.action, 'move(x=0, y=3)')
        self.assertEqual(decision.reason, "parse_fallback_no_suggestion_local_recovery")

    def test_decide_no_suggestion_uses_watering_autonomous_recovery(self) -> None:
        client = VLLMClient(api_key="dummy")

        class _FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Reasoning:\n"
                                    "1. The current tile is blocked.\n"
                                    "2. I should move toward the crop."
                                )
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 80,
                        "total_tokens": 180,
                    },
                }

        state = {
            "task": "water_5_crop_with_watering_can",
            "task_description": "water_5_crop_with_watering_can",
            "subtask_description": "The current subtask is select the Watering Can and water 5 crop.",
            "toolbar_information": "Currently selected item: slot_index 2: Watering Can",
            "gathered_info": {
                "selected_item_name": "Watering Can",
                "surroundings": "\n".join(
                    [
                        "[0, 3]: Parsnip Seeds (growing), HoeDirt",
                        "[1, 0]: Farmhouse",
                    ]
                ),
            },
        }

        with mock.patch("cradle.runner.vllm_client.requests.post", return_value=_FakeResponse()):
            with mock.patch(
                "cradle.runner.vllm_client.acquire_llm_endpoint_slot",
                side_effect=lambda **kwargs: contextlib.nullcontext(),
            ):
                with mock.patch("cradle.runner.vllm_client.increment_llm_call_counter", return_value=None):
                    decision = client.decide(
                        context_summary="ctx",
                        suggestion={"action": "", "reason": "big_brain_unavailable_decide_freely"},
                        execution_log=[],
                        mem0_reference="",
                        step=0,
                        total_steps=1,
                        skill_list="move(x, y)",
                        game_state=state,
                    )

        self.assertFalse(decision.escalate)
        self.assertEqual(decision.action, 'move(x=0, y=3)')
        self.assertEqual(decision.reason, "parse_fallback_no_suggestion_local_recovery")

    def test_decide_no_suggestion_does_not_use_crop_watering_recovery_for_pet_bowl_task(self) -> None:
        client = VLLMClient(api_key="dummy")

        class _FakeResponse:
            status_code = 200

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    "Reasoning:\n"
                                    "1. The current tile is blocked.\n"
                                    "2. I should move toward the crop."
                                )
                            }
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 80,
                        "total_tokens": 180,
                    },
                }

        state = {
            "task": "fill_1_pet_bowl_with_watering_can",
            "task_description": "fill_1_pet_bowl_with_watering_can",
            "subtask_description": "The current subtask is select the Watering Can and fill 1 pet bowl.",
            "toolbar_information": "Currently selected item: slot_index 2: Watering Can",
            "gathered_info": {
                "selected_item_name": "Watering Can",
                "surroundings": "\n".join(
                    [
                        "[0, 3]: Parsnip Seeds (growing), HoeDirt",
                        "[1, 0]: Farmhouse",
                    ]
                ),
            },
        }

        with mock.patch("cradle.runner.vllm_client.requests.post", return_value=_FakeResponse()):
            with mock.patch(
                "cradle.runner.vllm_client.acquire_llm_endpoint_slot",
                side_effect=lambda **kwargs: contextlib.nullcontext(),
            ):
                with mock.patch("cradle.runner.vllm_client.increment_llm_call_counter", return_value=None):
                    decision = client.decide(
                        context_summary="ctx",
                        suggestion={"action": "", "reason": "big_brain_unavailable_decide_freely"},
                        execution_log=[],
                        mem0_reference="",
                        step=0,
                        total_steps=1,
                        skill_list="move(x, y)",
                        game_state=state,
                    )

        self.assertFalse(decision.escalate)
        self.assertEqual(decision.action, "move(x=1, y=-2)")
        self.assertEqual(decision.reason, "parse_fallback_no_suggestion_local_recovery")

    def test_noop_move_with_placeable_item_rewrites_to_interact(self) -> None:
        suggestion = {"action": 'move(x=0, y=0)'}
        decision = VLLMDecision(action='move(x=0, y=0)', reason="stay_put", escalate=False)
        state = {
            "toolbar_information": "Currently selected item: slot_index 5: Basic Retaining Soil",
            "gathered_info": {
                "selected_item_name": "Basic Retaining Soil",
                "surroundings": "\n".join(
                    [
                        "[0, -1]: HoeDirt",
                        "[1, 0]: Grass",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'interact(direction="up")')
        self.assertEqual(validated.reason, "noop_move_placeable_fix")

    def test_seed_use_rewrites_to_interact_when_hoedirt_is_valid(self) -> None:
        suggestion = {"action": 'use(direction="up")'}
        decision = VLLMDecision(action='use(direction="up")', reason="plant seed", escalate=False)
        state = {
            "task": "sow_5_dirt_with_cauliflower_seeds",
            "toolbar_information": "Currently selected item: slot_index 6: Cauliflower Seeds",
            "gathered_info": {
                "selected_item_name": "Cauliflower Seeds",
                "surroundings": "\n".join(
                    [
                        "[0, -1]: HoeDirt",
                        "[1, 0]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'interact(direction="up")')
        self.assertEqual(validated.reason, "placeable_item_use_fix")

    def test_hay_clear_recovery_prefers_grass_over_weeds(self) -> None:
        action = VLLMClient._build_local_clear_recovery_action(
            game_state={
                "task": "forage_10_hay_with_scythe",
                "toolbar_information": "Currently selected item: slot_index 4: Scythe",
                "gathered_info": {
                    "selected_item_name": "Scythe",
                    "surroundings": "\n".join(
                        [
                            "[0, -1]: Weeds",
                            "[1, 0]: Grass",
                        ]
                    ),
                },
            },
            selected_item_name="Scythe",
            inventory=[],
            toolbar_information="Currently selected item: slot_index 4: Scythe",
        )

        self.assertEqual(action, 'use(direction="right")')

    def test_hay_clear_recovery_can_use_weeds_when_no_grass_is_visible(self) -> None:
        action = VLLMClient._build_local_clear_recovery_action(
            game_state={
                "task": "forage_10_hay_with_scythe",
                "toolbar_information": "Currently selected item: slot_index 4: Scythe",
                "gathered_info": {
                    "selected_item_name": "Scythe",
                    "surroundings": "\n".join(
                        [
                            "[0, -1]: Weeds",
                            "[1, 0]: empty",
                        ]
                    ),
                },
            },
            selected_item_name="Scythe",
            inventory=[],
            toolbar_information="Currently selected item: slot_index 4: Scythe",
        )

        self.assertEqual(action, 'use(direction="up")')

    def test_noop_move_without_grounded_fallback_escalates(self) -> None:
        suggestion = {"action": 'move(x=0, y=0)'}
        decision = VLLMDecision(action='move(x=0, y=0)', reason="stay_put", escalate=False)
        state = {
            "toolbar_information": "Currently selected item: slot_index 0: Axe",
            "gathered_info": {
                "selected_item_name": "Axe",
                "surroundings": "[0, -1]: Farmhouse",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertTrue(validated.escalate)
        self.assertEqual(validated.reason, "noop_move")

    def test_blocked_recovery_finds_tool_from_top_level_toolbar_information(self) -> None:
        state = {
            "toolbar_information": "\n".join(
                [
                    "Currently selected item: slot_index 3: Pickaxe",
                    "slot_index 3: Pickaxe",
                    "slot_index 4: Scythe",
                ]
            ),
            "gathered_info": {
                "facing_direction": "down",
                "surroundings": "[0, 1]: Weeds",
            },
        }
        execution_log = [
            {
                "action": "move(x=0, y=1)",
                "success": False,
                "errors_info": "path is likely blocked by an obstacle",
            }
        ]

        context = VLLMClient._build_front_obstacle_context(state, execution_log)

        self.assertTrue(context["blocked_override_action"].startswith("move("))
        self.assertIn("Prefer a grounded reroute", context["blocked_recovery_hint"])

    def test_fertilizer_interact_keeps_seeded_hoedirt_when_tile_is_unfertilized(self) -> None:
        suggestion = {"action": 'interact(direction="right")'}
        decision = VLLMDecision(action='interact(direction="right")', reason="fertilize", escalate=False)
        state = {
            "toolbar_information": "\n".join(
                [
                    "Currently selected item: slot_index 5: Basic Retaining Soil",
                    "slot_index 5: Basic Retaining Soil",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Basic Retaining Soil",
                "position": [65, 19],
                "surroundings": "\n".join(
                    [
                        "[1, 0]: HoeDirt",
                        "[0, 1]: HoeDirt",
                        "[-1, 0]: Grass",
                    ]
                ),
                "crops": "Parsnip Seeds (growing) at (66, 19)",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'interact(direction="right")')

    def test_fertilizer_interact_rewrites_away_from_explicitly_fertilized_tile(self) -> None:
        suggestion = {"action": 'interact(direction="right")'}
        decision = VLLMDecision(action='interact(direction="right")', reason="fertilize", escalate=False)
        state = {
            "toolbar_information": "\n".join(
                [
                    "Currently selected item: slot_index 5: Basic Retaining Soil",
                    "slot_index 5: Basic Retaining Soil",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Basic Retaining Soil",
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Parsnip Seeds (growing), Basic Retaining Soil, HoeDirt",
                        "[0, 1]: HoeDirt",
                        "[-1, 0]: Grass",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'interact(direction="down")')
        self.assertEqual(validated.reason, "placeable_item_interact_target_fix")

    def test_invalid_explicitly_fertilized_interact_escalates_after_zero_progress(self) -> None:
        suggestion = {"action": 'interact(direction="right")'}
        decision = VLLMDecision(action='interact(direction="right")', reason="fertilize", escalate=False)
        state = {
            "zero_progress_streak": 2,
            "toolbar_information": "\n".join(
                [
                    "Currently selected item: slot_index 5: Basic Retaining Soil",
                    "slot_index 5: Basic Retaining Soil",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Basic Retaining Soil",
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Parsnip Seeds (growing), Basic Retaining Soil, HoeDirt",
                        "[0, 1]: HoeDirt",
                        "[-1, 0]: Grass",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertTrue(validated.escalate)
        self.assertEqual(validated.reason, "placeable_item_invalid_target")

    def test_repeated_no_confirmation_fertilizer_interact_reroutes_to_another_tile(self) -> None:
        suggestion = {"action": 'interact(direction="right")'}
        decision = VLLMDecision(action='interact(direction="right")', reason="fertilize", escalate=False)
        state = {
            "zero_progress_streak": 1,
            "task_progress_quantity": 0,
            "last_action": 'interact(direction="right")',
            "last_exec_info": {
                "errors_info": "interact() returned no confirmation; action may not have taken effect.",
            },
            "toolbar_information": "\n".join(
                [
                    "Currently selected item: slot_index 5: Basic Retaining Soil",
                    "slot_index 5: Basic Retaining Soil",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Basic Retaining Soil",
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Parsnip Seeds (growing), HoeDirt",
                        "[0, 1]: HoeDirt",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'interact(direction="down")')
        self.assertEqual(validated.reason, "fertilize_target_effective_recovery")

    def test_parse_invalidated_clear_suggestion_recovers_to_grounded_local_use(self) -> None:
        suggestion = {"action": 'use(direction="down")'}
        decision = VLLMDecision(
            action='use(direction="down")',
            reason="parse_fallback_invalidated_suggestion",
            escalate=False,
        )
        state = {
            "task": "clear_10_weeds_with_scythe",
            "toolbar_information": "Currently selected item: slot_index 4: Scythe",
            "gathered_info": {
                "selected_item_name": "Scythe",
                "surroundings": "\n".join(
                    [
                        "[0, 1]: empty",
                        "[1, 0]: Weeds",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'use(direction="right")')
        self.assertEqual(validated.reason, "parse_fallback_invalidated_local_recovery")

    def test_clearable_tool_use_on_empty_target_escalates(self) -> None:
        suggestion = {"action": 'use(direction="up")'}
        decision = VLLMDecision(action='use(direction="up")', reason="mine", escalate=False)
        state = {
            "toolbar_information": "Currently selected item: slot_index 3: Pickaxe",
            "gathered_info": {
                "selected_item_name": "Pickaxe",
                "surroundings": "\n".join(
                    [
                        "[0, -1]: empty",
                        "[1, -1]: Stone",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertTrue(validated.escalate)
        self.assertEqual(validated.reason, "pickaxe_invalid_target")

    def test_invalid_tool_use_override_does_not_revert_to_invalid_suggestion(self) -> None:
        suggestion = {"action": 'use(direction="right")'}
        decision = VLLMDecision(action='use(direction="down")', reason="clear weeds", escalate=False)
        state = {
            "toolbar_information": "Currently selected item: slot_index 4: Scythe",
            "gathered_info": {
                "selected_item_name": "Scythe",
                "surroundings": "\n".join(
                    [
                        "[1, 0]: empty",
                        "[0, 1]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertTrue(validated.escalate)
        self.assertEqual(validated.reason, "scythe_invalid_target")

    def test_invalid_hoe_override_does_not_revert_to_invalid_suggestion(self) -> None:
        suggestion = {"action": 'use(direction="right")'}
        decision = VLLMDecision(action='use(direction="down")', reason="till", escalate=False)
        state = {
            "zero_progress_streak": 1,
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "gathered_info": {
                "selected_item_name": "Hoe",
                "surroundings": "\n".join(
                    [
                        "[1, 0]: empty",
                        "[0, 1]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.reason, "hoe_empty_target_recovery")
        self.assertNotEqual(validated.action, suggestion["action"])

    def test_supported_inventory_menu_stays_available(self) -> None:
        suggestion = {"action": "choose_item(slot_index=12)"}
        decision = VLLMDecision(
            action='menu(option="open", menu_name="inventory")',
            reason="open bag",
            escalate=False,
        )
        state = {
            "toolbar_information": "\n".join(
                [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 12: Potato Seeds (quantity: 1)",
                ]
            ),
            "gathered_info": {
                "inventory": [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 12: Potato Seeds (quantity: 1)",
                ],
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'menu(option="open", menu_name="inventory")')

    def test_choose_item_on_explicitly_empty_slot_escalates(self) -> None:
        suggestion = {"action": "choose_item(slot_index=12)"}
        decision = VLLMDecision(action="choose_item(slot_index=12)", reason="equip", escalate=False)
        state = {
            "toolbar_information": "slot_index 12: No item",
            "gathered_info": {
                "inventory": ["slot_index 12: No item"],
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertTrue(validated.escalate)
        self.assertEqual(validated.reason, "choose_item_empty_slot")

    def test_hoe_use_on_empty_target_recovers_after_zero_progress(self) -> None:
        suggestion = {"action": 'use(direction="right")'}
        decision = VLLMDecision(action='use(direction="right")', reason="till", escalate=False)
        state = {
            "zero_progress_streak": 2,
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "gathered_info": {
                "selected_item_name": "Hoe",
                "surroundings": "[1, 0]: empty",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.reason, "hoe_empty_target_recovery")
        self.assertNotEqual(validated.action, suggestion["action"])

    def test_hoe_use_on_empty_target_is_left_alone_without_failure_evidence(self) -> None:
        suggestion = {"action": 'use(direction="right")'}
        decision = VLLMDecision(action='use(direction="right")', reason="till", escalate=False)
        state = {
            "zero_progress_streak": 0,
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "gathered_info": {
                "selected_item_name": "Hoe",
                "surroundings": "[1, 0]: empty",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'use(direction="right")')

    def test_hoe_use_on_empty_target_in_till_context_reroutes_after_repeated_failure(self) -> None:
        suggestion = {"action": 'use(direction="right")'}
        decision = VLLMDecision(action='use(direction="right")', reason="till", escalate=False)
        state = {
            "zero_progress_streak": 2,
            "repeated_action_streak": 2,
            "task_description": "till_5_tile_with_hoe",
            "subtask_description": "The current subtask is select the Hoe and till one tile.",
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "last_exec_info": {
                "errors_info": "use() returned no confirmation; action may not have taken effect.",
            },
            "gathered_info": {
                "selected_item_name": "Hoe",
                "surroundings": "\n".join(
                    [
                        "[1, 0]: empty",
                        "[0, 1]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertNotEqual(validated.action, 'use(direction="right")')
        self.assertEqual(validated.reason, "hoe_empty_target_recovery")

    def test_till_choose_item_override_reverts_to_local_recovery_with_hoe_selected(self) -> None:
        suggestion = {"action": 'use(direction="down")'}
        decision = VLLMDecision(action="choose_item(slot_index=3)", reason="clear stone", escalate=False)
        state = {
            "task_description": "till_5_tile_with_hoe",
            "subtask_description": "The current subtask is select the Hoe and till one tile.",
            "toolbar_information": "\n".join(
                [
                    "Currently selected item: slot_index 1: Hoe",
                    "slot_index 1: Hoe",
                    "slot_index 3: Pickaxe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Hoe",
                "inventory": [
                    "slot_index 1: Hoe",
                    "slot_index 3: Pickaxe",
                ],
                "surroundings": "\n".join(
                    [
                        "[0, 1]: Stone",
                        "[1, 0]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertNotEqual(validated.action, "choose_item(slot_index=3)")
        self.assertIn(validated.action, {'use(direction="right")', 'move(x=1, y=0)'})

    def test_watering_choose_item_override_reverts_to_local_crop_recovery(self) -> None:
        suggestion = {"action": 'use(direction="down")'}
        decision = VLLMDecision(action="choose_item(slot_index=3)", reason="clear stone", escalate=False)
        state = {
            "task_description": "water_5_crop_with_watering_can",
            "subtask_description": "The current subtask is select the Watering Can and water 5 crop.",
            "toolbar_information": "\n".join(
                [
                    "Currently selected item: slot_index 2: Watering Can",
                    "slot_index 2: Watering Can",
                    "slot_index 3: Pickaxe",
                ]
            ),
            "gathered_info": {
                "selected_item_name": "Watering Can",
                "inventory": [
                    "slot_index 2: Watering Can",
                    "slot_index 3: Pickaxe",
                ],
                "surroundings": "\n".join(
                    [
                        "[0, 1]: Stone",
                        "[1, 0]: Parsnip Seeds (growing), HoeDirt",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'use(direction="right")')

    def test_non_crop_watering_can_use_is_not_rewritten_to_visible_crop(self) -> None:
        suggestion = {"action": 'use(direction="down")'}
        decision = VLLMDecision(action='use(direction="down")', reason="fill bowl", escalate=False)
        state = {
            "task": "fill_1_pet_bowl_with_watering_can",
            "task_description": "fill_1_pet_bowl_with_watering_can",
            "subtask_description": "Use the Watering Can to fill the pet bowl.",
            "toolbar_information": "Currently selected item: slot_index 2: Watering Can",
            "gathered_info": {
                "selected_item_name": "Watering Can",
                "surroundings": "\n".join(
                    [
                        "[0, 1]: Pet Bowl",
                        "[1, 0]: Parsnip Seeds (growing), HoeDirt",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'use(direction="down")')

    def test_find_alternative_watering_direction_ignores_visible_crops_outside_crop_context(self) -> None:
        state = {
            "task": "fill_1_pet_bowl_with_watering_can",
            "task_description": "fill_1_pet_bowl_with_watering_can",
            "subtask_description": "Use the Watering Can to fill the pet bowl.",
            "gathered_info": {
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Parsnip Seeds (growing), HoeDirt",
                        "[0, 1]: Pet Bowl",
                    ]
                ),
            },
        }

        self.assertEqual(
            VLLMClient._find_alternative_tool_use_direction(
                game_state=state,
                tool_name="Watering Can",
                invalid_direction="",
            ),
            "",
        )

    def test_runtime_watering_recovery_ignores_pet_bowl_tasks(self) -> None:
        from stardojo.utils.cortex_runtime_utils import _build_local_watering_recovery_action

        state = {
            "task": "fill_1_pet_bowl_with_watering_can",
            "task_description": "fill_1_pet_bowl_with_watering_can",
            "subtask_description": "Use the Watering Can to fill the pet bowl.",
            "toolbar_information": "Currently selected item: slot_index 2: Watering Can",
            "gathered_info": {
                "surroundings": "[0, 3]: Parsnip Seeds (growing), HoeDirt",
            },
        }

        self.assertEqual(
            _build_local_watering_recovery_action(
                result_state=state,
                vllm_cls=VLLMClient,
                selected_item_name="Watering Can",
                inventory=[],
                toolbar_information=state["toolbar_information"],
            ),
            "",
        )

    def test_move_into_explicit_chest_escalates(self) -> None:
        suggestion = {"action": 'move(x=0, y=1)'}
        decision = VLLMDecision(action='move(x=0, y=1)', reason="approach", escalate=False)
        state = {
            "gathered_info": {
                "surroundings": "[0, 1]: Chest",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertTrue(validated.escalate)
        self.assertEqual(validated.reason, "move_target_blocked")

    def test_move_into_adjacent_door_rewrites_to_interact_only_when_front_tile_matches(self) -> None:
        suggestion = {"action": 'move(x=0, y=1)'}
        decision = VLLMDecision(action='move(x=0, y=1)', reason="enter", escalate=False)
        state = {
            "gathered_info": {
                "surroundings": "[0, 1]: Door",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.reason, "adjacent_blocker_interact_fix")
        self.assertEqual(validated.action, 'interact(direction="down")')

    def test_move_into_adjacent_farmhouse_door_rewrites_to_interact_for_navigation_task(self) -> None:
        suggestion = {"action": 'move(x=0, y=1)'}
        decision = VLLMDecision(action='move(x=0, y=1)', reason="enter home", escalate=False)
        state = {
            "task": "return_home_and_sleep",
            "subtask_description": "The current subtask is enter the farmhouse and go to bed.",
            "gathered_info": {
                "surroundings": "[0, 1]: Farmhouse Door",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.reason, "enter home")
        self.assertEqual(validated.action, 'move(x=0, y=1)')

    def test_go_to_bus_stop_blocked_door_move_does_not_rewrite_to_interact(self) -> None:
        suggestion = {"action": 'move(x=0, y=-3)'}
        decision = VLLMDecision(action='move(x=0, y=-3)', reason="advance", escalate=False)
        state = {
            "task": "go_to_bus_stop",
            "subtask_description": "The current subtask is move toward the east exit of the farm to reach the Bus Stop.",
            "gathered_info": {
                "surroundings": "\n".join(
                    [
                        "[0, -1]: Farmhouse Door, exit: Farmhouse Entrance",
                        "[1, 0]: empty",
                        "[-1, 0]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertNotEqual(validated.reason, "adjacent_blocker_interact_fix")
        self.assertNotEqual(validated.action, 'interact(direction="up")')
        self.assertTrue(validated.action.startswith("move("))

    def test_shipping_task_blocked_door_move_with_parsnip_selected_does_not_rewrite_to_interact(self) -> None:
        suggestion = {"action": 'move(x=0, y=-1)'}
        decision = VLLMDecision(action='move(x=0, y=-1)', reason="exit house", escalate=False)
        state = {
            "task": "ship_1_parsnip_with_shipping_bin",
            "subtask_description": "The current subtask is move right and up to reach the Shipping Bin located near the silo and coop.",
            "toolbar_information": "Currently selected item: slot_index 6: Parsnip",
            "gathered_info": {
                "selected_item_name": "Parsnip",
                "surroundings": "\n".join(
                    [
                        "[0, -1]: Farmhouse Door, exit: Farmhouse Entrance",
                        "[1, 0]: empty",
                        "[-1, 0]: empty",
                    ]
                ),
                "current_menu": {"type": "No Menu"},
            },
            "current_menu": {"type": "No Menu"},
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertNotEqual(validated.reason, "adjacent_blocker_interact_fix")
        self.assertNotEqual(validated.action, 'interact(direction="up")')
        self.assertTrue(validated.action.startswith("move("))

    def test_outdoor_cultivation_task_does_not_auto_interact_with_adjacent_farmhouse_door(self) -> None:
        suggestion = {"action": 'move(x=0, y=-1)'}
        decision = VLLMDecision(action='move(x=0, y=-1)', reason="step off porch", escalate=False)
        state = {
            "task": "till_5_tile_with_hoe",
            "task_description": "till_5_tile_with_hoe",
            "subtask_description": "The current subtask is move away from the farmhouse and reach an open till patch outside.",
            "location": "Farm",
            "gathered_info": {
                "location": "Farm",
                "surroundings": "\n".join(
                    [
                        "[0, -1]: Farmhouse Door, exit: Farmhouse Entrance",
                        "[0, 1]: empty",
                        "[1, 0]: Farmhouse",
                        "[-1, 0]: Farmhouse",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertNotEqual(validated.reason, "adjacent_blocker_interact_fix")
        self.assertNotEqual(validated.action, 'interact(direction="up")')
        self.assertTrue(validated.action.startswith("move("))

    def test_move_into_building_does_not_auto_interact_without_explicit_actionable_front_tile(self) -> None:
        suggestion = {"action": 'move(x=0, y=1)'}
        decision = VLLMDecision(action='move(x=0, y=1)', reason="enter", escalate=False)
        state = {
            "gathered_info": {
                "surroundings": "\n".join(
                    [
                        "[0, 1]: Farmhouse",
                        "[1, 0]: empty",
                        "[-1, 0]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertNotEqual(validated.reason, "adjacent_blocker_interact_fix")
        self.assertNotEqual(validated.action, 'interact(direction="down")')

    def test_cultivation_same_axis_progress_move_is_kept_near_farmhouse(self) -> None:
        suggestion = {"action": 'move(x=0, y=2)'}
        decision = VLLMDecision(action='move(x=0, y=2)', reason="reposition", escalate=False)
        state = {
            "task": "till_5_tile_with_hoe",
            "gathered_info": {
                "surroundings": "\n".join(
                    [
                        "[0, 1]: Farmhouse",
                        "[0, 2]: empty",
                        "[0, 3]: empty",
                        "[1, 2]: empty",
                        "[-1, 2]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.reason, "blocked_structure_reroute")
        self.assertEqual(validated.action, 'move(x=1, y=2)')

    def test_menu_alias_for_crafting_is_normalized_and_allowed(self) -> None:
        suggestion = {"action": 'menu(option="open", menu_name="crafting")'}
        decision = VLLMDecision(
            action='menu(option="crafting", menu_name="crafting")',
            reason="open crafting",
            escalate=False,
        )
        state = {"gathered_info": {"current_menu": ""}}

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'menu(option="crafting", menu_name="crafting")')

    def test_move_into_farmhouse_reroutes_to_grounded_escape_move(self) -> None:
        suggestion = {"action": 'move(x=0, y=1)'}
        decision = VLLMDecision(action='move(x=0, y=1)', reason="approach", escalate=False)
        state = {
            "toolbar_information": "Currently selected item: slot_index 4: Scythe",
            "gathered_info": {
                "selected_item_name": "Scythe",
                "surroundings": "\n".join(
                    [
                        "[0, 1]: Farmhouse",
                        "[0, 2]: empty",
                        "[-1, 2]: empty",
                        "[-3, 3]: Weeds",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.reason, "blocked_structure_reroute")
        self.assertEqual(validated.action, 'move(x=-1, y=2)')

    def test_blocked_rightward_farmhouse_move_prefers_progressive_escape_not_upward_wall_bump(self) -> None:
        recovered = VLLMClient._build_structure_blocked_move_recovery(
            action_text="move(x=2, y=0)",
            game_state={
                "toolbar_information": "Currently selected item: slot_index 1: Hoe",
                "gathered_info": {
                    "selected_item_name": "Hoe",
                    "surroundings": "\n".join(
                        [
                            "[-3, -3]: empty",
                            "[-3, -2]: empty",
                            "[-3, -1]: empty",
                            "[-3, 0]: empty",
                            "[-3, 1]: empty",
                            "[-3, 2]: empty",
                            "[-3, 3]: Weeds",
                            "[-2, -3]: empty",
                            "[-2, -2]: empty",
                            "[-2, -1]: empty",
                            "[-2, 0]: empty",
                            "[-2, 1]: empty",
                            "[-2, 2]: empty",
                            "[-2, 3]: empty",
                            "[-1, -3]: empty",
                            "[-1, -2]: empty",
                            "[-1, -1]: Farmhouse",
                            "[-1, 0]: Farmhouse",
                            "[-1, 1]: Farmhouse",
                            "[-1, 2]: empty",
                            "[-1, 3]: empty",
                            "[0, -3]: empty",
                            "[0, -2]: empty",
                            "[0, -1]: Farmhouse",
                            "[0, 0]: Farmhouse",
                            "[0, 1]: Farmhouse",
                            "[0, 2]: empty",
                            "[0, 3]: empty",
                            "[1, -3]: empty",
                            "[1, -2]: empty",
                            "[1, -1]: Farmhouse",
                            "[1, 0]: Farmhouse",
                            "[1, 1]: Farmhouse",
                            "[1, 2]: empty",
                            "[1, 3]: empty",
                            "[2, -3]: empty",
                            "[2, -2]: empty",
                            "[2, -1]: empty",
                            "[2, 0]: empty",
                            "[2, 1]: empty",
                            "[2, 2]: empty",
                            "[2, 3]: empty",
                            "[3, -3]: empty",
                            "[3, -2]: empty",
                            "[3, -1]: empty",
                            "[3, 0]: empty",
                            "[3, 1]: empty",
                            "[3, 2]: empty",
                            "[3, 3]: empty",
                        ]
                    ),
                },
            },
            blocker="Farmhouse",
        )

        self.assertNotEqual(recovered, "move(x=0, y=-2)")
        self.assertTrue(recovered.startswith("move("))
        self.assertIn("x=1", recovered)

    def test_harvest_ready_crop_blocked_reroute_does_not_move_farther_from_crop_patch(self) -> None:
        state = {
            "task_description": "harvest_5_parsnip",
            "target_item": "Parsnip",
            "toolbar_information": "Currently selected item: slot_index 0: Axe",
            "gathered_info": {
                "selected_item_name": "Axe",
                "surroundings": "\n".join(
                    [
                        "[-3, -3]: empty",
                        "[-3, -2]: empty",
                        "[-3, -1]: empty",
                        "[-3, 0]: empty",
                        "[-3, 1]: empty",
                        "[-3, 2]: empty",
                        "[-3, 3]: Parsnip (ready to harvest), HoeDirt",
                        "[-2, -3]: empty",
                        "[-2, -2]: empty",
                        "[-2, -1]: empty",
                        "[-2, 0]: empty",
                        "[-2, 1]: empty",
                        "[-2, 2]: empty",
                        "[-2, 3]: Parsnip (ready to harvest), HoeDirt",
                        "[-1, -3]: empty",
                        "[-1, -2]: empty",
                        "[-1, -1]: Farmhouse",
                        "[-1, 0]: Farmhouse",
                        "[-1, 1]: Farmhouse",
                        "[-1, 2]: empty",
                        "[-1, 3]: Parsnip (ready to harvest), HoeDirt",
                        "[0, -3]: empty",
                        "[0, -2]: empty",
                        "[0, -1]: Farmhouse Door, exit: Farmhouse Entrance",
                        "[0, 0]: Farmhouse",
                        "[0, 1]: Farmhouse",
                        "[0, 2]: empty",
                        "[0, 3]: Parsnip (ready to harvest), HoeDirt",
                        "[1, -3]: empty",
                        "[1, -2]: empty",
                        "[1, -1]: Farmhouse",
                        "[1, 0]: Farmhouse",
                        "[1, 1]: Farmhouse",
                        "[1, 2]: empty",
                        "[1, 3]: Parsnip (ready to harvest), HoeDirt",
                        "[2, -3]: empty",
                        "[2, -2]: empty",
                        "[2, -1]: empty",
                        "[2, 0]: empty",
                        "[2, 1]: empty",
                        "[2, 2]: empty",
                        "[2, 3]: Parsnip (ready to harvest), HoeDirt",
                        "[3, -3]: empty",
                        "[3, -2]: empty",
                        "[3, -1]: empty",
                        "[3, 0]: empty",
                        "[3, 1]: empty",
                        "[3, 2]: empty",
                        "[3, 3]: Parsnip (ready to harvest), HoeDirt",
                    ]
                ),
            },
        }

        ready_cells = VLLMClient._nearby_ready_harvest_crop_cells(state)
        current_distance = VLLMClient._min_manhattan_distance_to_cells((0, 0), ready_cells)
        recovered = VLLMClient._build_structure_blocked_move_recovery(
            action_text="move(x=2, y=0)",
            game_state=state,
            blocker="Farmhouse",
        )

        self.assertNotEqual(recovered, "move(x=1, y=-3)")
        self.assertNotEqual(recovered, "")
        move = VLLMClient._parse_move_components(recovered)
        self.assertIsNotNone(move)
        candidate_distance = VLLMClient._min_manhattan_distance_to_cells(move, ready_cells)
        self.assertIsNotNone(current_distance)
        self.assertIsNotNone(candidate_distance)
        self.assertLessEqual(candidate_distance, current_distance)

    def test_blocked_move_prefers_route_aware_recovery_for_go_to_coop(self) -> None:
        suggestion = {"action": 'move(x=0, y=-2)'}
        decision = VLLMDecision(action='move(x=0, y=-2)', reason="approach", escalate=False)
        state = {
            "task": "go_to_coop",
            "target_item": "Coop",
            "source_type": "farm_building",
            "toolbar_information": "Currently selected item: slot_index 0: Axe",
            "gathered_info": {
                "selected_item_name": "Axe",
                "buildings": "Coop (door: 4 tiles right, relative offset: x=4, y=0)",
                "surroundings": "\n".join(
                    [
                        "[0, -1]: Farmhouse",
                        "[1, 0]: empty",
                        "[2, 0]: empty",
                        "[3, 0]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.reason, "blocked_route_recovery")
        self.assertEqual(validated.action, "move(x=4, y=0)")

    def test_grounded_waypoint_override_is_kept_when_bigbrain_points_opposite_direction(self) -> None:
        suggestion = {"action": 'move(x=4, y=0)'}
        decision = VLLMDecision(action='move(x=-2, y=0)', reason="local correction", escalate=False)
        state = {
            "task": "pet_3_animal",
            "source_type": "animal_housing",
            "gathered_info": {
                "selected_item_name": "Axe",
                "buildings": "Deluxe Barn (door: 15 tiles left, relative offset: x=-15, y=0)",
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Farmhouse",
                        "[-1, 0]: empty",
                        "[-2, 0]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=-2, y=0)')
        self.assertEqual(validated.reason, "local correction")

    def test_navigation_diagonal_bypass_is_kept_when_bigbrain_pushes_through_house(self) -> None:
        suggestion = {"action": 'move(x=10, y=0)'}
        decision = VLLMDecision(action='move(x=2, y=-2)', reason="house bypass", escalate=False)
        state = {
            "task": "go_to_bus_stop",
            "subtask_description": "The current subtask is route across the farm toward the east exit that leads to the Bus Stop.",
            "gathered_info": {
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Farmhouse",
                        "[2, 0]: empty",
                        "[0, -2]: empty",
                        "[2, -2]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=2, y=-2)')
        self.assertEqual(validated.reason, "house bypass")

    def test_tilling_structure_escape_move_override_is_kept_over_blocked_farmhouse_push(self) -> None:
        suggestion = {"action": 'move(x=-1, y=0)'}
        decision = VLLMDecision(action='move(x=2, y=0)', reason="escape porch footprint", escalate=False)
        state = {
            "task": "till_5_tile_with_hoe",
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "gathered_info": {
                "selected_item_name": "Hoe",
                "surroundings": "\n".join(
                    [
                        "[-1, 0]: Farmhouse",
                        "[2, 0]: empty",
                        "[0, -1]: Farmhouse",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=2, y=0)')
        self.assertEqual(validated.reason, "escape porch footprint")

    def test_tilling_grounded_empty_hoe_use_override_is_kept_over_move(self) -> None:
        suggestion = {"action": 'move(x=0, y=1)'}
        decision = VLLMDecision(action='use(direction="down")', reason="till now", escalate=False)
        state = {
            "task": "till_5_tile_with_hoe",
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "gathered_info": {
                "selected_item_name": "Hoe",
                "surroundings": "\n".join(
                    [
                        "[0, 1]: empty",
                        "[1, 1]: empty",
                        "[-1, 1]: empty",
                        "[0, 0]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'use(direction="down")')
        self.assertEqual(validated.reason, "till now")

    def test_backwoods_pet_bowl_waypoint_override_is_kept_over_blocked_north_push(self) -> None:
        suggestion = {"action": 'move(x=0, y=-4)'}
        decision = VLLMDecision(action='move(x=-1, y=0)', reason="sidestep to entrance", escalate=False)
        state = {
            "task": "go_to_backwoods",
            "target_item": "Backwoods",
            "gathered_info": {
                "exits": "Pet Bowl Entrance (relative offset: x=-1, y=-2)",
                "surroundings": "\n".join(
                    [
                        "[0, -1]: Pet Bowl",
                        "[-1, 0]: empty",
                        "[-1, -1]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=-1, y=0)')
        self.assertEqual(validated.reason, "sidestep to entrance")

    def test_navigation_anchor_override_keeps_local_farmhouse_exit_move(self) -> None:
        suggestion = {"action": 'move(x=4, y=-4)'}
        decision = VLLMDecision(action='move(x=0, y=-1)', reason="local exit move", escalate=False)
        state = {
            "task": "go_to_coop",
            "subtask_description": "The current subtask is route to the coop entrance and enter it.",
            "current_menu": {"type": "No Menu"},
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "surroundings": "\n".join(
                    [
                        "[0, -1]: Farmhouse Entrance",
                        "[0, 1]: Farmhouse",
                        "[1, 0]: Farmhouse",
                        "[-1, 0]: Farmhouse",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=0, y=-1)')
        self.assertEqual(validated.reason, "local exit move")

    def test_clear_task_move_is_rewritten_to_immediate_use_when_adjacent_stone_exists(self) -> None:
        suggestion = {"action": 'move(x=1, y=0)'}
        decision = VLLMDecision(action='move(x=1, y=0)', reason="approach stone", escalate=False)
        state = {
            "task": "clear_5_stone_with_pickaxe",
            "toolbar_information": "Currently selected item: slot_index 3: Pickaxe",
            "gathered_info": {
                "selected_item_name": "Pickaxe",
                "surroundings": "[1, 0]: Stone",
                "inventory": ["slot_index 3: Pickaxe"],
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'use(direction="right")')
        self.assertEqual(validated.reason, "immediate_local_progress_fix")

    def test_clear_weeds_task_ignores_adjacent_stone_and_uses_visible_weeds(self) -> None:
        suggestion = {"action": 'move(x=1, y=0)'}
        decision = VLLMDecision(action='move(x=1, y=0)', reason="approach debris", escalate=False)
        state = {
            "task": "clear_10_weeds_with_scythe",
            "toolbar_information": "Currently selected item: slot_index 4: Scythe",
            "gathered_info": {
                "selected_item_name": "Scythe",
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Stone",
                        "[0, 1]: Weeds",
                    ]
                ),
                "inventory": ["slot_index 4: Scythe"],
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertIn(validated.action, {'use(direction="down")', 'use(direction="right")'})
        self.assertEqual(validated.reason, "immediate_local_progress_fix")

    def test_craft_with_missing_materials_recovers_to_grounded_route(self) -> None:
        suggestion = {"action": 'menu(option="close", menu_name="current_menu")'}
        decision = VLLMDecision(action='craft(item="Cherry Bomb")', reason="force craft", escalate=False)
        state = {
            "task": "craft_1_cherry_bomb",
            "subtask_description": "The current subtask is go to the Mines to collect Copper Ore and Coal needed for crafting a cherry bomb.",
            "gathered_info": {
                "location": "Farm",
                "inventory": [
                    "slot_index 0: Axe",
                    "slot_index 1: Hoe",
                    "slot_index 2: Watering Can",
                    "slot_index 3: Pickaxe",
                    "slot_index 4: Scythe",
                    "slot_index 5: Rusty Sword",
                ],
                "toolbar_information": "Currently selected item: slot_index 0: Axe",
                "exits": "Backwoods (relative offset: x=-3, y=0)",
                "surroundings": "[1, 0]: Farmhouse",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=-1, y=0)')
        self.assertEqual(validated.reason, "craft_missing_materials_recovery")

    def test_till_task_move_is_preserved_when_adjacent_ground_exists(self) -> None:
        suggestion = {"action": 'move(x=0, y=1)'}
        decision = VLLMDecision(action='move(x=0, y=1)', reason="approach tile", escalate=False)
        state = {
            "task": "till_5_tile_with_hoe",
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "subtask_description": "The current subtask is till a nearby empty tile.",
            "gathered_info": {
                "selected_item_name": "Hoe",
                "surroundings": "\n".join(
                    [
                        "[0, 1]: empty",
                        "[1, 0]: empty",
                        "[1, 1]: empty",
                        "[-1, 1]: empty",
                    ]
                ),
                "inventory": ["slot_index 1: Hoe"],
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=0, y=1)')
        self.assertNotEqual(validated.reason, "immediate_local_progress_fix")

    def test_till_task_move_is_not_rewritten_to_use_when_stuck_on_same_tile(self) -> None:
        suggestion = {"action": 'move(x=1, y=0)'}
        decision = VLLMDecision(action='move(x=1, y=0)', reason="approach tile", escalate=False)
        state = {
            "task": "till_5_tile_with_hoe",
            "zero_progress_streak": 1,
            "repeated_action_streak": 3,
            "last_action": 'use(direction="right")',
            "last_exec_info": {
                "errors_info": "use() returned no confirmation; action may not have taken effect.",
            },
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "subtask_description": "The current subtask is till a nearby empty tile.",
            "gathered_info": {
                "selected_item_name": "Hoe",
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Sandy ground",
                        "[0, 1]: Sandy ground",
                        "[-1, 0]: Sandy ground",
                    ]
                ),
                "inventory": ["slot_index 1: Hoe"],
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertTrue(validated.action.startswith("move("))
        self.assertNotEqual(validated.reason, "immediate_local_progress_fix")

    def test_till_task_move_is_not_rewritten_to_use_after_recent_nonprogress_hoe_swing(self) -> None:
        suggestion = {"action": 'move(x=-1, y=-1)'}
        decision = VLLMDecision(action='move(x=-1, y=-1)', reason="reposition to next patch", escalate=False)
        state = {
            "task": "till_5_tile_with_hoe",
            "last_action": 'use(direction="up")',
            "task_progress_quantity": 3,
            "previous_task_progress_quantity": 3,
            "task_progress_delta": 0,
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "subtask_description": "The current subtask is use the Hoe on the next open soil patch.",
            "gathered_info": {
                "selected_item_name": "Hoe",
                "surroundings": "\n".join(
                    [
                        "[0, -1]: empty",
                        "[-1, -1]: empty",
                        "[1, 0]: Weeds",
                        "[0, 1]: HoeDirt",
                    ]
                ),
                "inventory": ["slot_index 1: Hoe"],
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=-1, y=-1)')
        self.assertNotEqual(validated.reason, "immediate_local_progress_fix")

    def test_hoe_use_on_empty_target_next_to_mailbox_recovers_to_safer_patch(self) -> None:
        suggestion = {"action": 'use(direction="left")'}
        decision = VLLMDecision(action='use(direction="left")', reason="till", escalate=False)
        state = {
            "task": "till_5_tile_with_hoe",
            "task_description": "till_5_tile_with_hoe",
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "gathered_info": {
                "selected_item_name": "Hoe",
                "inventory": ["slot_index 1: Hoe"],
                "surroundings": "\n".join(
                    [
                        "[-1, 0]: empty",
                        "[-1, 1]: Mailbox",
                        "[0, 1]: empty",
                        "[1, 0]: empty",
                        "[1, 1]: empty",
                        "[2, 0]: empty",
                        "[2, 1]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertNotEqual(validated.action, 'use(direction="left")')
        self.assertEqual(validated.reason, "hoe_empty_target_recovery")

    def test_blocked_tilling_move_prefers_grounded_till_recovery_over_house_reroute(self) -> None:
        suggestion = {"action": 'move(x=-3, y=0)'}
        decision = VLLMDecision(action='move(x=-3, y=0)', reason="route to open patch", escalate=False)
        state = {
            "task": "till_5_tile_with_hoe",
            "task_description": "till_5_tile_with_hoe",
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "gathered_info": {
                "selected_item_name": "Hoe",
                "inventory": ["slot_index 1: Hoe"],
                "surroundings": "\n".join(
                    [
                        "[-1, 0]: Farmhouse",
                        "[-1, -1]: Farmhouse",
                        "[0, -1]: Farmhouse Door, exit: Farmhouse Entrance",
                        "[1, 0]: empty",
                        "[1, 1]: empty",
                        "[2, 0]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.reason, "blocked_tilling_recovery")
        self.assertNotEqual(validated.action, 'move(x=-3, y=0)')
        self.assertNotEqual(validated.action, 'move(x=-1, y=-3)')
        self.assertIn(validated.action, {'use(direction="right")', 'move(x=1, y=0)', 'move(x=2, y=0)'})

    def test_crafting_menu_open_with_missing_stone_is_rewritten_to_grounded_material_recovery(self) -> None:
        suggestion = {"action": 'use(direction="down")'}
        decision = VLLMDecision(action='menu(option="open", menu_name="crafting")', reason="open menu", escalate=False)
        state = {
            "task": "craft_1_basic_retaining_soil",
            "target_item": "Basic Retaining Soil",
            "subtask_description": "The current subtask is mine the Stone located directly in front of the player using the Pickaxe.",
            "toolbar_information": "Currently selected item: slot_index 3: Pickaxe",
            "gathered_info": {
                "selected_item_name": "Pickaxe",
                "surroundings": "[0, 1]: Stone",
                "inventory": ["slot_index 3: Pickaxe"],
                "current_menu": {"type": "No Menu"},
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'use(direction="down")')
        self.assertEqual(validated.reason, "craft_missing_materials_recovery")

    def test_animal_housing_short_search_override_is_not_reverted_to_blocked_big_brain_move(self) -> None:
        suggestion = {"action": 'move(x=4, y=0)'}
        decision = VLLMDecision(action='move(x=-2, y=0)', reason="short open-ground search", escalate=False)
        state = {
            "task": "pet_3_animal",
            "target_item": "Animal",
            "source_type": "animal_housing",
            "gathered_info": {
                "surroundings": "\n".join(
                    [
                        "[0, -1]: Farmhouse Door, exit: Farmhouse Entrance",
                        "[1, 0]: Farmhouse",
                        "[2, 0]: empty",
                        "[-1, 0]: empty",
                        "[-2, 0]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, "move(x=-2, y=0)")
        self.assertEqual(validated.reason, "short open-ground search")

    def test_parse_invalidated_till_suggestion_uses_multi_tile_move_toward_visible_ground(self) -> None:
        suggestion = {"action": 'use(direction="down")'}
        decision = VLLMDecision(
            action='use(direction="down")',
            reason="parse_fallback_invalidated_suggestion",
            escalate=False,
        )
        state = {
            "task": "till_5_tile_with_hoe",
            "task_description": "till_5_tile_with_hoe",
            "subtask_description": "The current subtask is select the Hoe and till 5 tile.",
            "toolbar_information": "Currently selected item: slot_index 1: Hoe",
            "gathered_info": {
                "selected_item_name": "Hoe",
                "surroundings": "\n".join(
                    [
                        "[0, 3]: empty",
                        "[1, 0]: Farmhouse",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=0, y=3)')

    def test_parse_invalidated_watering_suggestion_uses_multi_tile_move_toward_visible_crop(self) -> None:
        suggestion = {"action": 'use(direction="down")'}
        decision = VLLMDecision(
            action='use(direction="down")',
            reason="parse_fallback_invalidated_suggestion",
            escalate=False,
        )
        state = {
            "task": "water_5_crop_with_watering_can",
            "task_description": "water_5_crop_with_watering_can",
            "subtask_description": "The current subtask is select the Watering Can and water 5 crop.",
            "toolbar_information": "Currently selected item: slot_index 2: Watering Can",
            "gathered_info": {
                "selected_item_name": "Watering Can",
                "surroundings": "\n".join(
                    [
                        "[0, 3]: Parsnip Seeds (growing), HoeDirt",
                        "[1, 0]: Farmhouse",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=0, y=3)')

    def test_little_brain_keeps_state_aware_blocked_structure_reroute(self) -> None:
        suggestion = {"action": 'move(x=0, y=2)', "reason": "advance"}
        little_brain = LittleBrain(
            vllm_client=_StubVLLMClient(
                VLLMDecision(
                    action='move(x=-1, y=3)',
                    reason="blocked_structure_reroute",
                    escalate=False,
                )
            ),
            execute_internally=False,
        )
        little_brain.load_plan([suggestion], "ctx", "task")

        result = little_brain.execute(
            {
                "current_step": 0,
                "suggestions": [suggestion],
                "context_summary": "ctx",
                "task": "task",
                "previous_actions": [],
                "zero_progress_streak": 0,
            }
        )

        self.assertEqual(result["planned_actions"], ['move(x=-1, y=3)'])

    def test_little_brain_keeps_blocked_recovery_override(self) -> None:
        suggestion = {"action": 'move(x=0, y=-2)', "reason": "advance"}
        little_brain = LittleBrain(
            vllm_client=_StubVLLMClient(
                VLLMDecision(
                    action='move(x=-2, y=0)',
                    reason="blocked_recovery",
                    escalate=False,
                )
            ),
            execute_internally=False,
        )
        little_brain.load_plan([suggestion], "ctx", "task")

        result = little_brain.execute(
            {
                "current_step": 0,
                "suggestions": [suggestion],
                "context_summary": "ctx",
                "task": "task",
                "previous_actions": [],
                "zero_progress_streak": 0,
            }
        )

        self.assertEqual(result["planned_actions"], ['move(x=-2, y=0)'])

    def test_little_brain_keeps_side_step_when_big_brain_move_targets_visible_bed_blocker(self) -> None:
        suggestion = {"action": 'move(x=0, y=1)', "reason": "exit farmhouse"}
        little_brain = LittleBrain(
            vllm_client=_StubVLLMClient(
                VLLMDecision(
                    action='move(x=1, y=0)',
                    reason="bed blocker sidestep",
                    escalate=False,
                )
            ),
            execute_internally=False,
        )
        little_brain.load_plan([suggestion], "ctx", "pet_8_animal")

        result = little_brain.execute(
            {
                "current_step": 0,
                "suggestions": [suggestion],
                "context_summary": "ctx",
                "task": "pet_8_animal",
                "previous_actions": [],
                "gathered_info": {
                    "surroundings": "\n".join(
                        [
                            "[0, 1]: Bed",
                            "[1, 0]: empty",
                        ]
                    ),
                },
            }
        )

        self.assertEqual(result["planned_actions"], ['move(x=1, y=0)'])

    def test_little_brain_still_reverts_axis_swap_without_blocker_evidence(self) -> None:
        suggestion = {"action": 'move(x=0, y=1)', "reason": "advance"}
        little_brain = LittleBrain(
            vllm_client=_StubVLLMClient(
                VLLMDecision(
                    action='move(x=1, y=0)',
                    reason="side_move",
                    escalate=False,
                )
            ),
            execute_internally=False,
        )
        little_brain.load_plan([suggestion], "ctx", "task")

        result = little_brain.execute(
            {
                "current_step": 0,
                "suggestions": [suggestion],
                "context_summary": "ctx",
                "task": "task",
                "previous_actions": [],
                "gathered_info": {
                    "surroundings": "[0, 1]: empty",
                },
            }
        )

        self.assertEqual(result["planned_actions"], ['move(x=0, y=1)'])

    def test_move_into_explicit_stone_prefers_reroute(self) -> None:
        suggestion = {"action": 'move(x=-2, y=0)'}
        decision = VLLMDecision(action='move(x=-2, y=0)', reason="reposition", escalate=False)
        state = {
            "gathered_info": {
                "surroundings": "[-1, 0]: Stone",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.reason, "blocked_structure_reroute")
        self.assertTrue(validated.action.startswith("move("))
        self.assertNotEqual(validated.action, suggestion["action"])

    def test_move_into_clearable_front_obstacle_prefers_grounded_reroute(self) -> None:
        suggestion = {"action": 'move(x=0, y=1)'}
        decision = VLLMDecision(action='move(x=0, y=1)', reason="advance", escalate=False)
        state = {
            "gathered_info": {
                "surroundings": "\n".join(
                    [
                        "[0, 1]: Weeds",
                        "[1, 0]: empty",
                        "[1, 1]: empty",
                    ]
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.reason, "blocked_structure_reroute")
        self.assertEqual(validated.action, 'move(x=1, y=1)')

    def test_move_through_empty_tile_remains_allowed(self) -> None:
        suggestion = {"action": 'move(x=0, y=1)'}
        decision = VLLMDecision(action='move(x=0, y=1)', reason="advance", escalate=False)
        state = {
            "gathered_info": {
                "surroundings": "[0, 1]: empty",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=0, y=1)')

    def test_no_menu_dict_is_not_treated_as_open(self) -> None:
        self.assertFalse(VLLMClient._is_menu_open({"type": "No Menu"}))

    def test_sleep_choose_option_without_menu_rewrites_to_bed_interact(self) -> None:
        suggestion = {"action": "choose_option(option_index=0, quantity=0)"}
        decision = VLLMDecision(
            action="choose_option(option_index=0, quantity=0)",
            reason="close menu",
            escalate=False,
        )
        state = {
            "task": "go_to_bed",
            "current_menu": {"type": "No Menu"},
            "gathered_info": {
                "facing_direction": "up",
                "surroundings": "[0, -1]: Bed",
                "current_menu": {"type": "No Menu"},
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertEqual(validated.action, 'interact(direction="up")')
        self.assertEqual(validated.reason, "sleep_bed_interact_fix")

    def test_sleep_dialogue_cancel_rewrites_to_confirm(self) -> None:
        suggestion = {"action": "choose_option(option_index=0, quantity=0)"}
        decision = VLLMDecision(
            action="choose_option(option_index=0, quantity=0)",
            reason="close menu",
            escalate=False,
        )
        state = {
            "task": "go_to_bed",
            "current_menu": {
                "type": "DialogueBox",
                "dialogues": ["Go to sleep for the night?"],
                "responses": ["Yes", "No"],
            },
            "gathered_info": {
                "facing_direction": "up",
                "surroundings": "[0, -1]: Bed",
                "current_menu": {
                    "type": "DialogueBox",
                    "dialogues": ["Go to sleep for the night?"],
                    "responses": ["Yes", "No"],
                },
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertEqual(validated.action, "choose_option(option_index=1, quantity=0)")
        self.assertEqual(validated.reason, "sleep_dialogue_confirm_fix")

    def test_sleep_dialogue_interact_rewrites_to_confirm(self) -> None:
        suggestion = {"action": 'interact(direction="up")'}
        decision = VLLMDecision(
            action='interact(direction="up")',
            reason="sleep",
            escalate=False,
        )
        state = {
            "task": "go_to_bed",
            "current_menu": {
                "type": "DialogueBox",
                "dialogues": ["Go to sleep for the night?"],
                "responses": ["Yes", "No"],
            },
            "gathered_info": {
                "facing_direction": "up",
                "surroundings": "[0, -1]: Bed",
                "current_menu": {
                    "type": "DialogueBox",
                    "dialogues": ["Go to sleep for the night?"],
                    "responses": ["Yes", "No"],
                },
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertEqual(validated.action, "choose_option(option_index=1, quantity=0)")
        self.assertEqual(validated.reason, "sleep_dialogue_confirm_fix")

    def test_consumable_dialogue_rewrites_world_action_to_decline(self) -> None:
        suggestion = {"action": 'move(x=1, y=0)'}
        decision = VLLMDecision(
            action='move(x=1, y=0)',
            reason="approach incubator",
            escalate=False,
        )
        state = {
            "task": "incubate_1_chicken_with_incubator",
            "current_menu": {
                "type": "DialogueBox",
                "dialogues": ["Eat Egg?"],
                "responses": [
                    {"responseKey": "Yes", "responseText": "Yes"},
                    {"responseKey": "No", "responseText": "No"},
                ],
            },
            "gathered_info": {
                "current_menu": {
                    "type": "DialogueBox",
                    "dialogues": ["Eat Egg?"],
                    "responses": [
                        {"responseKey": "Yes", "responseText": "Yes"},
                        {"responseKey": "No", "responseText": "No"},
                    ],
                },
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, "choose_option(option_index=2, quantity=0)")
        self.assertEqual(validated.reason, "consumable_dialogue_decline_fix")

    def test_object_dialogue_rewrites_action_to_confirm(self) -> None:
        suggestion = {"action": 'move(x=1, y=0)'}
        decision = VLLMDecision(
            action='move(x=1, y=0)',
            reason="clear debris",
            escalate=False,
        )
        state = {
            "task": "clear_30_debris_with_scythe_and_pickaxe_and_axe",
            "current_menu": {
                "type": "ObjectDialogue",
                "dialogues": ["You found a 'Geode'!"],
                "message": "You found a 'Geode'!",
            },
            "gathered_info": {
                "current_menu": {
                    "type": "ObjectDialogue",
                    "dialogues": ["You found a 'Geode'!"],
                    "message": "You found a 'Geode'!",
                },
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertEqual(validated.action, "choose_option(option_index=1, quantity=0)")
        self.assertEqual(validated.reason, "object_dialogue_dismiss_fix")

    def test_sleep_move_into_bed_is_preserved_when_bed_tiles_are_walkable(self) -> None:
        suggestion = {"action": 'move(x=0, y=-1)'}
        decision = VLLMDecision(action='move(x=0, y=-1)', reason="sleep", escalate=False)
        state = {
            "task": "go_to_bed",
            "current_menu": {"type": "No Menu"},
            "gathered_info": {
                "surroundings": "[0, -1]: Bed",
                "current_menu": {"type": "No Menu"},
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertEqual(validated.action, 'move(x=0, y=-1)')
        self.assertEqual(validated.reason, "sleep")

    def test_map_page_rewrites_world_action_to_close_menu(self) -> None:
        suggestion = {"action": 'use(direction="right")'}
        decision = VLLMDecision(
            action='use(direction="right")',
            reason="mine coal",
            escalate=False,
        )
        state = {
            "task": "mine_1_coal_with_pickaxe",
            "current_menu": {"type": "No Menu"},
            "gathered_info": {
                "description": (
                    "Overall, the image shows a map of the Stardew Valley area, "
                    "specifically highlighting the StarDojo Farm region."
                ),
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'menu(option="close", menu_name="map")')
        self.assertEqual(validated.reason, "map_menu_close_fix")

    def test_shop_menu_sell_rewrites_wrong_direction_to_canonical_sell_action(self) -> None:
        suggestion = {"action": 'choose_option(option_index=1, quantity=1, direction="in")'}
        decision = VLLMDecision(
            action='choose_option(option_index=1, quantity=1, direction="in")',
            reason="sell parsnips",
            escalate=False,
        )
        state = {
            "task": "sell_5_parsnip_to_pierre",
            "target_item": "Parsnip",
            "current_menu": {
                "type": "ShopMenu",
                "shopmenudata": [
                    {"name": "Parsnip Seeds", "price": 20},
                ],
            },
            "toolbar_information": "\n".join(
                [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 6: Parsnip (quantity: 5)",
                    "Currently selected item: slot_index 6: Parsnip",
                ]
            ),
            "gathered_info": {
                "current_menu": {
                    "type": "ShopMenu",
                    "shopmenudata": [
                        {"name": "Parsnip Seeds", "price": 20},
                    ],
                },
                "selected_item_name": "Parsnip",
                "inventory": [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 6: Parsnip (quantity: 5)",
                ],
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(
            validated.action,
            'choose_option(option_index=7, quantity=5, direction="out")',
        )
        self.assertEqual(validated.reason, "shop_menu_sell_fix")

    def test_shop_menu_sell_prefers_target_slot_selection_before_sale(self) -> None:
        suggestion = {"action": 'move(x=0, y=-1)'}
        decision = VLLMDecision(
            action='move(x=0, y=-1)',
            reason="approach pierre",
            escalate=False,
        )
        state = {
            "task": "sell_1_parsnip_to_pierre",
            "target_item": "Parsnip",
            "current_menu": {
                "type": "ShopMenu",
                "shopmenudata": [
                    {"name": "Parsnip Seeds", "price": 20},
                ],
            },
            "toolbar_information": "\n".join(
                [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 6: Parsnip (quantity: 1)",
                    "Currently selected item: slot_index 0: Axe",
                ]
            ),
            "gathered_info": {
                "current_menu": {
                    "type": "ShopMenu",
                    "shopmenudata": [
                        {"name": "Parsnip Seeds", "price": 20},
                    ],
                },
                "selected_item_name": "Axe",
                "inventory": [
                    "slot_index 0: Axe (quantity: 1)",
                    "slot_index 6: Parsnip (quantity: 1)",
                ],
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, "choose_item(slot_index=6)")
        self.assertEqual(validated.reason, "shop_menu_sell_fix")

    def test_house_adjacent_route_step_helper_clamps_diagonal_move_to_unit_step(self) -> None:
        action = VLLMClient._prefer_house_adjacent_route_step(
            'move(x=-1, y=-3)',
            {
                "task": "go_to_coop",
                "gathered_info": {
                    "surroundings": "\n".join(
                        [
                            "[-1, 0]: Farmhouse",
                            "[0, 1]: Farmhouse",
                            "[1, 0]: Farmhouse",
                            "[0, -1]: empty",
                        ]
                    ),
                },
            },
        )

        self.assertEqual(action, 'move(x=0, y=-1)')

    def test_sow_autonomous_local_recovery_prefers_seed_slot_setup(self) -> None:
        action = VLLMClient._build_autonomous_local_recovery_action(
            game_state={
                "task": "sow_5_dirt_with_cauliflower_seeds",
                "target_item": "Cauliflower Seeds",
                "toolbar_information": "\n".join(
                    [
                        "slot_index 1: Hoe (quantity: 1)",
                        "slot_index 6: Cauliflower Seeds (quantity: 1)",
                        "Currently selected item: slot_index 1: Hoe",
                    ]
                ),
                "gathered_info": {
                    "selected_item_name": "Hoe",
                    "inventory": [
                        "slot_index 1: Hoe (quantity: 1)",
                        "slot_index 6: Cauliflower Seeds (quantity: 1)",
                    ],
                    "surroundings": "\n".join(
                        [
                            "[1, 0]: Cauliflower Seeds (growing), HoeDirt",
                            "[2, 0]: Cauliflower Seeds (growing), HoeDirt",
                            "[3, 0]: Cauliflower Seeds (growing), HoeDirt",
                        ]
                    ),
                },
            }
        )

        self.assertEqual(action, "choose_item(slot_index=6)")

    def test_ship_task_inside_farmhouse_short_move_is_not_rewritten_to_service_interact(self) -> None:
        from stardojo.utils.cortex_runtime_utils import validate_runtime_pre_execution_action

        validated = validate_runtime_pre_execution_action(
            state={
                "task": "ship_1_parsnip_with_shipping_bin",
                "prompt_profile": "social",
                "location": "FarmHouse",
                "gathered_info": {
                    "location": "FarmHouse",
                    "current_menu": {"type": "No Menu"},
                    "surroundings": "[0, -1]: Door, exit: Farm",
                },
            },
            action_text='move(x=1, y=0)',
        )

        self.assertTrue(validated["is_valid"])
        self.assertEqual(validated.get("invalid_reason", ""), "")

    def test_pet_bowl_watering_move_is_not_rewritten_to_interact(self) -> None:
        suggestion = {"action": 'move(x=2, y=0)'}
        decision = VLLMDecision(
            action='move(x=2, y=0)',
            reason="approach pet bowl",
            escalate=False,
        )
        state = {
            "task": "fill_1_pet_bowl_with_watering_can",
            "current_menu": {"type": "No Menu"},
            "toolbar_information": "Currently selected item: slot_index 2: Watering Can",
            "gathered_info": {
                "selected_item_name": "Watering Can",
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Pet Bowl",
                        "[2, 0]: Pet Bowl",
                    ]
                ),
                "current_menu": {"type": "No Menu"},
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertFalse(validated.escalate)
        self.assertEqual(validated.action, 'move(x=2, y=0)')
        self.assertEqual(validated.reason, "approach pet bowl")

    def test_go_to_bed_inside_coop_prefers_local_exit_recovery(self) -> None:
        action = VLLMClient._build_local_route_recovery_action(
            game_state={
                "task": "go_to_bed",
                "location": "Coop",
                "gathered_info": {
                    "location": "Coop",
                    "surroundings": "[0, 1]: Door, exit: Farm",
                },
            }
        )

        self.assertEqual(action, 'interact(direction="down")')

    def test_go_home_route_recovery_prefers_farmhouse_door_interact_when_adjacent(self) -> None:
        action = VLLMClient._build_autonomous_local_recovery_action(
            game_state={
                "task": "go_home",
                "gathered_info": {
                    "buildings": "Farmhouse (door: 1 tiles down, relative offset: x=0, y=1)",
                    "surroundings": "[0, 1]: Farmhouse Door",
                    "toolbar_information": "Currently selected item: slot_index 0: Axe",
                },
            }
        )

        self.assertEqual(action, 'interact(direction="down")')

    def test_parse_fallback_escalates_under_instability(self) -> None:
        suggestion = {"action": 'move(x=0, y=1)', "reason": "advance"}
        decision = VLLMDecision(
            action='move(x=0, y=1)',
            reason="parse_fallback: advance",
            escalate=False,
        )
        state = {
            "zero_progress_streak": 1,
            "gathered_info": {
                "surroundings": "[0, 1]: empty",
            },
        }

        validated = VLLMClient._validate_decision_against_state(decision, suggestion, state)

        self.assertTrue(validated.escalate)
        self.assertEqual(validated.reason, "parse_fallback_under_instability")

    def test_external_execution_mode_defers_execution_log_until_feedback(self) -> None:
        suggestion = {"action": 'move(x=1, y=0)', "reason": "advance"}
        little_brain = LittleBrain(
            vllm_client=_StubVLLMClient(
                VLLMDecision(action='move(x=1, y=0)', reason="advance", escalate=False)
            ),
            execute_internally=False,
        )
        little_brain.load_plan([suggestion], "ctx", "task")

        result = little_brain.execute(
            {
                "current_step": 0,
                "suggestions": [suggestion],
                "context_summary": "ctx",
                "task": "task",
                "previous_actions": [],
            }
        )

        self.assertEqual(little_brain.execution_log, [])
        self.assertEqual(result["execution_log"], [])
        self.assertEqual(result["planned_actions"], ['move(x=1, y=0)'])
        self.assertTrue(result["execution_pending"])
        self.assertEqual(result["pending_action"], 'move(x=1, y=0)')
        self.assertEqual(result["pending_step_index"], 0)
        self.assertEqual(result["pending_suggested_action"], 'move(x=1, y=0)')
        self.assertIsNone(result["success"])
        self.assertEqual(result["brain_mode"], "little")
        self.assertNotIn("escalation_reason", result)
        self.assertFalse(result["has_execution_feedback"])

    def test_record_external_execution_feedback_appends_real_failure(self) -> None:
        little_brain = LittleBrain(
            vllm_client=_StubVLLMClient(
                VLLMDecision(action='move(x=1, y=0)', reason="advance", escalate=False)
            ),
            execute_internally=False,
        )

        little_brain.record_external_execution_feedback(
            action='move(x=1, y=0)',
            success=False,
            errors_info="path is likely blocked by an obstacle",
            step=0,
            suggested_action='move(x=1, y=0)',
        )

        self.assertEqual(len(little_brain.execution_log), 1)
        self.assertEqual(little_brain.execution_log[0]["action"], 'move(x=1, y=0)')
        self.assertFalse(little_brain.execution_log[0]["success"])
        self.assertIn("blocked", little_brain.execution_log[0]["errors_info"])

    def test_record_external_execution_feedback_keeps_final_failure_metadata(self) -> None:
        little_brain = LittleBrain(
            vllm_client=_StubVLLMClient(
                VLLMDecision(action='use(direction="right")', reason="till", escalate=False)
            ),
            execute_internally=False,
        )

        little_brain.record_external_execution_feedback(
            action='use(direction="right")',
            success=False,
            errors_info="returned no confirmation",
            step=0,
            suggested_action='use(direction="right")',
            state_changed=False,
            uncertain_execution=False,
            heightened_failure_signal=True,
            progress_delta=0,
            progress_quantity=0,
        )

        self.assertEqual(len(little_brain.execution_log), 1)
        self.assertFalse(little_brain.execution_log[0]["success"])
        self.assertFalse(little_brain.execution_log[0]["state_changed"])
        self.assertFalse(little_brain.execution_log[0]["uncertain_execution"])
        self.assertTrue(little_brain.execution_log[0]["heightened_failure_signal"])
        self.assertEqual(little_brain.execution_log[0]["progress_delta"], 0)
        self.assertEqual(little_brain.execution_log[0]["progress_quantity"], 0)

    def test_little_brain_backstop_guard_reverts_axis_swap(self) -> None:
        guarded = LittleBrain._guard_opposite_move(
            action='move(x=1, y=0)',
            suggested='move(x=0, y=1)',
            reason="unguarded_side_move",
        )

        self.assertEqual(guarded, 'move(x=0, y=1)')


if __name__ == "__main__":
    unittest.main()
