from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import stardojo.utils.cortex_runtime_utils as cortex_runtime_utils
from stardojo.utils.cortex_runtime_utils import (
    _load_object_name_to_id,
    record_cortex_no_execution,
    reset_cortex_no_execution_watchdog,
    validate_cultivation_pre_execution_action,
    validate_runtime_pre_execution_action,
)


class _StubCultivationVLLM:
    @staticmethod
    def _normalize_menu_type(value: object) -> str:
        if isinstance(value, dict):
            return str(value.get("type", "") or "").strip().lower()
        return str(value or "").strip().lower()

    @staticmethod
    def _parse_directional_skill(action: object):
        action_text = str(action or "").strip()
        if action_text.startswith('use(direction="') and action_text.endswith('")'):
            return "use", action_text[len('use(direction="'):-2]
        if action_text.startswith('interact(direction="') and action_text.endswith('")'):
            return "interact", action_text[len('interact(direction="'):-2]
        return None

    @staticmethod
    def _normalize_tool_name(value: object) -> str:
        return str(value or "").strip()

    @staticmethod
    def _extract_selected_item_name_from_toolbar(toolbar_information: object) -> str:
        text = str(toolbar_information or "")
        match = re.search(r"Currently selected item:\s*slot_index\s+\d+:\s*([^(]+)", text)
        return str(match.group(1)).strip() if match else ""

    @staticmethod
    def _get_directional_target(state: object, direction: str):
        if isinstance(state, dict):
            directional_targets = state.get("directional_targets", {})
            if isinstance(directional_targets, dict) and direction in directional_targets:
                override = directional_targets.get(direction)
                if isinstance(override, (tuple, list)) and len(override) >= 2:
                    return override[0], override[1]
                return override, ""
        if str(direction or "").strip() == "none":
            return "", ""
        return "HoeDirt", "Watering Can"

    @staticmethod
    def _find_tool_slot(inventory: object, tool_name: str):
        return None

    @staticmethod
    def _find_tool_slot_in_toolbar_text(toolbar_information: object, tool_name: str):
        return 2 if tool_name in str(toolbar_information or "") else None

    @staticmethod
    def _extract_inventory_slot_map(inventory: object, toolbar_information: object):
        return {2: "Parsnip Seeds"} if "Parsnip Seeds" in str(toolbar_information or "") else {}

    @staticmethod
    def _slot_is_explicitly_empty(item_name: object) -> bool:
        return not str(item_name or "").strip()

    @staticmethod
    def _is_valid_watering_target(state: object, direction: str, target_obj: object) -> bool:
        return str(direction or "").strip() == "right"

    @staticmethod
    def _find_alternative_tool_use_direction(game_state: object, tool_name: object, invalid_direction: str) -> str:
        return "right"

    @staticmethod
    def _selected_item_is_seed(item_name: object) -> bool:
        return "seed" in str(item_name or "").strip().lower()

    @staticmethod
    def _selected_item_requires_interact(item_name: object) -> bool:
        text = str(item_name or "").strip().lower()
        return "seed" in text or "fertilizer" in text or "soil" in text

    @staticmethod
    def _is_valid_placeable_target(state: object, direction: str, item_name: object, target_obj: object) -> bool:
        return str(direction or "").strip() == "right"

    @staticmethod
    def _collect_valid_placeable_directions(game_state: object, item_name: object, invalid_direction: str):
        return [] if str(invalid_direction or "").strip() == "right" else ["right"]

    @staticmethod
    def _parse_menu_action(action_text: object):
        text = str(action_text or "").strip().lower()
        if text in {
            'menu(option="open", menu_name="crafting")',
            'menu(option="open_crafting", menu_name="crafting")',
        }:
            return "open", "crafting"
        if text == 'menu(option="close", menu_name="current_menu")':
            return "close", "current_menu"
        return None

    @staticmethod
    def _direction_to_relative(direction: object):
        return {
            "up": (0, -1),
            "down": (0, 1),
            "left": (-1, 0),
            "right": (1, 0),
        }.get(str(direction or "").strip().lower())

    @classmethod
    def _get_structured_surroundings_map(cls, state: object):
        gathered = state.get("gathered_info", {}) if isinstance(state, dict) else {}
        surroundings = str(gathered.get("surroundings", "") or "")
        cells: dict[tuple[int, int], str] = {}
        for raw_line in surroundings.splitlines():
            line = raw_line.strip()
            match = re.match(r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*:\s*(.+)$", line)
            if not match:
                continue
            cells[(int(match.group(1)), int(match.group(2)))] = match.group(3).strip()
        return cells, {}

    @staticmethod
    def _is_open_ground_tile(value: object) -> bool:
        return str(value or "").strip().lower() in {"", "empty"}

    @staticmethod
    def _is_valid_hoe_target(target_obj: object) -> bool:
        text = str(target_obj or "").strip().lower()
        return "dirt" in text or "soil" in text

    @staticmethod
    def _is_allowed_empty_hoe_target(state: object, target_obj: object, direction: str) -> bool:
        return str(target_obj or "").strip().lower() == "empty"

    @staticmethod
    def _parse_move_components(action_text: object):
        match = re.match(r'^move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)$', str(action_text or "").strip())
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _adjacent_cell_to_direction(cell: tuple[int, int]) -> str:
        mapping = {
            (0, -1): "up",
            (0, 1): "down",
            (-1, 0): "left",
            (1, 0): "right",
        }
        return mapping.get(tuple(cell), "")

    @staticmethod
    def _is_door_or_entrance_text(obj_text: object) -> bool:
        lowered = str(obj_text or "").strip().lower()
        return any(token in lowered for token in ("door", "entrance", "exit"))

    @staticmethod
    def _target_text_is_ready_to_harvest(target_obj: object) -> bool:
        text = str(target_obj or "").strip().lower()
        return "ready to harvest" in text

    @classmethod
    def _nearby_ready_harvest_crop_cells(cls, game_state: object):
        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        ready_cells = []
        for cell, value in surroundings_map.items():
            if cell == (0, 0):
                continue
            if cls._target_text_is_ready_to_harvest(value):
                ready_cells.append(cell)
        return ready_cells

    @staticmethod
    def _get_single_axis_path_blocker(action_text: object, state: object):
        return state.get("path_blocker", (0, "")) if isinstance(state, dict) else (0, "")

    @staticmethod
    def _should_preserve_navigation_anchor_move(action_text: str, game_state: object, blocker: object) -> bool:
        return bool(game_state.get("preserve_anchor", False)) if isinstance(game_state, dict) else False

    @staticmethod
    def _build_structure_blocked_move_recovery(action_text: str, game_state: object, blocker: object) -> str:
        return str(game_state.get("reroute_action", "") or "") if isinstance(game_state, dict) else ""

    @staticmethod
    def _best_route_waypoint_candidate(game_state: object):
        return game_state.get("best_waypoint") if isinstance(game_state, dict) else None

    @staticmethod
    def _build_step_toward_cell_move(cell: tuple[int, int], game_state: object = None, *, max_stride: int = 3) -> str:
        step_x = 0 if cell[0] == 0 else (1 if cell[0] > 0 else -1)
        step_y = 0 if cell[1] == 0 else (1 if cell[1] > 0 else -1)
        return f"move(x={step_x}, y={step_y})"


class TestCortexRuntimeUtils(unittest.TestCase):
    def test_load_object_name_to_id_reads_content_wrapper(self) -> None:
        with mock.patch.object(cortex_runtime_utils, "_OBJECT_NAME_TO_ID", None):
            mapping = _load_object_name_to_id()

        self.assertEqual(mapping.get("wood"), "388")
        self.assertEqual(mapping.get("parsnip seeds"), "472")

    def test_reset_cortex_no_execution_watchdog_tolerates_invalid_counters(self) -> None:
        result = reset_cortex_no_execution_watchdog(
            state={
                "blocked_replan_count": "oops",
                "no_execution_return_count": object(),
            }
        )

        self.assertEqual(result["blocked_replan_count"], 0)
        self.assertEqual(result["no_execution_return_count"], 0)

    def test_runtime_validation_rewrites_forage_move_when_adjacent_target_is_grounded(self) -> None:
        state = {
            "task": "forage_1_clam",
            "main_task": "forage_1_clam",
            "prompt_profile": "navigation",
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "surroundings": "\n".join(
                    [
                        "[0, 1]: Clam",
                        "[1, 0]: Empty",
                    ]
                ),
            },
        }

        with mock.patch.object(cortex_runtime_utils, "_load_fastllm_runtime_classes", return_value=(_StubCultivationVLLM, None)):
            validated = validate_runtime_pre_execution_action(
                state=state,
                action_text='move(x=1, y=0)',
            )

        self.assertFalse(validated["is_valid"])
        self.assertEqual(
            validated["invalid_reason"],
            "runtime_validation:forage_adjacent_target_requires_interact",
        )
        self.assertEqual(validated["fallback_action"], 'interact(direction="down")')

    def test_runtime_validation_rewrites_forage_move_away_from_visible_image_target(self) -> None:
        state = {
            "task": "forage_1_clam",
            "main_task": "forage_1_clam",
            "prompt_profile": "navigation",
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Empty",
                        "[-1, 0]: Empty",
                        "[0, 1]: Empty",
                        "[0, -1]: Empty",
                    ]
                ),
                "description": (
                    "In grid (1,1), there is a clam on the sand. "
                    "In grid (1,3), the player character is standing near the beach hut."
                ),
            },
        }

        with mock.patch.object(cortex_runtime_utils, "_load_fastllm_runtime_classes", return_value=(_StubCultivationVLLM, None)):
            validated = validate_runtime_pre_execution_action(
                state=state,
                action_text='move(x=1, y=0)',
            )

        self.assertFalse(validated["is_valid"])
        self.assertEqual(
            validated["invalid_reason"],
            "runtime_validation:forage_visible_target_requires_local_alignment",
        )
        self.assertEqual(validated["fallback_action"], "move(x=-1, y=0)")

    def test_runtime_validation_allows_forage_move_toward_visible_image_target(self) -> None:
        state = {
            "task": "forage_1_clam",
            "main_task": "forage_1_clam",
            "prompt_profile": "navigation",
            "gathered_info": {
                "current_menu": {"type": "No Menu"},
                "surroundings": "\n".join(
                    [
                        "[1, 0]: Empty",
                        "[-1, 0]: Empty",
                        "[0, 1]: Empty",
                        "[0, -1]: Empty",
                    ]
                ),
                "description": (
                    "In grid (1,1), there is a clam on the sand. "
                    "In grid (1,3), the player character is standing near the beach hut."
                ),
            },
        }

        with mock.patch.object(cortex_runtime_utils, "_load_fastllm_runtime_classes", return_value=(_StubCultivationVLLM, None)):
            validated = validate_runtime_pre_execution_action(
                state=state,
                action_text='move(x=-1, y=0)',
            )

        self.assertTrue(validated["is_valid"])

    def test_same_step_no_execution_streak_resets_when_signature_changes(self) -> None:
        state = {
            "last_no_execution_step": 5,
            "last_no_execution_signature": "move_target_blocked | shot_a | subtask_a",
            "same_step_no_execution_streak": 4,
            "same_step_blocked_since_ts": 100.0,
        }

        updates = record_cortex_no_execution(
            state=state,
            step_num=5,
            blocked_reason="unsupported_menu_action",
            screenshot_path="shot_b",
            subtask_description="subtask_b",
            planning_sec=12.0,
            now_ts=160.0,
        )

        self.assertEqual(updates["same_step_no_execution_streak"], 1)
        self.assertEqual(updates["same_step_blocked_since_ts"], 160.0)
        self.assertEqual(updates["same_step_elapsed_sec"], 0.0)

    def test_same_step_no_execution_streak_preserves_timer_for_identical_signature(self) -> None:
        state = {
            "last_no_execution_step": 5,
            "last_no_execution_signature": "move_target_blocked | shot_a | subtask_a",
            "same_step_no_execution_streak": 2,
            "same_step_blocked_since_ts": 100.0,
        }

        updates = record_cortex_no_execution(
            state=state,
            step_num=5,
            blocked_reason="move_target_blocked",
            screenshot_path="shot_a",
            subtask_description="subtask_a",
            planning_sec=12.0,
            now_ts=160.0,
        )

        self.assertEqual(updates["same_step_no_execution_streak"], 3)
        self.assertEqual(updates["same_step_blocked_since_ts"], 100.0)
        self.assertEqual(updates["same_step_elapsed_sec"], 60.0)

    def test_watering_validation_requires_watering_can_selection(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": None,
                    "selected_item_name": "Hoe",
                    "toolbar_information": "slot_index 2: Watering Can",
                    "inventory": [],
                },
            ),
        ):
            result = validate_cultivation_pre_execution_action(
                state={"main_task": "water 5 crop with watering can", "gathered_info": {}},
                action_text='use(direction="down")',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_reason"], "cultivation_validation:water_requires_watering_can")
        self.assertEqual(result["failure_root_cause"], "stale_subtask")
        self.assertEqual(result["required_change_type"], "change_selected_item")

    def test_cultivation_dialogue_menu_open_provides_confirm_fallback(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": {"type": "DialogueBox", "dialogues": ["You found a Geode!"]},
                    "selected_item_name": "Axe",
                    "toolbar_information": "slot_index 0: Axe",
                    "inventory": [],
                },
            ),
        ):
            result = validate_cultivation_pre_execution_action(
                state={"main_task": "harvest_1_egg", "gathered_info": {}},
                action_text='move(x=4, y=0)',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_reason"], "cultivation_validation:menu_open:dialoguebox")
        self.assertEqual(result["required_change_type"], "close_menu")
        self.assertEqual(result["fallback_action"], "choose_option(option_index=1, quantity=0)")

    def test_cultivation_non_dialogue_menu_open_provides_close_fallback(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": {"type": "Inventory"},
                    "selected_item_name": "Hoe",
                    "toolbar_information": "slot_index 1: Hoe",
                    "inventory": [],
                },
            ),
        ):
            result = validate_cultivation_pre_execution_action(
                state={"main_task": "till_5_tile_with_hoe", "gathered_info": {}},
                action_text='move(x=1, y=0)',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_reason"], "cultivation_validation:menu_open:inventory")
        self.assertEqual(result["fallback_action"], 'menu(option="close", menu_name="current_menu")')

    def test_till_move_rewrites_to_use_when_adjacent_target_is_available(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": None,
                    "selected_item_name": "Hoe",
                    "toolbar_information": "slot_index 1: Hoe",
                    "inventory": [],
                },
            ),
        ):
            result = validate_cultivation_pre_execution_action(
                state={
                    "main_task": "till_5_tile_with_hoe",
                    "gathered_info": {
                        "surroundings": "\n".join(
                            [
                                "[0, -1]: empty",
                                "[1, 0]: Stone",
                                "[0, 1]: empty",
                                "[-1, 0]: empty",
                            ]
                        )
                    },
                },
                action_text='move(x=1, y=0)',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_reason"], "cultivation_validation:till_adjacent_target_requires_use")
        self.assertEqual(result["fallback_action"], 'use(direction="up")')

    def test_till_use_on_adjacent_empty_target_is_allowed_even_near_buildings(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": None,
                    "selected_item_name": "Hoe",
                    "toolbar_information": "slot_index 1: Hoe",
                    "inventory": [],
                },
            ),
        ):
            result = validate_cultivation_pre_execution_action(
                state={
                    "main_task": "till_5_tile_with_hoe",
                    "zero_progress_streak": 0,
                    "repeated_action_streak": 0,
                    "position_issue_detected": False,
                    "gathered_info": {
                        "surroundings": "\n".join(
                            [
                                "[-1, 0]: empty",
                                "[0, -1]: Coop",
                                "[0, 0]: empty",
                                "[0, 1]: empty",
                                "[1, 0]: empty",
                            ]
                        )
                    },
                },
                action_text='use(direction="down")',
            )

        self.assertTrue(result["is_valid"])

    def test_watering_validation_requests_facing_change_for_wrong_direction(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": None,
                    "selected_item_name": "Watering Can",
                    "toolbar_information": "slot_index 2: Watering Can",
                    "inventory": [],
                },
            ),
        ):
            result = validate_cultivation_pre_execution_action(
                state={"main_task": "water 5 crop with watering can", "gathered_info": {}},
                action_text='use(direction="down")',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_reason"], "cultivation_validation:water_invalid_target")
        self.assertEqual(result["failure_root_cause"], "wrong_facing_direction")
        self.assertEqual(result["required_change_type"], "change_facing")

    def test_sow_validation_requires_seed_selection(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": None,
                    "selected_item_name": "Hoe",
                    "toolbar_information": "slot_index 2: Parsnip Seeds",
                    "inventory": [],
                },
            ),
        ):
            result = validate_cultivation_pre_execution_action(
                state={"main_task": "sow 5 dirt with cauliflower seeds", "gathered_info": {}},
                action_text='interact(direction="down")',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_reason"], "cultivation_validation:sow_requires_seed_selected")
        self.assertEqual(result["failure_root_cause"], "stale_subtask")
        self.assertEqual(result["required_change_type"], "change_selected_item")

    def test_sow_validation_rejects_use_and_prefers_interact(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": None,
                    "selected_item_name": "Parsnip Seeds",
                    "toolbar_information": "slot_index 2: Parsnip Seeds",
                    "inventory": [],
                },
            ),
        ):
            result = validate_cultivation_pre_execution_action(
                state={"main_task": "sow 5 dirt with cauliflower seeds", "gathered_info": {}},
                action_text='use(direction="down")',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_reason"], "cultivation_validation:sow_requires_interact")
        self.assertEqual(result["failure_root_cause"], "stale_subtask")
        self.assertEqual(result["required_change_type"], "rebuild_subtask")

    def test_sow_validation_rewrites_use_to_interact_when_seed_target_is_valid(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": None,
                    "selected_item_name": "Parsnip Seeds",
                    "toolbar_information": "slot_index 2: Parsnip Seeds",
                    "inventory": [],
                },
            ),
        ):
            result = validate_cultivation_pre_execution_action(
                state={"main_task": "sow 5 dirt with cauliflower seeds", "gathered_info": {}},
                action_text='use(direction="right")',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_reason"], "cultivation_validation:sow_requires_interact")
        self.assertEqual(result["required_change_type"], "rebuild_subtask")
        self.assertEqual(result["fallback_action"], 'interact(direction="right")')

    def test_sow_validation_requests_facing_change_for_wrong_direction(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": None,
                    "selected_item_name": "Parsnip Seeds",
                    "toolbar_information": "slot_index 2: Parsnip Seeds",
                    "inventory": [],
                },
            ),
        ):
            result = validate_cultivation_pre_execution_action(
                state={"main_task": "sow 5 dirt with cauliflower seeds", "gathered_info": {}},
                action_text='interact(direction="down")',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_reason"], "cultivation_validation:sow_invalid_target")
        self.assertEqual(result["failure_root_cause"], "wrong_facing_direction")
        self.assertEqual(result["required_change_type"], "change_facing")

    def test_runtime_validation_rejects_zero_move(self) -> None:
        result = validate_runtime_pre_execution_action(
            state={"main_task": "go_to_coop"},
            action_text="move(x=0, y=0)",
        )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_reason"], "runtime_validation:zero_move_invalid")

    def test_runtime_validation_rewrites_consumable_dialogue_to_decline_for_farm_ops(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": {
                        "type": "DialogueBox",
                        "dialogues": ["Eat Egg?"],
                        "responses": [
                            {"responseKey": "Yes", "responseText": "Yes"},
                            {"responseKey": "No", "responseText": "No"},
                        ],
                    },
                    "selected_item_name": "Egg",
                    "toolbar_information": "slot_index 5: Egg",
                    "inventory": ["slot_index 5: Egg (quantity: 1)"],
                },
            ),
        ):
            result = validate_runtime_pre_execution_action(
                state={
                    "main_task": "incubate_1_chicken_with_incubator",
                    "prompt_profile": "farm_ops",
                    "gathered_info": {},
                },
                action_text='move(x=1, y=0)',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_reason"], "runtime_validation:menu_open:dialoguebox")
        self.assertEqual(result["failure_root_cause"], "menu_stuck")
        self.assertEqual(result["required_change_type"], "close_menu")
        self.assertEqual(result["fallback_action"], "choose_option(option_index=2, quantity=0)")

    def test_harvest_validation_rewrites_adjacent_move_to_interact(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": None,
                    "selected_item_name": "Axe",
                    "toolbar_information": "slot_index 0: Axe",
                    "inventory": [],
                },
            ),
        ):
            result = validate_cultivation_pre_execution_action(
                state={
                    "main_task": "harvest_5_parsnip",
                    "gathered_info": {
                        "surroundings": "\n".join(
                            [
                                "[-1, 0]: Parsnip (ready to harvest), HoeDirt",
                                "[0, 1]: HoeDirt",
                            ]
                        )
                    },
                },
                action_text='move(x=-1, y=0)',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(
            result["invalid_reason"],
            "cultivation_validation:harvest_adjacent_target_requires_interact",
        )
        self.assertEqual(result["failure_root_cause"], "stale_subtask")
        self.assertEqual(result["required_change_type"], "rebuild_subtask")
        self.assertEqual(result["fallback_action"], 'interact(direction="left")')

    def test_runtime_validation_rejects_combat_interact(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": None,
                    "selected_item_name": "Rusty Sword",
                    "toolbar_information": "slot_index 5: Rusty Sword",
                    "inventory": [],
                },
            ),
        ):
            result = validate_runtime_pre_execution_action(
                state={"main_task": "kill_10_green_slime_with_rusty_sword", "prompt_profile": "combat", "gathered_info": {}},
                action_text='interact(direction="down")',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_reason"], "runtime_validation:combat_interact_invalid")

    def test_runtime_validation_rejects_reopening_crafting_menu(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": {"type": "crafting"},
                    "selected_item_name": "",
                    "toolbar_information": "",
                    "inventory": [],
                },
            ),
        ):
            result = validate_runtime_pre_execution_action(
                state={"main_task": "craft_1_basic_retaining_soil", "prompt_profile": "crafting", "gathered_info": {}},
                action_text='menu(option="open", menu_name="crafting")',
            )

        self.assertFalse(result["is_valid"])
        self.assertTrue(str(result["invalid_reason"]).startswith("runtime_validation:craft_missing_materials:"))

    def test_navigation_adjacent_door_rewrites_repeated_face_move_to_interact(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": {"type": "No Menu"},
                    "facing_direction": "up",
                    "selected_item_name": "Axe",
                    "toolbar_information": "slot_index 0: Axe",
                    "inventory": [],
                },
            ),
        ):
            result = validate_runtime_pre_execution_action(
                state={
                    "main_task": "go_to_bed",
                    "prompt_profile": "navigation",
                    "last_action": 'move(x=0, y=-1)',
                    "last_errors_info": "move(x=0, y=-1) toward up FAILED - player position did not change, path is likely blocked by an obstacle",
                    "directional_targets": {
                        "up": ("Farmhouse Door, exit: Farmhouse Entrance", ""),
                    },
                    "gathered_info": {},
                },
                action_text='move(x=0, y=-1)',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(
            result["invalid_reason"],
            "runtime_validation:navigation_adjacent_door_requires_interact",
        )
        self.assertEqual(result["fallback_action"], 'interact(direction="up")')

    def test_runtime_validation_rejects_craft_without_materials(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": {"type": "No Menu"},
                    "selected_item_name": "",
                    "toolbar_information": "",
                    "inventory": ["slot_index 0: Axe (quantity: 1)"],
                },
            ),
        ):
            result = validate_runtime_pre_execution_action(
                state={"main_task": "craft_1_basic_retaining_soil", "prompt_profile": "crafting", "gathered_info": {}},
                action_text='craft(item="Basic Retaining Soil")',
            )

        self.assertFalse(result["is_valid"])
        self.assertTrue(str(result["invalid_reason"]).startswith("runtime_validation:craft_missing_materials:"))
        self.assertEqual(result["failure_root_cause"], "item_missing")
        self.assertEqual(result["required_change_type"], "switch_to_retrieval_subtask")

    def test_runtime_validation_allows_spring_seed_material_search_move(self) -> None:
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": {"type": "No Menu"},
                    "selected_item_name": "Axe",
                    "toolbar_information": "slot_index 0: Axe (quantity: 1)",
                    "inventory": ["slot_index 0: Axe (quantity: 1)"],
                },
            ),
        ):
            result = validate_runtime_pre_execution_action(
                state={
                    "main_task": "craft_1_spring_seeds",
                    "prompt_profile": "crafting",
                    "target_item": "Spring Seeds",
                    "gathered_info": {},
                },
                action_text="move(x=3, y=1)",
            )

        self.assertTrue(result["is_valid"], result)

    def test_runtime_validation_rewrites_spring_seed_craft_alias(self) -> None:
        inventory = [
            "slot_index 0: Wild Horseradish (quantity: 1)",
            "slot_index 1: Daffodil (quantity: 1)",
            "slot_index 2: Leek (quantity: 1)",
            "slot_index 3: Dandelion (quantity: 1)",
        ]
        with (
            mock.patch(
                "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
                return_value=(_StubCultivationVLLM, object()),
            ),
            mock.patch(
                "stardojo.utils.cortex_runtime_utils.extract_stardew_prompt_fact_fields",
                return_value={
                    "current_menu": {"type": "No Menu"},
                    "selected_item_name": "",
                    "toolbar_information": "",
                    "inventory": inventory,
                },
            ),
        ):
            result = validate_runtime_pre_execution_action(
                state={
                    "main_task": "craft_1_spring_seeds",
                    "prompt_profile": "crafting",
                    "target_item": "Spring Seeds",
                    "gathered_info": {},
                },
                action_text='craft(item="Spring Seeds")',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_reason"], "runtime_validation:craft_alias_requires_recipe_name")
        self.assertEqual(result["fallback_action"], 'craft(item="Wild Seeds (Sp)")')

    def test_runtime_validation_preserves_navigation_anchor_moves(self) -> None:
        with mock.patch(
            "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
            return_value=(_StubCultivationVLLM, object()),
        ):
            result = validate_runtime_pre_execution_action(
                state={
                    "main_task": "go_to_backwoods",
                    "prompt_profile": "navigation",
                    "path_blocker": (1, "Backwoods Exit"),
                    "preserve_anchor": True,
                    "reroute_action": "move(x=-2, y=-1)",
                    "gathered_info": {"current_menu": {"type": "No Menu"}},
                },
                action_text='move(x=0, y=-1)',
            )

        self.assertTrue(result["is_valid"])

    def test_runtime_validation_clamps_large_navigation_exit_move_to_unit_step(self) -> None:
        with mock.patch(
            "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
            return_value=(_StubCultivationVLLM, object()),
        ):
            result = validate_runtime_pre_execution_action(
                state={
                    "main_task": "go_to_bus_stop",
                    "prompt_profile": "navigation",
                    "zero_progress_streak": 1,
                    "best_waypoint": {"source": "exits", "offset": (-4, 0), "raw": "Bus Stop Exit"},
                    "gathered_info": {"current_menu": {"type": "No Menu"}},
                },
                action_text='move(x=-5, y=0)',
            )

        self.assertFalse(result["is_valid"])
        self.assertEqual(result["invalid_reason"], "runtime_validation:navigation_requires_unit_step")
        self.assertEqual(result["fallback_action"], "move(x=-1, y=0)")


if __name__ == "__main__":
    unittest.main()
