from __future__ import annotations

import os
import re
import statistics
import time
import json
from typing import Any, Dict, List, Optional

from stardojo.utils.stardew_prompt_state import extract_stardew_prompt_fact_fields
from stardojo.utils.task_bootstrap import build_task_acquisition_context


LEGACY_COMPACT_PROMPT_SOURCE = "legacy_compact_prompt"

_CULTIVATION_PREFIXES = (
    "till ",
    "fertilize ",
    "sow ",
    "plant ",
    "water ",
    "harvest ",
    "cultivate and harvest ",
)


_OBJECT_NAME_TO_ID: Optional[Dict[str, str]] = None
_OBJECT_ID_TO_NAME: Optional[Dict[str, str]] = None
_CRAFTING_RECIPE_TABLE: Optional[Dict[str, str]] = None

_CRAFTING_RECIPE_ALIASES = {
    "spring seeds": "Wild Seeds (Sp)",
    "summer seeds": "Wild Seeds (Su)",
    "fall seeds": "Wild Seeds (Fa)",
    "winter seeds": "Wild Seeds (Wi)",
}


class CortexConfigurationError(RuntimeError):
    """Raised when cortex runtime config and effective wiring disagree."""


def _normalize_task_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _infer_cultivation_task_kind(task_text: Any) -> str:
    normalized = _normalize_task_text(task_text)
    if normalized.startswith(("till ", "till_")):
        return "till"
    if normalized.startswith(("fertilize ", "fertilize_")):
        return "fertilize"
    if normalized.startswith(("sow ", "sow_", "plant ", "plant_")):
        return "sow"
    if normalized.startswith(("water ", "water_")):
        return "water"
    if normalized.startswith(("harvest ", "harvest_")):
        return "harvest"
    if normalized.startswith(("cultivate and harvest ", "cultivate_and_harvest_")):
        return "cultivate_and_harvest"
    return ""


def _normalize_item_name_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_animal_product_harvest_task(
    *,
    state: Optional[Dict[str, Any]],
    task_text: Any,
) -> bool:
    if _infer_cultivation_task_kind(task_text) != "harvest":
        return False

    normalized_fragments: List[str] = [_normalize_item_name_text(task_text)]
    state = state if isinstance(state, dict) else {}
    for field in (
        "main_task",
        "task",
        "task_description",
        "subtask_description",
        "target_item",
        "object",
        "source_type",
        "source_detail",
    ):
        normalized_value = _normalize_item_name_text(state.get(field, ""))
        if normalized_value:
            normalized_fragments.append(normalized_value)

    gathered = state.get("gathered_info", {})
    if isinstance(gathered, dict):
        for field in ("description", "target_item", "source_type", "source_detail"):
            normalized_value = _normalize_item_name_text(gathered.get(field, ""))
            if normalized_value:
                normalized_fragments.append(normalized_value)

    normalized_context = " ".join(fragment for fragment in normalized_fragments if fragment).strip()
    if not normalized_context:
        return False

    return any(
        re.search(rf"\b{re.escape(token)}\b", normalized_context)
        for token in ("egg", "duck egg", "milk", "goat milk", "wool", "milk pail", "shears")
    )


def _is_unit_step_move(move_components: Any) -> bool:
    if not isinstance(move_components, tuple) or len(move_components) != 2:
        return False
    try:
        move_x = int(move_components[0])
        move_y = int(move_components[1])
    except Exception:
        return False
    return (move_x != 0 or move_y != 0) and max(abs(move_x), abs(move_y)) == 1


def _build_unit_priority_navigation_reroute(
    *,
    vllm_cls: Any,
    state: Optional[Dict[str, Any]],
    action: str,
    reroute: str,
) -> str:
    safe_reroute = str(reroute or "").strip()
    if not safe_reroute or safe_reroute == action:
        return safe_reroute

    parse_move_components_fn = getattr(vllm_cls, "_parse_move_components", None)
    if not callable(parse_move_components_fn):
        return safe_reroute

    reroute_components = parse_move_components_fn(safe_reroute)
    if reroute_components is None:
        return safe_reroute

    refused_action = ""
    if isinstance(state, dict):
        refused_action = str(
            (state.get("last_exec_info", {}) or {}).get("refused_action", "") or ""
        ).strip()

    reroute_x = int(reroute_components[0])
    reroute_y = int(reroute_components[1])
    unit_candidates: List[str] = []

    build_step_toward_fn = getattr(vllm_cls, "_build_step_toward_cell_move", None)
    if callable(build_step_toward_fn):
        stepped_reroute = str(
            build_step_toward_fn(
                (reroute_x, reroute_y),
                state,
                max_stride=1,
            )
            or ""
        ).strip()
        if stepped_reroute:
            unit_candidates.append(stepped_reroute)

    direct_unit = (
        f"move(x={0 if reroute_x == 0 else (1 if reroute_x > 0 else -1)}, "
        f"y={0 if reroute_y == 0 else (1 if reroute_y > 0 else -1)})"
    )
    unit_candidates.append(direct_unit)

    for candidate in unit_candidates:
        parsed_candidate = parse_move_components_fn(candidate)
        if (
            not candidate
            or candidate == action
            or candidate == refused_action
            or not _is_unit_step_move(parsed_candidate)
        ):
            continue
        return candidate

    return safe_reroute


def _should_force_direct_craft_rewrite(
    *,
    state: Optional[Dict[str, Any]],
    craft_target_action: str,
) -> bool:
    state = state if isinstance(state, dict) else {}
    if not craft_target_action:
        return False

    zero_progress_streak = _safe_int(state.get("zero_progress_streak", 0) or 0, default=0)
    repeated_action_streak = _safe_int(state.get("repeated_action_streak", 0) or 0, default=0)
    last_action = str(state.get("last_action", "") or "").strip()

    if (
        last_action == craft_target_action
        and (zero_progress_streak >= 1 or repeated_action_streak >= 2)
    ):
        return False
    return True


def _find_visible_inventory_slot_for_target(
    *,
    vllm_cls: Any,
    inventory: Any,
    toolbar_information: Any,
    target_item: Any,
    item_kind: str = "",
) -> int | None:
    extract_slot_map = getattr(vllm_cls, "_extract_inventory_slot_map", None)
    if not callable(extract_slot_map):
        return None

    slot_map = extract_slot_map(inventory, toolbar_information)
    if not isinstance(slot_map, dict):
        return None

    target_normalized = _normalize_item_name_text(target_item)
    for slot_index, item_name in sorted(slot_map.items()):
        normalized_item = str(item_name or "").strip()
        if not normalized_item:
            continue
        if getattr(vllm_cls, "_slot_is_explicitly_empty", lambda x: False)(normalized_item):
            continue

        normalized_item_key = _normalize_item_name_text(normalized_item)
        if target_normalized and (
            target_normalized == normalized_item_key
            or target_normalized in normalized_item_key
            or normalized_item_key in target_normalized
        ):
            return int(slot_index)

    for slot_index, item_name in sorted(slot_map.items()):
        normalized_item = str(item_name or "").strip()
        if not normalized_item:
            continue
        if getattr(vllm_cls, "_slot_is_explicitly_empty", lambda x: False)(normalized_item):
            continue
        if item_kind == "sow" and vllm_cls._selected_item_is_seed(normalized_item):
            return int(slot_index)
        if item_kind == "fertilize" and vllm_cls._selected_item_is_fertilizer(normalized_item):
            return int(slot_index)

    return None


def _parse_choose_option_action(action_text: Any) -> Dict[str, Any]:
    text = str(action_text or "").strip()
    match = re.match(
        r'^choose_option\(\s*option_index\s*=\s*(-?\d+)(?:\s*,\s*quantity\s*=\s*(-?\d+))?(?:\s*,\s*direction\s*=\s*"?([a-zA-Z]+)"?)?\s*\)$',
        text,
        re.IGNORECASE,
    )
    if not match:
        return {}
    return {
        "option_index": int(match.group(1)),
        "quantity": int(match.group(2)) if match.group(2) is not None else None,
        "direction": str(match.group(3) or "").strip().lower(),
    }


def _build_adjacent_service_interact_fallback(
    *,
    state: Dict[str, Any],
    prompt_facts: Dict[str, Any],
    vllm_cls: Any,
) -> str:
    def _is_service_target(text: Any) -> bool:
        lowered = str(text or "").strip().lower()
        return any(
            token in lowered
            for token in ("counter", "shipping bin", "pet bowl", "feeding bench", "door")
        )

    def _is_open_ground(text: Any) -> bool:
        checker = getattr(vllm_cls, "_is_open_ground_tile", None)
        if callable(checker):
            return bool(checker(text))
        lowered = str(text or "").strip().lower()
        return lowered in {"", "empty", "ground", "open ground", "floor", "path", "dirt"}

    surroundings_map, _ = vllm_cls._get_structured_surroundings_map(state)
    if isinstance(surroundings_map, dict):
        for direction in ("up", "down", "left", "right"):
            relative = getattr(vllm_cls, "_direction_to_relative")(direction)
            if relative is None:
                continue
            target_text = str(surroundings_map.get(relative, "") or "").strip().lower()
            if _is_service_target(target_text):
                return f'interact(direction="{direction}")'
            if direction == "up":
                npc_text = str(surroundings_map.get((0, -2), "") or "").strip().lower()
                if ("npc:" in npc_text or "name:" in npc_text) and (
                    "counter" in target_text or "shop" in target_text
                ):
                    return 'interact(direction="up")'

        for step_x in (-1, 1):
            diagonal_text = str(surroundings_map.get((step_x, -1), "") or "").strip().lower()
            side_text = str(surroundings_map.get((step_x, 0), "") or "").strip().lower()
            if _is_service_target(diagonal_text) and _is_open_ground(side_text):
                return f"move(x={step_x}, y=0)"

    facing_direction = str(
        prompt_facts.get("facing_direction")
        or state.get("facing_direction")
        or state.get("gathered_info", {}).get("facing_direction", "")
        or ""
    ).strip().lower()
    if facing_direction not in {"up", "down", "left", "right"}:
        return ""

    target_obj, _ = vllm_cls._get_directional_target(state, facing_direction)
    target_text = str(target_obj or "").strip().lower()
    if _is_service_target(target_text):
        return f'interact(direction="{facing_direction}")'

    return ""


def _build_harvest_grounding_move_fallback(
    *,
    state: Dict[str, Any],
    vllm_cls: Any,
) -> str:
    surroundings_map, _ = vllm_cls._get_structured_surroundings_map(state)
    if not surroundings_map:
        return ""

    ready_cells = vllm_cls._nearby_ready_harvest_crop_cells(state)
    if not ready_cells:
        return ""

    def _is_safe_stand_tile(cell: tuple[int, int]) -> bool:
        text = str(surroundings_map.get(cell, "") or "").strip()
        if not text:
            return False
        if vllm_cls._target_text_contains_crop(text):
            return False
        if vllm_cls._is_hard_structure_blocker(text):
            return False
        return bool(
            vllm_cls._is_open_ground_tile(text)
            or vllm_cls._placeable_target_is_tilled(text)
            or vllm_cls._is_empty_like_tile(text)
        )

    candidates = []
    for crop_cell in ready_cells:
        crop_x, crop_y = crop_cell
        for direction in getattr(vllm_cls, "_CARDINAL_DIRECTIONS", ("up", "right", "down", "left")):
            relative = vllm_cls._direction_to_relative(direction)
            if relative is None:
                continue
            stand_cell = (crop_x - relative[0], crop_y - relative[1])
            if stand_cell == (0, 0) or abs(stand_cell[0]) + abs(stand_cell[1]) != 1:
                continue
            if not _is_safe_stand_tile(stand_cell):
                continue
            candidates.append(
                (
                    vllm_cls._cell_sort_key(crop_cell),
                    vllm_cls._cell_sort_key(stand_cell),
                    stand_cell,
                )
            )

    if not candidates:
        return ""

    _, _, stand_cell = min(candidates, key=lambda item: (item[0], item[1]))
    return f"move(x={stand_cell[0]}, y={stand_cell[1]})"


def _find_adjacent_door_interact_fallback(
    *,
    state: Dict[str, Any],
    vllm_cls: Any,
) -> str:
    is_door_fn = getattr(vllm_cls, "_is_door_or_entrance_text", None)

    for direction in ("up", "down", "left", "right"):
        try:
            target_obj, _ = vllm_cls._get_directional_target(state, direction)
        except Exception:
            continue
        target_text = str(target_obj or "").strip()
        normalized_target = target_text.lower()
        is_door = bool(
            callable(is_door_fn) and is_door_fn(target_text)
        ) or any(token in normalized_target for token in ("door", "entrance", "exit"))
        if is_door:
            return f'interact(direction="{direction}")'

    return ""


def _build_menu_blocker_fallback_action(
    *,
    current_menu: Any,
    current_menu_type: str,
    vllm_cls: Any,
) -> str:
    def _menu_dialogue_lines(menu_value: Any) -> List[str]:
        if not isinstance(menu_value, dict):
            return []
        lines: List[str] = []
        for field in ("dialogues", "chats", "message"):
            raw_value = menu_value.get(field)
            if isinstance(raw_value, (list, tuple, set)):
                lines.extend(str(item or "").strip() for item in raw_value if str(item or "").strip())
            elif str(raw_value or "").strip():
                lines.append(str(raw_value or "").strip())
        return lines

    def _menu_has_yes_no_responses(menu_value: Any) -> bool:
        if not isinstance(menu_value, dict):
            return False
        normalized_responses: List[str] = []
        responses = menu_value.get("responses")
        if not isinstance(responses, (list, tuple)):
            return False
        for response in responses:
            if isinstance(response, dict):
                label = response.get("responseText") or response.get("responseKey") or ""
            else:
                label = response
            normalized = str(label or "").strip().lower()
            if normalized:
                normalized_responses.append(normalized)
        return "yes" in normalized_responses and "no" in normalized_responses

    def _menu_prefers_negative_confirmation(menu_value: Any) -> bool:
        if not _menu_has_yes_no_responses(menu_value):
            return False
        normalized_dialogue = " ".join(
            _normalize_item_name_text(line)
            for line in _menu_dialogue_lines(menu_value)
        ).strip()
        if not normalized_dialogue:
            return False
        return bool(re.search(r"\b(eat|consume|drink)\b", normalized_dialogue))

    normalized_menu_type = str(current_menu_type or "").strip().lower()
    if normalized_menu_type in {"", "no menu", "none", "null"}:
        return ""

    if normalized_menu_type == "dialoguebox":
        if _menu_prefers_negative_confirmation(current_menu):
            return "choose_option(option_index=2, quantity=0)"
        return "choose_option(option_index=1, quantity=0)"

    if normalized_menu_type in {"objectdialogue", "notificationdialogue"}:
        return "choose_option(option_index=1, quantity=0)"

    return 'menu(option="close", menu_name="current_menu")'


def _find_adjacent_combat_attack_direction(
    *,
    state: Dict[str, Any],
    prompt_facts: Dict[str, Any],
    vllm_cls: Any,
) -> str:
    enemy_tokens = (
        "green slime",
        "slime",
        "bug",
        "fly",
        "duggy",
        "grub",
        "bat",
        "crab",
        "enemy",
        "monster",
    )

    for direction in ("up", "down", "left", "right"):
        try:
            target_obj, required_tool = vllm_cls._get_directional_target(state, direction)
        except Exception:
            continue
        target_text = str(target_obj or "").strip().lower()
        required_tool_text = str(required_tool or "").strip()
        if required_tool_text == "Rusty Sword":
            return direction
        if target_text and any(token in target_text for token in enemy_tokens):
            return direction

    facing_direction = str(
        prompt_facts.get("facing_direction")
        or state.get("facing_direction")
        or state.get("gathered_info", {}).get("facing_direction", "")
        or ""
    ).strip().lower()
    if facing_direction in {"up", "down", "left", "right"}:
        try:
            target_obj, required_tool = vllm_cls._get_directional_target(state, facing_direction)
        except Exception:
            target_obj, required_tool = "", ""
        target_text = str(target_obj or "").strip().lower()
        required_tool_text = str(required_tool or "").strip()
        if required_tool_text == "Rusty Sword":
            return facing_direction
        if target_text and any(token in target_text for token in enemy_tokens):
            return facing_direction

    return ""


def _resolve_task_target_item(
    *,
    state: Dict[str, Any],
    prompt_facts: Dict[str, Any],
) -> str:
    target_item = str(
        state.get("target_item")
        or prompt_facts.get("target_item")
        or ""
    ).strip()
    if target_item:
        return target_item

    task_text = str(state.get("main_task", "") or state.get("task", "") or "").strip()
    if not task_text:
        return ""

    acquisition_context = build_task_acquisition_context(task_text)
    target_item = str(acquisition_context.get("target_item", "") or "").strip()
    if target_item:
        return target_item

    generic_match = re.match(
        r"^(?:craft|cook|produce)[_\s]+\d+[_\s]+(.+)$",
        task_text,
        re.IGNORECASE,
    )
    if not generic_match:
        return ""

    raw_target = re.sub(r"[_\s]+", " ", generic_match.group(1)).strip()
    if not raw_target:
        return ""
    return " ".join(part.capitalize() for part in raw_target.split())


def _candidate_target_labels(target_item: Any) -> List[str]:
    normalized = _normalize_item_name_text(target_item)
    if not normalized:
        return []

    labels: List[str] = [normalized]
    if normalized.endswith("s") and len(normalized) > 3:
        singular = normalized[:-1].strip()
        if singular and singular not in labels:
            labels.append(singular)

    parts = normalized.split()
    if len(parts) > 1:
        tail = parts[-1].strip()
        if tail and tail not in labels:
            labels.append(tail)

    return labels


def _is_hay_with_scythe_forage_task(
    *,
    task_text: Any,
    state: Dict[str, Any],
    prompt_facts: Dict[str, Any],
) -> bool:
    context_fragments = [
        task_text,
        state.get("main_task", ""),
        state.get("task", ""),
        state.get("task_description", ""),
        state.get("subtask_description", ""),
        state.get("target_item", ""),
        state.get("object", ""),
        state.get("source_type", ""),
        state.get("tool", ""),
        prompt_facts.get("target_item", ""),
        prompt_facts.get("selected_item_name", ""),
    ]
    gathered = state.get("gathered_info", {})
    if isinstance(gathered, dict):
        context_fragments.extend(
            [
                gathered.get("task_description", ""),
                gathered.get("subtask_description", ""),
                gathered.get("target_item", ""),
                gathered.get("source_type", ""),
                gathered.get("tool", ""),
            ]
        )

    normalized = f" {_normalize_item_name_text(' '.join(str(item or '') for item in context_fragments))} "
    return (
        (" forage " in normalized or str(task_text or "").startswith("forage_"))
        and " hay " in normalized
        and " scythe " in normalized
    )


def _find_adjacent_hay_scythe_target_direction(
    *,
    vllm_cls: Any,
    state: Dict[str, Any],
) -> str:
    surroundings_map, _ = vllm_cls._get_structured_surroundings_map(state)
    if isinstance(surroundings_map, dict) and surroundings_map:
        candidates = []
        for cell, raw_text in surroundings_map.items():
            if cell == (0, 0):
                continue
            direction = vllm_cls._adjacent_cell_to_direction(cell)
            if not direction:
                continue
            clearable = vllm_cls._classify_clearable_object(raw_text)
            if not clearable:
                continue
            if clearable.get("tool") != "Scythe":
                continue
            if clearable.get("family") not in {"hay", "weeds"}:
                continue
            candidates.append((vllm_cls._cell_sort_key(cell), direction))
        if candidates:
            _, direction = min(candidates, key=lambda item: item[0])
            return direction

    image_target_offset = _infer_named_target_offset_from_image_description(
        state=state,
        target_labels=["grass", "hay", "weed", "weeds", "fiber", "fibre"],
    )
    if image_target_offset is None:
        return ""
    if abs(image_target_offset[0]) + abs(image_target_offset[1]) != 1:
        return ""
    return vllm_cls._adjacent_cell_to_direction(image_target_offset)


def _find_nearby_named_target_cell(
    *,
    vllm_cls: Any,
    state: Dict[str, Any],
    target_labels: List[str],
    max_distance: int = 3,
) -> Optional[tuple[int, int]]:
    if not target_labels:
        return None

    surroundings_map, _ = vllm_cls._get_structured_surroundings_map(state)
    if not isinstance(surroundings_map, dict) or not surroundings_map:
        return None

    candidates: List[tuple[int, int, int, int, int]] = []
    for cell, raw_text in surroundings_map.items():
        if cell == (0, 0):
            continue
        normalized = _normalize_item_name_text(raw_text)
        if not normalized:
            continue
        if not any(label in normalized for label in target_labels):
            continue
        cell_x, cell_y = int(cell[0]), int(cell[1])
        distance = abs(cell_x) + abs(cell_y)
        if distance > max_distance:
            continue
        candidates.append((distance, abs(cell_y), abs(cell_x), cell_x, cell_y))

    if not candidates:
        return None

    _, _, _, best_x, best_y = min(candidates)
    return int(best_x), int(best_y)


_GRID_SEGMENT_PATTERN = re.compile(
    r"in grid\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,?\s*(.*?)(?=(?:\bin grid\s*\(\s*\d+\s*,\s*\d+\s*\))|\Z)",
    re.IGNORECASE | re.DOTALL,
)


def _infer_named_target_offset_from_image_description(
    *,
    state: Dict[str, Any],
    target_labels: List[str],
) -> Optional[tuple[int, int]]:
    if not target_labels:
        return None

    gathered = state.get("gathered_info", {}) if isinstance(state, dict) else {}
    if not isinstance(gathered, dict):
        gathered = {}

    description_text = str(
        gathered.get("image_description")
        or state.get("image_description")
        or gathered.get("description")
        or state.get("description")
        or ""
    ).strip()
    if not description_text:
        return None

    target_cells: List[tuple[int, int]] = []
    player_cells: List[tuple[int, int]] = []
    player_markers = (
        "player character",
        "the player character",
        "player is standing",
        "player stands",
        "character is standing",
        "character standing",
    )

    for match in _GRID_SEGMENT_PATTERN.finditer(description_text):
        row = int(match.group(1))
        col = int(match.group(2))
        segment = match.group(3)
        normalized = _normalize_item_name_text(segment)
        if not normalized:
            continue
        if any(label in normalized for label in target_labels):
            target_cells.append((row, col))
        if any(marker in normalized for marker in player_markers):
            player_cells.append((row, col))

    if not target_cells or not player_cells:
        return None

    player_row, player_col, target_row, target_col = min(
        (
            (player_row, player_col, target_row, target_col)
            for player_row, player_col in player_cells
            for target_row, target_col in target_cells
        ),
        key=lambda item: abs(item[0] - item[2]) + abs(item[1] - item[3]),
    )
    delta_x = target_col - player_col
    delta_y = target_row - player_row
    if delta_x == 0 and delta_y == 0:
        return None
    return delta_x, delta_y


def _move_reduces_named_target_offset(
    *,
    move_components: tuple[int, int],
    target_offset: tuple[int, int],
) -> bool:
    target_x, target_y = int(target_offset[0]), int(target_offset[1])
    step_x = 0 if move_components[0] == 0 else (1 if move_components[0] > 0 else -1)
    step_y = 0 if move_components[1] == 0 else (1 if move_components[1] > 0 else -1)
    before = abs(target_x) + abs(target_y)
    after = abs(target_x - step_x) + abs(target_y - step_y)
    return after < before


def _is_cultivation_task_text(task_text: Any) -> bool:
    normalized = _normalize_task_text(task_text)
    return any(normalized.startswith(prefix) for prefix in _CULTIVATION_PREFIXES)


def should_initialize_cortex_state(
    *,
    current_state: Optional[Dict[str, Any]],
    task_description: Any,
) -> bool:
    if not isinstance(current_state, dict) or not current_state:
        return True

    expected_task = str(task_description or "").strip()
    if not expected_task:
        return False

    current_task = str(
        current_state.get("main_task", "")
        or current_state.get("task", "")
        or ""
    ).strip()
    return current_task != expected_task


def should_treat_cortex_attempt_as_first_step(
    *,
    state: Optional[Dict[str, Any]],
    step_num: Any,
) -> bool:
    try:
        current_step = int(step_num)
    except (TypeError, ValueError):
        current_step = 0

    if current_step != 0:
        return False

    if not isinstance(state, dict) or not state:
        return True

    try:
        planning_attempt_count = int(state.get("planning_attempt_count", 0) or 0)
    except (TypeError, ValueError):
        planning_attempt_count = 0

    if planning_attempt_count > 0:
        return False
    if bool(state.get("has_execution_feedback", False)):
        return False
    if bool(state.get("execution_pending", False)):
        return False
    if state.get("previous_actions") or state.get("previous_results"):
        return False
    return True


def resolve_little_brain_prompt_source(
    *,
    use_stardew_template: bool,
    little_brain_available: bool,
    template_path: str,
) -> str:
    if not use_stardew_template:
        return LEGACY_COMPACT_PROMPT_SOURCE

    if not little_brain_available:
        raise CortexConfigurationError(
            "LittleBrain stardew template requested but little_brain is unavailable."
        )

    if not os.path.exists(template_path):
        raise CortexConfigurationError(
            f"LittleBrain stardew template requested but missing: {template_path}"
        )

    return template_path


def select_cortex_suggestion_for_logging(
    result_state: Dict[str, Any],
    normalized_actions: List[str],
    normalized_suggestion_actions: List[str],
) -> Optional[str]:
    if not normalized_actions or not normalized_suggestion_actions:
        return None

    # BigBrain planning returns the full suggestion list. LittleBrain returns
    # only the action it just executed, and current_step already points to the
    # next step after execution.
    if len(normalized_actions) != 1:
        return normalized_suggestion_actions[0]

    try:
        current_step = int(result_state.get("current_step", 0) or 0)
    except (TypeError, ValueError):
        current_step = 0

    executed_step_index = max(current_step - 1, 0)
    if executed_step_index >= len(normalized_suggestion_actions):
        executed_step_index = 0

    return normalized_suggestion_actions[executed_step_index]


def resolve_cortex_executable_actions(
    *,
    result_state: Dict[str, Any],
    normalized_actions: List[str],
    normalized_suggestion_actions: List[str],
) -> Dict[str, Any]:
    escalation_reason = str(result_state.get("escalation_reason", "") or "").strip()
    brain_mode = str(result_state.get("brain_mode", "") or "").strip().lower()
    execution_pending = bool(result_state.get("execution_pending", False))
    has_execution_feedback = bool(result_state.get("has_execution_feedback", False))
    allow_suggestion_execution_fallback = bool(
        result_state.get("allow_suggestion_execution_fallback", False)
    )
    pending_action = str(result_state.get("pending_action", "") or "").strip()

    blocked_reason = ""
    if escalation_reason:
        blocked_reason = f"escalation_reason:{escalation_reason}"
    elif brain_mode == "big":
        blocked_reason = "awaiting_big_brain_replan"
    elif normalized_suggestion_actions:
        blocked_reason = "suggestion_fallback_disabled"
    else:
        blocked_reason = "empty_plan"

    if execution_pending:
        pending_actions = list(normalized_actions)
        if pending_action and not pending_actions:
            pending_actions = [pending_action]
        if pending_actions and not has_execution_feedback:
            return {
                "actions": [pending_actions[0]],
                "execution_source": "pending_action",
                "blocked_reason": "",
                "used_suggestion_fallback": False,
            }
        return {
            "actions": [],
            "execution_source": "pending_action",
            "blocked_reason": "awaiting_execution_feedback",
            "used_suggestion_fallback": False,
        }

    if str(result_state.get("pending_local_recovery_action", "") or "").strip():
        pending_recovery = _resolve_runtime_no_action_recovery(
            result_state=result_state,
            blocked_reason="pending_local_recovery",
            normalized_suggestion_actions=normalized_suggestion_actions,
        )
        if pending_recovery:
            return {
                "actions": [pending_recovery["action"]],
                "execution_source": pending_recovery["source"],
                "blocked_reason": "",
                "used_suggestion_fallback": pending_recovery["source"] == "suggestions",
            }

    if normalized_actions:
        return {
            "actions": list(normalized_actions),
            "execution_source": "planned_actions",
            "blocked_reason": "",
            "used_suggestion_fallback": False,
        }

    recovery = _resolve_runtime_no_action_recovery(
        result_state=result_state,
        blocked_reason=blocked_reason,
        normalized_suggestion_actions=normalized_suggestion_actions,
    )
    if recovery:
        return {
            "actions": [recovery["action"]],
            "execution_source": recovery["source"],
            "blocked_reason": "",
            "used_suggestion_fallback": recovery["source"] == "suggestions",
        }

    current_step_suggestion = _select_current_step_suggestion_for_execution(
        result_state=result_state,
        normalized_suggestion_actions=normalized_suggestion_actions,
    )
    if current_step_suggestion and allow_suggestion_execution_fallback:
        return {
            "actions": [current_step_suggestion],
            "execution_source": "suggestions",
            "blocked_reason": "",
            "used_suggestion_fallback": True,
        }

    if normalized_suggestion_actions:
        return {
            "actions": [],
            "execution_source": "blocked",
            "blocked_reason": blocked_reason,
            "used_suggestion_fallback": False,
        }

    return {
        "actions": [],
        "execution_source": "none",
        "blocked_reason": blocked_reason,
        "used_suggestion_fallback": False,
    }


def is_redundant_tool_selection_subtask(text: str, selected_item_name: str) -> bool:
    lowered = str(text or "").strip().lower()
    item_lower = str(selected_item_name or "").strip().lower()
    if not lowered or not item_lower:
        return False

    selection_verbs = ("select", "equip", "choose", "switch to", "pick")
    selection_targets = ("toolbar", "tool bar", "inventory", "slot")

    has_selection_verb = any(token in lowered for token in selection_verbs)
    has_selection_target = any(token in lowered for token in selection_targets)
    return has_selection_verb and has_selection_target and item_lower in lowered


def build_sanitized_subtask_hints(
    *,
    subtask_description: Any,
    subtask_reasoning: Any,
    selected_item_name: Any,
) -> Dict[str, Any]:
    description = str(subtask_description or "").strip()
    reasoning = str(subtask_reasoning or "").strip()
    item_name = str(selected_item_name or "").strip()

    if not item_name or not is_redundant_tool_selection_subtask(description, item_name):
        return {
            "sanitized_subtask_hint": "",
            "redundant_tool_selection": False,
            "selected_item_already_correct": False,
        }

    hint = (
        f"{item_name} is already selected in the toolbar, so do not spend another step "
        f"reselecting it. Continue the current subtask directly."
    )
    if reasoning:
        hint = f"{hint} Previous subtask reasoning: {reasoning}"

    return {
        "sanitized_subtask_hint": hint,
        "redundant_tool_selection": True,
        "selected_item_already_correct": True,
    }


def resolve_workflow_subtask_values(
    *,
    state: Optional[Dict[str, Any]],
    initial_subtask_description: Any,
    initial_subtask_reasoning: Any,
) -> Dict[str, str]:
    initial_description = str(initial_subtask_description or "").strip()
    initial_reasoning = str(initial_subtask_reasoning or "").strip()

    if isinstance(state, dict):
        state_description = str(state.get("subtask_description", "") or "").strip()
        state_reasoning = str(state.get("subtask_reasoning", "") or "").strip()
        if state_description or state_reasoning:
            return {
                "subtask_description": state_description or initial_description,
                "subtask_reasoning": state_reasoning or initial_reasoning,
            }

    return {
        "subtask_description": initial_description,
        "subtask_reasoning": initial_reasoning,
    }


def is_grounded_blocker_reason(reason: Any) -> bool:
    lowered = str(reason or "").strip().lower()
    if not lowered:
        return False
    grounded_markers = (
        "move_target_blocked",
        "scythe_invalid_target",
        "blocked by an obstacle",
        "path is likely blocked",
        "invalid_target",
    )
    return any(marker in lowered for marker in grounded_markers)


def is_recoverable_no_execution_reason(reason: Any) -> bool:
    lowered = str(reason or "").strip().lower()
    if not lowered:
        return False
    if lowered == "awaiting_execution_feedback":
        return True
    if lowered.startswith("escalation_reason:vllm_escalate:") and any(
        token in lowered for token in ("timeout", "api_error", "throttle_timeout")
    ):
        return True
    if is_grounded_blocker_reason(lowered):
        return True
    recoverable_markers = (
        "suggestion_fallback_disabled",
        "parse_fallback_invalidated_suggestion",
        "parse_fallback_invalidated_local_recovery",
        "position mismatch",
        "wrong position",
        "not adjacent",
        "too far away",
    )
    return any(marker in lowered for marker in recoverable_markers)


def _build_no_execution_signature(
    *,
    blocked_reason: Any,
    screenshot_path: Any,
    subtask_description: Any,
) -> str:
    return " | ".join(
        [
            str(blocked_reason or "").strip(),
            str(screenshot_path or "").strip(),
            str(subtask_description or "").strip(),
        ]
    )


def _append_recent_float(
    values: Any,
    value: Any,
    *,
    max_items: int,
) -> List[float]:
    history: List[float] = []
    if isinstance(values, list):
        for item in values:
            try:
                history.append(float(item))
            except (TypeError, ValueError):
                continue
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = None
    if numeric_value is not None and numeric_value >= 0.0:
        history.append(numeric_value)
    if max_items > 0 and len(history) > max_items:
        history = history[-max_items:]
    return history


def record_cortex_planning_latency(
    *,
    state: Optional[Dict[str, Any]],
    planning_sec: Any,
    max_planning_history: int = 12,
) -> Dict[str, Any]:
    state = state if isinstance(state, dict) else {}
    history = _append_recent_float(
        state.get("recent_planning_sec_window", []),
        planning_sec,
        max_items=max_planning_history,
    )
    last_planning_sec = history[-1] if history else None
    planning_sec_median = None
    if history:
        try:
            planning_sec_median = float(statistics.median(history))
        except statistics.StatisticsError:
            planning_sec_median = None
    return {
        "recent_planning_sec_window": history,
        "last_planning_sec": last_planning_sec,
        "planning_sec_median": planning_sec_median,
    }


def should_use_blocker_replan_only(
    *,
    state: Optional[Dict[str, Any]],
    step_num: int,
    blocked_reason: Any,
    screenshot_path: Any,
    subtask_description: Any,
    blocker_replan_threshold: int = 2,
) -> bool:
    if not isinstance(state, dict):
        return False
    if blocker_replan_threshold <= 0 or not is_grounded_blocker_reason(blocked_reason):
        return False
    previous_step_raw = state.get("last_no_execution_step", -1)
    try:
        previous_step = int(previous_step_raw) if previous_step_raw not in (None, "") else -1
    except (TypeError, ValueError):
        previous_step = -1
    current_step = _safe_int(step_num, default=-1)
    if previous_step != current_step:
        return False
    previous_signature = str(state.get("last_blocker_signature", "") or "").strip()
    current_signature = _build_no_execution_signature(
        blocked_reason=blocked_reason,
        screenshot_path=screenshot_path,
        subtask_description=subtask_description,
    )
    if not previous_signature or previous_signature != current_signature:
        return False
    try:
        blocker_replan_streak = int(state.get("blocker_replan_streak", 0) or 0)
    except (TypeError, ValueError):
        blocker_replan_streak = 0
    return blocker_replan_streak >= blocker_replan_threshold


def should_invalidate_stale_plan_after_no_execution(blocked_reason: Any) -> bool:
    lowered = str(blocked_reason or "").strip().lower()
    if not lowered:
        return True
    if is_recoverable_no_execution_reason(lowered):
        return False
    return True


def record_cortex_no_execution(
    *,
    state: Optional[Dict[str, Any]],
    step_num: int,
    blocked_reason: Any,
    screenshot_path: Any,
    subtask_description: Any,
    planning_sec: Any = None,
    now_ts: Optional[float] = None,
    blocker_replan_threshold: int = 2,
    no_execution_watchdog_threshold: int = 16,
    no_execution_watchdog_seconds: float = 300.0,
    no_execution_watchdog_multiplier: float = 2.5,
    no_execution_watchdog_cap_seconds: float = 900.0,
    no_execution_watchdog_min_samples: int = 3,
    no_execution_planning_history: int = 5,
) -> Dict[str, Any]:
    state = state if isinstance(state, dict) else {}
    now = float(time.time() if now_ts is None else now_ts)
    current_step = _safe_int(step_num, default=0)
    current_signature = _build_no_execution_signature(
        blocked_reason=blocked_reason,
        screenshot_path=screenshot_path,
        subtask_description=subtask_description,
    )
    task_text = str(
        state.get("main_task", "")
        or state.get("task", "")
        or ""
    ).strip()
    cultivation_task = _is_cultivation_task_text(task_text)
    previous_step_raw = state.get("last_no_execution_step", -1)
    try:
        previous_step = int(previous_step_raw) if previous_step_raw not in (None, "") else -1
    except (TypeError, ValueError):
        previous_step = -1
    previous_signature = str(state.get("last_no_execution_signature", "") or "").strip()
    same_step = previous_step == current_step
    same_signature = same_step and previous_signature == current_signature and bool(current_signature)

    blocked_since_ts = state.get("same_step_blocked_since_ts", None)
    if same_signature and blocked_since_ts not in (None, ""):
        try:
            blocked_since = float(blocked_since_ts)
        except (TypeError, ValueError):
            blocked_since = now
    else:
        blocked_since = now

    try:
        previous_same_step_streak = int(state.get("same_step_no_execution_streak", 0) or 0)
    except (TypeError, ValueError):
        previous_same_step_streak = 0
    same_step_no_execution_streak = previous_same_step_streak + 1 if same_signature else 1

    grounded_blocker = is_grounded_blocker_reason(blocked_reason)
    blocker_signature = current_signature if grounded_blocker else ""
    previous_blocker_signature = str(state.get("last_blocker_signature", "") or "").strip()
    try:
        previous_blocker_replan_streak = int(state.get("blocker_replan_streak", 0) or 0)
    except (TypeError, ValueError):
        previous_blocker_replan_streak = 0
    if blocker_signature and same_signature and blocker_signature == previous_blocker_signature:
        blocker_replan_streak = previous_blocker_replan_streak + 1
    elif blocker_signature:
        blocker_replan_streak = 1
    else:
        blocker_replan_streak = 0

    blocker_replan_only = bool(
        blocker_signature
        and blocker_replan_threshold > 0
        and blocker_replan_streak >= blocker_replan_threshold
    )

    watchdog_triggered = False
    watchdog_reason = ""
    same_step_elapsed_s = max(0.0, now - blocked_since)
    no_execution_history = _append_recent_float(
        state.get("recent_no_execution_planning_sec_window", []),
        planning_sec,
        max_items=no_execution_planning_history,
    )
    no_execution_planning_sec = no_execution_history[-1] if no_execution_history else None
    if len(no_execution_history) >= max(1, int(no_execution_watchdog_min_samples)):
        try:
            no_execution_planning_sec_median = float(statistics.median(no_execution_history))
        except statistics.StatisticsError:
            no_execution_planning_sec_median = None
    else:
        no_execution_planning_sec_median = None

    dynamic_timeout_s = float(no_execution_watchdog_seconds)
    if cultivation_task:
        no_execution_watchdog_seconds = max(float(no_execution_watchdog_seconds), 480.0)
        no_execution_watchdog_multiplier = max(float(no_execution_watchdog_multiplier), 3.0)
        no_execution_watchdog_cap_seconds = max(float(no_execution_watchdog_cap_seconds), 1200.0)
        dynamic_timeout_s = float(no_execution_watchdog_seconds)
    if (
        no_execution_planning_sec_median is not None
        and no_execution_watchdog_multiplier > 0
    ):
        dynamic_timeout_s = max(
            float(no_execution_watchdog_seconds),
            min(
                float(no_execution_watchdog_cap_seconds),
                float(no_execution_planning_sec_median) * float(no_execution_watchdog_multiplier),
            ),
        )
    failure_signature = str(state.get("failure_signature", "") or "").strip()
    previous_deadlock_signature = str(state.get("deadlock_signature", "") or "").strip()
    current_deadlock_signature = failure_signature or current_signature
    same_deadlock_signature = (
        same_step
        and bool(current_deadlock_signature)
        and current_deadlock_signature == previous_deadlock_signature
    )
    try:
        previous_deadlock_cycles = int(state.get("deadlock_reflection_cycles", 0) or 0)
    except (TypeError, ValueError):
        previous_deadlock_cycles = 0
    if same_deadlock_signature:
        deadlock_reflection_cycles = previous_deadlock_cycles + 1
    elif current_deadlock_signature:
        deadlock_reflection_cycles = 1
    else:
        deadlock_reflection_cycles = 0

    if cultivation_task and same_deadlock_signature and deadlock_reflection_cycles >= 3:
        watchdog_triggered = True
        watchdog_reason = (
            f"diagnosed_deadlock:{current_deadlock_signature}:cycles={deadlock_reflection_cycles}"
        )
    elif no_execution_watchdog_threshold > 0 and same_step_no_execution_streak >= no_execution_watchdog_threshold:
        if not cultivation_task or deadlock_reflection_cycles >= 3:
            watchdog_triggered = True
            watchdog_reason = f"same_step_no_execution_streak:{same_step_no_execution_streak}"
    elif (
        dynamic_timeout_s > 0
        and same_step
        and same_step_elapsed_s >= dynamic_timeout_s
    ):
        if not cultivation_task or deadlock_reflection_cycles >= 3:
            watchdog_triggered = True
            watchdog_reason = (
                f"same_step_no_execution_timeout:{same_step_elapsed_s:.1f}s"
                f">={dynamic_timeout_s:.1f}s"
            )

    try:
        no_execution_return_count = int(state.get("no_execution_return_count", 0) or 0) + 1
    except (TypeError, ValueError):
        no_execution_return_count = 1
    try:
        blocked_replan_count = int(state.get("blocked_replan_count", 0) or 0)
    except (TypeError, ValueError):
        blocked_replan_count = 0
    if blocker_replan_only:
        blocked_replan_count += 1

    updates = {
        "no_execution_return_count": no_execution_return_count,
        "last_no_execution_signature": current_signature,
        "last_no_execution_step": current_step,
        "same_step_no_execution_streak": same_step_no_execution_streak,
        "same_step_blocked_since_ts": blocked_since,
        "last_blocker_signature": blocker_signature,
        "blocker_replan_streak": blocker_replan_streak,
        "blocker_replan_only": blocker_replan_only,
        "blocked_replan_count": blocked_replan_count,
        "watchdog_triggered": watchdog_triggered,
        "watchdog_reason": watchdog_reason,
        "recent_no_execution_planning_sec_window": no_execution_history,
        "last_no_execution_planning_sec": no_execution_planning_sec,
        "no_execution_planning_sec_median": no_execution_planning_sec_median,
        "no_execution_planning_sample_count": len(no_execution_history),
        "watchdog_dynamic_timeout_sec": dynamic_timeout_s,
        "same_step_elapsed_sec": same_step_elapsed_s,
        "deadlock_signature": current_deadlock_signature,
        "deadlock_reflection_cycles": deadlock_reflection_cycles,
    }

    if should_invalidate_stale_plan_after_no_execution(blocked_reason):
        updates.update({
            # Repeated planning-only returns must discard the stale plan so the
            # next scheduler decision cannot hand the same suggestion back to
            # LittleBrain before BigBrain replans.
            "force_big_brain_replan": True,
            "suggestions": [],
            "planned_actions": [],
            "current_step": 0,
            "completed_steps": [],
            "brain_mode": "big",
            "execution_pending": False,
            "pending_action": "",
            "pending_step_index": None,
            "pending_suggested_action": "",
        })
        existing_escalation_reason = str(
            state.get("escalation_reason", "") or ""
        ).strip()
        if existing_escalation_reason:
            updates["escalation_reason"] = existing_escalation_reason
        elif str(blocked_reason or "").strip():
            updates["escalation_reason"] = str(blocked_reason).strip()

    return updates


def reset_cortex_no_execution_watchdog(
    *,
    state: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    state = state if isinstance(state, dict) else {}
    return {
        "last_no_execution_signature": "",
        "last_no_execution_step": None,
        "same_step_no_execution_streak": 0,
        "same_step_blocked_since_ts": None,
        "last_blocker_signature": "",
        "blocker_replan_streak": 0,
        "blocker_replan_only": False,
        "watchdog_triggered": False,
        "watchdog_reason": "",
        "blocked_replan_count": _safe_int(state.get("blocked_replan_count", 0) or 0, default=0),
        "no_execution_return_count": _safe_int(state.get("no_execution_return_count", 0) or 0, default=0),
        "recent_no_execution_planning_sec_window": [],
        "last_no_execution_planning_sec": None,
        "no_execution_planning_sec_median": None,
        "no_execution_planning_sample_count": 0,
        "watchdog_dynamic_timeout_sec": None,
        "same_step_elapsed_sec": 0.0,
        "deadlock_signature": "",
        "deadlock_reflection_cycles": 0,
    }


def _select_current_step_suggestion_for_execution(
    *,
    result_state: Dict[str, Any],
    normalized_suggestion_actions: List[str],
) -> str:
    if not normalized_suggestion_actions:
        return ""

    try:
        current_step = int(result_state.get("current_step", 0) or 0)
    except (TypeError, ValueError):
        current_step = 0

    if current_step < 0 or current_step >= len(normalized_suggestion_actions):
        return ""
    return normalized_suggestion_actions[current_step]


def validate_cultivation_pre_execution_action(
    *,
    state: Optional[Dict[str, Any]],
    action_text: Any,
) -> Dict[str, Any]:
    state = state if isinstance(state, dict) else {}
    action = str(action_text or "").strip()
    if not action:
        return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}

    task_text = str(state.get("main_task", "") or state.get("task", "") or "").strip()
    task_text_lower = task_text.lower()
    task_kind = _infer_cultivation_task_kind(task_text)
    if task_kind not in {"till", "fertilize", "sow", "water", "harvest"}:
        return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}

    animal_harvest_task = _is_animal_product_harvest_task(
        state=state,
        task_text=task_text_lower,
    )

    vllm_cls, _ = _load_fastllm_runtime_classes()
    if vllm_cls is None:
        return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}

    prompt_facts = extract_stardew_prompt_fact_fields(
        state=state,
        gathered_info=state.get("gathered_info", {}),
    )
    current_menu = prompt_facts.get("current_menu")
    current_menu_type = vllm_cls._normalize_menu_type(prompt_facts.get("current_menu"))
    if current_menu_type in {"{'type': 'crafting'}", '{"type": "crafting"}'}:
        current_menu_type = "crafting"
    if current_menu_type not in {"", "no menu", "none", "null"}:
        fallback_action = _build_menu_blocker_fallback_action(
            current_menu=current_menu,
            current_menu_type=current_menu_type,
            vllm_cls=vllm_cls,
        )
        if fallback_action and action == fallback_action:
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}
        return {
            "is_valid": False,
            "invalid_reason": f"cultivation_validation:menu_open:{current_menu_type}",
            "failure_root_cause": "menu_stuck",
            "required_change_type": "close_menu",
            "fallback_action": fallback_action,
        }

    inside_house_for_cultivation = bool(
        getattr(vllm_cls, "_is_inside_house_for_cultivation", lambda _state: False)(state)
    )
    inside_house_exit_action = ""
    if inside_house_for_cultivation:
        build_inside_house_exit = getattr(vllm_cls, "_build_inside_house_exit_recovery_action", None)
        if callable(build_inside_house_exit):
            inside_house_exit_action = str(build_inside_house_exit(game_state=state) or "").strip()
        if inside_house_exit_action and action == inside_house_exit_action:
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}
        return {
            "is_valid": False,
            "invalid_reason": f"cultivation_validation:{task_kind}_inside_house_requires_exit",
            "failure_root_cause": "wrong_tile_alignment",
            "required_change_type": "change_position",
            "fallback_action": inside_house_exit_action,
        }

    menu_action = vllm_cls._parse_menu_action(action)
    if menu_action is not None and menu_action == ("open", "inventory") and task_kind in {"sow", "fertilize"}:
        toolbar_information = (
            state.get("toolbar_information")
            or prompt_facts.get("toolbar_information")
            or ""
        )
        inventory = (
            prompt_facts.get("inventory")
            or state.get("inventory")
            or []
        )
        target_item = (
            state.get("target_item")
            or prompt_facts.get("target_item")
            or ""
        )
        slot_index = _find_visible_inventory_slot_for_target(
            vllm_cls=vllm_cls,
            inventory=inventory,
            toolbar_information=toolbar_information,
            target_item=target_item,
            item_kind=task_kind,
        )
        if slot_index is not None:
            return {
                "is_valid": False,
                "invalid_reason": f"cultivation_validation:{task_kind}_menu_open_instead_of_select",
                "failure_root_cause": "stale_subtask",
                "required_change_type": "change_selected_item",
                "fallback_action": f"choose_item(slot_index={slot_index})",
            }

    if task_kind == "harvest":
        parse_move_components = getattr(vllm_cls, "_parse_move_components", None)
        move_components = parse_move_components(action) if callable(parse_move_components) else None
        if move_components is not None:
            ready_cells = vllm_cls._nearby_ready_harvest_crop_cells(state)
            adjacent_ready = [cell for cell in ready_cells if abs(cell[0]) + abs(cell[1]) == 1]
            if adjacent_ready:
                fallback_direction = vllm_cls._adjacent_cell_to_direction(adjacent_ready[0])
                if fallback_direction:
                    return {
                        "is_valid": False,
                        "invalid_reason": "cultivation_validation:harvest_adjacent_target_requires_interact",
                        "failure_root_cause": "stale_subtask",
                        "required_change_type": "rebuild_subtask",
                        "fallback_action": f'interact(direction="{fallback_direction}")',
                    }

    selected_item_name = str(prompt_facts.get("selected_item_name", "") or "").strip()

    if task_kind == "till":
        if action.startswith("move("):
            zero_progress_streak = _safe_int(state.get("zero_progress_streak", 0) or 0, default=0)
            repeated_action_streak = _safe_int(state.get("repeated_action_streak", 0) or 0, default=0)
            position_issue_detected = bool(state.get("position_issue_detected", False))
            if (
                zero_progress_streak >= 2
                or repeated_action_streak >= 2
                or position_issue_detected
            ):
                return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}

            selected_tool_name = vllm_cls._normalize_tool_name(selected_item_name)
            if selected_tool_name == "Hoe":
                surroundings_map, _ = vllm_cls._get_structured_surroundings_map(state)
                if isinstance(surroundings_map, dict) and surroundings_map:
                    adjacent_candidates = []
                    for relative in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                        target_text = surroundings_map.get(relative, "")
                        if not target_text:
                            continue
                        if vllm_cls._is_valid_hoe_target(target_text):
                            adjacent_candidates.append(relative)
                            continue
                        if vllm_cls._is_allowed_empty_hoe_target(
                            state,
                            target_text,
                            vllm_cls._adjacent_cell_to_direction(relative),
                        ):
                            adjacent_candidates.append(relative)
                    if adjacent_candidates:
                        fallback_direction = vllm_cls._adjacent_cell_to_direction(adjacent_candidates[0])
                        if fallback_direction:
                            return {
                                "is_valid": False,
                                "invalid_reason": "cultivation_validation:till_adjacent_target_requires_use",
                                "failure_root_cause": "stale_subtask",
                                "required_change_type": "rebuild_subtask",
                                "fallback_action": f'use(direction="{fallback_direction}")',
                            }

        directional_skill = vllm_cls._parse_directional_skill(action)
        if directional_skill is None:
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}

        skill_name, direction = directional_skill
        target_obj, _required_tool = vllm_cls._get_directional_target(state, direction)
        if skill_name != "use":
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}
        selected_tool_name = vllm_cls._normalize_tool_name(selected_item_name)
        zero_progress_streak = _safe_int(state.get("zero_progress_streak", 0) or 0, default=0)
        repeated_action_streak = _safe_int(state.get("repeated_action_streak", 0) or 0, default=0)
        position_issue_detected = bool(state.get("position_issue_detected", False))
        if selected_tool_name and selected_tool_name != "Hoe":
            return {
                "is_valid": False,
                "invalid_reason": "cultivation_validation:till_requires_hoe",
                "failure_root_cause": "stale_subtask",
                "required_change_type": "change_selected_item",
            }

        if not target_obj:
            return {
                "is_valid": False,
                "invalid_reason": "cultivation_validation:till_requires_grounded_target",
                "failure_root_cause": "invalid_target_tile",
                "required_change_type": "change_target_tile",
            }
        if vllm_cls._is_valid_hoe_target(target_obj):
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}
        if vllm_cls._is_allowed_empty_hoe_target(state, target_obj, direction):
            repeated_empty_target_stuck = bool(
                zero_progress_streak >= 2
                or repeated_action_streak >= 2
                or position_issue_detected
            )
            if not repeated_empty_target_stuck:
                return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}

            toolbar_information = (
                state.get("toolbar_information")
                or prompt_facts.get("toolbar_information")
                or ""
            )
            inventory = (
                prompt_facts.get("inventory")
                or state.get("inventory")
                or []
            )
            recovery_action = str(
                getattr(vllm_cls, "_build_local_tilling_recovery_action")(
                    game_state=state,
                    selected_item_name=selected_item_name,
                    inventory=inventory,
                    toolbar_information=toolbar_information,
                    invalid_direction=direction,
                )
                or ""
            ).strip()
            return {
                "is_valid": False,
                "invalid_reason": "cultivation_validation:till_repeated_empty_target",
                "failure_root_cause": "wrong_tile_alignment" if recovery_action else "invalid_target_tile",
                "required_change_type": "change_position" if recovery_action else "change_target_tile",
                "fallback_action": recovery_action,
            }
        if vllm_cls._is_empty_like_tile(target_obj) or getattr(vllm_cls, "_is_open_ground_tile")(target_obj):
            toolbar_information = (
                state.get("toolbar_information")
                or prompt_facts.get("toolbar_information")
                or ""
            )
            inventory = (
                prompt_facts.get("inventory")
                or state.get("inventory")
                or []
            )
            recovery_action = str(
                getattr(vllm_cls, "_build_local_tilling_recovery_action")(
                    game_state=state,
                    selected_item_name=selected_item_name,
                    inventory=inventory,
                    toolbar_information=toolbar_information,
                    invalid_direction=direction,
                )
                or ""
            ).strip()
            return {
                "is_valid": False,
                "invalid_reason": "cultivation_validation:till_invalid_target",
                "failure_root_cause": "wrong_tile_alignment" if recovery_action else "invalid_target_tile",
                "required_change_type": "change_position" if recovery_action else "change_target_tile",
                "fallback_action": recovery_action,
            }
        alternative_direction = vllm_cls._find_alternative_tool_use_direction(
            game_state=state,
            tool_name="Hoe",
            invalid_direction=direction,
        )
        return {
            "is_valid": False,
            "invalid_reason": "cultivation_validation:till_invalid_target",
            "failure_root_cause": "wrong_facing_direction" if alternative_direction else "invalid_target_tile",
            "required_change_type": "change_facing" if alternative_direction else "change_target_tile",
        }

    directional_skill = vllm_cls._parse_directional_skill(action)
    if directional_skill is None:
        return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}

    skill_name, direction = directional_skill
    target_obj, _required_tool = vllm_cls._get_directional_target(state, direction)

    if task_kind == "fertilize":
        if skill_name != "interact":
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}
        toolbar_information = (
            state.get("toolbar_information")
            or prompt_facts.get("toolbar_information")
            or ""
        )
        inventory = (
            prompt_facts.get("inventory")
            or state.get("inventory")
            or []
        )
        slot_map = vllm_cls._extract_inventory_slot_map(inventory, toolbar_information)
        has_fertilizer_available = any(
            not vllm_cls._slot_is_explicitly_empty(item_name)
            and vllm_cls._selected_item_is_fertilizer(item_name)
            for item_name in slot_map.values()
        )
        if not vllm_cls._selected_item_is_fertilizer(selected_item_name):
            return {
                "is_valid": False,
                "invalid_reason": "cultivation_validation:fertilize_requires_fertilizer_selected",
                "failure_root_cause": "stale_subtask" if has_fertilizer_available else "item_missing",
                "required_change_type": "change_selected_item" if has_fertilizer_available else "switch_to_retrieval_subtask",
            }
        if not target_obj:
            return {
                "is_valid": False,
                "invalid_reason": "cultivation_validation:fertilize_requires_grounded_target",
                "failure_root_cause": "wrong_tile_alignment",
                "required_change_type": "change_position",
            }
        if vllm_cls._is_valid_placeable_target(state, direction, selected_item_name, target_obj):
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}
        alternative_directions = vllm_cls._collect_valid_placeable_directions(
            game_state=state,
            item_name=selected_item_name,
            invalid_direction=direction,
        )
        return {
            "is_valid": False,
            "invalid_reason": "cultivation_validation:fertilize_invalid_target",
            "failure_root_cause": "wrong_facing_direction" if alternative_directions else "invalid_target_tile",
            "required_change_type": "change_facing" if alternative_directions else "change_target_tile",
        }

    if task_kind == "sow":
        toolbar_information = (
            state.get("toolbar_information")
            or prompt_facts.get("toolbar_information")
            or ""
        )
        inventory = (
            prompt_facts.get("inventory")
            or state.get("inventory")
            or []
        )
        extract_slot_map = getattr(vllm_cls, "_extract_inventory_slot_map", None)
        has_seed_available = False
        if callable(extract_slot_map):
            slot_map = extract_slot_map(inventory, toolbar_information)
            if isinstance(slot_map, dict):
                has_seed_available = any(
                    not getattr(vllm_cls, "_slot_is_explicitly_empty", lambda _: False)(item_name)
                    and vllm_cls._selected_item_is_seed(item_name)
                    for item_name in slot_map.values()
                )
        target_item = _resolve_task_target_item(state=state, prompt_facts=prompt_facts)
        seed_slot_index = _find_visible_inventory_slot_for_target(
            vllm_cls=vllm_cls,
            inventory=inventory,
            toolbar_information=toolbar_information,
            target_item=target_item,
            item_kind="sow",
        )
        selected_seed = bool(vllm_cls._selected_item_is_seed(selected_item_name))
        if skill_name == "use":
            if selected_seed:
                fallback_action = ""
                if target_obj and vllm_cls._is_valid_placeable_target(state, direction, selected_item_name, target_obj):
                    fallback_action = f'interact(direction="{direction}")'
                return {
                    "is_valid": False,
                    "invalid_reason": "cultivation_validation:sow_requires_interact",
                    "failure_root_cause": "stale_subtask",
                    "required_change_type": "rebuild_subtask",
                    "fallback_action": fallback_action,
                }
            return {
                "is_valid": False,
                "invalid_reason": "cultivation_validation:sow_requires_seed_selected",
                "failure_root_cause": "stale_subtask" if has_seed_available else "item_missing",
                "required_change_type": "change_selected_item" if has_seed_available else "switch_to_retrieval_subtask",
                "fallback_action": f"choose_item(slot_index={seed_slot_index})" if seed_slot_index is not None else "",
            }
        if skill_name != "interact":
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}
        if not selected_seed:
            return {
                "is_valid": False,
                "invalid_reason": "cultivation_validation:sow_requires_seed_selected",
                "failure_root_cause": "stale_subtask" if has_seed_available else "item_missing",
                "required_change_type": "change_selected_item" if has_seed_available else "switch_to_retrieval_subtask",
                "fallback_action": f"choose_item(slot_index={seed_slot_index})" if seed_slot_index is not None else "",
            }
        if not target_obj:
            return {
                "is_valid": False,
                "invalid_reason": "cultivation_validation:sow_requires_grounded_target",
                "failure_root_cause": "wrong_tile_alignment",
                "required_change_type": "change_position",
            }
        if vllm_cls._is_valid_placeable_target(state, direction, selected_item_name, target_obj):
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}
        alternative_directions = vllm_cls._collect_valid_placeable_directions(
            game_state=state,
            item_name=selected_item_name,
            invalid_direction=direction,
        )
        return {
            "is_valid": False,
            "invalid_reason": "cultivation_validation:sow_invalid_target",
            "failure_root_cause": "wrong_facing_direction" if alternative_directions else "invalid_target_tile",
            "required_change_type": "change_facing" if alternative_directions else "change_target_tile",
        }

    if task_kind == "water":
        if skill_name != "use":
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}
        toolbar_information = (
            state.get("toolbar_information")
            or prompt_facts.get("toolbar_information")
            or ""
        )
        inventory = (
            prompt_facts.get("inventory")
            or state.get("inventory")
            or []
        )
        selected_tool_name = vllm_cls._normalize_tool_name(selected_item_name)
        has_watering_can_available = (
            vllm_cls._find_tool_slot(inventory, "Watering Can") is not None
            or vllm_cls._find_tool_slot_in_toolbar_text(toolbar_information, "Watering Can") is not None
        )
        if selected_tool_name != "Watering Can":
            return {
                "is_valid": False,
                "invalid_reason": "cultivation_validation:water_requires_watering_can",
                "failure_root_cause": "stale_subtask" if has_watering_can_available else "item_missing",
                "required_change_type": "change_selected_item" if has_watering_can_available else "switch_to_retrieval_subtask",
            }
        if not target_obj:
            return {
                "is_valid": False,
                "invalid_reason": "cultivation_validation:water_requires_grounded_target",
                "failure_root_cause": "wrong_tile_alignment",
                "required_change_type": "change_position",
            }
        if vllm_cls._is_valid_watering_target(state, direction, target_obj):
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}
        alternative_direction = vllm_cls._find_alternative_tool_use_direction(
            game_state=state,
            tool_name="Watering Can",
            invalid_direction=direction,
        )
        return {
            "is_valid": False,
            "invalid_reason": "cultivation_validation:water_invalid_target",
            "failure_root_cause": "wrong_facing_direction" if alternative_direction else "invalid_target_tile",
            "required_change_type": "change_facing" if alternative_direction else "change_target_tile",
        }

    if task_kind == "harvest":
        if skill_name == "use":
            if target_obj and vllm_cls._target_text_is_ready_to_harvest(target_obj):
                return {
                    "is_valid": False,
                    "invalid_reason": "cultivation_validation:harvest_requires_interact",
                    "failure_root_cause": "stale_subtask",
                    "required_change_type": "rebuild_subtask",
                    "fallback_action": f'interact(direction="{direction}")',
                }
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}
        if skill_name != "interact":
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}
        if animal_harvest_task:
            # Animal product harvest (milk/egg/wool) does not expose crop-style grounded targets.
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}
        if target_obj and vllm_cls._target_text_is_ready_to_harvest(target_obj):
            return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}
        ready_cells = vllm_cls._nearby_ready_harvest_crop_cells(state)
        adjacent_ready = [cell for cell in ready_cells if abs(cell[0]) + abs(cell[1]) == 1]
        if adjacent_ready:
            fallback_direction = vllm_cls._adjacent_cell_to_direction(adjacent_ready[0])
            if fallback_direction:
                return {
                    "is_valid": False,
                    "invalid_reason": "cultivation_validation:harvest_wrong_facing",
                    "failure_root_cause": "wrong_facing_direction",
                    "required_change_type": "change_facing",
                    "fallback_action": f'interact(direction="{fallback_direction}")',
                }
        return {
            "is_valid": False,
            "invalid_reason": "cultivation_validation:harvest_requires_grounded_target",
            "failure_root_cause": "wrong_tile_alignment",
            "required_change_type": "change_position",
            "fallback_action": _build_harvest_grounding_move_fallback(
                state=state,
                vllm_cls=vllm_cls,
            ),
        }

    return {"is_valid": True, "invalid_reason": "", "failure_root_cause": "", "required_change_type": ""}


def validate_runtime_pre_execution_action(
    *,
    state: Optional[Dict[str, Any]],
    action_text: Any,
) -> Dict[str, Any]:
    state = state if isinstance(state, dict) else {}
    action = str(action_text or "").strip()
    if not action:
        return {
            "is_valid": True,
            "invalid_reason": "",
            "failure_root_cause": "",
            "required_change_type": "",
        }

    move_named = re.match(
        r"^move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)$",
        action,
        re.IGNORECASE,
    )
    if move_named:
        if int(move_named.group(1)) == 0 and int(move_named.group(2)) == 0:
            return {
                "is_valid": False,
                "invalid_reason": "runtime_validation:zero_move_invalid",
                "failure_root_cause": "stale_subtask",
                "required_change_type": "rebuild_subtask",
            }

    task_text = str(state.get("main_task", "") or state.get("task", "") or "").strip().lower()
    prompt_profile = str(state.get("prompt_profile", "") or "").strip().lower()
    vllm_cls, _ = _load_fastllm_runtime_classes()
    prompt_facts = extract_stardew_prompt_fact_fields(
        state=state,
        gathered_info=state.get("gathered_info", {}),
    )

    if prompt_profile in {"", "none"}:
        if task_text.startswith(("go_to_", "forage ", "go to ", "sleep", "go_to_bed")):
            prompt_profile = "navigation"
        elif task_text.startswith("kill "):
            prompt_profile = "combat"
        elif task_text.startswith(("craft ", "craft_", "cook ", "produce ")):
            prompt_profile = "crafting"

    cultivation_validation = validate_cultivation_pre_execution_action(
        state=state,
        action_text=action,
    )
    if not cultivation_validation.get("is_valid", True):
        return cultivation_validation

    if vllm_cls is None:
        return {
            "is_valid": True,
            "invalid_reason": "",
            "failure_root_cause": "",
            "required_change_type": "",
        }

    current_menu_type = vllm_cls._normalize_menu_type(prompt_facts.get("current_menu"))
    location_text = re.sub(
        r"[^a-z0-9]+",
        " ",
        str(
            prompt_facts.get("location")
            or state.get("location")
            or state.get("gathered_info", {}).get("location", "")
            or ""
        ).lower(),
    ).strip()
    ship_task_inside_farmhouse = bool(
        task_text.startswith(("ship ", "ship_"))
        and any(token in location_text for token in ("farmhouse", "house", "home"))
    )
    blocking_menu_profiles = {
        "navigation",
        "combat",
        "crafting",
        "farm_clearup",
        "farm_ops",
    }
    if current_menu_type in {"dialoguebox", "objectdialogue", "notificationdialogue"}:
        fallback_action = _build_menu_blocker_fallback_action(
            current_menu=prompt_facts.get("current_menu"),
            current_menu_type=current_menu_type,
            vllm_cls=vllm_cls,
        )
        if action == fallback_action:
            return {
                "is_valid": True,
                "invalid_reason": "",
                "failure_root_cause": "",
                "required_change_type": "",
            }
        if prompt_profile in blocking_menu_profiles or not action.startswith("choose_option("):
            return {
                "is_valid": False,
                "invalid_reason": f"runtime_validation:menu_open:{current_menu_type}",
                "failure_root_cause": "menu_stuck",
                "required_change_type": "close_menu",
                "fallback_action": fallback_action,
            }

    selected_item_name = str(prompt_facts.get("selected_item_name", "") or "").strip()
    toolbar_information = (
        state.get("toolbar_information")
        or prompt_facts.get("toolbar_information")
        or ""
    )
    inventory = (
        prompt_facts.get("inventory")
        or state.get("inventory")
        or []
    )

    directional_skill = vllm_cls._parse_directional_skill(action)

    selected_tool_for_hay = str(
        vllm_cls._normalize_tool_name(
            selected_item_name
            or vllm_cls._extract_selected_item_name_from_toolbar(toolbar_information)
        )
        or ""
    ).strip()
    if (
        action.startswith("move(")
        and selected_tool_for_hay == "Scythe"
        and _is_hay_with_scythe_forage_task(
            task_text=task_text,
            state=state,
            prompt_facts=prompt_facts,
        )
    ):
        hay_scythe_direction = _find_adjacent_hay_scythe_target_direction(
            vllm_cls=vllm_cls,
            state=state,
        )
        if hay_scythe_direction:
            return {
                "is_valid": False,
                "invalid_reason": "runtime_validation:hay_scythe_adjacent_target_requires_use",
                "failure_root_cause": "stale_subtask",
                "required_change_type": "rebuild_subtask",
                "fallback_action": f'use(direction="{hay_scythe_direction}")',
            }

    if prompt_profile == "combat":
        selected_tool_name = str(
            vllm_cls._normalize_tool_name(selected_item_name) or selected_item_name or ""
        ).strip()
        adjacent_attack_direction = _find_adjacent_combat_attack_direction(
            state=state,
            prompt_facts=prompt_facts,
            vllm_cls=vllm_cls,
        )
        if action.startswith("interact("):
            return {
                "is_valid": False,
                "invalid_reason": "runtime_validation:combat_interact_invalid",
                "failure_root_cause": "stale_subtask",
                "required_change_type": "rebuild_subtask",
            }
        if (
            adjacent_attack_direction
            and selected_tool_name == "Rusty Sword"
            and action.startswith("move(")
        ):
            return {
                "is_valid": False,
                "invalid_reason": "runtime_validation:combat_adjacent_enemy_requires_attack",
                "failure_root_cause": "stale_subtask",
                "required_change_type": "rebuild_subtask",
                "fallback_action": f'use(direction="{adjacent_attack_direction}")',
            }
        if directional_skill and directional_skill[0] == "use":
            _skill_name, direction = directional_skill
            target_obj, required_tool = vllm_cls._get_directional_target(state, direction)
            if selected_tool_name != "Rusty Sword":
                sword_slot = vllm_cls._find_tool_slot(inventory, "Rusty Sword")
                if sword_slot is None:
                    sword_slot = vllm_cls._find_tool_slot_in_toolbar_text(toolbar_information, "Rusty Sword")
                return {
                    "is_valid": False,
                    "invalid_reason": "runtime_validation:combat_requires_rusty_sword",
                    "failure_root_cause": "stale_subtask" if sword_slot is not None else "item_missing",
                    "required_change_type": "change_selected_item" if sword_slot is not None else "switch_to_retrieval_subtask",
                }
            if required_tool and required_tool != "Rusty Sword":
                return {
                    "is_valid": False,
                    "invalid_reason": "runtime_validation:combat_wrong_target_type",
                    "failure_root_cause": "stale_subtask",
                    "required_change_type": "rebuild_subtask",
                }
            if not target_obj:
                return {
                    "is_valid": False,
                    "invalid_reason": "runtime_validation:combat_use_without_enemy_target",
                    "failure_root_cause": "wrong_tile_alignment",
                    "required_change_type": "change_position",
                }

    if prompt_profile == "crafting":
        craft_target = _resolve_task_target_item(state=state, prompt_facts=prompt_facts)
        craft_recipe_target = _canonical_crafting_recipe_name(craft_target) if craft_target else ""
        craft_target_action = f'craft(item="{craft_recipe_target}")' if craft_recipe_target else ""
        craft_target_is_direct_craft = bool(task_text.startswith(("craft ", "craft_")) and craft_target_action)
        force_direct_craft_rewrite = _should_force_direct_craft_rewrite(
            state=state,
            craft_target_action=craft_target_action,
        )
        inventory_text = " ".join(str(item or "") for item in inventory)
        normalized_inventory = re.sub(r"[^a-z0-9]+", " ", inventory_text.lower()).strip()
        craft_target_lower = re.sub(r"[^a-z0-9]+", " ", craft_target.lower()).strip()
        missing_materials = (
            _crafting_recipe_missing_materials(craft_target, inventory)
            if craft_target_is_direct_craft
            else None
        )
        if (
            craft_target_is_direct_craft
            and force_direct_craft_rewrite
            and not action.startswith("craft(")
        ):
            if not missing_materials and not (craft_target_lower and craft_target_lower in normalized_inventory):
                return {
                    "is_valid": False,
                    "invalid_reason": "runtime_validation:crafting_should_direct_craft",
                    "failure_root_cause": "stale_subtask",
                    "required_change_type": "rebuild_subtask",
                    "fallback_action": craft_target_action,
                }
        materials_missing_for_direct_craft = bool(
            craft_target_is_direct_craft and isinstance(missing_materials, list) and len(missing_materials) > 0
        )
        if action.startswith("use(") or action.startswith("interact("):
            if materials_missing_for_direct_craft:
                return {
                    "is_valid": True,
                    "invalid_reason": "",
                    "failure_root_cause": "",
                    "required_change_type": "",
                }
            return {
                "is_valid": False,
                "invalid_reason": "runtime_validation:crafting_world_action_invalid",
                "failure_root_cause": "stale_subtask",
                "required_change_type": "rebuild_subtask",
            }
        if action.startswith("menu("):
            parsed_menu = vllm_cls._parse_menu_action(action)
            current_menu_text = str(prompt_facts.get("current_menu", "") or "").strip().lower()
            if materials_missing_for_direct_craft and parsed_menu and parsed_menu[0] == "open":
                missing_label = ", ".join(missing_materials[:3])
                return {
                    "is_valid": False,
                    "invalid_reason": f"runtime_validation:craft_missing_materials:{missing_label}",
                    "failure_root_cause": "item_missing",
                    "required_change_type": "switch_to_retrieval_subtask",
                }
            if parsed_menu == ("open", "crafting") and (
                current_menu_type == "crafting" or "craft" in current_menu_text
            ):
                return {
                    "is_valid": False,
                    "invalid_reason": "runtime_validation:crafting_menu_already_open",
                    "failure_root_cause": "menu_stuck",
                    "required_change_type": "rebuild_subtask",
                    "fallback_action": craft_target_action if craft_target_is_direct_craft else "",
                }
        craft_match = re.match(
            r'^craft\(\s*item\s*=\s*"([^"]+)"\s*\)$',
            action,
            re.IGNORECASE,
        )
        if craft_match:
            raw_craft_item = craft_match.group(1).strip()
            craft_item = _canonical_crafting_recipe_name(raw_craft_item)
            try:
                from env.actions import _crafting_recipes
            except Exception:
                _crafting_recipes = {}
            recipe_table = {}
            if isinstance(_crafting_recipes, dict):
                recipe_table = _crafting_recipes.get("content", _crafting_recipes)
            if isinstance(recipe_table, dict) and craft_item not in recipe_table:
                return {
                    "is_valid": False,
                    "invalid_reason": "runtime_validation:unknown_crafting_recipe",
                    "failure_root_cause": "stale_subtask",
                    "required_change_type": "rebuild_subtask",
                }
            recipe_missing_materials = _crafting_recipe_missing_materials(craft_item, inventory)
            if recipe_missing_materials:
                missing_label = ", ".join(recipe_missing_materials[:3])
                return {
                    "is_valid": False,
                    "invalid_reason": f"runtime_validation:craft_missing_materials:{missing_label}",
                    "failure_root_cause": "item_missing",
                    "required_change_type": "switch_to_retrieval_subtask",
                }
            if craft_item != raw_craft_item:
                return {
                    "is_valid": False,
                    "invalid_reason": "runtime_validation:craft_alias_requires_recipe_name",
                    "failure_root_cause": "stale_subtask",
                    "required_change_type": "rebuild_subtask",
                    "fallback_action": f'craft(item="{craft_item}")',
                }
            craft_lower = re.sub(r"[^a-z0-9]+", " ", craft_item.lower()).strip()
            if craft_lower and craft_lower in normalized_inventory:
                return {
                    "is_valid": False,
                    "invalid_reason": "runtime_validation:craft_target_already_present",
                    "failure_root_cause": "stale_subtask",
                    "required_change_type": "rebuild_subtask",
                }

    if prompt_profile == "navigation":
        forage_navigation_task = task_text.startswith(("forage ", "forage_"))
        if forage_navigation_task:
            target_item = _resolve_task_target_item(state=state, prompt_facts=prompt_facts)
            target_labels = _candidate_target_labels(target_item)
            move_components = vllm_cls._parse_move_components(action)
            nearby_target_cell = _find_nearby_named_target_cell(
                vllm_cls=vllm_cls,
                state=state,
                target_labels=target_labels,
            )
            if move_components is not None and nearby_target_cell is not None:
                distance = abs(nearby_target_cell[0]) + abs(nearby_target_cell[1])
                if distance == 1:
                    fallback_direction = vllm_cls._adjacent_cell_to_direction(nearby_target_cell)
                    if fallback_direction:
                        return {
                            "is_valid": False,
                            "invalid_reason": "runtime_validation:forage_adjacent_target_requires_interact",
                            "failure_root_cause": "stale_subtask",
                            "required_change_type": "rebuild_subtask",
                            "fallback_action": f'interact(direction="{fallback_direction}")',
                        }
                fallback_move = str(
                    getattr(vllm_cls, "_build_step_toward_cell_move")(
                        nearby_target_cell,
                        state,
                        max_stride=1,
                    )
                    or ""
                ).strip()
                if (
                    fallback_move
                    and fallback_move != action
                    and not _move_reduces_named_target_offset(
                        move_components=move_components,
                        target_offset=nearby_target_cell,
                    )
                ):
                    return {
                        "is_valid": False,
                        "invalid_reason": "runtime_validation:forage_nearby_target_requires_local_alignment",
                        "failure_root_cause": "stale_subtask",
                        "required_change_type": "change_position",
                        "fallback_action": fallback_move,
                    }

            if move_components is not None:
                image_target_offset = _infer_named_target_offset_from_image_description(
                    state=state,
                    target_labels=target_labels,
                )
                if image_target_offset is not None:
                    fallback_move = str(
                        getattr(vllm_cls, "_build_step_toward_cell_move")(
                            image_target_offset,
                            state,
                            max_stride=1,
                        )
                        or ""
                    ).strip()
                    if (
                        fallback_move
                        and fallback_move != action
                        and not _move_reduces_named_target_offset(
                            move_components=move_components,
                            target_offset=image_target_offset,
                        )
                    ):
                        return {
                            "is_valid": False,
                            "invalid_reason": "runtime_validation:forage_visible_target_requires_local_alignment",
                            "failure_root_cause": "stale_subtask",
                            "required_change_type": "change_position",
                            "fallback_action": fallback_move,
                        }

        if (
            task_text.startswith("go_to_bed")
            and current_menu_type == "dialoguebox"
            and vllm_cls._menu_contains_sleep_prompt(prompt_facts.get("current_menu"))
            and action != "choose_option(option_index=1, quantity=0)"
        ):
            return {
                "is_valid": False,
                "invalid_reason": "runtime_validation:sleep_dialogue_requires_confirm",
                "failure_root_cause": "stale_subtask",
                "required_change_type": "rebuild_subtask",
                "fallback_action": "choose_option(option_index=1, quantity=0)",
            }
        move_components = vllm_cls._parse_move_components(action)
        adjacent_door_interact = _find_adjacent_door_interact_fallback(
            state=state,
            vllm_cls=vllm_cls,
        )
        sleep_navigation_task = task_text.startswith(("go_to_bed", "sleep", "go to bed"))
        if sleep_navigation_task and move_components is not None and adjacent_door_interact:
            interact_match = re.match(
                r'^interact\(\s*direction\s*=\s*"(up|down|left|right)"\s*\)$',
                adjacent_door_interact,
                re.IGNORECASE,
            )
            if interact_match:
                door_direction = interact_match.group(1).lower()
                direction_to_relative = getattr(vllm_cls, "_direction_to_relative", None)
                door_relative = (
                    direction_to_relative(door_direction)
                    if callable(direction_to_relative)
                    else None
                )
                facing_direction = str(
                    prompt_facts.get("facing_direction")
                    or state.get("facing_direction")
                    or state.get("gathered_info", {}).get("facing_direction", "")
                    or ""
                ).strip().lower()
                move_distance = abs(int(move_components[0])) + abs(int(move_components[1]))
                last_errors_info = str(state.get("last_errors_info", "") or "").strip().lower()
                last_action_text = str(
                    state.get("last_action", "")
                    or state.get("action", "")
                    or ""
                ).strip()
                blocked_same_move = (
                    "path is likely blocked" in last_errors_info
                    and last_action_text == action
                )
                if (
                    move_distance > 1
                    or facing_direction == door_direction
                    or (
                        door_relative is not None
                        and tuple(move_components) == tuple(door_relative)
                        and blocked_same_move
                    )
                ):
                    return {
                        "is_valid": False,
                        "invalid_reason": "runtime_validation:navigation_adjacent_door_requires_interact",
                        "failure_root_cause": "stale_subtask",
                        "required_change_type": "rebuild_subtask",
                        "fallback_action": adjacent_door_interact,
                    }
        description_text = str(
            prompt_facts.get("description")
            or state.get("gathered_info", {}).get("description", "")
            or state.get("description", "")
            or ""
        ).strip().lower()
        if (
            move_components is not None
            and "porch" in description_text
            and move_components[1] <= 0
            and abs(move_components[0]) >= 1
        ):
            down_target, _ = vllm_cls._get_directional_target(state, "down")
            if getattr(vllm_cls, "_is_open_ground_tile")(down_target):
                return {
                    "is_valid": False,
                    "invalid_reason": "runtime_validation:navigation_leave_porch_first",
                    "failure_root_cause": "wrong_tile_alignment",
                    "required_change_type": "change_position",
                    "fallback_action": "move(x=0, y=1)",
                }
        if move_components is not None:
            best_waypoint = None
            best_waypoint_fn = getattr(vllm_cls, "_best_route_waypoint_candidate", None)
            if callable(best_waypoint_fn):
                best_waypoint = best_waypoint_fn(state)
            move_distance = abs(int(move_components[0])) + abs(int(move_components[1]))
            if best_waypoint and move_distance > 1:
                waypoint_offset = tuple(best_waypoint.get("offset", (0, 0)))
                waypoint_text = str(best_waypoint.get("raw", "") or best_waypoint.get("name", "") or "").strip().lower()
                near_waypoint = abs(int(waypoint_offset[0])) + abs(int(waypoint_offset[1])) <= 6
                repeated_blocking = (
                    _safe_int(state.get("zero_progress_streak", 0) or 0, default=0) >= 1
                    or _safe_int(state.get("repeated_action_streak", 0) or 0, default=0) >= 1
                )
                if (
                    best_waypoint.get("source") == "exits"
                    or any(token in waypoint_text for token in ("exit", "entrance", "door"))
                ) and (near_waypoint or repeated_blocking):
                    unit_step = str(
                        getattr(vllm_cls, "_build_step_toward_cell_move")(
                            best_waypoint["offset"],
                            state,
                            max_stride=1,
                        )
                        or ""
                    ).strip()
                    if unit_step and unit_step != action:
                        return {
                            "is_valid": False,
                            "invalid_reason": "runtime_validation:navigation_requires_unit_step",
                            "failure_root_cause": "wrong_tile_alignment",
                            "required_change_type": "change_position",
                            "fallback_action": unit_step,
                        }
            blocker_index, blocker = vllm_cls._get_single_axis_path_blocker(action, state)
            blocker_text = str(blocker or "").strip()
            if blocker_index > 0 and blocker_text:
                preserve_anchor_fn = getattr(vllm_cls, "_should_preserve_navigation_anchor_move", None)
                preserve_anchor = bool(
                    callable(preserve_anchor_fn)
                    and preserve_anchor_fn(
                        action_text=action,
                        game_state=state,
                        blocker=blocker_text,
                    )
                )
                if not preserve_anchor:
                    reroute = vllm_cls._build_structure_blocked_move_recovery(
                        action_text=action,
                        game_state=state,
                        blocker=blocker_text,
                    )
                    if reroute and reroute != action:
                        safe_reroute = _build_unit_priority_navigation_reroute(
                            vllm_cls=vllm_cls,
                            state=state,
                            action=action,
                            reroute=reroute,
                        )
                        return {
                            "is_valid": False,
                            "invalid_reason": "runtime_validation:navigation_blocked_route",
                            "failure_root_cause": "wrong_tile_alignment",
                            "required_change_type": "change_position",
                            "fallback_action": safe_reroute,
                        }
            elif move_distance > 3:
                short_probe = str(
                    getattr(vllm_cls, "_build_step_toward_cell_move")(
                        (int(move_components[0]), int(move_components[1])),
                        state,
                        max_stride=3,
                    )
                    or ""
                ).strip()
                if short_probe and short_probe != action:
                    return {
                        "is_valid": False,
                        "invalid_reason": "runtime_validation:navigation_requires_short_probe",
                        "failure_root_cause": "wrong_tile_alignment",
                        "required_change_type": "change_position",
                        "fallback_action": short_probe,
                    }
        if action.startswith("craft("):
            return {
                "is_valid": False,
                "invalid_reason": "runtime_validation:navigation_craft_invalid",
                "failure_root_cause": "stale_subtask",
                "required_change_type": "rebuild_subtask",
            }
        if task_text.startswith("go_to_coop") and action.startswith("interact("):
            target_obj, _ = vllm_cls._get_directional_target(
                state,
                directional_skill[1] if directional_skill else "up",
            ) if directional_skill else ("", "")
            if "door" not in str(target_obj or "").lower() and "coop" not in str(target_obj or "").lower():
                return {
                    "is_valid": False,
                    "invalid_reason": "runtime_validation:navigation_interact_without_door",
                    "failure_root_cause": "wrong_tile_alignment",
                    "required_change_type": "change_position",
                }

    if prompt_profile == "farm_ops":
        move_components = vllm_cls._parse_move_components(action)
        if move_components is not None:
            best_waypoint = None
            best_waypoint_fn = getattr(vllm_cls, "_best_route_waypoint_candidate", None)
            if callable(best_waypoint_fn):
                best_waypoint = best_waypoint_fn(state)

            if best_waypoint:
                route_fallback = str(
                    getattr(vllm_cls, "_build_step_toward_cell_move")(
                        best_waypoint["offset"],
                        state,
                        max_stride=3,
                    )
                    or ""
                ).strip()
                if (
                    route_fallback
                    and route_fallback != action
                    and not getattr(vllm_cls, "_move_reduces_waypoint_distance")(action, best_waypoint)
                ):
                    return {
                        "is_valid": False,
                        "invalid_reason": "runtime_validation:farm_ops_route_conflicts_with_waypoint",
                        "failure_root_cause": "wrong_tile_alignment",
                        "required_change_type": "change_position",
                        "fallback_action": route_fallback,
                    }

    if (
        prompt_profile in {"shopping", "social"}
        and current_menu_type in {"", "no menu", "none", "null"}
        and not ship_task_inside_farmhouse
    ):
        move_components = vllm_cls._parse_move_components(action)
        if move_components is not None and abs(int(move_components[0])) + abs(int(move_components[1])) <= 2:
            interact_fallback = _build_adjacent_service_interact_fallback(
                state=state,
                prompt_facts=prompt_facts,
                vllm_cls=vllm_cls,
            )
            if interact_fallback and interact_fallback != action:
                return {
                    "is_valid": False,
                    "invalid_reason": "runtime_validation:service_move_while_aligned",
                    "failure_root_cause": "wrong_tile_alignment",
                    "required_change_type": "change_position",
                    "fallback_action": interact_fallback,
                }

    if prompt_profile in {"shopping", "social"}:
        choose_option = _parse_choose_option_action(action)
        current_menu_text = str(prompt_facts.get("current_menu", "") or "").strip().lower()
        if choose_option and current_menu_type in {"", "no menu", "none", "null"} and not ship_task_inside_farmhouse:
            service_fallback = _build_adjacent_service_interact_fallback(
                state=state,
                prompt_facts=prompt_facts,
                vllm_cls=vllm_cls,
            )
            if service_fallback:
                return {
                    "is_valid": False,
                    "invalid_reason": "runtime_validation:service_menu_requires_context",
                    "failure_root_cause": "wrong_tile_alignment" if service_fallback.startswith("move(") else "stale_subtask",
                    "required_change_type": "change_position" if service_fallback.startswith("move(") else "rebuild_subtask",
                    "fallback_action": service_fallback,
                }
            return {
                "is_valid": False,
                "invalid_reason": "runtime_validation:service_menu_requires_context",
                "failure_root_cause": "stale_subtask",
                "required_change_type": "rebuild_subtask",
            }
        if choose_option and choose_option.get("direction") == "out" and (
            current_menu_type not in {"", "no menu", "none", "null"}
            or task_text.startswith(("sell ", "sell_"))
        ):
            selected_slot = prompt_facts.get("selected_position")
            if selected_slot is None:
                selected_slot = state.get("selected_position")
            target_item = (
                state.get("target_item")
                or prompt_facts.get("target_item")
                or selected_item_name
            )
            if selected_slot is None:
                selected_slot = _find_visible_inventory_slot_for_target(
                    vllm_cls=vllm_cls,
                    inventory=inventory,
                    toolbar_information=toolbar_information,
                    target_item=target_item,
                )
            try:
                expected_option_index = int(selected_slot) + 1 if selected_slot is not None else None
            except (TypeError, ValueError):
                expected_option_index = None
            if expected_option_index is not None and choose_option.get("option_index") != expected_option_index:
                fallback_quantity = choose_option.get("quantity")
                if fallback_quantity is None:
                    fallback_quantity = 0
                return {
                    "is_valid": False,
                    "invalid_reason": "runtime_validation:shop_sell_wrong_slot",
                    "failure_root_cause": "stale_subtask",
                    "required_change_type": "rebuild_subtask",
                    "fallback_action": (
                        f'choose_option(option_index={expected_option_index}, '
                        f'quantity={int(fallback_quantity)}, direction="out")'
                    ),
                }

    return {
        "is_valid": True,
        "invalid_reason": "",
        "failure_root_cause": "",
        "required_change_type": "",
    }


def _load_fastllm_runtime_classes():
    try:
        from cradle.runner.vllm_client import VLLMClient, VLLMDecision
    except Exception:
        return None, None
    return VLLMClient, VLLMDecision


def _load_object_name_to_id() -> Dict[str, str]:
    global _OBJECT_NAME_TO_ID
    if _OBJECT_NAME_TO_ID is not None:
        return _OBJECT_NAME_TO_ID

    mapping: Dict[str, str] = {}
    try:
        objects_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "env", "game_data", "Objects.json")
        )
        with open(objects_path, "r", encoding="utf-8") as fp:
            raw = json.load(fp) or {}
        content = raw.get("content", raw)
        if isinstance(content, dict):
            for object_id, object_data in content.items():
                if not isinstance(object_data, dict):
                    continue
                object_name = str(object_data.get("Name", "") or "").strip()
                if not object_name:
                    continue
                normalized = re.sub(r"[^a-z0-9]+", " ", object_name.lower()).strip()
                if normalized:
                    mapping.setdefault(normalized, str(object_id))
    except Exception:
        mapping = {}

    _OBJECT_NAME_TO_ID = mapping
    return mapping


def _load_object_id_to_name() -> Dict[str, str]:
    global _OBJECT_ID_TO_NAME
    if _OBJECT_ID_TO_NAME is not None:
        return _OBJECT_ID_TO_NAME

    mapping: Dict[str, str] = {}
    try:
        objects_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "env", "game_data", "Objects.json")
        )
        with open(objects_path, "r", encoding="utf-8") as fp:
            raw = json.load(fp) or {}
        content = raw.get("content", raw)
        if isinstance(content, dict):
            for object_id, object_data in content.items():
                if not isinstance(object_data, dict):
                    continue
                object_name = str(object_data.get("Name", "") or "").strip()
                normalized = re.sub(r"[^a-z0-9]+", " ", object_name.lower()).strip()
                if normalized:
                    mapping[str(object_id)] = normalized
    except Exception:
        mapping = {}

    _OBJECT_ID_TO_NAME = mapping
    return mapping


def _load_crafting_recipe_table() -> Dict[str, str]:
    global _CRAFTING_RECIPE_TABLE
    if _CRAFTING_RECIPE_TABLE is not None:
        return _CRAFTING_RECIPE_TABLE

    recipes: Dict[str, str] = {}
    try:
        recipes_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "env", "game_data", "CraftingRecipes.json")
        )
        with open(recipes_path, "r", encoding="utf-8") as fp:
            raw = json.load(fp) or {}
        content = raw.get("content", raw)
        if isinstance(content, dict):
            recipes = {
                str(recipe_name).strip(): str(recipe_spec).strip()
                for recipe_name, recipe_spec in content.items()
                if str(recipe_name).strip() and str(recipe_spec).strip()
            }
    except Exception:
        recipes = {}

    _CRAFTING_RECIPE_TABLE = recipes
    return recipes


def _normalize_crafting_recipe_lookup(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _canonical_crafting_recipe_name(item_name: Any) -> str:
    recipe_name = str(item_name or "").strip()
    if not recipe_name:
        return ""

    recipes = _load_crafting_recipe_table()
    if recipe_name in recipes:
        return recipe_name

    normalized = _normalize_crafting_recipe_lookup(recipe_name)
    alias = _CRAFTING_RECIPE_ALIASES.get(normalized)
    if alias and alias in recipes:
        return alias

    for existing_name in recipes:
        if _normalize_crafting_recipe_lookup(existing_name) == normalized:
            return existing_name

    return recipe_name


def _inventory_name_to_count(inventory: Any) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not isinstance(inventory, list):
        return counts

    pattern = re.compile(r"slot_index\s+\d+:\s*(.+?)(?:\s+\(quantity:\s*([^)]+)\))?$", re.IGNORECASE)
    for item in inventory:
        text = str(item or "").strip()
        if not text:
            continue
        match = pattern.match(text)
        if match:
            item_name = match.group(1).strip()
            if item_name.lower() == "no item":
                continue
            quantity_raw = str(match.group(2) or "1").strip()
            try:
                quantity = int(float(quantity_raw))
            except (TypeError, ValueError):
                quantity = 1
        else:
            item_name = text
            quantity = 1

        normalized = re.sub(r"[^a-z0-9]+", " ", item_name.lower()).strip()
        if not normalized:
            continue
        counts[normalized] = counts.get(normalized, 0) + max(quantity, 0)

    return counts


def _crafting_recipe_missing_materials(item_name: str, inventory: Any) -> Optional[List[str]]:
    recipes = _load_crafting_recipe_table()
    if not recipes:
        return None

    recipe_spec = recipes.get(_canonical_crafting_recipe_name(item_name))
    if not recipe_spec:
        return None

    object_id_to_name = _load_object_id_to_name()
    inventory_counts = _inventory_name_to_count(inventory)

    ingredient_part = recipe_spec.split("/", 1)[0]
    tokens = [token for token in ingredient_part.split() if token]
    if len(tokens) < 2:
        return []

    missing: List[str] = []
    for idx in range(0, len(tokens) - 1, 2):
        ingredient_key = tokens[idx]
        try:
            needed_qty = int(float(tokens[idx + 1]))
        except (TypeError, ValueError):
            continue

        ingredient_name = ""
        if ingredient_key in object_id_to_name:
            ingredient_name = object_id_to_name.get(ingredient_key, "")
        else:
            ingredient_name = re.sub(r"[^a-z0-9]+", " ", ingredient_key.lower()).strip()

        if not ingredient_name:
            continue

        available_qty = inventory_counts.get(ingredient_name, 0)
        if available_qty < needed_qty:
            missing.append(ingredient_name)

    return missing


def _crafting_missing_material_source_hints(missing_materials: Any) -> List[Dict[str, Any]]:
    normalized_missing = {
        re.sub(r"[^a-z0-9]+", " ", str(material or "").lower()).strip()
        for material in (missing_materials or [])
        if str(material or "").strip()
    }
    hints: List[Dict[str, Any]] = []

    if "stone" in normalized_missing:
        hints.append({
            "material": "stone",
            "tokens": ("stone", "stones", "rock", "rocks", "boulder", "boulders"),
            "tool": "pickaxe",
        })
    if "wood" in normalized_missing:
        hints.append({
            "material": "wood",
            "tokens": ("twig", "twigs", "branch", "branches", "wood", "log", "logs", "stump", "stumps"),
            "tool": "axe",
        })
    if "fiber" in normalized_missing:
        hints.append({
            "material": "fiber",
            "tokens": ("weed", "weeds", "grass", "fiber"),
            "tool": "scythe",
        })
    if "clay" in normalized_missing:
        hints.append({
            "material": "clay",
            "tokens": ("artifact spot", "artifact spots", "worm", "worms"),
            "tool": "hoe",
        })

    return hints


def _crafting_gather_action_matches_missing_materials(
    *,
    action: str,
    state: Dict[str, Any],
    vllm_cls: Any,
    missing_materials: Any,
    selected_item_name: str,
) -> bool:
    directional_skill = vllm_cls._parse_directional_skill(action)
    if directional_skill is None:
        return False

    skill_name, direction = directional_skill
    target_obj, required_tool = vllm_cls._get_directional_target(state, direction)
    target_text = re.sub(r"[^a-z0-9]+", " ", str(target_obj or "").lower()).strip()
    normalized_selected_tool = str(vllm_cls._normalize_tool_name(selected_item_name) or "").strip().lower()
    normalized_required_tool = str(required_tool or "").strip().lower()

    for hint in _crafting_missing_material_source_hints(missing_materials):
        if not any(token in target_text for token in hint["tokens"]):
            continue
        if skill_name != "use":
            return False
        expected_tool = str(hint["tool"] or "").strip().lower()
        if normalized_required_tool and normalized_required_tool != expected_tool:
            return False
        return normalized_selected_tool == expected_tool

    return False


def _build_synthetic_execution_log_for_recovery(
    *,
    result_state: Dict[str, Any],
    blocked_reason: str,
    hint_action: str,
) -> List[Dict[str, Any]]:
    last_action = str(
        result_state.get("last_action", "")
        or result_state.get("pre_action", "")
        or hint_action
        or ""
    ).strip()
    last_errors = str(
        result_state.get("last_errors_info", "")
        or result_state.get("latest_execution_summary", "")
        or blocked_reason
        or ""
    ).strip()
    if not last_action and not last_errors:
        return []
    return [
        {
            "action": last_action or hint_action,
            "success": False,
            "errors_info": last_errors or blocked_reason,
        }
    ]


def _build_local_watering_recovery_action(
    *,
    result_state: Dict[str, Any],
    vllm_cls: Any,
    selected_item_name: str,
    inventory: Any,
    toolbar_information: Any,
) -> str:
    normalized_tool = str(vllm_cls._normalize_tool_name(selected_item_name) or "").strip()
    if not vllm_cls._is_watering_context(result_state):
        return ""

    surroundings_map, _ = vllm_cls._get_structured_surroundings_map(result_state)
    if not surroundings_map:
        return ""

    candidates = []
    for cell, raw_text in surroundings_map.items():
        if cell == (0, 0):
            continue
        text = str(raw_text or "").strip()
        if not text:
            continue
        if not (
            vllm_cls._placeable_target_is_tilled(text)
            or vllm_cls._target_text_contains_crop(text)
        ):
            continue
        candidates.append((vllm_cls._cell_sort_key(cell), cell))

    if not candidates:
        return ""

    _, cell = min(candidates, key=lambda item: item[0])
    direction = vllm_cls._adjacent_cell_to_direction(cell)
    if direction:
        if normalized_tool == "Watering Can":
            return f'use(direction="{direction}")'
        tool_slot = vllm_cls._find_tool_slot(inventory, "Watering Can")
        if tool_slot is None:
            tool_slot = vllm_cls._find_tool_slot_in_toolbar_text(
                toolbar_information,
                "Watering Can",
            )
        if tool_slot is not None:
            return f"choose_item(slot_index={tool_slot})"

    return vllm_cls._build_step_toward_cell_move(cell)


def _validate_runtime_candidate_action(
    *,
    result_state: Dict[str, Any],
    action: str,
    suggestion_action: str,
    decision_reason: str,
    vllm_cls: Any,
    vllm_decision_cls: Any,
) -> str:
    candidate = str(action or "").strip()
    if not candidate:
        return ""

    suggestion = {"action": suggestion_action or candidate, "reason": "runtime_recovery"}
    validated = vllm_cls._validate_decision_against_state(
        vllm_decision_cls(
            action=candidate,
            reason=decision_reason,
            escalate=False,
        ),
        suggestion,
        result_state,
    )
    if getattr(validated, "escalate", False):
        return ""
    return str(getattr(validated, "action", "") or "").strip()


def _alternate_local_recovery_action(
    primary_action: str,
    secondary_action: str,
    *,
    variation_seed: Any = 1,
) -> str:
    try:
        seed = int(variation_seed)
    except (TypeError, ValueError):
        seed = 1
    if seed <= 0:
        seed = 1
    return primary_action if seed % 2 == 1 else secondary_action


def _build_minimal_local_recovery_action(
    stuck_action: Any,
    *,
    variation_seed: Any = 1,
) -> str:
    action_text = str(stuck_action or "").strip()
    if not action_text:
        return ""

    menu_open_match = re.match(
        r'menu\(\s*option\s*=\s*"open"\s*,\s*menu_name\s*=\s*"([^"]+)"\s*\)',
        action_text,
        re.IGNORECASE,
    )
    if menu_open_match:
        return f'menu(option="close", menu_name="{menu_open_match.group(1)}")'

    directional_match = re.match(
        r'(?:use|interact)\(\s*direction\s*=\s*"(up|down|left|right)"\s*\)',
        action_text,
        re.IGNORECASE,
    )
    if directional_match:
        direction = directional_match.group(1).lower()
        if direction in {"up", "down"}:
            return _alternate_local_recovery_action(
                "move(x=1, y=0)",
                "move(x=-1, y=0)",
                variation_seed=variation_seed,
            )
        return _alternate_local_recovery_action(
            "move(x=0, y=1)",
            "move(x=0, y=-1)",
            variation_seed=variation_seed,
        )

    if action_text.startswith(("choose_item(", "attach_item(", "unattach_item(", "choose_option(", "menu(")):
        return _alternate_local_recovery_action(
            "move(x=1, y=0)",
            "move(x=-1, y=0)",
            variation_seed=variation_seed,
        )

    move_match = re.match(
        r'^move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)$',
        action_text,
        re.IGNORECASE,
    )
    if move_match:
        x_val = int(move_match.group(1))
        y_val = int(move_match.group(2))
        if abs(x_val) >= abs(y_val) and x_val != 0:
            return _alternate_local_recovery_action(
                "move(x=0, y=1)",
                "move(x=0, y=-1)",
                variation_seed=variation_seed,
            )
        if y_val != 0:
            return _alternate_local_recovery_action(
                "move(x=1, y=0)",
                "move(x=-1, y=0)",
                variation_seed=variation_seed,
            )

    return ""


def build_runtime_local_recovery_action(
    *,
    result_state: Dict[str, Any],
    suggestion_action: str = "",
    failed_action: str = "",
    decision_reason: str = "runtime_local_recovery",
    variation_seed: Any = 1,
) -> str:
    vllm_cls, vllm_decision_cls = _load_fastllm_runtime_classes()
    if vllm_cls is None or vllm_decision_cls is None or not isinstance(result_state, dict):
        return ""

    gathered = result_state.get("gathered_info", {})
    if not isinstance(gathered, dict):
        gathered = {}

    toolbar_information = (
        result_state.get("toolbar_information")
        or gathered.get("toolbar_information")
        or ""
    )
    inventory = gathered.get("inventory") or result_state.get("inventory", [])
    selected_item_name = (
        vllm_cls._extract_selected_item_name(gathered)
        or vllm_cls._extract_selected_item_name_from_toolbar(toolbar_information)
    )

    anchor_action = str(failed_action or suggestion_action or "").strip()
    reference_action = str(suggestion_action or anchor_action or "").strip()
    if not reference_action:
        reference_action = anchor_action

    minimal_recovery = _build_minimal_local_recovery_action(
        anchor_action,
        variation_seed=variation_seed,
    )
    validated_minimal = _validate_runtime_candidate_action(
        result_state=result_state,
        action=minimal_recovery,
        suggestion_action=reference_action or minimal_recovery,
        decision_reason=decision_reason,
        vllm_cls=vllm_cls,
        vllm_decision_cls=vllm_decision_cls,
    )
    if validated_minimal:
        return validated_minimal

    if anchor_action:
        invalidated_builder = getattr(vllm_cls, "_build_invalidated_suggestion_local_recovery", None)
        if callable(invalidated_builder):
            corrected = invalidated_builder(
                game_state=result_state,
                suggestion_action=anchor_action,
                selected_item_name=selected_item_name,
                inventory=inventory,
                toolbar_information=toolbar_information,
            )
            validated_corrected = _validate_runtime_candidate_action(
                result_state=result_state,
                action=corrected,
                suggestion_action=reference_action or anchor_action,
                decision_reason=f"{decision_reason}_invalidated",
                vllm_cls=vllm_cls,
                vllm_decision_cls=vllm_decision_cls,
            )
            if validated_corrected:
                return validated_corrected

    autonomous_builder = getattr(vllm_cls, "_build_autonomous_local_recovery_action", None)
    if callable(autonomous_builder):
        autonomous_action = autonomous_builder(game_state=result_state)
        validated_autonomous = _validate_runtime_candidate_action(
            result_state=result_state,
            action=autonomous_action,
            suggestion_action=reference_action or autonomous_action,
            decision_reason=f"{decision_reason}_autonomous",
            vllm_cls=vllm_cls,
            vllm_decision_cls=vllm_decision_cls,
        )
        if validated_autonomous:
            return validated_autonomous

    return ""


def _resolve_runtime_no_action_recovery(
    *,
    result_state: Dict[str, Any],
    blocked_reason: str,
    normalized_suggestion_actions: List[str],
) -> Dict[str, str]:
    vllm_cls, vllm_decision_cls = _load_fastllm_runtime_classes()
    if vllm_cls is None or vllm_decision_cls is None or not isinstance(result_state, dict):
        return {}

    gathered = result_state.get("gathered_info", {})
    if not isinstance(gathered, dict):
        gathered = {}

    current_step_suggestion = _select_current_step_suggestion_for_execution(
        result_state=result_state,
        normalized_suggestion_actions=normalized_suggestion_actions,
    )
    pending_local_recovery_action = str(
        result_state.get("pending_local_recovery_action", "") or ""
    ).strip()
    pending_local_recovery_reason = str(
        result_state.get("pending_local_recovery_reason", "") or "runtime_pending_local_recovery"
    ).strip()
    if pending_local_recovery_action:
        validated_pending = _validate_runtime_candidate_action(
            result_state=result_state,
            action=pending_local_recovery_action,
            suggestion_action=current_step_suggestion or pending_local_recovery_action,
            decision_reason=pending_local_recovery_reason or "runtime_pending_local_recovery",
            vllm_cls=vllm_cls,
            vllm_decision_cls=vllm_decision_cls,
        )
        if validated_pending:
            return {
                "action": validated_pending,
                "source": "runtime_recovery",
            }

    current_runtime_step_raw = result_state.get("step_count", result_state.get("current_step", 0))
    try:
        current_runtime_step = int(current_runtime_step_raw) if current_runtime_step_raw not in (None, "") else 0
    except (TypeError, ValueError):
        current_runtime_step = 0
    previous_no_execution_step_raw = result_state.get("last_no_execution_step", -1)
    try:
        previous_no_execution_step = int(previous_no_execution_step_raw) if previous_no_execution_step_raw not in (None, "") else -1
    except (TypeError, ValueError):
        previous_no_execution_step = -1
    previous_no_execution_signature = str(
        result_state.get("last_no_execution_signature", "") or ""
    ).strip()
    current_no_execution_signature = _build_no_execution_signature(
        blocked_reason=blocked_reason,
        screenshot_path=result_state.get("screenshot_path", ""),
        subtask_description=result_state.get("subtask_description", ""),
    )
    same_no_execution_signature = (
        previous_no_execution_step == current_runtime_step
        and bool(current_no_execution_signature)
        and previous_no_execution_signature == current_no_execution_signature
    )
    try:
        same_step_no_execution_streak = int(result_state.get("same_step_no_execution_streak", 0) or 0)
    except (TypeError, ValueError):
        same_step_no_execution_streak = 0
    recoverable_planning_gap = (
        blocked_reason in {"empty_plan", "awaiting_big_brain_replan", "suggestion_fallback_disabled"}
        or is_recoverable_no_execution_reason(blocked_reason)
    )
    if same_no_execution_signature and same_step_no_execution_streak >= 1 and recoverable_planning_gap:
        repeated_local_recovery = build_runtime_local_recovery_action(
            result_state=result_state,
            suggestion_action=current_step_suggestion,
            failed_action=current_step_suggestion,
            decision_reason="runtime_repeated_no_execution_local_recovery",
            variation_seed=same_step_no_execution_streak + 1,
        )
        if repeated_local_recovery:
            return {
                "action": repeated_local_recovery,
                "source": "runtime_recovery",
            }

    task_text = str(result_state.get("main_task", "") or result_state.get("task", "") or "").strip()
    cultivation_kind = _infer_cultivation_task_kind(task_text)
    execution_log = _build_synthetic_execution_log_for_recovery(
        result_state=result_state,
        blocked_reason=blocked_reason,
        hint_action=current_step_suggestion,
    )
    front_context = vllm_cls._build_front_obstacle_context(
        result_state,
        execution_log,
        hint_action=current_step_suggestion,
    )
    blocked_override_action = str(
        front_context.get("blocked_override_action", "") or ""
    ).strip()
    if blocked_override_action and cultivation_kind not in {"till", "fertilize"}:
        validated_override = _validate_runtime_candidate_action(
            result_state=result_state,
            action=blocked_override_action,
            suggestion_action=current_step_suggestion,
            decision_reason="runtime_blocked_recovery",
            vllm_cls=vllm_cls,
            vllm_decision_cls=vllm_decision_cls,
        )
        if validated_override:
            return {
                "action": validated_override,
                "source": "runtime_recovery",
            }

    toolbar_information = (
        result_state.get("toolbar_information")
        or gathered.get("toolbar_information")
        or ""
    )
    inventory = gathered.get("inventory") or result_state.get("inventory", [])
    selected_item_name = (
        vllm_cls._extract_selected_item_name(gathered)
        or vllm_cls._extract_selected_item_name_from_toolbar(toolbar_information)
    )

    if current_step_suggestion:
        if (
            not blocked_reason.startswith("escalation_reason:")
            or is_recoverable_no_execution_reason(blocked_reason)
            or "parse_fallback_invalidated_suggestion" in blocked_reason
        ):
            invalidated_reason = (
                "parse_fallback_invalidated_suggestion"
                if "parse_fallback_invalidated_suggestion" in blocked_reason
                else "runtime_current_step_suggestion_fallback"
            )
            validated_suggestion = _validate_runtime_candidate_action(
                result_state=result_state,
                action=current_step_suggestion,
                suggestion_action=current_step_suggestion,
                decision_reason=invalidated_reason,
                vllm_cls=vllm_cls,
                vllm_decision_cls=vllm_decision_cls,
            )
            if validated_suggestion:
                return {
                    "action": validated_suggestion,
                    "source": (
                        "suggestions"
                        if validated_suggestion == current_step_suggestion
                        else "runtime_recovery"
                    ),
                }

        if (
            "parse_fallback_invalidated_suggestion" in blocked_reason
            or is_grounded_blocker_reason(blocked_reason)
        ) and cultivation_kind not in {"till", "fertilize"}:
            corrected = vllm_cls._build_invalidated_suggestion_local_recovery(
                game_state=result_state,
                suggestion_action=current_step_suggestion,
                selected_item_name=selected_item_name,
                inventory=inventory,
                toolbar_information=toolbar_information,
            )
            validated_corrected = _validate_runtime_candidate_action(
                result_state=result_state,
                action=corrected,
                suggestion_action=current_step_suggestion,
                decision_reason="runtime_invalidated_suggestion_recovery",
                vllm_cls=vllm_cls,
                vllm_decision_cls=vllm_decision_cls,
            )
            if validated_corrected:
                return {
                    "action": validated_corrected,
                    "source": "runtime_recovery",
                }

    current_menu = (
        result_state.get("current_menu")
        or result_state.get("CurrentMenuData")
        or gathered.get("current_menu")
        or gathered.get("CurrentMenuData")
        or ""
    )
    if vllm_cls._is_menu_open(current_menu):
        return {}

    immediate_use = ""
    if cultivation_kind not in {"till", "fertilize"}:
        immediate_use = str(vllm_cls._build_immediate_use_if_aligned(result_state) or "").strip()
    validated_immediate_use = _validate_runtime_candidate_action(
        result_state=result_state,
        action=immediate_use,
        suggestion_action=current_step_suggestion,
        decision_reason="runtime_immediate_use",
        vllm_cls=vllm_cls,
        vllm_decision_cls=vllm_decision_cls,
    )
    if validated_immediate_use:
        return {
            "action": validated_immediate_use,
            "source": "runtime_recovery",
        }

    recovery_actions = [
        vllm_cls._build_local_clear_recovery_action(
            game_state=result_state,
            selected_item_name=selected_item_name,
            inventory=inventory,
            toolbar_information=toolbar_information,
        ),
        _build_local_watering_recovery_action(
            result_state=result_state,
            vllm_cls=vllm_cls,
            selected_item_name=selected_item_name,
            inventory=inventory,
            toolbar_information=toolbar_information,
        ),
    ]
    if cultivation_kind == "harvest":
        build_local_harvest_recovery_action = getattr(vllm_cls, "_build_local_harvest_recovery_action", None)
        if callable(build_local_harvest_recovery_action):
            recovery_actions.append(
                build_local_harvest_recovery_action(
                    game_state=result_state,
                )
            )
    if cultivation_kind not in {"till", "fertilize"}:
        recovery_actions.append(
            vllm_cls._build_local_placeable_recovery_action(
                game_state=result_state,
                item_name=selected_item_name,
            )
        )

    for recovery_action in recovery_actions:
        validated_recovery = _validate_runtime_candidate_action(
            result_state=result_state,
            action=recovery_action,
            suggestion_action=current_step_suggestion,
            decision_reason="runtime_deterministic_recovery",
            vllm_cls=vllm_cls,
            vllm_decision_cls=vllm_decision_cls,
        )
        if validated_recovery:
            return {
                "action": validated_recovery,
                "source": "runtime_recovery",
            }

    return {}
