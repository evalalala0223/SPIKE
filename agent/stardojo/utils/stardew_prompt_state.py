from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from stardojo.utils.task_grounding import (
    build_clear_task_profile,
    classify_clearable_target,
    classify_tilling_target,
    clear_target_matches_profile,
)


_SURROUNDINGS_LINE_RE = re.compile(
    r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\](?:\([^)]*\))?\s*:\s*(.*)$"
)


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return True


def _pick_first(*values: Any, default: Any = "") -> Any:
    for value in values:
        if _has_value(value):
            return value
    return default


def _normalize_menu_type(menu_value: Any) -> str:
    if isinstance(menu_value, dict):
        menu_type = menu_value.get("type", "")
    else:
        raw_text = str(menu_value or "").strip()
        match = re.search(r"type['\"]?\s*[:=]\s*['\"]?([A-Za-z ]+)", raw_text, re.IGNORECASE)
        menu_type = match.group(1) if match else raw_text

    normalized = re.sub(r"[^a-z0-9]+", " ", str(menu_type or "").lower()).strip()
    if normalized in {"", "none", "null"}:
        return "no menu"
    return normalized


def _description_implies_map_menu(*descriptions: Any) -> bool:
    combined = "\n".join(str(item or "").strip().lower() for item in descriptions if str(item or "").strip())
    if not combined:
        return False

    if "map interface" in combined:
        return True
    if "overall, the image shows a map" in combined:
        return True
    if "shows a map of the stardew valley area" in combined:
        return True
    return False


def _resolve_current_menu(
    *,
    state_dict: Dict[str, Any],
    gathered_dict: Dict[str, Any],
) -> Any:
    state_has_menu_observation = any(
        key in state_dict for key in ("current_menu", "CurrentMenuData")
    )
    gathered_has_menu_observation = any(
        key in gathered_dict for key in ("current_menu", "CurrentMenuData")
    )

    state_menu = _pick_first(
        state_dict.get("current_menu"),
        state_dict.get("CurrentMenuData"),
        default="",
    )
    gathered_menu = _pick_first(
        gathered_dict.get("current_menu"),
        gathered_dict.get("CurrentMenuData"),
        default="",
    )

    explicit_candidates: List[Any] = []
    if gathered_has_menu_observation:
        explicit_candidates.append(gathered_menu if _has_value(gathered_menu) else "No Menu")
    if state_has_menu_observation:
        explicit_candidates.append(state_menu if _has_value(state_menu) else "No Menu")

    for candidate in explicit_candidates:
        if _normalize_menu_type(candidate) != "no menu":
            return candidate

    if explicit_candidates:
        current_menu = explicit_candidates[0]
    else:
        current_menu = ""

    if _normalize_menu_type(current_menu) == "no menu" and _description_implies_map_menu(
        gathered_dict.get("description"),
        state_dict.get("description"),
        gathered_dict.get("image_description"),
        state_dict.get("image_description"),
    ):
        return {
            "type": "MapPage",
            "inferred_from_description": True,
        }

    return current_menu


def _canonicalize_surroundings_string(text: Any) -> str:
    if not isinstance(text, str):
        return str(text or "").strip()

    stripped = text.strip()
    if not stripped:
        return ""

    normalized_lines: List[str] = []
    matched_any = False
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        match = _SURROUNDINGS_LINE_RE.match(line)
        if match:
            rel_x = int(match.group(1))
            rel_y = int(match.group(2))
            label = match.group(3).strip() or "empty"
            normalized_lines.append(f"[{rel_x}, {rel_y}]: {label}")
            matched_any = True
        else:
            normalized_lines.append(line)

    if matched_any:
        return "\n".join(normalized_lines)
    return stripped


def _normalize_position(position: Any) -> Optional[Tuple[int, int]]:
    if isinstance(position, (list, tuple)) and len(position) >= 2:
        x, y = position[0], position[1]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return int(x), int(y)
    if isinstance(position, dict):
        x = position.get("x", position.get("X"))
        y = position.get("y", position.get("Y"))
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return int(x), int(y)
    return None


def _flatten_surroundings_value(value: Any) -> List[str]:
    if value in (None, "", []):
        return []
    if isinstance(value, dict):
        for key in ("seed_name", "name", "Name", "item_name", "id"):
            candidate = value.get(key)
            if candidate not in (None, "", []):
                return [str(candidate).strip()]
        return [str(value).strip()]
    if isinstance(value, (list, tuple, set)):
        parts: List[str] = []
        for item in value:
            parts.extend(_flatten_surroundings_value(item))
        return parts
    return [str(value).strip()]


def _normalize_surroundings_text(surroundings: Any, player_position: Any = None) -> str:
    if isinstance(surroundings, str):
        return _canonicalize_surroundings_string(surroundings)
    if surroundings in (None, "", []):
        return ""

    normalized_player = _normalize_position(player_position)

    if isinstance(surroundings, list):
        lines: List[str] = []
        for tile in surroundings:
            if not isinstance(tile, dict):
                continue

            tile_position = _normalize_position(tile.get("position"))
            if tile_position is None:
                continue

            rel_x, rel_y = tile_position
            if normalized_player is not None:
                rel_x -= normalized_player[0]
                rel_y -= normalized_player[1]

            descriptors: List[str] = []
            for key, value in tile.items():
                if key == "position":
                    continue
                descriptors.extend(_flatten_surroundings_value(value))

            deduped: List[str] = []
            seen = set()
            for descriptor in descriptors:
                normalized = str(descriptor).strip()
                if not normalized:
                    continue
                lowered = normalized.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                deduped.append(normalized)

            summary = ", ".join(deduped) if deduped else "empty"
            lines.append(f"[{rel_x}, {rel_y}]: {summary}")

        if lines:
            return "\n".join(lines)

    return _canonicalize_surroundings_string(str(surroundings))


def _parse_surroundings_map(surroundings: Any) -> Dict[Tuple[int, int], str]:
    cells: Dict[Tuple[int, int], str] = {}
    for raw_line in _canonicalize_surroundings_string(surroundings).splitlines():
        line = raw_line.strip()
        match = _SURROUNDINGS_LINE_RE.match(line)
        if not match:
            continue
        cell = (int(match.group(1)), int(match.group(2)))
        cells[cell] = str(match.group(3) or "").strip()
    return cells


def _pick_best_surroundings_text(
    state_dict: Dict[str, Any],
    gathered_dict: Dict[str, Any],
) -> str:
    reference_position = _pick_first(
        state_dict.get("current_position"),
        state_dict.get("position"),
        gathered_dict.get("current_position"),
        gathered_dict.get("position"),
        default=None,
    )
    structured_candidates = (
        state_dict.get("surroundings"),
        gathered_dict.get("surroundings"),
        gathered_dict.get("description"),
        state_dict.get("description"),
    )
    for candidate in structured_candidates:
        text = _normalize_surroundings_text(candidate, reference_position)
        if text and _parse_surroundings_map(text):
            return text

    # Fall back to the first non-empty natural-language surroundings text.
    for candidate in structured_candidates:
        text = _normalize_surroundings_text(candidate, reference_position)
        if text:
            return text

    return ""


def _direction_to_relative(direction: Any) -> Optional[Tuple[int, int]]:
    return {
        "up": (0, -1),
        "down": (0, 1),
        "left": (-1, 0),
        "right": (1, 0),
    }.get(str(direction or "").strip().lower())


def _classify_clearable_object(obj_text: Any) -> Optional[Dict[str, str]]:
    return classify_clearable_target(obj_text)


def _looks_like_structural_blocker(obj_text: Any) -> bool:
    text = str(obj_text or "").strip().lower()
    if not text:
        return False
    return any(
        token in text
        for token in (
            "farmhouse",
            "house",
            "porch",
            "wall",
            "building",
            "barn",
            "coop",
            "shed",
            "silo",
        )
    )


def _looks_like_cultivation_target(obj_text: Any) -> bool:
    text = str(obj_text or "").strip().lower()
    if not text:
        return False
    return any(
        token in text
        for token in (
            "hoedirt",
            "tilled soil",
            "tilled dirt",
            "seed",
            "sprout",
            "crop",
            "growing",
        )
    )


def _looks_like_interaction_target(obj_text: Any) -> bool:
    text = str(obj_text or "").strip().lower()
    if not text:
        return False
    if any(token in text for token in ("door", "entrance", "gate")):
        return True
    if _looks_like_structural_blocker(text):
        return False
    return any(
        token in text
        for token in (
            "shipping bin",
            "mailbox",
            "counter",
            "chest",
            "barrel",
            "quest",
        )
    )


def _looks_like_open_reroute_tile(obj_text: Any) -> bool:
    text = str(obj_text or "").strip().lower()
    if not text:
        return False
    if _classify_clearable_object(text) or _looks_like_cultivation_target(text):
        return False
    if _looks_like_structural_blocker(text) or _looks_like_interaction_target(text):
        return False
    return any(
        token in text
        for token in (
            "empty",
            "grass",
            "dirt",
            "soil",
            "ground",
            "floor",
            "path",
            "stone path",
            "sand",
        )
    )


def _distance_key(cell: Tuple[int, int]) -> Tuple[int, int, int, int, int]:
    return (abs(cell[0]) + abs(cell[1]), abs(cell[1]), abs(cell[0]), cell[1], cell[0])


def _has_grounded_blocker_signal(state_dict: Dict[str, Any]) -> bool:
    escalation_reason = str(state_dict.get("escalation_reason", "") or "").strip().lower()
    last_errors_info = str(state_dict.get("last_errors_info", "") or "").strip().lower()
    return bool(state_dict.get("blocker_replan_only", False)) or bool(
        str(state_dict.get("last_blocker_signature", "") or "").strip()
    ) or any(
        marker in escalation_reason or marker in last_errors_info
        for marker in (
            "move_target_blocked",
            "invalid_target",
            "blocked by an obstacle",
            "path is likely blocked",
        )
    )


def _build_current_blocker_signature(
    *,
    state_dict: Dict[str, Any],
    gathered_dict: Dict[str, Any],
    direction: str,
    relative_cell: Optional[Tuple[int, int]],
    front_obj: str,
    clearable: Optional[Dict[str, str]],
    grounded_blocker: bool,
) -> str:
    explicit = _pick_first(
        state_dict.get("current_blocker_signature"),
        gathered_dict.get("current_blocker_signature"),
        state_dict.get("last_blocker_signature"),
        gathered_dict.get("last_blocker_signature"),
        default="",
    )
    if relative_cell is None:
        return str(explicit or "(none)")
    if front_obj and (_looks_like_structural_blocker(front_obj) or grounded_blocker):
        if clearable:
            signature = (
                f"Immediate blocker on the current facing line: {front_obj} at {relative_cell} "
                f"toward {direction}; it is clearable with {clearable['tool']} if rerouting is not grounded."
            )
        else:
            signature = (
                f"Immediate blocker on the current facing line: {front_obj} at {relative_cell} "
                f"toward {direction}."
            )
        explicit_text = str(explicit or "").strip()
        if explicit_text and explicit_text not in signature:
            return f"{signature} Previous blocker note: {explicit_text}."
        return signature
    return str(explicit or "(none)")


def _build_nearest_grounded_target_summary(
    *,
    state_dict: Dict[str, Any],
    gathered_dict: Dict[str, Any],
    surroundings_map: Dict[Tuple[int, int], str],
    front_obj: str,
    grounded_blocker: bool,
) -> str:
    explicit = _pick_first(
        state_dict.get("nearest_grounded_target_summary"),
        gathered_dict.get("nearest_grounded_target_summary"),
        default="",
    )
    exits = _pick_first(
        state_dict.get("exits"),
        gathered_dict.get("exits"),
        default="",
    )
    if not surroundings_map:
        if explicit:
            return str(explicit)
        if _has_value(exits):
            return f"Nearest grounded route context: {str(exits).strip()}."
        return "(none)"

    task_text = str(
        state_dict.get("main_task")
        or state_dict.get("task")
        or state_dict.get("task_description")
        or state_dict.get("subtask_description")
        or gathered_dict.get("main_task")
        or gathered_dict.get("task")
        or gathered_dict.get("task_description")
        or gathered_dict.get("subtask_description")
        or ""
    ).strip().lower()
    cultivation_kind = ""
    for prefix, kind in (
        ("till_", "till"),
        ("till ", "till"),
        ("sow_", "sow"),
        ("sow ", "sow"),
        ("fertilize_", "fertilize"),
        ("fertilize ", "fertilize"),
        ("water_", "water"),
        ("water ", "water"),
        ("harvest_", "harvest"),
        ("harvest ", "harvest"),
        ("cultivate_and_harvest_", "cultivate_and_harvest"),
        ("cultivate and harvest ", "cultivate_and_harvest"),
    ):
        if task_text.startswith(prefix):
            cultivation_kind = kind
            break
    if not cultivation_kind:
        normalized_task_text = f" {re.sub(r'[^a-z0-9]+', ' ', task_text)} "
        for token, kind in (
            (" till ", "till"),
            (" sow ", "sow"),
            (" fertilize ", "fertilize"),
            (" water ", "water"),
            (" harvest ", "harvest"),
            (" cultivate and harvest ", "cultivate_and_harvest"),
        ):
            if token in normalized_task_text:
                cultivation_kind = kind
                break
    clear_profile = build_clear_task_profile(task_text)
    primary_candidates = []
    reroute_candidates = []
    for cell, raw_text in surroundings_map.items():
        if cell == (0, 0):
            continue
        text = str(raw_text or "").strip()
        if not text:
            continue
        distance_key = _distance_key(cell)
        if cultivation_kind == "till":
            till_candidate = classify_tilling_target(surroundings_map, cell)
            if till_candidate:
                primary_candidates.append(
                    (
                        1,
                        (
                            till_candidate["nearby_structures"],
                            -till_candidate["nearby_open_ground"],
                            distance_key,
                        ),
                        f"Nearest grounded till target: {text} at {cell}; it looks like a {till_candidate['label']}.",
                    )
                )
                continue
        elif cultivation_kind and _looks_like_cultivation_target(text):
            primary_candidates.append(
                (
                    1,
                    distance_key,
                    f"Nearest grounded cultivation target: {text} at {cell}.",
                )
            )
            continue
        clearable = _classify_clearable_object(text)
        if clearable:
            if cultivation_kind:
                continue
            if clear_profile and not clear_target_matches_profile(text, clear_profile):
                continue
            label = str(clear_profile.get("summary_label", "Nearest grounded debris target") or "Nearest grounded debris target")
            primary_candidates.append(
                (
                    2,
                    distance_key,
                    f"{label}: {text} at {cell}; it requires {clearable['tool']}.",
                )
            )
            continue
        if _looks_like_cultivation_target(text):
            primary_candidates.append(
                (
                    2 if cultivation_kind else 3,
                    distance_key,
                    f"Nearest grounded cultivation target: {text} at {cell}.",
                )
            )
            continue
        if _looks_like_interaction_target(text):
            primary_candidates.append(
                (
                    4,
                    distance_key,
                    f"Nearest grounded interaction target: {text} at {cell}.",
                )
            )
            continue
        if _looks_like_open_reroute_tile(text):
            reroute_candidates.append(
                (
                    distance_key,
                    f"Nearest open reroute tile: {text} at {cell}. Use it to step off blocked structures or line up a valid action.",
                )
            )

    blocker_like = grounded_blocker or _looks_like_structural_blocker(front_obj)
    primary_summary = ""
    if primary_candidates:
        _, _, primary_summary = min(primary_candidates, key=lambda item: (item[0], item[1]))
    reroute_summary = ""
    if reroute_candidates:
        _, reroute_summary = min(reroute_candidates, key=lambda item: item[0])

    if primary_summary and blocker_like and reroute_summary:
        return f"{reroute_summary} {primary_summary}"
    if primary_summary:
        return primary_summary
    if reroute_summary:
        return reroute_summary
    if _has_value(exits):
        return f"Nearest grounded route context: {str(exits).strip()}."
    if explicit:
        return str(explicit)
    return "(none)"


def _build_front_obstacle_fields(
    *,
    state_dict: Dict[str, Any],
    gathered_dict: Dict[str, Any],
    surroundings_text: Any,
    facing_direction: Any,
) -> Dict[str, str]:
    explicit_summary = _pick_first(
        state_dict.get("front_tile_summary"),
        gathered_dict.get("front_tile_summary"),
        default="",
    )
    explicit_hint = _pick_first(
        state_dict.get("blocked_recovery_hint"),
        gathered_dict.get("blocked_recovery_hint"),
        default="",
    )
    explicit_blocker_signature = _pick_first(
        state_dict.get("current_blocker_signature"),
        gathered_dict.get("current_blocker_signature"),
        state_dict.get("last_blocker_signature"),
        gathered_dict.get("last_blocker_signature"),
        default="",
    )
    explicit_target_summary = _pick_first(
        state_dict.get("nearest_grounded_target_summary"),
        gathered_dict.get("nearest_grounded_target_summary"),
        default="",
    )
    direction = str(facing_direction or "").strip().lower()
    relative_cell = _direction_to_relative(direction)
    surroundings_map = _parse_surroundings_map(surroundings_text)
    if not surroundings_map:
        exits = _pick_first(
            state_dict.get("exits"),
            gathered_dict.get("exits"),
            default="",
        )
        if explicit_summary or explicit_hint or explicit_blocker_signature or explicit_target_summary:
            return {
                "front_tile_summary": str(explicit_summary or "(none)"),
                "blocked_recovery_hint": str(explicit_hint or ""),
                "current_blocker_signature": str(explicit_blocker_signature or "(none)"),
                "nearest_grounded_target_summary": str(
                    explicit_target_summary
                    or (f"Nearest grounded route context: {str(exits).strip()}." if _has_value(exits) else "(none)")
                ),
            }
        return {
            "front_tile_summary": "(none)",
            "blocked_recovery_hint": "",
            "current_blocker_signature": str(explicit_blocker_signature or "(none)"),
            "nearest_grounded_target_summary": str(
                f"Nearest grounded route context: {str(exits).strip()}." if _has_value(exits) else "(none)"
            ),
        }
    if relative_cell is None:
        exits = _pick_first(
            state_dict.get("exits"),
            gathered_dict.get("exits"),
            default="",
        )
        nearest_grounded_target_summary = _build_nearest_grounded_target_summary(
            state_dict=state_dict,
            gathered_dict=gathered_dict,
            surroundings_map=surroundings_map,
            front_obj="",
            grounded_blocker=_has_grounded_blocker_signal(state_dict),
        )
        if explicit_summary or explicit_hint or explicit_blocker_signature or explicit_target_summary:
            return {
                "front_tile_summary": str(explicit_summary or "(none)"),
                "blocked_recovery_hint": str(explicit_hint or ""),
                "current_blocker_signature": str(explicit_blocker_signature or "(none)"),
                "nearest_grounded_target_summary": str(
                    explicit_target_summary
                    or nearest_grounded_target_summary
                    or (f"Nearest grounded route context: {str(exits).strip()}." if _has_value(exits) else "(none)")
                ),
            }
        return {
            "front_tile_summary": "(none)",
            "blocked_recovery_hint": "",
            "current_blocker_signature": str(explicit_blocker_signature or "(none)"),
            "nearest_grounded_target_summary": str(
                nearest_grounded_target_summary
                or (f"Nearest grounded route context: {str(exits).strip()}." if _has_value(exits) else "(none)")
            ),
        }

    front_obj = str(surroundings_map.get(relative_cell, "") or "").strip()
    clearable = _classify_clearable_object(front_obj)

    if not front_obj:
        front_tile_summary = (
            f"Front tile {relative_cell} toward {direction}: no explicit object in current surroundings."
        )
    elif clearable:
        front_tile_summary = (
            f"Front tile {relative_cell} toward {direction}: {front_obj}. "
            f'Clearable with {clearable["tool"]}.'
        )
    else:
        front_tile_summary = (
            f"Front tile {relative_cell} toward {direction}: {front_obj}. "
            "Not an obvious clearable obstacle."
        )

    grounded_blocker = _has_grounded_blocker_signal(state_dict)

    blocked_recovery_hint = ""
    if grounded_blocker and front_obj:
        if clearable:
            blocked_recovery_hint = (
                f'Recent planning or execution already grounded the front tile as the blocker. '
                "Do not repeat the same-direction move/use/interact into it. "
                "Prefer rerouting to a grounded nearby tile or waypoint first. "
                f'Only clear the {clearable["label"]} ahead with {clearable["tool"]} '
                "if rerouting is not grounded or rerouting still fails."
            )
        else:
            blocked_recovery_hint = (
                "Recent planning or execution already showed the front tile blocks direct progress. "
                "Do not repeat the same-direction move/use/interact into it; route around it or "
                "pick a different nearby waypoint first."
            )
    elif grounded_blocker:
        blocked_recovery_hint = (
            "Recent planning or execution already showed this direction is blocked or misaligned. "
            "Do not repeat the same-direction move/use until current surroundings provide a "
            "concrete target."
        )

    current_blocker_signature = _build_current_blocker_signature(
        state_dict=state_dict,
        gathered_dict=gathered_dict,
        direction=direction,
        relative_cell=relative_cell,
        front_obj=front_obj,
        clearable=clearable,
        grounded_blocker=grounded_blocker,
    )
    nearest_grounded_target_summary = _build_nearest_grounded_target_summary(
        state_dict=state_dict,
        gathered_dict=gathered_dict,
        surroundings_map=surroundings_map,
        front_obj=front_obj,
        grounded_blocker=grounded_blocker,
    )

    return {
        "front_tile_summary": front_tile_summary,
        "blocked_recovery_hint": blocked_recovery_hint,
        "current_blocker_signature": current_blocker_signature,
        "nearest_grounded_target_summary": nearest_grounded_target_summary,
    }


def extract_stardew_prompt_fact_fields(
    state: Any = None,
    gathered_info: Any = None,
) -> Dict[str, Any]:
    state_dict = state if isinstance(state, dict) else {}
    gathered_dict = gathered_info if isinstance(gathered_info, dict) else {}

    position = _pick_first(
        state_dict.get("position"),
        state_dict.get("current_position"),
        gathered_dict.get("position"),
        gathered_dict.get("current_position"),
        default="",
    )
    current_position = _pick_first(
        state_dict.get("current_position"),
        state_dict.get("position"),
        gathered_dict.get("current_position"),
        gathered_dict.get("position"),
        default=position,
    )

    current_menu = _resolve_current_menu(
        state_dict=state_dict,
        gathered_dict=gathered_dict,
    )

    facing_direction = _pick_first(
        state_dict.get("facing_direction"),
        gathered_dict.get("facing_direction"),
        default="",
    )
    surroundings = _pick_best_surroundings_text(state_dict, gathered_dict)
    front_obstacle_fields = _build_front_obstacle_fields(
        state_dict=state_dict,
        gathered_dict=gathered_dict,
        surroundings_text=surroundings,
        facing_direction=facing_direction,
    )

    return {
        "basic_knowledge": _pick_first(
            state_dict.get("basic_knowledge"),
            gathered_dict.get("basic_knowledge"),
            default=[],
        ),
        "location": _pick_first(
            state_dict.get("location"),
            gathered_dict.get("location"),
            default="",
        ),
        "time": _pick_first(
            state_dict.get("time"),
            gathered_dict.get("time"),
            default="",
        ),
        "day": _pick_first(
            state_dict.get("day"),
            gathered_dict.get("day"),
            default="",
        ),
        "season": _pick_first(
            state_dict.get("season"),
            gathered_dict.get("season"),
            default="",
        ),
        "health": _pick_first(
            state_dict.get("health"),
            gathered_dict.get("health"),
            default="",
        ),
        "energy": _pick_first(
            state_dict.get("energy"),
            gathered_dict.get("energy"),
            default="",
        ),
        "money": _pick_first(
            state_dict.get("money"),
            gathered_dict.get("money"),
            default="",
        ),
        "position": position,
        "current_position": current_position,
        "facing_direction": facing_direction,
        "facing_position": _pick_first(
            state_dict.get("facing_position"),
            gathered_dict.get("facing_position"),
            default="",
        ),
        "current_menu": current_menu,
        "inventory": _pick_first(
            state_dict.get("inventory"),
            gathered_dict.get("inventory"),
            default=[],
        ),
        "chosen_item": _pick_first(
            state_dict.get("chosen_item"),
            gathered_dict.get("chosen_item"),
            default="",
        ),
        "selected_position": _pick_first(
            state_dict.get("selected_position"),
            gathered_dict.get("selected_position"),
            default=None,
        ),
        "selected_item_name": _pick_first(
            state_dict.get("selected_item_name"),
            gathered_dict.get("selected_item_name"),
            default="",
        ),
        "surroundings": surroundings,
        "front_tile_summary": front_obstacle_fields["front_tile_summary"],
        "blocked_recovery_hint": front_obstacle_fields["blocked_recovery_hint"],
        "current_blocker_signature": front_obstacle_fields["current_blocker_signature"],
        "nearest_grounded_target_summary": front_obstacle_fields["nearest_grounded_target_summary"],
        "toolbar_information": _pick_first(
            state_dict.get("toolbar_information"),
            gathered_dict.get("toolbar_information"),
            default="",
        ),
        "crops": _pick_first(
            state_dict.get("crops"),
            gathered_dict.get("crops"),
            default="",
        ),
        "buildings": _pick_first(
            state_dict.get("buildings"),
            gathered_dict.get("buildings"),
            default="",
        ),
        "furniture": _pick_first(
            state_dict.get("furniture"),
            gathered_dict.get("furniture"),
            default="",
        ),
        "npcs": _pick_first(
            state_dict.get("npcs"),
            gathered_dict.get("npcs"),
            default="",
        ),
        "exits": _pick_first(
            state_dict.get("exits"),
            gathered_dict.get("exits"),
            default="",
        ),
        "failure_root_cause": _pick_first(
            state_dict.get("failure_root_cause"),
            gathered_dict.get("failure_root_cause"),
            default="",
        ),
        "failure_signature": _pick_first(
            state_dict.get("failure_signature"),
            gathered_dict.get("failure_signature"),
            default="",
        ),
        "required_change_type": _pick_first(
            state_dict.get("required_change_type"),
            gathered_dict.get("required_change_type"),
            default="",
        ),
        "deadlock_signature": _pick_first(
            state_dict.get("deadlock_signature"),
            gathered_dict.get("deadlock_signature"),
            default="",
        ),
        "deadlock_reflection_cycles": str(_pick_first(
            state_dict.get("deadlock_reflection_cycles"),
            gathered_dict.get("deadlock_reflection_cycles"),
            default="0",
        )),
    }


__all__ = ["extract_stardew_prompt_fact_fields"]
