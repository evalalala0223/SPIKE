from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Tuple

from stardojo.utils.task_bootstrap import get_task_spec


_EMPTY_TILES = {"", "empty", "none", "null", "air"}
_OPEN_GROUND_TILES = {"empty", "open ground", "ground", "floor", "path", "road", "dirt", "soil"}
_HARD_STRUCTURE_TOKENS = (
    "farmhouse",
    "house",
    "barn",
    "coop",
    "shed",
    "silo",
    "wall",
    "fence",
    "shipping bin",
    "mailbox",
    "pet bowl",
    "counter",
    "door",
)
_CLEAR_TASK_PROFILES = {
    "weeds": {
        "family": "weeds",
        "allowed_families": {"weeds"},
        "desired_tools": {"Scythe"},
        "label": "weeds",
        "summary_label": "Nearest grounded debris target",
    },
    "stone": {
        "family": "stone",
        "allowed_families": {"stone"},
        "desired_tools": {"Pickaxe"},
        "label": "stone",
        "summary_label": "Nearest grounded debris target",
    },
    "twig": {
        "family": "twig",
        "allowed_families": {"twig"},
        "desired_tools": {"Axe"},
        "label": "twig",
        "summary_label": "Nearest grounded debris target",
    },
    "debris": {
        "family": "debris",
        "allowed_families": {"weeds", "stone", "twig"},
        "desired_tools": {"Scythe", "Pickaxe", "Axe"},
        "label": "debris",
        "summary_label": "Nearest grounded debris target",
    },
    "hay": {
        "family": "hay",
        "allowed_families": {"hay", "weeds"},
        "desired_tools": {"Scythe"},
        "label": "hay",
        "summary_label": "Nearest grounded hay target",
    },
    "scythe_generic": {
        "family": "scythe_generic",
        "allowed_families": {"weeds", "hay"},
        "desired_tools": {"Scythe"},
        "label": "scythe target",
        "summary_label": "Nearest grounded debris target",
    },
}


def normalize_free_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify_clearable_target(obj_text: Any) -> Optional[Dict[str, str]]:
    text = normalize_free_text(obj_text)
    if not text or text in _EMPTY_TILES:
        return None
    if any(
        token in text
        for token in (
            "green slime",
            "slime",
            "bug",
            "fly",
            "duggy",
            "grub",
            "rock crab",
            "crab",
            "monster",
            "enemy",
        )
    ):
        return {"family": "monster", "tool": "Rusty Sword", "label": "monster"}
    if any(token in text for token in ("weed", "fiber", "fibre")):
        return {"family": "weeds", "tool": "Scythe", "label": "weeds"}
    if any(token in text for token in ("grass", "hay")):
        return {"family": "hay", "tool": "Scythe", "label": "grass or hay"}
    if any(token in text for token in ("stone", "rock", "boulder", "ore")):
        return {"family": "stone", "tool": "Pickaxe", "label": "stone"}
    if any(token in text for token in ("twig", "branch", "wood", "log", "stump")):
        return {"family": "twig", "tool": "Axe", "label": "wood or twigs"}
    return None


def build_clear_task_profile(
    task_name: Any,
    *,
    spec: Optional[Dict[str, Any]] = None,
    tool_name: Any = "",
    object_name: Any = "",
) -> Dict[str, Any]:
    resolved_spec = dict(spec or get_task_spec(str(task_name or "")) or {})
    task_text = normalize_free_text(task_name)
    evaluator = normalize_free_text(resolved_spec.get("evaluator", ""))
    object_text = normalize_free_text(object_name or resolved_spec.get("object", ""))
    tool_text = normalize_free_text(tool_name or resolved_spec.get("tool", ""))

    if evaluator == "silo" or (
        "hay" in task_text and any(token in task_text for token in ("forage", "collect", "cut", "scythe", "silo"))
    ):
        return dict(_CLEAR_TASK_PROFILES["hay"])
    if evaluator == "clear":
        if object_text == "weeds":
            return dict(_CLEAR_TASK_PROFILES["weeds"])
        if object_text == "stone":
            return dict(_CLEAR_TASK_PROFILES["stone"])
        if object_text == "twig":
            return dict(_CLEAR_TASK_PROFILES["twig"])
        if object_text == "debris":
            return dict(_CLEAR_TASK_PROFILES["debris"])

    if "debris" in task_text:
        return dict(_CLEAR_TASK_PROFILES["debris"])
    if "weed" in task_text:
        return dict(_CLEAR_TASK_PROFILES["weeds"])
    if any(token in task_text for token in ("stone", "rock", "boulder", "ore")):
        return dict(_CLEAR_TASK_PROFILES["stone"])
    if any(token in task_text for token in ("twig", "branch", "wood", "log", "stump")):
        return dict(_CLEAR_TASK_PROFILES["twig"])
    if "scythe" in tool_text:
        return dict(_CLEAR_TASK_PROFILES["scythe_generic"])
    if "pickaxe" in tool_text:
        return dict(_CLEAR_TASK_PROFILES["stone"])
    if "axe" in tool_text:
        return dict(_CLEAR_TASK_PROFILES["twig"])
    return {}


def clear_target_matches_profile(obj_text: Any, profile: Mapping[str, Any]) -> bool:
    clearable = classify_clearable_target(obj_text)
    if not clearable:
        return False
    allowed_families = set(profile.get("allowed_families", set()) or set())
    desired_tools = set(profile.get("desired_tools", set()) or set())
    if allowed_families and clearable["family"] not in allowed_families:
        return False
    if desired_tools and clearable["tool"] not in desired_tools:
        return False
    return True


def is_empty_like_tile(obj_text: Any) -> bool:
    return normalize_free_text(obj_text) in _EMPTY_TILES


def is_open_ground_tile(obj_text: Any) -> bool:
    return normalize_free_text(obj_text) in _OPEN_GROUND_TILES


def is_hard_structure_text(obj_text: Any) -> bool:
    text = normalize_free_text(obj_text)
    if not text:
        return False
    return any(token in text for token in _HARD_STRUCTURE_TOKENS)


def is_explicit_tillable_ground(obj_text: Any) -> bool:
    text = normalize_free_text(obj_text)
    if not text or text in _EMPTY_TILES:
        return False

    invalid_tokens = (
        "hoedirt",
        "hoe dirt",
        "tilled dirt",
        "tilled soil",
        "crop",
        "seed",
        "seeds",
        "water",
        "pond",
        "river",
        "lake",
        "tree",
        "stone",
        "rock",
        "twig",
        "branch",
        "wood",
        "log",
        "stump",
        "weed",
        "grass",
        "fiber",
        "fibre",
        "bush",
        "fence",
        "farmhouse",
        "house",
        "barn",
        "coop",
        "wall",
        "door",
        "bed",
        "chest",
        "shipping bin",
        "mailbox",
    )
    if any(token in text for token in invalid_tokens):
        return False
    return any(token in text for token in ("dirt", "soil", "ground", "mud", "sand"))


def count_nearby_hard_structures(
    surroundings_map: Mapping[Tuple[int, int], Any],
    cell: Tuple[int, int],
    *,
    radius: int = 1,
) -> int:
    cell_x, cell_y = cell
    count = 0
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            if is_hard_structure_text(surroundings_map.get((cell_x + dx, cell_y + dy), "")):
                count += 1
    return count


def count_nearby_open_ground_tiles(
    surroundings_map: Mapping[Tuple[int, int], Any],
    cell: Tuple[int, int],
    *,
    radius: int = 1,
) -> int:
    cell_x, cell_y = cell
    count = 0
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            candidate = surroundings_map.get((cell_x + dx, cell_y + dy), "")
            if is_open_ground_tile(candidate) or is_explicit_tillable_ground(candidate):
                count += 1
    return count


def classify_tilling_target(
    surroundings_map: Mapping[Tuple[int, int], Any],
    cell: Tuple[int, int],
) -> Optional[Dict[str, Any]]:
    raw_text = str(surroundings_map.get(cell, "") or "").strip()
    if not raw_text:
        return None

    nearby_structures = count_nearby_hard_structures(surroundings_map, cell, radius=1)
    nearby_open_ground = count_nearby_open_ground_tiles(surroundings_map, cell, radius=1)

    if is_explicit_tillable_ground(raw_text):
        return {
            "kind": "explicit_ground",
            "text": raw_text,
            "label": "tillable ground",
            "nearby_structures": nearby_structures,
            "nearby_open_ground": nearby_open_ground,
        }
    if (
        (is_empty_like_tile(raw_text) or is_open_ground_tile(raw_text))
        and nearby_open_ground >= 2
        and nearby_structures == 0
    ):
        return {
            "kind": "open_patch",
            "text": raw_text,
            "label": "open till patch",
            "nearby_structures": nearby_structures,
            "nearby_open_ground": nearby_open_ground,
        }
    return None


def is_safe_empty_till_use_target(
    surroundings_map: Mapping[Tuple[int, int], Any],
    target_cell: Tuple[int, int],
    *,
    current_cell: Tuple[int, int] = (0, 0),
) -> bool:
    raw_text = str(surroundings_map.get(target_cell, "") or "").strip()
    if not raw_text or not (is_empty_like_tile(raw_text) or is_open_ground_tile(raw_text)):
        return False

    candidate = classify_tilling_target(surroundings_map, target_cell)
    return bool(candidate and candidate["kind"] == "open_patch")


__all__ = [
    "build_clear_task_profile",
    "classify_clearable_target",
    "classify_tilling_target",
    "clear_target_matches_profile",
    "count_nearby_hard_structures",
    "count_nearby_open_ground_tiles",
    "is_empty_like_tile",
    "is_explicit_tillable_ground",
    "is_hard_structure_text",
    "is_open_ground_tile",
    "is_safe_empty_till_use_target",
    "normalize_free_text",
]
