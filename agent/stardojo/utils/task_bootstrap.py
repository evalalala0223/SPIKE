from __future__ import annotations

import os
import re
from functools import lru_cache
from typing import Any, Dict

import yaml


def _task_suite_dir() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "env", "tasks", "task_suite")
    )


def _normalize_task_lookup_key(value: str) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=1)
def _load_task_specs() -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    task_suite_dir = _task_suite_dir()
    if not os.path.isdir(task_suite_dir):
        return specs

    for filename in sorted(os.listdir(task_suite_dir)):
        if not filename.endswith(".yaml"):
            continue
        full_path = os.path.join(task_suite_dir, filename)
        try:
            with open(full_path, "r", encoding="utf-8") as file:
                raw = yaml.safe_load(file) or {}
        except Exception:
            continue

        if not isinstance(raw, dict):
            continue

        for task_name, task_spec in raw.items():
            if isinstance(task_name, str) and isinstance(task_spec, dict):
                specs.setdefault(task_name, task_spec)

    return specs


@lru_cache(maxsize=1)
def _load_normalized_task_specs() -> Dict[str, Dict[str, Any]]:
    normalized_specs: Dict[str, Dict[str, Any]] = {}
    for task_name, task_spec in _load_task_specs().items():
        normalized_key = _normalize_task_lookup_key(task_name)
        if normalized_key:
            normalized_specs.setdefault(normalized_key, task_spec)
    return normalized_specs


def _get_task_spec(task_name: str) -> Dict[str, Any]:
    exact_key = str(task_name or "").strip()
    exact_match = _load_task_specs().get(exact_key)
    if isinstance(exact_match, dict):
        return dict(exact_match)

    normalized_key = _normalize_task_lookup_key(exact_key)
    if not normalized_key:
        return {}

    return dict(_load_normalized_task_specs().get(normalized_key, {}) or {})


def get_task_spec(task_name: str) -> Dict[str, Any]:
    """Public helper for looking up benchmark task metadata by task name."""
    return _get_task_spec(task_name)


def _humanize_task_name(task_name: str) -> str:
    text = re.sub(r"[_\-]+", " ", str(task_name or "").strip())
    text = re.sub(r"\s+", " ", text).strip()
    return text or "the current task"


def _format_display_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    text = re.sub(r"[_\-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _format_quantity(value: Any) -> str:
    if value in (None, "", 0):
        return ""
    if value == 1:
        return "one"
    return str(value)


def _format_target_phrase(quantity: Any, object_name: str) -> str:
    qty = _format_quantity(quantity)
    object_text = str(object_name or "").strip().lower()
    if qty and object_text:
        return f"{qty} {object_text}"
    if object_text:
        return object_text
    if qty:
        return qty
    return "the target"


_DIRECT_OBJECT_ACQUISITION_EVALUATORS = {
    "accept",
    "backpack",
    "break",
    "build",
    "complete_help",
    "complete_story",
    "demolish",
    "exchange",
    "jojamart",
    "location",
    "move",
    "museum",
    "purchase",
    "purchase_animal",
    "read",
    "repair",
    "reward",
    "sell",
    "sell_animal",
    "upgrade_farmhouse",
    "upgrade_tool",
}

_TOOL_ACQUISITION_EVALUATORS = {
    "clear",
    "fertilize",
    "fill",
    "gift",
    "kill",
    "sow",
    "till",
    "water",
}

_SEED_OR_FERTILIZER_KEYWORDS = (
    "seed",
    "seeds",
    "fertilizer",
    "soil",
    "speed-gro",
    "speed gro",
    "retaining soil",
)

_SOURCE_HINTS_BY_EVALUATOR = {
    "accept": ("quest_board", "Route to the quest board or quest giver for {target}."),
    "backpack": ("seed_shop", "Route to Pierre's General Store counter for {target}."),
    "break": ("blacksmith", "Route to Clint's Blacksmith counter to process {target}."),
    "build": ("carpenter", "Route to Robin's Carpenter Shop menu for {target}."),
    "demolish": ("carpenter", "Route to Robin's Carpenter Shop menu to demolish {target}."),
    "exchange": ("exchange", "Route to the relevant exchange or reward context for {target}."),
    "jojamart": ("jojamart", "Route to the JojaMart service counter for {target}."),
    "location": ("navigation", "Route through map exits toward {target}."),
    "move": ("carpenter", "Route to Robin's Carpenter Shop menu to move {target}."),
    "purchase": ("shop", "Route to the relevant shop counter or store menu for {target}."),
    "purchase_animal": ("animal_shop", "Route to Marnie's Ranch animal purchase menu for {target}."),
    "read": ("interaction", "Route to the interaction point for {target}."),
    "repair": ("objective", "Route to the repair interaction for {target}."),
    "reward": ("quest_reward", "Route to the reward hand-in point for {target}."),
    "sell": ("seller", "Route to the relevant seller or selling point for {target}."),
    "sell_animal": ("animal_shop", "Route to Marnie's Ranch animal sale menu for {target}."),
    "silo": ("grass_patch_or_farm_area", "Route to a grassy farm area and cut grass with the scythe to collect {target}."),
    "upgrade_farmhouse": ("carpenter", "Route to Robin's Carpenter Shop menu for the farmhouse upgrade."),
    "upgrade_tool": ("blacksmith", "Route to Clint's Blacksmith counter for {target}."),
}

_ANIMAL_TRADE_KEYWORDS = (
    "chicken",
    "cow",
    "goat",
    "duck",
    "rabbit",
    "sheep",
    "pig",
    "dinosaur",
)


def _select_acquisition_target(
    evaluator: str,
    object_name: str,
    tool_name: str,
) -> str:
    if evaluator == "harvest" and object_name:
        return object_name
    if evaluator in _DIRECT_OBJECT_ACQUISITION_EVALUATORS and object_name:
        return object_name
    if evaluator in _TOOL_ACQUISITION_EVALUATORS and tool_name:
        return tool_name
    return ""


def _looks_like_seed_or_fertilizer(item_name: str) -> bool:
    lowered = str(item_name or "").strip().lower()
    return any(keyword in lowered for keyword in _SEED_OR_FERTILIZER_KEYWORDS)


_LEADING_COUNT_PATTERN = re.compile(
    r"^(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an)\s+",
    re.IGNORECASE,
)


def _capitalize_words(text: str) -> str:
    normalized = _format_display_name(text).lower()
    if not normalized:
        return ""
    return " ".join(word[:1].upper() + word[1:] for word in normalized.split())


def _cleanup_fallback_target(text: str, strip_leading_count: bool = True) -> str:
    cleaned = str(text or "").strip().lower()
    cleaned = re.sub(r"\b(?:from|to|at|inside|in|on|via|using|with)\b.*$", "", cleaned).strip()
    if strip_leading_count:
        cleaned = _LEADING_COUNT_PATTERN.sub("", cleaned).strip()
    cleaned = re.sub(r"^(?:the|a|an)\s+", "", cleaned).strip()
    return _capitalize_words(cleaned)


def _extract_seed_or_fertilizer_target(text: str) -> str:
    normalized = str(text or "").strip().lower()
    for keyword in (
        "retaining soil",
        "speed gro",
        "speed-gro",
        "fertilizer",
        "seeds",
        "seed",
        "soil",
    ):
        index = normalized.find(keyword)
        if index >= 0:
            candidate = normalized[: index + len(keyword)].strip()
            return _cleanup_fallback_target(candidate)
    return ""


def _contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = str(text or "").strip().lower()
    return any(keyword in lowered for keyword in keywords)


def _extract_route_suffix_target(normalized_task: str, pattern: str) -> str:
    match = re.search(pattern, normalized_task)
    if not match:
        return ""
    return _cleanup_fallback_target(match.group(1), strip_leading_count=False)


def _looks_like_animal_trade_target(text: str) -> bool:
    return _contains_any_keyword(text, _ANIMAL_TRADE_KEYWORDS)


def _infer_task_spec_from_text(task_name: str) -> Dict[str, Any]:
    normalized_task = _normalize_task_lookup_key(task_name)
    if not normalized_task:
        return {}

    route_target = _extract_route_suffix_target(
        normalized_task,
        r"\b(?:go|walk|head|travel|navigate|route|move)\s+(?:to|toward|towards|into)\s+(.+?)(?:\s+and\s+.+)?$",
    )

    breakup_match = re.search(r"\bbreak\s+up\s+with\s+(.+)$", normalized_task)
    if breakup_match:
        return {
            "evaluator": "breakup",
            "object": _cleanup_fallback_target(breakup_match.group(1), strip_leading_count=False),
            "tool": "Wilted Bouquet",
        }

    propose_match = re.search(r"\bpropose(?:\s+to)?\s+(.+)$", normalized_task)
    if propose_match:
        return {
            "evaluator": "propose",
            "object": _cleanup_fallback_target(propose_match.group(1), strip_leading_count=False),
            "tool": "Mermaid's Pendant",
        }

    date_match = re.search(r"\bdate\s+(.+)$", normalized_task)
    if date_match:
        return {
            "evaluator": "date",
            "object": _cleanup_fallback_target(date_match.group(1), strip_leading_count=False),
            "tool": "Bouquet",
        }

    if route_target and re.search(r"\band\s+break\s+up$", normalized_task):
        return {
            "evaluator": "breakup",
            "object": route_target,
            "tool": "Wilted Bouquet",
        }

    if route_target and re.search(r"\band\s+propose$", normalized_task):
        return {
            "evaluator": "propose",
            "object": route_target,
            "tool": "Mermaid's Pendant",
        }

    if route_target and re.search(r"\band\s+date$", normalized_task):
        return {
            "evaluator": "date",
            "object": route_target,
            "tool": "Bouquet",
        }

    if "upgrade farmhouse" in normalized_task or "upgrade the farmhouse" in normalized_task:
        return {
            "evaluator": "upgrade_farmhouse",
            "object": "Farmhouse",
        }

    upgrade_match = re.match(r"^upgrade(?:\s+to)?\s+(.+)$", normalized_task)
    if upgrade_match:
        return {
            "evaluator": "upgrade_tool",
            "object": _cleanup_fallback_target(upgrade_match.group(1)),
        }

    purchase_animal_match = re.search(r"\b(?:purchase|buy|get|obtain)\s+(.+)$", normalized_task)
    if purchase_animal_match and _looks_like_animal_trade_target(purchase_animal_match.group(1)):
        return {
            "evaluator": "purchase_animal",
            "object": _cleanup_fallback_target(purchase_animal_match.group(1)),
        }

    sell_animal_match = re.search(r"\bsell\s+(.+)$", normalized_task)
    if sell_animal_match and _looks_like_animal_trade_target(sell_animal_match.group(1)):
        return {
            "evaluator": "sell_animal",
            "object": _cleanup_fallback_target(sell_animal_match.group(1)),
        }

    build_match = re.search(r"\bbuild\s+(.+)$", normalized_task)
    if build_match:
        return {
            "evaluator": "build",
            "object": _cleanup_fallback_target(build_match.group(1)),
        }

    move_match = re.search(r"\bmove\s+(.+)$", normalized_task)
    if move_match and not normalized_task.startswith("move to "):
        return {
            "evaluator": "move",
            "object": _cleanup_fallback_target(move_match.group(1)),
        }

    demolish_match = re.search(r"\bdemolish\s+(.+)$", normalized_task)
    if demolish_match:
        return {
            "evaluator": "demolish",
            "object": _cleanup_fallback_target(demolish_match.group(1)),
        }

    break_match = re.search(r"\bbreak\s+(.+)$", normalized_task)
    if break_match:
        return {
            "evaluator": "break",
            "object": _cleanup_fallback_target(break_match.group(1)),
        }

    ship_match = re.search(r"\bship\s+(.+)$", normalized_task)
    if ship_match:
        return {
            "evaluator": "sell",
            "object": _cleanup_fallback_target(ship_match.group(1)),
            "tool": "Shipping Bin",
        }

    sell_match = re.search(r"\bsell\s+(.+)$", normalized_task)
    if sell_match:
        inferred = {
            "evaluator": "sell",
            "object": _cleanup_fallback_target(sell_match.group(1)),
        }
        if "shipping bin" in normalized_task:
            inferred["tool"] = "Shipping Bin"
        return inferred

    purchase_match = re.search(r"\b(?:purchase|buy|get|obtain)\s+(.+)$", normalized_task)
    if purchase_match:
        return {
            "evaluator": "purchase",
            "object": _cleanup_fallback_target(purchase_match.group(1)),
        }

    give_to_match = re.match(r"^(?:give|gift)\s+(.+?)\s+to\s+.+$", normalized_task)
    if give_to_match:
        return {
            "evaluator": "gift",
            "tool": _cleanup_fallback_target(give_to_match.group(1)),
        }

    give_count_match = re.search(
        r"(?:^|\s)(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+(.+)$",
        normalized_task,
    )
    if normalized_task.startswith(("give ", "gift ")) and give_count_match:
        return {
            "evaluator": "gift",
            "tool": _cleanup_fallback_target(give_count_match.group(1)),
        }

    sow_with_match = re.match(r"^(sow|plant|fertilize)\s+.+?\s+with\s+(.+)$", normalized_task)
    if sow_with_match:
        evaluator = "fertilize" if sow_with_match.group(1) == "fertilize" else "sow"
        return {
            "evaluator": evaluator,
            "tool": _cleanup_fallback_target(sow_with_match.group(2)),
        }

    sow_match = re.match(r"^(sow|plant|fertilize)\s+(.+)$", normalized_task)
    if sow_match:
        evaluator = "fertilize" if sow_match.group(1) == "fertilize" else "sow"
        target_item = _extract_seed_or_fertilizer_target(sow_match.group(2))
        if target_item:
            return {
                "evaluator": evaluator,
                "tool": target_item,
            }

    clear_with_match = re.match(r"^(clear|kill)\s+.+?\s+with\s+(.+)$", normalized_task)
    if clear_with_match:
        return {
            "evaluator": clear_with_match.group(1),
            "tool": _cleanup_fallback_target(clear_with_match.group(2)),
        }

    if "hay" in normalized_task and any(token in normalized_task for token in ("forage", "collect", "cut", "scythe", "silo")):
        return {
            "evaluator": "silo",
            "object": "Hay",
            "tool": "Scythe" if "scythe" in normalized_task else "",
        }

    if normalized_task.startswith("till "):
        return {"evaluator": "till", "tool": "Hoe"}
    if normalized_task.startswith("water "):
        return {"evaluator": "water", "tool": "Watering Can"}

    location_match = re.match(
        r"^(?:go|walk|head|travel|navigate|route|move)\s+(?:to|toward|towards|into)\s+(.+)$",
        normalized_task,
    )
    if location_match:
        return {
            "evaluator": "location",
            "object": _cleanup_fallback_target(location_match.group(1), strip_leading_count=False),
        }

    return {}


def build_task_acquisition_context(task_name: str) -> Dict[str, str]:
    """Build soft acquisition hints for prompt templates from benchmark task metadata."""
    normalized_task = _normalize_task_lookup_key(task_name)
    spec = _get_task_spec(task_name)
    if not spec:
        spec = _infer_task_spec_from_text(task_name)
    evaluator = str(spec.get("evaluator", "") or "").strip().lower()
    object_name = _format_display_name(spec.get("object", ""))
    tool_name = _format_display_name(spec.get("tool", ""))
    preloaded_seed_item = _get_preloaded_seed_or_fertilizer_item(spec)
    generic_animal_pet_task = evaluator == "pet" and object_name.lower() == "animal"
    animal_building_location_task = _is_animal_building_location_task(evaluator, object_name)
    animal_door_task = _is_animal_door_task(evaluator, object_name)
    pet_bowl_fill_task = _is_pet_bowl_fill_task(evaluator, object_name)
    feeding_bench_fill_task = _is_feeding_bench_fill_task(evaluator, object_name)
    egg_harvest_task = _is_egg_harvest_task(evaluator, object_name)
    milk_harvest_task = _is_milk_harvest_task(evaluator, object_name)
    incubate_chicken_task = _is_incubate_chicken_task(evaluator, object_name)
    pet_friendship_task = _is_pet_friendship_task(evaluator, object_name)
    relationship_task = _is_relationship_task(evaluator)
    shipping_bin_sell_task = _is_shipping_bin_sell_task(task_name, spec)
    hay_or_silo_task = _is_hay_or_silo_task(task_name, spec)
    normalized_object_name = re.sub(r"[^a-z0-9]+", " ", object_name.lower()).strip()
    tool_required_harvest_task = _is_tool_required_harvest_task(
        task_name,
        evaluator,
        tool_name,
        egg_harvest_task,
        spec,
    )

    if _is_cultivate_and_harvest_task(task_name, spec) and preloaded_seed_item:
        target_item = preloaded_seed_item
    elif generic_animal_pet_task:
        target_item = object_name or "Animal"
    elif milk_harvest_task:
        target_item = tool_name or "Milk Pail"
    elif tool_required_harvest_task:
        target_item = tool_name
    elif evaluator in {"sell", "sell_animal"}:
        target_item = object_name or "the item"
    elif hay_or_silo_task:
        target_item = object_name or "Hay"
    elif relationship_task:
        target_item = tool_name or object_name
    elif any(
        (
            animal_building_location_task,
            animal_door_task,
            pet_bowl_fill_task,
            feeding_bench_fill_task,
            egg_harvest_task,
            milk_harvest_task,
            incubate_chicken_task,
            pet_friendship_task,
        )
    ):
        target_item = object_name or tool_name or "Animal"
    else:
        target_item = _select_acquisition_target(
            evaluator=evaluator,
            object_name=object_name,
            tool_name=tool_name,
        )
    source_type = ""
    source_detail = ""

    if target_item:
        hint = _SOURCE_HINTS_BY_EVALUATOR.get(evaluator)
        if _is_cultivate_and_harvest_task(task_name, spec) and preloaded_seed_item:
            source_type = "inventory_preloaded"
            source_detail = (
                f"{preloaded_seed_item} is preloaded for this benchmark. Start by tilling soil, planting it, "
                f"watering once today, then keep watering on later days until {object_name or 'the crop'} is ready to harvest."
            )
        elif generic_animal_pet_task:
            source_type = "animal_housing"
            source_detail = (
                "If no animals are visible nearby, route to the coop first and enter it to look for chickens. "
                "If the coop has no reachable animals, check the barn next."
            )
        elif animal_building_location_task:
            source_type = "farm_building"
            source_detail = (
                f"Stay on the farm, route to the {object_name or 'animal building'} entrance, "
                "and enter it instead of circling around outside."
            )
        elif animal_door_task:
            source_type = "animal_door"
            source_detail = (
                f"Route to the outside of the {object_name or 'animal building'} and interact with its animal door hatch. "
                "Entering the building does not change the animal-door state."
            )
        elif pet_bowl_fill_task:
            source_type = "pet_area"
            source_detail = (
                "Route to the pet bowl area by the farmhouse, fill the Watering Can first if needed, "
                "then fill the bowl."
            )
        elif feeding_bench_fill_task:
            source_type = "animal_housing"
            source_detail = (
                "Route into the coop or barn interior, take hay from the hopper if needed, "
                "and place it onto the feeding bench."
            )
        elif egg_harvest_task:
            source_type = "animal_housing"
            source_detail = "Route into the coop interior and collect a visible egg from the floor or nesting area."
        elif milk_harvest_task and _task_starts_with_item(spec, target_item):
            source_type = "inventory_preloaded"
            source_detail = (
                f"{target_item} is preloaded for this benchmark. Stay on the farm, route into the barn interior, "
                "and use it on an adult milkable animal."
            )
        elif milk_harvest_task:
            source_type = "animal_tool"
            source_detail = (
                f"If {tool_name or 'the Milk Pail'} is missing, route to Marnie's Ranch counter for one first. "
                "Then enter the barn interior and use it on an adult milkable animal."
            )
        elif tool_required_harvest_task:
            source_type = "inventory_or_tool"
            source_detail = (
                f"Check toolbar or inventory for {tool_name} before changing routes. "
                f"If {tool_name} is ready, use it to collect {object_name or 'the target'}."
            )
        elif shipping_bin_sell_task:
            source_type = "shipping_bin"
            source_detail = (
                f"Return to the farm shipping bin and deposit {target_item}. "
                "Do not route to a shop counter when the task explicitly calls for the shipping bin."
            )
        elif evaluator == "sell_animal":
            source_type = "animal_shop"
            source_detail = (
                f"Route to Marnie's Ranch animal sale menu and sell {target_item} there."
            )
        elif hay_or_silo_task:
            source_type = "grass_patch_or_farm_area"
            source_detail = (
                "Route to a grassy farm area, equip the Scythe, and cut grass so the hay is stored through the silo path."
            )
        elif incubate_chicken_task:
            source_type = "animal_housing"
            source_detail = (
                "Route into the coop interior, locate the incubator, and interact with it while holding a hatchable egg if one is available."
            )
        elif pet_friendship_task:
            source_type = "pet_routine"
            source_detail = (
                f"{object_name or 'The pet'} gains friendship over multiple days. Route to it, pet it once today, "
                "then sleep and repeat on later days."
            )
        elif relationship_task:
            source_type = "inventory_or_source"
            source_detail = (
                f"Check inventory first for {target_item}; once it is ready, route to {object_name or 'the target NPC'} and complete the relationship interaction."
            )
        elif evaluator == "kill" and target_item:
            source_type = "enemy_search"
            source_detail = (
                f"Keep {target_item} combat-focused: equip {target_item}, search the current combat area for a visible target enemy, "
                "and only attack once the enemy is adjacent or clearly reachable."
            )
        elif evaluator == "harvest" and normalized_task.startswith(("forage ", "forage_")):
            source_type = "forage_search"
            source_detail = (
                f"Route through outdoor forageable areas and keep searching until {target_item} is explicitly visible; "
                "only interact when the target forage item is grounded in the current facts."
            )
        elif evaluator == "location" and normalized_object_name in {"bus stop", "busstop"}:
            source_type = "navigation"
            source_detail = (
                "If the player is still on the Farm map, route toward the east exit to reach the Bus Stop. "
                "Do not drift toward the pet bowl path or the southern forest when the task specifically targets Bus Stop."
            )
        elif evaluator == "location" and normalized_object_name == "backwoods":
            source_type = "navigation"
            source_detail = (
                "If the player is still on the Farm map, route toward the north exit near the pet bowl to reach the Backwoods. "
                "Do not treat the Bus Stop path as the Backwoods route."
            )
        elif hint is not None:
            source_type = hint[0]
            source_detail = hint[1].format(target=target_item)
        elif evaluator in {"sow", "fertilize"} and _task_starts_with_item(spec, target_item):
            source_type = "inventory_preloaded"
            source_detail = (
                f"{target_item} is preloaded for this benchmark. Check inventory or toolbar first "
                f"and do not route to a shop unless {target_item} is genuinely missing."
            )
        elif evaluator in {"sow", "fertilize"} and _looks_like_seed_or_fertilizer(target_item):
            source_type = "inventory_or_shop"
            source_detail = (
                f"Check the current Inventory facts first; if {target_item} is missing there, route to Pierre's General Store "
                f"or another confirmed seller for {target_item}. When the Inventory already lists slot_index entries, treat that "
                "as the current known inventory state instead of reopening the menu just to search hidden slots."
            )
        elif evaluator == "fill" and target_item:
            source_type = "inventory_or_container"
            source_detail = (
                f"Check inventory or the relevant storage source for {target_item} before committing to the fill action."
            )
        elif evaluator == "gift" and target_item:
            source_type = "inventory_or_source"
            source_detail = (
                f"Check inventory first; if {target_item} is missing, route toward a confirmed source for {target_item}."
            )
        elif evaluator in {"clear", "kill", "till", "water"} and target_item:
            source_type = "inventory_or_tool"
            source_detail = (
                f"Check toolbar or inventory for {target_item} before changing routes or committing to tool actions."
            )

    return {
        "target_item": target_item,
        "source_type": source_type,
        "source_detail": source_detail,
    }


def _extract_preloaded_item_names(spec: Dict[str, Any]) -> list[str]:
    item_names: list[str] = []
    init_commands = spec.get("init_commands", [])
    if not isinstance(init_commands, list):
        return item_names

    for command in init_commands:
        if not isinstance(command, str):
            continue
        match = re.search(r'add_item_by_name\("([^"]+)"', command)
        if match:
            item_names.append(_format_display_name(match.group(1)))

    return item_names


def _task_starts_with_item(spec: Dict[str, Any], item_name: str) -> bool:
    normalized_item = _format_display_name(item_name).lower()
    if not normalized_item:
        return False

    preloaded_items = {
        item.lower()
        for item in _extract_preloaded_item_names(spec)
        if item
    }
    return normalized_item in preloaded_items


def _get_preloaded_seed_or_fertilizer_item(spec: Dict[str, Any]) -> str:
    for item_name in _extract_preloaded_item_names(spec):
        if _looks_like_seed_or_fertilizer(item_name):
            return item_name
    return ""


def _is_cultivate_and_harvest_task(task_name: str, spec: Dict[str, Any]) -> bool:
    normalized_task = _normalize_task_lookup_key(task_name)
    if "cultivate and harvest" in normalized_task:
        return True

    evaluator = str(spec.get("evaluator", "") or "").strip().lower()
    return evaluator == "harvest" and bool(_get_preloaded_seed_or_fertilizer_item(spec))


def _normalized_object_name(value: Any) -> str:
    return _format_display_name(value).lower()


def _is_animal_building_name(value: Any) -> bool:
    normalized = _normalized_object_name(value)
    return "coop" in normalized or "barn" in normalized


def _is_animal_building_location_task(evaluator: str, object_name: str) -> bool:
    return evaluator == "location" and _is_animal_building_name(object_name)


def _is_animal_door_task(evaluator: str, object_name: str) -> bool:
    return evaluator in {"open", "close"} and _is_animal_building_name(object_name)


def _is_pet_bowl_fill_task(evaluator: str, object_name: str) -> bool:
    return evaluator == "fill" and _normalized_object_name(object_name) == "pet bowl"


def _is_feeding_bench_fill_task(evaluator: str, object_name: str) -> bool:
    return evaluator == "fill" and _normalized_object_name(object_name) == "feeding bench"


def _is_egg_harvest_task(evaluator: str, object_name: str) -> bool:
    return evaluator == "harvest" and _normalized_object_name(object_name) == "egg"


def _is_milk_harvest_task(evaluator: str, object_name: str) -> bool:
    return evaluator == "harvest" and _normalized_object_name(object_name) == "milk"


def _is_incubate_chicken_task(evaluator: str, object_name: str) -> bool:
    return evaluator == "incubate" and "chicken" in _normalized_object_name(object_name)


def _is_pet_friendship_task(evaluator: str, object_name: str) -> bool:
    return evaluator == "friendship" and _normalized_object_name(object_name) in {"cat", "dog", "pet"}


def _is_relationship_task(evaluator: str) -> bool:
    return evaluator in {"date", "breakup", "propose"}


def _is_shipping_bin_sell_task(task_name: str, spec: Dict[str, Any]) -> bool:
    normalized_task = _normalize_task_lookup_key(task_name)
    tool_name = _format_display_name(spec.get("tool", "")).lower()
    return str(spec.get("evaluator", "") or "").strip().lower() == "sell" and (
        "shipping bin" in normalized_task or tool_name == "shipping bin"
    )


def _is_hay_or_silo_task(task_name: str, spec: Dict[str, Any]) -> bool:
    normalized_task = _normalize_task_lookup_key(task_name)
    evaluator = str(spec.get("evaluator", "") or "").strip().lower()
    object_name = _normalized_object_name(spec.get("object", ""))
    tool_name = _format_display_name(spec.get("tool", "")).lower()
    if evaluator == "silo":
        return True
    return "hay" in normalized_task and any(token in normalized_task for token in ("forage", "collect", "cut", "scythe")) and (
        object_name == "hay" or tool_name == "scythe" or not object_name
    )


def _is_tool_required_harvest_task(
    task_name: str,
    evaluator: str,
    tool_name: str,
    egg_harvest_task: bool,
    spec: Dict[str, Any],
) -> bool:
    return (
        evaluator == "harvest"
        and bool(str(tool_name or "").strip())
        and not egg_harvest_task
        and not _is_cultivate_and_harvest_task(task_name, spec)
    )


def _is_multi_tool_clear_toolset(tool_name: str) -> bool:
    normalized = str(tool_name or "").strip()
    if not normalized:
        return False
    parts = [
        part.strip()
        for part in re.split(r",|\band\b|/|\|", normalized, flags=re.IGNORECASE)
        if part.strip()
    ]
    return len(parts) > 1


def build_initial_subtask(task_name: str) -> str:
    task_name = str(task_name or "").strip()
    spec = _get_task_spec(task_name)
    if not spec:
        spec = _infer_task_spec_from_text(task_name)
    evaluator = str(spec.get("evaluator", "") or "").strip().lower()
    object_name = str(spec.get("object", "") or "").strip()
    tool_name = str(spec.get("tool", "") or "").strip()
    target_phrase = _format_target_phrase(spec.get("quantity"), object_name)
    starts_with_tool = _task_starts_with_item(spec, tool_name)
    preloaded_seed_item = _get_preloaded_seed_or_fertilizer_item(spec)
    generic_animal_pet_task = evaluator == "pet" and object_name.lower() == "animal"
    animal_building_location_task = _is_animal_building_location_task(evaluator, object_name)
    animal_door_task = _is_animal_door_task(evaluator, object_name)
    pet_bowl_fill_task = _is_pet_bowl_fill_task(evaluator, object_name)
    feeding_bench_fill_task = _is_feeding_bench_fill_task(evaluator, object_name)
    egg_harvest_task = _is_egg_harvest_task(evaluator, object_name)
    milk_harvest_task = _is_milk_harvest_task(evaluator, object_name)
    incubate_chicken_task = _is_incubate_chicken_task(evaluator, object_name)
    pet_friendship_task = _is_pet_friendship_task(evaluator, object_name)
    relationship_task = _is_relationship_task(evaluator)
    shipping_bin_sell_task = _is_shipping_bin_sell_task(task_name, spec)
    hay_or_silo_task = _is_hay_or_silo_task(task_name, spec)
    normalized_object_name = re.sub(r"[^a-z0-9]+", " ", object_name.lower()).strip()
    tool_required_harvest_task = _is_tool_required_harvest_task(
        task_name,
        evaluator,
        tool_name,
        egg_harvest_task,
        spec,
    )

    if _is_cultivate_and_harvest_task(task_name, spec):
        crop_text = target_phrase or str(object_name or "the crop").strip().lower() or "the crop"
        seed_text = preloaded_seed_item or "the seeds"
        return (
            f"The current subtask is start the growth cycle for {crop_text}: till nearby soil if needed, "
            f"plant {seed_text}, water it once today, then return home and sleep so later days can advance until it is ready for the next watering or harvest."
        )

    if evaluator == "fertilize":
        if starts_with_tool:
            return (
                f"The current subtask is select {tool_name or 'the fertilizer'} and fertilize {target_phrase}."
            )
        if _looks_like_seed_or_fertilizer(tool_name):
            return (
                f"The current subtask is verify whether {tool_name or 'the fertilizer'} is explicitly visible in the current toolbar or inventory facts; "
                f"if it is not present there, route to Pierre's General Store to buy it, then prepare {target_phrase} for fertilizing."
            )
        return (
            f"The current subtask is obtain {tool_name or 'the fertilizer'} and get ready to fertilize {target_phrase}."
        )
    if evaluator == "sow":
        if starts_with_tool:
            return (
                f"The current subtask is select {tool_name or 'the seeds'} and sow {target_phrase}."
            )
        if _looks_like_seed_or_fertilizer(tool_name):
            return (
                f"The current subtask is verify whether {tool_name or 'the seeds'} are explicitly visible in the current toolbar or inventory facts; "
                f"if they are not present there, route to Pierre's General Store to buy them, then prepare {target_phrase} for sowing."
            )
        return (
            f"The current subtask is obtain {tool_name or 'the seeds'} and prepare {target_phrase} for sowing."
        )
    if evaluator == "till":
        return f"The current subtask is select the Hoe and till {target_phrase}."
    if evaluator == "water":
        return f"The current subtask is select the Watering Can and water {target_phrase}."
    if evaluator == "harvest":
        if _normalize_task_lookup_key(task_name).startswith(("forage ", "forage_")):
            return (
                f"The current subtask is route through forageable outdoor areas, keep searching until {target_phrase} is explicitly visible, "
                "and pick it up once it is grounded in the current facts."
            )
        if egg_harvest_task:
            return (
                "The current subtask is route into the coop interior and collect a visible egg from the floor or nesting area."
            )
        if milk_harvest_task:
            return (
                f"The current subtask is select {tool_name or 'the Milk Pail'}, route into the barn, "
                "and use it on a milkable animal."
            )
        if tool_required_harvest_task:
            return (
                f"The current subtask is select {tool_name}, move into position near {target_phrase}, "
                "and collect it with the required tool."
            )
        return f"The current subtask is move to {target_phrase} and harvest it."
    if evaluator == "fill":
        if pet_bowl_fill_task:
            return (
                f"The current subtask is select {tool_name or 'the Watering Can'}, route to the pet bowl by the farmhouse, "
                "and fill it."
            )
        if feeding_bench_fill_task:
            return (
                f"The current subtask is select {tool_name or 'Hay'}, route into the coop or barn, "
                "and place hay on the feeding bench."
            )
        if tool_name:
            return f"The current subtask is select {tool_name} and move into position to fill {target_phrase}."
        return f"The current subtask is move into position to fill {target_phrase}."
    if evaluator == "pet":
        if generic_animal_pet_task:
            return (
                "The current subtask is route to the coop first, enter it, and pet visible animals there; "
                "if none are reachable in the coop, check the barn next."
            )
        return f"The current subtask is reach {target_phrase} and start petting them."
    if evaluator == "open":
        if animal_door_task:
            return (
                f"The current subtask is route to the outside of the {str(object_name or 'animal building').strip().lower()} "
                "and open its animal door hatch."
            )
        return f"The current subtask is navigate to {target_phrase} and open it."
    if evaluator == "close":
        if animal_door_task:
            return (
                f"The current subtask is route to the outside of the {str(object_name or 'animal building').strip().lower()} "
                "and close its animal door hatch."
            )
        return f"The current subtask is navigate to {target_phrase} and close it."
    if evaluator == "clear":
        if tool_name:
            if _is_multi_tool_clear_toolset(tool_name):
                debris_text = str(object_name or "debris").strip().lower() or "debris"
                return (
                    "The current subtask is move off blocking structures toward the nearest visible "
                    f"{debris_text}, select the matching tool for that reachable target, and clear it."
                )
            return f"The current subtask is select {tool_name} and clear a nearby {str(object_name or 'target').strip().lower()}."
        return f"The current subtask is clear a nearby {str(object_name or 'target').strip().lower()}."
    if evaluator == "kill":
        enemy_text = str(object_name or target_phrase or "the target enemy").strip().lower() or "the target enemy"
        weapon_text = str(tool_name or "the weapon").strip() or "the weapon"
        return (
            f"The current subtask is select {weapon_text}, search for a visible {enemy_text}, "
            "move into attack range, and strike only when the enemy is adjacent or clearly reachable."
        )
    if evaluator == "craft":
        return f"The current subtask is gather the required materials and craft {target_phrase}."
    if evaluator == "purchase_animal":
        return f"The current subtask is route to Marnie's Ranch counter and buy {target_phrase}."
    if evaluator == "purchase":
        return f"The current subtask is navigate to the seller and buy {target_phrase}."
    if evaluator == "sell_animal":
        return f"The current subtask is route to Marnie's Ranch animal sale menu and sell {target_phrase}."
    if evaluator == "sell":
        if shipping_bin_sell_task:
            return f"The current subtask is return to the farm shipping bin and deposit {target_phrase}."
        return f"The current subtask is gather {target_phrase} and sell it."
    if evaluator == "break":
        return (
            f"The current subtask is route inside Clint's Blacksmith to the upper-right furnace/anvil "
            f"processing station and process {target_phrase} there, not the ore shop counter."
        )
    if evaluator == "build":
        return f"The current subtask is route to Robin's Carpenter Shop menu and build {target_phrase}."
    if evaluator == "move":
        return f"The current subtask is route to Robin's Carpenter Shop menu and move {target_phrase}."
    if evaluator == "upgrade_farmhouse":
        return "The current subtask is route to Robin's Carpenter Shop menu and start the farmhouse upgrade."
    if evaluator == "demolish":
        return f"The current subtask is route to Robin's Carpenter Shop menu and demolish {target_phrase}."
    if evaluator == "upgrade_tool":
        return f"The current subtask is route to Clint's counter and upgrade to {target_phrase}."
    if evaluator == "backpack":
        return f"The current subtask is route to Pierre's counter and buy {target_phrase}."
    if evaluator == "jojamart":
        return f"The current subtask is route to JojaMart's service counter and purchase {target_phrase}."
    if hay_or_silo_task:
        return "The current subtask is route to a grassy farm area, equip the Scythe, and cut grass to collect hay through the silo path."
    if evaluator == "talk":
        return f"The current subtask is reach {target_phrase} and talk to them."
    if evaluator == "gift":
        return f"The current subtask is prepare the gift and reach {target_phrase}."
    if evaluator == "friendship":
        if pet_friendship_task:
            return (
                f"The current subtask is route to the {str(object_name or 'pet').strip().lower()}, pet it once today, "
                "then sleep and repeat the routine on later days."
            )
        return f"The current subtask is reach {target_phrase} and keep making friendship progress."
    if evaluator == "date":
        return f"The current subtask is make sure Bouquet is ready, then reach {target_phrase} and ask them on a date."
    if evaluator == "breakup":
        return f"The current subtask is make sure Wilted Bouquet is ready, then reach {target_phrase} and break up with them."
    if evaluator == "propose":
        return f"The current subtask is make sure Mermaid's Pendant is ready, then reach {target_phrase} and propose."
    if evaluator == "location":
        if animal_building_location_task:
            return (
                f"The current subtask is route to the {str(object_name or 'animal building').strip().lower()} entrance "
                "and enter it."
            )
        if normalized_object_name in {"bus stop", "busstop"}:
            return "The current subtask is route across the farm toward the east exit that leads to the Bus Stop."
        if normalized_object_name == "backwoods":
            return "The current subtask is route across the farm toward the north exit near the pet bowl that leads to the Backwoods."
        return f"The current subtask is follow the map exits toward {target_phrase}."
    if evaluator == "incubate" and incubate_chicken_task:
        return "The current subtask is route into the coop, move next to the incubator, and place an egg into it if available."
    if evaluator == "sleep":
        return "The current subtask is return home, move next to the bed, and confirm the sleep prompt when it appears."
    if evaluator == "read":
        return f"The current subtask is navigate to {target_phrase} and read it."
    if evaluator == "repair":
        return f"The current subtask is reach {target_phrase} and repair it."
    if evaluator == "accept":
        return f"The current subtask is reach {target_phrase} and accept it."
    if evaluator == "reward":
        return f"The current subtask is reach {target_phrase} and claim the reward."
    if relationship_task:
        return f"The current subtask is prepare the required item and reach {target_phrase}."

    return f"The current subtask is start making concrete progress on {_humanize_task_name(task_name)}."


def build_initial_subtask_reasoning(task_name: str) -> str:
    return (
        "This is the bootstrap subtask derived from the benchmark task definition. "
        f"If task inference is unavailable, continue from this initial task for {str(task_name or '').strip() or 'the current task'}."
    )
