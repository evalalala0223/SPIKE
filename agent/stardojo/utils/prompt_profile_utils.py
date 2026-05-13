import os
import re
from copy import deepcopy
from typing import Any, Dict, Iterable, Optional, Tuple

from stardojo.log import Logger
from stardojo.utils.file_utils import assemble_project_path, read_resource_file
from stardojo.utils.task_bootstrap import get_task_spec

logger = Logger()


_GENERAL_ACTION_PLANNING_TEMPLATE = "./res/stardew/prompts/templates/action_planning_cortex.prompt"
_GENERAL_TASK_INFERENCE_TEMPLATE = "./res/stardew/prompts/templates/task_inference_cortex.prompt"
_GENERAL_SELF_REFLECTION_TEMPLATE = "./res/stardew/prompts/templates/self_reflection_general.prompt"
_GENERIC_INFORMATION_GATHERING_TEMPLATE = "./res/stardew/prompts/templates/information_gathering_cultivation.prompt"
_GENERIC_TOOLBAR_TEMPLATE = "./res/stardew/prompts/templates/information_toolbar_gathering_cultivation.prompt"
_PROFILE_SPECIFIC_BIGBRAIN_PROFILES = {"cultivation", "farm_clearup", "farm_ops", "shopping"}
_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}
_PROFILE_TEMPLATE_FALLBACKS: Dict[str, str] = {
    "action_planning": _GENERAL_ACTION_PLANNING_TEMPLATE,
    "information_gathering": _GENERIC_INFORMATION_GATHERING_TEMPLATE,
    "self_reflection": _GENERAL_SELF_REFLECTION_TEMPLATE,
    "task_inference": _GENERAL_TASK_INFERENCE_TEMPLATE,
    "information_toolbar_gathering": _GENERIC_TOOLBAR_TEMPLATE,
}


def runtime_single_action_big_brain_enabled() -> bool:
    return str(os.getenv("STARDOJO_BIG_BRAIN_SINGLE_ACTION", "")).strip().lower() in _TRUE_ENV_VALUES


def _strip_extra_action_example_steps(template_text: str) -> str:
    code_block_pattern = re.compile(r"```(?:python)?\s*\n(?P<body>.*?)```", re.IGNORECASE | re.DOTALL)

    def _rewrite_code_block(match: re.Match[str]) -> str:
        body = match.group("body")
        kept_lines = []
        skipping_extra_step = False
        for line in body.splitlines():
            if re.match(r"\s*#\s*Step\s+[2-9]\b", line, flags=re.IGNORECASE):
                skipping_extra_step = True
                continue
            if skipping_extra_step:
                if re.match(r"\s*#\s*Step\s+1\b", line, flags=re.IGNORECASE):
                    skipping_extra_step = False
                    kept_lines.append(line)
                continue
            kept_lines.append(line)

        return "```python\n" + "\n".join(kept_lines).rstrip() + "\n```"

    return code_block_pattern.sub(_rewrite_code_block, template_text)


def apply_runtime_action_planning_template_overrides(template_text: str) -> str:
    if not runtime_single_action_big_brain_enabled():
        return template_text

    text = str(template_text or "")
    replacements = (
        ("Plan 4 sequential actions", "Plan exactly 1 immediately executable action"),
        ("in this 4-step budget", "in this 1-step budget"),
        ("Plan a sequence of exactly 4 actions", "Plan exactly 1 immediate action"),
        ("These 4 actions should form", "This action should form"),
        ("You MUST output exactly 4 actions as a numbered sequence.", "You MUST output exactly 1 action."),
        ("The 4 actions should be", "The action should be"),
        ("Output exactly 4 concrete actions", "Output exactly 1 concrete action"),
        ("Your 4 actions must", "Your action must"),
        ("Your 4 actions should", "Your action should"),
        ("Prefer a 4-step horizon", "Prefer a single immediately executable action"),
        ("the 4-step horizon", "the single-action decision"),
        ("Your 4 actions should focus", "Your action should focus"),
    )
    for old, new in replacements:
        text = text.replace(old, new)

    regex_replacements = (
        (r"exactly\s+4\s+actions", "exactly 1 action"),
        (r"\b4-step\b", "1-step"),
        (r"\b4\s+actions\b", "1 action"),
        (r"multi-step plan", "single-action decision"),
        (r"as many steps as possible", "the current step"),
    )
    for pattern, new in regex_replacements:
        text = re.sub(pattern, new, text, flags=re.IGNORECASE)

    text = _strip_extra_action_example_steps(text)

    if "SINGLE-ACTION RUN MODE:" not in text:
        text = (
            text.rstrip()
            + "\n\nSINGLE-ACTION RUN MODE:\n"
            + "- For this run, ignore any instruction or example that asks for multiple actions.\n"
            + "- Output exactly 1 action in the Actions code block.\n"
            + "- Step 1 must be immediately executable from the current facts and must be the single action to execute now.\n"
            + "- Do not include any additional action.\n"
            + "\nExpected output shape:\n"
            + "Reasoning:\n"
            + "1. ...\n"
            + "Actions:\n"
            + "```python\n"
            + "# Step 1\n"
            + "move(x=1, y=0)\n"
            + "```\n"
        )
    return text

PROMPT_PROFILE_TEMPLATE_PATHS: Dict[str, Dict[str, str]] = {
    "cultivation": {
        "action_planning": "./res/stardew/prompts/templates/action_planning_cultivation.prompt",
        "information_gathering": "./res/stardew/prompts/templates/information_gathering_cultivation.prompt",
        "self_reflection": "./res/stardew/prompts/templates/self_reflection_cultivation.prompt",
        "task_inference": "./res/stardew/prompts/templates/task_inference_cultivation.prompt",
        "information_toolbar_gathering": "./res/stardew/prompts/templates/information_toolbar_gathering_cultivation.prompt",
    },
    "farm_clearup": {
        "action_planning": "./res/stardew/prompts/templates/action_planning_farm_clearup.prompt",
        "information_gathering": "./res/stardew/prompts/templates/information_gathering_farm_clearup.prompt",
        "self_reflection": "./res/stardew/prompts/templates/self_reflection_farm_clearup.prompt",
        "task_inference": "./res/stardew/prompts/templates/task_inference_farm_clearup.prompt",
        "information_toolbar_gathering": "./res/stardew/prompts/templates/information_toolbar_gathering_farm_clearup.prompt",
    },
    "shopping": {
        "action_planning": "./res/stardew/prompts/templates/action_planning_shopping.prompt",
        "information_gathering": _GENERIC_INFORMATION_GATHERING_TEMPLATE,
        "self_reflection": _GENERAL_SELF_REFLECTION_TEMPLATE,
        "task_inference": "./res/stardew/prompts/templates/task_inference_shopping.prompt",
        "information_toolbar_gathering": _GENERIC_TOOLBAR_TEMPLATE,
    },
    "farm_ops": {
        "action_planning": "./res/stardew/prompts/templates/action_planning_farm_ops.prompt",
        "information_gathering": _GENERIC_INFORMATION_GATHERING_TEMPLATE,
        "self_reflection": _GENERAL_SELF_REFLECTION_TEMPLATE,
        "task_inference": "./res/stardew/prompts/templates/task_inference_farm_ops.prompt",
        "information_toolbar_gathering": _GENERIC_TOOLBAR_TEMPLATE,
    },
    "navigation": {
        "action_planning": _GENERAL_ACTION_PLANNING_TEMPLATE,
        "information_gathering": _GENERIC_INFORMATION_GATHERING_TEMPLATE,
        "self_reflection": _GENERAL_SELF_REFLECTION_TEMPLATE,
        "task_inference": _GENERAL_TASK_INFERENCE_TEMPLATE,
        "information_toolbar_gathering": _GENERIC_TOOLBAR_TEMPLATE,
    },
    "social": {
        "action_planning": _GENERAL_ACTION_PLANNING_TEMPLATE,
        "information_gathering": _GENERIC_INFORMATION_GATHERING_TEMPLATE,
        "self_reflection": _GENERAL_SELF_REFLECTION_TEMPLATE,
        "task_inference": _GENERAL_TASK_INFERENCE_TEMPLATE,
        "information_toolbar_gathering": _GENERIC_TOOLBAR_TEMPLATE,
    },
    "crafting": {
        "action_planning": _GENERAL_ACTION_PLANNING_TEMPLATE,
        "information_gathering": _GENERIC_INFORMATION_GATHERING_TEMPLATE,
        "self_reflection": _GENERAL_SELF_REFLECTION_TEMPLATE,
        "task_inference": _GENERAL_TASK_INFERENCE_TEMPLATE,
        "information_toolbar_gathering": _GENERIC_TOOLBAR_TEMPLATE,
    },
    "combat": {
        "action_planning": _GENERAL_ACTION_PLANNING_TEMPLATE,
        "information_gathering": _GENERIC_INFORMATION_GATHERING_TEMPLATE,
        "self_reflection": _GENERAL_SELF_REFLECTION_TEMPLATE,
        "task_inference": _GENERAL_TASK_INFERENCE_TEMPLATE,
        "information_toolbar_gathering": _GENERIC_TOOLBAR_TEMPLATE,
    },
}

_CLEAR_EVALUATORS = {"clear"}
_CULTIVATION_EVALUATORS = {"till", "fertilize", "sow", "water"}
_FARM_OPS_EVALUATORS = {"pet", "open", "close", "fill", "incubate", "silo"}
_NAVIGATION_EVALUATORS = {"sleep", "location", "read", "repair", "accept", "reward", "quit", "complete_help", "complete_story"}
_SHOPPING_EVALUATORS = {
    "purchase",
    "purchase_animal",
    "sell",
    "sell_animal",
    "upgrade_tool",
    "break",
    "jojamart",
    "backpack",
    "build",
    "move",
    "upgrade_farmhouse",
    "demolish",
}
_SOCIAL_EVALUATORS = {
    "talk",
    "gift",
    "date",
    "breakup",
    "propose",
}
_CRAFTING_EVALUATORS = {"craft", "cook", "produce"}
_COMBAT_EVALUATORS = {"kill"}

_CROP_OR_FARM_PRODUCT_KEYWORDS = (
    "seed",
    "seeds",
    "crop",
    "parsnip",
    "garlic",
    "potato",
    "cauliflower",
    "bean",
    "strawberry",
    "tulip",
    "kale",
)
_ANIMAL_OR_PET_KEYWORDS = (
    "cat",
    "dog",
    "chicken",
    "cow",
    "goat",
    "duck",
    "rabbit",
    "sheep",
    "pig",
    "dinosaur",
    "pet",
)
_ANIMAL_PRODUCT_KEYWORDS = (
    "egg",
    "milk",
    "wool",
    "mayonnaise",
    "truffle",
)
_PURE_NAVIGATION_FALLBACK_KEYWORDS = (
    "go to",
    "go_to",
    "head to",
    "walk to",
    "travel to",
    "navigate to",
    "route to",
    "move to",
    "move into",
    "return home",
    "sleep",
    "bed",
    "door",
    "exit",
    "enter",
    "leave",
    "forage",
    "mine",
    "dig",
    "chop",
    "quest",
    "reward",
)
_SHOPPING_FALLBACK_KEYWORDS = (
    "purchase",
    "buy",
    "ship",
    "shipping bin",
    "shop",
    "store",
    "counter",
    "service",
    "joja",
    "sell",
    "shipping",
    "upgrade",
    "geode",
    "build",
    "demolish",
    "backpack",
)
_SOCIAL_FALLBACK_KEYWORDS = (
    "gift",
    "talk",
    "friendship",
    "date",
    "break up",
    "breakup",
    "propose",
    "bouquet",
    "wilted bouquet",
    "pendant",
)
_FARM_OPS_FALLBACK_KEYWORDS = (
    "pet bowl",
    "feeding bench",
    "incubator",
    "animal door",
    "coop door",
    "barn door",
    "hatch",
    "hay",
    "silo",
    "pet animal",
)
_BUILDING_SERVICE_OBJECT_KEYWORDS = (
    "coop",
    "barn",
    "silo",
    "farmhouse",
    "building",
    "house",
    "stable",
    "shed",
    "well",
    "mill",
    "slime hutch",
    "cabin",
)
_CRAFTING_FALLBACK_KEYWORDS = ("craft", "recipe", "cook", "furnace", "bomb", "sprinkler", "fence")
_COMBAT_FALLBACK_KEYWORDS = ("kill", "slime", "bug", "fly", "duggy", "grub", "crab", "sword")
_CLEAR_FALLBACK_KEYWORDS = (
    "clear",
    "weed",
    "weeds",
    "stone",
    "stones",
    "twig",
    "twigs",
    "debris",
    "grass",
    "obstacle",
    "obstacles",
    "boulder",
    "boulders",
    "log",
    "logs",
    "stump",
    "stumps",
)


def _normalize_text(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _normalize_profile_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _contains_phrase(text: str, keyword: str) -> bool:
    pattern = r"\b" + re.escape(keyword).replace(r"\ ", r"\s+") + r"\b"
    return re.search(pattern, text) is not None


def _contains_any_phrase(text: str, keywords: tuple[str, ...]) -> bool:
    return any(_contains_phrase(text, keyword) for keyword in keywords)


def _is_animal_or_pet(text: str) -> bool:
    return _contains_any(text, _ANIMAL_OR_PET_KEYWORDS)


def _looks_like_crop_harvest(task_text: str, object_text: str, tool_text: str) -> bool:
    if task_text.startswith("cultivate and harvest"):
        return True
    if task_text.startswith("harvest ") and _contains_any(object_text, _CROP_OR_FARM_PRODUCT_KEYWORDS):
        return True
    if _contains_any(object_text, _CROP_OR_FARM_PRODUCT_KEYWORDS):
        return True
    return False


def _looks_like_animal_product_harvest(object_text: str, tool_text: str) -> bool:
    if _contains_any(object_text, _ANIMAL_PRODUCT_KEYWORDS):
        return True
    if any(token in tool_text for token in ("milk pail", "shears")):
        return True
    return False


def _contains_shopping_intent(text: str) -> bool:
    if not text:
        return False
    if "break up" in text or "breakup" in text:
        return False
    if _contains_any_phrase(text, _SHOPPING_FALLBACK_KEYWORDS):
        return True
    if "break" in text and "geode" in text:
        return True
    if re.search(r"\bmove\b", text) and _contains_any_phrase(text, _BUILDING_SERVICE_OBJECT_KEYWORDS):
        return True
    return False


def _contains_social_intent(text: str) -> bool:
    return _contains_any_phrase(text, _SOCIAL_FALLBACK_KEYWORDS)


def _contains_farm_ops_intent(text: str) -> bool:
    return _contains_any_phrase(text, _FARM_OPS_FALLBACK_KEYWORDS)


def _is_pure_navigation_task(text: str) -> bool:
    if not _contains_any_phrase(text, _PURE_NAVIGATION_FALLBACK_KEYWORDS):
        return False
    if _contains_shopping_intent(text) or _contains_social_intent(text):
        return False
    if "hay" in text and any(token in text for token in ("purchase", "buy", "sell", "ship")):
        return False
    return True


def _looks_like_hay_or_silo_task(text: str) -> bool:
    if "silo" in text:
        return True
    if "hay" not in text:
        return False
    return any(token in text for token in ("forage", "collect", "cut", "scythe"))


def _infer_profile_from_task_spec(task_description: str, task_spec: Optional[Dict[str, Any]]) -> str:
    normalized_task = _normalize_text(task_description)
    spec = dict(task_spec or {})
    explicit_domain = _normalize_profile_name(spec.get("prompt_domain"))
    if explicit_domain in PROMPT_PROFILE_TEMPLATE_PATHS:
        return explicit_domain

    evaluator = _normalize_text(spec.get("evaluator"))
    object_text = _normalize_text(spec.get("object"))
    tool_text = _normalize_text(spec.get("tool"))

    if evaluator == "silo":
        return "farm_ops"

    if evaluator in _COMBAT_EVALUATORS:
        return "combat"
    if evaluator in _CRAFTING_EVALUATORS:
        return "crafting"
    if evaluator in _CLEAR_EVALUATORS:
        return "farm_clearup"
    if evaluator == "friendship":
        return "farm_ops" if _is_animal_or_pet(object_text) else "social"
    if evaluator in _SHOPPING_EVALUATORS:
        return "shopping"
    if evaluator in _NAVIGATION_EVALUATORS:
        return "navigation"
    if evaluator in _FARM_OPS_EVALUATORS:
        return "farm_ops"
    if evaluator in _CULTIVATION_EVALUATORS:
        return "cultivation"
    if evaluator in _SOCIAL_EVALUATORS:
        return "social"
    if evaluator == "harvest":
        if _looks_like_crop_harvest(normalized_task, object_text, tool_text):
            return "cultivation"
        if _looks_like_animal_product_harvest(object_text, tool_text):
            return "farm_ops"
        if normalized_task.startswith("chop "):
            return "farm_clearup"
        if normalized_task.startswith(("forage ", "mine ", "dig ")):
            return "navigation"
        if _contains_any(object_text, ("wood", "ore", "coal", "quartz", "amethyst", "clam", "daffodil", "leek", "horseradish", "cave carrot")):
            if "wood" in object_text:
                return "farm_clearup"
            return "navigation"
        return "cultivation"

    return ""


def infer_stardew_prompt_profile(task_description: str, task_spec: Optional[Dict[str, Any]] = None) -> str:
    spec = dict(task_spec or get_task_spec(task_description) or {})
    inferred = _infer_profile_from_task_spec(task_description, spec)
    if inferred:
        return inferred

    normalized = _normalize_text(task_description)
    if _looks_like_hay_or_silo_task(normalized):
        return "farm_ops"
    if normalized.startswith(("forage ", "forage_")):
        return "navigation"
    if _contains_any(normalized, _COMBAT_FALLBACK_KEYWORDS):
        return "combat"
    if _contains_any(normalized, _CRAFTING_FALLBACK_KEYWORDS):
        return "crafting"
    if "friendship" in normalized and _is_animal_or_pet(normalized):
        return "farm_ops"
    if _contains_shopping_intent(normalized):
        return "shopping"
    if _contains_social_intent(normalized):
        return "social"
    if _contains_farm_ops_intent(normalized):
        return "farm_ops"
    if _is_pure_navigation_task(normalized):
        return "navigation"
    if _contains_any(normalized, _CLEAR_FALLBACK_KEYWORDS):
        return "farm_clearup"
    return "cultivation"


def resolve_dual_brain_bigbrain_template_paths(prompt_profile: str) -> Tuple[Dict[str, str], bool]:
    resolved_paths, _audit, preserve_profile_templates = resolve_prompt_profile_template_paths(
        prompt_profile,
        template_keys=("action_planning", "task_inference"),
    )
    return (
        {
            "action_planning": resolved_paths["action_planning"],
            "task_inference": resolved_paths["task_inference"],
        },
        preserve_profile_templates,
    )


def _iter_template_keys(template_keys: Optional[Iterable[str]] = None) -> Tuple[str, ...]:
    if template_keys is None:
        return tuple(_PROFILE_TEMPLATE_FALLBACKS.keys())
    ordered = []
    seen = set()
    for key in template_keys:
        normalized = str(key or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return tuple(ordered)


def _resolve_single_prompt_profile_template_path(
    prompt_profile: str,
    template_key: str,
) -> Tuple[str, Dict[str, Any]]:
    normalized_profile = _normalize_profile_name(prompt_profile)
    profile_templates = PROMPT_PROFILE_TEMPLATE_PATHS.get(normalized_profile, {})
    requested_path = str(profile_templates.get(template_key, "") or "").strip()
    fallback_path = str(_PROFILE_TEMPLATE_FALLBACKS.get(template_key, "") or "").strip()

    requested_abs = assemble_project_path(requested_path) if requested_path else ""
    fallback_abs = assemble_project_path(fallback_path) if fallback_path else ""

    fallback_used = False
    if requested_path and os.path.exists(requested_abs):
        resolved_path = requested_path
    else:
        if requested_path:
            logger.warn(
                f"[PromptProfile] Missing template for profile={normalized_profile or 'default'}, "
                f"key={template_key}: {requested_abs}"
            )
        if not fallback_path:
            raise FileNotFoundError(
                f"No fallback template configured for profile={normalized_profile or 'default'}, key={template_key}"
            )
        if not os.path.exists(fallback_abs):
            raise FileNotFoundError(
                f"Fallback template is missing for key={template_key}: {fallback_abs}"
            )
        resolved_path = fallback_path
        fallback_used = requested_path != fallback_path

    return resolved_path, {
        "profile": normalized_profile or "default",
        "template_key": template_key,
        "requested_path": requested_path or fallback_path,
        "resolved_path": resolved_path,
        "fallback_used": fallback_used,
    }


def resolve_prompt_profile_template_paths(
    prompt_profile: str,
    *,
    existing_templates: Optional[Dict[str, str]] = None,
    template_keys: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]], bool]:
    resolved_templates = dict(existing_templates or {})
    template_audit: Dict[str, Dict[str, Any]] = {}
    for template_key in _iter_template_keys(template_keys):
        resolved_path, audit = _resolve_single_prompt_profile_template_path(
            prompt_profile,
            template_key,
        )
        resolved_templates[template_key] = resolved_path
        template_audit[template_key] = audit

    preserve_profile_templates = (
        _normalize_profile_name(prompt_profile) in _PROFILE_SPECIFIC_BIGBRAIN_PROFILES
    )
    return resolved_templates, template_audit, preserve_profile_templates


def load_prompt_profile_template_texts(
    prompt_profile: str,
    *,
    template_keys: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]], bool]:
    resolved_paths, template_audit, preserve_profile_templates = resolve_prompt_profile_template_paths(
        prompt_profile,
        template_keys=template_keys,
    )
    template_texts: Dict[str, str] = {}
    for template_key in _iter_template_keys(template_keys):
        resolved_path = resolved_paths[template_key]
        template_text = read_resource_file(assemble_project_path(resolved_path))
        if template_key == "action_planning":
            template_text = apply_runtime_action_planning_template_overrides(template_text)
        template_texts[template_key] = template_text
    return template_texts, template_audit, preserve_profile_templates


def sync_planner_prompt_templates(
    planner: Any,
    prompt_profile: str,
    *,
    template_keys: Optional[Iterable[str]] = None,
) -> Tuple[Dict[str, str], Dict[str, Dict[str, Any]], bool]:
    template_texts, template_audit, preserve_profile_templates = load_prompt_profile_template_texts(
        prompt_profile,
        template_keys=template_keys,
    )
    for template_key in _iter_template_keys(template_keys):
        template_text = template_texts[template_key]
        if isinstance(getattr(planner, "templates", None), dict):
            planner.templates[template_key] = template_text
        if template_key == "action_planning" and getattr(planner, "action_planning_", None) is not None:
            planner.action_planning_.template = template_text
        elif template_key == "task_inference" and getattr(planner, "task_inference_", None) is not None:
            planner.task_inference_.template = template_text
        elif template_key == "self_reflection" and getattr(planner, "self_reflection_", None) is not None:
            planner.self_reflection_.template = template_text
        elif template_key == "information_gathering" and getattr(planner, "information_gathering_", None) is not None:
            planner.information_gathering_.template = template_text

    resolved_paths = {
        template_key: str(template_audit[template_key]["resolved_path"])
        for template_key in _iter_template_keys(template_keys)
        if template_key in template_audit
    }
    return resolved_paths, template_audit, preserve_profile_templates


def build_task_specific_planner_params(
    planner_params: Dict[str, Any],
    task_description: str,
) -> Tuple[Dict[str, Any], str]:
    task_spec = get_task_spec(task_description)
    prompt_profile = infer_stardew_prompt_profile(task_description, task_spec=task_spec)
    resolved_params = deepcopy(planner_params or {})
    resolved_params["prompt_profile"] = prompt_profile

    prompt_paths = resolved_params.get("prompt_paths", {})
    templates = prompt_paths.get("templates", {})

    if not isinstance(templates, dict):
        return resolved_params, prompt_profile

    resolved_templates, template_audit, _preserve_profile_templates = resolve_prompt_profile_template_paths(
        prompt_profile,
        existing_templates=templates,
    )
    prompt_paths["templates"] = resolved_templates
    resolved_params["prompt_paths"] = prompt_paths
    resolved_params["_prompt_template_audit"] = template_audit

    return resolved_params, prompt_profile
