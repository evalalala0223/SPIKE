import re
from typing import Dict, Any, List, Optional
from copy import deepcopy

from cradle import constants
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider import BaseProvider
from stardojo.utils.task_bootstrap import build_task_acquisition_context
from stardojo.utils.stardew_prompt_state import extract_stardew_prompt_fact_fields

config = Config()
logger = Logger()
memory = LocalMemory()


_SURROUNDINGS_LINE_RE = re.compile(
    r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\](?:\([^)]*\))?\s*:\s*(.*)$"
)


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


def _get_latest_working_image() -> str:
    if hasattr(memory, "get_working_area_snapshot"):
        working_area = memory.get_working_area_snapshot()
    else:
        working_area = getattr(memory, "working_area", {}) or {}

    current_intro = working_area.get(constants.IMAGES_INPUT_TAG_NAME, [])
    if isinstance(current_intro, list):
        for item in reversed(current_intro):
            if not isinstance(item, dict):
                continue
            path = item.get(constants.IMAGE_PATH_TAG_NAME)
            if isinstance(path, str) and path.strip():
                return path.strip()
            if isinstance(path, list):
                for candidate in reversed(path):
                    if isinstance(candidate, str) and candidate.strip():
                        return candidate.strip()

    current_paths = working_area.get("image_paths", [])
    if isinstance(current_paths, str):
        text = current_paths.strip()
        if text:
            return text
    elif isinstance(current_paths, list):
        for candidate in reversed(current_paths):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()

    return ""


def _normalize_surroundings_text(surroundings: Any, player_position: Any = None) -> str:
    if isinstance(surroundings, str):
        return _canonicalize_surroundings_string(surroundings)
    if surroundings in (None, "", []):
        return ""

    def _normalize_position(position: Any) -> Optional[tuple[int, int]]:
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

    def _flatten_value(value: Any) -> List[str]:
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
                parts.extend(_flatten_value(item))
            return parts
        return [str(value).strip()]

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
                descriptors.extend(_flatten_value(value))

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

class TaskInferencePreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 use_screenshot_augmented = False,
                 use_video = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm
        self.use_screenshot_augmented = use_screenshot_augmented
        self.use_video = use_video

    def __call__(self):

        if not self.use_video:
            screenshot_path = memory.get_latest(constants.IMAGES_MEM_BUCKET, "")
            screenshot_augmnented_path = memory.get_latest(constants.AUGMENTED_IMAGES_MEM_BUCKET, "")

            if not self.use_screenshot_augmented:
                image_introduction = [
                    {
                        "introduction": "This screenshot is the current step of the game.",
                        "path": screenshot_path,
                        "assistant": ""
                    }
                ]
            else:
                image_introduction = [
                    {
                        "introduction": "This screenshot is the current step of the game.",
                        "path": screenshot_augmnented_path,
                        "assistant": ""
                    }
                ]

            processed_params = {
                "image_introduction": image_introduction
            }

        else:
            images = memory.get_recent_history(constants.IMAGES_MEM_BUCKET, config.event_count)
            reasonings = memory.get_recent_history('decision_making_reasoning', config.event_count)

            image_introduction = [
                {
                    "path": images[event_i],
                    "assistant": "",
                    "introduction": 'This is the {} screenshot of recent events. The description of this image: {}'.format(
                        ['first', 'second', 'third', 'fourth', 'fifth'][event_i], reasonings[event_i])
                } for event_i in range(config.event_count)
            ]

            processed_params = {
                "image_introduction": image_introduction,
                "event_count": config.event_count
            }

        if hasattr(memory, "update_working_area"):
            memory.update_working_area(processed_params)
        else:
            memory.working_area.update(processed_params)

        return processed_params

class TaskInferencePostprocessProvider(BaseProvider):

    def __init__(self,
                 *args,
                 use_subtask = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.use_subtask = use_subtask

    def __call__(self, response: Dict):

        processed_response = deepcopy(response)

        subtask_description = processed_response.get("subtask", "")
        if not subtask_description:
            logger.warn("[TaskInference] Missing 'subtask' in response; using empty fallback")

        processed_response.update({
            "subtask_description": subtask_description
        })

        if not self.use_subtask:
            processed_response_keys = list(processed_response.keys())
            for key in processed_response_keys:
                if "subtask" in key:
                    processed_response.pop(key)

        memory.update_info_history(processed_response)

        return processed_response


class RDR2TaskInferencePreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm

    def __call__(self):

        logger.write(f'RDR2 Task Inference Preprocess')

        task_description = memory.get_latest("task_description", "")
        screenshot_path = memory.get_latest(constants.IMAGES_MEM_BUCKET, "")

        processed_params = {
            "task_description": task_description,
            constants.IMAGES_MEM_BUCKET: screenshot_path
        }

        # Information summary preparation
        if len(memory.get_recent_history("decision_making_reasoning",
                                         memory.max_recent_steps)) == memory.max_recent_steps:
            logger.write(f'> Information summary call...')

            images = memory.get_recent_history(constants.IMAGES_MEM_BUCKET, config.event_count)
            reasonings = memory.get_recent_history('decision_making_reasoning', config.event_count)

            image_introduction = [
                {
                    "path": images[event_i], "assistant": "",
                    "introduction": 'This is the {} screenshot of recent events. The description of this image: {}'.format(
                        ['first', 'second', 'third', 'fourth', 'fifth'][event_i], reasonings[event_i])
                } for event_i in range(config.event_count)
            ]

            previous_summarization = memory.get_summarization()
            event_count = str(config.event_count)

            processed_params.update({
                "image_introduction": image_introduction,
                "previous_summarization": previous_summarization,
                "event_count": event_count
            })

        if hasattr(memory, "update_working_area"):
            memory.update_working_area(processed_params)
        else:
            memory.working_area.update(processed_params)

        return processed_params

class RDR2TaskInferencePostprocessProvider(BaseProvider):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, response: Dict):

        logger.write(f'RDR2 Task Inference Postprocess')

        processed_response = deepcopy(response)

        if "info_summary" not in response:
            response["info_summary"] = ""

        info_summary = response["info_summary"]

        processed_response.update({
            "summarization": info_summary
        })

        memory.update_info_history(processed_response)

        return processed_response


class StardewTaskInferencePreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm

    def __call__(self):

        logger.write(f'Stardew Task Inference Preprocess')

        def _safe_recent(key: str, default=None):
            try:
                if hasattr(memory, "get_latest"):
                    return memory.get_latest(key, default)
                values = memory.get_recent_history(key, k=1)
                if values:
                    latest = values[0]
                    if latest not in ("", None):
                        return latest
            except Exception:
                pass
            return default

        prompts = [
            "This screenshot is the current step of the game. The blue band represents the left side and the yellow band represents the right side."
        ]

        task_description = _safe_recent("task_description", "")
        previous_summarization = _safe_recent("summarization", "")
        substask_description = _safe_recent("subtask_description", "")
        substask_reasoning = _safe_recent("subtask_reasoning", "")
        toolbar_information = _safe_recent("toolbar_information", "")
        decision_making_reasoning = _safe_recent("decision_making_reasoning", "")
        self_reflection_reasoning = _safe_recent("self_reflection_reasoning", "")
        self_reflection_progress = _safe_recent("self_reflection_progress", 0)
        self_reflection_status_summary = _safe_recent("self_reflection_status_summary", "")
        previous_action = _safe_recent("pre_action", _safe_recent("action", ""))

        if hasattr(memory, "get_working_area_snapshot"):
            working_area = memory.get_working_area_snapshot()
        else:
            working_area = dict(getattr(memory, "working_area", {}) or {})
        if "gathered_info" not in working_area:
            working_area["gathered_info"] = _safe_recent("gathered_info", {})
        if task_description not in (None, ""):
            working_area["task_description"] = task_description
            working_area.setdefault("task", task_description)
            working_area.setdefault("main_task", task_description)
        if substask_description not in (None, ""):
            working_area["subtask_description"] = substask_description
        prompt_fact_fields = extract_stardew_prompt_fact_fields(
            state=working_area,
            gathered_info=working_area.get("gathered_info"),
        )
        if not toolbar_information:
            toolbar_information = prompt_fact_fields.get("toolbar_information", "")

        current_position = prompt_fact_fields.get("current_position") or _safe_recent("position", "")
        surroundings = _normalize_surroundings_text(
            prompt_fact_fields.get("surroundings", ""),
            current_position,
        )
        if not surroundings:
            surroundings = _normalize_surroundings_text(
                working_area.get("surroundings", ""),
                working_area.get("position", current_position),
            )
        if not surroundings:
            surroundings = str(_safe_recent("description", "") or "").strip()

        selected_position = prompt_fact_fields.get("selected_position")
        chosen_item = prompt_fact_fields.get("chosen_item")
        selected_item_name = str(prompt_fact_fields.get("selected_item_name", "") or "").strip()
        if not selected_item_name and isinstance(chosen_item, dict):
            selected_item_name = str(
                chosen_item.get("currentitem")
                or chosen_item.get("current_item")
                or chosen_item.get("item_name")
                or chosen_item.get("name")
                or chosen_item.get("item")
                or ""
            ).strip()

        current_toolbar_fact = ""
        if selected_item_name:
            if selected_position not in (None, ""):
                current_toolbar_fact = (
                    f"Current toolbar fact: {selected_item_name} is currently selected in "
                    f"slot_index {selected_position}. If this conflicts with previous_summarization, "
                    "trust the current toolbar fact and current toolbar information."
                )
            else:
                current_toolbar_fact = (
                    f"Current toolbar fact: {selected_item_name} is currently selected now. "
                    "If this conflicts with previous_summarization, trust the current toolbar fact "
                    "and current toolbar information."
                )
        elif toolbar_information:
            current_toolbar_fact = (
                "Current toolbar fact: trust the current toolbar information over "
                "previous_summarization when they conflict."
            )

        image_description = str(
            _safe_recent("image_description", "")
            or working_area.get("gathered_info", {}).get("image_description", "")
            or working_area.get("gathered_info", {}).get("description", "")
            or ""
        ).strip()

        latest_image = memory.get_latest(constants.AUGMENTED_IMAGES_MEM_BUCKET, "")
        if latest_image in ("", constants.NO_IMAGE, None):
            latest_image = memory.get_latest(constants.IMAGES_MEM_BUCKET, "")
        if latest_image in ("", constants.NO_IMAGE, None):
            latest_image = _get_latest_working_image()

        image_introduction = []
        if latest_image and latest_image != constants.NO_IMAGE:
            image_introduction.append(
                {
                    "introduction": prompts[-1],
                    "path": latest_image,
                    "assistant": ""
                }
            )

        acquisition_context = build_task_acquisition_context(task_description)

        processed_params = {
            "image_introduction": image_introduction,
            "image_paths": [latest_image] if image_introduction else [],
            "previous_summarization": previous_summarization,
            "task_description": task_description,
            "subtask_description": substask_description,
            "subtask_reasoning": substask_reasoning,
            "previous_reasoning": decision_making_reasoning,
            "previous_action": previous_action,
            "self_reflection_reasoning": self_reflection_reasoning,
            "self_reflection_progress": self_reflection_progress,
            "self_reflection_status_summary": self_reflection_status_summary,
            **prompt_fact_fields,
            "toolbar_information": toolbar_information,
            "current_toolbar_fact": current_toolbar_fact,
            "image_description": image_description,
            "surroundings": surroundings,
            "selected_position": selected_position,
            "selected_item_name": selected_item_name,
            **acquisition_context,
        }

        if hasattr(memory, "update_working_area"):
            memory.update_working_area(processed_params)
        else:
            memory.working_area.update(processed_params)

        return processed_params


class StardewTaskInferencePostprocessProvider(BaseProvider):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, response: Dict):

        logger.write(f'Stardew Task Inference Postprocess')

        processed_response = deepcopy(response)

        history_summary = response.get('history_summary') or response.get('summarization', '')

        subtask_description = response.get('subtask') or response.get('subtask_description', '')
        subtask_reasoning = response.get('subtask_reasoning', '')

        if not subtask_description:
            logger.warn("[TaskInference] Missing 'subtask' in response; using empty fallback")

        processed_response.update({
            'summarization': history_summary,
            'subtask_description': subtask_description,
            'subtask_reasoning': subtask_reasoning
        })

        memory.update_info_history(processed_response)

        return processed_response


class SkylinesTaskInferencePreprocessProvider(BaseProvider):
    """Skylines-specific preprocessor that provides all required parameters for task inference."""

    def __init__(self, *args,
                 gm: Any,
                 use_screenshot_augmented=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm
        self.use_screenshot_augmented = use_screenshot_augmented

    def __call__(self):

        logger.write("Skylines Task Inference Preprocess")

        # Get screenshot
        screenshot_path = memory.get_latest(constants.IMAGES_MEM_BUCKET, "")
        screenshot_augmented_path = memory.get_latest(constants.AUGMENTED_IMAGES_MEM_BUCKET, "")

        if not self.use_screenshot_augmented:
            image_introduction = [{
                "introduction": "This screenshot is the current step of the game.",
                "path": screenshot_path,
                "assistant": ""
            }]
        else:
            path = screenshot_augmented_path if screenshot_augmented_path else screenshot_path
            image_introduction = [{
                "introduction": "This screenshot is the current step of the game.",
                "path": path,
                "assistant": ""
            }]

        # Get parameters from memory with safe defaults
        def safe_get(key, default=""):
            return memory.get_latest(key, default)

        # Required parameters for Skylines task_inference.prompt
        task_description = safe_get("task_description", "Build and manage the city")
        subtask_description = safe_get("subtask_description", "")
        if not subtask_description:
            subtask_description = "Start building roads to connect to the highway"

        subtask_reasoning = safe_get("subtask_reasoning", "")
        budget = safe_get("budget", "Unknown")
        population = safe_get("population", "0")
        actions = safe_get("actions", "No previous action.")
        self_reflection_reasoning = safe_get("self_reflection_reasoning", "No previous reflection.")
        error_message = safe_get("error_message", "")
        previous_summarization = safe_get("summarization", "This is the beginning of the task.")

        processed_params = {
            "image_introduction": image_introduction,
            "task_description": task_description,
            "subtask_description": subtask_description,
            "subtask_reasoning": subtask_reasoning,
            "budget": str(budget),
            "population": str(population),
            "actions": actions,
            "self_reflection_reasoning": self_reflection_reasoning,
            "error_message": error_message,
            "previous_summarization": previous_summarization,
        }

        if hasattr(memory, "update_working_area"):
            memory.update_working_area(processed_params)
        else:
            memory.working_area.update(processed_params)

        return processed_params
