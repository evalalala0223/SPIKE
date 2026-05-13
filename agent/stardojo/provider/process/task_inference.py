import re
from typing import Dict, Any, List, Optional
from copy import deepcopy

from stardojo import constants
from stardojo.log import Logger
from stardojo.config import Config
from stardojo.memory import LocalMemory
from stardojo.provider import BaseProvider
from stardojo.utils.task_bootstrap import build_task_acquisition_context
from stardojo.utils.stardew_prompt_state import extract_stardew_prompt_fact_fields
from stardojo.utils.execution_feedback_utils import stable_snapshot_text

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
    working_area = getattr(memory, "working_area", {}) or {}

    current_intro = working_area.get(constants.IMAGES_INPUT_TAG_NAME, [])
    if isinstance(current_intro, list):
        for item in reversed(current_intro):
            if isinstance(item, dict):
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


def _prompt_text_value(value: Any) -> str:
    """Convert prompt params into a stable text form accepted by the LLM layer."""
    if value in (None, "", [], {}):
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (bool, int, float)):
        return str(value)
    if isinstance(value, dict):
        return stable_snapshot_text(value)
    if isinstance(value, list):
        if all(isinstance(item, str) for item in value):
            return "\n".join(str(item).strip() for item in value if str(item).strip())
        return stable_snapshot_text(value)
    return str(value).strip()

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
            screenshot_path = memory.get_recent_history(constants.IMAGES_MEM_BUCKET)[-1]
            screenshot_augmnented_path = memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET)[-1]

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

        subtask_description = processed_response["subtask"]

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

        task_description = memory.get_recent_history("task_description", k=1)[0]
        screenshot_path = memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=1)[0]

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

        prompts = [
            "This screenshot is the current step of the game. The blue band represents the left side and the yellow band represents the right side."
        ]
        working_area = dict(getattr(memory, "working_area", {}) or {})
        task_description = memory.get_recent_history("task_description", k=1)[0]
        substask_description = memory.get_recent_history("subtask_description", k=1)[0]
        if task_description not in (None, ""):
            working_area["task_description"] = task_description
            working_area.setdefault("task", task_description)
            working_area.setdefault("main_task", task_description)
        if substask_description not in (None, ""):
            working_area["subtask_description"] = substask_description
        prompt_fact_fields = extract_stardew_prompt_fact_fields(
            state=working_area,
            gathered_info=working_area.get("gathered_info", {}),
        )
        previous_summarization = memory.get_recent_history("summarization", 1)[0]
        substask_reasoning = memory.get_recent_history("subtask_reasoning", 1)[0]
        toolbar_information = memory.get_recent_history("toolbar_information", 1)[0]
        current_position = memory.get_recent_history("position", 1)[0]
        location = prompt_fact_fields.get("location", "") or working_area.get("location", "")
        facing_direction = (
            prompt_fact_fields.get("facing_direction", "")
            or working_area.get("facing_direction", "")
        )
        facing_position = (
            prompt_fact_fields.get("facing_position", "")
            or working_area.get("facing_position", "")
        )
        current_menu = (
            prompt_fact_fields.get("current_menu", "")
            or working_area.get("current_menu", "")
        )
        buildings = (
            prompt_fact_fields.get("buildings", "")
            or working_area.get("buildings", "")
            or working_area.get("gathered_info", {}).get("buildings", "")
        )
        buildings = _prompt_text_value(buildings)
        furniture = (
            prompt_fact_fields.get("furniture", "")
            or working_area.get("furniture", "")
            or working_area.get("gathered_info", {}).get("furniture", "")
        )
        furniture = _prompt_text_value(furniture)
        npcs = (
            prompt_fact_fields.get("npcs", "")
            or working_area.get("npcs", "")
            or working_area.get("gathered_info", {}).get("npcs", "")
        )
        npcs = _prompt_text_value(npcs)
        exits = (
            prompt_fact_fields.get("exits", "")
            or working_area.get("exits", "")
            or working_area.get("gathered_info", {}).get("exits", "")
        )
        exits = _prompt_text_value(exits)
        surroundings = _normalize_surroundings_text(
            memory.get_recent_history("surroundings", 1)[0],
            current_position,
        )
        if not surroundings:
            surroundings = _normalize_surroundings_text(
                working_area.get("surroundings", ""),
                working_area.get("position", current_position),
            )
        if not surroundings:
            surroundings = str(memory.get_recent_history("description", 1)[0] or "").strip()
        selected_position = memory.get_recent_history("selected_position", 1)[0]
        chosen_item = memory.get_recent_history("chosen_item", 1)[0]
        images = memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, 1)
        if not images or images[-1] in ("", constants.NO_IMAGE, None):
            images = memory.get_recent_history(constants.IMAGES_MEM_BUCKET, 1)
        decision_making_reasoning = memory.get_recent_history('decision_making_reasoning', 1)[0]
        self_reflection_reasoning = memory.get_recent_history('self_reflection_reasoning', 1)[0]
        pre_action = memory.get_recent_history("pre_action", k=1)[0]
        latest_image = images[-1] if images else ""
        if latest_image in ("", constants.NO_IMAGE, None):
            latest_image = _get_latest_working_image()
        image_description = str(
            memory.get_recent_history("image_description", 1)[0]
            or working_area.get("gathered_info", {}).get("image_description", "")
            or working_area.get("gathered_info", {}).get("description", "")
            or ""
        ).strip()

        selected_item_name = ""
        if isinstance(chosen_item, dict):
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
                    f"Current toolbar fact: {selected_item_name} is currently selected in slot_index {selected_position}. "
                    "If this conflicts with previous_summarization, trust the current toolbar fact and current toolbar information."
                )
            else:
                current_toolbar_fact = (
                    f"Current toolbar fact: {selected_item_name} is currently selected now. "
                    "If this conflicts with previous_summarization, trust the current toolbar fact and current toolbar information."
                )
        elif toolbar_information:
            current_toolbar_fact = (
                "Current toolbar fact: trust the current toolbar information over previous_summarization when they conflict."
            )

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
            "self_reflection_reasoning": self_reflection_reasoning,
            "toolbar_information": toolbar_information,
            "image_description": image_description,
            "surroundings": surroundings,
            "location": location,
            "current_position": prompt_fact_fields.get("current_position", current_position),
            "facing_direction": facing_direction,
            "facing_position": facing_position,
            "current_menu": current_menu,
            "buildings": buildings,
            "furniture": furniture,
            "npcs": npcs,
            "exits": exits,
            "previous_action": pre_action,
            "selected_position": selected_position,
            "selected_item_name": selected_item_name,
            "current_toolbar_fact": current_toolbar_fact,
            "action_feedback": working_area.get("action_feedback", ""),
            "nearest_grounded_target_summary": prompt_fact_fields.get(
                "nearest_grounded_target_summary",
                "",
            ),
            "failure_root_cause": prompt_fact_fields.get("failure_root_cause", ""),
            "failure_signature": prompt_fact_fields.get("failure_signature", ""),
            "required_change_type": prompt_fact_fields.get("required_change_type", ""),
            "deadlock_signature": prompt_fact_fields.get("deadlock_signature", ""),
            "deadlock_reflection_cycles": str(prompt_fact_fields.get("deadlock_reflection_cycles", "0")),
            **acquisition_context,
        }

        memory.working_area.update(processed_params)

        return processed_params


class StardewTaskInferencePostprocessProvider(BaseProvider):

    def __init__(self,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, response: Dict):

        processed_response = deepcopy(response)

        if 'subtask' not in response:
            response['subtask'] = ''
            print(f"Subtask not in response. Response: {response}")
            logger.error(f"Subtask not in response. Response: {response}")
        if 'history_summary' not in response:
            response['history_summary'] = ''
            print(f"History summary not in response. Response: {response}")
            logger.error(f"History summary not in response. Response: {response}")
        if 'subtask_reasoning' not in response:
            response['subtask_reasoning'] = ''
            print(f"Subtask reasoning not in response. Response: {response}")
            logger.error(f"Subtask reasoning not in response. Response: {response}")

        history_summary = response['history_summary']

        subtask_description = response['subtask']
        subtask_reasoning = response['subtask_reasoning']

        processed_response.update({
            'summarization': history_summary,
            'history_summary': history_summary,
            'subtask_description': subtask_description,
            'subtask_reasoning': subtask_reasoning
        })

        memory.update_info_history(processed_response)

        return processed_response
