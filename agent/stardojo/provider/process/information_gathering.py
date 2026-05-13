import os
import re
from typing import Dict, Any, List, Optional
from copy import deepcopy

load_image = None


def _get_groundingdino_load_image():
    """GroundingDINO can be very slow to import; keep it off the hot path."""
    global load_image
    if load_image is not None:
        return load_image
    if str(os.environ.get("STARDOJO_IMPORT_GROUNDINGDINO", "")).lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return None
    try:
        from groundingdino.util.inference import load_image as imported_load_image
    except Exception:
        return None
    load_image = imported_load_image
    return load_image

from stardojo.log import Logger
from stardojo.config import Config
from stardojo.memory import LocalMemory
from stardojo.provider import BaseProvider
from stardojo.provider.others.task_guidance import TaskGuidanceProvider
from stardojo import constants
from stardojo.utils.check import is_valid_value
from stardojo.provider import VideoRecordProvider
from stardojo.utils.image_utils import save_annotate_frame
from stardojo.utils.image_utils import segment_toolbar, segment_new_icon, segement_inventory
try:
    from cradle.config.enhanced_config import EnhancedConfig
except Exception:
    EnhancedConfig = None


config = Config()
logger = Logger()


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


def _looks_like_surroundings_text(text: Any) -> bool:
    normalized = _canonicalize_surroundings_string(text)
    if not normalized:
        return False
    first_line = normalized.splitlines()[0].strip()
    return bool(re.match(r"^\[\s*-?\d+\s*,\s*-?\d+\s*\]\s*:", first_line))


def _get_latest_working_image(memory_obj: LocalMemory) -> str:
    working_area = getattr(memory_obj, "working_area", {}) or {}

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


def _coerce_frame_id(frame_id: Any, default: int = -1) -> int:
    if frame_id is None:
        return default
    if isinstance(frame_id, int):
        return frame_id
    if isinstance(frame_id, float):
        return int(frame_id)
    try:
        return int(str(frame_id).strip())
    except (TypeError, ValueError):
        logger.warn(
            f"[InfoGathering] Invalid frame id {frame_id!r}, using {default}"
        )
        return default


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


class InformationGatheringPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any = None,
                 use_screenshot_augmented = False,
                 use_task_guidance = False,
                 task_description = "",
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.gm = gm
        self.memory = LocalMemory()
        self.task_guidance = TaskGuidanceProvider(task_description=task_description)

        self.use_screenshot_augmented = use_screenshot_augmented
        self.use_task_guidance = use_task_guidance


    def __call__(self):

        screenshot_path = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET)[-1]
        screenshot_augmnented_path = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET)[-1]
        working_image_path = _get_latest_working_image(self.memory)

        if screenshot_path in ("", constants.NO_IMAGE, None):
            screenshot_path = working_image_path
        if screenshot_augmnented_path in ("", constants.NO_IMAGE, None):
            screenshot_augmnented_path = working_image_path

        if not self.use_screenshot_augmented:
            image_introduction = [
                {
                    "introduction": "This is a screenshot of the current moment in the game.",
                    "path": screenshot_path,
                    "assistant": ""
                }
            ]
        else:
            image_introduction = [
                {
                    "introduction": "This is a screenshot of the current moment in the game with multiple augmentation to help you understand it better. The screenshot is organized into a grid layout with 15 segments, arranged in 3 rows and 5 columns. Each segment in the grid is uniquely identified by coordinates, which are displayed at the center of each segment in white text. The layout also features color-coded bands for orientation: a blue band on the left side and a yellow band on the right side of the screenshot.",
                    "path": screenshot_augmnented_path,
                    "assistant": ""
                }
            ]

        processed_params: Dict[str, Any] = {
            "image_introduction": image_introduction,
            "image_paths": [image_introduction[0]["path"]] if image_introduction and image_introduction[0].get("path") else [],
        }

        if self.use_task_guidance:
            task_description = self.task_guidance.get_task_guidance(use_last=False)
            processed_params["task_description"] = task_description

        self.memory.working_area.update(processed_params)

        return processed_params


class InformationGatheringPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.memory = LocalMemory()


    def __call__(self, response: Dict):

        processed_response = deepcopy(response)

        if (constants.LAST_TASK_HORIZON in response and is_valid_value(response[constants.LAST_TASK_HORIZON])):
            long_horizon = True
        else:
            long_horizon = False

        if (constants.SCREEN_CLASSIFICATION in response and is_valid_value(response[constants.SCREEN_CLASSIFICATION])):
            screen_classification = response[constants.SCREEN_CLASSIFICATION]
        else:
            screen_classification = "None"

        processed_response.update({
            "long_horizon": long_horizon,
            "screen_classification": screen_classification
        })

        self.memory.update_info_history(processed_response)

        return processed_response


class RDR2InformationGatheringPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any = None,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.gm = gm
        self.memory = LocalMemory()
        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))
        self.task_guidance = TaskGuidanceProvider()

    def __call__(self):

        logger.write("RDR2 Information Gathering Preprocess")

        prompts = [
            "This is a screenshot of the current moment in the game"
        ]

        start_frame_id = _coerce_frame_id(
            self.memory.get_recent_history("start_frame_id", k=1)[0],
            default=0,
        )
        end_frame_id = _coerce_frame_id(
            self.memory.get_recent_history("end_frame_id", k=1)[0],
            default=start_frame_id,
        )
        screenshot_path = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=1)[0]

        # Gather information preparation
        logger.write(f'Gather Information Start Frame ID: {start_frame_id}, End Frame ID: {end_frame_id}')

        text_input = {
            "image_introduction": [
                {
                    "introduction": "This is a screenshot of the current moment in the game",
                    "path": "",
                    "assistant": ""
                }
            ],
            "information_type": ["Item_status",
                                 "Notification",
                                 "Environment_information",
                                 "Action_guidance",
                                 "Task_guidance",
                                 "Dialogue",
                                 "Others"]
        }

        video_clip_path = self.video_recorder.get_video(start_frame_id, end_frame_id)
        task_description = self.task_guidance.get_task_guidance(use_last=False)

        get_text_image_introduction = [
            {
                "introduction": prompts[-1],
                "path": screenshot_path,
                "assistant": ""
            }
        ]

        # Configure the gather_information module
        gather_information_configurations = {
            "frame_extractor": True,
            "icon_replacer": True,
            "llm_description": True,
            "object_detector": True
        }

        # Modify the general input for gather_information here
        image_introduction = [get_text_image_introduction[-1]]
        task_description = task_description

        # Modify the input for get_text module in gather_information here
        text_input["image_introduction"] = get_text_image_introduction

        processed_params = {
            "image_introduction": image_introduction,
            "task_description": task_description,
            "text_input": text_input,
            "video_clip_path": video_clip_path,
            "gather_information_configurations": gather_information_configurations
        }

        self.memory.working_area.update(processed_params)

        return processed_params


class StardewInformationGatheringPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any = None,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.gm = gm
        self.memory = LocalMemory()
        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))

    @staticmethod
    def _scan_latest_screenshot(path: str) -> str:
        if not path or path == constants.NO_IMAGE:
            return ""

        scan_dir = os.path.dirname(path)
        if not scan_dir or not os.path.isdir(scan_dir):
            return ""

        import glob

        try:
            found = sorted(
                [f for f in glob.glob(os.path.join(scan_dir, "screenshot_*.jpeg")) if os.path.exists(f)],
                key=lambda f: os.path.getmtime(f),
                reverse=True,
            )
        except (OSError, FileNotFoundError):
            found = []
        if found:
            logger.warn(
                f"[InfoGathering] Stale screenshot path {path}, using latest: {found[0]}"
            )
            return found[0]

        return ""

    @classmethod
    def _ensure_screenshot_exists(cls, path: str, fallback_path: str = "") -> str:
        for candidate in (path, fallback_path):
            if not candidate or candidate == constants.NO_IMAGE:
                continue
            if os.path.exists(candidate):
                return candidate

            latest = cls._scan_latest_screenshot(candidate)
            if latest:
                return latest

        return ""


    def __call__(self):

        prompts = [
            "This is a screenshot of the current moment in the game with multiple augmentation to help you understand it better. The screenshot is organized into a grid layout with 15 segments, arranged in 3 rows and 5 columns. Each segment in the grid is uniquely identified by coordinates, which are displayed at the center of each segment in white text. The layout also features color-coded bands for orientation: a blue band on the left side and a yellow band on the right side of the screenshot."
        ]

        start_frame_id = _coerce_frame_id(
            self.memory.get_recent_history("start_frame_id", k=1)[0],
            default=0,
        )
        end_frame_id = _coerce_frame_id(
            self.memory.get_recent_history("end_frame_id", k=1)[0],
            default=start_frame_id,
        )
        working_image_path = _get_latest_working_image(self.memory)
        screenshot_path = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=1)[0]
        screenshot_path = self._ensure_screenshot_exists(
            screenshot_path,
            fallback_path=working_image_path,
        )
        augmented_screenshot_path = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, k=1)[0]
        augmented_screenshot_path = self._ensure_screenshot_exists(
            augmented_screenshot_path,
            fallback_path=screenshot_path,
        )
        task_description = self.memory.get_recent_history("task_description", k=1)[0]

        video_clip_path = self.video_recorder.get_video(start_frame_id, end_frame_id)

        # Configure the test
        # if you want to test with a pre-defined screenshot, you can replace the cur_screenshot_path with the path to the screenshot
        pre_defined_sreenshot = None
        pre_defined_sreenshot_augmented = None
        if pre_defined_sreenshot is not None:
            cur_screenshot_path = pre_defined_sreenshot
            cur_screenshot_path_augmented = pre_defined_sreenshot_augmented
        else:
            cur_screenshot_path = screenshot_path
            cur_screenshot_path_augmented = augmented_screenshot_path

        visual_path = cur_screenshot_path_augmented or cur_screenshot_path

        use_toolbar_scan = True
        if EnhancedConfig is not None:
            try:
                enhanced_cfg = EnhancedConfig()
                use_toolbar_scan = bool(
                    enhanced_cfg._raw_config.get("performance", {})
                    .get("vision", {})
                    .get("enable_toolbar_scan", True)
                )
            except Exception as e:
                logger.warn(f"[InfoGathering] Failed to read enable_toolbar_scan config: {e}")

        if use_toolbar_scan and cur_screenshot_path:
            try:
                cur_toolbar_shot_path = segment_toolbar(cur_screenshot_path)
                cur_new_icon_image_shot_path, cur_new_icon_name_image_shot_path = segment_new_icon(cur_screenshot_path)
                cur_inventories_shot_paths = segement_inventory(r"{}".format(cur_toolbar_shot_path))
            except Exception as e:
                logger.warn(f"[InfoGathering] Toolbar scan failed, disabling toolbar scan for this step: {e}")
                use_toolbar_scan = False
                cur_new_icon_image_shot_path, cur_new_icon_name_image_shot_path = None, None
                cur_inventories_shot_paths = []
        else:
            if use_toolbar_scan and not cur_screenshot_path:
                logger.warn("[InfoGathering] No valid base screenshot available, skip toolbar scan for this step.")
                use_toolbar_scan = False
            cur_new_icon_image_shot_path, cur_new_icon_name_image_shot_path = None, None
            cur_inventories_shot_paths = []

        image_introduction = [
            {
                "introduction": prompts[-1],
                "path": visual_path,
                "assistant": ""
            }
        ] if visual_path else []

        # Configure the gather_information module
        gather_information_configurations = {
            "frame_extractor": False,  # extract text from the video clip
            "icon_replacer": False,
            "llm_description": True,  # get the description of the current screenshot
            "object_detector": False,
            "get_item_number": use_toolbar_scan,  # use llm to get item number in the toolbox
            "use_toolbar": use_toolbar_scan,
        }

        processed_params = {
            "image_introduction": image_introduction,
            "image_paths": [visual_path] if visual_path else [],
            "task_description": task_description,
            constants.IMAGES_MEM_BUCKET: cur_screenshot_path,
            constants.AUGMENTED_IMAGES_MEM_BUCKET: cur_screenshot_path_augmented or constants.NO_IMAGE,
            "cur_inventories_shot_paths": cur_inventories_shot_paths,
            "cur_new_icon_image_shot_path": cur_new_icon_image_shot_path,
            "cur_new_icon_name_image_shot_path": cur_new_icon_name_image_shot_path,
            "video_clip_path": video_clip_path,
            "gather_information_configurations": gather_information_configurations
        }

        self.memory.working_area.update(processed_params)

        return processed_params


class RDR2InformationGatheringPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.memory = LocalMemory()


    def __call__(self, response: Dict):

        logger.write("RDR2 Information Gathering Postprocess")

        processed_response = deepcopy(response)
        task_description = self.memory.working_area["task_description"]

        gathered_information_JSON = response['gathered_information_JSON']
        screenshot_path = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=1)[0]

        all_generated_actions = gathered_information_JSON.search_type_across_all_indices(constants.ACTION_GUIDANCE)

        if constants.LAST_TASK_GUIDANCE in response:
            last_task_guidance = response[constants.LAST_TASK_GUIDANCE]
            if constants.LAST_TASK_HORIZON in response:
                long_horizon = bool(
                    int(response[constants.LAST_TASK_HORIZON][0]))  # Only first character is relevant
            else:
                long_horizon = False
        else:
            logger.warn(f"No {constants.LAST_TASK_GUIDANCE} in response.")
            last_task_guidance = ""
            long_horizon = False

        if constants.IMAGE_DESCRIPTION in response:
            if constants.SCREEN_CLASSIFICATION in response:
                screen_classification = response[constants.SCREEN_CLASSIFICATION]
            else:
                screen_classification = "None"
        else:
            logger.warn(f"No {constants.IMAGE_DESCRIPTION} in response.")
            screen_classification = "None"

        if constants.TARGET_OBJECT_NAME in response:
            target_object_name = response[constants.TARGET_OBJECT_NAME]
        else:
            logger.write("> No target object")
            target_object_name = ""

        groundingdino_load_image = _get_groundingdino_load_image()
        if "boxes" in response and groundingdino_load_image is not None:
            image_source, image = groundingdino_load_image(screenshot_path)
            boxes = response["boxes"]
            logits = response["logits"]
            phrases = response["phrases"]
            directory, filename = os.path.split(screenshot_path)
            bb_image_path = os.path.join(directory, "bb_" + filename)
            save_annotate_frame(image_source,
                                boxes,
                                logits,
                                phrases,
                                target_object_name.title(),
                                bb_image_path)

            if boxes is not None and boxes.numel() != 0:
                # Add the screenshot with bounding boxes into working memory

                self.memory.update_info_history({
                        constants.AUGMENTED_IMAGES_MEM_BUCKET: bb_image_path
                    }
                )
            else:
                self.memory.update_info_history({
                        constants.AUGMENTED_IMAGES_MEM_BUCKET: constants.NO_IMAGE
                    }
                )
        else:
            if "boxes" in response and load_image is None:
                logger.warn("groundingdino is unavailable, skip bbox visualization.")

            self.memory.update_info_history({
                    constants.AUGMENTED_IMAGES_MEM_BUCKET: constants.NO_IMAGE
                }
            )

        processed_response.update({
            "long_horizon": long_horizon,
            "last_task_guidance": last_task_guidance,
            "all_generated_actions": all_generated_actions,
            "screen_classification": screen_classification,
            "task_description": task_description,
            "response_keys": list(response.keys()),
            "response": response,
        })

        self.memory.update_info_history(processed_response)

        return processed_response


class StardewInformationGatheringPostprocessProvider(BaseProvider):

    def __init__(self, *args, base_toolbar_objects, **kwargs):

        super().__init__(*args, **kwargs)

        self.base_toolbar_objects = base_toolbar_objects

        self.memory = LocalMemory()


    def prepare_toolbar_information(self,
                                    tool_dict_list: List[Dict[str, Any]],
                                    selected_position: Optional[int]):

        toolbar_information = "The items in the toolbar are arranged from left to right in the following order.\n"
        selected_item = None

        for item in tool_dict_list:

            name = item["name"]
            number = item["number"]
            position = item["position"]

            if name in self.base_toolbar_objects:
                toolbar_object = self.base_toolbar_objects[name]
                true_name = toolbar_object["name"]
                type = toolbar_object["type"]
                description = toolbar_object["description"]

                if type == "Tool":
                    toolbar_information += f"{position}. {true_name}: {type}. {description}\n"
                elif type == "Blank":
                    toolbar_information += f"{position}. {true_name}: {description}\n"
                else:
                    toolbar_information += f"{position}. {true_name}: {type}. {description} Quality: {number}.\n"

                if selected_position is not None and selected_position == position:
                    selected_item = true_name
            else:
                toolbar_object = self.base_toolbar_objects["unknown"]
                true_name = toolbar_object["name"]
                type = toolbar_object["type"]
                description = toolbar_object["description"]
                toolbar_information += f"{position}. {true_name}: {description}\n"

        # selected item
        if selected_item is not None:
            toolbar_information += f"Now the item you selected is: {selected_position}. {selected_item}\n"
        else:
            toolbar_information += f"Now you are not selecting any item.\n"

        return toolbar_information


    def __call__(self, response: Dict):
        response = deepcopy(response) if isinstance(response, dict) else {}

        toolbar_dict_list = response.get('toolbar_dict_list') or []
        selected_position = response.get('selected_position')
        previous_toolbar_information = self.memory.get_recent_history("toolbar_information", k=1)[0]
        previous_selected_position = self.memory.get_recent_history("selected_position", k=1)[0]
        previous_position = self.memory.get_recent_history("position", k=1)[0]
        previous_image_description = self.memory.get_recent_history("image_description", k=1)[0]
        previous_surroundings = _normalize_surroundings_text(
            self.memory.get_recent_history("surroundings", k=1)[0],
            previous_position,
        )

        if toolbar_dict_list:
            response['toolbar_information'] = self.prepare_toolbar_information(
                toolbar_dict_list,
                selected_position)
        else:
            response['toolbar_information'] = previous_toolbar_information
            if selected_position is None:
                response['selected_position'] = previous_selected_position

        description = str(
            response.get('description')
            or response.get('image_description')
            or ""
        ).strip()
        if not description:
            if previous_image_description not in (None, "", []):
                description = str(previous_image_description).strip()
            elif previous_surroundings:
                description = previous_surroundings

        raw_surroundings = response.get('surroundings', "")
        if raw_surroundings in (None, "", []) and _looks_like_surroundings_text(description):
            raw_surroundings = description

        response['description'] = description
        response['image_description'] = description
        response['surroundings'] = _normalize_surroundings_text(
            raw_surroundings or previous_surroundings,
            response.get('position', previous_position),
        ) or previous_surroundings

        processed_response = deepcopy(response)

        toolbar_information = None
        selected_position = response.get('selected_position')
        surroundings = response.get('surroundings', "")

        energy = None
        dialog = None
        date_time = None

        if constants.IMAGE_DESCRIPTION in response:
            if 'toolbar_information' in response:
                toolbar_information = response['toolbar_information']
            if 'energy' in response:
                energy = response['energy']
            if 'dialog' in response:
                dialog = response['dialog']
            if 'date_time' in response:
                date_time = response['date_time']
        else:
            logger.warn(f"No {constants.IMAGE_DESCRIPTION} in response.")

        processed_response.update({
            "response_keys": list(response.keys()),
            "response": response,
            "toolbar_information": toolbar_information,
            "previous_toolbar_information": previous_toolbar_information,
            "selected_position": selected_position,
            "surroundings": surroundings,
            "previous_surroundings": previous_surroundings,
            "energy": energy,
            "dialog": dialog,
            "date_time": date_time,
        })

        self.memory.update_info_history(processed_response)

        return processed_response
