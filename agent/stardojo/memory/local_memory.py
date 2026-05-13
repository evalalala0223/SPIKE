from collections import deque
from typing import (
    Any,
    List,
    Dict,
    Union,
    Tuple,
    Optional,
)
import json
import os

from stardojo.config import Config
from stardojo import constants
from stardojo.log import Logger
from stardojo.memory.base import BaseMemory, Image
from stardojo.utils.json_utils import load_json, save_json
from stardojo.utils.singleton import Singleton

config = Config()
logger = Logger()


class LocalMemory(BaseMemory, metaclass=Singleton):

    storage_filename = "memory.json"
    MEMORY_DEBUG_KEYS = {
        "history_summary",
        "summarization",
        "subtask_description",
        "subtask_reasoning",
        "toolbar_information",
        "selected_position",
        "chosen_item",
        "action",
        "pre_action",
        "exec_info",
        "gathered_info",
        "screenshot_path",
        "screenshot_augmented_path",
    }

    def __init__(
        self,
        memory_path: str = config.work_dir,
        max_recent_steps: int = config.max_recent_steps,
    ) -> None:

        self.max_recent_steps = max_recent_steps
        self.memory_path = memory_path

        # Public working space for the agent to store information during loop
        self.working_area: Dict[str, Any] = {}

        self.task_duration = 3

        self.recent_history = self._build_default_recent_history()
        self._current_task_scope = ""
        self._normalize_recent_history_buckets()

        self._memory_debug_enabled = False
        self._memory_debug_max_len = 240
        self._load_memory_debug_settings()


    def _build_default_recent_history(self) -> Dict[str, Any]:
        # @TODO First memory summary should be based on environment spec
        return {
            constants.IMAGES_MEM_BUCKET: [],
            constants.AUGMENTED_IMAGES_MEM_BUCKET: [],
            "action": [],
            "action_error": [],
            "decision_making_reasoning": [],
            "success_detection_reasoning": [],
            "self_reflection_reasoning": [],
            "image_description": [],
            "task_guidance": [],
            "dialogue": [],
            "task_description": [],
            constants.SKIIL_LIB_MEM_BUCKET: [],
            constants.SUMMARIZATION_MEM_BUCKET: ["The user is using the target application on the PC."],
            constants.LAST_TASK_GUIDANCE: [],
            "long_horizon_task": [],
            constants.LAST_TASK_DURATION: [self.task_duration],
            constants.KEY_REASON_OF_LAST_ACTION: [],
            constants.SUCCESS_DETECTION: [],
            "start_frame_id": [],
            "end_frame_id": [],
            "screenshot_path": [],
            "screenshot_augmented_path": [],
            "subtask_description": [],
            "subtask_reasoning": [],
            "history_summary": [],
            "toolbar_information": [],
            "selected_position": [],
            "selected_item_name": [],
            "chosen_item": [],
            "pre_action": [],
            "exec_info": [],
            "gathered_info": [],
        }


    def reset_runtime_state(
        self,
        task_scope: str = "",
        work_dir: Optional[str] = None,
    ) -> None:
        previous_scope = self._current_task_scope
        if task_scope:
            self._current_task_scope = task_scope
        self.working_area = {}
        self.recent_history = self._build_default_recent_history()
        self.memory_path = work_dir or Config().work_dir
        if previous_scope and task_scope and previous_scope != task_scope:
            logger.write(f"[Memory] Task switched: {previous_scope} -> {task_scope}")
        self._log_memory_debug(
            "reset",
            {
                "task_scope": self._current_task_scope,
                "memory_path": self.memory_path,
                "working_area_keys": 0,
                "recent_history_keys": len(self.recent_history),
            },
        )


    def _load_memory_debug_settings(self) -> None:
        config_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "conf", "enhanced_config.yaml")
        )
        try:
            import yaml

            if not os.path.exists(config_path):
                return

            with open(config_path, "r", encoding="utf-8") as file:
                config_data = yaml.safe_load(file) or {}

            logging_cfg = config_data.get("logging", {}) or {}
            self._memory_debug_enabled = bool(logging_cfg.get("memory_debug", False))
            self._memory_debug_max_len = int(logging_cfg.get("memory_debug_max_len", 240))
        except Exception as e:
            logger.warn(f"[MemoryDebug][StardewLocalMemory] Failed to load debug settings: {e}")


    def _summarize_debug_value(self, value: Any) -> Any:
        if isinstance(value, str):
            if len(value) > self._memory_debug_max_len:
                return value[:self._memory_debug_max_len] + "...<truncated>"
            return value

        if isinstance(value, list):
            summarized = [self._summarize_debug_value(item) for item in value[:3]]
            if len(value) > 3:
                summarized.append(f"...<total={len(value)}>")
            return summarized

        if isinstance(value, dict):
            summary = {}
            for idx, (key, item) in enumerate(value.items()):
                if idx >= 6:
                    summary["..."] = f"<total_keys={len(value)}>"
                    break
                summary[key] = self._summarize_debug_value(item)
            return summary

        return value


    def _log_memory_debug(self, operation: str, payload: Dict[str, Any]) -> None:
        if not self._memory_debug_enabled or not payload:
            return

        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            text = repr(payload)
        logger.write(f"[MemoryDebug][StardewLocalMemory][{operation}] {text}")


    def _tracked_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:
        tracked = {}
        for key, value in data.items():
            if key in self.MEMORY_DEBUG_KEYS:
                tracked[key] = self._summarize_debug_value(value)
        return tracked


    def _normalize_recent_history_buckets(self) -> None:
        normalized_history: Dict[str, Any] = {}
        for key, value in dict(self.recent_history or {}).items():
            if isinstance(value, deque):
                normalized_history[key] = deque(value, maxlen=self.max_recent_steps)
            elif isinstance(value, list):
                normalized_history[key] = deque(value, maxlen=self.max_recent_steps)
            elif value in (None, ""):
                normalized_history[key] = deque(maxlen=self.max_recent_steps)
            else:
                normalized_history[key] = deque([value], maxlen=self.max_recent_steps)
        self.recent_history = normalized_history


    def _ensure_recent_history_bucket(self, key: str) -> deque:
        if key not in self.recent_history:
            self.recent_history[key] = deque(maxlen=self.max_recent_steps)
        elif isinstance(self.recent_history[key], list):
            self.recent_history[key] = deque(self.recent_history[key], maxlen=self.max_recent_steps)
        elif not isinstance(self.recent_history[key], deque):
            existing = self.recent_history[key]
            if existing in (None, ""):
                self.recent_history[key] = deque(maxlen=self.max_recent_steps)
            else:
                self.recent_history[key] = deque([existing], maxlen=self.max_recent_steps)
        return self.recent_history[key]


    def add(self, **kwargs) -> None:
        if kwargs:
            self.update_info_history(kwargs)


    def similarity_search(
        self,
        data: Union[str, Image],
        top_k: int,
        **kwargs: Any,
    ) -> List[Union[str, Image]]:
        if top_k <= 0:
            return []

        if isinstance(data, str) and data in self.recent_history:
            history = list(self.recent_history.get(data, []))
            return history[-top_k:] if len(history) >= top_k else history

        image_history = list(self.recent_history.get(constants.IMAGES_MEM_BUCKET, []))
        return image_history[-top_k:] if len(image_history) >= top_k else image_history


    def add_recent_history_kv(
        self,
        key: str,
        info: Any,
    ) -> None:

        """Add recent info (skill/image/reasoning) to memory."""
        self._ensure_recent_history_bucket(key).append(info)


    def add_recent_history(
        self,
        information
    ) -> None:

        """Add recent info to memory."""
        for key, value in information.items():
            self._ensure_recent_history_bucket(key).append(value)


    def get_recent_history(
        self,
        key: str,
        k: int = 1,
    ) -> List[Any]:

        """Query recent info (skill/image/reasoning) from memory."""

        if key not in self.recent_history or len(self.recent_history[key]) == 0:
            result = [""]
            return result

        if k is None:
            k = 1

        history = self.recent_history[key]
        result = list(history)[-k:] if len(history) >= k else list(history)
        if key in self.MEMORY_DEBUG_KEYS:
            latest = result[-1] if result else ""
            self._log_memory_debug("read", {
                "key": key,
                "k": k,
                "latest": self._summarize_debug_value(latest),
            })
        return result


    def update_info_history(self, data: Dict[str, Any]):
        self.working_area.update(data)
        self.add_recent_history(data)
        tracked = self._tracked_payload(data)
        if tracked:
            self._log_memory_debug("update", tracked)


    def add_summarization(self, summary: str) -> None:
        self.recent_history[constants.SUMMARIZATION_MEM_BUCKET] = [summary]


    def get_summarization(self) -> str:
        return self.recent_history[constants.SUMMARIZATION_MEM_BUCKET][-1]


    def add_task_guidance(self, task_description: str, long_horizon: bool) -> None:
        self.update_info_history({
            constants.LAST_TASK_GUIDANCE: task_description,
            constants.LAST_TASK_DURATION: self.task_duration,
        })
        if long_horizon:
            self.update_info_history({'long_horizon_task': task_description})


    def get_task_guidance(self, use_last = True) -> str:
        last_task_guidance = self.get_recent_history(constants.LAST_TASK_GUIDANCE, k=1)[0]
        if use_last:
            return last_task_guidance
        else:
            last_duration = self.get_recent_history(constants.LAST_TASK_DURATION, k=1)[0]
            try:
                last_duration = int(last_duration)
            except (TypeError, ValueError):
                last_duration = self.task_duration

            current_duration = last_duration - 1
            self.update_info_history({constants.LAST_TASK_DURATION: current_duration})

            if current_duration >= 0:
                return last_task_guidance
            else:
                return self.get_recent_history('long_horizon_task', k=1)[0]


    def load(self, load_path=None) -> None:
        """Load the memory from the local file."""
        # @TODO load and store whole memory
        if load_path != None:
            if os.path.exists(os.path.join(load_path)):
                self.recent_history = load_json(load_path)
                self._normalize_recent_history_buckets()
                logger.write(f"{load_path} has been loaded.")
            else:
                logger.error(f"{load_path} does not exist.")


    def save(self, local_path=None) -> None:
        """Save the memory to the local file."""
        # @TODO load and store whole memory
        recent_history_without_image = dict(self.recent_history)
        if 'ScreenShot' in recent_history_without_image:
            recent_history_without_image['ScreenShot'] = []
        if local_path:
            save_json(file_path=local_path, json_dict=recent_history_without_image, indent=4)
        else:
            save_json(file_path=os.path.join(self.memory_path, self.storage_filename), json_dict=recent_history_without_image,
                      indent=4)
