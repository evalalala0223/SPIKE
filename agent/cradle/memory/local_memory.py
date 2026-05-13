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
import threading
from contextlib import contextmanager
from copy import deepcopy

from cradle.config import Config
from cradle import constants
from cradle.log import Logger
from cradle.memory.base import BaseMemory, Image
from cradle.utils.json_utils import load_json, save_json
from cradle.utils.singleton import Singleton

config = Config()
logger = Logger()


def normalize_sakg_experience(experience: Any) -> Dict[str, Any]:

    if not isinstance(experience, dict):
        return {
            "state_description": "",
            "action": "",
            "similarity": 0.0,
            "success_rate": 0.0,
            "raw_experience": experience,
        }

    normalized = dict(experience)

    state_obj = normalized.get("state")
    state_description = normalized.get("state_description")
    if not isinstance(state_description, str) or not state_description.strip():
        if isinstance(state_obj, dict):
            state_description = state_obj.get("description", "")
        else:
            state_description = getattr(state_obj, "description", "")
    state_description = str(state_description or "").strip()

    action_obj = normalized.get("action")
    action_text = action_obj
    if isinstance(action_text, dict):
        action_text = action_text.get("action", "")
    elif not isinstance(action_text, str):
        action_text = getattr(action_text, "action", "")
    action_text = str(action_text or "").strip()

    normalized["state_description"] = state_description
    normalized["action"] = action_text
    if "state_node" not in normalized and state_obj is not None and not isinstance(state_obj, str):
        normalized["state_node"] = state_obj
    if "action_edge" not in normalized and action_obj is not None and not isinstance(action_obj, str):
        normalized["action_edge"] = action_obj

    return normalized


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
        "memory_hits",
        "memory_actions",
        "memory_confidence",
        "memory_retrieval_mode",
    }

    def __init__(
        self,
        memory_path: str = config.work_dir,
        max_recent_steps: int = config.max_recent_steps,
    ) -> None:

        self._state_lock = threading.RLock()
        self._scope_local = threading.local()
        self.max_recent_steps = max_recent_steps
        self.memory_path = memory_path
        
        # Initialize SA-KG (State-Action Knowledge Graph) if enabled
        self.sa_kg = None
        try:
            from cradle.memory.sa_kg import SAKG
            namespace = getattr(config, 'env_short_name', None) or getattr(config, 'env_name', None)
            sakg = SAKG()
            sakg.initialize(namespace=namespace)
            self.sa_kg = sakg
            if self.sa_kg is not None and self.sa_kg.enabled:
                logger.write("SA-KG initialized and enabled in LocalMemory")
        except Exception as e:
            logger.warn(f"SA-KG initialization failed; continuing without SA-KG: {e}")
            self.sa_kg = None

        # Public working space for the agent to store information during loop
        self._working_area_global: Dict[str, Any] = {}

        self.task_duration = 3

        self._recent_history_global = self._build_default_recent_history()
        self._current_task_scope = ""

        self._normalize_recent_history_buckets()
        self._memory_debug_enabled = False
        self._memory_debug_max_len = 240
        self._load_memory_debug_settings()

    def _singleton_reconfigure(
        self,
        memory_path: str = config.work_dir,
        max_recent_steps: int = config.max_recent_steps,
    ) -> None:
        with self._state_lock:
            if memory_path:
                self.memory_path = memory_path
            if isinstance(max_recent_steps, int) and max_recent_steps > 0 and max_recent_steps != self.max_recent_steps:
                self.max_recent_steps = max_recent_steps
                for key, bucket in list(self._recent_history_global.items()):
                    if not isinstance(bucket, list):
                        bucket = [bucket]
                    if len(bucket) > self.max_recent_steps:
                        bucket = bucket[-self.max_recent_steps:]
                    self._recent_history_global[key] = bucket

    def _get_scope_state(self) -> Dict[str, Any]:
        scope_stack = getattr(self._scope_local, "stack", None)
        if scope_stack:
            return scope_stack[-1]
        return {}

    @property
    def working_area(self) -> Dict[str, Any]:
        scoped = self._get_scope_state().get("working_area")
        if isinstance(scoped, dict):
            return scoped
        return self._working_area_global

    @working_area.setter
    def working_area(self, value: Dict[str, Any]) -> None:
        scope_state = self._get_scope_state()
        if scope_state:
            scope_state["working_area"] = value if isinstance(value, dict) else {}
        else:
            self._working_area_global = value if isinstance(value, dict) else {}

    @property
    def recent_history(self) -> Dict[str, Any]:
        scoped = self._get_scope_state().get("recent_history")
        if isinstance(scoped, dict):
            return scoped
        return self._recent_history_global

    @recent_history.setter
    def recent_history(self, value: Dict[str, Any]) -> None:
        scope_state = self._get_scope_state()
        if scope_state:
            scope_state["recent_history"] = value if isinstance(value, dict) else {}
        else:
            self._recent_history_global = value if isinstance(value, dict) else {}

    @contextmanager
    def isolated_scope(
        self,
        *,
        working_area: Optional[Dict[str, Any]] = None,
        recent_history: Optional[Dict[str, Any]] = None,
    ):
        scope_state = {
            "working_area": deepcopy(working_area if working_area is not None else self.get_working_area_snapshot()),
            "recent_history": deepcopy(recent_history if recent_history is not None else self.get_recent_history_snapshot()),
        }
        stack = getattr(self._scope_local, "stack", None)
        if stack is None:
            stack = []
            self._scope_local.stack = stack
        stack.append(scope_state)
        try:
            yield self
        finally:
            stack.pop()
            if not stack:
                try:
                    del self._scope_local.stack
                except AttributeError:
                    pass

    def run_with_isolated_scope(self, func, *args, **kwargs):
        with self.isolated_scope():
            return func(*args, **kwargs)

    def get_working_area_snapshot(self) -> Dict[str, Any]:
        with self._state_lock:
            return deepcopy(self.working_area)

    def get_recent_history_snapshot(self) -> Dict[str, Any]:
        with self._state_lock:
            return deepcopy(self.recent_history)

    def update_working_area(self, data: Dict[str, Any]) -> None:
        if not data:
            return
        with self._state_lock:
            self.working_area.update(data)
        tracked = self._tracked_payload(data)
        if tracked:
            self._log_memory_debug("working_area_update", tracked)

    def get_working_area_value(self, key: str, default: Any = None) -> Any:
        with self._state_lock:
            return self.working_area.get(key, default)

    def get_latest(self, key: str, default: Any = None) -> Any:
        values = self.get_recent_history(key, k=1)
        if not values:
            return default
        latest = values[-1]
        if latest in ("", None):
            return default
        return latest


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
        }


    def reset_runtime_state(
        self,
        task_scope: str = "",
        work_dir: str = "",
    ) -> None:
        with self._state_lock:
            previous_scope = self._current_task_scope
            if task_scope:
                self._current_task_scope = task_scope
            self._working_area_global = {}
            self._recent_history_global = self._build_default_recent_history()
            self._normalize_recent_history_buckets()
            if work_dir:
                self.memory_path = work_dir
            self._debug_read_cache = {}
        if previous_scope and task_scope and previous_scope != task_scope:
            logger.write(f"[Memory] Task switched: {previous_scope} -> {task_scope}")
        self._log_memory_debug(
            "reset",
            {
                "task_scope": self._current_task_scope,
                "memory_path": self.memory_path,
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
            logger.warn(f"[MemoryDebug][CradleLocalMemory] Failed to load debug settings: {e}")


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

        # Dedup: skip read logs if same key/value was already logged this step
        if operation == "read":
            key = payload.get("key", "")
            latest = payload.get("latest", "")
            if not hasattr(self, "_debug_read_cache"):
                self._debug_read_cache: Dict[str, Any] = {}
            cache_key = str(key)
            if cache_key in self._debug_read_cache and self._debug_read_cache[cache_key] == latest:
                return
            self._debug_read_cache[cache_key] = latest

        try:
            text = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception:
            text = repr(payload)
        logger.write(f"[MemoryDebug][CradleLocalMemory][{operation}] {text}")


    def _tracked_payload(self, data: Dict[str, Any]) -> Dict[str, Any]:

        tracked = {}
        for key, value in data.items():
            if key in self.MEMORY_DEBUG_KEYS:
                tracked[key] = self._summarize_debug_value(value)
        return tracked


    def _normalize_recent_history_buckets(self) -> None:

        """Normalize recent_history so every bucket is list-like.

        This keeps backward compatibility with old snapshots where some buckets
        (e.g. last_task_duration) were stored as scalar values.
        """
        for key, value in list(self.recent_history.items()):
            if isinstance(value, list):
                continue
            self.recent_history[key] = [value]


    def _append_recent_history_value(self, key: str, value: Any) -> None:
        bucket = self.recent_history
        if key not in bucket:
            bucket[key] = []
        elif not isinstance(bucket[key], list):
            bucket[key] = [bucket[key]]

        bucket[key].append(value)

        if len(bucket[key]) > self.max_recent_steps:
            bucket[key].pop(0)


    def add(self, **kwargs) -> None:

        """Add data to memory.

        LocalMemory stores key-value style runtime data, so this method maps
        incoming kwargs directly into working/recent history.
        """
        if not kwargs:
            return

        self.update_info_history(kwargs)


    def similarity_search(
        self,
        data: Union[str, Image],
        top_k: int,
        **kwargs,
    ) -> List[Union[str, Image]]:

        """Retrieve similar recent items from a history bucket.

        Args:
            data: query text/image placeholder.
            top_k: max number of results.
            **kwargs: supports `key` to select a recent_history bucket.
        """
        key = kwargs.get("key", constants.SUMMARIZATION_MEM_BUCKET)
        with self._state_lock:
            bucket = deepcopy(self.recent_history.get(key, []))

        if not isinstance(bucket, list):
            bucket = [bucket]

        if top_k is None or top_k <= 0:
            top_k = 1

        if not bucket:
            return []

        query = str(data).strip().lower() if data is not None else ""
        if not query:
            return bucket[-top_k:]

        scored_items = []
        query_tokens = set(query.split())

        for item in bucket:
            item_text = str(item)
            lower_text = item_text.lower()
            score = 0

            if query in lower_text:
                score += 2

            if query_tokens:
                score += len(query_tokens.intersection(set(lower_text.split())))

            scored_items.append((score, item))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored_items[:top_k]]


    def add_recent_history_kv(
        self,
        key: str,
        info: Any,
    ) -> None:

        """Add recent info (skill/image/reasoning) to memory."""
        with self._state_lock:
            self._append_recent_history_value(key, info)


    def add_recent_history(
        self,
        information
    ) -> None:

        """Add recent info to memory."""
        with self._state_lock:
            for key, value in information.items():
                self._append_recent_history_value(key, value)


    def get_recent_history(
        self,
        key: str,
        k: int = 1,
    ) -> List[Any]:

        """Query recent info (skill/image/reasoning) from memory."""
        with self._state_lock:
            if key not in self.recent_history:
                result = [""]
                return result

            bucket = self.recent_history[key]
            if not isinstance(bucket, list):
                bucket = [bucket]
                self.recent_history[key] = bucket

            if len(bucket) == 0:
                result = [""]
                return result

            if k is None:
                k = 1

            result = list(bucket[-k:] if len(bucket) >= k else bucket)
        if key in self.MEMORY_DEBUG_KEYS:
            latest = result[-1] if result else ""
            self._log_memory_debug("read", {
                "key": key,
                "k": k,
                "latest": self._summarize_debug_value(latest),
            })
        return result


    def update_info_history(self, data: Dict[str, Any]):
        if not data:
            return
        with self._state_lock:
            self.working_area.update(data)
            for key, value in data.items():
                self._append_recent_history_value(key, value)
        tracked = self._tracked_payload(data)
        if tracked:
            self._log_memory_debug("update", tracked)


    def add_summarization(self, summary: str) -> None:
        with self._state_lock:
            self.recent_history[constants.SUMMARIZATION_MEM_BUCKET] = [summary]


    def get_summarization(self) -> str:
        with self._state_lock:
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
        if load_path is not None:
            if os.path.exists(load_path):
                with self._state_lock:
                    self.recent_history = load_json(load_path)
                    self._normalize_recent_history_buckets()
                logger.write(f"{load_path} has been loaded.")
            else:
                logger.error(f"{load_path} does not exist.")


    def save(self, local_path=None) -> None:
        """Save the memory to the local file."""
        # @TODO load and store whole memory
        with self._state_lock:
            recent_history_snapshot = deepcopy(self.recent_history)
        if local_path:
            save_json(file_path=local_path, json_dict=recent_history_snapshot, indent=4)
        else:
            save_json(file_path=os.path.join(self.memory_path, self.storage_filename), json_dict=recent_history_snapshot,
                      indent=4)
        
        # Save SA-KG metadata if enabled
        if self.sa_kg and self.sa_kg.enabled:
            try:
                self.sa_kg.save()
                logger.debug(f"SA-KG metadata saved")
            except Exception as e:
                logger.warn(f"Failed to save SA-KG metadata: {e}")
    
    
    def retrieve_similar_experiences(self, current_state: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve similar experiences from SA-KG for action planning.
        
        Args:
            current_state: Current game state description (image description + context)
            top_k: Number of similar experiences to retrieve
            
        Returns:
            List of similar experience dictionaries with state, action, success_rate
        """
        if not self.sa_kg or not self.sa_kg.enabled:
            return []
        
        try:
            similar_states = self.sa_kg.retrieve_similar_states(
                state_description=current_state,
                top_k=top_k
            )
            return [normalize_sakg_experience(experience) for experience in similar_states]
        except Exception as e:
            logger.warn(f"Failed to retrieve similar experiences from SA-KG: {e}")
            return []
    
    
    def add_experience_to_sakg(self, state_description: str, action: str, 
                                reward: float, success: bool) -> None:
        """Add a validated experience to SA-KG after self-reflection.
        
        Args:
            state_description: Game state description
            action: Action taken
            reward: Reward received
            success: Whether the action was successful
        """
        if not self.sa_kg or not self.sa_kg.enabled:
            return
        
        # Only add if we have self-reflection confirmation (validation already in SAKG)
        self_reflection = self.get_recent_history("self_reflection_reasoning", k=1)
        if not self_reflection or not self_reflection[0]:
            logger.debug("Skipping SA-KG experience addition: no self-reflection available")
            return
        
        try:
            self.sa_kg.add_experience(
                state_description=state_description,
                action=action,
                screenshot_path="",  # Screenshot path not available in this context
                action_params={},  # Action params not available in this context
                success=success,
                metadata={
                    "self_reflection": self_reflection[0],
                    "reward": reward
                }
            )
            logger.debug(f"Experience added to SA-KG: action={action}, success={success}")
        except Exception as e:
            logger.warn(f"Failed to add experience to SA-KG: {e}")
