"""
LangGraph 节点适配器 (Phase 1)

将现有的 Provider 包装成 LangGraph 节点函数。

设计原则：
1. 最小化修改现有 Provider 代码（适配器模式）
2. 节点函数只负责状态转换，业务逻辑在 Provider
3. 添加详细日志和错误处理
4. 保持接口兼容性

作者: AI Development Team
日期: 2026-02-01
版本: 1.0.0
"""
from typing import Dict, Any, Optional, List, cast
import os
import re
import threading
import time
import traceback
import json

from cradle.runner.game_state import GameState, ProviderOutput
from cradle.log import Logger
from cradle import constants
try:
    from cradle.memory import Mem0Provider
except ImportError:
    Mem0Provider = None
from cradle.utils.file_utils import assemble_project_path
from stardojo.utils.task_bootstrap import (
    build_initial_subtask,
    build_initial_subtask_reasoning,
    build_task_acquisition_context,
    get_task_spec,
)
from stardojo.utils.cortex_runtime_utils import is_redundant_tool_selection_subtask
from stardojo.utils.execution_feedback_utils import (
    execution_counts_as_recent_success,
    execution_has_explicit_failure,
    execution_has_no_confirmation,
    infer_execution_success_raw,
)
from stardojo.utils.prompt_profile_utils import infer_stardew_prompt_profile
from stardojo.utils.stardew_prompt_state import extract_stardew_prompt_fact_fields
from stardojo.utils.task_grounding import (
    build_clear_task_profile,
    classify_clearable_target,
    classify_tilling_target,
    clear_target_matches_profile,
)

# Thread-local persistent event loop for parallel node execution.
# Never closed, so httpx AsyncClient (from LangChain) can clean up properly.
_node_loop_storage = threading.local()

logger = Logger()
_RELATIVE_OFFSET_RE = re.compile(
    r"relative offset:\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)",
    re.IGNORECASE,
)


def sanitize_for_checkpoint(data):
    """
    递归清理数据，移除不可序列化的Skill对象

    Args:
        data: 任意数据结构（dict/list/primitive）

    Returns:
        清理后的可序列化数据
    """
    # Check for Skill objects from both cradle and stardojo packages
    _skill_classes = []
    try:
        from cradle.environment.skill import Skill as CradleSkill
        _skill_classes.append(CradleSkill)
    except Exception:
        pass
    try:
        from stardojo.environment.skill import Skill as StardojoSkill
        _skill_classes.append(StardojoSkill)
    except Exception:
        pass

    if _skill_classes and isinstance(data, tuple(_skill_classes)):
        logger.debug(f"sanitize_for_checkpoint: Found Skill object: {getattr(data, 'skill_name', '?')}")
        return {
            "skill_name": getattr(data, 'skill_name', ''),
            "skill_code": getattr(data, 'skill_code', ''),
        }

    # Also catch any duck-typed Skill-like object (has skill_name + skill_function)
    if hasattr(data, 'skill_name') and hasattr(data, 'skill_function'):
        return {
            "skill_name": getattr(data, 'skill_name', ''),
            "skill_code": getattr(data, 'skill_code', ''),
        }

    if isinstance(data, dict):
        return {k: sanitize_for_checkpoint(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        cleaned = [sanitize_for_checkpoint(item) for item in data]
        return cleaned if not isinstance(data, tuple) else tuple(cleaned)
    elif isinstance(data, (str, int, float, bool, type(None))):
        return data
    else:
        # 对于其他不可序列化类型，尝试JSON序列化测试
        try:
            json.dumps(data)
            return data
        except (TypeError, ValueError):
            return str(data)


class LangGraphNodes:
    """
    LangGraph 节点集合
    
    每个方法对应一个 StateGraph 节点，包装现有的 Provider。
    
    节点函数签名：
        def node_function(state: GameState) -> ProviderOutput
    
    返回值：
        只返回需要更新的状态字段，LangGraph 会自动合并。
        例如：return {"gathered_info": {...}, "step_count": state["step_count"] + 1}
    
    错误处理：
        节点内部捕获所有异常，返回 {"error": "..."} 而不是抛出异常。
        这样工作流可以通过条件边处理错误（重试或结束）。
    """
    
    def __init__(self, providers: Dict[str, Any], gm: Optional[Any] = None, augment_provider: Optional[Any] = None,
                 action_planning_preprocess: Optional[Any] = None, action_planning_postprocess: Optional[Any] = None,
                 self_reflection_preprocess: Optional[Any] = None, self_reflection_postprocess: Optional[Any] = None,
                 info_gathering_preprocess: Optional[Any] = None, info_gathering_postprocess: Optional[Any] = None,
                 task_inference_preprocess: Optional[Any] = None, task_inference_postprocess: Optional[Any] = None,
                 runtime_memory: Optional[Any] = None):
        """
        初始化节点适配器
        
        Args:
            providers: Provider 实例字典
                {
                    'video_clip': VideoClipProvider,
                    'self_reflection': SelfReflectionProvider,
                    'task_inference': TaskInferenceProvider,
                    'action_planning': ActionPlanningProvider,
                    'skill_execute': SkillExecuteProvider
                }
        
        Raises:
            KeyError: 如果缺少必需的 Provider
        """
        required_providers = [
            'video_clip',
            'self_reflection',
            'task_inference',
            'action_planning',
            'skill_execute'
        ]
        
        for name in required_providers:
            if name not in providers:
                raise KeyError(f"Missing required provider: {name}")
        
        self.video_clip_provider = providers['video_clip']
        self.self_reflection_provider = providers['self_reflection']
        self.task_inference_provider = providers['task_inference']
        self.action_planning_provider = providers['action_planning']
        self.skill_execute_provider = providers['skill_execute']
        self.gm = gm
        self.augment_provider = augment_provider
        self.action_planning_preprocess = action_planning_preprocess
        self.action_planning_postprocess = action_planning_postprocess
        self.self_reflection_preprocess = self_reflection_preprocess
        self.self_reflection_postprocess = self_reflection_postprocess
        self.info_gathering_preprocess = info_gathering_preprocess
        self.info_gathering_postprocess = info_gathering_postprocess
        self.task_inference_preprocess = task_inference_preprocess
        self.task_inference_postprocess = task_inference_postprocess
        self.runtime_memory = runtime_memory

        # Phase 2.2: Mem0 provider (generic framework)
        self.mem0_enabled = False
        self.mem0_provider = None
        self.mem0_feature_enabled = False
        self.mem0_config_enabled = False
        self.big_brain_throttle_enabled = False
        self.big_brain_throttle_scope = "heavy_nodes"
        self.big_brain_lock_path = ""
        self.big_brain_lock_timeout_s = 180
        self.big_brain_lock_poll_s = 0.5
        self.big_brain_lock_stale_s = 300
        self.mem0_quick_path_max_consecutive_hits = 2
        self.mem0_quick_path_repeat_action_limit = 2
        self.mem0_quick_path_disable_without_embedding = True
        self.mem0_quick_path_execute_threshold = 0.92
        self.mem0_quick_path_max_retry_for_execute = 0
        self.no_progress_repeat_action_limit = 3
        self.mem0_store_require_reflection_success = False
        self.mem0_store_require_execution_success = True
        self.mem0_store_require_meaningful_progress = True
        self.mem0_store_progress_min_chars = 8
        try:
            import yaml
            config_path = assemble_project_path('./conf/enhanced_config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                    features_cfg = cfg.get('features', {})
                    mem0_cfg = cfg.get('mem0', {})
                    dual_brain_cfg = cfg.get('dual_brain', {})
                    big_brain_cfg = dual_brain_cfg.get('big_brain', {})
                    throttle_cfg = big_brain_cfg.get('throttle', {})
                    self.mem0_feature_enabled = bool(features_cfg.get('use_mem0', False))
                    self.mem0_config_enabled = bool(mem0_cfg.get('enabled', False))
                    self.big_brain_throttle_enabled = bool(throttle_cfg.get('enabled', False))
                    self.big_brain_throttle_scope = str(throttle_cfg.get('scope', 'heavy_nodes'))
                    self.big_brain_lock_path = assemble_project_path(
                        throttle_cfg.get('lock_path', './cache/locks/big_brain_planning.lock')
                    )
                    self.big_brain_lock_timeout_s = int(throttle_cfg.get('lock_timeout_s', 180))
                    self.big_brain_lock_poll_s = float(throttle_cfg.get('poll_interval_s', 0.5))
                    self.big_brain_lock_stale_s = int(throttle_cfg.get('stale_after_s', 300))
                    self.mem0_quick_path_max_consecutive_hits = int(mem0_cfg.get('quick_path_max_consecutive_hits', 2))
                    self.mem0_quick_path_repeat_action_limit = int(mem0_cfg.get('quick_path_repeat_action_limit', 2))
                    self.mem0_quick_path_disable_without_embedding = bool(mem0_cfg.get('quick_path_disable_without_embedding', True))
                    self.mem0_quick_path_execute_threshold = float(mem0_cfg.get('quick_path_execute_threshold', 0.92))
                    self.mem0_quick_path_max_retry_for_execute = int(mem0_cfg.get('quick_path_max_retry_for_execute', 0))
                    self.no_progress_repeat_action_limit = int(mem0_cfg.get('no_progress_repeat_action_limit', 3))
                    self.mem0_store_require_reflection_success = bool(
                        mem0_cfg.get('store_require_reflection_success', False)
                    )
                    self.mem0_store_require_execution_success = bool(mem0_cfg.get('store_require_execution_success', True))
                    self.mem0_store_require_meaningful_progress = bool(mem0_cfg.get('store_require_meaningful_progress', True))
                    self.mem0_store_progress_min_chars = int(mem0_cfg.get('store_progress_min_chars', 8))
                    self.mem0_enabled = self.mem0_feature_enabled and self.mem0_config_enabled
                    if self.mem0_enabled:
                        embedding_provider = getattr(self.gm, 'embedding_provider', None) if self.gm is not None else None
                        namespace = getattr(self.gm, 'env_name', None) if self.gm is not None else None
                        self.mem0_provider = Mem0Provider(
                            enabled=True,
                            embedding_provider=embedding_provider,
                            namespace=namespace,
                            storage_path=mem0_cfg.get('storage_path'),
                            quick_path_threshold=float(mem0_cfg.get('quick_path_threshold', 0.85)),
                            max_results=int(mem0_cfg.get('max_results', 3)),
                            require_meaningful_progress=self.mem0_store_require_meaningful_progress,
                            progress_min_chars=self.mem0_store_progress_min_chars,
                        )
        except Exception as e:
            logger.warn(f"[Mem0] Failed to init provider: {e}")

        if self.mem0_enabled and self.mem0_provider is not None:
            logger.write(
                f"[Mem0] Enabled for debugging (feature={self.mem0_feature_enabled}, config={self.mem0_config_enabled}, quick_path_max_hits={self.mem0_quick_path_max_consecutive_hits})"
            )
        else:
            logger.write(
                f"[Mem0] Disabled (features.use_mem0={self.mem0_feature_enabled}, mem0.enabled={self.mem0_config_enabled})"
            )
        
        logger.write("[LangGraphNodes] Initialized with all providers")

    def _get_runtime_memory(self):
        if self.runtime_memory is not None:
            return self.runtime_memory
        from cradle.memory import LocalMemory
        return LocalMemory()  # type: ignore

    def _acquire_heavy_node_slot(self, node_name: str) -> Optional[int]:
        if (
            not self.big_brain_throttle_enabled
            or self.big_brain_throttle_scope != 'heavy_nodes'
            or not self.big_brain_lock_path
        ):
            return None

        lock_dir = os.path.dirname(self.big_brain_lock_path)
        if lock_dir:
            os.makedirs(lock_dir, exist_ok=True)

        start = time.time()
        wait_logged = False
        while True:
            try:
                fd = os.open(self.big_brain_lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                payload = {
                    'pid': os.getpid(),
                    'node': node_name,
                    'time': time.time(),
                }
                os.write(fd, json.dumps(payload, ensure_ascii=False).encode('utf-8'))
                if wait_logged:
                    logger.write(
                        f"[LangGraph] Heavy-node throttle slot acquired for {node_name} after {time.time() - start:.1f}s"
                    )
                return fd
            except FileExistsError:
                try:
                    age = time.time() - os.path.getmtime(self.big_brain_lock_path)
                    if age > self.big_brain_lock_stale_s:
                        logger.warn(
                            f"[LangGraph] Removing stale heavy-node throttle lock (age={age:.1f}s)"
                        )
                        os.remove(self.big_brain_lock_path)
                        continue
                except FileNotFoundError:
                    continue

                waited = time.time() - start
                if waited >= self.big_brain_lock_timeout_s:
                    logger.warn(
                        f"[LangGraph] Heavy-node throttle wait exceeded {waited:.1f}s for {node_name}; proceeding without lock"
                    )
                    return None

                if not wait_logged:
                    logger.write(f"[LangGraph] Waiting for heavy-node throttle slot before {node_name}...")
                    wait_logged = True
                time.sleep(self.big_brain_lock_poll_s)

    def _release_heavy_node_slot(self, lock_fd: Optional[int]) -> None:
        if lock_fd is None or not self.big_brain_lock_path:
            return

        try:
            os.close(lock_fd)
        except Exception:
            pass

        try:
            if os.path.exists(self.big_brain_lock_path):
                os.remove(self.big_brain_lock_path)
        except FileNotFoundError:
            pass

    @staticmethod
    def _normalize_action_sequence(actions: Any) -> tuple:
        if isinstance(actions, (list, tuple)):
            normalized = []
            for action in actions:
                if isinstance(action, str):
                    stripped = action.strip()
                    if stripped:
                        normalized.append(stripped)
                elif action is not None:
                    normalized.append(str(action).strip())
            return tuple(normalized)
        if isinstance(actions, str):
            stripped = actions.strip()
            return (stripped,) if stripped else tuple()
        return tuple()

    def _count_same_action_tail(self, history: Any, current_actions: Any) -> int:
        target = self._normalize_action_sequence(current_actions)
        if not target or not isinstance(history, list):
            return 0

        streak = 0
        for item in reversed(history):
            if self._normalize_action_sequence(item) == target:
                streak += 1
            else:
                break
        return streak

    @staticmethod
    def _count_recent_successes(previous_results: Any) -> int:
        if not isinstance(previous_results, list):
            return 0

        streak = 0
        for item in reversed(previous_results):
            if execution_counts_as_recent_success(item):
                streak += 1
                continue
            break
        return streak

    @staticmethod
    def _count_recent_objective_successes(
        main_task: str,
        previous_results: Any,
    ) -> int:
        prompt_profile = infer_stardew_prompt_profile(main_task)
        if prompt_profile != "cultivation":
            return LangGraphNodes._count_recent_successes(previous_results)

        if not isinstance(previous_results, list):
            return 0

        streak = 0
        for item in reversed(previous_results):
            if not isinstance(item, dict):
                break
            if item.get("completed") is True:
                streak += 1
                continue
            progress_delta = item.get("progress_delta", None)
            if progress_delta not in (None, "", 0, 0.0):
                streak += 1
                continue
            break
        return streak

    @staticmethod
    def _has_recent_instability(state: GameState) -> bool:
        if int(state.get("zero_progress_streak", 0) or 0) >= 1:
            return True
        if int(state.get("repeated_action_streak", 0) or 0) >= 2:
            return True
        if int(state.get("consecutive_failures", 0) or 0) > 0:
            return True
        if bool(state.get("position_issue_detected", False)):
            return True
        if state.get("has_execution_feedback", False) and state.get("last_state_changed") is False:
            return True
        return execution_has_explicit_failure(state.get("last_exec_info", {}))

    @staticmethod
    def _extract_tool_name_from_text(text: Any) -> str:
        if not text:
            return ""

        normalized = str(text)
        tool_patterns = [
            r"selected tool\s*:\s*([a-zA-Z ]+)",
            r"([a-zA-Z ]+)\s+is currently selected",
            r"containing an .*?\b([A-Za-z ]+)\s*\(selected\)",
        ]
        for pattern in tool_patterns:
            match = re.search(pattern, normalized, re.IGNORECASE)
            if match:
                return match.group(1).strip().lower()

        known_tools = [
            "scythe", "axe", "hoe", "watering can", "pickaxe", "rusty sword", "sword"
        ]
        lowered = normalized.lower()
        for tool in known_tools:
            if tool in lowered and "selected" in lowered:
                return tool
        return ""

    def _extract_active_tool_from_state(self, state: GameState) -> str:
        gathered_info = state.get('gathered_info', {})
        if isinstance(gathered_info, dict):
            direct_candidates = [
                gathered_info.get('selected_item_name'),
                gathered_info.get('selected_tool'),
                gathered_info.get('current_tool'),
            ]
            for candidate in direct_candidates:
                if candidate:
                    return str(candidate).strip().lower()

            chosen_item = gathered_info.get('chosen_item')
            if isinstance(chosen_item, dict):
                for key in ('currentitem', 'current_item', 'item_name', 'name', 'item'):
                    candidate = chosen_item.get(key)
                    if candidate:
                        return str(candidate).strip().lower()

            for key in ('other', 'description', 'toolbar_information'):
                extracted = self._extract_tool_name_from_text(gathered_info.get(key, ''))
                if extracted:
                    return extracted

        for key in ('toolbar_information', 'history_summary', 'summarization'):
            extracted = self._extract_tool_name_from_text(state.get(key, ''))
            if extracted:
                return extracted

        return ""

    @staticmethod
    def _extract_target_tool_from_subtask(subtask: str) -> str:
        lowered = (subtask or '').lower()
        known_tools = [
            "scythe", "axe", "hoe", "watering can", "pickaxe", "rusty sword", "sword"
        ]
        for tool in known_tools:
            if tool in lowered:
                return tool
        return ""

    def _subtask_tool_selection_already_satisfied(self, subtask: str, state: GameState) -> bool:
        lowered = (subtask or '').lower()
        if not lowered:
            return False

        if not any(keyword in lowered for keyword in ('select', 'equip', 'choose')):
            return False
        if 'toolbar' not in lowered and 'tool' not in lowered:
            return False

        target_tool = self._extract_target_tool_from_subtask(subtask)
        active_tool = self._extract_active_tool_from_state(state)
        return bool(target_tool and active_tool and target_tool == active_tool)

    @staticmethod
    def _normalize_free_text(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()

    @staticmethod
    def _extract_relative_offset_from_text(value: Any) -> Optional[tuple[int, int]]:
        match = _RELATIVE_OFFSET_RE.search(str(value or ""))
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    @classmethod
    def _extract_named_relative_targets(cls, raw_value: Any) -> List[Dict[str, Any]]:
        if isinstance(raw_value, list):
            lines = [str(item or "").strip() for item in raw_value]
        else:
            lines = [line.strip() for line in str(raw_value or "").splitlines()]

        targets: List[Dict[str, Any]] = []
        for line in lines:
            if not line:
                continue
            offset = cls._extract_relative_offset_from_text(line)
            if offset is None:
                continue
            name = line.split("(", 1)[0].strip() or line
            targets.append({
                "name": name,
                "offset": offset,
                "raw": line,
            })
        return targets

    @staticmethod
    def _navigation_direction_phrase(dx: int, dy: int) -> str:
        parts: List[str] = []
        if dy < 0:
            parts.append("north")
        elif dy > 0:
            parts.append("south")
        if dx < 0:
            parts.append("west")
        elif dx > 0:
            parts.append("east")
        return " and ".join(parts)

    @staticmethod
    def _navigation_axis_words(text: str) -> set[str]:
        normalized = f" {str(text or '').strip().lower()} "
        words: set[str] = set()
        if any(token in normalized for token in (" north ", " up ", " upward ", " move up ")):
            words.add("north")
        if any(token in normalized for token in (" south ", " down ", " downward ", " move down ")):
            words.add("south")
        if any(token in normalized for token in (" west ", " left ", " leftward ", " move left ")):
            words.add("west")
        if any(token in normalized for token in (" east ", " right ", " rightward ", " move right ")):
            words.add("east")
        return words

    def _find_navigation_waypoint(
        self,
        main_task: str,
        prompt_fact_fields: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        task_text = self._normalize_free_text(main_task)
        location_text = self._normalize_free_text(prompt_fact_fields.get("location", ""))
        acquisition_context = build_task_acquisition_context(main_task)
        target_item = self._normalize_free_text(acquisition_context.get("target_item", ""))

        preferences: List[tuple[str, str, ...]] = []

        def _add(source: str, *tokens: str) -> None:
            normalized_tokens = tuple(
                token
                for token in (self._normalize_free_text(value) for value in tokens)
                if token
            )
            if normalized_tokens:
                preferences.append((source, *normalized_tokens))

        if "go to coop" in task_text or "go_to_coop" in task_text or target_item == "coop":
            _add("buildings", "coop")
        elif "go to barn" in task_text or "go_to_barn" in task_text or target_item == "barn":
            _add("buildings", "barn")
        elif "go to bus stop" in task_text or "go_to_bus_stop" in task_text or target_item == "bus stop":
            _add("exits", "bus stop", "bus stop exit")
        elif "go to backwoods" in task_text or "go_to_backwoods" in task_text or target_item == "backwoods":
            _add("exits", "backwoods", "pet bowl entrance", "backwoods exit")
        elif "pierre" in task_text or "general store" in task_text or "pierre" in target_item or "general store" in target_item:
            if "farm" in location_text and "bus stop" not in location_text and "town" not in location_text:
                _add("exits", "bus stop")
            elif "bus stop" in location_text:
                _add("exits", "town")
            else:
                _add("buildings", "pierre", "general store")
        elif "marnie" in task_text or "ranch" in task_text or "marnie" in target_item or "ranch" in target_item:
            if "farm" in location_text and "bus stop" not in location_text and "town" not in location_text:
                _add("exits", "bus stop")
            elif "bus stop" in location_text:
                _add("exits", "town")
            else:
                _add("buildings", "marnie", "ranch")
        elif "fish shop" in task_text or "fish shop" in target_item:
            if "farm" in location_text and "bus stop" not in location_text and "town" not in location_text:
                _add("exits", "bus stop")
            elif "bus stop" in location_text:
                _add("exits", "town")
            else:
                _add("buildings", "fish shop")
        elif target_item:
            _add("buildings", target_item)
            _add("exits", target_item)

        candidates: List[Dict[str, Any]] = []
        for preference_index, preference in enumerate(preferences):
            source = preference[0]
            tokens = preference[1:]
            for target in self._extract_named_relative_targets(prompt_fact_fields.get(source, "")):
                haystack = self._normalize_free_text(f"{target['name']} {target['raw']}")
                if any(token in haystack for token in tokens):
                    candidates.append({
                        **target,
                        "source": source,
                        "preference_index": preference_index,
                    })

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda item: (
                int(item.get("preference_index", 99)),
                abs(item["offset"][0]) + abs(item["offset"][1]),
                abs(item["offset"][1]),
                abs(item["offset"][0]),
            ),
        )

    def _navigation_direction_conflicts_with_waypoint(
        self,
        normalized_subtask: str,
        waypoint: Dict[str, Any],
    ) -> bool:
        direction_words = self._navigation_axis_words(normalized_subtask)
        if not direction_words:
            return False

        dx, dy = waypoint.get("offset", (0, 0))
        if dx < 0 and "east" in direction_words:
            return True
        if dx > 0 and "west" in direction_words:
            return True
        if dy < 0 and "south" in direction_words:
            return True
        if dy > 0 and "north" in direction_words:
            return True
        return False

    @staticmethod
    def _looks_like_precise_subtask_text(value: Any) -> bool:
        text = str(value or "").strip()
        if not text:
            return False

        lowered = text.lower()
        if "\n" in text or "\r" in text:
            return False
        if re.match(r"^\s*(?:\d+[\.\)]|[-*])\s+", text):
            return False
        if any(
            marker in lowered
            for marker in (
                "history_summary",
                "subtask_reasoning",
                "current task:",
                "toolbar:",
                "surroundings:",
                "analyze_the",
                "i need to",
                "input analysis",
                "decision:",
            )
        ):
            return False
        if lowered.startswith("the current subtask is"):
            return True
        if len(text) > 220:
            return False
        return text.count(".") <= 2 and len(text.split()) <= 32

    def _extract_named_item_mentions(self, text: Any) -> List[str]:
        normalized = self._normalize_free_text(text)
        if not normalized:
            return []

        known_items = [
            "potato seeds",
            "cauliflower seeds",
            "parsnip seeds",
            "bean starter",
            "basic retaining soil",
            "quality retaining soil",
            "deluxe retaining soil",
            "retaining soil",
            "speed gro",
            "fertilizer",
            "scythe",
            "axe",
            "hoe",
            "watering can",
            "pickaxe",
            "rusty sword",
            "sword",
        ]
        mentions: List[str] = []
        for item in known_items:
            normalized_item = self._normalize_free_text(item)
            if normalized_item in normalized:
                mentions.append(normalized_item)
        return mentions

    def _normalize_target_labels(self, *labels: Any) -> List[str]:
        normalized_labels: List[str] = []
        for raw_label in labels:
            normalized_label = self._normalize_free_text(raw_label)
            if not normalized_label:
                continue
            candidates = [normalized_label]
            if normalized_label.endswith("es") and len(normalized_label) > 3:
                candidates.append(normalized_label[:-2])
            if normalized_label.endswith("s") and len(normalized_label) > 2:
                candidates.append(normalized_label[:-1])
            for candidate in candidates:
                if candidate and candidate not in normalized_labels:
                    normalized_labels.append(candidate)
        return normalized_labels

    @staticmethod
    def _extract_named_target_offset_from_text(
        raw_text: Any,
        target_labels: List[str],
    ) -> Optional[tuple[int, int]]:
        text = str(raw_text or "").strip()
        if not text or not target_labels:
            return None

        tuple_patterns = (
            r"at\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
            r"at\s*\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]",
        )
        for label in target_labels:
            escaped_label = re.escape(label)
            for tuple_pattern in tuple_patterns:
                match = re.search(
                    rf"{escaped_label}.*?{tuple_pattern}",
                    text,
                    flags=re.IGNORECASE,
                )
                if match:
                    return int(match.group(1)), int(match.group(2))

        return None

    def _find_local_named_target(
        self,
        prompt_fact_fields: Dict[str, Any],
        *,
        target_labels: List[str],
        nearby_distance: int = 2,
    ) -> Optional[Dict[str, Any]]:
        if not target_labels:
            return None

        surroundings_map = self._parse_surroundings_cells(prompt_fact_fields.get("surroundings", ""))
        candidates: List[tuple[int, int, int, int, int, str]] = []
        for (cell_x, cell_y), cell_text in surroundings_map.items():
            if cell_x == 0 and cell_y == 0:
                continue
            lowered = self._normalize_free_text(cell_text)
            if not lowered:
                continue
            if any(label in lowered for label in target_labels):
                distance = abs(cell_x) + abs(cell_y)
                candidates.append(
                    (
                        distance,
                        abs(cell_y),
                        abs(cell_x),
                        cell_x,
                        cell_y,
                        str(cell_text or "").strip(),
                    )
                )

        if candidates:
            distance, _, _, cell_x, cell_y, label = min(candidates)
            return {
                "source": "surroundings",
                "offset": (cell_x, cell_y),
                "distance": distance,
                "nearby": distance <= max(1, int(nearby_distance)),
                "adjacent": distance == 1,
                "label": label,
            }

        front_tile_summary = self._normalize_free_text(prompt_fact_fields.get("front_tile_summary", ""))
        if any(label in front_tile_summary for label in target_labels):
            return {
                "source": "front_tile_summary",
                "offset": None,
                "distance": 1,
                "nearby": True,
                "adjacent": True,
                "label": str(prompt_fact_fields.get("front_tile_summary", "") or "").strip(),
            }

        nearest_grounded_summary_raw = str(
            prompt_fact_fields.get("nearest_grounded_target_summary", "") or ""
        ).strip()
        nearest_grounded_summary = self._normalize_free_text(nearest_grounded_summary_raw)
        if any(label in nearest_grounded_summary for label in target_labels):
            summary_offset = self._extract_named_target_offset_from_text(
                nearest_grounded_summary_raw,
                target_labels,
            )
            if summary_offset is None:
                return None
            distance = abs(summary_offset[0]) + abs(summary_offset[1])
            return {
                "source": "nearest_grounded_target_summary",
                "offset": summary_offset,
                "distance": distance,
                "nearby": distance <= max(1, int(nearby_distance)),
                "adjacent": distance == 1,
                "label": nearest_grounded_summary_raw,
            }

        return None

    def _subtask_claims_completion_without_progress(self, subtask_text: Any, state: GameState) -> bool:
        normalized_subtask = self._normalize_free_text(subtask_text)
        if not normalized_subtask:
            return False

        normalized_main_task = self._normalize_free_text(
            state.get("main_task", "") or state.get("task", "")
        )
        sleep_task = any(
            token in normalized_main_task
            for token in ("go to bed", "go_to_bed", "sleep")
        )
        if not any(
            marker in normalized_subtask
            for marker in (
                "task is completed",
                "task completed",
                "objective is completed",
                "objective completed",
                "already completed",
                "subtask is complete",
                "task has been successfully finished",
                "successfully finished",
            )
        ) and not (
            sleep_task
            and any(
                marker in normalized_subtask
                for marker in ("start of a new day", "new day has started")
            )
        ):
            return False

        progress_snapshot = self._extract_mem0_progress_snapshot(state)
        return progress_snapshot["completed"] is False and not bool(progress_snapshot["hard_progress"])

    def _current_task_has_nearby_target(self, main_task: str, state: GameState) -> bool:
        gathered_info = state.get('gathered_info', {})
        prompt_fact_fields = extract_stardew_prompt_fact_fields(
            state=state,
            gathered_info=gathered_info if isinstance(gathered_info, dict) else {},
        )
        image_description = ""
        if isinstance(gathered_info, dict):
            image_description = str(
                gathered_info.get('image_description', '')
                or gathered_info.get('description', '')
                or ''
            )

        prompt_profile = infer_stardew_prompt_profile(main_task)
        local_fact_text = self._normalize_free_text(
            " ".join(
                str(prompt_fact_fields.get(key, ''))
                for key in (
                    'surroundings',
                    'furniture',
                    'npcs',
                    'toolbar_information',
                    'chosen_item',
                    'selected_item_name',
                )
            )
            + " "
            + image_description
        )
        fact_text = self._normalize_free_text(
            " ".join(
                str(prompt_fact_fields.get(key, ''))
                for key in (
                    'surroundings',
                    'crops',
                    'buildings',
                    'furniture',
                    'npcs',
                    'toolbar_information',
                    'chosen_item',
                    'selected_item_name',
                )
            )
            + " "
            + image_description
        )
        task_text = self._normalize_free_text(main_task)
        if not task_text:
            return False

        if task_text.startswith(("forage ", "forage_")):
            spec = get_task_spec(main_task)
            acquisition_context = build_task_acquisition_context(main_task)
            forage_target = self._find_local_named_target(
                prompt_fact_fields,
                target_labels=self._normalize_target_labels(
                    acquisition_context.get("target_item", ""),
                    spec.get("object", ""),
                ),
            )
            return bool(forage_target and forage_target.get("nearby", False))

        def _normalize_nav_target_label(label: str) -> str:
            spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(label or ""))
            return self._normalize_free_text(spaced)

        clear_profile = build_clear_task_profile(main_task)
        if clear_profile:
            for raw_line in str(prompt_fact_fields.get("surroundings", "") or "").splitlines():
                if ":" not in raw_line:
                    continue
                _, _, candidate = raw_line.partition(":")
                if clear_target_matches_profile(candidate, clear_profile):
                    return True
            if clear_profile.get("family") == "hay" and "grass" in fact_text:
                return True

        if task_text.startswith("fertilize ") or task_text.startswith("sow "):
            acquisition_context = build_task_acquisition_context(main_task)
            target_item = acquisition_context.get("target_item", "")
            if not self._local_task_item_is_available(prompt_fact_fields, target_item):
                return False

            surroundings_map = self._parse_surroundings_cells(prompt_fact_fields.get("surroundings", ""))
            return self._has_visible_cultivation_target(
                surroundings_map,
                task_kind="sow" if task_text.startswith("sow ") else "fertilize",
            )

        if prompt_profile == "navigation" and any(
            token in task_text
            for token in ("go to bed", "go_to_bed", "sleep", "enter door and sleep", "enter_door_and_sleep")
        ):
            return "bed" in local_fact_text

        if prompt_profile == "farm_ops":
            return self._farm_ops_task_has_nearby_target(
                task_text=task_text,
                prompt_fact_fields=prompt_fact_fields,
            )

        if prompt_profile == "navigation":
            acquisition_context = build_task_acquisition_context(main_task)
            nav_target = _normalize_nav_target_label(acquisition_context.get("target_item", ""))
            nav_fact_text = self._normalize_free_text(
                " ".join(
                    str(prompt_fact_fields.get(key, ""))
                    for key in (
                        "surroundings",
                        "buildings",
                        "furniture",
                        "npcs",
                        "exits",
                        "front_tile_summary",
                        "current_blocker_signature",
                        "nearest_grounded_target_summary",
                    )
                )
                + " "
                + image_description
            )
            if nav_target and nav_target in nav_fact_text:
                return True
            if ("go to bus stop" in task_text or "go_to_bus_stop" in task_text) and "bus stop" in nav_fact_text:
                return True
            if ("go to backwoods" in task_text or "go_to_backwoods" in task_text) and (
                "backwoods" in nav_fact_text or "pet bowl entrance" in nav_fact_text
            ):
                return True
            if ("go to coop" in task_text or "go_to_coop" in task_text) and "coop" in nav_fact_text:
                return True
            if ("go to bed" in task_text or "go_to_bed" in task_text or "sleep" in task_text) and "bed" in nav_fact_text:
                return True

        if prompt_profile == "combat":
            combat_terms = (
                "slime",
                "green slime",
                "bug",
                "fly",
                "duggy",
                "grub",
                "bat",
                "crab",
                "enemy",
                "monster",
            )
            return any(term in fact_text for term in combat_terms)

        return False

    @staticmethod
    def _is_farm_ops_route_subtask(normalized_subtask: str) -> bool:
        if not normalized_subtask:
            return False

        route_terms = (
            "route to the coop",
            "route to the coop first",
            "enter the coop",
            "check the barn",
            "route into the barn",
            "route into the coop",
            "route into the coop or barn",
            "animal building",
            "route to the farmhouse",
            "pet bowl area",
            "outside of the coop",
            "outside of the barn",
            "route to the pet bowl",
        )
        return any(term in normalized_subtask for term in route_terms)

    def _farm_ops_nearby_fact_text(self, prompt_fact_fields: Dict[str, Any]) -> str:
        raw_surroundings = str(prompt_fact_fields.get("surroundings", "") or "")
        surroundings_map = self._parse_surroundings_cells(raw_surroundings)
        nearby_cells: List[str] = []
        for (cell_x, cell_y), cell_text in surroundings_map.items():
            if cell_x == 0 and cell_y == 0:
                continue
            if abs(cell_x) + abs(cell_y) <= 2:
                nearby_cells.append(str(cell_text or ""))

        return self._normalize_free_text(
            " ".join(
                nearby_cells
                + [
                    raw_surroundings,
                    str(prompt_fact_fields.get("front_tile_summary", "") or ""),
                    str(prompt_fact_fields.get("nearest_grounded_target_summary", "") or ""),
                    str(prompt_fact_fields.get("furniture", "") or ""),
                    str(prompt_fact_fields.get("npcs", "") or ""),
                ]
            )
        )

    def _farm_ops_task_has_nearby_target(
        self,
        *,
        task_text: str,
        prompt_fact_fields: Dict[str, Any],
    ) -> bool:
        nearby_fact_text = self._farm_ops_nearby_fact_text(prompt_fact_fields)
        if not nearby_fact_text:
            return False
        clear_profile = build_clear_task_profile(task_text)

        if any(token in task_text for token in ("pet", "friendship")) and any(
            token in nearby_fact_text
            for token in ("chicken", "cow", "goat", "duck", "rabbit", "sheep", "pig", "dinosaur", "animal")
        ):
            return True
        if any(token in task_text for token in ("open", "close")) and any(
            token in nearby_fact_text
            for token in ("animal door", "coop door", "barn door", "hatch")
        ):
            return True
        if "pet bowl" in task_text and "pet bowl" in nearby_fact_text:
            return True
        if "feeding bench" in task_text and any(
            token in nearby_fact_text
            for token in ("feeding bench", "hopper", "feeder")
        ):
            return True
        if "egg" in task_text and "egg" in nearby_fact_text:
            return True
        if "milk" in task_text and any(token in nearby_fact_text for token in ("cow", "goat", "milk")):
            return True
        if "incubat" in task_text and "incubator" in nearby_fact_text:
            return True
        if clear_profile.get("family") == "hay" and any(
            token in nearby_fact_text for token in ("grass", "hay", "weed", "weeds", "fiber", "fibre")
        ):
            return True

        return False

    def _farm_ops_named_target_direction_mismatch(
        self,
        *,
        normalized_subtask: str,
        prompt_fact_fields: Dict[str, Any],
        target_labels: List[str],
    ) -> str:
        if not normalized_subtask or not target_labels:
            return ""

        surroundings_map = self._parse_surroundings_cells(prompt_fact_fields.get("surroundings", ""))
        if not surroundings_map:
            return ""

        direction_phrases = {
            (-1, 0): ("immediately to the left", "directly to the left", "immediately left"),
            (1, 0): ("immediately to the right", "directly to the right", "immediately right"),
            (0, -1): ("immediately above", "directly above", "immediately up"),
            (0, 1): ("immediately below", "directly below", "immediately down"),
        }
        normalized_labels = [self._normalize_free_text(label) for label in target_labels if self._normalize_free_text(label)]
        for (cell_x, cell_y), phrases in direction_phrases.items():
            if not any(phrase in normalized_subtask for phrase in phrases):
                continue
            cell_text = self._normalize_free_text(surroundings_map.get((cell_x, cell_y), ""))
            if any(label in cell_text for label in normalized_labels):
                return ""
            return {(-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down"}[(cell_x, cell_y)]

        target_cells: List[tuple[int, int]] = []
        for (cell_x, cell_y), cell_text in surroundings_map.items():
            if cell_x == 0 and cell_y == 0:
                continue
            lowered = self._normalize_free_text(cell_text)
            if lowered and any(label in lowered for label in normalized_labels):
                target_cells.append((cell_x, cell_y))

        if not target_cells:
            return ""

        movement_phrases = {
            (-1, 0): ("move left", "head left", "go left", "move west", "head west", "go west"),
            (1, 0): ("move right", "head right", "go right", "move east", "head east", "go east"),
            (0, -1): ("move up", "head up", "go up", "move north", "head north", "go north"),
            (0, 1): ("move down", "head down", "go down", "move south", "head south", "go south"),
        }
        current_best_distance = min(abs(cell_x) + abs(cell_y) for cell_x, cell_y in target_cells)
        for (step_x, step_y), phrases in movement_phrases.items():
            if not any(phrase in normalized_subtask for phrase in phrases):
                continue
            moved_best_distance = min(
                abs(cell_x - step_x) + abs(cell_y - step_y)
                for cell_x, cell_y in target_cells
            )
            if moved_best_distance > current_best_distance:
                return {(-1, 0): "left", (1, 0): "right", (0, -1): "up", (0, 1): "down"}[(step_x, step_y)]

        return ""

    @staticmethod
    def _parse_surroundings_cells(surroundings_text: Any) -> Dict[tuple[int, int], str]:
        cells: Dict[tuple[int, int], str] = {}
        for raw_line in str(surroundings_text or "").splitlines():
            line = raw_line.strip()
            match = re.match(r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*:\s*(.+)$", line)
            if not match:
                continue
            cells[(int(match.group(1)), int(match.group(2)))] = match.group(3).strip()
        return cells

    @staticmethod
    def _classify_clear_recovery_target(cell_text: Any) -> Optional[Dict[str, str]]:
        return classify_clearable_target(cell_text)

    def _select_nearest_clear_recovery_target(
        self,
        main_task: str,
        tool_name: str,
        object_name: str,
        surroundings_map: Dict[tuple[int, int], str],
    ) -> Dict[str, Any]:
        if not surroundings_map:
            return {}

        clear_profile = build_clear_task_profile(
            main_task,
            tool_name=tool_name,
            object_name=object_name,
        )
        desired_tools = set(clear_profile.get("desired_tools", set()) or set())
        if not desired_tools:
            desired_tools = {"Scythe", "Pickaxe", "Axe"}

        candidates: List[tuple[tuple[int, int, int, int], tuple[int, int], Dict[str, str], str]] = []
        for (cell_x, cell_y), cell_text in surroundings_map.items():
            if cell_x == 0 and cell_y == 0:
                continue
            clearable = self._classify_clear_recovery_target(cell_text)
            if not clearable or clearable["tool"] not in desired_tools:
                continue
            if clear_profile and not clear_target_matches_profile(cell_text, clear_profile):
                continue
            candidates.append(
                (
                    (abs(cell_x) + abs(cell_y), abs(cell_x), cell_x, cell_y),
                    (cell_x, cell_y),
                    clearable,
                    str(cell_text or "").strip(),
                )
            )

        if not candidates:
            return {}

        _, cell, clearable, raw_text = min(candidates, key=lambda item: item[0])
        return {
            "cell": cell,
            "tool": clearable["tool"],
            "label": clearable["label"],
            "raw_text": raw_text,
        }

    def _crafting_material_source_specs(
        self,
        missing_materials: List[str],
    ) -> List[Dict[str, Any]]:
        normalized_missing = {
            self._normalize_free_text(material)
            for material in (missing_materials or [])
            if material
        }
        specs: List[Dict[str, Any]] = []

        if "wood" in normalized_missing and "sap" in normalized_missing:
            specs.append({
                "material": "wood and sap",
                "tokens": ("tree", "trees", "wood", "log", "logs", "stump", "stumps"),
                "tool": "Axe",
                "label": "tree",
                "verb": "chop",
            })
        if "stone" in normalized_missing:
            specs.append({
                "material": "stone",
                "tokens": ("stone", "stones", "rock", "rocks", "boulder", "boulders"),
                "tool": "Pickaxe",
                "label": "stone",
                "verb": "mine",
            })
        if "wood" in normalized_missing:
            specs.append({
                "material": "wood",
                "tokens": ("twig", "twigs", "branch", "branches", "wood", "log", "logs", "stump", "stumps"),
                "tool": "Axe",
                "label": "wood or twigs",
                "verb": "chop",
            })
        if "sap" in normalized_missing:
            specs.append({
                "material": "sap",
                "tokens": ("tree", "trees", "sap", "wood", "log", "logs", "stump", "stumps"),
                "tool": "Axe",
                "label": "tree",
                "verb": "chop",
            })
        if "fiber" in normalized_missing:
            specs.append({
                "material": "fiber",
                "tokens": ("weed", "weeds", "grass", "fiber"),
                "tool": "Scythe",
                "label": "weeds",
                "verb": "clear",
            })
        if "clay" in normalized_missing:
            specs.append({
                "material": "clay",
                "tokens": ("artifact spot", "artifact spots", "worm", "worms"),
                "tool": "Hoe",
                "label": "artifact spot",
                "verb": "dig",
            })

        return specs

    def _select_nearest_crafting_material_target(
        self,
        missing_materials: List[str],
        surroundings_map: Dict[tuple[int, int], str],
    ) -> Dict[str, Any]:
        if not surroundings_map:
            return {}

        source_specs = self._crafting_material_source_specs(missing_materials)
        if not source_specs:
            return {}

        candidates: List[tuple[tuple[int, int, int, int, int], tuple[int, int], Dict[str, Any], str]] = []
        for (cell_x, cell_y), cell_text in surroundings_map.items():
            if cell_x == 0 and cell_y == 0:
                continue
            lowered = self._normalize_free_text(cell_text)
            if not lowered:
                continue
            for source_index, spec in enumerate(source_specs):
                if any(token in lowered for token in spec["tokens"]):
                    candidates.append(
                        (
                            (source_index, abs(cell_x) + abs(cell_y), abs(cell_x), cell_x, cell_y),
                            (cell_x, cell_y),
                            spec,
                            str(cell_text or "").strip(),
                        )
                    )
                    break

        if not candidates:
            return {}

        _, cell, spec, raw_text = min(candidates, key=lambda item: item[0])
        return {
            "cell": cell,
            "tool": spec["tool"],
            "label": spec["label"],
            "verb": spec["verb"],
            "material": spec["material"],
            "raw_text": raw_text,
        }

    def _crafting_missing_material_route_tokens(self, missing_materials: List[str]) -> set[str]:
        tokens = {
            self._normalize_free_text(material)
            for material in (missing_materials or [])
            if material
        }
        for spec in self._crafting_material_source_specs(missing_materials):
            tokens.add(self._normalize_free_text(spec["label"]))
            tokens.add(self._normalize_free_text(spec["tool"]))
            tokens.add(self._normalize_free_text(spec["verb"]))
            tokens.update(self._normalize_free_text(token) for token in spec["tokens"])
        return {token for token in tokens if token}

    @staticmethod
    def _recent_positive_progress_signal(state: GameState) -> bool:
        progress_delta = state.get("task_progress_delta", None)
        return progress_delta not in (None, "", 0, 0.0)

    @staticmethod
    def _recent_no_confirmation_signal(state: GameState) -> bool:
        if execution_has_no_confirmation(state.get("last_exec_info", {})):
            return True
        recent_feedback = state.get("recent_execution_feedback", [])
        if not isinstance(recent_feedback, list):
            return False
        for entry in recent_feedback[-2:]:
            if execution_has_no_confirmation(entry):
                return True
            lowered_errors = str(entry.get("errors_info", "") or "").strip().lower()
            if "no confirmation" in lowered_errors:
                return True
        return False

    def _build_current_fact_recovery_subtask(
        self,
        main_task: str,
        state: GameState,
    ) -> tuple[str, str]:
        prompt_fact_fields = extract_stardew_prompt_fact_fields(
            state=state,
            gathered_info=state.get("gathered_info", {}),
        )
        prompt_profile = infer_stardew_prompt_profile(main_task)
        spec = get_task_spec(main_task)
        evaluator = str(spec.get("evaluator", "") or "").strip().lower()
        tool_name = str(spec.get("tool", "") or "").strip()
        object_name = str(spec.get("object", "") or "").strip().lower()
        selected_item_name = str(prompt_fact_fields.get("selected_item_name", "") or "").strip()
        front_tile_summary = self._normalize_free_text(prompt_fact_fields.get("front_tile_summary", ""))
        porch_blocked = any(token in front_tile_summary for token in ("farmhouse", "house"))
        recent_positive_progress = self._recent_positive_progress_signal(state)
        recent_no_confirmation = self._recent_no_confirmation_signal(state)
        surroundings_map = self._parse_surroundings_cells(prompt_fact_fields.get("surroundings", ""))
        current_menu_type = self._normalize_menu_type(prompt_fact_fields.get("current_menu"))
        target_item = str(
            prompt_fact_fields.get("target_item")
            or spec.get("object", "")
            or build_task_acquisition_context(main_task).get("target_item", "")
            or ""
        ).strip()
        local_context_text = self._normalize_free_text(
            " ".join(
                str(prompt_fact_fields.get(key, "") or "")
                for key in (
                    "surroundings",
                    "buildings",
                    "furniture",
                    "npcs",
                    "exits",
                    "image_description",
                    "description",
                )
            )
            + " "
            + " ".join(
                str((state.get("gathered_info", {}) or {}).get(key, "") or "")
                for key in ("description", "image_description")
            )
        )
        target_item_available = self._local_task_item_is_available(prompt_fact_fields, target_item)

        if prompt_profile in {"cultivation", "farm_ops"} and current_menu_type != "no menu":
            task_focus = str(target_item or spec.get("object", "") or main_task).strip() or "the task"
            if current_menu_type in {"dialoguebox", "objectdialogue", "notificationdialogue"}:
                return (
                    f"The current subtask is dismiss the current dialogue so progress toward {task_focus} can continue.",
                    "Current facts show an active dialogue or notification blocking world control, so the next subtask should clear that menu before resuming local farm progress.",
                )
            return (
                f"The current subtask is close the current menu and return to the world state needed for {task_focus}.",
                "Current facts show an active local menu that blocks world actions, so the next subtask should close it before resuming the farm operation.",
            )

        if main_task.startswith(("forage_", "forage ")):
            forage_target = self._find_local_named_target(
                prompt_fact_fields,
                target_labels=self._normalize_target_labels(target_item, spec.get("object", "")),
            )
            if forage_target and bool(forage_target.get("nearby", False)):
                forage_label = (
                    str(spec.get("object", "") or target_item or "the forage target").strip()
                    or "the forage target"
                )
                if forage_target.get("adjacent", False):
                    return (
                        f"The current subtask is face the adjacent {forage_label} and interact to pick it up now.",
                        "Current structured facts already ground the forage target on a local tile, so the next subtask should stop remote searching and finish the pickup immediately.",
                    )
                if recent_no_confirmation:
                    return (
                        f"The current subtask is make one short alignment step toward the nearby {forage_label}, then interact to pick it up.",
                        "Recent execution feedback was low-signal, but the forage target is still grounded nearby in current structured facts, so the next subtask should stay local and retry the pickup alignment instead of falling back to area search.",
                    )
                return (
                    f"The current subtask is line up with the nearby {forage_label} and interact to pick it up.",
                    "Current structured facts already ground the forage target nearby, so the next subtask should stay local and finish the pickup instead of resuming a remote forage-search route.",
                )
            location_text = self._normalize_free_text(prompt_fact_fields.get("location", ""))
            if (
                str(spec.get("object", "") or "").strip().lower() == "daffodil"
                and "town" in location_text
                and any(token in local_context_text for token in ("saloon", "town square", "pelican town"))
            ):
                return (
                    "The current subtask is move south out of the town square toward the bus stop road and search the grassy edges for a Daffodil.",
                    "No Daffodil is grounded nearby, and the player is still in the Town square. For this specific forage target, the next subtask should leave the stone plaza and search the grassy southern road edges instead of wandering around the square center.",
                )

        if evaluator in {"craft", "cook", "produce"}:
            craft_target = str(spec.get("object", "") or "the target item").strip().lower() or "the target item"
            missing_materials = self._crafting_recipe_missing_materials_for_task(
                main_task,
                prompt_fact_fields,
            )
            if missing_materials == []:
                return (
                    f"The current subtask is craft {craft_target} now that all recipe materials are already present.",
                    "Current toolbar and inventory facts already satisfy the recipe, and direct craft(item) is available, so the next subtask should stop routing for materials and finish the crafting interaction immediately.",
                )
            if missing_materials:
                material_target = self._select_nearest_crafting_material_target(
                    missing_materials,
                    surroundings_map,
                )
                missing_label = ", ".join(
                    str(material or "").strip().lower()
                    for material in missing_materials[:2]
                    if str(material or "").strip()
                ) or "recipe materials"
                if material_target:
                    target_phrase = f"nearest visible {material_target['label']}"
                    tool_phrase = str(material_target["tool"]).strip()
                    verb_phrase = str(material_target["verb"]).strip()
                    material_name = str(material_target["material"]).strip()
                    if porch_blocked:
                        return (
                            f"The current subtask is route off the farmhouse porch toward the {target_phrase}, then {verb_phrase} it with the {tool_phrase} to collect {material_name} for crafting {craft_target}.",
                            "The recipe is still missing materials, and the current surroundings already show a grounded local source, so the next subtask should gather that material instead of reopening the crafting flow.",
                        )
                    if recent_positive_progress:
                        return (
                            f"The current subtask is keep gathering materials by moving toward the {target_phrase} and {verb_phrase} it with the {tool_phrase} to collect {material_name} for crafting {craft_target}.",
                            "Task progress changed recently and the recipe is still missing materials, so the next subtask should continue from the nearest grounded material source instead of resetting to menu actions.",
                        )
                    if recent_no_confirmation:
                        return (
                            f"The current subtask is make a short local reposition toward the {target_phrase} and {verb_phrase} it with the {tool_phrase} to collect {material_name} for crafting {craft_target}.",
                            "Recent execution feedback was low-signal, but the recipe is still missing materials and a local grounded source is visible, so the next subtask should stay local and retry the gather route.",
                        )
                    return (
                        f"The current subtask is move toward the {target_phrase} and {verb_phrase} it with the {tool_phrase} to collect {material_name} for crafting {craft_target}.",
                        "The recipe is still missing materials, and the current surroundings already show a grounded local source, so the next subtask should gather that material instead of reopening the crafting flow.",
                    )
                return (
                    f"The current subtask is collect the missing {missing_label} needed to craft {craft_target}.",
                    "The recipe is still missing materials, so the next subtask should stay in material-retrieval mode instead of attempting to craft immediately.",
                )

        if prompt_profile in {"shopping", "social"}:
            service_target = "shop counter"
            if "pierre" in local_context_text or "seed shop" in local_context_text or "general store" in local_context_text:
                service_target = "Pierre's counter"
            elif "shipping bin" in local_context_text:
                service_target = "shipping bin"
            elif "counter" in local_context_text:
                service_target = "store counter"

            if current_menu_type != "no menu":
                if main_task.startswith(("sell_", "sell ")) and target_item and target_item_available:
                    return (
                        f"The current subtask is keep the current shop menu open, make sure {target_item} is selected from the inventory bar, and sell it through the current menu.",
                        "The current facts already show an active menu and the sale item is locally available, so the next subtask should stay in the local sell flow instead of reopening the counter or moving again.",
                    )
                if main_task.startswith(("purchase_", "purchase ")) and target_item:
                    return (
                        f"The current subtask is stay in the current shop menu and buy {target_item}.",
                        "The current facts already show an active shopping context, so the next subtask should stay in the local purchase flow instead of reopening the counter or wandering.",
                    )

            if (
                "counter" in local_context_text
                or "shop" in local_context_text
                or "shipping bin" in local_context_text
            ):
                if main_task.startswith(("sell_", "sell ")) and target_item and target_item_available:
                    return (
                        f"The current subtask is stay at {service_target}, open the sale menu if needed, select {target_item}, and sell it.",
                        "The current facts already place the player in the correct selling context with the target item locally available, so the next subtask should stay local and finish the sale instead of re-routing.",
                    )
                if main_task.startswith(("purchase_", "purchase ")) and target_item:
                    return (
                        f"The current subtask is stay at {service_target}, open the menu if needed, and buy {target_item}.",
                        "The current facts already place the player in the correct shopping context, so the next subtask should stay local and finish the purchase instead of re-routing.",
                    )

        if not surroundings_map:
            return "", ""

        def _nearest_label(match_tokens: tuple[str, ...]) -> str:
            candidates: List[tuple[int, int, int, int, str]] = []
            for (cell_x, cell_y), cell_text in surroundings_map.items():
                lowered = str(cell_text or "").strip().lower()
                if not lowered or (cell_x == 0 and cell_y == 0):
                    continue
                if any(token in lowered for token in match_tokens):
                    candidates.append((abs(cell_x) + abs(cell_y), abs(cell_x), cell_x, cell_y, lowered))
            if not candidates:
                return ""
            _, _, _, _, label = sorted(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))[0]
            return label

        if evaluator == "clear":
            clear_target = self._select_nearest_clear_recovery_target(
                main_task,
                tool_name,
                object_name,
                surroundings_map,
            )
            target_phrase = "nearby visible debris"
            tool_clause = f"clear it with the {tool_name or selected_item_name or 'required tool'}"
            if clear_target:
                target_phrase = f"visible {clear_target['label']} at {clear_target['cell']}"
                target_tool = str(clear_target["tool"]).strip()
                active_tool = str(selected_item_name or tool_name or "").strip().lower()
                if active_tool == target_tool.lower():
                    tool_clause = f"clear it with the {target_tool}"
                else:
                    tool_clause = f"select {target_tool} if needed, then clear it"
            if porch_blocked:
                return (
                    f"The current subtask is route off the farmhouse porch toward the nearest {target_phrase}, then {tool_clause}.",
                    "The farmhouse still blocks the direct porch line, so the next subtask must route locally toward the nearest grounded debris target instead of restarting a generic clear-up subtask.",
                )
            if clear_target:
                if recent_positive_progress:
                    return (
                        f"The current subtask is keep the local clear-up going by moving toward the nearest {target_phrase} and {tool_clause}.",
                        "Task progress increased recently, so task inference should continue from the nearest grounded debris target instead of resetting to a generic clear bootstrap subtask.",
                    )
                if recent_no_confirmation:
                    return (
                        f"The current subtask is make a short local reposition toward the nearest {target_phrase} and {tool_clause}.",
                        "Recent execution feedback included no-confirmation or low-signal failure, so the next subtask should stay local and re-ground on the nearest concrete debris target.",
                    )
                return (
                    f"The current subtask is move toward the nearest {target_phrase} and {tool_clause}.",
                    "Current surroundings already show a grounded debris target, so the next subtask should stay local and concrete.",
                )

        if evaluator == "till":
            if porch_blocked:
                return (
                    "The current subtask is route off the farmhouse porch to reachable open ground, then line up a nearby clear tile for the Hoe.",
                    "The Hoe is already selected, and the farmhouse blocks the direct porch move, so the next subtask must first reach open ground before tilling.",
                )
            till_candidates: List[tuple[tuple[int, int, int, int], tuple[int, int], Dict[str, Any]]] = []
            for (cell_x, cell_y), raw_text in surroundings_map.items():
                if (cell_x, cell_y) == (0, 0):
                    continue
                candidate = classify_tilling_target(surroundings_map, (cell_x, cell_y))
                if not candidate:
                    continue
                till_candidates.append(
                    (
                        (abs(cell_x) + abs(cell_y), abs(cell_x), cell_x, cell_y),
                        (cell_x, cell_y),
                        candidate,
                    )
                )
            if till_candidates:
                _, (cell_x, cell_y), till_candidate = min(
                    till_candidates,
                    key=lambda item: item[0],
                )
                direction_phrase = self._navigation_direction_phrase(cell_x, cell_y) or "nearby"
                target_label = (
                    "adjacent tillable ground"
                    if till_candidate["kind"] == "explicit_ground"
                    else "adjacent open ground patch"
                )
                if abs(cell_x) + abs(cell_y) == 1:
                    return (
                        f"The current subtask is line up with the {target_label} {direction_phrase} of the player and till it with the Hoe.",
                        "Current structured facts already ground a valid local till target, so the next subtask should stay on that nearby grounded patch instead of drifting back to generic house-adjacent routing.",
                    )
                return (
                    f"The current subtask is move {direction_phrase} toward the nearest grounded open till patch and till it with the Hoe.",
                    "Current structured facts already ground the next till target, so the next subtask should follow that nearby patch directly instead of resetting to a generic cultivation bootstrap.",
                )

        if evaluator == "sow":
            if porch_blocked:
                return (
                    f"The current subtask is route off the farmhouse porch to reachable open ground, then line up a valid nearby tilled soil target for {tool_name or selected_item_name or 'the seeds'}.",
                    "The seeds are already selected, and the farmhouse blocks the direct porch move, so the next subtask must first reach open ground and then align with a valid sowing tile.",
                )

        if evaluator == "fertilize":
            if porch_blocked:
                return (
                    f"The current subtask is route off the farmhouse porch to reachable open ground, then line up a valid nearby tilled soil target for {tool_name or selected_item_name or 'the fertilizer'}.",
                    "The fertilizer is already selected, and the farmhouse blocks the direct porch move, so the next subtask must first reach open ground and then align with a valid fertilizing tile.",
                )

        if prompt_profile == "farm_ops" and evaluator == "fill" and "pet bowl" in self._normalize_free_text(target_item):
            pet_bowl_target = self._find_local_named_target(
                prompt_fact_fields,
                target_labels=["pet bowl", target_item],
                nearby_distance=3,
            )
            if pet_bowl_target:
                bowl_label = str(target_item or "pet bowl").strip() or "pet bowl"
                tool_phrase = str(tool_name or selected_item_name or "Watering Can").strip() or "Watering Can"
                if pet_bowl_target.get("adjacent", False):
                    return (
                        f"The current subtask is face the adjacent {bowl_label} and fill it with the {tool_phrase}.",
                        "Current structured facts already ground the pet bowl on an adjacent tile, so the next subtask should stop routing and finish the fill action locally.",
                    )
                offset = pet_bowl_target.get("offset")
                direction_phrase = ""
                if isinstance(offset, tuple) and len(offset) >= 2:
                    direction_phrase = self._navigation_direction_phrase(int(offset[0]), int(offset[1]))
                direction_phrase = direction_phrase or "toward the visible pet bowl"
                return (
                    f"The current subtask is move {direction_phrase} toward the visible {bowl_label}, then fill it with the {tool_phrase}.",
                    "Current structured facts already show the pet bowl nearby, but not adjacent, so the next subtask should stay local and align with the visible bowl tile instead of treating the pet area as immediate fill range.",
                )

        if prompt_profile == "farm_ops" and evaluator == "harvest" and "egg" in self._normalize_free_text(target_item or object_name or main_task):
            building_targets = self._extract_named_relative_targets(prompt_fact_fields.get("buildings", ""))
            coop_targets: List[Dict[str, Any]] = []
            barn_targets: List[Dict[str, Any]] = []
            for target in building_targets:
                haystack = self._normalize_free_text(f"{target.get('name', '')} {target.get('raw', '')}")
                if "coop" in haystack:
                    coop_targets.append(target)
                elif "barn" in haystack:
                    barn_targets.append(target)

            if coop_targets:
                waypoint = min(
                    coop_targets,
                    key=lambda item: (
                        abs(item["offset"][0]) + abs(item["offset"][1]),
                        abs(item["offset"][1]),
                        abs(item["offset"][0]),
                    ),
                )
                dx, dy = waypoint["offset"]
                direction_phrase = self._navigation_direction_phrase(dx, dy) or "toward the coop"
                target_name = str(waypoint.get("name", "") or "coop").strip() or "coop"
                nearby_target = abs(dx) + abs(dy) <= 2
                if porch_blocked:
                    return (
                        f"The current subtask is step off the farmhouse porch and move {direction_phrase} toward the {target_name} entrance to collect an egg.",
                        "Egg harvest tasks are coop-only, and current building facts already provide a grounded coop entrance offset, so the next subtask should follow that coop waypoint instead of drifting toward other animal housing.",
                    )
                if nearby_target:
                    return (
                        f"The current subtask is line up with the {target_name} entrance and enter it to collect an egg.",
                        "Egg harvest tasks are coop-only, and the current building facts place the coop entrance within a short local offset, so the next subtask should stay local and finish the coop entry.",
                    )
                return (
                    f"The current subtask is move {direction_phrase} toward the {target_name} entrance to collect an egg.",
                    "Egg harvest tasks are coop-only, and current building facts already provide a grounded coop entrance offset, so the next subtask should follow that coop waypoint instead of guessing from nearby barns.",
                )

            if barn_targets:
                return (
                    "The current subtask is keep searching for the coop entrance near the animal buildings without entering the barn.",
                    "Egg harvest tasks are coop-only; the current facts show a barn but not a confirmed coop entrance nearby, so the next subtask should keep the search focused on the coop instead of treating the barn as a proxy target.",
                )

        if prompt_profile == "farm_ops" and evaluator == "harvest" and "milk" in self._normalize_free_text(target_item or object_name or main_task):
            location_text = self._normalize_free_text(
                prompt_fact_fields.get("location", "")
                or state.get("location", "")
                or state.get("gathered_info", {}).get("location", "")
            )
            visible_milk_target = "goat" if "goat" in local_context_text else ("cow" if "cow" in local_context_text else "")
            if "barn" in location_text:
                if visible_milk_target:
                    tool_phrase = str(tool_name or selected_item_name or "Milk Pail").strip() or "Milk Pail"
                    return (
                        f"The current subtask is make one short local reposition inside the barn to line up with the visible {visible_milk_target}, then use the {tool_phrase} on it.",
                        "Current facts already place the player inside the barn and image-level evidence shows a milkable animal, but structured local tiles do not yet ground an adjacent target, so the next subtask should stay local and align before using the Milk Pail.",
                    )
                return (
                    "The current subtask is search a little deeper inside the barn for a visible cow or goat while keeping the Milk Pail ready.",
                    "Current facts already place the player inside the barn, but no milkable animal is grounded on local tiles yet, so the next subtask should stay inside and do a short local search instead of restarting barn-entry routing.",
                )

        if prompt_profile == "combat":
            enemy_terms = (
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
            enemy_candidates: List[tuple[int, int, int, int, str]] = []
            for (cell_x, cell_y), cell_text in surroundings_map.items():
                lowered = self._normalize_free_text(cell_text)
                if not lowered or (cell_x == 0 and cell_y == 0):
                    continue
                if any(term in lowered for term in enemy_terms):
                    enemy_candidates.append(
                        (abs(cell_x) + abs(cell_y), abs(cell_x), cell_x, cell_y, lowered)
                    )

            if enemy_candidates:
                _, _, cell_x, cell_y, enemy_label = sorted(
                    enemy_candidates,
                    key=lambda item: (item[0], item[1], item[2], item[3]),
                )[0]
                attack_target = "nearby enemy"
                for term in enemy_terms:
                    if term in enemy_label:
                        attack_target = term
                        break
                selected_weapon = str(selected_item_name or tool_name or "Rusty Sword").strip()
                adjacent_enemy = abs(cell_x) + abs(cell_y) == 1
                if adjacent_enemy and selected_weapon.lower() == "rusty sword":
                    return (
                        f"The current subtask is attack the adjacent {attack_target} with the Rusty Sword.",
                        "Current surroundings already show an adjacent combat target and the correct weapon is selected, so task inference should stop enemy-search routing and switch to the local attack now.",
                    )
                if selected_weapon.lower() == "rusty sword":
                    return (
                        f"The current subtask is move into immediate attack range of the nearby {attack_target} and strike it with the Rusty Sword.",
                        "Current surroundings already show a nearby combat target, so the next subtask should stay local and finish the attack setup instead of continuing a generic mine search.",
                    )
                return (
                    f"The current subtask is keep the nearby {attack_target} in view, equip the Rusty Sword, and attack it as soon as it is adjacent.",
                    "Current surroundings already show a nearby combat target, so the next subtask should switch to weapon-ready local combat instead of continuing a generic mine search.",
                )

        if prompt_profile == "navigation":
            waypoint = self._find_navigation_waypoint(main_task, prompt_fact_fields)
            if waypoint:
                dx, dy = waypoint["offset"]
                direction_phrase = self._navigation_direction_phrase(dx, dy)
                target_name = str(waypoint.get("name", "") or "target").strip()
                nearby_target = abs(dx) + abs(dy) <= 2
                if waypoint.get("source") == "buildings":
                    if porch_blocked:
                        return (
                            f"The current subtask is step off the farmhouse porch and move {direction_phrase} toward the {target_name} entrance.",
                            "Current building facts already provide a grounded entrance offset, so the next subtask should route off the porch and follow that entrance waypoint instead of resetting to a generic navigation bootstrap.",
                        )
                    if nearby_target:
                        return (
                            f"The current subtask is line up with the {target_name} entrance and enter it.",
                            "Current building facts already place the entrance within a short local offset, so the next subtask should stay local and finish the building-entry alignment instead of reusing a stale route.",
                        )
                    return (
                        f"The current subtask is move {direction_phrase} toward the {target_name} entrance.",
                        "Current building facts already provide a grounded entrance offset, so the next subtask should follow that waypoint instead of inventing a generic farm layout.",
                    )
                if waypoint.get("source") == "exits":
                    exit_target_phrase = f"{target_name} exit"
                    normalized_main_task = self._normalize_free_text(main_task)
                    normalized_target_name = self._normalize_free_text(target_name)
                    if normalized_main_task.startswith(("go to backwoods", "go_to_backwoods")) and "pet bowl entrance" in normalized_target_name:
                        exit_target_phrase = "pet bowl entrance exit path that leads to the Backwoods"
                    elif normalized_main_task.startswith(("go to bus stop", "go_to_bus_stop")) and "bus stop" not in normalized_target_name:
                        exit_target_phrase = f"{target_name} path that leads to the Bus Stop"
                    if porch_blocked:
                        return (
                            f"The current subtask is step off the farmhouse porch and move {direction_phrase} toward the {exit_target_phrase}.",
                            "Current exit facts already provide a grounded waypoint, so the next subtask should route off the porch and follow that exit instead of resetting to a generic navigation bootstrap.",
                        )
                    return (
                        f"The current subtask is move {direction_phrase} toward the {exit_target_phrase}.",
                        "Current exit facts already provide a grounded waypoint, so the next subtask should follow that exit instead of inventing a generic route.",
                    )
            location_text = self._normalize_free_text(prompt_fact_fields.get("location", ""))
            normalized_main_task = self._normalize_free_text(main_task)
            if "farm" in location_text and "bus stop" not in location_text and "backwoods" not in location_text:
                if normalized_main_task.startswith(("go to bus stop", "go_to_bus_stop")):
                    return (
                        "The current subtask is move east across the farm toward the Bus Stop exit.",
                        "The target is still Bus Stop and the player remains on the Farm map, so use the known eastward Farm-to-Bus-Stop route instead of guessing another map edge.",
                    )
                if normalized_main_task.startswith(("go to backwoods", "go_to_backwoods")):
                    return (
                        "The current subtask is move north across the farm toward the pet bowl entrance path that leads to the Backwoods.",
                        "The target is still Backwoods and the player remains on the Farm map, so use the known northward pet-bowl route instead of drifting toward the Bus Stop side.",
                    )

        return "", ""

    def _subtask_conflicts_with_current_facts(self, previous_subtask: str, main_task: str, state: GameState) -> str:
        normalized_subtask = self._normalize_free_text(previous_subtask)
        if not normalized_subtask:
            return ""

        # Screenshot grid labels like "grid (3, 2)" are visual annotation coordinates,
        # not actionable world-space facts. When they leak into subtasks, later planning
        # misreads them as relative map coordinates and starts chasing non-existent tiles.
        if re.search(r"\bgrid\s*\(\s*\d+\s*,\s*\d+\s*\)", str(previous_subtask or ""), re.IGNORECASE):
            return "subtask uses screenshot grid coordinates"

        prompt_fact_fields = extract_stardew_prompt_fact_fields(
            state=state,
            gathered_info=state.get('gathered_info', {}),
        )
        selected_item_name = str(prompt_fact_fields.get("selected_item_name", "") or "").strip()
        if (
            bool(state.get("selected_item_already_correct", False))
            and selected_item_name
            and is_redundant_tool_selection_subtask(previous_subtask, selected_item_name)
        ):
            return "selected item already correct; refresh subtask"
        current_menu = prompt_fact_fields.get("current_menu")
        current_menu_type = self._normalize_menu_type(current_menu)
        no_menu_observed = current_menu_type in {"", "no menu", "none", "null"}
        menu_related_subtask = any(
            token in normalized_subtask
            for token in (
                "menu",
                "inventory",
                "dialogue",
                "dialog",
                "sleep prompt",
                "sleep dialogue",
                "shop",
                "chest",
                "letter",
            )
        )
        if menu_related_subtask:
            if bool(state.get("last_menu_changed", False)):
                return f"menu state changed:{current_menu_type or 'unknown'}"
            if no_menu_observed and any(
                token in normalized_subtask
                for token in ("close", "exit", "dismiss", "cancel", "finish")
            ):
                return "menu subtask stale:observed_no_menu"
            expected_menu_type = self._infer_expected_menu_type(normalized_subtask)
            if expected_menu_type and not no_menu_observed and current_menu_type and current_menu_type != expected_menu_type:
                return f"subtask menu mismatch:{expected_menu_type}!={current_menu_type}"

        feedback_text = self._normalize_free_text(
            " ".join(
                str(state.get(key, '') or '')
                for key in (
                    'latest_execution_summary',
                    'failure_signals',
                    'recent_execution_feedback',
                    'task_progress_summary',
                )
            )
        )
        if any(
            token in feedback_text
            for token in ("blocked", "no progress", "position mismatch", "wrong position", "not adjacent")
        ):
            return "recent feedback requires refreshed inference"

        spec = get_task_spec(main_task)
        evaluator = str(spec.get("evaluator", "") or "").strip().lower()
        tool_name = self._normalize_free_text(spec.get("tool", ""))
        target_item = self._normalize_free_text(build_task_acquisition_context(main_task).get('target_item', ''))
        normalized_main_task = self._normalize_free_text(main_task)
        if (
            "cultivate and harvest" in normalized_main_task
            and target_item
            and "harvest" in normalized_subtask
            and target_item not in normalized_subtask
        ):
            return f"cultivation task requires growth setup before harvest:{target_item}"
        mentioned_items = self._extract_named_item_mentions(previous_subtask)
        allows_required_tool_focus = bool(
            tool_name
            and tool_name in mentioned_items
            and (
                evaluator in {"clear", "till", "water", "fertilize", "sow", "kill", "fill", "open", "close", "harvest", "silo"}
                or normalized_main_task.startswith(("forage ", "forage_"))
            )
        )
        if (
            target_item
            and mentioned_items
            and target_item not in mentioned_items
            and target_item not in normalized_subtask
            and not allows_required_tool_focus
        ):
            return f"subtask item mismatch:{mentioned_items[0]}!={target_item}"

        prompt_profile = infer_stardew_prompt_profile(main_task)
        if prompt_profile == "crafting":
            missing_materials = self._crafting_recipe_missing_materials_for_task(
                main_task,
                prompt_fact_fields,
            )
            normalized_missing_materials = {
                self._normalize_free_text(material)
                for material in (missing_materials or [])
                if material
            }
            mentioned_material_tokens = {
                token
                for token in ("stone", "wood", "fiber", "fibre", "sap", "coal", "ore", "clay")
                if token in normalized_subtask.split()
            }
            missing_route_tokens = self._crafting_missing_material_route_tokens(missing_materials or [])
            explicit_material_recovery = bool(
                missing_materials
                and any(token in normalized_subtask for token in missing_route_tokens)
                and any(
                    token in normalized_subtask
                    for token in (
                        "collect",
                        "gather",
                        "needed to craft",
                        "for crafting",
                        "to craft",
                    )
                )
            )
            if missing_materials and normalized_subtask.startswith("the current subtask is craft "):
                return "crafting task missing materials require retrieval"
            unsupported_material_mentions = mentioned_material_tokens - normalized_missing_materials
            if missing_materials and unsupported_material_mentions:
                return "crafting task mentions unsupported material"
            if missing_materials and any(
                term in normalized_subtask
                for term in (
                    "crafting menu",
                    "open the crafting menu",
                    "open crafting menu",
                    "menu(option=\"craft\"",
                    "menu(option=\"open\"",
                )
            ) and not explicit_material_recovery:
                return "crafting task missing materials require retrieval"
            if missing_materials == [] and any(
                term in normalized_subtask
                for term in (
                    "gather",
                    "obtain",
                    "collect",
                    "route to",
                    "navigate to",
                    "search for",
                    "buy",
                    "purchase",
                    "store",
                    "counter",
                    "mine",
                    "mines",
                    "coal",
                    "wood",
                    "fiber",
                    "sap",
                )
            ):
                return "crafting recipe already satisfied by current materials"
            stale_clear_terms = (
                "clear debris",
                "clear the debris",
                "clear nearby weeds",
                "approach the nearby weeds",
                "patch of weeds",
                "clear the weeds",
                "clear weeds",
                "clear the rocks",
                "clear rocks",
                "clear twigs",
            )
            if any(term in normalized_subtask for term in stale_clear_terms) and not explicit_material_recovery:
                return "crafting task conflicts with unrelated clearup subtask"
        location_navigation_task = evaluator == "location" or normalized_main_task.startswith(
            ("go to ", "go_to_", "sleep", "go_to_bed")
        )
        if prompt_profile == "navigation" and location_navigation_task:
            location_text = self._normalize_free_text(
                prompt_fact_fields.get("location", "")
                or state.get("location", "")
                or state.get("gathered_info", {}).get("location", "")
            )
            waypoint = self._find_navigation_waypoint(main_task, prompt_fact_fields)
            if normalized_main_task.startswith(("go to backwoods", "go_to_backwoods")):
                on_farm_outdoors = (
                    "farm" in location_text
                    and "farmhouse" not in location_text
                    and "house" not in location_text
                    and "home" not in location_text
                    and "backwoods" not in location_text
                    and "bus stop" not in location_text
                )
                waypoint_name = self._normalize_free_text(
                    waypoint.get("name", "") if isinstance(waypoint, dict) else ""
                )
                stale_house_exit_subtask = (
                    any(term in normalized_subtask for term in ("farmhouse", "house"))
                    and any(
                        term in normalized_subtask
                        for term in (
                            "exit",
                            "leave",
                            "door",
                            "entrance",
                            "south",
                            "southern",
                            "interior",
                            "inside",
                            "building",
                        )
                    )
                    and "pet bowl" not in normalized_subtask
                    and "backwoods" not in normalized_subtask
                )
                if (
                    on_farm_outdoors
                    and stale_house_exit_subtask
                    and isinstance(waypoint, dict)
                    and waypoint.get("source") == "exits"
                    and ("pet bowl entrance" in waypoint_name or "backwoods" in waypoint_name)
                ):
                    return "outdoor backwoods navigation conflicts with stale farmhouse-exit subtask"
            stale_clear_terms = (
                "clear debris",
                "clear the debris",
                "clear nearby weeds",
                "clear the weeds",
                "clear weeds",
                "clear the rocks",
                "clear rocks",
                "clear immediate debris",
                "patch of weeds",
                "pickaxe",
                "scythe",
                "axe",
            )
            if any(term in normalized_subtask for term in stale_clear_terms) and not self._current_task_has_nearby_target(main_task, state):
                return "navigation task conflicts with unrelated clearup subtask"
            stale_search_terms = (
                "search for the coop",
                "search for coop",
                "search for the bus stop",
                "search for bus stop",
                "search for the building",
                "search for the entrance",
            )
            if any(term in normalized_subtask for term in stale_search_terms):
                return "navigation task conflicts with vague search subtask"
            unsupported_landmark_terms = (
                "large tree",
                "pink tree",
                "flower patch",
                "mailbox",
                "bush",
            )
            if (
                any(term in normalized_subtask for term in unsupported_landmark_terms)
                and not self._current_task_has_nearby_target(main_task, state)
            ):
                return "navigation task conflicts with unsupported landmark route subtask"
        if prompt_profile == "combat":
            stale_noncombat_terms = (
                "clear debris",
                "clear the debris",
                "clear nearby weeds",
                "approach the nearby weeds",
                "search for seeds",
                "begin cultivation",
                "water the crop",
                "fertiliz",
                "plant the seed",
                "plant seeds",
                "route to pierre",
                "general store",
                "buy ",
                "purchase ",
                "craft ",
                "cook ",
            )
            if any(term in normalized_subtask for term in stale_noncombat_terms):
                return "combat task conflicts with unrelated farming or shopping subtask"
            if self._current_task_has_nearby_target(main_task, state):
                stale_search_terms = (
                    "unexplored",
                    "adjacent unexplored",
                    "search the mines",
                    "within the mines",
                    "search for a visible",
                    "locate a visible",
                    "move to adjacent unexplored",
                    "move through unexplored",
                    "keep searching until",
                )
                if any(term in normalized_subtask for term in stale_search_terms):
                    return "combat nearby target conflicts with stale enemy-search subtask"
        if prompt_profile in {"shopping", "social"}:
            current_menu_type = self._normalize_menu_type(prompt_fact_fields.get("current_menu"))
            if current_menu_type != "no menu":
                stale_counter_terms = (
                    "move to pierre",
                    "walk to pierre",
                    "approach pierre",
                    "talk to pierre",
                    "reach the counter",
                    "move into the counter",
                    "open the shop",
                    "open the store",
                    "open the menu",
                    "reopen",
                    "close the menu",
                )
                if any(term in normalized_subtask for term in stale_counter_terms):
                    return "shopping local menu context conflicts with stale counter/menu subtask"
            if (
                main_task.startswith(("sell_", "sell "))
                and self._local_task_item_is_available(prompt_fact_fields, target_item)
                and current_menu_type != "no menu"
                and any(
                    term in normalized_subtask
                    for term in (
                        "counter",
                        "open the shop menu",
                        "open shop menu",
                        "open the store menu",
                        "talk to pierre",
                        "reach pierre",
                    )
                )
            ):
                return "shopping local sell context conflicts with stale counter subtask"
        if prompt_profile == "farm_clearup":
            stale_cultivation_terms = (
                "tilled farmland",
                "begin cultivation",
                "search for seeds",
                "fertiliz",
                "water the crop",
                "water crops",
                "plant the seed",
                "plant seeds",
                "hoe dirt",
                "hoedirt",
            )
            if any(term in normalized_subtask for term in stale_cultivation_terms):
                return "clearup task conflicts with unrelated cultivation subtask"
        if prompt_profile in {"farm_clearup", "cultivation"} and self._current_task_has_nearby_target(main_task, state):
            stale_route_terms = (
                "inventory",
                "check for",
                "shop",
                "store",
                "pierre",
                "counter",
                "buy",
                "purchase",
                "menu",
            )
            if any(term in normalized_subtask for term in stale_route_terms):
                return "nearby grounded target conflicts with route/acquisition subtask"
        if prompt_profile == "farm_ops" and self._current_task_has_nearby_target(main_task, state):
            stale_route_terms = (
                "inventory",
                "check for",
                "menu",
                "shop",
                "store",
                "counter",
            )
            if any(term in normalized_subtask for term in stale_route_terms) or self._is_farm_ops_route_subtask(normalized_subtask):
                return "farm_ops nearby target conflicts with route/acquisition subtask"
        if prompt_profile == "farm_ops":
            if evaluator == "harvest" and "egg" in normalized_main_task:
                if "barn" in normalized_subtask and "coop" not in normalized_subtask:
                    return "egg task conflicts with barn subtask"
            if evaluator == "harvest" and "milk" in normalized_main_task:
                grounded_milk_target = self._find_local_named_target(
                    prompt_fact_fields,
                    target_labels=["goat", "cow"],
                    nearby_distance=2,
                )
                claims_nearby_milk_target = any(
                    term in normalized_subtask
                    for term in (
                        "nearby goat",
                        "nearest goat",
                        "visible goat",
                        "nearby cow",
                        "nearest cow",
                        "visible cow",
                        "milkable animal",
                    )
                )
                if claims_nearby_milk_target and not bool(grounded_milk_target and grounded_milk_target.get("nearby", False)):
                    return "milk task claims nearby animal without grounded target"

            if "pet bowl" in normalized_main_task:
                directional_mismatch = self._farm_ops_named_target_direction_mismatch(
                    normalized_subtask=normalized_subtask,
                    prompt_fact_fields=prompt_fact_fields,
                    target_labels=["pet bowl"],
                )
                if directional_mismatch:
                    return f"pet bowl directional mismatch:{directional_mismatch}"

                pet_bowl_target = self._find_local_named_target(
                    prompt_fact_fields,
                    target_labels=["pet bowl"],
                    nearby_distance=3,
                )
                if (
                    "pet bowl" in normalized_subtask
                    and any(term in normalized_subtask for term in ("adjacent", "immediately", "right next"))
                    and not bool(pet_bowl_target and pet_bowl_target.get("adjacent", False))
                ):
                    return "pet bowl subtask claims adjacent target without adjacent bowl"

            acquisition_context = build_task_acquisition_context(main_task)
            source_type = self._normalize_free_text(acquisition_context.get("source_type", ""))
            buildings_text = self._normalize_free_text(prompt_fact_fields.get("buildings", ""))
            generic_pet_task = evaluator == "pet" and target_item == "animal"
            if (
                generic_pet_task
                and source_type == "animal housing"
                and "barn" in normalized_subtask
                and "coop" not in normalized_subtask
                and not self._current_task_has_nearby_target(main_task, state)
                and not buildings_text
            ):
                return "generic animal pet task conflicts with ungrounded barn-first subtask"
        if prompt_profile == "navigation" and self._current_task_has_nearby_target(main_task, state):
            stale_route_terms = (
                "walk through the farmhouse door",
                "enter the interior",
                "walk outside",
                "outside the farmhouse",
                "outside",
                "debris",
                "weeds",
                "stones",
                "twigs",
                "patch of ground",
                "farmhouse door",
            )
            if any(term in normalized_subtask for term in stale_route_terms):
                return "navigation nearby target conflicts with stale route/search subtask"
            stale_clear_terms = (
                "clear ",
                "clear the weeds",
                "clear weeds",
                "clear the rocks",
                "clear rocks",
                "clear debris",
                "weeds",
                "stones",
                "twigs",
                "fiber",
                "grass",
                "pickaxe",
                "scythe",
                "axe",
            )
            if any(term in normalized_subtask for term in stale_clear_terms):
                return "navigation nearby target conflicts with unrelated clearup subtask"
            if main_task.startswith(("forage_", "forage ")):
                stale_search_terms = (
                    "bus stop",
                    "mountain lake",
                    "search the area",
                    "search another area",
                    "move towards the bus stop",
                    "move toward the bus stop",
                    "route through outdoor forageable areas",
                    "keep searching until",
                )
                if any(term in normalized_subtask for term in stale_search_terms):
                    return "forage nearby target conflicts with stale remote-search subtask"
            waypoint = self._find_navigation_waypoint(main_task, prompt_fact_fields)
            if waypoint and self._navigation_direction_conflicts_with_waypoint(normalized_subtask, waypoint):
                return "navigation direction conflicts with grounded waypoint"

        return ""

    @staticmethod
    def _normalize_menu_type(menu_value: Any) -> str:
        if isinstance(menu_value, dict):
            menu_type = menu_value.get("type", "")
        else:
            menu_type = str(menu_value or "").strip()

        normalized = re.sub(r"[^a-z0-9]+", " ", str(menu_type or "").lower()).strip()
        if normalized in {"", "none", "null"}:
            return "no menu"
        return normalized

    @staticmethod
    def _infer_expected_menu_type(normalized_subtask: str) -> str:
        if "inventory" in normalized_subtask:
            return "inventory"
        if "chest" in normalized_subtask or "wooden box" in normalized_subtask or "itemgrabmenu" in normalized_subtask:
            return "chest"
        if "shop" in normalized_subtask or "store" in normalized_subtask or "purchase" in normalized_subtask:
            return "shopmenu"
        if any(token in normalized_subtask for token in ("dialogue", "dialog", "sleep prompt", "sleep dialogue")):
            return "dialoguebox"
        if "letter" in normalized_subtask or "mail" in normalized_subtask:
            return "letter"
        return ""

    def _is_embedding_provider_ready(self) -> bool:
        if self.mem0_provider is None:
            return False

        sa_kg = getattr(self.mem0_provider, 'sa_kg', None)
        if sa_kg is None:
            return False

        return getattr(sa_kg, 'embedding_provider', None) is not None

    @staticmethod
    def _local_task_item_is_available(prompt_fact_fields: Dict[str, Any], target_item: str) -> bool:
        normalized_target = re.sub(r"[^a-z0-9]+", " ", str(target_item or "").strip().lower()).strip()
        if not normalized_target:
            return True

        local_item_text = " ".join(
            str(prompt_fact_fields.get(key, "") or "")
            for key in (
                "inventory",
                "toolbar_information",
                "chosen_item",
                "selected_item_name",
            )
        ).lower()
        normalized_local = re.sub(r"[^a-z0-9]+", " ", local_item_text).strip()
        return normalized_target in normalized_local

    @staticmethod
    def _collect_inventory_lines(prompt_fact_fields: Dict[str, Any]) -> List[str]:
        entries: List[str] = []

        inventory = prompt_fact_fields.get("inventory", [])
        if isinstance(inventory, list):
            entries.extend(str(item or "").strip() for item in inventory if str(item or "").strip())
        elif inventory not in ("", None):
            entries.extend(
                line.strip()
                for line in str(inventory or "").splitlines()
                if line.strip()
            )

        toolbar_information = str(prompt_fact_fields.get("toolbar_information", "") or "")
        entries.extend(
            line.strip()
            for line in toolbar_information.splitlines()
            if "slot_index" in line.lower()
        )
        return entries

    def _crafting_recipe_missing_materials_for_task(
        self,
        main_task: str,
        prompt_fact_fields: Dict[str, Any],
    ) -> Optional[List[str]]:
        spec = get_task_spec(main_task)
        evaluator = str(spec.get("evaluator", "") or "").strip().lower()
        if evaluator not in {"craft", "cook", "produce"}:
            return None

        recipe_name = str(spec.get("object", "") or "").strip()
        if not recipe_name:
            return None

        inventory_lines = self._collect_inventory_lines(prompt_fact_fields)
        inventory_counts: Dict[str, int] = {}
        line_pattern = re.compile(
            r"slot_index\s+\d+\s*:\s*(.+?)(?:\s+\(quantity:\s*([^)]+)\))?$",
            re.IGNORECASE,
        )
        for entry in inventory_lines:
            text = str(entry or "").strip()
            if not text:
                continue
            match = line_pattern.search(text)
            if match:
                item_name = match.group(1).strip()
                quantity_raw = str(match.group(2) or "1").strip()
            else:
                item_name = text
                quantity_raw = "1"
            if item_name.lower() == "no item":
                continue
            normalized_item = self._normalize_free_text(item_name)
            if not normalized_item:
                continue
            try:
                quantity = int(float(quantity_raw))
            except (TypeError, ValueError):
                quantity = 1
            inventory_counts[normalized_item] = inventory_counts.get(normalized_item, 0) + max(quantity, 0)

        game_data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", "env", "game_data")
        )
        recipes_path = os.path.join(game_data_dir, "CraftingRecipes.json")
        objects_path = os.path.join(game_data_dir, "Objects.json")

        try:
            with open(recipes_path, "r", encoding="utf-8") as fp:
                recipes_raw = json.load(fp) or {}
            recipes = recipes_raw.get("content", recipes_raw)
            recipe_spec = str(recipes.get(recipe_name, "") or "").strip()
        except Exception:
            return None
        if not recipe_spec:
            return None

        try:
            with open(objects_path, "r", encoding="utf-8") as fp:
                objects_raw = json.load(fp) or {}
            object_content = objects_raw.get("content", objects_raw)
        except Exception:
            object_content = {}

        id_to_name: Dict[str, str] = {}
        if isinstance(object_content, dict):
            for object_id, object_data in object_content.items():
                if not isinstance(object_data, dict):
                    continue
                object_name = str(object_data.get("Name", "") or "").strip()
                normalized_name = self._normalize_free_text(object_name)
                if normalized_name:
                    id_to_name[str(object_id)] = normalized_name

        ingredient_tokens = [token for token in recipe_spec.split("/", 1)[0].split() if token]
        missing: List[str] = []
        for index in range(0, len(ingredient_tokens) - 1, 2):
            ingredient_key = ingredient_tokens[index]
            try:
                needed_quantity = int(float(ingredient_tokens[index + 1]))
            except (TypeError, ValueError):
                continue
            ingredient_name = id_to_name.get(str(ingredient_key), self._normalize_free_text(ingredient_key))
            if not ingredient_name:
                continue
            if inventory_counts.get(ingredient_name, 0) < needed_quantity:
                missing.append(ingredient_name)

        return missing

    @staticmethod
    def _has_visible_cultivation_target(
        surroundings_map: Dict[tuple[int, int], str],
        *,
        task_kind: str,
    ) -> bool:
        if not surroundings_map:
            return False

        allow_existing_crop = task_kind == "fertilize"
        allow_pre_fertilized = task_kind == "sow"

        allowed_tokens = {
            "watered",
            "unwatered",
            "dry",
            "wet",
            "empty",
            "tile",
            "soil",
            "dirt",
        }
        if allow_pre_fertilized:
            allowed_tokens.update(
                {
                    "fertilized",
                    "fertilizer",
                    "basic",
                    "quality",
                    "deluxe",
                    "speed",
                    "gro",
                    "retaining",
                }
            )

        crop_tokens = (
            "seed",
            "seeds",
            "crop",
            "growing",
            "mature",
            "sprout",
            "seedling",
            "parsnip",
            "bean",
            "potato",
            "cauliflower",
            "garlic",
            "kale",
            "tulip",
            "jazz",
            "melon",
            "blueberry",
            "tomato",
            "pepper",
            "wheat",
            "corn",
            "pumpkin",
            "cranberry",
            "yam",
            "amaranth",
            "artichoke",
            "beet",
            "bok choy",
            "eggplant",
            "radish",
            "sunflower",
            "hops",
            "strawberry",
            "coffee",
            "tea",
            "rice",
            "pineapple",
            "taro",
            "ancient fruit",
            "grape",
        )
        fertilized_tokens = {
            "fertilized",
            "fertilizer",
            "basic",
            "quality",
            "deluxe",
            "speed",
            "gro",
            "retaining",
        }

        for (cell_x, cell_y), cell_text in surroundings_map.items():
            if cell_x == 0 and cell_y == 0:
                continue

            lowered = str(cell_text or "").strip().lower()
            if not lowered or "hoedirt" not in lowered.replace(" ", ""):
                continue

            remainder = re.sub(r"\bhoe\s*dirt\b|\bhoedirt\b", " ", lowered)
            remainder = re.sub(r"[^a-z0-9]+", " ", remainder).strip()
            if not remainder:
                return True

            remainder_tokens = [token for token in remainder.split() if token]
            has_crop = any(token in lowered for token in crop_tokens)
            has_explicit_fertilizer = any(token in remainder_tokens for token in fertilized_tokens)

            if has_crop and not allow_existing_crop:
                continue
            if has_explicit_fertilizer and not allow_pre_fertilized:
                continue
            if has_crop and allow_existing_crop:
                return True
            if remainder_tokens and all(token in allowed_tokens for token in remainder_tokens):
                return True

        return False

    def _is_meaningful_progress(self, progress_text: Any) -> bool:
        if progress_text is None:
            return False
        normalized = str(progress_text).strip()
        if len(normalized) < max(1, int(self.mem0_store_progress_min_chars)):
            return False

        lowered = normalized.lower()
        fraction_match = re.search(r"(?<!\d)(\d+)\s*/\s*(\d+)", lowered)
        if fraction_match:
            return int(fraction_match.group(1)) > 0

        delta_match = re.search(r"delta\s*=\s*(-?\d+(?:\.\d+)?)", lowered)
        if delta_match:
            return float(delta_match.group(1)) != 0.0

        change_match = re.search(
            r"(?:increased|changed)\s+from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)",
            lowered,
        )
        if change_match:
            return float(change_match.group(1)) != float(change_match.group(2))

        if "task is completed" in lowered:
            return True

        weak_patterns = {
            "none",
            "n/a",
            "unknown",
            "no progress",
            "no significant progress",
            "nothing changed",
            "same as before",
        }
        if lowered in weak_patterns:
            return False

        negative_markers = (
            "0/",
            "stayed at 0",
            "recorded task progress is 0",
            "task progress stayed at",
            "task is not completed yet",
            "not completed",
            "not yet completed",
            "no observable effect",
            "without progress",
            "no action was executed",
            "no new action was performed",
            "no fertilizer application has been performed yet",
            "no such action was executed",
            "no interaction",
            "did not advance",
            "did not make progress",
            "did not change",
            "still need",
            "still needs",
            "still missing",
            "still unavailable",
            "unsuccessful",
            "failed",
            "pending",
        )
        if any(marker in lowered for marker in negative_markers):
            return False

        positive_markers = (
            "task progress increased",
            "cleared ",
            "removed ",
            "chopped ",
            "cut down",
            "broken ",
            "harvested ",
            "collected ",
            "fertilized ",
            "planted ",
            "watered ",
            "tilled ",
            "mined ",
            "filled ",
            "petted ",
            "deposited ",
        )
        return any(marker in lowered for marker in positive_markers)

    def _derive_mem0_progress_text(self, state: GameState, reflection_result: Any) -> str:
        progress_snapshot = self._extract_mem0_progress_snapshot(state)
        if progress_snapshot["completed"] is True:
            return "The task is completed."

        if progress_snapshot["hard_progress"]:
            previous_quantity = progress_snapshot["previous_progress_quantity"]
            current_quantity = progress_snapshot["progress_quantity"]
            progress_delta = progress_snapshot["progress_delta"]
            if previous_quantity is not None and current_quantity is not None:
                candidate = (
                    f"Task progress changed from {previous_quantity} "
                    f"to {current_quantity} (delta={progress_delta})."
                )
            elif progress_delta not in (None, "", 0, 0.0):
                candidate = f"Task progress delta recorded: {progress_delta}."
            else:
                candidate = "Task progress increased."
            if self._is_meaningful_progress(candidate):
                return candidate

        if isinstance(reflection_result, dict):
            for key in ("progress", "status_summary", "reasoning"):
                candidate = str(reflection_result.get(key, "") or "").strip()
                if self._is_meaningful_progress(candidate):
                    return candidate

        latest_execution_summary = str(state.get("latest_execution_summary", "") or "").strip()
        if self._is_meaningful_progress(latest_execution_summary):
            return latest_execution_summary

        return ""

    @staticmethod
    def _clean_mem0_actions(actions: Any) -> List[str]:
        if isinstance(actions, str):
            raw_actions = [actions]
        elif isinstance(actions, list):
            raw_actions = actions
        else:
            raw_actions = []

        cleaned: List[str] = []
        for action in raw_actions:
            if not isinstance(action, str):
                continue
            stripped = action.strip()
            if not stripped or stripped.lower() == "nop()":
                continue
            cleaned.append(stripped)
        return cleaned

    @staticmethod
    def _is_setup_only_mem0_actions(actions: List[str]) -> bool:
        if not actions:
            return False
        return all(
            action.lower().startswith(("choose_item(", "attach_item(", "unattach_item("))
            for action in actions
        )

    @staticmethod
    def _is_move_only_mem0_actions(actions: List[str]) -> bool:
        if not actions:
            return False
        return all(action.lower().startswith("move(") for action in actions)

    @staticmethod
    def _coerce_mem0_numeric(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

    def _extract_mem0_progress_snapshot(self, state: GameState) -> Dict[str, Any]:
        latest_task_eval = state.get("latest_task_eval", {})
        if not isinstance(latest_task_eval, dict):
            latest_task_eval = {}

        completed_value = latest_task_eval.get("completed", state.get("completed", None))
        completed = completed_value if isinstance(completed_value, bool) else None

        progress_delta = self._coerce_mem0_numeric(state.get("task_progress_delta", None))
        progress_quantity = self._coerce_mem0_numeric(
            state.get("task_progress_quantity", None)
        )
        previous_progress_quantity = self._coerce_mem0_numeric(
            state.get("previous_task_progress_quantity", None)
        )
        if (
            progress_delta is None
            and progress_quantity is not None
            and previous_progress_quantity is not None
        ):
            progress_delta = progress_quantity - previous_progress_quantity

        hard_progress = bool(
            completed is True
            or (
                progress_delta is not None
                and progress_delta != 0.0
            )
            or (
                progress_quantity is not None
                and previous_progress_quantity is not None
                and progress_quantity != previous_progress_quantity
            )
        )

        return {
            "completed": completed,
            "progress_delta": progress_delta,
            "progress_quantity": progress_quantity,
            "previous_progress_quantity": previous_progress_quantity,
            "hard_progress": hard_progress,
        }

    def _record_mem0_store_skip(self, reason: str, message: str) -> None:
        logger.write(f"[Mem0] Skip store: {message}")
        if self.mem0_provider is None:
            return
        try:
            self.mem0_provider.record_store_skip(reason)
        except Exception as e:
            logger.warn(f"[Mem0] Failed to record store skip reason: {e}")

    def _resolve_mem0_store_actions(self, state: GameState) -> List[str]:
        actions = self._clean_mem0_actions(state.get('planned_actions', []))
        if actions:
            return actions

        previous_actions = state.get('previous_actions', [])
        if isinstance(previous_actions, list) and previous_actions:
            last_actions = previous_actions[-1]
            if isinstance(last_actions, list):
                return self._clean_mem0_actions(last_actions)
            if isinstance(last_actions, str) and last_actions.strip():
                return self._clean_mem0_actions([last_actions.strip()])

        return []

    def _derive_mem0_reward(
        self,
        success: bool,
        progress: str,
        *,
        hard_progress: bool,
        completed: Optional[bool] = None,
    ) -> float:
        if completed is True:
            return 1.0
        if hard_progress:
            return 1.0 if success else 0.25
        if not success:
            return -0.5
        if self._is_meaningful_progress(progress):
            return 0.25
        return 0.0

    def _commit_mem0_store(
        self,
        state: GameState,
        *,
        reflection_result: Any,
        execution_success: bool,
        reflection_confirmed_success: bool,
        reflection_status: str,
        store_source: str,
    ) -> bool:
        if self.mem0_provider is None:
            return False

        actions = self._resolve_mem0_store_actions(state)
        if not actions:
            self._record_mem0_store_skip("empty_actions_precheck", "no useful executed actions")
            return False

        if self._is_setup_only_mem0_actions(actions):
            self._record_mem0_store_skip(
                "setup_only_actions",
                f"setup-only actions are not persisted to Mem0 ({actions})",
            )
            return False

        progress_snapshot = self._extract_mem0_progress_snapshot(state)
        hard_progress = bool(progress_snapshot["hard_progress"])
        move_only = self._is_move_only_mem0_actions(actions)

        progress = self._derive_mem0_progress_text(state, reflection_result)
        if move_only and not hard_progress:
            self._record_mem0_store_skip(
                "move_only_no_progress",
                f"move-only actions without hard progress are not persisted to Mem0 ({actions})",
            )
            return False

        if not hard_progress:
            if self.mem0_store_require_meaningful_progress and not self._is_meaningful_progress(progress):
                self._record_mem0_store_skip(
                    "progress_not_meaningful",
                    f"progress not meaningful (progress={progress})",
                )
            else:
                self._record_mem0_store_skip(
                    "no_hard_progress",
                    f"no hard task progress evidence for Mem0 store (progress={progress})",
                )
            return False

        task = state.get('task', '')
        gathered_info = str(state.get('gathered_info', ''))[:300]
        reflection = str(reflection_result)[:300]
        latest_execution_summary = str(state.get('latest_execution_summary', '') or '')[:240]
        state_desc = (
            f"task={task} | progress={progress} | execution={latest_execution_summary} "
            f"| gathered={gathered_info} | reflection={reflection}"
        )
        reward = self._derive_mem0_reward(
            execution_success,
            progress,
            hard_progress=hard_progress,
            completed=progress_snapshot["completed"],
        )

        metadata = {
            "task": task,
            "progress": progress,
            "step_count": state.get('step_count', 0),
            "task_changed": state.get('task_changed', False),
            "memory_quick_path": state.get('memory_quick_path', False),
            "store_source": store_source,
            "reflection_status": reflection_status or "",
            "setup_only": self._is_setup_only_mem0_actions(actions),
            "move_only": move_only,
            "state_changed": bool(state.get("last_state_changed", False)),
            "completed": progress_snapshot["completed"],
            "progress_delta": progress_snapshot["progress_delta"],
            "progress_quantity": progress_snapshot["progress_quantity"],
            "previous_progress_quantity": progress_snapshot["previous_progress_quantity"],
            "hard_progress": hard_progress,
        }

        logger.write(f"[Mem0] Storing record (actions={len(actions)}, reward={reward})")
        self.mem0_provider.store(
            state=state_desc,
            actions=actions,
            success=execution_success if not reflection_confirmed_success else bool(state.get('success', execution_success)),
            reward=reward,
            metadata=metadata,
        )
        logger.write("[Mem0] Store completed")
        return True

    @staticmethod
    def _should_surface_memory_reference(
        state: GameState,
        confidence: float,
        retrieval_mode: str,
    ) -> bool:
        if retrieval_mode == "execute":
            return False
        return confidence > 0.0

    def _format_memory_reference(self, state: GameState) -> str:
        hits = state.get("memory_hits", [])
        actions = self._clean_mem0_actions(state.get("memory_actions", []))
        confidence = float(state.get("memory_confidence", 0.0) or 0.0)
        retrieval_mode = str(state.get("memory_retrieval_mode", "hint") or "hint")

        if not hits and not actions:
            return ""
        if not self._should_surface_memory_reference(state, confidence, retrieval_mode):
            return ""

        successful_hits = [
            hit for hit in (hits or [])
            if isinstance(hit, dict) and (
                bool(hit.get("success", False))
                or float(hit.get("successes", 0) or 0) > 0
            )
        ]
        if not successful_hits:
            return ""
        hits = successful_hits
        if retrieval_mode != "execute":
            confidence = max(confidence, 0.01)

        top_hit = hits[0] if isinstance(hits, list) and hits else {}
        if not isinstance(top_hit, dict):
            top_hit = {}

        state_desc = str(top_hit.get("state", "") or "").replace("\n", " ").strip()
        hit_actions = self._clean_mem0_actions(top_hit.get("actions", []))
        metadata = top_hit.get("metadata", {})
        progress = ""
        if isinstance(metadata, dict):
            progress = str(metadata.get("progress", "") or "").strip()

        chosen_actions = hit_actions or actions
        if not chosen_actions:
            return ""
        if self._is_setup_only_mem0_actions(chosen_actions):
            return ""

        lines = [
            (
                "Long-term memory reference "
                f"(confidence={confidence:.2f}). "
                "Use as historical successful experience when it matches the current state."
            ),
            (
                "If no current target, exit, building, or blocker gives a better grounded next step, "
                f"prefer this same-task route prefix for the next planning horizon: {chosen_actions[:4]}"
            ),
            "If the current facts already show the target adjacent/reachable, act on the current target instead.",
            f"Historical successful action chain: {chosen_actions}",
        ]
        if progress:
            lines.append(f"Historical progress: {progress[:120]}")
        if state_desc:
            lines.append(f"Historical context: {state_desc[:160]}")

        return "\n".join(lines)

    @staticmethod
    def _memory_state_fields(state: GameState) -> Dict[str, Any]:
        return {
            "memory_hits": state.get("memory_hits", []),
            "memory_confidence": state.get("memory_confidence", 0.0),
            "memory_actions": state.get("memory_actions", []),
            "memory_reference": state.get("memory_reference", ""),
            "memory_quick_path": state.get("memory_quick_path", False),
            "memory_retrieval_mode": state.get("memory_retrieval_mode", "hint"),
            "quick_path_consecutive_hits": state.get("quick_path_consecutive_hits", 0),
            "quick_path_guard_reason": state.get("quick_path_guard_reason", ""),
        }
    
    def info_gathering_node(self, state: GameState) -> ProviderOutput:
        """
        节点 1: 信息收集（带性能优化）
        
        调用: VideoClipProvider.gather_information()
        
        性能优化:
        1. 动态帧数: 首次6帧，重试12帧（减少50%处理时间）
        2. 并行编码: 6个worker并行处理图像（70%性能提升）
        3. 流式输出: 边生成边解析（感知延迟↓30-40%）
        
        输入状态：
            - frame_ids: Tuple[int, int]
            - screenshot_path: str
            - video_clip: Dict (可选)
            - retry_count: int (用于动态调整帧数)
        
        输出状态：
            - gathered_info: Dict
                {
                    'description': str,
                    'target_object': str,
                    'minimap_info': Dict,
                    'ui_elements': List,
                    ...
                }
        
        错误处理:
            如果失败，返回 {"error": "Info gathering failed: ..."}
        """
        step_id = state.get('step_id', 0)
        logger.write(f"[step_id={step_id}][node=info_gathering] → Starting")
        frame_ids = state.get('frame_ids', (0, 0))
        retry_count = state.get('retry_count', 0)
        use_stardew_original_input = bool(getattr(self.video_clip_provider, 'use_stardew_original_input', False))

        if use_stardew_original_input:
            target_frame_count = 1 if retry_count == 0 else 2
            logger.write("[InfoGathering] Using Stardew original input (1-2 images + game text)")
            logger.write(f"[LangGraph Node] info_gathering | Frames: {frame_ids} | Image target: {target_frame_count}")
        else:
            # 🚀 优化1: 动态帧数策略
            import yaml
            import os
            from cradle.utils.file_utils import assemble_project_path
            
            # 加载性能配置
            enhanced_config_path = assemble_project_path('./conf/enhanced_config.yaml')
            performance_cfg = {}
            vision_cfg = {}
            
            try:
                if os.path.exists(enhanced_config_path):
                    with open(enhanced_config_path, 'r', encoding='utf-8') as f:
                        enhanced_config = yaml.safe_load(f)
                        performance_cfg = enhanced_config.get('performance', {})
                        vision_cfg = performance_cfg.get('vision', {})
                        logger.write(f"[Config] Loaded vision config: default={vision_cfg.get('default_frame_count')}, reduced={vision_cfg.get('reduced_frame_count')}, dynamic={vision_cfg.get('dynamic_frame_count')}")
                else:
                    logger.write(f"[WARNING] enhanced_config.yaml not found at {enhanced_config_path}")
            except Exception as e:
                logger.write(f"[WARNING] Failed to load enhanced_config.yaml: {e}, using defaults")
            
            if vision_cfg.get('dynamic_frame_count', True):
                if retry_count == 0:
                    # 首次执行：使用减少的帧数
                    target_frame_count = vision_cfg.get('reduced_frame_count', 6)
                    logger.write(f"[Performance] 🎯 First attempt: Using {target_frame_count} frames (optimized)")
                else:
                    # 重试时：使用完整帧数获取更多信息
                    target_frame_count = vision_cfg.get('default_frame_count', 12)
                    logger.write(f"[Performance] 🔄 Retry {retry_count}: Using {target_frame_count} frames (full)")
            else:
                target_frame_count = vision_cfg.get('default_frame_count', 12)
                logger.write(f"[Performance] Using default {target_frame_count} frames")
            
            logger.write(f"[LangGraph Node] info_gathering | Frames: {frame_ids} | Target: {target_frame_count} frames")
        
        try:
            # 关键：同步state到memory.working_area（Provider从memory读取数据）
            from cradle.memory import LocalMemory
            memory = self._get_runtime_memory()
            memory.update_info_history({
                "frame_ids": frame_ids,
                "screenshot_path": state.get('screenshot_path', ''),
                "video_clip": state.get('video_clip', {}),
                "target_frame_count": target_frame_count  # 传递给Provider
            })
            
            # 预处理（可选）：构建图像输入与上下文
            if self.info_gathering_preprocess is not None:
                self.info_gathering_preprocess()

            # 调用现有 Provider（修复：使用 __call__() 标准接口）
            # Provider 会从 memory.working_area 读取必要参数
            heavy_lock_fd = self._acquire_heavy_node_slot("info_gathering")
            try:
                result = self.video_clip_provider()
            finally:
                self._release_heavy_node_slot(heavy_lock_fd)
            logger.write(f"[LangGraph Node] info_gathering raw result: {str(result)[:2000]}")

            # 后处理（可选）：归一化结果
            if self.info_gathering_postprocess is not None:
                result = self.info_gathering_postprocess(result)
            
            if not isinstance(result, dict) or len(result) == 0:
                error_msg = "Info gathering produced empty result"
                logger.warn(f"[LangGraph Node] {error_msg}")
                return {"error": error_msg}

            logger.write(f"[LangGraph Node] ✓ info_gathering completed")

            return sanitize_for_checkpoint({"gathered_info": result})  # type: ignore[return-value]
            
        except Exception as e:
            error_msg = f"Info gathering failed: {str(e)}"
            logger.error(f"[LangGraph Node] {error_msg}")
            logger.error(f"[LangGraph Node] Traceback:\n{traceback.format_exc()}")
            
            return {"error": error_msg}

    def memory_retrieve_node(self, state: GameState) -> ProviderOutput:
        """
        Phase 2.2: Mem0 记忆检索节点

        输出状态：
            - memory_hits
            - memory_confidence
            - memory_actions
            - memory_quick_path
            - planned_actions (若启用快速路径)
        """
        if not self.mem0_enabled or self.mem0_provider is None:
            return sanitize_for_checkpoint({
                "memory_hits": [],
                "memory_confidence": 0.0,
                "memory_actions": [],
                "memory_quick_path": False
            })  # type: ignore[return-value]

        query_parts = [
            f"task: {state.get('task', '')}",
            f"gathered: {str(state.get('gathered_info', ''))[:200]}",
            f"reflection: {str(state.get('reflection_result', ''))[:200]}",
            f"previous_actions: {str(state.get('previous_actions', []))[:160]}",
            f"previous_results: {str(state.get('previous_results', []))[:160]}",
        ]
        query_text = " | ".join([p for p in query_parts if p])

        result = self.mem0_provider.retrieve(query_text)
        memory_actions = result.get("memory_actions", [])
        memory_confidence = float(result.get("memory_confidence", 0.0))

        # Quick path decision
        quick_path = False
        quick_path_guard_reason = ""
        retrieval_mode = "hint"
        prev_quick_hits = int(state.get('quick_path_consecutive_hits', 0))
        quick_path_max_hits = self.mem0_quick_path_max_consecutive_hits
        quick_path_repeat_limit = self.mem0_quick_path_repeat_action_limit
        quick_path_disable_no_embedding = self.mem0_quick_path_disable_without_embedding
        execute_threshold = self.mem0_quick_path_execute_threshold
        max_retry_for_execute = self.mem0_quick_path_max_retry_for_execute
        current_retry_count = int(state.get('retry_count', 0))

        try:
            import yaml
            config_path = assemble_project_path('./conf/enhanced_config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                    mem0_cfg = cfg.get('mem0', {})
                    quick_enabled = False  # Mem0 is reference-only; never execute high-confidence quick path.
                    threshold = float(mem0_cfg.get('quick_path_threshold', 0.85))
                    quick_path_max_hits = int(mem0_cfg.get('quick_path_max_consecutive_hits', quick_path_max_hits))
                    quick_path_repeat_limit = int(mem0_cfg.get('quick_path_repeat_action_limit', quick_path_repeat_limit))
                    quick_path_disable_no_embedding = bool(mem0_cfg.get('quick_path_disable_without_embedding', quick_path_disable_no_embedding))
                    execute_threshold = float(mem0_cfg.get('quick_path_execute_threshold', execute_threshold))
                    max_retry_for_execute = int(mem0_cfg.get('quick_path_max_retry_for_execute', max_retry_for_execute))

                    if quick_path_disable_no_embedding and not self._is_embedding_provider_ready():
                        quick_path = False
                        quick_path_guard_reason = "quick_path_disabled_without_embedding"
                    else:
                        quick_path = (
                            quick_enabled
                            and memory_confidence >= threshold
                            and len(memory_actions) > 0
                            and state.get('error') in (None, '', False)
                        )

                    if quick_path:
                        retrieval_mode = "execute"
                        same_action_streak = self._count_same_action_tail(state.get('previous_actions', []), memory_actions)
                        if same_action_streak >= quick_path_repeat_limit:
                            quick_path = False
                            retrieval_mode = "hint"
                            quick_path_guard_reason = f"quick_path_repeat_guard(streak={same_action_streak})"
                        elif prev_quick_hits >= quick_path_max_hits:
                            quick_path = False
                            retrieval_mode = "hint"
                            quick_path_guard_reason = f"quick_path_consecutive_guard(hits={prev_quick_hits})"
                        elif memory_confidence < execute_threshold:
                            quick_path = False
                            retrieval_mode = "hint"
                            quick_path_guard_reason = (
                                f"quick_path_hint_only(conf={memory_confidence:.3f}<execute_threshold={execute_threshold:.3f})"
                            )
                        elif current_retry_count > max_retry_for_execute:
                            quick_path = False
                            retrieval_mode = "hint"
                            quick_path_guard_reason = (
                                f"quick_path_retry_guard(retry={current_retry_count}>max={max_retry_for_execute})"
                            )
        except Exception:
            pass

        quick_path_consecutive_hits = prev_quick_hits + 1 if quick_path else 0
        if quick_path_guard_reason:
            logger.warn(f"[Mem0] Quick path guarded: {quick_path_guard_reason}")

        output: Dict[str, Any] = {
            "memory_hits": result.get("memory_hits", []),
            "memory_confidence": memory_confidence,
            "memory_actions": memory_actions,
            "memory_quick_path": quick_path,
            "quick_path_consecutive_hits": quick_path_consecutive_hits,
            "quick_path_guard_reason": quick_path_guard_reason,
            "memory_retrieval_mode": retrieval_mode,
        }
        output["memory_reference"] = self._format_memory_reference({**state, **output})
        if output["memory_hits"]:
            logger.write(
                f"[Mem0] Retrieved {len(output['memory_hits'])} successful reference hits "
                f"(source={result.get('memory_source', 'unknown')}, conf={memory_confidence:.3f}, "
                f"actions={len(memory_actions)})"
            )

        try:
            memory_source = str(result.get("memory_source", "unknown"))
            self.mem0_provider.record_quick_path_decision(
                hit=quick_path,
                confidence=memory_confidence,
                memory_source=memory_source,
            )
        except Exception as e:
            logger.warn(f"[Mem0] Failed to record quick-path metrics: {e}")

        if quick_path:
            output.update({
                "planned_actions": memory_actions,
                "planning_reasoning": "memory_retrieve_quick_path"
            })

        return sanitize_for_checkpoint(output)  # type: ignore[return-value]

    def memory_store_node(self, state: GameState) -> ProviderOutput:
        """
        Phase 2.2: Mem0 记忆写入节点
        """
        if not self.mem0_enabled or self.mem0_provider is None:
            logger.write("[Mem0] Store disabled (mem0_enabled=False or provider missing)")
            return self._memory_state_fields(state)

        reflection_result = state.get('reflection_result', {})
        reflection_success = None
        reflection_status = None
        if isinstance(reflection_result, dict):
            reflection_success = reflection_result.get('success')
            reflection_status = reflection_result.get('status')

        reflection_confirmed_success = isinstance(reflection_result, dict) and reflection_success is True
        reflection_explicit_failure = (
            isinstance(reflection_result, dict)
            and str(reflection_status or '').strip().lower() == 'failure'
        )

        if self.mem0_store_require_reflection_success and not reflection_confirmed_success:
            logger.write(
                f"[Mem0] Skip store: reflection not confirmed success (success={reflection_success}, status={reflection_status})"
            )
            try:
                self.mem0_provider.record_store_skip("reflection_not_confirmed")
            except Exception as e:
                logger.warn(f"[Mem0] Failed to record store skip reason: {e}")
            return self._memory_state_fields(state)

        execution_success = bool(state.get('execution_success_raw', state.get('success', False)))
        if reflection_explicit_failure and not reflection_confirmed_success:
            logger.write(
                f"[Mem0] Skip store: reflection explicitly marked failure (success={reflection_success}, status={reflection_status})"
            )
            try:
                self.mem0_provider.record_store_skip("reflection_failure")
            except Exception as e:
                logger.warn(f"[Mem0] Failed to record store skip reason: {e}")
            return self._memory_state_fields(state)

        if self.mem0_store_require_execution_success and not execution_success:
            self._record_mem0_store_skip("execution_not_success", "execution not successful")
            return self._memory_state_fields(state)

        self._commit_mem0_store(
            state,
            reflection_result=reflection_result,
            execution_success=execution_success,
            reflection_confirmed_success=reflection_confirmed_success,
            reflection_status=str(reflection_status or ""),
            store_source="reflection" if reflection_confirmed_success else "execution_feedback",
        )
        return self._memory_state_fields(state)
    
    def self_reflection_node(self, state: GameState) -> ProviderOutput:
        """
        节点 2: 自我反思
        
        调用: SelfReflectionProvider.reflect()
        
        优化: 首步跳过（由路由函数控制，这里不应该被调用）
        
        输入状态：
            - gathered_info: Dict
            - execution_result: Dict (optional)
            - is_first_step: bool
        
        输出状态：
            - reflection_result: Dict
                {
                    'reasoning': str,
                    'success': bool,
                    'skipped': bool (如果跳过)
                }
        
        错误处理:
            如果失败，返回 {"error": "Self reflection failed: ..."}
        """
        logger.write("[LangGraph Node] → self_reflection")
        
        # 安全检查：如果是首步，不应该被调用（路由应该跳过）
        if state.get('is_first_step', False):
            logger.warn("[LangGraph Node] self_reflection called on first step (should be skipped by routing)")
            return sanitize_for_checkpoint({
                "reflection_result": {
                    "reasoning": "First step, no previous action to reflect on",
                    "success": True,
                    "skipped": True
                }
            })  # type: ignore[return-value]
        
        try:
            # 关键：同步state到memory.working_area
            from cradle.memory import LocalMemory
            memory = self._get_runtime_memory()
            latest_action = str(
                state.get("last_action", "")
                or state.get("pre_action", "")
                or state.get("action", "")
                or ""
            ).strip()
            latest_reasoning = str(
                state.get("decision_making_reasoning", "")
                or state.get("pre_decision_making_reasoning", "")
                or state.get("planning_reasoning", "")
                or ""
            ).strip()
            latest_exec_info = state.get("last_exec_info", state.get("exec_info", {}))
            if not isinstance(latest_exec_info, dict):
                latest_exec_info = {}
            prompt_fact_fields = extract_stardew_prompt_fact_fields(
                state=state,
                gathered_info=state.get('gathered_info', {}),
            )
            memory.update_info_history({
                "gathered_info": state.get('gathered_info', {}),
                "execution_result": state.get('execution_result', {}),
                "pre_action": latest_action,
                "action": latest_action,
                "pre_decision_making_reasoning": latest_reasoning,
                "decision_making_reasoning": latest_reasoning,
                "exec_info": latest_exec_info,
                "toolbar_information": state.get("toolbar_information", ""),
                "previous_toolbar_information": state.get("previous_toolbar_information", ""),
                "history_summary": state.get("history_summary", "") or state.get("summarization", ""),
                "summarization": state.get("summarization", "") or state.get("history_summary", ""),
                "subtask_description": state.get("subtask_description", ""),
                "subtask_reasoning": state.get("subtask_reasoning", ""),
                "action_feedback": state.get("action_feedback", ""),
                "previous_actions": state.get('previous_actions', []),
                "previous_results": state.get('previous_results', []),
                **prompt_fact_fields,
            })
            
            # 预处理（可选）：构建图像与上下文输入
            if self.self_reflection_preprocess is not None:
                self.self_reflection_preprocess()

            # 调用现有 Provider（修复：使用 __call__() 标准接口）
            result = self.self_reflection_provider()

            # 后处理（可选）：严格 success 判定与字段归一化
            if self.self_reflection_postprocess is not None:
                result = self.self_reflection_postprocess(result)

            if isinstance(result, dict) and bool(state.get("uncertain_execution", False)):
                result["success"] = False
                result["status"] = "uncertain_execution"
                reasoning_text = str(result.get("reasoning", "") or "").strip()
                uncertainty_note = (
                    "Execution feedback remained uncertain because the action returned no explicit confirmation "
                    "and no reliable state change was observed."
                )
                if uncertainty_note.lower() not in reasoning_text.lower():
                    result["reasoning"] = (
                        f"{reasoning_text}\n{uncertainty_note}".strip()
                        if reasoning_text
                        else uncertainty_note
                    )

            if isinstance(result, dict) and result.get('success') is None:
                if state.get('has_execution_feedback', False):
                    inferred_success = bool(state.get('success', False))
                elif 'execution_success_raw' in state:
                    inferred_success = bool(state.get('execution_success_raw', False))
                else:
                    inferred_success = infer_execution_success_raw(state.get('last_exec_info', {}))
                result['success'] = inferred_success
                result.setdefault('status', 'inferred_from_execution')
            
            success = result.get('success', False)
            logger.write(f"[LangGraph Node] self_reflection completed | Success: {success}")
            
            return sanitize_for_checkpoint({"reflection_result": result})  # type: ignore[return-value]
            
        except Exception as e:
            error_msg = f"Self reflection failed: {str(e)}"
            logger.error(f"[LangGraph Node] {error_msg}")
            logger.error(f"[LangGraph Node] Traceback:\n{traceback.format_exc()}")
            
            return {"error": error_msg}
    
    def task_inference_node(self, state: GameState) -> ProviderOutput:
        """
        节点 3: 任务推理
        
        调用: TaskInferenceProvider.infer_task()
        
        优化: 检测任务变化，减少不必要调用（由路由控制）
        
        输入状态：
            - gathered_info: Dict
            - task: str (optional, previous task)
        
        输出状态：
            - task: str
            - long_horizon_task: str
            - task_changed: bool
        
        错误处理:
            如果失败，返回 {"error": "Task inference failed: ..."}
        """
        logger.write("[LangGraph Node] → task_inference")

        try:
            config_path = assemble_project_path('./conf/enhanced_config.yaml')
            skip_task_inference_threshold = 2
            try:
                import yaml
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        cfg = yaml.safe_load(f) or {}
                        dual_brain_cfg = cfg.get('dual_brain', {}) or {}
                        big_brain_cfg = dual_brain_cfg.get('big_brain', {}) or {}
                        skip_task_inference_cfg = big_brain_cfg.get('skip_task_inference', {}) or {}
                        skip_task_inference_threshold = int(skip_task_inference_cfg.get('consecutive_success_threshold', 2))
            except Exception as e:
                logger.debug(f"[LangGraph Node] Failed to load task inference skip config: {e}")

            previous_subtask = str(state.get('subtask_description', '') or '')
            previous_subtask_reasoning = str(state.get('subtask_reasoning', '') or '')
            previous_history_summary = str(
                state.get('history_summary', '') or state.get('summarization', '') or ''
            )
            main_task = str(
                state.get('main_task', '')
                or state.get('env_config', {}).get('target_task_description', '')
                or state.get('task', '')
            )
            if bool(state.get("blocker_replan_only", False)) and previous_subtask:
                logger.write(
                    "[LangGraph Node] Skipping task_inference and reusing current subtask "
                    "(blocker_replan_only)"
                )
                return sanitize_for_checkpoint({
                    "task": main_task,
                    "main_task": main_task,
                    "subtask_description": previous_subtask,
                    "subtask_reasoning": previous_subtask_reasoning,
                    "summarization": previous_history_summary,
                    "history_summary": previous_history_summary,
                    "long_horizon_task": state.get('long_horizon_task', ''),
                    "task_changed": False
                })  # type: ignore[return-value]
            consecutive_successes = self._count_recent_objective_successes(
                main_task,
                state.get('previous_results', []),
            )
            refresh_requires_new_subtask = False
            refresh_reason = ""

            if (
                state.get('dual_brain_enabled', False)
                and skip_task_inference_threshold > 0
                and consecutive_successes >= skip_task_inference_threshold
                and previous_subtask
            ):
                if self._subtask_tool_selection_already_satisfied(previous_subtask, state):
                    refresh_requires_new_subtask = True
                    refresh_reason = "previous subtask is stale because target tool is already selected"
                    logger.write(
                        f"[LangGraph Node] Not skipping task_inference: previous subtask is stale because target tool is already selected ({self._extract_active_tool_from_state(state)})"
                    )
                elif stale_subtask_reason := self._subtask_conflicts_with_current_facts(previous_subtask, main_task, state):
                    refresh_requires_new_subtask = True
                    refresh_reason = stale_subtask_reason
                    logger.write(
                        f"[LangGraph Node] Not skipping task_inference: previous subtask conflicts with current facts ({stale_subtask_reason})"
                    )
                elif self._has_recent_instability(state):
                    logger.write(
                        "[LangGraph Node] Not skipping task_inference: recent execution feedback is unstable "
                        f"(zero_progress_streak={int(state.get('zero_progress_streak', 0) or 0)}, "
                        f"repeated_action_streak={int(state.get('repeated_action_streak', 0) or 0)}, "
                        f"consecutive_failures={int(state.get('consecutive_failures', 0) or 0)}, "
                        f"position_issue_detected={bool(state.get('position_issue_detected', False))}, "
                        f"last_state_changed={state.get('last_state_changed', True)})"
                    )
                else:
                    logger.write(
                        f"[LangGraph Node] Skipping task_inference (recent_successes={consecutive_successes} >= threshold={skip_task_inference_threshold})"
                    )
                    return sanitize_for_checkpoint({
                        "task": main_task,
                        "main_task": main_task,
                        "subtask_description": previous_subtask,
                        "subtask_reasoning": previous_subtask_reasoning,
                        "summarization": previous_history_summary,
                        "history_summary": previous_history_summary,
                        "long_horizon_task": state.get('long_horizon_task', ''),
                        "task_changed": False
                    })  # type: ignore[return-value]

            # 关键：同步state到memory.working_area
            from cradle.memory import LocalMemory
            memory = self._get_runtime_memory()
            acquisition_context = build_task_acquisition_context(main_task)
            prompt_fact_fields = extract_stardew_prompt_fact_fields(
                state=state,
                gathered_info=state.get('gathered_info', {}),
            )
            memory.update_info_history({
                "gathered_info": state.get('gathered_info', {}),
                "reflection_result": state.get('reflection_result', {}),
                "task": main_task,
                "step_count": state.get('step_count', 0),
                # Field aliases: template variable names <- actual state/memory field names
                # Use "None" instead of empty string to prevent template paragraph skipping
                "previous_action": state.get('pre_action', '') or "None",
                "previous_reasoning": state.get('pre_decision_making_reasoning', '') or "None",
                "self_reflection_reasoning": (
                    state.get('reflection_result', {}).get('self_reflection_reasoning', '')
                    if isinstance(state.get('reflection_result'), dict) else ''
                ) or "None",
                "previous_summarization": state.get('history_summary', '') or "None",
                "subtask_description": state.get('subtask_description', '') or "None",
                "subtask_reasoning": state.get('subtask_reasoning', '') or "None",
                "memory_reference": state.get("memory_reference", "") or self._format_memory_reference(state),
                "image_description": (
                    state.get('gathered_info', {}).get('image_description', '')
                    if isinstance(state.get('gathered_info'), dict) else ''
                ) or (
                    state.get('gathered_info', {}).get('description', '')
                    if isinstance(state.get('gathered_info'), dict) else ''
                ) or "None",
                **prompt_fact_fields,
                **acquisition_context,
            })
            
            # 预处理（可选）：构建图像输入与上下文
            if self.task_inference_preprocess is not None:
                self.task_inference_preprocess()

            # 调用现有 Provider（修复：使用 __call__() 标准接口）
            result = self.task_inference_provider()

            # 后处理（可选）：归一化字段
            if self.task_inference_postprocess is not None:
                result = self.task_inference_postprocess(result)
            
            # 从postprocess结果中提取task
            # Stardew: subtask_description字段包含当前子任务
            # RDR2/其他: 可能使用task_guidance或reasoning
            explicit_subtask = result.get('subtask_description', '') or result.get('task_guidance', '')
            explicit_subtask = str(explicit_subtask or '').strip()
            reasoning_fallback = str(result.get('reasoning', '') or '').strip()
            if not self._looks_like_precise_subtask_text(explicit_subtask):
                explicit_subtask = ""
            if not explicit_subtask and self._looks_like_precise_subtask_text(reasoning_fallback):
                new_subtask = reasoning_fallback
            else:
                new_subtask = explicit_subtask
            new_subtask_reasoning = str(
                result.get('subtask_reasoning', '')
                or result.get('reasoning', '')
                or previous_subtask_reasoning
                or build_initial_subtask_reasoning(main_task)
                or ''
            ).strip()
            history_summary = str(
                result.get('history_summary', '')
                or result.get('summarization', '')
                or previous_history_summary
                or ''
            ).strip()
            
            # 防御性检查：如果 subtask 为空，保持上一个子任务
            if not new_subtask or not new_subtask.strip():
                if previous_subtask and not refresh_requires_new_subtask:
                    logger.write(f"[LangGraph Node] Task inference returned empty subtask, keeping previous subtask: '{previous_subtask}'")
                    new_subtask = previous_subtask
                    if not new_subtask_reasoning:
                        new_subtask_reasoning = previous_subtask_reasoning
                else:
                    recovery_subtask, recovery_reasoning = self._build_current_fact_recovery_subtask(main_task, state)
                    new_subtask = recovery_subtask or build_initial_subtask(main_task)
                    if not new_subtask_reasoning:
                        new_subtask_reasoning = recovery_reasoning or build_initial_subtask_reasoning(main_task)
                    if refresh_reason:
                        logger.write(
                            f"[LangGraph Node] Task inference returned empty subtask after forced refresh ({refresh_reason}), using default: '{new_subtask}'"
                        )
                    else:
                        logger.write(f"[LangGraph Node] Task inference returned empty subtask, using default: '{new_subtask}'")
            
            # 检测子任务变化
            if self._subtask_claims_completion_without_progress(new_subtask, state):
                recovery_subtask, recovery_reasoning = self._build_current_fact_recovery_subtask(main_task, state)
                fallback_subtask = recovery_subtask or build_initial_subtask(main_task)
                logger.write(
                    "[LangGraph Node] Task inference claimed completion without evaluator progress, "
                    f"using default: '{fallback_subtask}'"
                )
                new_subtask = fallback_subtask
                new_subtask_reasoning = recovery_reasoning or build_initial_subtask_reasoning(main_task)

            if stale_new_subtask_reason := self._subtask_conflicts_with_current_facts(new_subtask, main_task, state):
                recovery_subtask, recovery_reasoning = self._build_current_fact_recovery_subtask(main_task, state)
                fallback_subtask = recovery_subtask or build_initial_subtask(main_task)
                logger.write(
                    "[LangGraph Node] Task inference produced a stale subtask against current facts "
                    f"({stale_new_subtask_reason}), using default: '{fallback_subtask}'"
                )
                new_subtask = fallback_subtask
                new_subtask_reasoning = recovery_reasoning or build_initial_subtask_reasoning(main_task)

            task_changed = (new_subtask != previous_subtask) if previous_subtask else True
            
            if task_changed:
                logger.write(f"[LangGraph Node] Subtask changed: '{previous_subtask}' → '{new_subtask}'")
            
            return sanitize_for_checkpoint({
                "task": main_task,
                "main_task": main_task,
                "subtask_description": new_subtask,
                "subtask_reasoning": new_subtask_reasoning,
                "summarization": history_summary,
                "history_summary": history_summary,
                "long_horizon_task": result.get('long_horizon_task', ''),
                "task_changed": task_changed
            })  # type: ignore[return-value]
            
        except Exception as e:
            error_msg = f"Task inference failed: {str(e)}"
            logger.error(f"[LangGraph Node] {error_msg}")
            logger.error(f"[LangGraph Node] Traceback:\n{traceback.format_exc()}")
            
            return {"error": error_msg}
    
    def action_planning_node(self, state: GameState) -> ProviderOutput:
        """
        节点 4: 动作规划
        
        调用: ActionPlanningProvider.plan_action()
        
        这是核心节点，消耗时间最长（60s），必须调用。
        
        输入状态：
            - task: str
            - gathered_info: Dict
            - reflection_result: Dict (optional)
        
        输出状态：
            - planned_actions: List[str]
            - planning_reasoning: str
        
        错误处理:
            如果失败，返回 {"error": "Action planning failed: ..."}
        """
        logger.write(f"[LangGraph Node] → action_planning | Task: {state.get('task', 'N/A')[:50]}...")
        
        try:
            # 关键：同步state到memory.working_area（修复Bug #21：必须包含图像数据才能用vision mode）
            from cradle.memory import LocalMemory
            memory = self._get_runtime_memory()
            acquisition_context = build_task_acquisition_context(state.get('task', ''))
            prompt_fact_fields = extract_stardew_prompt_fact_fields(
                state=state,
                gathered_info=state.get('gathered_info', {}),
            )
            memory.update_info_history({
                "task": state.get('task', ''),
                "task_description": state.get('task', '') or state.get('main_task', ''),
                "main_task": state.get('main_task', '') or state.get('task', ''),
                "gathered_info": state.get('gathered_info', {}),
                "reflection_result": state.get('reflection_result', {}),
                "summarization": state.get('summarization', '') or state.get('history_summary', ''),
                "history_summary": state.get('history_summary', '') or state.get('summarization', ''),
                "subtask_description": state.get('subtask_description', ''),
                "subtask_reasoning": state.get('subtask_reasoning', ''),
                "previous_actions": state.get('previous_actions', []),
                "last_action": state.get('last_action', ''),
                "last_exec_info": state.get('last_exec_info', {}),
                "latest_task_eval": state.get('latest_task_eval', {}),
                "previous_task_progress_quantity": state.get('previous_task_progress_quantity', None),
                "task_progress_quantity": state.get('task_progress_quantity', None),
                "task_progress_delta": state.get('task_progress_delta', None),
                "zero_progress_streak": state.get('zero_progress_streak', 0),
                "repeated_action_streak": state.get('repeated_action_streak', 0),
                "position_issue_detected": state.get('position_issue_detected', False),
                "last_state_changed": state.get('last_state_changed', True),
                "latest_execution_summary": state.get('latest_execution_summary', ''),
                "action_feedback": state.get('action_feedback', '') or state.get('latest_execution_summary', ''),
                "recent_execution_feedback": state.get('recent_execution_feedback', []),
                "memory_hits": state.get('memory_hits', []),
                "memory_actions": state.get('memory_actions', []),
                "memory_confidence": state.get('memory_confidence', 0.0),
                "memory_retrieval_mode": state.get('memory_retrieval_mode', 'hint'),
                "memory_reference": state.get("memory_reference", "") or self._format_memory_reference(state),
                "skill_library": state.get('skill_library', ''),
                "action": state.get('pre_action', '') or state.get('action', ''),
                "action_planning_reasoning": (
                    state.get('pre_decision_making_reasoning', '')
                    or state.get('decision_making_reasoning', '')
                ),
                # Bug #21修复：添加图像数据，ActionPlanningProvider需要这些才能用vision mode
                "screenshot_path": state.get('screenshot_path', ''),
                "video_clip": state.get('video_clip', {}),
                "augmented_image_path": state.get('augmented_image_path', ''),
                # Dual-brain mode flag for postprocess to bypass truncation
                "dual_brain_enabled": state.get('dual_brain_enabled', False),
                "sanitized_subtask_hint": state.get('sanitized_subtask_hint', ''),
                "redundant_tool_selection": state.get('redundant_tool_selection', False),
                "selected_item_already_correct": state.get('selected_item_already_correct', False),
                **prompt_fact_fields,
                **acquisition_context,
            })

            # 允许流式过程中提前执行动作（不停止生成）
            memory.update_info_history({
                "early_action_executed": False,
                "early_action_in_flight": False,
                "early_action_steps": [],
                "early_action_exec_info": {},
                "early_action_frame_ids": (0, 0)
            })

            def _schedule_early_action(action_text):
                # 避免重复触发
                if memory.get_working_area_value("early_action_in_flight") or memory.get_working_area_value("early_action_executed"):
                    return

                def _run():
                    try:
                        # 允许传入单个动作或动作列表
                        action_list = action_text if isinstance(action_text, list) else [action_text]
                        memory.update_info_history({
                            "early_action_in_flight": True,
                            "early_action_steps": action_list,
                            "skill_steps": action_list,
                            "screen_classification": state.get('screen_classification', ''),
                            "pre_screen_classification": state.get('pre_screen_classification', ''),
                            "pre_action": state.get('pre_action', '')
                        })
                        logger.write(f"[EarlyAction] ⚡ Executing actions immediately: {action_list}")
                        result = self.skill_execute_provider()
                        memory.update_info_history({
                            "early_action_executed": True,
                            "early_action_in_flight": False,
                            "early_action_exec_info": result.get('exec_info', {}),
                            "early_action_frame_ids": (
                                result.get('start_frame_id', 0),
                                result.get('end_frame_id', 0)
                            )
                        })
                        logger.write("[EarlyAction] ✓ Action executed (early)")
                    except Exception as early_error:
                        memory.update_info_history({"early_action_in_flight": False})
                        logger.warn(f"[EarlyAction] Failed: {early_error}")

                threading.Thread(target=_run, daemon=True).start()

            # 将回调注入 ActionPlanning（StardewPlanner 路径）
            try:
                planner = getattr(self.action_planning_provider, 'planner', None)
                if planner is not None and hasattr(planner, 'action_planning_'):
                    planner.action_planning_.on_action_callback = _schedule_early_action
            except Exception as cb_error:
                logger.warn(f"[EarlyAction] Callback injection failed: {cb_error}")

            # Detect consecutive move failures and inject stuck warning
            previous_actions = state.get('previous_actions', [])
            previous_results = state.get('previous_results', [])
            if previous_actions and len(previous_actions) >= 2:
                consecutive_move_fails = 0
                for i in range(len(previous_actions) - 1, -1, -1):
                    action_str = str(previous_actions[i])
                    # Check if it's a move that failed (exec_info errors or zero_progress)
                    if action_str.startswith("move("):
                        # Check result if available
                        result_ok = True
                        if i < len(previous_results):
                            r = previous_results[i] if isinstance(previous_results[i], dict) else {}
                            result_ok = r.get("success", True)
                        # Also check zero_progress_streak
                        zero_streak = int(state.get("zero_progress_streak", 0) or 0)
                        if not result_ok or zero_streak >= 2:
                            consecutive_move_fails += 1
                        else:
                            break
                    else:
                        break

                if consecutive_move_fails >= 2:
                    stuck_warning = (
                        f"[CRITICAL WARNING] The last {consecutive_move_fails} move attempts all FAILED. "
                        "The player is stuck and cannot reach the target by repeating the same move. "
                        "You MUST try a different strategy: "
                        "1) Move in the opposite direction or try the other axis first to find a clear path. "
                        "2) Use a tool (Pickaxe/Axe/Scythe) to clear obstacles in front. "
                        "3) Change the axis order or sidestep 3-5 tiles, for example move(x=0, y=3) before moving right. "
                        "DO NOT repeat the same blocked move or the same blocked axis order."
                    )
                    memory.update_info_history({"action_feedback": stuck_warning})
                    logger.warn(f"[LangGraph Node] Injected stuck warning into action planning context")

            # 预处理（可选）：构建图像输入与上下文
            if self.action_planning_preprocess is not None:
                self.action_planning_preprocess()

            # 调用现有 Provider（修复：使用 __call__() 而不是 plan_action()）
            heavy_lock_fd = self._acquire_heavy_node_slot("action_planning")
            try:
                result = self.action_planning_provider()
            finally:
                self._release_heavy_node_slot(heavy_lock_fd)

            # 清理回调，避免影响后续步骤
            try:
                planner = getattr(self.action_planning_provider, 'planner', None)
                if planner is not None and hasattr(planner, 'action_planning_'):
                    planner.action_planning_.on_action_callback = None
            except Exception:
                pass

            # 后处理（可选）：归一 actions/skill_steps
            if self.action_planning_postprocess is not None:
                result = self.action_planning_postprocess(result)
            
            actions = result.get('skill_steps', result.get('actions', []))
            reasoning = result.get('decision_making_reasoning', result.get('reasoning', ''))
            
            logger.write(f"[LangGraph Node] ✓ action_planning | {len(actions)} actions: {actions}")
            
            return sanitize_for_checkpoint({
                "planned_actions": actions,
                "planning_reasoning": reasoning
            })  # type: ignore[return-value]
            
        except Exception as e:
            error_msg = f"Action planning failed: {str(e)}"
            logger.error(f"[LangGraph Node] {error_msg}")
            logger.error(f"[LangGraph Node] Traceback:\n{traceback.format_exc()}")
            
            return {"error": error_msg}
    
    def skill_execute_node(self, state: GameState) -> ProviderOutput:
        """
        节点 5: 技能执行
        
        调用: SkillExecuteProvider.execute()
        
        输入状态：
            - planned_actions: List[str]
        
        输出状态：
            - execution_result: Dict
            - executed_frames: Tuple[int, int]
            - success: bool
            - step_count: int (incremented)
            - is_first_step: bool (set to False)
            - consecutive_failures: int (updated)
        
        错误处理:
            如果失败，返回 {"error": "...", "success": False}
        """
        planned_actions = state.get('planned_actions', [])
        actions = planned_actions if isinstance(planned_actions, list) else []
        logger.write(f"[LangGraph Node] → skill_execute | Actions: {len(actions)}")

        if len(actions) == 0:
            consecutive_failures = state.get('consecutive_failures', 0) + 1
            retry_count = state.get('retry_count', 0) + 1
            logger.warn("[LangGraph Node] No planned actions, forcing replanning")
            return {
                "error": "no_actions_planned",
                "success": False,
                "execution_success_raw": False,
                "consecutive_failures": consecutive_failures,
                "retry_count": retry_count,
            }
        
        try:
            # 关键：同步state到memory.working_area（skill_steps是必需的）
            from cradle.memory import LocalMemory
            memory = self._get_runtime_memory()
            # 如果早执行仍在进行，等待完成（避免重复执行）
            if memory.get_working_area_value("early_action_in_flight"):
                for _ in range(50):  # 最多等待5秒
                    time.sleep(0.1)
                    if not memory.get_working_area_value("early_action_in_flight"):
                        break

            early_executed = memory.get_working_area_value("early_action_executed", False)
            early_steps = memory.get_working_area_value("early_action_steps", [])
            if early_executed and isinstance(early_steps, list) and actions[:len(early_steps)] == early_steps:
                logger.write("[EarlyAction] ✓ Skipping already executed prefix actions")
                exec_info = memory.get_working_area_value("early_action_exec_info", {})
                frame_ids = memory.get_working_area_value("early_action_frame_ids", (0, 0))
                # 清理早执行标记，避免影响后续步骤
                memory.update_info_history({
                    "early_action_executed": False,
                    "early_action_in_flight": False,
                    "early_action_steps": [],
                    "early_action_exec_info": {},
                    "early_action_frame_ids": (0, 0)
                })

                # 仍需执行剩余动作（如果有）
                remaining_actions = actions[len(early_steps):]
                if remaining_actions:
                    memory.update_info_history({
                        "skill_steps": remaining_actions,
                        "screen_classification": state.get('screen_classification', ''),
                        "pre_screen_classification": state.get('pre_screen_classification', ''),
                        "pre_action": state.get('pre_action', '')
                    })
                    result = self.skill_execute_provider()
                else:
                    result = {
                        "exec_info": exec_info,
                        "start_frame_id": frame_ids[0],
                        "end_frame_id": frame_ids[1]
                    }
            else:
                memory.update_info_history({
                    "skill_steps": actions,  # Provider读取skill_steps而不是planned_actions
                    "screen_classification": state.get('screen_classification', ''),
                    "pre_screen_classification": state.get('pre_screen_classification', ''),
                    "pre_action": state.get('pre_action', '')
                })
                
                # 调用现有 Provider（修复：使用 __call__() 标准接口）
                result = self.skill_execute_provider()
            
            # 修复Bug #20：SkillExecuteProvider返回exec_info，通过exec_info['done']和executed_skills判断成功
            exec_info = result.get('exec_info', {})
            # 成功条件：1) exec_info['done']=True 或 2) 有技能被成功执行
            executed_skills = exec_info.get('executed_skills', [])
            is_done = exec_info.get('done', False)
            execution_success_raw = infer_execution_success_raw(exec_info)
            success = execution_success_raw

            # Phase 2.2 guard: 防止在同一动作上“成功但无进展”自激振荡
            repeated_action_streak = self._count_same_action_tail(state.get('previous_actions', []), actions)
            no_progress_repeat_limit = self.no_progress_repeat_action_limit
            try:
                import yaml
                config_path = assemble_project_path('./conf/enhanced_config.yaml')
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as f:
                        cfg = yaml.safe_load(f)
                        mem0_cfg = cfg.get('mem0', {})
                        no_progress_repeat_limit = int(mem0_cfg.get('no_progress_repeat_action_limit', no_progress_repeat_limit))
            except Exception:
                pass

            if success and repeated_action_streak >= no_progress_repeat_limit:
                success = False
                result['error'] = f"no_progress_repeated_actions(streak={repeated_action_streak})"
                logger.warn(f"[LangGraph Node] Repeated action detected without progress, forcing replanning: streak={repeated_action_streak}")
            
            # 修复：从result中获取正确的frame_ids字段
            start_frame = result.get('start_frame_id', 0)
            end_frame = result.get('end_frame_id', 0)
            frame_ids = (start_frame, end_frame)
            
            # 每步截图：动作执行后立即截图并增强（保证下一步使用最新图像）
            latest_screenshot_path = ''
            latest_augmented_path = ''
            try:
                if self.gm is not None:
                    latest_screenshot_path = self.gm.capture_screen()
                    memory.update_info_history({
                        "screenshot_path": latest_screenshot_path,
                        constants.IMAGES_MEM_BUCKET: latest_screenshot_path
                    })
                    if self.augment_provider is not None:
                        self.augment_provider()
                        augmented_list = memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, k=1)
                        if augmented_list:
                            latest_augmented_path = augmented_list[0]
            except Exception as capture_error:
                logger.warn(f"[LangGraph Node] Screenshot capture/augment failed: {capture_error}")

            # 更新连续失败计数
            consecutive_failures = state.get('consecutive_failures', 0)
            # 修复Bug #23：同时更新retry_count用于路由判断
            retry_count = state.get('retry_count', 0)
            
            if not success:
                consecutive_failures += 1
                retry_count += 1  # Bug #23修复：失败时递增retry_count
                logger.warn(f"[LangGraph Node] Execution failed (consecutive failures: {consecutive_failures}, retry: {retry_count})")
            else:
                consecutive_failures = 0
                retry_count = 0  # Bug #23修复：成功时重置retry_count
                logger.write("[LangGraph Node] Execution succeeded")

                # 长期记忆补强：仅在“首步且无反思结果”时兜底写入，避免与memory_store双写污染
                reflection_result = state.get('reflection_result')
                allow_fallback_store = bool(state.get('is_first_step', False)) and not isinstance(reflection_result, dict)
                if self.mem0_enabled and self.mem0_provider is not None and allow_fallback_store:
                    try:
                        if self.mem0_store_require_reflection_success:
                            self._record_mem0_store_skip(
                                "reflection_not_confirmed",
                                "fallback store blocked because reflection is required",
                            )
                        else:
                            self._commit_mem0_store(
                                state,
                                reflection_result=reflection_result,
                                execution_success=True,
                                reflection_confirmed_success=False,
                                reflection_status="",
                                store_source="skill_execute_success_fallback_first_step",
                            )
                            logger.write("[Mem0] Evaluated successful action from skill_execute fallback (first step)")
                    except Exception as mem0_store_error:
                        logger.warn(f"[Mem0] skill_execute success store failed: {mem0_store_error}")
            
            # 更新步数计数
            step_count = state.get('step_count', 0) + 1
            
            # 修复Bug：累积previous_actions历史，让LLM知道已执行的动作
            previous_actions = list(state.get('previous_actions', []))
            previous_actions.append(actions)  # type: ignore[arg-type]
            
            # 同样累积previous_results
            previous_results = list(state.get('previous_results', []))
            previous_results.append({
                "action": actions,
                "success": success,
                "execution_success_raw": execution_success_raw,
                "errors": bool(exec_info.get('errors', False) or result.get('error')),
                "errors_info": str(exec_info.get('errors_info', '') or ''),
                "executed_skills": list(executed_skills) if isinstance(executed_skills, list) else [],
                "last_skill": str(exec_info.get('last_skill', '') or ''),
                "exec_info": exec_info,
            })
            
            logger.write(f"[LangGraph Node] ✓ skill_execute | Success: {success} | Step: {step_count} | History: {len(previous_actions)}")
            
            # 修复Bug：清理返回数据中的Skill对象
            return sanitize_for_checkpoint({
                "execution_result": result,
                "executed_frames": frame_ids,
                "success": success,
                "execution_success_raw": execution_success_raw,
                "step_count": step_count,
                "is_first_step": False,  # 从第二步开始都不是首步
                "consecutive_failures": consecutive_failures,
                "retry_count": retry_count,  # Bug #23修复：返回更新后的retry_count
                "screenshot_path": latest_screenshot_path,
                "augmented_image_path": latest_augmented_path,
                "previous_actions": previous_actions,  # 返回更新后的历史
                "previous_results": previous_results
            })  # type: ignore[return-value]
            
        except Exception as e:
            error_msg = f"Skill execution failed: {str(e)}"
            logger.error(f"[LangGraph Node] {error_msg}")
            logger.error(f"[LangGraph Node] Traceback:\n{traceback.format_exc()}")
            
            # 增加失败计数
            consecutive_failures = state.get('consecutive_failures', 0) + 1
            
            return {
                "error": error_msg,
                "success": False,
                "consecutive_failures": consecutive_failures
            }
    
    def parallel_info_and_reflect_node(self, state: GameState) -> ProviderOutput:
        """
        🚀 Phase 2.1: 并行节点 - 同时执行info_gathering和self_reflection
        
        性能优化:
            串行: info_gathering(15s) + self_reflection(22s) = 37s
            并行: max(15s, 22s) = 22s
            节省: 15s (40%提升)
        
        实现方式:
            使用asyncio.gather()并行执行两个节点
            合并输出到单个state更新
        
        输入状态:
            - 所有info_gathering和self_reflection需要的字段
        
        输出状态:
            - gathered_info: Dict (来自info_gathering)
            - reflection: Dict (来自self_reflection)
        
        错误处理:
            - 如果任一失败，捕获错误并记录
            - 部分成功: 返回成功的结果 + error字段
            - 完全失败: 返回error
        """
        import asyncio
        import time
        
        step_id = state.get('step_id', 0)
        step_count = state.get('step_count', 0)
        logger.write(f"[step_id={step_id}][node=parallel] → Starting PARALLEL execution")
        
        start_time = time.time()

        # 是否要求 self_reflection 使用更新后的 gathered_info/截图
        reflect_after_info = False
        try:
            import yaml
            from cradle.utils.file_utils import assemble_project_path
            config_path = assemble_project_path('./conf/enhanced_config.yaml')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = yaml.safe_load(f)
                    reflect_after_info = cfg.get('langgraph', {}).get('parallel', {}).get('reflect_after_info', False)
        except Exception:
            pass
        
        # 🔧 Phase 2.1 Fix: 检查是否是首步，首步跳过self_reflection
        is_first_step = bool(state.get('is_first_step', False))
        
        async def run_parallel():
            """异步执行两个节点"""
            # 🔧 P1 CRITICAL FIX: 使用ThreadPoolExecutor将同步节点包装为真正的并行任务
            # asyncio.gather对同步代码无效，必须用executor
            from concurrent.futures import ThreadPoolExecutor
            loop = asyncio.get_running_loop()

            def _run_node_in_isolated_memory(node_fn, node_state):
                runtime_memory = self._get_runtime_memory()
                if hasattr(runtime_memory, "run_with_isolated_scope"):
                    return runtime_memory.run_with_isolated_scope(node_fn, node_state)
                return node_fn(node_state)
            
            async def run_info_gathering():
                task_start = time.time()
                logger.write(f"[Parallel Timing] → info_gathering started at {task_start:.3f}")
                try:
                    # 在独立线程中运行同步代码，避免阻塞event loop
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        result = await loop.run_in_executor(
                            executor,
                            _run_node_in_isolated_memory,
                            self.info_gathering_node,
                            state,
                        )
                    task_end = time.time()
                    task_duration = task_end - task_start
                    logger.write(f"[Parallel Timing] ✓ info_gathering completed in {task_duration:.2f}s (ended at {task_end:.3f})")
                    return ("info", result, task_start, task_end)
                except Exception as e:
                    task_end = time.time()
                    logger.error(f"[Parallel Timing] ✗ info_gathering failed after {task_end - task_start:.2f}s: {e}")
                    return ("info", {"error": str(e)}, task_start, task_end)
            
            async def run_self_reflection(run_state: GameState):
                if is_first_step:
                    logger.write("[Parallel Timing] → self_reflection SKIPPED (first step optimization)")
                    return ("reflect", {}, 0, 0)  # 返回空结果，节省22s
                
                task_start = time.time()
                logger.write(f"[Parallel Timing] → self_reflection started at {task_start:.3f}")
                try:
                    # 在独立线程中运行同步代码，避免阻塞event loop
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        result = await loop.run_in_executor(
                            executor,
                            _run_node_in_isolated_memory,
                            self.self_reflection_node,
                            run_state,
                        )
                    task_end = time.time()
                    task_duration = task_end - task_start
                    logger.write(f"[Parallel Timing] ✓ self_reflection completed in {task_duration:.2f}s (ended at {task_end:.3f})")
                    return ("reflect", result, task_start, task_end)
                except Exception as e:
                    task_end = time.time()
                    logger.error(f"[Parallel Timing] ✗ self_reflection failed after {task_end - task_start:.2f}s: {e}")
                    return ("reflect", {"error": str(e)}, task_start, task_end)
            
            # 如果需要使用更新后的 gathered_info/截图，则先完成 info_gathering 再反思
            if not is_first_step and reflect_after_info:
                logger.write("[Parallel] 🔁 self_reflection waits for updated gathered_info")
                info_key, info_result, info_start, info_end = await run_info_gathering()
                if isinstance(info_result, dict) and info_result.get("error"):
                    return [(info_key, info_result, info_start, info_end), ("reflect", {"error": "self_reflection skipped due to info_gathering error"}, 0, 0)]

                updated_state = cast(GameState, dict(state))
                if isinstance(info_result, dict):
                    updated_state.update(info_result)  # type: ignore[arg-type]

                reflect_key, reflect_result, reflect_start, reflect_end = await run_self_reflection(updated_state)
                return [(info_key, info_result, info_start, info_end), (reflect_key, reflect_result, reflect_start, reflect_end)]

            # 并行执行 - 现在真正会并行运行
            results = await asyncio.gather(
                run_info_gathering(),
                run_self_reflection(state),
                return_exceptions=True
            )
            
            return results
        
        try:
            # 运行异步任务 - 使用线程局部持久event loop
            # 避免关闭loop后httpx AsyncClient cleanup失败
            _loop = getattr(_node_loop_storage, 'loop', None)
            if _loop is None or _loop.is_closed():
                _loop = asyncio.new_event_loop()
                asyncio.set_event_loop(_loop)
                _node_loop_storage.loop = _loop
            results = _loop.run_until_complete(run_parallel())
            
            elapsed = time.time() - start_time
            
            # 🔍 P1 Fix: 分析并行性能
            logger.write(f"[Parallel Timing] PERFORMANCE ANALYSIS")
            logger.write(f"[Parallel Timing] Total elapsed: {elapsed:.2f}s")
            
            # 提取任务时间
            task_times = {}
            for item in results:
                if isinstance(item, tuple) and len(item) == 4:
                    task_name, task_result, task_start, task_end = item
                    if task_start > 0:  # 排除跳过的任务
                        task_times[task_name] = {
                            'start': task_start,
                            'end': task_end,
                            'duration': task_end - task_start
                        }
            
            # 计算并行度
            if len(task_times) >= 2:
                times_list = list(task_times.values())
                earliest_start = min(t['start'] for t in times_list)
                latest_end = max(t['end'] for t in times_list)
                total_span = latest_end - earliest_start
                
                # 计算重叠时间
                overlap_start = max(t['start'] for t in times_list)
                overlap_end = min(t['end'] for t in times_list)
                overlap = max(0, overlap_end - overlap_start)
                
                concurrency_pct = (overlap / total_span * 100) if total_span > 0 else 0
                
                logger.write(f"[Parallel Timing] Task durations:")
                for name, times in task_times.items():
                    logger.write(f"[Parallel Timing]   {name}: {times['duration']:.2f}s")
                logger.write(f"[Parallel Timing] Time span: {total_span:.2f}s")
                logger.write(f"[Parallel Timing] Overlap: {overlap:.2f}s")
                logger.write(f"[Parallel Timing] Concurrency: {concurrency_pct:.1f}%")
                
                if concurrency_pct > 50:
                    logger.write(f"[Parallel Timing] ✅ GOOD CONCURRENCY - Tasks executed in parallel")
                else:
                    logger.write(f"[Parallel Timing] LOW CONCURRENCY - Tasks may have run sequentially")
            else:
                logger.write(f"[Parallel Timing] Only one task executed (first step optimization)")
            
            # 合并结果
            combined_output = {}
            for item in results:
                if isinstance(item, tuple) and len(item) >= 2:
                    task_name, task_result = item[0], item[1]
                    if isinstance(task_result, dict):
                        combined_output.update(task_result)
                elif isinstance(item, Exception):
                    logger.error(f"[Parallel] Task raised exception: {item}")
            
            logger.write(f"[Parallel] ✓ Parallel execution completed")
            return sanitize_for_checkpoint(combined_output)  # type: ignore[return-value]
            
        except Exception as e:
            error_msg = f"Parallel execution failed: {str(e)}"
            logger.error(f"[Parallel] {error_msg}")
            logger.error(f"[Parallel] Traceback:\n{traceback.format_exc()}")
            
            return {"error": error_msg}


# ========== 辅助函数 ==========

def create_nodes_from_runner(runner) -> LangGraphNodes:
    """
    从 Runner 实例创建 LangGraphNodes
    
    这是一个便捷函数，自动提取 Runner 中的 Provider 实例。
    
    Args:
        runner: Runner 实例
    
    Returns:
        LangGraphNodes: 节点适配器实例
    
    Example:
        >>> from cradle.runner import Runner
        >>> runner = Runner(config)
        >>> nodes = create_nodes_from_runner(runner)
    """
    providers = {
        'video_clip': runner.video_clip_provider,
        'self_reflection': runner.self_reflection_provider,
        'task_inference': runner.task_inference_provider,
        'action_planning': runner.action_planning_provider,
        'skill_execute': runner.skill_execute_provider
    }
    
    gm = getattr(runner, 'gm', None)
    augment_provider = getattr(runner, 'augment', None)
    action_planning_preprocess = getattr(runner, 'action_planning_preprocess', None)
    action_planning_postprocess = getattr(runner, 'action_planning_postprocess', None)
    self_reflection_preprocess = getattr(runner, 'self_reflection_preprocess', None)
    self_reflection_postprocess = getattr(runner, 'self_reflection_postprocess', None)
    info_gathering_preprocess = getattr(runner, 'information_gathering_preprocess', None)
    info_gathering_postprocess = getattr(runner, 'information_gathering_postprocess', None)
    task_inference_preprocess = getattr(runner, 'task_inference_preprocess', None)
    task_inference_postprocess = getattr(runner, 'task_inference_postprocess', None)

    return LangGraphNodes(
        providers,
        gm=gm,
        augment_provider=augment_provider,
        action_planning_preprocess=action_planning_preprocess,
        action_planning_postprocess=action_planning_postprocess,
        self_reflection_preprocess=self_reflection_preprocess,
        self_reflection_postprocess=self_reflection_postprocess,
        info_gathering_preprocess=info_gathering_preprocess,
        info_gathering_postprocess=info_gathering_postprocess,
        task_inference_preprocess=task_inference_preprocess,
        task_inference_postprocess=task_inference_postprocess,
        runtime_memory=getattr(runner, "cortex_memory", None),
    )
