"""
Fast LLM client for little-brain autonomous decisions (Phase 3.2).

Uses DashScope OpenAI-compatible API with nothinking mode (enable_thinking=false)
for fast action decisions. Differentiates from big brain's thinking mode.

Originally designed for local vLLM, refactored to use cloud API for
zero-deployment convenience and higher model capability.
"""
import ast
import hashlib
import json
import os
import re
import threading
import time
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests

from cradle.log import Logger
from cradle.utils.file_utils import assemble_project_path
from cradle.utils.llm_call_budget import increment_llm_call_counter
from cradle.utils.llm_endpoint_throttle import (
    LLMEndpointThrottleTimeout,
    acquire_llm_endpoint_slot,
    get_llm_endpoint_wait_timeout,
    resolve_remaining_llm_request_timeout,
)
from stardojo.utils.execution_feedback_utils import (
    execution_has_no_confirmation,
    execution_refusal_type,
    execution_refused_action,
)
from stardojo.utils.llm_timing_utils import add_llm_retry_overhead
from stardojo.utils.stardew_prompt_state import extract_stardew_prompt_fact_fields
from stardojo.utils.task_grounding import (
    build_clear_task_profile,
    classify_clearable_target,
    classify_tilling_target,
    clear_target_matches_profile,
    count_nearby_hard_structures as grounding_count_nearby_hard_structures,
    count_nearby_open_ground_tiles as grounding_count_nearby_open_ground_tiles,
    is_empty_like_tile as grounding_is_empty_like_tile,
    is_explicit_tillable_ground,
    is_hard_structure_text,
    is_open_ground_tile as grounding_is_open_ground_tile,
    is_safe_empty_till_use_target,
)

logger = Logger()

# Regex pattern for stardojo-style template variables: <$variable_name$>
_PLACEHOLDER_RE = re.compile(r"<\$([^$]+)\$>")


@dataclass
class VLLMDecision:
    """Result of a fast LLM decision call."""

    action: str  # Skill call string, e.g. "choose_item(slot_index=1)"
    reason: str  # Brief reason (20 chars)
    escalate: bool  # True = request big-brain intervention


class VLLMClient:
    """Fast LLM client using DashScope API with nothinking mode.

    Uses the same model family as big brain but with enable_thinking=false,
    producing fast direct outputs (~1-3s) without chain-of-thought overhead.

    Config is read from openai_config.json (same as big brain):
        - api_key: llm_api_key
        - base_url: dashscope compatible-mode endpoint
        - model: qwen-plus (nothinking)

    Class name kept as VLLMClient for backward compatibility.
    """

    _TOOL_DISPLAY_NAMES = {
        "watering can": "Watering Can",
        "pickaxe": "Pickaxe",
        "scythe": "Scythe",
        "axe": "Axe",
        "hoe": "Hoe",
    }
    _TERMINAL_COMPOSITE_SKILLS = set()
    _CARDINAL_DIRECTIONS = ("up", "right", "down", "left")
    _SHARED_HEALTH_CACHE_SUCCESS_TTL_S = 15.0
    _SHARED_HEALTH_CACHE_FAILURE_TTL_S = 3.0
    _SHARED_HEALTH_LOCK_POLL_S = 0.1
    _SHARED_HEALTH_LOCK_STALE_S = 30.0

    def __init__(
        self,
        base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1",
        model: str = "qwen-plus",
        api_key: str = "",
        secondary_api_key: str = "",
        max_tokens: int = 300,
        request_timeout_s: int = 30,
        health_check_timeout_s: Optional[float] = None,
        primary_weight: int = 5,
        secondary_weight: int = 1,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.primary_api_key = str(api_key or "").strip()
        self.secondary_api_key = str(secondary_api_key or "").strip()
        self.api_key = self.primary_api_key
        self.max_tokens = max_tokens
        self.request_timeout_s = request_timeout_s
        if health_check_timeout_s is None:
            health_check_timeout_s = self._default_health_check_timeout(
                self.request_timeout_s
            )
        self.health_check_timeout_s = max(1.0, float(health_check_timeout_s))
        self.primary_weight = max(1, int(primary_weight))
        self.secondary_weight = max(1, int(secondary_weight))
        self._rr_lock = threading.Lock()
        self._rr_counter = 0
        self.last_effective_duration_s: float = 0.0
        # Template for stardojo-style prompt (set externally after init)
        self.template: Optional[str] = None

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_retry_config() -> Dict[str, float]:
        defaults = {"max_retries": 10.0, "retry_interval_s": 8.0}
        try:
            import yaml

            cfg_path = assemble_project_path("./conf/enhanced_config.yaml")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                streaming = ((raw.get("performance") or {}).get("streaming") or {})
                defaults["max_retries"] = float(streaming.get("llm_request_max_retries", defaults["max_retries"]))
                defaults["retry_interval_s"] = float(
                    streaming.get("llm_request_retry_interval_seconds", defaults["retry_interval_s"])
                )
        except Exception:
            pass
        return defaults

    @staticmethod
    def _is_retryable_503_error(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        try:
            if int(status_code) == 503:
                return True
        except (TypeError, ValueError):
            pass
        lowered = str(exc or "").lower()
        return "503" in lowered and "service" in lowered and "unavailable" in lowered

    @staticmethod
    def _default_health_check_timeout(request_timeout_s: float) -> float:
        return min(15.0, max(5.0, float(request_timeout_s)))

    @classmethod
    def from_openai_config(
        cls,
        config_path: str = "conf/openai_config.json",
        model_override: Optional[str] = None,
    ) -> "VLLMClient":
        """Create from the project's openai_config.json.

        Reads the same config file as big brain, using llm_api_key.
        """
        from cradle.utils.file_utils import assemble_project_path

        resolved = assemble_project_path(config_path)
        with open(resolved, "r", encoding="utf-8") as f:
            conf = json.load(f)

        api_key = conf.get("llm_api_key") or conf.get("api_key") or ""
        secondary_key = conf.get("secondary_api_key") or ""
        # Use same base_url as big brain (coding.dashscope or compatible-mode)
        base_url = conf.get("base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        model = model_override or conf.get("comp_model", "qwen-plus")

        if secondary_key and isinstance(secondary_key, str) and secondary_key.strip():
            logger.write(
                "[FastLLM] Dual-key load distribution enabled "
                "(primary:secondary ~= 5:1)"
            )

        return cls(
            base_url=base_url,
            model=model,
            api_key=api_key,
            secondary_api_key=secondary_key,
        )

    def _iter_available_keys(self) -> List[tuple[str, str]]:
        keys: List[tuple[str, str]] = []
        if self.primary_api_key:
            keys.append(("primary", self.primary_api_key))
        if (
            self.secondary_api_key
            and self.secondary_api_key != self.primary_api_key
        ):
            keys.append(("secondary", self.secondary_api_key))
        return keys

    def _get_next_api_key(self) -> tuple[str, str]:
        keys = self._iter_available_keys()
        if not keys:
            return "primary", ""
        if len(keys) == 1:
            return keys[0]

        cycle_size = self.primary_weight + self.secondary_weight
        with self._rr_lock:
            cycle_pos = self._rr_counter % cycle_size
            self._rr_counter += 1

        if cycle_pos < self.primary_weight:
            return "primary", self.primary_api_key
        return "secondary", self.secondary_api_key

    def _shared_health_cache_key(self) -> str:
        payload = f"{self.base_url}|{self.model}".encode("utf-8", errors="ignore")
        return hashlib.sha1(payload).hexdigest()[:16]

    def _get_shared_health_check_paths(self) -> tuple[str, str]:
        cache_dir = assemble_project_path("./cache/locks/fastllm_health")
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = self._shared_health_cache_key()
        cache_path = os.path.join(cache_dir, f"{cache_key}.json")
        lock_path = os.path.join(cache_dir, f"{cache_key}.lock")
        return cache_path, lock_path

    def _read_shared_health_check_cache(self, now_s: Optional[float] = None) -> Optional[bool]:
        cache_path, _ = self._get_shared_health_check_paths()
        current_time = time.time() if now_s is None else float(now_s)
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
        except FileNotFoundError:
            return None
        except Exception:
            return None

        available = bool(payload.get("available", False))
        timestamp = payload.get("time")
        try:
            recorded_time = float(timestamp)
        except (TypeError, ValueError):
            try:
                recorded_time = os.path.getmtime(cache_path)
            except OSError:
                return None

        max_age_s = (
            self._SHARED_HEALTH_CACHE_SUCCESS_TTL_S
            if available
            else self._SHARED_HEALTH_CACHE_FAILURE_TTL_S
        )
        if current_time - recorded_time > max_age_s:
            return None
        return available

    def _write_shared_health_check_cache(
        self,
        available: bool,
        now_s: Optional[float] = None,
    ) -> None:
        cache_path, _ = self._get_shared_health_check_paths()
        current_time = time.time() if now_s is None else float(now_s)
        temp_path = f"{cache_path}.{os.getpid()}.tmp"
        payload = {
            "available": bool(available),
            "time": current_time,
            "model": self.model,
            "base_url": self.base_url,
        }
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False)
            os.replace(temp_path, cache_path)
        except Exception:
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass

    def _acquire_shared_health_probe_lock(self) -> tuple[Optional[int], Optional[str]]:
        _, lock_path = self._get_shared_health_check_paths()
        stale_after_s = max(
            self._SHARED_HEALTH_LOCK_STALE_S,
            float(self.health_check_timeout_s) * 2.0,
        )
        while True:
            try:
                lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(
                    lock_fd,
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "time": time.time(),
                            "model": self.model,
                        },
                        ensure_ascii=False,
                    ).encode("utf-8"),
                )
                return lock_fd, lock_path
            except FileExistsError:
                try:
                    age_s = time.time() - os.path.getmtime(lock_path)
                    if age_s > stale_after_s:
                        os.remove(lock_path)
                        continue
                except FileNotFoundError:
                    continue
                except OSError:
                    pass
                return None, lock_path

    @staticmethod
    def _release_shared_health_probe_lock(lock_fd: Optional[int], lock_path: Optional[str]) -> None:
        if lock_fd is not None:
            try:
                os.close(lock_fd)
            except OSError:
                pass
        if lock_path:
            try:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
            except OSError:
                pass

    def _reuse_shared_health_result_while_probe_in_progress(
        self,
        effective_timeout: float,
    ) -> Optional[bool]:
        deadline = time.time() + min(3.0, max(0.5, effective_timeout))
        while time.time() < deadline:
            cached_result = self._read_shared_health_check_cache()
            if cached_result is not None:
                return cached_result
            time.sleep(self._SHARED_HEALTH_LOCK_POLL_S)
        return self._read_shared_health_check_cache()

    def health_check(self, timeout_s: Optional[float] = None) -> bool:
        """Verify API connectivity with a minimal request."""
        key_candidates = self._iter_available_keys()
        if not key_candidates:
            logger.warn("[FastLLM] No API key configured")
            return False

        effective_timeout = max(
            1.0,
            float(
                timeout_s
                if timeout_s is not None
                else getattr(self, "health_check_timeout_s", self._default_health_check_timeout(self.request_timeout_s))
            ),
        )

        cached_result = self._read_shared_health_check_cache()
        if cached_result is not None:
            logger.write(
                "[FastLLM] Reusing recent shared health result "
                f"({self.model}: {'available' if cached_result else 'unavailable'})"
            )
            return cached_result

        probe_lock_fd, probe_lock_path = self._acquire_shared_health_probe_lock()
        if probe_lock_fd is None:
            reused_result = self._reuse_shared_health_result_while_probe_in_progress(
                effective_timeout,
            )
            if reused_result is not None:
                logger.write(
                    "[FastLLM] Reusing shared health result from another worker "
                    f"({self.model}: {'available' if reused_result else 'unavailable'})"
                )
                return reused_result
            logger.warn(
                "[FastLLM] Another worker is already probing FastLLM health; "
                "deferring duplicate startup probe"
            )
            return False

        available = False
        try:
            for key_name, api_key in key_candidates:
                try:
                    slot_wait_timeout_s = get_llm_endpoint_wait_timeout(
                        self.model,
                        total_timeout_s=effective_timeout,
                    )
                    with acquire_llm_endpoint_slot(
                        model_name=self.model,
                        purpose="little_brain_healthcheck",
                        logger_obj=logger,
                        timeout_s=slot_wait_timeout_s,
                    ) as slot_info:
                        request_timeout_s = resolve_remaining_llm_request_timeout(
                            effective_timeout,
                            slot_info,
                        )
                        resp = requests.post(
                            f"{self.base_url}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {api_key}",
                                "Content-Type": "application/json",
                            },
                            json={
                                "model": self.model,
                                "messages": [{"role": "user", "content": "ping"}],
                                "max_tokens": 1,
                                "chat_template_kwargs": {"enable_thinking": False},
                            },
                            timeout=request_timeout_s,
                        )
                    if resp.status_code == 200:
                        available = True
                        logger.write(
                            f"[FastLLM] Health check passed "
                            f"(model={self.model}, key={key_name}, nothinking)"
                        )
                        return True
                    logger.warn(
                        f"[FastLLM] Health check returned {resp.status_code} "
                        f"for {key_name} key: {resp.text[:200]}"
                    )
                except LLMEndpointThrottleTimeout as e:
                    logger.warn(
                        "[FastLLM] Health check skipped because the shared endpoint queue "
                        f"did not clear in time: {e}"
                    )
                except Exception as e:
                    logger.warn(f"[FastLLM] Health check failed for {key_name} key: {e}")
            return False
        finally:
            self._write_shared_health_check_cache(available)
            self._release_shared_health_probe_lock(probe_lock_fd, probe_lock_path)

    @staticmethod
    def _detect_stuck_warning(execution_log: List[Dict[str, Any]], game_state: Optional[Dict[str, Any]] = None) -> str:
        """Detect consecutive move failures and generate a stuck warning for the LLM."""
        if not execution_log or len(execution_log) < 2:
            return ""

        # Check last 5 entries for consecutive move failures
        recent = execution_log[-5:]
        consecutive_move_fails = 0
        failed_moves = []
        for entry in reversed(recent):
            action_str = str(entry.get("action", ""))
            success = entry.get("success", True)
            if action_str.startswith("move(") and not success:
                consecutive_move_fails += 1
                failed_moves.append(action_str)
            else:
                break

        if consecutive_move_fails < 2:
            return ""

        # Build warning with context
        warning_parts = [
            f"\n[WARNING: YOU ARE STUCK] The last {consecutive_move_fails} move attempts all failed to change player position.",
            f"Failed moves: {', '.join(reversed(failed_moves))}.",
            "DO NOT repeat the same move. Try one of these strategies:",
            "1. Move in the opposite direction, try the other axis first, or sidestep to find a clear path.",
            "2. If the tile in front of you contains a clearable obstacle, try rerouting around it first. Only clear it when rerouting is not grounded or repeated reroutes still fail: Pickaxe for stone, Axe for wood/twig/log, Scythe for weeds/grass/fiber.",
            "3. If the target needs both x and y movement, change the order: try the clear axis first, e.g. move(x=0, y=3) before moving right, or move(x=-3, y=0) to backtrack.",
        ]

        return "\n".join(warning_parts)

    @staticmethod
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

    @classmethod
    def _is_menu_open(cls, menu_value: Any) -> bool:
        return cls._normalize_menu_type(menu_value) not in {"", "no menu", "none", "null"}

    @staticmethod
    def _get_current_menu_value(
        game_state: Optional[Dict[str, Any]],
        gathered: Optional[Dict[str, Any]] = None,
    ) -> Any:
        if not isinstance(game_state, dict):
            return None
        if not isinstance(gathered, dict):
            candidate = game_state.get("gathered_info", {})
            gathered = candidate if isinstance(candidate, dict) else {}

        prompt_facts = extract_stardew_prompt_fact_fields(
            state=game_state,
            gathered_info=gathered,
        )
        return prompt_facts.get("current_menu")

    @staticmethod
    def _menu_has_response_options(menu_value: Any) -> bool:
        if not isinstance(menu_value, dict):
            return False

        responses = menu_value.get("responses")
        if isinstance(responses, (list, tuple, set)):
            return len(responses) > 0
        return responses not in (None, "", [])

    @staticmethod
    def _menu_prefers_negative_confirmation(menu_value: Any) -> bool:
        if not isinstance(menu_value, dict):
            return False

        responses = menu_value.get("responses")
        if not isinstance(responses, (list, tuple)):
            return False

        normalized_responses: set[str] = set()
        for response in responses:
            if isinstance(response, dict):
                label = response.get("responseText") or response.get("responseKey") or ""
            else:
                label = response
            normalized = str(label or "").strip().lower()
            if normalized:
                normalized_responses.add(normalized)

        if not {"yes", "no"}.issubset(normalized_responses):
            return False

        dialogue_lines: List[str] = []
        for field in ("dialogues", "chats", "message"):
            raw_value = menu_value.get(field)
            if isinstance(raw_value, (list, tuple, set)):
                dialogue_lines.extend(str(item or "").strip() for item in raw_value if str(item or "").strip())
            elif str(raw_value or "").strip():
                dialogue_lines.append(str(raw_value or "").strip())

        normalized_dialogue = " ".join(
            re.sub(r"[^a-z0-9]+", " ", line.lower()).strip()
            for line in dialogue_lines
            if line
        ).strip()
        if not normalized_dialogue:
            return False

        return bool(re.search(r"\b(eat|consume|drink)\b", normalized_dialogue))

    @staticmethod
    def _format_menu_action(option: str, menu_name: str) -> str:
        return f'menu(option={json.dumps(option)}, menu_name={json.dumps(menu_name)})'

    @staticmethod
    def _parse_choose_option_index(action_text: Any) -> Optional[int]:
        match = re.match(
            r"^choose_option\(\s*option_index\s*=\s*(-?\d+)",
            str(action_text or "").strip(),
            re.IGNORECASE,
        )
        if not match:
            return None
        return int(match.group(1))

    @classmethod
    def _extract_task_quantity(
        cls,
        game_state: Optional[Dict[str, Any]],
        *prefixes: str,
    ) -> Optional[int]:
        if not isinstance(game_state, dict):
            return None

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        normalized_prefixes = tuple(
            cls._normalize_context_text(prefix) for prefix in prefixes if str(prefix or "").strip()
        )
        if not normalized_prefixes:
            return None

        for raw_text in (
            game_state.get("task"),
            game_state.get("main_task"),
            game_state.get("task_description"),
            gathered.get("task"),
            gathered.get("main_task"),
            gathered.get("task_description"),
        ):
            normalized_text = cls._normalize_context_text(raw_text)
            if not normalized_text:
                continue
            for prefix in normalized_prefixes:
                match = re.search(rf"\b{re.escape(prefix)}\s+(\d+)\b", normalized_text)
                if match:
                    quantity = int(match.group(1))
                    if quantity > 0:
                        return quantity
        return None

    @classmethod
    def _extract_shop_menu_entries(cls, menu_value: Any) -> List[Dict[str, Any]]:
        if not isinstance(menu_value, dict):
            return []

        raw_entries = menu_value.get("shopmenudata")
        if not isinstance(raw_entries, list):
            return []

        entries: List[Dict[str, Any]] = []
        for option_index, raw_entry in enumerate(raw_entries, start=1):
            if isinstance(raw_entry, dict):
                raw_name = (
                    raw_entry.get("name")
                    or raw_entry.get("Name")
                    or raw_entry.get("item")
                    or ""
                )
            else:
                raw_name = raw_entry
            normalized_name = cls._normalize_context_text(raw_name)
            if not normalized_name:
                continue
            entries.append(
                {
                    "option_index": option_index,
                    "name": normalized_name,
                }
            )
        return entries

    @classmethod
    def _find_target_inventory_slot(
        cls,
        inventory_slot_map: Dict[int, str],
        target_item: str,
    ) -> Optional[int]:
        normalized_target = cls._normalize_context_text(target_item)
        if not normalized_target:
            return None

        candidates: List[tuple[int, int]] = []
        for slot_index, item_name in inventory_slot_map.items():
            normalized_item = cls._normalize_context_text(item_name)
            if not normalized_item or cls._slot_is_explicitly_empty(item_name):
                continue
            if (
                normalized_item == normalized_target
                or normalized_target in normalized_item
                or normalized_item in normalized_target
            ):
                score = 0 if normalized_item == normalized_target else 1
                candidates.append((score, int(slot_index)))

        if not candidates:
            return None
        candidates.sort()
        return candidates[0][1]

    @classmethod
    def _build_shop_menu_purchase_action(
        cls,
        game_state: Optional[Dict[str, Any]],
        *,
        current_menu: Any,
    ) -> str:
        if not isinstance(game_state, dict):
            return ""
        if cls._normalize_menu_type(current_menu) != "shopmenu":
            return ""

        target_item = cls._target_item_text(game_state)
        if not target_item:
            return ""

        option_index: Optional[int] = None
        for entry in cls._extract_shop_menu_entries(current_menu):
            entry_name = str(entry.get("name") or "").strip()
            if (
                entry_name == target_item
                or target_item in entry_name
                or entry_name in target_item
            ):
                option_index = int(entry["option_index"])
                break
        if option_index is None:
            return ""

        quantity = cls._extract_task_quantity(game_state, "purchase") or 1
        return (
            f'choose_option(option_index={option_index}, '
            f'quantity={int(quantity)}, direction="in")'
        )

    @classmethod
    def _build_shop_menu_sell_action(
        cls,
        game_state: Optional[Dict[str, Any]],
        *,
        current_menu: Any,
        inventory_slot_map: Dict[int, str],
        selected_slot: Optional[int],
    ) -> str:
        if not isinstance(game_state, dict):
            return ""
        if cls._normalize_menu_type(current_menu) != "shopmenu":
            return ""

        target_item = cls._target_item_text(game_state)
        target_slot = cls._find_target_inventory_slot(inventory_slot_map, target_item)
        if target_slot is None:
            return ""

        if selected_slot != target_slot:
            return f"choose_item(slot_index={target_slot})"

        quantity = cls._extract_task_quantity(game_state, "sell") or 1
        return (
            f'choose_option(option_index={int(target_slot) + 1}, '
            f'quantity={int(quantity)}, direction="out")'
        )

    @classmethod
    def _is_sleep_task_context(cls, game_state: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(game_state, dict):
            return False
        task_text = " ".join(
            str(game_state.get(key, "") or "")
            for key in ("task", "main_task", "task_description", "subtask_description")
        ).lower()
        normalized = re.sub(r"[^a-z0-9]+", " ", task_text).strip()
        return any(
            token in normalized
            for token in (
                "go to bed",
                "go_to_bed",
                "sleep",
                "enter door and sleep",
                "enter_door_and_sleep",
            )
        )

    @classmethod
    def _find_adjacent_bed_direction(cls, game_state: Optional[Dict[str, Any]]) -> str:
        if not isinstance(game_state, dict):
            return ""

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        facing_direction = str(gathered.get("facing_direction", "") or "").strip().lower()
        preferred_directions = list(cls._CARDINAL_DIRECTIONS)
        if facing_direction in preferred_directions:
            preferred_directions.remove(facing_direction)
            preferred_directions.insert(0, facing_direction)

        for direction in preferred_directions:
            target_obj, _ = cls._get_directional_target(game_state, direction)
            if "bed" in str(target_obj or "").strip().lower():
                return direction
        return ""

    @classmethod
    def _sleep_scene_text(cls, game_state: Optional[Dict[str, Any]]) -> str:
        if not isinstance(game_state, dict):
            return ""
        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}
        parts: List[str] = []
        for source in (game_state, gathered):
            for key in (
                "location",
                "description",
                "other",
                "surroundings",
                "subtask_description",
                "task_description",
            ):
                value = source.get(key) if isinstance(source, dict) else None
                if value:
                    parts.append(str(value))
        return cls._normalize_context_text(" ".join(parts))

    @classmethod
    def _is_inside_farmhouse_sleep_context(cls, game_state: Optional[Dict[str, Any]]) -> bool:
        scene_text = cls._sleep_scene_text(game_state)
        location_text = cls._location_text(game_state)
        return (
            "farmhouse" in location_text
            or "inside farmhouse" in scene_text
            or "inside the farmhouse" in scene_text
            or "farmhouse interior" in scene_text
            or "inside of the player s small farmhouse" in scene_text
        )

    @classmethod
    def _sleep_task_at_farmhouse_entrance(cls, game_state: Optional[Dict[str, Any]]) -> bool:
        scene_text = cls._sleep_scene_text(game_state)
        if "bed" not in scene_text:
            return False
        if not cls._is_inside_farmhouse_sleep_context(game_state):
            return False
        return any(
            token in scene_text
            for token in (
                "doorway area",
                "bottom entrance",
                "bottom doorway",
                "standing in the doorway",
                "near the bottom entrance",
                "standing near the bottom entrance",
                "standing near the lower entrance",
            )
        )

    @staticmethod
    def _menu_contains_sleep_prompt(menu_value: Any) -> bool:
        text_parts: List[str] = []
        if isinstance(menu_value, dict):
            for key in ("dialogues", "chats", "message", "responses"):
                value = menu_value.get(key)
                if value is None:
                    continue
                if isinstance(value, list):
                    text_parts.extend(str(item) for item in value)
                else:
                    text_parts.append(str(value))
        else:
            text_parts.append(str(menu_value or ""))

        text = " ".join(text_parts).lower()
        return any(
            token in text
            for token in (
                "sleep",
                "go to sleep",
                "for the night",
                "yes",
                "no",
            )
        )

    @staticmethod
    def _extract_selected_item_name(gathered: Dict[str, Any]) -> str:
        selected_item_name = (
            gathered.get("selected_item_name")
            or gathered.get("selected_tool")
            or gathered.get("current_tool")
        )
        if selected_item_name:
            return str(selected_item_name).strip()

        chosen_item = gathered.get("chosen_item")
        if isinstance(chosen_item, dict):
            for key in ("currentitem", "current_item", "item_name", "name", "item"):
                candidate = chosen_item.get(key)
                if candidate:
                    return str(candidate).strip()
        if chosen_item:
            return str(chosen_item).strip()
        return ""

    @staticmethod
    def _parse_move_direction(action_text: Any) -> Optional[str]:
        match = re.match(
            r"^move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)$",
            str(action_text or "").strip(),
            re.IGNORECASE,
        )
        if not match:
            return None

        x = int(match.group(1))
        y = int(match.group(2))
        if x and y:
            return None
        if x > 0:
            return "right"
        if x < 0:
            return "left"
        if y > 0:
            return "down"
        if y < 0:
            return "up"
        return None

    @staticmethod
    def _parse_move_components(action_text: Any) -> Optional[tuple[int, int]]:
        match = re.match(
            r"^move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)$",
            str(action_text or "").strip(),
            re.IGNORECASE,
        )
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    @staticmethod
    def _parse_skill_name(action_text: Any) -> str:
        match = re.match(r"^\s*([A-Za-z_]\w*)\s*\(", str(action_text or "").strip())
        if not match:
            return ""
        return match.group(1)

    @staticmethod
    def _normalize_action_text(action_text: Any) -> str:
        text = str(action_text or "").strip()
        if not text:
            return ""
        text = text.replace("'", '"')
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\s*=\s*", "=", text)
        text = re.sub(r"\s*,\s*", ", ", text)
        text = re.sub(r"\(\s+", "(", text)
        text = re.sub(r"\s+\)", ")", text)
        return text

    @classmethod
    def _build_refused_move_recovery_action(cls, action_text: Any) -> str:
        move = cls._parse_move_components(action_text)
        if move is None:
            return ""

        x_val, y_val = move
        if abs(x_val) >= abs(y_val) and x_val != 0:
            y_dir = 1 if y_val > 0 else -1 if y_val < 0 else (1 if x_val > 0 else -1)
            return f"move(x=0, y={y_dir})"

        if x_val != 0:
            x_dir = 1 if x_val > 0 else -1
        else:
            x_dir = -1 if y_val > 0 else 1
        return f"move(x={x_dir}, y=0)"

    @staticmethod
    def _direction_to_relative(direction: str) -> Optional[tuple]:
        direction = str(direction or "").strip().lower()
        if direction == "up":
            return (0, -1)
        if direction == "down":
            return (0, 1)
        if direction == "left":
            return (-1, 0)
        if direction == "right":
            return (1, 0)
        return None

    @staticmethod
    def _parse_surroundings_map(surroundings: Any) -> Dict[tuple, str]:
        cells: Dict[tuple, str] = {}
        for raw_line in str(surroundings or "").splitlines():
            line = raw_line.strip()
            match = re.match(r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]\s*:\s*(.+)$", line)
            if not match:
                continue
            cell = (int(match.group(1)), int(match.group(2)))
            obj_text = match.group(3).strip()
            cells[cell] = obj_text
        return cells

    @staticmethod
    def _classify_clearable_object(obj_text: Any) -> Optional[Dict[str, str]]:
        return classify_clearable_target(obj_text)

    @staticmethod
    def _find_tool_slot(inventory: Any, tool_name: str) -> Optional[int]:
        if not isinstance(inventory, list):
            return None

        tool_lower = str(tool_name or "").strip().lower()
        if not tool_lower:
            return None

        for item in inventory:
            match = re.search(
                r"slot_index\s+(\d+)\s*:\s*([^()]+)",
                str(item),
                re.IGNORECASE,
            )
            if not match:
                continue
            slot_index = int(match.group(1))
            item_name = match.group(2).strip().lower()
            if item_name == tool_lower:
                return slot_index
        return None

    @staticmethod
    def _normalize_toolbar_information_text(toolbar_information: Any) -> str:
        if isinstance(toolbar_information, str):
            return toolbar_information

        if isinstance(toolbar_information, (list, tuple)):
            lines: List[str] = []
            for item in toolbar_information:
                line = str(item or "").strip()
                if line:
                    lines.append(line)
            return "\n".join(lines)

        return str(toolbar_information or "")

    @staticmethod
    def _find_tool_slot_in_toolbar_text(toolbar_information: Any, tool_name: str) -> Optional[int]:
        tool_lower = str(tool_name or "").strip().lower()
        if not tool_lower:
            return None

        toolbar_text = VLLMClient._normalize_toolbar_information_text(toolbar_information)
        patterns = (
            (re.compile(r"slot_index\s+(\d+)\s*:\s*([^()]+)", re.IGNORECASE), False),
            (re.compile(r"^\s*(\d+)\.\s*([^:]+):", re.IGNORECASE), True),
        )
        for raw_line in toolbar_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for pattern, one_based in patterns:
                match = pattern.search(line)
                if not match:
                    continue
                raw_slot_index = int(match.group(1))
                slot_index = max(raw_slot_index - 1, 0) if one_based else raw_slot_index
                item_name = match.group(2).strip().lower()
                if item_name == tool_lower:
                    return slot_index
        return None

    @staticmethod
    def _parse_grid_position(position: Any) -> Optional[tuple[int, int]]:
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            x, y = position[0], position[1]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return int(x), int(y)

        if isinstance(position, dict):
            x = position.get("x", position.get("X"))
            y = position.get("y", position.get("Y"))
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return int(x), int(y)

        match = re.search(r"(-?\d+)\s*,\s*(-?\d+)", str(position or ""))
        if not match:
            return None

        return int(match.group(1)), int(match.group(2))

    @classmethod
    def _get_target_absolute_position(
        cls,
        game_state: Optional[Dict[str, Any]],
        direction: str,
    ) -> Optional[tuple[int, int]]:
        if not isinstance(game_state, dict):
            return None

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        current_position = cls._parse_grid_position(
            gathered.get("position")
            or gathered.get("current_position")
            or game_state.get("position")
            or game_state.get("current_position")
        )
        relative = cls._direction_to_relative(direction)
        if current_position is None or relative is None:
            return None

        return current_position[0] + relative[0], current_position[1] + relative[1]

    @classmethod
    def _extract_crop_positions(cls, game_state: Optional[Dict[str, Any]]) -> set[tuple[int, int]]:
        if not isinstance(game_state, dict):
            return set()

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        crop_text = gathered.get("crops") or game_state.get("crops", "")
        positions: set[tuple[int, int]] = set()
        for raw_line in str(crop_text or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = re.search(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", line)
            if not match:
                match = re.search(r"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", line)
            if match:
                positions.add((int(match.group(1)), int(match.group(2))))

        return positions

    @staticmethod
    def _is_empty_like_tile(target_obj: Any) -> bool:
        return grounding_is_empty_like_tile(target_obj)

    @classmethod
    def _normalize_tool_name(cls, item_name: Any) -> str:
        item_lower = str(item_name or "").strip().lower()
        if not item_lower:
            return ""

        for tool_lower, display_name in cls._TOOL_DISPLAY_NAMES.items():
            if tool_lower in item_lower:
                return display_name
        return ""

    @classmethod
    def _is_valid_hoe_target(cls, target_obj: Any) -> bool:
        return is_explicit_tillable_ground(target_obj)

    @classmethod
    def _is_tilling_or_digging_context(cls, game_state: Optional[Dict[str, Any]]) -> bool:
        if not isinstance(game_state, dict):
            return False

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        context_text = " ".join(
            str(
                value
                or ""
            )
            for value in (
                game_state.get("task"),
                game_state.get("main_task"),
                game_state.get("task_description"),
                game_state.get("subtask_description"),
                gathered.get("task_description"),
                gathered.get("subtask_description"),
            )
        ).lower()
        normalized = f" {re.sub(r'[^a-z0-9]+', ' ', context_text)} "
        relevant_tokens = (
            " till ",
            " dig ",
            " cultivate ",
            " sow ",
            " plant ",
            " seed ",
            " seeds ",
            " growth cycle ",
            " cave carrot ",
        )
        return any(token in normalized for token in relevant_tokens)

    @classmethod
    def _is_allowed_empty_hoe_target(
        cls,
        game_state: Optional[Dict[str, Any]],
        target_obj: Any,
        direction: str = "",
    ) -> bool:
        if not cls._is_tilling_or_digging_context(game_state) and cls._task_context_text(game_state):
            return False
        if not (cls._is_empty_like_tile(target_obj) or cls._is_open_ground_tile(target_obj)):
            return False
        relative_cell = cls._direction_to_relative(direction)
        if relative_cell is None:
            return False
        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return False
        return is_safe_empty_till_use_target(
            surroundings_map,
            relative_cell,
            current_cell=(0, 0),
        )

    @classmethod
    def _is_watering_context(cls, game_state: Optional[Dict[str, Any]]) -> bool:
        task_text = cls._task_context_text(game_state)
        if not task_text:
            return False
        normalized = f" {task_text} "
        if any(
            token in normalized
            for token in (
                " pet bowl ",
                " dog bowl ",
                " cat bowl ",
                " animal bowl ",
            )
        ):
            return False

        has_watering_intent = any(
            token in normalized
            for token in (
                " water ",
                " watered ",
                " watering ",
            )
        )
        if not has_watering_intent:
            return False

        crop_markers = (
            " crop ",
            " crops ",
            " hoedirt ",
            " hoe dirt ",
            " tilled soil ",
            " tilled tile ",
            " seed ",
            " seeds ",
            " plant ",
            " plants ",
            " growing ",
        )
        return any(token in normalized for token in crop_markers)

    @classmethod
    def _is_valid_watering_target(
        cls,
        game_state: Optional[Dict[str, Any]],
        direction: str,
        target_obj: Any,
    ) -> bool:
        text = str(target_obj or "").strip()
        if not text:
            return False
        return cls._placeable_target_has_crop(game_state, direction, target_obj)

    @classmethod
    def _find_alternative_tool_use_direction(
        cls,
        game_state: Optional[Dict[str, Any]],
        tool_name: Any,
        invalid_direction: str,
    ) -> str:
        normalized_tool = cls._normalize_tool_name(tool_name)
        if not normalized_tool:
            return ""
        if normalized_tool == "Watering Can" and not cls._is_watering_context(game_state):
            return ""

        for direction in cls._CARDINAL_DIRECTIONS:
            if direction == invalid_direction:
                continue
            target_obj, required_tool = cls._get_directional_target(game_state, direction)
            if not target_obj:
                continue
            if normalized_tool == "Watering Can":
                if cls._is_valid_watering_target(game_state, direction, target_obj):
                    return direction
                continue
            if normalized_tool == "Hoe":
                if cls._is_valid_hoe_target(target_obj) or cls._is_allowed_empty_hoe_target(
                    game_state,
                    target_obj,
                    direction,
                ):
                    return direction
                continue
            if required_tool.lower() == normalized_tool.lower():
                return direction
        return ""

    @classmethod
    def _use_suggestion_is_grounded_fallback(
        cls,
        game_state: Optional[Dict[str, Any]],
        suggested_action: Any,
        selected_tool_name: Any,
    ) -> bool:
        action_text = str(suggested_action or "").strip()
        directional_skill = cls._parse_directional_skill(action_text)
        if directional_skill is None or directional_skill[0] != "use":
            return True

        normalized_tool = cls._normalize_tool_name(selected_tool_name)
        if not normalized_tool:
            return True

        _, direction = directional_skill
        target_obj, required_tool = cls._get_directional_target(game_state, direction)
        if not target_obj:
            return False

        zero_progress_streak = 0
        repeated_action_streak = 0
        if isinstance(game_state, dict):
            zero_progress_streak = int(game_state.get("zero_progress_streak", 0) or 0)
            repeated_action_streak = int(game_state.get("repeated_action_streak", 0) or 0)

        if normalized_tool == "Watering Can":
            return cls._is_valid_watering_target(game_state, direction, target_obj)

        if normalized_tool == "Hoe":
            if cls._is_valid_hoe_target(target_obj):
                return True
            if cls._is_allowed_empty_hoe_target(game_state, target_obj, direction):
                return zero_progress_streak < 2 and repeated_action_streak < 2
            return False

        return bool(required_tool and required_tool.lower() == normalized_tool.lower())

    @classmethod
    def _should_preserve_hay_forage_use_suggestion(
        cls,
        *,
        game_state: Optional[Dict[str, Any]],
        suggested_action: Any,
        selected_item_name: Any,
    ) -> bool:
        task_text = cls._task_context_text(game_state)
        target_text = cls._target_item_text(game_state)
        source_text = cls._source_type_text(game_state)
        context_tokens = set(f"{task_text} {target_text} {source_text}".split())
        if "forage" not in context_tokens:
            return False
        if not ({"hay", "grass"} & context_tokens):
            return False
        if cls._normalize_tool_name(selected_item_name) != "Scythe":
            return False

        directional_suggestion = cls._parse_directional_skill(suggested_action)
        if directional_suggestion is None or directional_suggestion[0] != "use":
            return False

        _skill_name, direction = directional_suggestion
        target_obj, _required_tool = cls._get_directional_target(game_state, direction)
        if not target_obj:
            return True

        clearable = cls._classify_clearable_object(target_obj)
        return bool(
            clearable
            and clearable.get("tool") == "Scythe"
            and clearable.get("family") in {"hay", "weeds"}
        )

    @staticmethod
    def _target_text_contains_crop(target_obj: Any) -> bool:
        text = str(target_obj or "").strip().lower()
        if not text:
            return False

        crop_tokens = (
            "seed",
            "seeds",
            "crop",
            "growing",
            "mature",
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
        return any(token in text for token in crop_tokens)

    @classmethod
    def _target_text_is_ready_to_harvest(cls, target_obj: Any) -> bool:
        normalized = cls._normalize_context_text(target_obj)
        if not normalized:
            return False
        return "ready to harvest" in normalized or (
            "ready" in normalized and "harvest" in normalized
        )

    @staticmethod
    def _min_manhattan_distance_to_cells(
        origin: tuple[int, int],
        cells: List[tuple[int, int]],
    ) -> Optional[int]:
        if not cells:
            return None
        origin_x, origin_y = origin
        return min(abs(cell_x - origin_x) + abs(cell_y - origin_y) for cell_x, cell_y in cells)

    @classmethod
    def _nearby_ready_harvest_crop_cells(
        cls,
        game_state: Optional[Dict[str, Any]],
    ) -> List[tuple[int, int]]:
        if not isinstance(game_state, dict):
            return []

        task_text = cls._task_context_text(game_state)
        if "harvest" not in task_text:
            return []

        target_item_text = cls._target_item_text(game_state)
        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return []

        ready_cells: List[tuple[int, int]] = []
        for cell, raw_text in surroundings_map.items():
            if cell == (0, 0):
                continue
            if not cls._target_text_contains_crop(raw_text):
                continue
            if not cls._target_text_is_ready_to_harvest(raw_text):
                continue

            normalized = cls._normalize_context_text(raw_text)
            if (
                target_item_text
                and target_item_text not in {"crop", "ready", "harvest"}
                and target_item_text not in normalized
            ):
                continue
            ready_cells.append(cell)

        return ready_cells

    @classmethod
    def _build_local_harvest_recovery_action(
        cls,
        *,
        game_state: Optional[Dict[str, Any]],
    ) -> str:
        ready_cells = cls._nearby_ready_harvest_crop_cells(game_state)
        if not ready_cells:
            return ""

        adjacent = [cell for cell in ready_cells if abs(cell[0]) + abs(cell[1]) == 1]
        if adjacent:
            best = min(adjacent, key=cls._cell_sort_key)
            direction = cls._adjacent_cell_to_direction(best)
            if direction:
                return f'interact(direction="{direction}")'

        best = min(ready_cells, key=cls._cell_sort_key)
        return cls._build_step_toward_cell_move(best, game_state, max_stride=2)

    @classmethod
    def _placeable_target_has_crop(
        cls,
        game_state: Optional[Dict[str, Any]],
        direction: str,
        target_obj: Any,
    ) -> bool:
        if cls._target_text_contains_crop(target_obj):
            return True

        target_position = cls._get_target_absolute_position(game_state, direction)
        if target_position is None:
            return False

        return target_position in cls._extract_crop_positions(game_state)

    @staticmethod
    def _selected_item_is_fertilizer(item_name: Any) -> bool:
        text = str(item_name or "").strip().lower()
        if not text:
            return False
        fertilizer_tokens = (
            "fertilizer",
            "speed-gro",
            "speed gro",
            "retaining soil",
            "basic retaining soil",
            "quality retaining soil",
            "deluxe retaining soil",
        )
        return any(token in text for token in fertilizer_tokens)

    @classmethod
    def _selected_item_is_seed(cls, item_name: Any) -> bool:
        text = str(item_name or "").strip().lower()
        if not text or cls._selected_item_is_fertilizer(text):
            return False
        return any(token in text for token in ("seed", "seeds", "starter"))

    @staticmethod
    def _placeable_target_is_tilled(target_obj: Any) -> bool:
        target_lower = str(target_obj or "").strip().lower()
        return any(
            token in target_lower
            for token in ("hoedirt", "hoe dirt", "tilled dirt", "tilled soil")
        )

    @staticmethod
    def _placeable_target_is_explicitly_fertilized(target_obj: Any) -> bool:
        normalized = re.sub(r"[^a-z0-9]+", " ", str(target_obj or "").strip().lower()).strip()
        if not normalized:
            return False
        fertilizer_markers = (
            "fertilized",
            "fertilizer",
            "speed gro",
            "retaining soil",
            "basic retaining soil",
            "quality retaining soil",
            "deluxe retaining soil",
        )
        return any(marker in normalized for marker in fertilizer_markers)

    @classmethod
    def _is_valid_visible_placeable_target(
        cls,
        item_name: Any,
        target_obj: Any,
    ) -> bool:
        if not cls._placeable_target_is_tilled(target_obj):
            return False
        if cls._selected_item_is_seed(item_name):
            return not cls._target_text_contains_crop(target_obj)
        if cls._selected_item_is_fertilizer(item_name):
            return not cls._placeable_target_is_explicitly_fertilized(target_obj)
        return not cls._target_text_contains_crop(target_obj)

    @classmethod
    def _is_valid_placeable_target(
        cls,
        game_state: Optional[Dict[str, Any]],
        direction: str,
        item_name: Any,
        target_obj: Any,
    ) -> bool:
        item_lower = str(item_name or "").strip().lower()
        if not item_lower or not str(target_obj or "").strip():
            return False

        if not cls._selected_item_requires_interact(item_lower):
            return True

        if not cls._placeable_target_is_tilled(target_obj):
            return False

        if cls._selected_item_is_seed(item_lower):
            return not cls._placeable_target_has_crop(game_state, direction, target_obj)

        if cls._selected_item_is_fertilizer(item_lower):
            return not cls._placeable_target_is_explicitly_fertilized(target_obj)

        return not cls._placeable_target_has_crop(game_state, direction, target_obj)

    @classmethod
    def _find_alternative_placeable_direction(
        cls,
        game_state: Optional[Dict[str, Any]],
        item_name: Any,
        invalid_direction: str,
    ) -> str:
        alternatives = cls._collect_valid_placeable_directions(
            game_state=game_state,
            item_name=item_name,
            invalid_direction=invalid_direction,
        )
        return alternatives[0] if alternatives else ""

    @classmethod
    def _collect_valid_placeable_directions(
        cls,
        game_state: Optional[Dict[str, Any]],
        item_name: Any,
        invalid_direction: str,
    ) -> List[str]:
        alternatives: List[str] = []
        for direction in cls._CARDINAL_DIRECTIONS:
            if direction == invalid_direction:
                continue
            target_obj, _ = cls._get_directional_target(game_state, direction)
            if not target_obj:
                continue
            if cls._is_valid_placeable_target(game_state, direction, item_name, target_obj):
                alternatives.append(direction)
        return alternatives

    @staticmethod
    def _extract_inventory_slot_map(
        inventory: Any,
        toolbar_information: Any,
    ) -> Dict[int, str]:
        slot_map: Dict[int, str] = {}

        if isinstance(inventory, list):
            for item in inventory:
                match = re.search(
                    r"slot_index\s+(\d+)\s*:\s*([^()]+)",
                    str(item),
                    re.IGNORECASE,
                )
                if not match:
                    continue
                slot_map[int(match.group(1))] = match.group(2).strip()

        toolbar_text = VLLMClient._normalize_toolbar_information_text(toolbar_information)
        patterns = (
            (re.compile(r"slot_index\s+(\d+)\s*:\s*([^\n]+)", re.IGNORECASE), False),
            (re.compile(r"^\s*(\d+)\.\s*([^:]+):", re.IGNORECASE), True),
        )
        for raw_line in toolbar_text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for pattern, one_based in patterns:
                match = pattern.search(line)
                if not match:
                    continue
                raw_slot = int(match.group(1))
                slot_index = max(raw_slot - 1, 0) if one_based else raw_slot
                slot_map[slot_index] = match.group(2).strip()
                break

        return slot_map

    @staticmethod
    def _slot_is_explicitly_empty(item_name: Any) -> bool:
        text = str(item_name or "").strip().lower()
        if not text:
            return False
        return text in {"no item", "empty", "blank", "none"}

    @staticmethod
    def _extract_selected_slot_index(
        game_state: Optional[Dict[str, Any]],
        gathered: Optional[Dict[str, Any]] = None,
    ) -> Optional[int]:
        for source in (game_state, gathered):
            if not isinstance(source, dict):
                continue
            for key in ("selected_position", "slot_index", "selected_slot"):
                value = source.get(key)
                if isinstance(value, int):
                    return value
                if isinstance(value, str) and value.strip().isdigit():
                    return int(value.strip())

        toolbar_information = ""
        if isinstance(game_state, dict):
            toolbar_information = VLLMClient._normalize_toolbar_information_text(
                game_state.get("toolbar_information", "") or ""
            )
        if not toolbar_information and isinstance(gathered, dict):
            toolbar_information = VLLMClient._normalize_toolbar_information_text(
                gathered.get("toolbar_information", "") or ""
            )

        selected_match = re.search(
            r"Currently selected item:\s*slot_index\s+(\d+)\s*:",
            toolbar_information,
            re.IGNORECASE,
        )
        if selected_match:
            return int(selected_match.group(1))

        selected_match = re.search(
            r"Now the item you selected is:\s*(\d+)\s*\.",
            toolbar_information,
            re.IGNORECASE,
        )
        if selected_match:
            return max(int(selected_match.group(1)) - 1, 0)
        return None

    @staticmethod
    def _extract_selected_item_name_from_toolbar(toolbar_information: Any) -> str:
        toolbar_text = VLLMClient._normalize_toolbar_information_text(toolbar_information).strip()
        if not toolbar_text:
            return ""

        selected_match = re.search(
            r"Currently selected item:\s*slot_index\s+\d+\s*:\s*([^\n]+)",
            toolbar_text,
            re.IGNORECASE,
        )
        if selected_match:
            return selected_match.group(1).strip()

        selected_match = re.search(
            r"Now the item you selected is:\s*\d+\s*\.\s*([^\n]+)",
            toolbar_text,
            re.IGNORECASE,
        )
        if selected_match:
            return selected_match.group(1).strip()

        return ""

    @staticmethod
    def _extract_choose_item_slot(action_text: Any) -> Optional[int]:
        match = re.match(
            r"^choose_item\(\s*slot_index\s*=\s*(-?\d+)\s*\)$",
            str(action_text or "").strip(),
            re.IGNORECASE,
        )
        if not match:
            return None
        return int(match.group(1))

    @classmethod
    def _parse_menu_action(cls, action_text: Any) -> Optional[tuple[str, str]]:
        text = str(action_text or "").strip()
        named_match = re.match(
            r"^menu\(\s*option\s*=\s*[\"']?([a-zA-Z_]+)[\"']?"
            r"(?:\s*,\s*menu_name\s*=\s*[\"']?([a-zA-Z_]+)[\"']?)?\s*\)$",
            text,
            re.IGNORECASE,
        )
        positional_match = re.match(
            r"^menu\(\s*[\"']?([a-zA-Z_]+)[\"']?"
            r"(?:\s*,\s*[\"']?([a-zA-Z_]+)[\"']?)?\s*\)$",
            text,
            re.IGNORECASE,
        )
        match = named_match or positional_match
        if not match:
            return None

        option = str(match.group(1) or "").strip().lower()
        menu_name = str(match.group(2) or "").strip().lower()
        menu_aliases = {
            "backpack": "inventory",
            "bag": "inventory",
            "items": "inventory",
            "item": "inventory",
            "craft": "crafting",
        }
        option_open_aliases = {
            "open",
            "inventory",
            "current_menu",
            "craft",
            "crafting",
            "open_inventory",
            "open_crafting",
        }
        option_close_aliases = {"close", "exit"}
        if option in option_open_aliases:
            normalized_option = "open"
            normalized_menu = menu_aliases.get(menu_name, menu_name)
            if not normalized_menu:
                if option in {"craft", "crafting", "open_crafting"}:
                    normalized_menu = "crafting"
                elif option in {"inventory", "open_inventory"}:
                    normalized_menu = "inventory"
                else:
                    normalized_menu = "current_menu"
        elif option in option_close_aliases:
            normalized_option = "close"
            normalized_menu = menu_aliases.get(menu_name, menu_name) or "current_menu"
        else:
            return None

        if not normalized_menu:
            return None
        return normalized_option, normalized_menu

    @staticmethod
    def _parse_directional_skill(action_text: Any) -> Optional[tuple[str, str]]:
        match = re.match(
            r"^(use|interact)\(\s*direction\s*=\s*[\"'](up|down|left|right)[\"']\s*\)$",
            str(action_text or "").strip(),
            re.IGNORECASE,
        )
        if not match:
            return None
        return match.group(1).lower(), match.group(2).lower()

    @classmethod
    def _extract_explicit_tool_name(cls, text: Any) -> str:
        lowered = str(text or "").strip().lower()
        if not lowered:
            return ""

        direct_patterns = (
            re.compile(
                r"(?:^|[;,.]\s*|\b)"
                r"(?:need|needs|needed|require|requires|required|use|equip|select|choose|switch to|swap to|grab|take)\s+"
                r"(?:the\s+)?(watering can|pickaxe|scythe|axe|hoe)\b"
            ),
            re.compile(
                r"\b(watering can|pickaxe|scythe|axe|hoe)\b"
                r"(?=\s+(?:needed|required|to clear|to break|to chop|to cut|to till))"
            ),
        )
        for pattern in direct_patterns:
            match = pattern.search(lowered)
            if match:
                return cls._TOOL_DISPLAY_NAMES.get(match.group(1), "")

        target_markers = (
            "need",
            "needs",
            "needed",
            "require",
            "requires",
            "required",
            "use",
            "using",
            "equip",
            "equipped",
            "select",
            "selected",
            "choose",
            "chosen",
            "switch to",
            "swap to",
            "grab",
            "take",
            "with",
            "for",
        )
        current_state_markers = (
            "currently holding",
            "currently using",
            "currently selected",
            "currently equipped",
            "already holding",
            "already using",
            "already selected",
            "already equipped",
            "holding",
            "using",
            "selected",
            "equipped",
            "current tool",
        )

        best_tool = ""
        best_score = 0
        for tool_name, display_name in cls._TOOL_DISPLAY_NAMES.items():
            for match in re.finditer(rf"\b{re.escape(tool_name)}\b", lowered):
                prefix = lowered[max(0, match.start() - 40):match.start()]
                suffix = lowered[match.end():match.end() + 24]
                score = 1

                if any(marker in prefix for marker in target_markers) or any(
                    suffix.startswith(f" {marker}") for marker in ("needed", "required")
                ):
                    score += 4

                if any(marker in prefix for marker in current_state_markers) or any(
                    marker in suffix for marker in (" selected", " equipped", " in hand")
                ):
                    score -= 5

                if score > best_score:
                    best_score = score
                    best_tool = display_name

        if best_score > 0:
            return best_tool
        return ""

    @classmethod
    def _get_structured_surroundings_map(
        cls,
        game_state: Optional[Dict[str, Any]],
    ) -> tuple[Dict[tuple, str], str]:
        if not isinstance(game_state, dict):
            return {}, ""

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        for candidate in (
            gathered.get("surroundings"),
            game_state.get("surroundings"),
            gathered.get("description"),
            game_state.get("description"),
        ):
            text = str(candidate or "").strip()
            if not text:
                continue
            surroundings_map = cls._parse_surroundings_map(text)
            if surroundings_map:
                return surroundings_map, text

        return {}, ""

    @classmethod
    def _get_surroundings_summary_text(
        cls,
        game_state: Optional[Dict[str, Any]],
    ) -> str:
        surroundings_map, structured_text = cls._get_structured_surroundings_map(game_state)
        if surroundings_map and structured_text:
            return structured_text

        if not isinstance(game_state, dict):
            return ""

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        for candidate in (
            gathered.get("surroundings"),
            game_state.get("surroundings"),
            gathered.get("description"),
            game_state.get("description"),
        ):
            text = str(candidate or "").strip()
            if text:
                return text
        return ""

    @classmethod
    def _get_directional_target(
        cls,
        game_state: Optional[Dict[str, Any]],
        direction: str,
    ) -> tuple[str, str]:
        if not isinstance(game_state, dict):
            return "", ""

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return "", ""

        relative_cell = cls._direction_to_relative(direction)
        if relative_cell is None:
            return "", ""

        target_obj = str(surroundings_map.get(relative_cell, "") or "").strip()
        if not target_obj:
            return "", ""

        clearable = cls._classify_clearable_object(target_obj)
        required_tool = clearable["tool"] if clearable else ""
        return target_obj, required_tool

    @classmethod
    def _current_tile_text(
        cls,
        game_state: Optional[Dict[str, Any]],
    ) -> str:
        if not isinstance(game_state, dict):
            return ""

        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return ""
        return str(surroundings_map.get((0, 0), "") or "").strip()

    @classmethod
    def _current_tile_is_hard_structure(
        cls,
        game_state: Optional[Dict[str, Any]],
    ) -> bool:
        current_tile = cls._current_tile_text(game_state)
        if not current_tile:
            return False
        return cls._is_hard_structure_blocker(current_tile)

    @classmethod
    def _is_explicit_blocking_tile(cls, obj_text: Any) -> bool:
        text = str(obj_text or "").strip().lower()
        if not text or text in {"empty", "none", "null", "air"}:
            return False

        non_blocking_tokens = (
            "open ground",
            "ground",
            "floor",
            "path",
            "road",
            "bridge",
            "dirt",
            "tilled dirt",
            "soil",
            "hoe dirt",
            "grass starter",
        )
        if any(token in text for token in non_blocking_tokens):
            return False

        clearable = cls._classify_clearable_object(text)
        if clearable:
            return True

        blocking_tokens = (
            "wall",
            "water",
            "pond",
            "river",
            "lake",
            "fence",
            "farmhouse",
            "house",
            "barn",
            "coop",
            "shop",
            "counter",
            "door",
            "bed",
            "chest",
            "shipping bin",
            "pet bowl",
            "tree",
            "bush",
            "npc",
            "villager",
            "slime",
            "bug",
            "fly",
            "grub",
            "crab",
            "crop",
            "plant",
            "parsnip",
            "cauliflower",
            "garlic",
            "potato",
            "bean",
            "seedling",
            "animal",
        )
        return any(token in text for token in blocking_tokens)

    @classmethod
    def _get_move_axis_blockers(
        cls,
        game_state: Optional[Dict[str, Any]],
        x: int,
        y: int,
    ) -> Dict[str, str]:
        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        blockers: Dict[str, str] = {}
        if not surroundings_map:
            return blockers

        if x > 0:
            candidate = str(surroundings_map.get((1, 0), "") or "").strip()
            if cls._is_explicit_blocking_tile(candidate):
                blockers["x+"] = candidate
        elif x < 0:
            candidate = str(surroundings_map.get((-1, 0), "") or "").strip()
            if cls._is_explicit_blocking_tile(candidate):
                blockers["x-"] = candidate

        if y > 0:
            candidate = str(surroundings_map.get((0, 1), "") or "").strip()
            if cls._is_explicit_blocking_tile(candidate):
                blockers["y+"] = candidate
        elif y < 0:
            candidate = str(surroundings_map.get((0, -1), "") or "").strip()
            if cls._is_explicit_blocking_tile(candidate):
                blockers["y-"] = candidate

        return blockers

    @classmethod
    def _get_single_axis_move_blocker(
        cls,
        action_text: Any,
        game_state: Optional[Dict[str, Any]],
    ) -> str:
        direction = cls._parse_move_direction(action_text)
        move = cls._parse_move_components(action_text)
        if direction is None or move is None:
            return ""

        x, y = move
        blockers = cls._get_move_axis_blockers(game_state, x, y)
        axis_key = {
            "right": "x+",
            "left": "x-",
            "down": "y+",
            "up": "y-",
        }.get(direction, "")
        if not axis_key:
            return ""
        return blockers.get(axis_key, "")

    @classmethod
    def _get_single_axis_path_blocker(
        cls,
        action_text: Any,
        game_state: Optional[Dict[str, Any]],
    ) -> tuple[int, str]:
        move = cls._parse_move_components(action_text)
        if move is None:
            return 0, ""

        x, y = move
        if (x != 0 and y != 0) or (x == 0 and y == 0):
            return 0, ""

        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return 0, ""

        if x > 0:
            cells = [(step, 0) for step in range(1, abs(x) + 1)]
        elif x < 0:
            cells = [(-step, 0) for step in range(1, abs(x) + 1)]
        elif y > 0:
            cells = [(0, step) for step in range(1, abs(y) + 1)]
        else:
            cells = [(0, -step) for step in range(1, abs(y) + 1)]

        for index, cell in enumerate(cells, start=1):
            candidate = str(surroundings_map.get(cell, "") or "").strip()
            if cls._is_explicit_blocking_tile(candidate):
                return index, candidate

        return 0, ""

    @classmethod
    def _should_keep_same_axis_progress_move(
        cls,
        action_text: Any,
        game_state: Optional[Dict[str, Any]],
        blocker: Any,
    ) -> bool:
        if not isinstance(game_state, dict):
            return False

        blocker_text = str(blocker or "").strip().lower()
        if not any(
            token in blocker_text
            for token in ("farmhouse", "house", "barn", "coop", "shed", "silo", "wall")
        ):
            return False

        move = cls._parse_move_components(action_text)
        direction = cls._parse_move_direction(action_text)
        if move is None or not direction:
            return False

        x, y = move
        if (x == 0 and y == 0) or (x != 0 and y != 0):
            return False

        stride = abs(x) + abs(y)
        if stride < 2:
            return False

        task_text = cls._task_context_text(game_state)
        if not any(
            token in task_text
            for token in ("till", "sow", "fertiliz", "water", "cultivat", "seed", "crop", "hoe")
        ):
            return False

        relative = cls._direction_to_relative(direction)
        if relative is None:
            return False

        next_progress_cell = (relative[0] * 2, relative[1] * 2)
        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return False
        if cls._is_hard_structure_blocker(str(surroundings_map.get((0, 0), "") or "").strip()):
            return False
        if cls._count_nearby_hard_structures(surroundings_map, (0, 0), radius=1) > 0:
            return False

        next_text = str(surroundings_map.get(next_progress_cell, "") or "").strip()
        if not next_text:
            return False

        return cls._is_open_ground_tile(next_text) or cls._placeable_target_is_tilled(next_text)

    @staticmethod
    def _is_open_ground_tile(obj_text: Any) -> bool:
        return grounding_is_open_ground_tile(obj_text)

    @staticmethod
    def _is_hard_structure_blocker(obj_text: Any) -> bool:
        return is_hard_structure_text(obj_text)

    @staticmethod
    def _is_actionable_front_tile(obj_text: Any) -> bool:
        text = str(obj_text or "").strip().lower()
        if not text:
            return False
        return any(
            token in text
            for token in (
                "door",
                "shipping bin",
                "counter",
                "bed",
                "stairs",
                "staircase",
                "ladder",
                "elevator",
                "pet bowl",
                "feeding bench",
                "incubator",
                "gate",
            )
        )

    @classmethod
    def _build_adjacent_blocker_interact(
        cls,
        *,
        action_text: str,
        game_state: Optional[Dict[str, Any]],
        blocker: Any,
    ) -> str:
        if not isinstance(game_state, dict):
            return ""
        move_direction = cls._parse_move_direction(action_text)
        if not move_direction:
            return ""
        relative_cell = cls._direction_to_relative(move_direction)
        if relative_cell is None:
            return ""
        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return ""

        blocker_text = str(blocker or "").strip().lower()
        front_text = str(surroundings_map.get(relative_cell, "") or "").strip().lower()
        if not blocker_text or not front_text:
            return ""
        if (
            blocker_text != front_text
            and blocker_text not in front_text
            and front_text not in blocker_text
        ):
            return ""
        if (
            cls._is_door_or_entrance_text(front_text)
            and cls._cultivation_outdoor_task_should_stay_outside(game_state)
        ):
            return ""
        if (
            cls._is_door_or_entrance_text(front_text)
            and "farmhouse" in front_text
            and "farmhouse" not in cls._location_text(game_state)
            and not cls._is_sleep_task_context(game_state)
            and not any(
                token in cls._task_context_text(game_state)
                for token in ("go_home", "go home", "return home", "enter house", "farmhouse")
            )
        ):
            return ""
        if cls._should_block_adjacent_door_interact_rewrite(
            game_state=game_state,
            front_text=front_text,
        ):
            return ""
        if not cls._is_actionable_front_tile(front_text):
            return ""
        return f'interact(direction="{move_direction}")'

    @classmethod
    def _should_block_adjacent_door_interact_rewrite(
        cls,
        *,
        game_state: Optional[Dict[str, Any]],
        front_text: str,
    ) -> bool:
        if not cls._is_door_or_entrance_text(front_text):
            return False
        task_text = cls._task_context_text(game_state)
        if not task_text:
            return False
        if cls._is_sleep_task_context(game_state):
            return cls._is_inside_farmhouse_sleep_context(game_state)
        return any(
            token in task_text
            for token in (
                "go to bus stop",
                "go_to_bus_stop",
                "forage ",
                "forage_",
                "ship ",
                "ship_",
            )
        )

    @classmethod
    def _build_structure_blocked_move_recovery(
        cls,
        *,
        action_text: str,
        game_state: Optional[Dict[str, Any]],
        blocker: Any,
    ) -> str:
        blocker_text = str(blocker or "").strip()
        if not blocker_text:
            return ""
        if not (
            cls._is_hard_structure_blocker(blocker_text)
            or cls._classify_clearable_object(blocker_text)
        ):
            return ""
        if not isinstance(game_state, dict):
            return ""

        move = cls._parse_move_components(action_text)
        if move is None:
            return ""

        x, y = move
        suggested_direction = cls._parse_move_direction(action_text)
        if cls._should_preserve_navigation_anchor_move(
            action_text=action_text,
            game_state=game_state,
            blocker=blocker_text,
        ):
            return ""
        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return ""

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        selected_item_name = (
            cls._extract_selected_item_name(gathered)
            or cls._extract_selected_item_name_from_toolbar(
                game_state.get("toolbar_information") or gathered.get("toolbar_information") or ""
            )
        ).strip().lower()
        refused_move_text = cls._normalize_action_text(
            execution_refused_action(game_state.get("last_exec_info", {}))
        )
        ready_harvest_cells = cls._nearby_ready_harvest_crop_cells(game_state)
        current_ready_crop_distance = cls._min_manhattan_distance_to_cells((0, 0), ready_harvest_cells)

        visible_target_cells: List[tuple[int, int, int, int, str]] = []
        open_ground_cells: List[tuple[int, int, int, int, str]] = []
        for (cell_x, cell_y), cell_text in surroundings_map.items():
            text = str(cell_text or "").strip()
            if not text or (cell_x == 0 and cell_y == 0):
                continue

            clearable = cls._classify_clearable_object(text)
            if clearable and (
                not selected_item_name or clearable["tool"].lower() == selected_item_name
            ):
                visible_target_cells.append(
                    (abs(cell_x) + abs(cell_y), abs(cell_x), cell_x, cell_y, text)
                )
                continue

            if cls._selected_item_requires_interact(selected_item_name) and cls._is_valid_visible_placeable_target(
                selected_item_name,
                text,
            ):
                visible_target_cells.append(
                    (abs(cell_x) + abs(cell_y), abs(cell_x), cell_x, cell_y, text)
                )
                continue

            if cls._is_open_ground_tile(text):
                open_ground_cells.append(
                    (abs(cell_x) + abs(cell_y), abs(cell_x), cell_x, cell_y, text)
                )

        def _select_candidate(cells: List[tuple[int, int, int, int, str]]) -> str:
            if not cells:
                return ""
            ranked: List[tuple[tuple[int, int, int, int, int, int, int, int], tuple[int, int]]] = []
            for _, _, cell_x, cell_y, _ in cells:
                # Avoid another same-axis move directly through the blocker.
                if suggested_direction in {"up", "down"} and cell_x == 0:
                    continue
                if suggested_direction in {"left", "right"} and cell_y == 0:
                    continue

                ready_crop_distance = 0
                if current_ready_crop_distance is not None:
                    candidate_ready_crop_distance = cls._min_manhattan_distance_to_cells(
                        (cell_x, cell_y),
                        ready_harvest_cells,
                    )
                    if candidate_ready_crop_distance is None:
                        continue
                    # When a harvest task already has visible ready crops nearby,
                    # never reroute farther away from that patch just to escape a
                    # structure tile. Force a replan instead of drifting off-patch.
                    if candidate_ready_crop_distance > current_ready_crop_distance:
                        continue
                    ready_crop_distance = candidate_ready_crop_distance

                preserves_blocked_axis_progress = 1
                if suggested_direction == "right":
                    preserves_blocked_axis_progress = 0 if cell_x > 0 else 1
                elif suggested_direction == "left":
                    preserves_blocked_axis_progress = 0 if cell_x < 0 else 1
                elif suggested_direction == "down":
                    preserves_blocked_axis_progress = 0 if cell_y > 0 else 1
                elif suggested_direction == "up":
                    preserves_blocked_axis_progress = 0 if cell_y < 0 else 1

                nearby_structures = cls._count_nearby_hard_structures(
                    surroundings_map,
                    (cell_x, cell_y),
                )
                nearby_open_ground = cls._count_nearby_open_ground_tiles(
                    surroundings_map,
                    (cell_x, cell_y),
                )
                intended_axis_progress = 0
                perpendicular_escape = 0
                if suggested_direction in {"left", "right"}:
                    intended_axis_progress = -abs(cell_x)
                    perpendicular_escape = -abs(cell_y)
                elif suggested_direction in {"up", "down"}:
                    intended_axis_progress = -abs(cell_y)
                    perpendicular_escape = -abs(cell_x)

                ranked.append(
                    (
                        (
                            ready_crop_distance,
                            preserves_blocked_axis_progress,
                            nearby_structures,
                            -nearby_open_ground,
                            cell_x == 0 and cell_y == 0,
                            abs(cell_x) + abs(cell_y),
                            perpendicular_escape,
                            intended_axis_progress,
                            abs(cell_x) + abs(cell_y),
                        ),
                        (cell_x, cell_y),
                    )
                )

            ranked.sort(key=lambda item: item[0])
            for _, (cell_x, cell_y) in ranked:
                candidate_action = f"move(x={cell_x}, y={cell_y})"
                if refused_move_text and cls._normalize_action_text(candidate_action) == refused_move_text:
                    continue
                return candidate_action
            return ""

        candidate = _select_candidate(open_ground_cells)
        if candidate:
            return candidate
        candidate = _select_candidate(visible_target_cells)
        if candidate:
            return candidate

        # Progressive offset fallback: when surroundings show no open ground
        # (e.g., player is surrounded by building tiles), generate moves that
        # offset perpendicular to the blocked direction with increasing distance.
        # This helps escape large buildings where a ±1 offset is insufficient.
        step_x = 0 if x == 0 else (1 if x > 0 else -1)
        step_y = 0 if y == 0 else (1 if y > 0 else -1)
        fallback_step_x = step_x if step_x != 0 else 1
        fallback_step_y = step_y if step_y != 0 else 1
        if suggested_direction in {"up", "down"}:
            # Blocked vertically → try horizontal offsets of increasing size
            offsets = [(-2, step_y), (2, step_y), (-3, step_y), (3, step_y), (-4, 0), (4, 0)]
        elif suggested_direction in {"left", "right"}:
            # Blocked horizontally → try vertical offsets of increasing size
            offsets = [(step_x, -2), (step_x, 2), (step_x, -3), (step_x, 3), (0, -4), (0, 4)]
        else:
            # Diagonal or unknown → try perpendicular and larger offsets
            offsets = [
                (-2, fallback_step_y),
                (2, fallback_step_y),
                (fallback_step_x, -2),
                (fallback_step_x, 2),
                (-3, -3),
                (3, 3),
            ]

        for ox, oy in offsets:
            if ox == 0 and oy == 0:
                continue
            cell_key = (ox, oy)
            cell_text = str(surroundings_map.get(cell_key, "") or "").strip().lower()
            if current_ready_crop_distance is not None:
                candidate_ready_crop_distance = cls._min_manhattan_distance_to_cells(
                    (ox, oy),
                    ready_harvest_cells,
                )
                if (
                    candidate_ready_crop_distance is None
                    or candidate_ready_crop_distance > current_ready_crop_distance
                ):
                    continue
            # Skip if this tile is a known hard blocker
            if cell_text and any(
                kw in cell_text for kw in ("farmhouse", "barn", "coop", "water", "fence", "wall")
            ):
                continue
            # Accept: empty, unknown, or non-blocker tile
            candidate_action = f"move(x={ox}, y={oy})"
            if refused_move_text and cls._normalize_action_text(candidate_action) == refused_move_text:
                continue
            logger.write(
                f"[FastLLM] Progressive offset recovery: {candidate_action} "
                f"(original blocked direction: {suggested_direction})"
            )
            return candidate_action

        return ""

    @staticmethod
    def _is_same_direction_shorter_stride_override(
        suggested_move: tuple[int, int],
        override_move: tuple[int, int],
    ) -> bool:
        sx, sy = suggested_move
        ax, ay = override_move

        if (sx, sy) == (ax, ay):
            return False
        if (sx, sy) == (0, 0) or (ax, ay) == (0, 0):
            return False

        if sx != 0 and sy == 0 and ax != 0 and ay == 0:
            return (sx > 0) == (ax > 0) and abs(ax) < abs(sx)

        if sy != 0 and sx == 0 and ay != 0 and ax == 0:
            return (sy > 0) == (ay > 0) and abs(ay) < abs(sy)

        if sx != 0 and sy != 0 and ax != 0 and ay != 0:
            same_quadrant = (sx > 0) == (ax > 0) and (sy > 0) == (ay > 0)
            if not same_quadrant:
                return False
            return (
                abs(ax) <= abs(sx)
                and abs(ay) <= abs(sy)
                and (abs(ax) < abs(sx) or abs(ay) < abs(sy))
            )

        return False

    @classmethod
    def _move_override_is_justified(
        cls,
        suggested_action: str,
        action: str,
        game_state: Optional[Dict[str, Any]],
    ) -> tuple[bool, str]:
        suggested_move = cls._parse_move_components(suggested_action)
        override_move = cls._parse_move_components(action)
        if suggested_move is None or override_move is None:
            return True, "not_move_pair"

        sx, sy = suggested_move
        ax, ay = override_move
        if (sx, sy) == (ax, ay):
            return True, "same_move"

        surroundings_map = {}
        if isinstance(game_state, dict):
            gathered = game_state.get("gathered_info", {})
            if not isinstance(gathered, dict):
                gathered = {}
            surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
            current_menu = gathered.get("current_menu") or gathered.get("CurrentMenuData") or game_state.get("current_menu")
            if cls._is_menu_open(current_menu):
                return True, "menu_contradiction"

            inventory = gathered.get("inventory", [])
            toolbar_information = game_state.get("toolbar_information") or gathered.get("toolbar_information") or ""
            selected_item_name = (
                cls._extract_selected_item_name(gathered)
                or cls._extract_selected_item_name_from_toolbar(toolbar_information)
            )
            grounded_local_recovery = cls._build_invalidated_suggestion_local_recovery(
                game_state=game_state,
                suggestion_action=suggested_action,
                selected_item_name=selected_item_name,
                inventory=inventory,
                toolbar_information=toolbar_information,
            )
            if grounded_local_recovery and grounded_local_recovery == action:
                return True, "grounded_local_recovery_override"

            route_recovery = cls._build_local_route_recovery_action(game_state=game_state)
            if route_recovery and route_recovery == action:
                return True, "route_recovery_override"

            zero_progress_streak = int(game_state.get("zero_progress_streak", 0) or 0)
            repeated_action_streak = int(game_state.get("repeated_action_streak", 0) or 0)
            position_issue_detected = bool(game_state.get("position_issue_detected", False))
            unstable_execution = (
                zero_progress_streak >= 1
                or repeated_action_streak >= 2
                or position_issue_detected
            )
            task_text = cls._task_context_text(game_state)
            combat_context = any(
                token in task_text
                for token in (
                    "kill ",
                    " kill ",
                    " combat ",
                    " slime ",
                    " bug ",
                    " fly ",
                    " duggy ",
                    " grub ",
                    " rock crab ",
                    " enemy ",
                    " monster ",
                )
            )
            if cls._is_same_direction_shorter_stride_override(
                suggested_move,
                override_move,
            ) and (unstable_execution or combat_context):
                if unstable_execution:
                    return True, "short_stride_stability_override"
                return True, "short_stride_combat_override"

            best_waypoint = cls._best_route_waypoint_candidate(game_state)
            if (
                best_waypoint
                and cls._move_reduces_waypoint_distance(action, best_waypoint)
                and cls._move_conflicts_with_waypoint(suggested_action, best_waypoint)
            ):
                return True, f"grounded_waypoint_override:{best_waypoint.get('name', '')}"

            source_type = str(game_state.get("source_type", "") or "").strip().lower()
            task_text = cls._task_context_text(game_state)
            if source_type in {"animal_housing", "pet_routine"} or "pet_3_animal" in task_text:
                stride = abs(ax) + abs(ay)
                blockers = cls._get_move_axis_blockers(game_state, sx, sy)
                if sx != 0 and sy == 0:
                    expected_key = "x+" if sx > 0 else "x-"
                    if (
                        expected_key in blockers
                        and stride <= 2
                        and not (ay == 0 and ax != 0 and (ax > 0) == (sx > 0))
                    ):
                        return True, f"animal_housing_short_search:{blockers[expected_key]}"
                if sy != 0 and sx == 0:
                    expected_key = "y+" if sy > 0 else "y-"
                    if (
                        expected_key in blockers
                        and stride <= 2
                        and not (ax == 0 and ay != 0 and (ay > 0) == (sy > 0))
                    ):
                        return True, f"animal_housing_short_search:{blockers[expected_key]}"

            action_move = cls._parse_move_components(action)
            action_direction = cls._parse_move_direction(action)
            if action_move is not None and action_direction:
                stride = abs(action_move[0]) + abs(action_move[1])
                task_text = cls._task_context_text(game_state)
                if stride <= 2 and any(
                    token in task_text
                    for token in (
                        "go to",
                        "go_to",
                        "route to",
                        "follow the map exits",
                        "return home",
                        "enter",
                        "exit",
                        "sleep",
                        "purchase",
                        "sell",
                        "talk",
                        "upgrade",
                        "break geode",
                    )
                ):
                    target_obj, _ = cls._get_directional_target(game_state, action_direction)
                    target_text = str(target_obj or "").strip().lower()
                    if any(
                        token in target_text
                        for token in (
                            "door",
                            "entrance",
                            "exit",
                            "stairs",
                            "staircase",
                            "ladder",
                            "elevator",
                        )
                    ):
                        return True, f"navigation_anchor:{target_text}"

        blockers = cls._get_move_axis_blockers(game_state, sx, sy)

        # BigBrain suggested a single-axis move.
        if sx != 0 and sy == 0:
            expected_key = "x+" if sx > 0 else "x-"
            if expected_key in blockers and ax == 0 and ay != 0:
                return True, f"x_axis_blocked:{blockers[expected_key]}"
            if (
                expected_key in blockers
                and ax != 0
                and ay != 0
                and (ax > 0) == (sx > 0)
                and abs(ax) <= max(abs(sx), 3)
                and abs(ay) <= 3
            ):
                return True, f"x_axis_blocked_diagonal_bypass:{blockers[expected_key]}"
            path_blocker_distance, path_blocker = cls._get_single_axis_path_blocker(
                suggested_action,
                game_state,
            )
            if (
                path_blocker_distance > 0
                and ay == 0
                and ax != 0
                and (ax > 0) == (sx > 0)
                and abs(ax) < path_blocker_distance
            ):
                return True, f"x_path_blocked:{path_blocker}"
            override_target_text = str(surroundings_map.get((ax, ay), "") or "").strip()
            if (
                expected_key in blockers
                and cls._is_hard_structure_blocker(blockers[expected_key])
                and override_target_text
                and cls._is_open_ground_tile(override_target_text)
                and abs(ax) + abs(ay) <= 3
            ):
                return True, f"x_structure_escape:{blockers[expected_key]}"
            return False, "single_axis_move_mismatch"

        if sy != 0 and sx == 0:
            expected_key = "y+" if sy > 0 else "y-"
            if expected_key in blockers and ay == 0 and ax != 0:
                return True, f"y_axis_blocked:{blockers[expected_key]}"
            if (
                expected_key in blockers
                and ay != 0
                and ax != 0
                and (ay > 0) == (sy > 0)
                and abs(ay) <= max(abs(sy), 3)
                and abs(ax) <= 3
            ):
                return True, f"y_axis_blocked_diagonal_bypass:{blockers[expected_key]}"
            path_blocker_distance, path_blocker = cls._get_single_axis_path_blocker(
                suggested_action,
                game_state,
            )
            if (
                path_blocker_distance > 0
                and ax == 0
                and ay != 0
                and (ay > 0) == (sy > 0)
                and abs(ay) < path_blocker_distance
            ):
                return True, f"y_path_blocked:{path_blocker}"
            override_target_text = str(surroundings_map.get((ax, ay), "") or "").strip()
            if (
                expected_key in blockers
                and cls._is_hard_structure_blocker(blockers[expected_key])
                and override_target_text
                and cls._is_open_ground_tile(override_target_text)
                and abs(ax) + abs(ay) <= 3
            ):
                return True, f"y_structure_escape:{blockers[expected_key]}"
            return False, "single_axis_move_mismatch"

        # BigBrain suggested a diagonal move. Allow splitting only when the removed axis is visibly blocked.
        if sx != 0 and sy != 0:
            x_key = "x+" if sx > 0 else "x-"
            y_key = "y+" if sy > 0 else "y-"
            if ax == 0 and ay == sy and x_key in blockers:
                return True, f"diagonal_split_x_blocked:{blockers[x_key]}"
            if ay == 0 and ax == sx and y_key in blockers:
                return True, f"diagonal_split_y_blocked:{blockers[y_key]}"
            suggested_target_text = str(surroundings_map.get((sx, sy), "") or "").strip()
            override_target_text = str(surroundings_map.get((ax, ay), "") or "").strip()
            if (
                suggested_target_text
                and cls._is_hard_structure_blocker(suggested_target_text)
                and override_target_text
                and cls._is_open_ground_tile(override_target_text)
                and abs(ax) + abs(ay) <= 2
            ):
                return True, f"diagonal_target_blocked:{suggested_target_text}"
            return False, "diagonal_override_without_blocker"

        return False, "move_override_unjustified"

    @classmethod
    def _should_preserve_navigation_anchor_move(
        cls,
        action_text: str,
        game_state: Optional[Dict[str, Any]],
        blocker: Any,
    ) -> bool:
        if not isinstance(game_state, dict):
            return False

        blocker_text = str(blocker or "").strip().lower()

        move = cls._parse_move_components(action_text)
        direction = cls._parse_move_direction(action_text)
        if move is None or not direction:
            return False

        stride = abs(move[0]) + abs(move[1])
        if stride > 1:
            return False

        task_text = cls._task_context_text(game_state)
        if not any(
            token in task_text
            for token in (
                "go to",
                "go_to",
                "route to",
                "follow the map exits",
                "return home",
                "enter",
                "exit",
                "sleep",
                "purchase",
                "sell",
                "talk",
                "upgrade",
                "break geode",
            )
        ):
            return False

        target_obj, _ = cls._get_directional_target(game_state, direction)
        target_text = str(target_obj or "").strip().lower()
        if not any(token in (blocker_text or target_text) for token in ("door", "entrance", "exit")) and not any(
            token in target_text for token in ("door", "entrance", "exit")
        ):
            return False
        return any(token in target_text for token in ("door", "entrance", "exit"))

    @classmethod
    def _composite_override_is_justified(
        cls,
        suggested_action: str,
        action: str,
        game_state: Optional[Dict[str, Any]],
    ) -> tuple[bool, str]:
        suggested_name = cls._parse_skill_name(suggested_action)
        if suggested_name not in cls._TERMINAL_COMPOSITE_SKILLS:
            return True, "not_terminal_composite"
        if action == suggested_action:
            return True, "same_composite"

        action_name = cls._parse_skill_name(action)
        if not isinstance(game_state, dict):
            return False, "terminal_composite_follow_default"

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        current_menu = gathered.get("current_menu") or gathered.get("CurrentMenuData") or game_state.get("current_menu")
        if cls._is_menu_open(current_menu) and action_name in {"choose_option", "menu"}:
            return True, "menu_override"

        location_text = re.sub(
            r"[^a-z0-9]+",
            " ",
            str(gathered.get("location") or game_state.get("location") or "").lower(),
        ).strip()

        if suggested_name == "go_home" and any(token in location_text for token in ("farmhouse", "house", "home")):
            return True, "already_home"
        if suggested_name == "go_to_store" and any(token in location_text for token in ("seedshop", "pierre", "store")):
            return True, "already_at_store"
        if suggested_name == "get_out_of_house" and not any(token in location_text for token in ("farmhouse", "house", "home")):
            return True, "already_outside"
        if suggested_name == "buy_item" and action_name in {"choose_option", "menu"} and cls._is_menu_open(current_menu):
            return True, "shop_menu_override"

        return False, "terminal_composite_follow_default"

    @classmethod
    def _build_immediate_use_if_aligned(cls, game_state: Optional[Dict[str, Any]]) -> str:
        if not isinstance(game_state, dict):
            return ""

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        facing_direction = str(gathered.get("facing_direction", "") or "").strip().lower()
        if not facing_direction:
            return ""

        target_obj, required_tool = cls._get_directional_target(game_state, facing_direction)
        if not target_obj or not required_tool:
            return ""

        selected_item_name = cls._extract_selected_item_name(gathered).lower()
        if selected_item_name != required_tool.lower():
            return ""

        return f'use(direction="{facing_direction}")'

    @classmethod
    def _selected_item_requires_interact(cls, item_name: Any) -> bool:
        text = str(item_name or "").strip().lower()
        if not text:
            return False

        placeable_tokens = (
            "seed",
            "seeds",
            "fertilizer",
            "speed-gro",
            "speed gro",
            "retaining soil",
            "basic retaining soil",
            "quality retaining soil",
            "deluxe retaining soil",
            "soil",
        )
        return any(token in text for token in placeable_tokens)

    @staticmethod
    def _parse_craft_item_name(action_text: Any) -> str:
        match = re.match(
            r'^craft\(\s*item\s*=\s*["\']?([^"\')]+)["\']?\s*\)$',
            str(action_text or "").strip(),
            re.IGNORECASE,
        )
        if not match:
            return ""
        return str(match.group(1) or "").strip()

    @staticmethod
    def _craft_missing_materials(item_name: str, inventory: Any) -> Optional[List[str]]:
        if not item_name:
            return None
        try:
            from stardojo.utils.cortex_runtime_utils import (
                _canonical_crafting_recipe_name,
                _crafting_recipe_missing_materials,
                _load_crafting_recipe_table,
            )
        except Exception:
            return None

        item_name = _canonical_crafting_recipe_name(item_name)
        recipe_table = _load_crafting_recipe_table()
        if isinstance(recipe_table, dict) and item_name not in recipe_table:
            return ["unknown_recipe"]

        missing = _crafting_recipe_missing_materials(item_name, inventory)
        if missing is None:
            return []
        return list(missing)

    @staticmethod
    def _normalize_context_text(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()

    @classmethod
    def _task_context_text(cls, game_state: Optional[Dict[str, Any]]) -> str:
        if not isinstance(game_state, dict):
            return ""
        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}
        raw = " ".join(
            str(candidate or "")
            for candidate in (
                game_state.get("task"),
                game_state.get("main_task"),
                game_state.get("task_description"),
                game_state.get("subtask_description"),
                gathered.get("task_description"),
                gathered.get("subtask_description"),
            )
        )
        return cls._normalize_context_text(raw)

    @classmethod
    def _source_type_text(cls, game_state: Optional[Dict[str, Any]]) -> str:
        if not isinstance(game_state, dict):
            return ""
        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}
        return cls._normalize_context_text(
            game_state.get("source_type")
            or gathered.get("source_type")
            or ""
        )

    @classmethod
    def _target_item_text(cls, game_state: Optional[Dict[str, Any]]) -> str:
        if not isinstance(game_state, dict):
            return ""
        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}
        return cls._normalize_context_text(
            game_state.get("target_item")
            or gathered.get("target_item")
            or ""
        )

    @classmethod
    def _location_text(cls, game_state: Optional[Dict[str, Any]]) -> str:
        if not isinstance(game_state, dict):
            return ""
        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}
        return cls._normalize_context_text(
            gathered.get("location")
            or game_state.get("location")
            or ""
        )

    @classmethod
    def _is_cultivation_context(cls, game_state: Optional[Dict[str, Any]]) -> bool:
        task_text = cls._task_context_text(game_state)
        if not task_text:
            return False
        return any(
            token in task_text
            for token in (
                " till ",
                " hoe ",
                " fertiliz ",
                " sow ",
                " seed ",
                " plant ",
                " water ",
                " crop ",
                " cultivate ",
                " harvest ",
            )
        )

    @classmethod
    def _is_inside_house_for_cultivation(
        cls,
        game_state: Optional[Dict[str, Any]],
    ) -> bool:
        location_text = cls._location_text(game_state)
        if not location_text:
            return False
        if "greenhouse" in location_text:
            return False
        if "farmhouse" in location_text:
            return True
        if "house" in location_text or "cabin" in location_text:
            return "inside" in location_text or "farm" not in location_text
        return False

    @staticmethod
    def _is_door_or_entrance_text(obj_text: Any) -> bool:
        text = str(obj_text or "").strip().lower()
        if not text:
            return False
        return any(token in text for token in ("door", "entrance", "exit"))

    @classmethod
    def _cultivation_outdoor_task_should_stay_outside(
        cls,
        game_state: Optional[Dict[str, Any]],
    ) -> bool:
        if not cls._is_cultivation_context(game_state):
            return False
        if cls._is_inside_house_for_cultivation(game_state):
            return False

        task_text = cls._task_context_text(game_state)
        source_text = cls._source_type_text(game_state)
        if any(
            token in task_text
            for token in (
                " sleep ",
                " bed ",
                " return home ",
                " go home ",
                " farmhouse entrance ",
                " enter farmhouse ",
                " enter the farmhouse ",
                " inside farmhouse ",
                " exit farmhouse ",
                " leave farmhouse ",
                " leave the farmhouse ",
                " return to the farmhouse ",
            )
        ):
            return False
        if any(token in source_text for token in ("sleep", "bed", "home")):
            return False
        return True

    @staticmethod
    def _extract_relative_offset_from_text(text: Any) -> Optional[tuple[int, int]]:
        match = re.search(
            r"relative offset:\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)",
            str(text or ""),
            re.IGNORECASE,
        )
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))

    @classmethod
    def _iter_named_relative_targets(cls, raw_value: Any) -> List[Dict[str, Any]]:
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

    @classmethod
    def _route_target_preferences(cls, game_state: Optional[Dict[str, Any]]) -> List[tuple[str, ...]]:
        task_text = cls._task_context_text(game_state)
        source_type = cls._source_type_text(game_state)
        target_item = cls._target_item_text(game_state)
        location_text = cls._location_text(game_state)
        source_detail = cls._normalize_context_text(
            game_state.get("source_detail", "") if isinstance(game_state, dict) else ""
        )

        preferences: List[tuple[str, ...]] = []

        def _add(*tokens: str) -> None:
            normalized = tuple(
                token
                for token in (
                    cls._normalize_context_text(value)
                    for value in tokens
                )
                if token
            )
            if normalized and normalized not in preferences:
                preferences.append(normalized)

        if "pet bowl" in task_text or source_type == "pet_area":
            if any(token in location_text for token in ("farmhouse", "house", "home")):
                _add("door", "entrance", "exit")
            _add("pet bowl")
            _add("farmhouse", "house")
            return preferences

        if any(
            token in task_text
            for token in (
                "go_to_bed",
                "go to bed",
                " sleep ",
            )
        ):
            if any(token in location_text for token in ("farmhouse", "home")):
                _add("bed")
                return preferences
            if any(
                token in location_text
                for token in ("coop", "barn", "shed", "cabin", "animalshop", "sciencehouse")
            ):
                _add("door", "entrance", "exit")
                return preferences
            _add("farmhouse", "house", "home")
            _add("door", "entrance")
            return preferences

        if any(
            token in task_text
            for token in (
                "go_home",
                "go home",
                "return home",
                "enter house",
                "enter the house",
                "home entrance",
            )
        ):
            _add("farmhouse", "house")
            _add("door", "entrance")
            return preferences

        if (
            cls._is_inside_house_for_cultivation(game_state)
            and any(
                token in task_text
                for token in (
                    " till ",
                    " sow ",
                    " fertilize ",
                    " water ",
                    " cultivate ",
                    " seed ",
                    " crop ",
                    " hoe ",
                )
            )
        ):
            _add("door", "entrance", "exit")
            _add("farmhouse", "house")
            return preferences

        if "animal door" in task_text or source_type == "animal_door":
            if "barn" in task_text or "barn" in target_item or "barn" in source_detail:
                _add("barn")
            else:
                _add("coop")
            _add("animal door", "hatch", "door")
            return preferences

        if "incubat" in task_text:
            _add("incubator")
            _add("coop")
            return preferences

        if "pet " in task_text or source_type == "pet_routine":
            _add("cat", "dog", "pet")
            _add("farmhouse", "house")
            return preferences

        if "milk" in task_text or "milk pail" in task_text or source_type == "animal_tool":
            _add("cow", "goat", "sheep", "barn")
            return preferences

        if "egg" in task_text:
            _add("egg")
            _add("coop")
            return preferences

        if "feed" in task_text or "feeding bench" in task_text:
            _add("feeding bench", "hopper")
            _add("coop", "barn")
            return preferences

        if "animal" in task_text or source_type == "animal_housing":
            _add("animal", "chicken", "cow", "goat", "duck", "rabbit")
            _add("coop")
            _add("barn")
            return preferences

        if (
            source_type in {"blacksmith"}
            or any(
                token in target_item or token in task_text or token in source_detail
                for token in (
                    "blacksmith",
                    "clint",
                    "geode",
                    "copper pickaxe",
                    "steel pickaxe",
                    "gold pickaxe",
                    "iridium pickaxe",
                    "tool upgrade",
                )
            )
        ):
            if "farm" in location_text and "bus stop" not in location_text and "town" not in location_text:
                _add("bus stop")
            elif "bus stop" in location_text:
                _add("town")
            else:
                _add("blacksmith", "clint")
            return preferences

        if (
            source_type in {"carpenter"}
            or any(
                token in target_item or token in task_text or token in source_detail
                for token in (
                    "carpenter",
                    "robin",
                    "farmhouse upgrade",
                    "shipping bin",
                    "demolish",
                    "large pack",
                    "big coop",
                    "big barn",
                    "deluxe coop",
                    "deluxe barn",
                    "build 1",
                    "move 1 coop",
                    "move coop",
                )
            )
        ):
            if (
                "farm" in location_text
                and "backwoods" not in location_text
                and "mountain" not in location_text
                and "carpenter" not in location_text
            ):
                _add("backwoods")
            elif "bus stop" in location_text:
                _add("backwoods")
            else:
                _add("carpenter", "robin")
            return preferences

        if (
            "joja" in source_type
            or any(
                token in target_item or token in task_text or token in source_detail
                for token in ("joja", "membership", "minecarts development project")
            )
        ):
            if "farm" in location_text and "bus stop" not in location_text and "town" not in location_text:
                _add("bus stop")
            elif "bus stop" in location_text:
                _add("town")
            else:
                _add("joja", "jojamart")
            return preferences

        if any(
            token in target_item or token in task_text or token in source_detail
            for token in ("backpack", "large pack")
        ):
            if "farm" in location_text and "bus stop" not in location_text and "town" not in location_text:
                _add("bus stop")
            elif "bus stop" in location_text:
                _add("town")
            else:
                _add("pierre", "general store", "seed shop", "store")
            return preferences

        if (
            source_type == "enemy_search"
            or any(
                token in task_text
                for token in (
                    "green slime",
                    "slime",
                    "bug",
                    "fly",
                    "duggy",
                    "grub",
                    "rock crab",
                    "enemy",
                    "monster",
                )
            )
        ):
            if "green slime" in task_text:
                _add("green slime", "slime")
            if "rock crab" in task_text:
                _add("rock crab", "crab")
            if "bug" in task_text:
                _add("bug")
            if "fly" in task_text:
                _add("fly")
            if "duggy" in task_text:
                _add("duggy")
            if "grub" in task_text:
                _add("grub")
            _add("enemy", "monster", "slime", "bug", "fly", "duggy", "grub", "rock crab")
            return preferences

        if any(
            token in task_text or token in source_detail or token in target_item
            for token in (
                "mine",
                "mines",
                "mine floor",
                "green slime",
                "slime",
                "bug",
                "fly",
                "duggy",
                "grub",
                "rock crab",
                "copper ore",
                "coal",
                "amethyst",
                "quartz",
                "cave carrot",
            )
        ):
            if (
                "farm" in location_text
                and "backwoods" not in location_text
                and "mountain" not in location_text
                and "mine" not in location_text
            ):
                _add("backwoods")
            elif "bus stop" in location_text:
                _add("backwoods")
            else:
                _add("mine", "mines", "elevator", "ladder")
            return preferences

        if "coop" in task_text or "coop" in target_item:
            _add("coop")
        if "barn" in task_text or "barn" in target_item:
            _add("barn")
        if "farmhouse" in task_text or "house" in target_item:
            _add("farmhouse", "house")
        if preferences:
            return preferences

        if "bus stop" in task_text or "bus stop" in target_item:
            _add("bus stop", "bus stop exit")
            return preferences
        if "backwoods" in task_text or "backwoods" in target_item:
            _add("backwoods", "pet bowl entrance", "backwoods exit")
            return preferences

        if any(
            token in target_item or token in task_text or token in source_detail
            for token in ("pierre", "general store")
        ):
            if "farm" in location_text and "bus stop" not in location_text and "town" not in location_text:
                _add("bus stop")
            elif "bus stop" in location_text:
                _add("town")
            else:
                _add("pierre", "general store", "seed shop", "store")
            return preferences

        if (
            "fish shop" in target_item
            or "fish shop" in task_text
            or "fishshop" in target_item
            or "fish shop" in source_detail
            or "fishshop" in source_detail
        ):
            if "farm" in location_text and "bus stop" not in location_text and "town" not in location_text:
                _add("bus stop")
            elif "bus stop" in location_text:
                _add("town")
            else:
                _add("fish shop", "fishshop")
            return preferences

        if (
            "marnie" in target_item
            or "ranch" in target_item
            or "marnie" in task_text
            or "ranch" in task_text
            or "marnie" in source_detail
            or "ranch" in source_detail
        ):
            if "farm" in location_text and "bus stop" not in location_text and "town" not in location_text:
                _add("bus stop")
            elif "bus stop" in location_text:
                _add("town")
            else:
                _add("marnie", "ranch")
            return preferences

        return preferences

    @staticmethod
    def _cell_sort_key(cell: tuple[int, int]) -> tuple[int, int, int, int, int]:
        cell_x, cell_y = cell
        return (abs(cell_x) + abs(cell_y), abs(cell_y), abs(cell_x), cell_y, cell_x)

    @classmethod
    def _adjacent_cell_to_direction(cls, cell: tuple[int, int]) -> str:
        for direction in cls._CARDINAL_DIRECTIONS:
            if cls._direction_to_relative(direction) == cell:
                return direction
        return ""

    @classmethod
    def _build_step_toward_cell_move(
        cls,
        cell: tuple[int, int],
        game_state: Optional[Dict[str, Any]] = None,
        *,
        max_stride: int = 3,
    ) -> str:
        cell_x, cell_y = cell
        if cell_x == 0 and cell_y == 0:
            return ""

        stride = max(1, int(max_stride))
        step_x = 0 if cell_x == 0 else (1 if cell_x > 0 else -1) * min(abs(cell_x), stride)
        step_y = 0 if cell_y == 0 else (1 if cell_y > 0 else -1) * min(abs(cell_y), stride)
        proposed = f"move(x={step_x}, y={step_y})"

        if not isinstance(game_state, dict):
            return proposed

        blockers = cls._get_move_axis_blockers(game_state, step_x, step_y)
        x_blocked = (step_x > 0 and "x+" in blockers) or (step_x < 0 and "x-" in blockers)
        y_blocked = (step_y > 0 and "y+" in blockers) or (step_y < 0 and "y-" in blockers)

        if step_x != 0 and step_y != 0:
            if x_blocked and not y_blocked:
                return f"move(x=0, y={step_y})"
            if y_blocked and not x_blocked:
                return f"move(x={step_x}, y=0)"
            if x_blocked and y_blocked:
                for fallback in (f"move(x={step_x}, y=0)", f"move(x=0, y={step_y})"):
                    blocker = cls._get_single_axis_move_blocker(fallback, game_state)
                    if not blocker:
                        return fallback
                    reroute = cls._build_structure_blocked_move_recovery(
                        action_text=fallback,
                        game_state=game_state,
                        blocker=blocker,
                    )
                    if reroute and reroute != fallback:
                        return reroute
                return proposed

        if step_x != 0 and step_y == 0 and x_blocked:
            blocker = cls._get_single_axis_move_blocker(proposed, game_state)
            if blocker:
                reroute = cls._build_structure_blocked_move_recovery(
                    action_text=proposed,
                    game_state=game_state,
                    blocker=blocker,
                )
                if reroute and reroute != proposed:
                    return reroute

        if step_y != 0 and step_x == 0 and y_blocked:
            blocker = cls._get_single_axis_move_blocker(proposed, game_state)
            if blocker:
                reroute = cls._build_structure_blocked_move_recovery(
                    action_text=proposed,
                    game_state=game_state,
                    blocker=blocker,
                )
                if reroute and reroute != proposed:
                    return reroute

        return proposed

    @staticmethod
    def _clamp_move_action_to_unit_step(action_text: str) -> str:
        move = VLLMClient._parse_move_components(action_text)
        if move is None:
            return action_text

        move_x, move_y = move
        if abs(move_x) + abs(move_y) <= 1:
            return action_text

        if abs(move_y) > abs(move_x) and move_y != 0:
            return f'move(x=0, y={1 if move_y > 0 else -1})'
        if move_x != 0:
            return f'move(x={1 if move_x > 0 else -1}, y=0)'
        return f'move(x=0, y={1 if move_y > 0 else -1})'

    @classmethod
    def _has_adjacent_house_footprint(cls, game_state: Optional[Dict[str, Any]]) -> bool:
        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return False

        for (cell_x, cell_y), raw_text in surroundings_map.items():
            if abs(int(cell_x)) + abs(int(cell_y)) != 1:
                continue
            normalized = cls._normalize_context_text(raw_text)
            if any(token in normalized for token in ("farmhouse", "house", "home")):
                return True
        return False

    @classmethod
    def _prefer_house_adjacent_route_step(
        cls,
        action_text: str,
        game_state: Optional[Dict[str, Any]],
    ) -> str:
        move = cls._parse_move_components(action_text)
        if move is None or not isinstance(game_state, dict):
            return action_text

        move_x, move_y = move
        if not cls._has_adjacent_house_footprint(game_state):
            return action_text
        if move_x == 0 and move_y == 0:
            return action_text

        diagonal_or_block_probe = bool(
            (move_x != 0 and move_y != 0) or abs(move_x) + abs(move_y) == 1
        )
        if not diagonal_or_block_probe and abs(move_x) + abs(move_y) <= 2:
            return action_text

        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return cls._clamp_move_action_to_unit_step(action_text)

        preferred_steps: List[tuple[int, int]] = []
        if abs(move_x) >= abs(move_y) and move_x != 0:
            preferred_steps.append((1 if move_x > 0 else -1, 0))
        if abs(move_y) >= abs(move_x) and move_y != 0:
            preferred_steps.append((0, 1 if move_y > 0 else -1))
        if move_x != 0:
            preferred_steps.append((1 if move_x > 0 else -1, 0))
        if move_y != 0:
            preferred_steps.append((0, 1 if move_y > 0 else -1))
        for fallback in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if fallback not in preferred_steps:
                preferred_steps.append(fallback)

        for step_x, step_y in preferred_steps:
            target_text = str(surroundings_map.get((step_x, step_y), "") or "").strip()
            if target_text and cls._is_hard_structure_blocker(target_text):
                continue
            if (
                not target_text
                or cls._is_open_ground_tile(target_text)
                or cls._is_actionable_front_tile(target_text)
            ):
                return f"move(x={step_x}, y={step_y})"

        return cls._clamp_move_action_to_unit_step(action_text)

    @classmethod
    def _count_nearby_open_ground_tiles(
        cls,
        surroundings_map: Dict[tuple[int, int], Any],
        cell: tuple[int, int],
        *,
        radius: int = 1,
    ) -> int:
        return grounding_count_nearby_open_ground_tiles(
            surroundings_map,
            cell,
            radius=radius,
        )

    @classmethod
    def _count_nearby_hard_structures(
        cls,
        surroundings_map: Dict[tuple[int, int], Any],
        cell: tuple[int, int],
        *,
        radius: int = 1,
    ) -> int:
        return grounding_count_nearby_hard_structures(
            surroundings_map,
            cell,
            radius=radius,
        )

    @classmethod
    def _collect_route_waypoint_candidates(
        cls,
        game_state: Optional[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not isinstance(game_state, dict):
            return []

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        preferences = cls._route_target_preferences(game_state)
        if not preferences:
            return []

        candidates: List[Dict[str, Any]] = []
        source_priority = {
            "surroundings": 0,
            "furniture": 1,
            "npcs": 2,
            "buildings": 3,
            "exits": 4,
        }

        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        for cell, raw_text in surroundings_map.items():
            if cell == (0, 0):
                continue
            normalized = cls._normalize_context_text(raw_text)
            if not normalized:
                continue
            for preference_index, tokens in enumerate(preferences):
                if any(token in normalized for token in tokens):
                    candidates.append(
                        {
                            "source": "surroundings",
                            "offset": cell,
                            "raw": str(raw_text or "").strip(),
                            "name": str(raw_text or "").strip(),
                            "preference_index": preference_index,
                            "source_index": source_priority["surroundings"],
                        }
                    )
                    break

        for source_name in ("furniture", "npcs", "buildings", "exits"):
            for target in cls._iter_named_relative_targets(gathered.get(source_name, "")):
                normalized = cls._normalize_context_text(
                    f"{target['name']} {target['raw']}"
                )
                for preference_index, tokens in enumerate(preferences):
                    if any(token in normalized for token in tokens):
                        candidates.append(
                            {
                                "source": source_name,
                                "offset": target["offset"],
                                "raw": target["raw"],
                                "name": target["name"],
                                "preference_index": preference_index,
                                "source_index": source_priority[source_name],
                            }
                        )
                        break

        return candidates

    @classmethod
    def _best_route_waypoint_candidate(
        cls,
        game_state: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        candidates = cls._collect_route_waypoint_candidates(game_state)
        if not candidates:
            return None

        def _sort_key(candidate: Dict[str, Any]) -> tuple[int, int, int, int, int, int]:
            cell_x, cell_y = candidate["offset"]
            return (
                int(candidate.get("preference_index", 99)),
                int(candidate.get("source_index", 99)),
                abs(cell_x) + abs(cell_y),
                abs(cell_y),
                abs(cell_x),
                0 if candidate.get("source") != "exits" else 1,
            )

        return min(candidates, key=_sort_key)

    @classmethod
    def _move_reduces_waypoint_distance(
        cls,
        action_text: Any,
        waypoint: Optional[Dict[str, Any]],
    ) -> bool:
        move = cls._parse_move_components(action_text)
        if move is None or waypoint is None:
            return False

        step_x, step_y = move
        if step_x == 0 and step_y == 0:
            return False

        target_x, target_y = waypoint.get("offset", (0, 0))
        if target_x and step_x and (target_x > 0) != (step_x > 0):
            return False
        if target_y and step_y and (target_y > 0) != (step_y > 0):
            return False

        before = abs(target_x) + abs(target_y)
        after = abs(target_x - step_x) + abs(target_y - step_y)
        return after < before

    @classmethod
    def _move_conflicts_with_waypoint(
        cls,
        action_text: Any,
        waypoint: Optional[Dict[str, Any]],
    ) -> bool:
        move = cls._parse_move_components(action_text)
        if move is None or waypoint is None:
            return False

        step_x, step_y = move
        target_x, target_y = waypoint.get("offset", (0, 0))
        if target_x and step_x and (target_x > 0) != (step_x > 0):
            return True
        if target_y and step_y and (target_y > 0) != (step_y > 0):
            return True
        return False

    @classmethod
    def _route_recovery_conflicts_with_task_direction(
        cls,
        action_text: Any,
        game_state: Optional[Dict[str, Any]],
    ) -> bool:
        move = cls._parse_move_components(action_text)
        if move is None:
            return False

        step_x, _step_y = move
        task_text = cls._task_context_text(game_state)
        location_text = cls._location_text(game_state)
        if (
            "bus stop" in task_text
            and "farm" in location_text
            and "farmhouse" not in location_text
            and step_x < 0
        ):
            return True
        return False

    @classmethod
    def _build_local_route_recovery_action(
        cls,
        *,
        game_state: Optional[Dict[str, Any]],
    ) -> str:
        best = cls._best_route_waypoint_candidate(game_state)
        if not best:
            if cls._is_sleep_task_context(game_state) and "farmhouse" in cls._location_text(game_state):
                # Conservative deterministic fallback for the farmhouse interior:
                # keep drifting toward the lower-right bed area instead of
                # oscillating back toward the entrance.
                return "move(x=1, y=1)"
            return ""
        cell = best["offset"]
        raw_text = str(best.get("raw", "") or "").strip().lower()
        direction = cls._adjacent_cell_to_direction(cell)
        if direction and best.get("source") != "exits":
            if any(
                token in raw_text
                for token in (
                    "door",
                    "entrance",
                    "hatch",
                    "pet bowl",
                    "feeding bench",
                    "incubator",
                    "animal",
                    "chicken",
                    "cow",
                    "goat",
                    "duck",
                    "rabbit",
                    "egg",
                    "pet",
                    "cat",
                    "dog",
                )
            ):
                if (
                    cls._is_door_or_entrance_text(raw_text)
                    and cls._cultivation_outdoor_task_should_stay_outside(game_state)
                ):
                    return ""
                return f'interact(direction="{direction}")'

        max_stride = 1 if best.get("source") == "exits" else 4
        action = cls._build_step_toward_cell_move(cell, game_state, max_stride=max_stride)
        if cls._route_recovery_conflicts_with_task_direction(action, game_state):
            return ""
        return action

    @classmethod
    def _build_inside_house_exit_recovery_action(
        cls,
        *,
        game_state: Optional[Dict[str, Any]],
    ) -> str:
        route_action = cls._build_local_route_recovery_action(game_state=game_state)
        if route_action:
            return route_action

        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if isinstance(surroundings_map, dict) and surroundings_map:
            for direction in ("down", "left", "right", "up"):
                relative = cls._direction_to_relative(direction)
                if relative is None:
                    continue
                cell_text = str(surroundings_map.get(relative, "") or "").strip()
                if not cell_text or cls._is_open_ground_tile(cell_text):
                    return f"move(x={relative[0]}, y={relative[1]})"
                if cls._is_actionable_front_tile(cell_text):
                    return f'interact(direction="{direction}")'

        # Conservative deterministic fallback for typical farmhouse interior layout.
        return "move(x=0, y=1)"

    @classmethod
    def _desired_clear_tools(
        cls,
        game_state: Optional[Dict[str, Any]],
        selected_item_name: Any,
    ) -> set[str]:
        task_text = cls._task_context_text(game_state)
        clear_profile = build_clear_task_profile(task_text)
        desired_tools = set(clear_profile.get("desired_tools", set()) or set())
        if len(desired_tools) > 1:
            return {"Scythe", "Pickaxe", "Axe"}

        selected_tool = cls._normalize_tool_name(selected_item_name)
        if not desired_tools and selected_tool in {"Scythe", "Pickaxe", "Axe"}:
            desired_tools.add(selected_tool)

        return desired_tools or {"Scythe", "Pickaxe", "Axe"}

    @classmethod
    def _build_local_clear_recovery_action(
        cls,
        *,
        game_state: Optional[Dict[str, Any]],
        selected_item_name: Any,
        inventory: Any,
        toolbar_information: Any,
    ) -> str:
        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return ""

        clear_profile = build_clear_task_profile(cls._task_context_text(game_state))
        desired_tools = cls._desired_clear_tools(game_state, selected_item_name)
        selected_tool = cls._normalize_tool_name(selected_item_name)
        candidates: List[tuple[tuple[int, int, int, int, int], tuple[int, int], str]] = []
        for cell, raw_text in surroundings_map.items():
            if cell == (0, 0):
                continue
            clearable = cls._classify_clearable_object(raw_text)
            if not clearable or clearable["tool"] not in desired_tools:
                continue
            if clear_profile and not clear_target_matches_profile(raw_text, clear_profile):
                continue
            candidates.append((cls._cell_sort_key(cell), cell, clearable["tool"]))

        if not candidates:
            return ""

        _, cell, required_tool = min(candidates, key=lambda item: item[0])
        direction = cls._adjacent_cell_to_direction(cell)
        if direction:
            if selected_tool == required_tool:
                return f'use(direction="{direction}")'
            tool_slot = cls._find_tool_slot(inventory, required_tool)
            if tool_slot is None:
                tool_slot = cls._find_tool_slot_in_toolbar_text(toolbar_information, required_tool)
            if tool_slot is not None:
                return f"choose_item(slot_index={tool_slot})"

        return cls._build_step_toward_cell_move(cell, game_state, max_stride=3)

    @classmethod
    def _build_local_tilling_recovery_action(
        cls,
        *,
        game_state: Optional[Dict[str, Any]],
        selected_item_name: Any,
        inventory: Any,
        toolbar_information: Any,
        invalid_direction: str = "",
    ) -> str:
        selected_tool = cls._normalize_tool_name(selected_item_name)
        if not cls._is_tilling_or_digging_context(game_state) and selected_tool != "Hoe":
            return ""

        if selected_tool != "Hoe":
            tool_slot = cls._find_tool_slot(inventory, "Hoe")
            if tool_slot is None:
                tool_slot = cls._find_tool_slot_in_toolbar_text(toolbar_information, "Hoe")
            if tool_slot is not None:
                return f"choose_item(slot_index={tool_slot})"

        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return ""

        if cls._is_inside_house_for_cultivation(game_state):
            exit_action = cls._build_inside_house_exit_recovery_action(
                game_state=game_state,
            )
            if exit_action:
                return exit_action

        nearby_current_structures = cls._count_nearby_hard_structures(
            surroundings_map,
            (0, 0),
            radius=1,
        )

        zero_progress_streak = 0
        repeated_action_streak = 0
        position_issue_detected = False
        last_exec_info = {}
        last_action_text = ""
        last_action_direction = ""
        if isinstance(game_state, dict):
            zero_progress_streak = int(game_state.get("zero_progress_streak", 0) or 0)
            repeated_action_streak = int(game_state.get("repeated_action_streak", 0) or 0)
            position_issue_detected = bool(game_state.get("position_issue_detected", False))
            last_exec_info = game_state.get("last_exec_info", {})
            last_action_text = str(game_state.get("last_action", "") or "").strip()

        last_directional_skill = cls._parse_directional_skill(last_action_text)
        if last_directional_skill and last_directional_skill[0] == "use":
            last_action_direction = last_directional_skill[1]

        stalled_tilling_context = bool(
            zero_progress_streak >= 1
            or repeated_action_streak >= 2
            or position_issue_detected
            or execution_has_no_confirmation(last_exec_info)
        )

        nearby_tilled_cells = sum(
            1 for cell, raw_text in surroundings_map.items()
            if cell != (0, 0) and cls._placeable_target_is_tilled(raw_text)
        )
        patch_exhausted = bool(
            nearby_tilled_cells >= 4
            or (
                nearby_tilled_cells >= 2
                and (zero_progress_streak >= 1 or repeated_action_streak >= 2)
            )
        )
        prefer_unit_step_reposition = bool(
            stalled_tilling_context or patch_exhausted or nearby_tilled_cells >= 2
        )

        allow_adjacent_empty_use = not (stalled_tilling_context or patch_exhausted)

        candidates: List[tuple[tuple[Any, ...], tuple[int, int], str, Dict[str, Any]]] = []
        for cell, raw_text in surroundings_map.items():
            if cell == (0, 0):
                continue
            text = str(raw_text or "").strip()
            if not text:
                continue
            candidate_info = classify_tilling_target(surroundings_map, cell)
            if not candidate_info:
                continue
            priority = 0 if candidate_info["kind"] == "explicit_ground" else 1
            candidates.append(((priority, cls._cell_sort_key(cell)), cell, text, candidate_info))

        if not candidates:
            if allow_adjacent_empty_use and not patch_exhausted:
                adjacent_empty_use_candidates: List[tuple[tuple[int, int, int, int, int], str]] = []
                for cell, raw_text in surroundings_map.items():
                    if cell == (0, 0) or abs(cell[0]) + abs(cell[1]) != 1:
                        continue
                    text = str(raw_text or "").strip()
                    if not (cls._is_empty_like_tile(text) or cls._is_open_ground_tile(text)):
                        continue
                    if cls._count_nearby_hard_structures(surroundings_map, cell, radius=1) > 0:
                        continue
                    direction = cls._adjacent_cell_to_direction(cell)
                    if not direction or direction == invalid_direction:
                        continue
                    adjacent_empty_use_candidates.append((cls._cell_sort_key(cell), direction))
                if adjacent_empty_use_candidates:
                    _, direction = min(adjacent_empty_use_candidates, key=lambda item: item[0])
                    return f'use(direction="{direction}")'

            fallback_moves: List[tuple[tuple[Any, ...], tuple[int, int]]] = []
            for cell, raw_text in surroundings_map.items():
                if cell == (0, 0):
                    continue
                text = str(raw_text or "").strip()
                if not text:
                    continue
                if not (cls._is_empty_like_tile(text) or cls._is_open_ground_tile(text)):
                    continue
                fallback_moves.append(
                    (
                        (
                            cls._count_nearby_hard_structures(surroundings_map, cell, radius=1),
                            abs(cell[0]) + abs(cell[1]),
                            cls._cell_sort_key(cell),
                        ),
                        cell,
                    )
                )
            if not fallback_moves:
                return ""
            _, cell = min(fallback_moves, key=lambda item: item[0])
            fallback_move = cls._build_step_toward_cell_move(
                cell,
                game_state,
                max_stride=1 if prefer_unit_step_reposition else 3,
            )
            if prefer_unit_step_reposition:
                fallback_move = cls._clamp_move_action_to_unit_step(fallback_move)
            return fallback_move

        force_patch_reposition = bool(patch_exhausted or nearby_current_structures > 0)
        if force_patch_reposition:
            far_candidates = [
                (sort_key, cell, raw_text, candidate_info)
                for sort_key, cell, raw_text, candidate_info in candidates
                if abs(cell[0]) + abs(cell[1]) >= 2
            ]
            if far_candidates:
                candidates = far_candidates

        if force_patch_reposition:
            ranked_candidates = []
            for sort_key, cell, raw_text, candidate_info in candidates:
                ranked_candidates.append(
                    (
                        (
                            candidate_info["nearby_structures"],
                            -candidate_info["nearby_open_ground"],
                            abs(cell[0]) + abs(cell[1]),
                            sort_key,
                        ),
                        cell,
                        raw_text,
                        candidate_info,
                    )
                )
            _, cell, raw_text, candidate_info = min(ranked_candidates, key=lambda item: item[0])
        else:
            _, cell, raw_text, candidate_info = min(candidates, key=lambda item: item[0])
        direction = cls._adjacent_cell_to_direction(cell)
        if direction and direction != invalid_direction:
            if cls._is_valid_hoe_target(raw_text):
                if not stalled_tilling_context:
                    return f'use(direction="{direction}")'
                if (
                    direction != last_action_direction
                    and nearby_current_structures == 0
                    and candidate_info["nearby_structures"] == 0
                    and candidate_info["nearby_open_ground"] >= 2
                ):
                    return f'use(direction="{direction}")'
            if (
                candidate_info["kind"] == "open_patch"
                and allow_adjacent_empty_use
                and not patch_exhausted
                and cls._is_allowed_empty_hoe_target(game_state, raw_text, direction)
            ):
                return f'use(direction="{direction}")'

        step_move = cls._build_step_toward_cell_move(
            cell,
            game_state,
            max_stride=1 if prefer_unit_step_reposition else 3,
        )
        if not step_move:
            return ""
        if prefer_unit_step_reposition:
            step_move = cls._clamp_move_action_to_unit_step(step_move)
        blocker = cls._get_single_axis_move_blocker(step_move, game_state)
        if blocker:
            reroute = cls._build_structure_blocked_move_recovery(
                action_text=step_move,
                game_state=game_state,
                blocker=blocker,
            )
            if reroute:
                if prefer_unit_step_reposition:
                    reroute = cls._clamp_move_action_to_unit_step(reroute)
                return reroute
        return step_move

    @classmethod
    def _build_local_watering_recovery_action(
        cls,
        *,
        game_state: Optional[Dict[str, Any]],
        selected_item_name: Any,
        inventory: Any,
        toolbar_information: Any,
        invalid_direction: str = "",
    ) -> str:
        if not cls._is_watering_context(game_state):
            return ""

        selected_tool = cls._normalize_tool_name(selected_item_name)
        if selected_tool != "Watering Can":
            tool_slot = cls._find_tool_slot(inventory, "Watering Can")
            if tool_slot is None:
                tool_slot = cls._find_tool_slot_in_toolbar_text(toolbar_information, "Watering Can")
            if tool_slot is not None:
                return f"choose_item(slot_index={tool_slot})"

        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return ""

        candidates: List[tuple[tuple[int, int, int, int, int], tuple[int, int], str]] = []
        for cell, raw_text in surroundings_map.items():
            if cell == (0, 0):
                continue
            text = str(raw_text or "").strip()
            if not text:
                continue
            direction = cls._adjacent_cell_to_direction(cell)
            if direction:
                if cls._is_valid_watering_target(game_state, direction, text):
                    candidates.append((cls._cell_sort_key(cell), cell, text))
            elif cls._target_text_contains_crop(text):
                candidates.append((cls._cell_sort_key(cell), cell, text))

        if not candidates:
            return ""

        _, cell, raw_text = min(candidates, key=lambda item: item[0])
        direction = cls._adjacent_cell_to_direction(cell)
        if direction and direction != invalid_direction and cls._is_valid_watering_target(game_state, direction, raw_text):
            return f'use(direction="{direction}")'

        step_move = cls._build_step_toward_cell_move(cell, game_state, max_stride=3)
        if not step_move:
            return ""
        blocker = cls._get_single_axis_move_blocker(step_move, game_state)
        if blocker:
            reroute = cls._build_structure_blocked_move_recovery(
                action_text=step_move,
                game_state=game_state,
                blocker=blocker,
            )
            if reroute:
                return reroute
        return step_move

    @classmethod
    def _build_local_placeable_recovery_action(
        cls,
        *,
        game_state: Optional[Dict[str, Any]],
        item_name: Any,
        invalid_direction: str = "",
    ) -> str:
        if not cls._selected_item_requires_interact(item_name):
            return ""

        alternative_directions = cls._collect_valid_placeable_directions(
            game_state=game_state,
            item_name=item_name,
            invalid_direction=invalid_direction,
        )
        if alternative_directions:
            return f'interact(direction="{alternative_directions[0]}")'

        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return ""

        candidates: List[tuple[tuple[int, int, int, int, int], tuple[int, int]]] = []
        for cell, raw_text in surroundings_map.items():
            if cell == (0, 0):
                continue
            if not cls._is_valid_visible_placeable_target(item_name, raw_text):
                continue
            direction = cls._adjacent_cell_to_direction(cell)
            if direction and direction == invalid_direction:
                continue
            candidates.append((cls._cell_sort_key(cell), cell))

        if not candidates:
            return ""

        _, cell = min(candidates, key=lambda item: item[0])
        direction = cls._adjacent_cell_to_direction(cell)
        if direction:
            return f'interact(direction="{direction}")'
        return cls._build_step_toward_cell_move(cell, game_state, max_stride=3)

    @classmethod
    def _build_invalidated_suggestion_local_recovery(
        cls,
        *,
        game_state: Optional[Dict[str, Any]],
        suggestion_action: Any,
        selected_item_name: Any,
        inventory: Any,
        toolbar_information: Any,
    ) -> str:
        if not isinstance(game_state, dict):
            return ""

        suggestion_text = str(suggestion_action or "").strip()
        if not suggestion_text:
            return ""

        task_text = cls._task_context_text(game_state)
        directional_skill = cls._parse_directional_skill(suggestion_text)
        normalized_selected_tool = cls._normalize_tool_name(selected_item_name)

        if task_text.startswith("clear ") or (
            directional_skill
            and directional_skill[0] == "use"
            and normalized_selected_tool in {"Scythe", "Pickaxe", "Axe"}
        ):
            corrected = cls._build_local_clear_recovery_action(
                game_state=game_state,
                selected_item_name=selected_item_name,
                inventory=inventory,
                toolbar_information=toolbar_information,
            )
            if corrected and corrected != suggestion_text:
                return corrected

        if (
            cls._is_tilling_or_digging_context(game_state)
            or (
                directional_skill
                and directional_skill[0] == "use"
                and normalized_selected_tool == "Hoe"
            )
        ):
            invalid_direction = directional_skill[1] if directional_skill and directional_skill[0] == "use" else ""
            corrected = cls._build_local_tilling_recovery_action(
                game_state=game_state,
                selected_item_name=selected_item_name,
                inventory=inventory,
                toolbar_information=toolbar_information,
                invalid_direction=invalid_direction,
            )
            if corrected and corrected != suggestion_text:
                return corrected

        if cls._is_watering_context(game_state):
            invalid_direction = directional_skill[1] if directional_skill and directional_skill[0] == "use" else ""
            corrected = cls._build_local_watering_recovery_action(
                game_state=game_state,
                selected_item_name=selected_item_name,
                inventory=inventory,
                toolbar_information=toolbar_information,
                invalid_direction=invalid_direction,
            )
            if corrected and corrected != suggestion_text:
                return corrected

        if (
            task_text.startswith("fertilize ")
            or task_text.startswith("sow ")
            or cls._selected_item_requires_interact(selected_item_name)
            or (directional_skill and directional_skill[0] == "interact")
        ):
            invalid_direction = directional_skill[1] if directional_skill and directional_skill[0] == "interact" else ""
            corrected = cls._build_local_placeable_recovery_action(
                game_state=game_state,
                item_name=selected_item_name,
                invalid_direction=invalid_direction,
            )
            if corrected and corrected != suggestion_text:
                return corrected

        if suggestion_text.startswith("move("):
            route_recovery = cls._build_local_route_recovery_action(
                game_state=game_state,
            )
            if route_recovery and route_recovery != suggestion_text:
                return route_recovery

        if suggestion_text.startswith("move("):
            blocker = cls._get_single_axis_move_blocker(suggestion_text, game_state)
            if blocker:
                corrected = cls._build_structure_blocked_move_recovery(
                    action_text=suggestion_text,
                    game_state=game_state,
                    blocker=blocker,
                )
                if corrected and corrected != suggestion_text:
                    return corrected

        return ""

    @classmethod
    def _task_prefers_inventory_setup(
        cls,
        game_state: Optional[Dict[str, Any]],
    ) -> bool:
        task_text = cls._task_context_text(game_state)
        if not task_text:
            return False

        inventory_setup_markers = (
            "feed",
            "feeding bench",
            "hay",
            "pet bowl",
            "milk",
            "milk pail",
            "egg",
            "incubat",
            "animal door",
            "coop",
            "barn",
        )
        return any(marker in task_text for marker in inventory_setup_markers)

    @classmethod
    def _build_placeable_inventory_setup_action(
        cls,
        *,
        game_state: Optional[Dict[str, Any]],
        selected_item_name: Any,
        inventory: Any,
        toolbar_information: Any,
    ) -> str:
        if not isinstance(game_state, dict):
            return ""

        task_text = cls._task_context_text(game_state)
        target_item = cls._target_item_text(game_state)
        slot_map = cls._extract_inventory_slot_map(inventory, toolbar_information)
        if not slot_map:
            return ""

        normalized_selected = cls._normalize_context_text(selected_item_name)

        def _pick_matching_slot(kind: str) -> str:
            candidates: List[tuple[tuple[int, int], int]] = []
            for slot_index, item_name in slot_map.items():
                normalized_item = cls._normalize_context_text(item_name)
                if not normalized_item or cls._slot_is_explicitly_empty(item_name):
                    continue
                if kind == "seed":
                    if not cls._selected_item_is_seed(item_name):
                        continue
                elif kind == "fertilizer":
                    if not cls._selected_item_is_fertilizer(item_name):
                        continue
                else:
                    continue

                if normalized_item == normalized_selected:
                    return ""

                exact_target = 0 if target_item and (
                    normalized_item == target_item
                    or target_item in normalized_item
                    or normalized_item in target_item
                ) else 1
                candidates.append(((exact_target, int(slot_index)), int(slot_index)))

            if not candidates:
                return ""
            candidates.sort(key=lambda item: item[0])
            return f"choose_item(slot_index={candidates[0][1]})"

        if task_text.startswith(("sow ", "sow_")) and not cls._selected_item_is_seed(selected_item_name):
            return _pick_matching_slot("seed")

        if task_text.startswith(("fertilize ", "fertilize_")) and not cls._selected_item_is_fertilizer(selected_item_name):
            return _pick_matching_slot("fertilizer")

        return ""

    @classmethod
    def _expected_task_setup_tool(
        cls,
        game_state: Optional[Dict[str, Any]],
        decision_reason: Any = "",
    ) -> str:
        task_text = cls._task_context_text(game_state)
        explicit_tool = cls._extract_explicit_tool_name(task_text)
        if explicit_tool:
            return explicit_tool

        explicit_tool = cls._extract_explicit_tool_name(decision_reason)
        if explicit_tool:
            return explicit_tool

        if cls._is_tilling_or_digging_context(game_state):
            return "Hoe"
        if cls._is_watering_context(game_state):
            return "Watering Can"

        clear_profile = build_clear_task_profile(task_text)
        desired_tools = set(clear_profile.get("desired_tools", set()) or set()) if clear_profile else set()
        if len(desired_tools) == 1:
            return next(iter(desired_tools))

        return ""

    @classmethod
    def _task_critical_choose_item_override_before_move(
        cls,
        *,
        game_state: Optional[Dict[str, Any]],
        choose_slot: Optional[int],
        chosen_item_name: Any,
        selected_item_name: Any,
        decision_reason: Any = "",
    ) -> bool:
        expected_tool = cls._expected_task_setup_tool(
            game_state=game_state,
            decision_reason=decision_reason,
        )
        if not expected_tool:
            return False

        normalized_selected_tool = cls._normalize_tool_name(selected_item_name)
        if normalized_selected_tool == expected_tool:
            return False

        normalized_chosen_tool = cls._normalize_tool_name(chosen_item_name)
        if normalized_chosen_tool:
            return normalized_chosen_tool == expected_tool

        return choose_slot is not None and 0 <= choose_slot <= 11

    @classmethod
    def _build_autonomous_local_recovery_action(
        cls,
        *,
        game_state: Optional[Dict[str, Any]],
    ) -> str:
        if not isinstance(game_state, dict):
            return ""

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        inventory = gathered.get("inventory") or game_state.get("inventory", [])
        toolbar_information = game_state.get("toolbar_information") or gathered.get("toolbar_information") or ""
        selected_item_name = (
            cls._extract_selected_item_name(gathered)
            or cls._extract_selected_item_name_from_toolbar(toolbar_information)
        )

        front_context = cls._build_front_obstacle_context(
            game_state,
            [],
            hint_action="",
        )
        blocked_override_action = str(
            front_context.get("blocked_override_action", "") or ""
        ).strip()
        if blocked_override_action:
            return blocked_override_action

        immediate_use = str(cls._build_immediate_use_if_aligned(game_state) or "").strip()
        if immediate_use:
            return immediate_use

        placeable_setup = str(
            cls._build_placeable_inventory_setup_action(
                game_state=game_state,
                selected_item_name=selected_item_name,
                inventory=inventory,
                toolbar_information=toolbar_information,
            )
            or ""
        ).strip()
        if placeable_setup:
            return placeable_setup

        route_recovery = str(
            cls._build_local_route_recovery_action(game_state=game_state) or ""
        ).strip()
        if route_recovery:
            return route_recovery

        for action in (
            cls._build_local_tilling_recovery_action(
                game_state=game_state,
                selected_item_name=selected_item_name,
                inventory=inventory,
                toolbar_information=toolbar_information,
            ),
            cls._build_local_watering_recovery_action(
                game_state=game_state,
                selected_item_name=selected_item_name,
                inventory=inventory,
                toolbar_information=toolbar_information,
            ),
            cls._build_local_clear_recovery_action(
                game_state=game_state,
                selected_item_name=selected_item_name,
                inventory=inventory,
                toolbar_information=toolbar_information,
            ),
            cls._build_local_placeable_recovery_action(
                game_state=game_state,
                item_name=selected_item_name,
            ),
        ):
            candidate = str(action or "").strip()
            if candidate:
                return candidate

        if cls._task_prefers_inventory_setup(game_state):
            target_text = " ".join(
                str(value or "")
                for value in (
                    game_state.get("target_item"),
                    game_state.get("source_type"),
                    game_state.get("source_detail"),
                    game_state.get("task"),
                    game_state.get("subtask_description"),
                )
            ).lower()
            slot_map = cls._extract_inventory_slot_map(inventory, toolbar_information)
            for slot_index, item_name in sorted(slot_map.items()):
                normalized_item = str(item_name or "").strip()
                lowered_item = normalized_item.lower()
                if not normalized_item or cls._slot_is_explicitly_empty(normalized_item):
                    continue
                if lowered_item == str(selected_item_name or "").strip().lower():
                    continue
                if any(
                    token in target_text
                    for token in (
                        lowered_item,
                        "hay" if "hay" in lowered_item else "",
                        "milk pail" if "milk pail" in lowered_item else "",
                        "egg" if "egg" in lowered_item else "",
                    )
                    if token
                ):
                    return f"choose_item(slot_index={slot_index})"

        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if surroundings_map:
            candidates: List[tuple[tuple[int, int, int, int, int], tuple[int, int]]] = []
            for cell, raw_text in surroundings_map.items():
                if cell == (0, 0):
                    continue
                cell_text = str(raw_text or "").strip()
                if not cls._is_open_ground_tile(cell_text):
                    continue
                candidates.append((cls._cell_sort_key(cell), cell))

            if candidates:
                _, cell = min(candidates, key=lambda item: item[0])
                return cls._build_step_toward_cell_move(cell, game_state, max_stride=3)

        return ""

    @classmethod
    def _validate_decision_against_state(
        cls,
        decision: VLLMDecision,
        suggestion: Dict[str, str],
        game_state: Optional[Dict[str, Any]],
    ) -> VLLMDecision:
        if decision.escalate or not isinstance(game_state, dict):
            return decision

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        action = str(decision.action or "").strip()
        if not action:
            return decision

        inventory = gathered.get("inventory", [])
        toolbar_information = game_state.get("toolbar_information") or gathered.get("toolbar_information") or ""
        selected_item_name = (
            cls._extract_selected_item_name(gathered)
            or cls._extract_selected_item_name_from_toolbar(toolbar_information)
        )
        selected_slot = cls._extract_selected_slot_index(game_state, gathered)
        inventory_slot_map = cls._extract_inventory_slot_map(inventory, toolbar_information)
        suggested_action = str(suggestion.get("action", "") or "").strip()
        zero_progress_streak = int(game_state.get("zero_progress_streak", 0) or 0)
        repeated_action_streak = int(game_state.get("repeated_action_streak", 0) or 0)
        position_issue_detected = bool(game_state.get("position_issue_detected", False))
        decision_reason = str(decision.reason or "").strip().lower()
        current_menu = cls._get_current_menu_value(game_state, gathered)
        current_menu_type = cls._normalize_menu_type(current_menu)
        sleep_task = cls._is_sleep_task_context(game_state)
        bed_direction = cls._find_adjacent_bed_direction(game_state)
        choose_option_index = cls._parse_choose_option_index(action)
        last_action_text = str(game_state.get("last_action", "") or "").strip()
        last_exec_info = game_state.get("last_exec_info", {})
        task_progress_quantity = game_state.get("task_progress_quantity", None)
        previous_task_progress_quantity = game_state.get("previous_task_progress_quantity", None)
        task_progress_delta = game_state.get("task_progress_delta", None)
        try:
            numeric_task_progress_quantity = float(task_progress_quantity)
        except (TypeError, ValueError):
            numeric_task_progress_quantity = None
        try:
            numeric_previous_task_progress_quantity = float(previous_task_progress_quantity)
        except (TypeError, ValueError):
            numeric_previous_task_progress_quantity = None
        try:
            numeric_task_progress_delta = float(task_progress_delta)
        except (TypeError, ValueError):
            numeric_task_progress_delta = None
        task_text = cls._task_context_text(game_state)
        craft_target = cls._target_item_text(game_state)
        canonical_purchase_action = cls._build_shop_menu_purchase_action(
            game_state,
            current_menu=current_menu,
        )
        canonical_sell_action = cls._build_shop_menu_sell_action(
            game_state,
            current_menu=current_menu,
            inventory_slot_map=inventory_slot_map,
            selected_slot=selected_slot,
        )

        if sleep_task and current_menu_type in {"", "no menu"}:
            if cls._sleep_task_at_farmhouse_entrance(game_state):
                corrected = "move(x=0, y=-1)"
                if action != corrected:
                    logger.write(
                        "[FastLLM] Rewrote sleep farmhouse entrance action into "
                        f"interior bed-route step: {action} -> {corrected}"
                    )
                    return VLLMDecision(
                        action=corrected,
                        reason="sleep_farmhouse_entrance_route_fix",
                        escalate=False,
                    )

        refused_action = execution_refused_action(last_exec_info)
        refusal_type = execution_refusal_type(last_exec_info)
        normalized_action = cls._normalize_action_text(action)
        normalized_suggested = cls._normalize_action_text(suggested_action)
        normalized_refused = cls._normalize_action_text(refused_action)
        last_directional_skill = cls._parse_directional_skill(last_action_text)
        recent_no_progress_productive_action = bool(
            last_directional_skill
            and last_directional_skill[0] in {"use", "interact"}
            and (
                (numeric_task_progress_delta is not None and numeric_task_progress_delta <= 0)
                or (
                    numeric_task_progress_quantity is not None
                    and numeric_previous_task_progress_quantity is not None
                    and numeric_task_progress_quantity <= numeric_previous_task_progress_quantity
                )
            )
        )

        if current_menu_type == "mappage":
            corrected = 'menu(option="close", menu_name="map")'
            if action != corrected:
                logger.write(
                    "[FastLLM] Rewrote world action while the map is open into map-close: "
                    f"{action} -> {corrected}"
                )
                return VLLMDecision(
                    action=corrected,
                    reason="map_menu_close_fix",
                    escalate=False,
                )

        if current_menu_type == "shopmenu":
            if canonical_sell_action and action != canonical_sell_action and task_text.startswith(("sell ", "sell_")):
                logger.write(
                    "[FastLLM] Rewrote open shop-menu action into canonical sell action: "
                    f"{action} -> {canonical_sell_action}"
                )
                return VLLMDecision(
                    action=canonical_sell_action,
                    reason="shop_menu_sell_fix",
                    escalate=False,
                )
            if canonical_purchase_action and action != canonical_purchase_action and task_text.startswith(("purchase ", "purchase_")):
                logger.write(
                    "[FastLLM] Rewrote open shop-menu action into canonical purchase action: "
                    f"{action} -> {canonical_purchase_action}"
                )
                return VLLMDecision(
                    action=canonical_purchase_action,
                    reason="shop_menu_purchase_fix",
                    escalate=False,
                )

        if current_menu_type == "dialoguebox" and cls._menu_prefers_negative_confirmation(current_menu):
            decline_action = "choose_option(option_index=2, quantity=0)"
            if action != decline_action:
                logger.write(
                    "[FastLLM] Rewrote consumable confirmation dialogue into decline: "
                    f"{action} -> {decline_action}"
                )
                return VLLMDecision(
                    action=decline_action,
                    reason="consumable_dialogue_decline_fix",
                    escalate=False,
                )

        if normalized_refused and normalized_action == normalized_refused:
            if normalized_suggested and normalized_suggested != normalized_refused and suggested_action:
                logger.write(
                    "[FastLLM] Reverted repeated refused action to BigBrain suggestion: "
                    f"{action} -> {suggested_action} | refusal_type={refusal_type or 'unknown'}"
                )
                return cls._validate_decision_against_state(
                    VLLMDecision(
                        action=suggested_action,
                        reason="avoid_refused_action:follow_suggestion",
                        escalate=False,
                    ),
                    suggestion,
                    game_state,
                )

            corrected = ""
            if action.startswith("move("):
                corrected = cls._build_refused_move_recovery_action(action)

            if not corrected:
                corrected = cls._build_invalidated_suggestion_local_recovery(
                    game_state=game_state,
                    suggestion_action=action,
                    selected_item_name=selected_item_name,
                    inventory=inventory,
                    toolbar_information=toolbar_information,
                )

            if not corrected:
                corrected = cls._build_local_route_recovery_action(game_state=game_state)

            if corrected and cls._normalize_action_text(corrected) != normalized_refused:
                logger.write(
                    "[FastLLM] Rewrote repeated refused action into grounded alternative: "
                    f"{action} -> {corrected} | refusal_type={refusal_type or 'unknown'}"
                )
                corrected_suggestion = dict(suggestion or {})
                corrected_suggestion["action"] = corrected
                return cls._validate_decision_against_state(
                    VLLMDecision(
                        action=corrected,
                        reason="avoid_refused_action:local_recovery",
                        escalate=False,
                    ),
                    corrected_suggestion,
                    game_state,
                )

            logger.write(
                "[FastLLM] Escalating repeated refused action with no grounded alternative: "
                f"{action} | refusal_type={refusal_type or 'unknown'}"
            )
            return VLLMDecision(
                action="",
                reason="repeat_refused_action",
                escalate=True,
            )

        if (
            choose_option_index is not None
            and current_menu_type in {"", "no menu"}
            and not sleep_task
            and any(
                token in task_text
                for token in (
                    "purchase",
                    "sell",
                    "ship",
                    "upgrade",
                    "build",
                    "joja",
                    "pierre",
                    "clint",
                    "robin",
                    "marnie",
                    "shop",
                    "counter",
                    "shipping bin",
                    "backpack",
                )
            )
        ):
            corrected = cls._build_local_route_recovery_action(game_state=game_state)
            if corrected and corrected != action:
                logger.write(
                    "[FastLLM] Rewrote choose_option() with no open service menu into grounded service context action: "
                    f"{action} -> {corrected}"
                )
                return cls._validate_decision_against_state(
                    VLLMDecision(
                        action=corrected,
                        reason="service_menu_no_menu_fix",
                        escalate=False,
                    ),
                    suggestion,
                    game_state,
                )

        if decision_reason == "parse_fallback_invalidated_suggestion":
            corrected = cls._build_invalidated_suggestion_local_recovery(
                game_state=game_state,
                suggestion_action=suggested_action or action,
                selected_item_name=selected_item_name,
                inventory=inventory,
                toolbar_information=toolbar_information,
            )
            if corrected:
                logger.write(
                    "[FastLLM] Recovered parse-invalidated suggestion with grounded local action: "
                    f"{suggested_action or action} -> {corrected}"
                )
                return cls._validate_decision_against_state(
                    VLLMDecision(
                        action=corrected,
                        reason="parse_fallback_invalidated_local_recovery",
                        escalate=False,
                    ),
                    suggestion,
                    game_state,
                )
            logger.write(
                "[FastLLM] Escalating parse-invalidated suggestion with no grounded local recovery: "
                f"{suggested_action or action}"
            )
            return VLLMDecision(
                action="",
                reason="parse_fallback_invalidated_suggestion",
                escalate=True,
            )

        menu_action = cls._parse_menu_action(action)
        if (
            menu_action is not None
            and menu_action[0] == "open"
            and craft_target
            and task_text.startswith(("craft ", "craft_"))
        ):
            missing_materials = cls._craft_missing_materials(craft_target, inventory)
            if missing_materials:
                suggested_menu_action = cls._parse_menu_action(suggested_action)
                suggested_is_spurious_menu_close = (
                    suggested_menu_action is not None
                    and suggested_menu_action[0] == "close"
                    and current_menu_type in {"", "no menu"}
                )
                corrected = cls._build_autonomous_local_recovery_action(
                    game_state=game_state,
                )
                if corrected and corrected != action:
                    logger.write(
                        "[FastLLM] Rewrote crafting menu open with missing materials into grounded recovery action: "
                        f"{action} -> {corrected} | missing={', '.join(missing_materials[:3])}"
                    )
                    return cls._validate_decision_against_state(
                        VLLMDecision(
                            action=corrected,
                            reason="craft_missing_materials_recovery",
                            escalate=False,
                        ),
                        suggestion,
                        game_state,
                    )
                if suggested_action and suggested_action != action and not suggested_is_spurious_menu_close:
                    logger.write(
                        "[FastLLM] Reverted crafting menu open with missing materials to the planning suggestion: "
                        f"{action} -> {suggested_action} | missing={', '.join(missing_materials[:3])}"
                    )
                    return cls._validate_decision_against_state(
                        VLLMDecision(
                            action=suggested_action,
                            reason="craft_missing_materials_revert",
                            escalate=False,
                        ),
                        suggestion,
                        game_state,
                    )
                logger.write(
                    "[FastLLM] Escalating crafting menu open with missing materials and no grounded recovery: "
                    f"{action} | missing={', '.join(missing_materials[:3])}"
                )
                return VLLMDecision(
                    action="",
                    reason="craft_missing_materials",
                    escalate=True,
                )

        craft_item = cls._parse_craft_item_name(action)
        if craft_item:
            missing_materials = cls._craft_missing_materials(craft_item, inventory)
            if missing_materials:
                suggested_menu_action = cls._parse_menu_action(suggested_action)
                suggested_is_spurious_menu_close = (
                    suggested_menu_action is not None
                    and suggested_menu_action[0] == "close"
                    and current_menu_type in {"", "no menu"}
                )
                corrected = cls._build_autonomous_local_recovery_action(
                    game_state=game_state,
                )
                if corrected and corrected != action:
                    logger.write(
                        "[FastLLM] Rewrote invalid craft() with missing materials into grounded recovery action: "
                        f"{action} -> {corrected} | missing={', '.join(missing_materials[:3])}"
                    )
                    return cls._validate_decision_against_state(
                        VLLMDecision(
                            action=corrected,
                            reason="craft_missing_materials_recovery",
                            escalate=False,
                        ),
                        suggestion,
                        game_state,
                    )
                if suggested_action and suggested_action != action and not suggested_is_spurious_menu_close:
                    logger.write(
                        "[FastLLM] Reverted craft() with missing materials to the planning suggestion: "
                        f"{action} -> {suggested_action} | missing={', '.join(missing_materials[:3])}"
                    )
                    return cls._validate_decision_against_state(
                        VLLMDecision(
                            action=suggested_action,
                            reason="craft_missing_materials_revert",
                            escalate=False,
                        ),
                        suggestion,
                        game_state,
                    )
                logger.write(
                    "[FastLLM] Escalating craft() with missing materials and no grounded recovery: "
                    f"{action} | missing={', '.join(missing_materials[:3])}"
                )
                return VLLMDecision(
                    action="",
                    reason="craft_missing_materials",
                    escalate=True,
                )

        if (
            sleep_task
            and current_menu_type == "dialoguebox"
            and (bed_direction or cls._menu_contains_sleep_prompt(current_menu))
        ):
            confirm_action = "choose_option(option_index=1, quantity=0)"
            if choose_option_index is not None and choose_option_index <= 0:
                logger.write(
                    "[FastLLM] Rewrote sleep dialogue cancel into confirm: "
                    f"{action} -> {confirm_action}"
                )
                return VLLMDecision(
                    action=confirm_action,
                    reason="sleep_dialogue_confirm_fix",
                    escalate=False,
                )
            directional_skill = cls._parse_directional_skill(action)
            if directional_skill and directional_skill[0] == "interact":
                logger.write(
                    "[FastLLM] Rewrote sleep dialogue interact into confirm: "
                    f"{action} -> {confirm_action}"
                )
                return VLLMDecision(
                    action=confirm_action,
                    reason="sleep_dialogue_confirm_fix",
                    escalate=False,
                )

        if (
            current_menu_type in {"objectdialogue", "notificationdialogue"}
            and not cls._menu_has_response_options(current_menu)
        ):
            confirm_action = "choose_option(option_index=1, quantity=0)"
            if action != confirm_action:
                logger.write(
                    "[FastLLM] Rewrote action into confirming object dialogue: "
                    f"{action} -> {confirm_action}"
                )
                return VLLMDecision(
                    action=confirm_action,
                    reason="object_dialogue_dismiss_fix",
                    escalate=False,
                )

        if sleep_task and current_menu_type in {"", "no menu"} and choose_option_index is not None and bed_direction:
            corrected = f'interact(direction="{bed_direction}")'
            logger.write(
                "[FastLLM] Rewrote sleep choose_option() with no open menu into bed interact: "
                f"{action} -> {corrected}"
            )
            return VLLMDecision(
                action=corrected,
                reason="sleep_bed_interact_fix",
                escalate=False,
            )

        if (
            decision_reason.startswith("parse_fallback:")
            and (
                zero_progress_streak >= 1
                or repeated_action_streak >= 2
                or position_issue_detected
            )
        ):
            logger.write(
                "[FastLLM] Escalating parse fallback under unstable execution feedback: "
                f"{action} | zero_progress_streak={zero_progress_streak}, "
                f"repeated_action_streak={repeated_action_streak}, "
                f"position_issue_detected={position_issue_detected}"
            )
            return VLLMDecision(
                action="",
                reason="parse_fallback_under_instability",
                escalate=True,
            )

        if action.startswith("move("):
            action_move = cls._parse_move_components(action)
            suggested_move = cls._parse_move_components(suggested_action)
            action_is_noop_move = action_move == (0, 0)
            suggested_is_noop_move = suggested_move == (0, 0)
            if action_is_noop_move:
                if cls._selected_item_requires_interact(selected_item_name):
                    alternative_directions = cls._collect_valid_placeable_directions(
                        game_state=game_state,
                        item_name=selected_item_name,
                        invalid_direction="",
                    )
                    if alternative_directions:
                        corrected = f'interact(direction="{alternative_directions[0]}")'
                        logger.write(
                            "[FastLLM] Rewrote no-op move into adjacent placeable interact: "
                            f"{action} -> {corrected} | selected={selected_item_name or '(none)'}"
                        )
                        return VLLMDecision(
                            action=corrected,
                            reason="noop_move_placeable_fix",
                            escalate=False,
                        )

                selected_tool_name = cls._normalize_tool_name(selected_item_name)
                if selected_tool_name:
                    alternative_direction = cls._find_alternative_tool_use_direction(
                        game_state=game_state,
                        tool_name=selected_tool_name,
                        invalid_direction="",
                    )
                    if alternative_direction:
                        corrected = f'use(direction="{alternative_direction}")'
                        logger.write(
                            "[FastLLM] Rewrote no-op move into adjacent tool use: "
                            f"{action} -> {corrected} | selected={selected_tool_name}"
                        )
                        return VLLMDecision(
                            action=corrected,
                            reason="noop_move_tool_fix",
                            escalate=False,
                        )

                if suggested_action and suggested_action != action and not suggested_is_noop_move:
                    logger.write(
                        "[FastLLM] Reverted no-op move to grounded BigBrain suggestion: "
                        f"{action} -> {suggested_action}"
                    )
                    return cls._validate_decision_against_state(
                        VLLMDecision(
                            action=suggested_action,
                            reason="noop_move_revert",
                            escalate=False,
                        ),
                        suggestion,
                        game_state,
                    )

                logger.write(
                    "[FastLLM] Escalating no-op move with no grounded fallback: "
                    f"{action}"
                )
                return VLLMDecision(
                    action="",
                    reason="noop_move",
                    escalate=True,
                )

        menu_action = cls._parse_menu_action(action)
        if menu_action is not None:
            option, menu_name = menu_action
            if sleep_task and option == "close" and current_menu_type in {"", "no menu"} and bed_direction:
                corrected = f'interact(direction="{bed_direction}")'
                logger.write(
                    "[FastLLM] Rewrote sleep menu-close with no open menu into bed interact: "
                    f"{action} -> {corrected}"
                )
                return VLLMDecision(
                    action=corrected,
                    reason="sleep_bed_interact_fix",
                    escalate=False,
                )
            if (
                sleep_task
                and option == "close"
                and current_menu_type == "dialoguebox"
                and (bed_direction or cls._menu_contains_sleep_prompt(current_menu))
            ):
                corrected = "choose_option(option_index=1, quantity=0)"
                logger.write(
                    "[FastLLM] Rewrote sleep menu-close on dialogue into confirm: "
                    f"{action} -> {corrected}"
                )
                return VLLMDecision(
                    action=corrected,
                    reason="sleep_dialogue_confirm_fix",
                    escalate=False,
                )
            supported_menu = option == "close" or (
                option == "open"
                and menu_name in {"map", "inventory", "current_menu", "crafting"}
            )
            if not supported_menu:
                if suggested_action and suggested_action != action:
                    logger.write(
                        "[FastLLM] Reverted unsupported menu() action to suggestion: "
                        f"{action} -> {suggested_action}"
                    )
                    return cls._validate_decision_against_state(
                        VLLMDecision(
                            action=suggested_action,
                            reason="unsupported_menu_revert",
                            escalate=False,
                        ),
                        suggestion,
                        game_state,
                    )
                logger.write(
                    "[FastLLM] Escalating unsupported menu() action with no grounded fallback: "
                    f"{action}"
                )
                return VLLMDecision(
                    action="",
                    reason="unsupported_menu_action",
                    escalate=True,
                )

        if cls._parse_skill_name(suggested_action) in cls._TERMINAL_COMPOSITE_SKILLS and action != suggested_action:
            composite_allowed, composite_reason = cls._composite_override_is_justified(
                suggested_action=suggested_action,
                action=action,
                game_state=game_state,
            )
            if not composite_allowed:
                logger.write(
                    "[FastLLM] Reverted ungrounded override of BigBrain composite skill: "
                    f"{action} -> {suggested_action}"
                )
                return VLLMDecision(
                    action=suggested_action,
                    reason=f"follow_big_brain_composite:{composite_reason}",
                    escalate=False,
                )

        if suggested_action.startswith("move(") and action.startswith("move("):
            move_allowed, move_reason = cls._move_override_is_justified(
                suggested_action=suggested_action,
                action=action,
                game_state=game_state,
            )
            if not move_allowed:
                logger.write(
                    "[FastLLM] Reverted unjustified move-to-move override of BigBrain move: "
                    f"{action} -> {suggested_action}"
                )
                return VLLMDecision(
                    action=suggested_action,
                    reason=f"follow_big_brain_move:{move_reason}",
                    escalate=False,
                )

        if (
            action.startswith("move(")
            and suggested_action
            and suggested_action.startswith("use(")
            and cls._should_preserve_hay_forage_use_suggestion(
                game_state=game_state,
                suggested_action=suggested_action,
                selected_item_name=selected_item_name,
            )
        ):
            logger.write(
                "[FastLLM] Reverted hay/grass forage move override to grounded Scythe use suggestion: "
                f"{action} -> {suggested_action}"
            )
            return VLLMDecision(
                action=suggested_action,
                reason="follow_big_brain_hay_scythe_use",
                escalate=False,
            )

        if suggested_action.startswith("move(") and not action.startswith("move("):
            suggested_move = cls._parse_move_components(suggested_action)
            if suggested_move == (0, 0):
                grounded_local_recovery = cls._build_invalidated_suggestion_local_recovery(
                    game_state=game_state,
                    suggestion_action=suggested_action,
                    selected_item_name=selected_item_name,
                    inventory=inventory,
                    toolbar_information=toolbar_information,
                )
                if grounded_local_recovery and grounded_local_recovery == action:
                    logger.write(
                        "[FastLLM] Allowing grounded local override of BigBrain no-op move: "
                        f"{action} | suggestion={suggested_action}"
                    )
                    return VLLMDecision(
                        action=action,
                        reason="noop_move_grounded_override",
                        escalate=False,
                    )

            suggested_move_blocker = cls._get_single_axis_move_blocker(
                suggested_action,
                game_state,
            )
            choose_slot = cls._extract_choose_item_slot(action)
            slot_item_name = inventory_slot_map.get(choose_slot, "") if choose_slot is not None else ""
            allow_inventory_setup_override = bool(
                choose_slot is not None
                and slot_item_name
                and not cls._slot_is_explicitly_empty(slot_item_name)
                and (
                    cls._task_prefers_inventory_setup(game_state)
                    or (
                        cls._selected_item_requires_interact(slot_item_name)
                        and not cls._selected_item_requires_interact(selected_item_name)
                    )
                )
            )
            allow_task_critical_tool_setup_override = bool(
                choose_slot is not None
                and cls._task_critical_choose_item_override_before_move(
                    game_state=game_state,
                    choose_slot=choose_slot,
                    chosen_item_name=slot_item_name,
                    selected_item_name=selected_item_name,
                    decision_reason=decision.reason,
                )
            )
            if (
                suggested_move_blocker
                and allow_inventory_setup_override
            ):
                logger.write(
                    "[FastLLM] Allowing task-relevant inventory setup override for blocked BigBrain move: "
                    f"{action} | blocked_suggestion={suggested_action} | blocker={suggested_move_blocker}"
                )
                return VLLMDecision(
                    action=action,
                    reason="farm_ops_inventory_setup_fix",
                    escalate=False,
                )

            if suggested_move_blocker and allow_task_critical_tool_setup_override:
                logger.write(
                    "[FastLLM] Allowing task-critical tool setup override for blocked BigBrain move: "
                    f"{action} | blocked_suggestion={suggested_action} | blocker={suggested_move_blocker}"
                )
                return VLLMDecision(
                    action=action,
                    reason="tool_setup_before_blocked_move_fix",
                    escalate=False,
                )

            if allow_inventory_setup_override:
                logger.write(
                    "[FastLLM] Allowing task-relevant inventory setup override before BigBrain move: "
                    f"{action} | suggestion={suggested_action}"
                )
                return VLLMDecision(
                    action=action,
                    reason="inventory_setup_before_move_fix",
                    escalate=False,
                )

            if allow_task_critical_tool_setup_override:
                logger.write(
                    "[FastLLM] Allowing task-critical tool setup override before BigBrain move: "
                    f"{action} | suggestion={suggested_action}"
                )
                return VLLMDecision(
                    action=action,
                    reason="tool_setup_before_move_fix",
                    escalate=False,
                )

            directional_skill = cls._parse_directional_skill(action)
            if directional_skill:
                skill_name, direction = directional_skill
                target_obj, required_tool = cls._get_directional_target(game_state, direction)
                selected_tool_name = cls._normalize_tool_name(selected_item_name)
                if suggested_move == (0, 0):
                    corrected = cls._build_invalidated_suggestion_local_recovery(
                        game_state=game_state,
                        suggestion_action=suggested_action,
                        selected_item_name=selected_item_name,
                        inventory=inventory,
                        toolbar_information=toolbar_information,
                    )
                    if corrected and corrected != suggested_action:
                        logger.write(
                            "[FastLLM] Recovered grounded override of BigBrain no-op move: "
                            f"{suggested_action} -> {corrected}"
                        )
                        return cls._validate_decision_against_state(
                            VLLMDecision(
                                action=corrected,
                                reason="noop_move_grounded_override",
                                escalate=False,
                            ),
                            suggestion,
                            game_state,
                        )
                if not target_obj:
                    logger.write(
                        "[FastLLM] Reverted non-grounded override of BigBrain move: "
                        f"{action} -> {suggested_action} | no explicit target in {direction}"
                    )
                    return VLLMDecision(
                        action=suggested_action,
                        reason="follow_big_brain_move:no_target",
                        escalate=False,
                    )
                allow_hoe_use_override = bool(
                    skill_name == "use"
                    and selected_tool_name == "Hoe"
                    and (
                        cls._is_valid_hoe_target(target_obj)
                        or cls._is_allowed_empty_hoe_target(game_state, target_obj, direction)
                    )
                )
                if skill_name == "use" and not required_tool and not allow_hoe_use_override:
                    logger.write(
                        "[FastLLM] Reverted invalid tool-use override of BigBrain move: "
                        f"{action} -> {suggested_action} | target={target_obj}"
                    )
                    return VLLMDecision(
                        action=suggested_action,
                        reason="follow_big_brain_move:invalid_use",
                        escalate=False,
                    )
                if skill_name == "interact" and required_tool:
                    logger.write(
                        "[FastLLM] Reverted obstacle-interact override of BigBrain move: "
                        f"{action} -> {suggested_action} | target={target_obj}"
                    )
                    return VLLMDecision(
                        action=suggested_action,
                        reason="follow_big_brain_move:tool_target",
                        escalate=False,
                    )
            elif (
                zero_progress_streak < 2
                and repeated_action_streak < 2
                and not position_issue_detected
                and current_menu_type in {"", "no menu"}
            ):
                logger.write(
                    "[FastLLM] Reverted ungrounded override of BigBrain move: "
                    f"{action} -> {suggested_action}"
                )
                return VLLMDecision(
                    action=suggested_action,
                    reason="follow_big_brain_move:default",
                    escalate=False,
                )

        if (
            action.startswith("move(")
            and suggested_action
            and not suggested_action.startswith("move(")
            and decision_reason.startswith("parse_fallback")
        ):
            directional_suggestion = cls._parse_directional_skill(suggested_action)
            grounded_non_move_suggestion = False

            if directional_suggestion:
                skill_name, direction = directional_suggestion
                if skill_name == "use":
                    grounded_non_move_suggestion = cls._use_suggestion_is_grounded_fallback(
                        game_state=game_state,
                        suggested_action=suggested_action,
                        selected_tool_name=selected_item_name,
                    )
                elif skill_name == "interact":
                    target_obj, required_tool = cls._get_directional_target(game_state, direction)
                    if cls._selected_item_requires_interact(selected_item_name):
                        grounded_non_move_suggestion = bool(target_obj) and cls._is_valid_placeable_target(
                            game_state,
                            direction,
                            selected_item_name,
                            target_obj,
                        )
                    else:
                        grounded_non_move_suggestion = bool(target_obj) and not required_tool
            elif suggested_action.startswith(("choose_item(", "choose_option(", "menu(", "craft(")):
                grounded_non_move_suggestion = True

            if grounded_non_move_suggestion:
                logger.write(
                    "[FastLLM] Reverted move override over grounded BigBrain non-move suggestion: "
                    f"{action} -> {suggested_action}"
                )
                return cls._validate_decision_against_state(
                    VLLMDecision(
                        action=suggested_action,
                        reason="follow_big_brain_non_move:grounded",
                        escalate=False,
                    ),
                    suggestion,
                    game_state,
                )

        if action.startswith("move("):
            task_text = cls._task_context_text(game_state)
            selected_item_name = (
                cls._extract_selected_item_name(gathered)
                or cls._extract_selected_item_name_from_toolbar(toolbar_information)
            )
            immediate_progress_override = ""
            clear_profile = build_clear_task_profile(task_text)
            if clear_profile:
                immediate_progress_override = cls._build_local_clear_recovery_action(
                    game_state=game_state,
                    selected_item_name=selected_item_name,
                    inventory=inventory,
                    toolbar_information=toolbar_information,
                )
            if (
                immediate_progress_override.startswith("use(")
                and cls._current_tile_is_hard_structure(game_state)
            ):
                immediate_progress_override = ""
            if (
                immediate_progress_override.startswith("use(")
                and cls._is_tilling_or_digging_context(game_state)
                and (
                    zero_progress_streak >= 1
                    or repeated_action_streak >= 2
                    or position_issue_detected
                    or execution_has_no_confirmation(last_exec_info)
                    or recent_no_progress_productive_action
                )
            ):
                if recent_no_progress_productive_action:
                    logger.write(
                        "[FastLLM] Suppressed low-value move -> Hoe use replacement after recent "
                        "no-progress productive action: "
                        f"{action} -> {immediate_progress_override} | last_action={last_action_text}"
                    )
                immediate_progress_override = ""
            if immediate_progress_override.startswith("use("):
                logger.write(
                    "[FastLLM] Replaced low-value move with immediate local progress action: "
                    f"{action} -> {immediate_progress_override}"
                )
                return VLLMDecision(
                    action=immediate_progress_override,
                    reason="immediate_local_progress_fix",
                    escalate=False,
                )

            blocker = cls._get_single_axis_move_blocker(action, game_state)
            if blocker:
                if cls._should_keep_same_axis_progress_move(action, game_state, blocker):
                    logger.write(
                        "[FastLLM] Kept same-axis cultivation progress move despite adjacent structure blocker: "
                        f"{action} | blocker={blocker}"
                    )
                    return VLLMDecision(
                        action=action,
                        reason="same_axis_progress_keep",
                        escalate=False,
                    )
                move_direction = cls._parse_move_direction(action)
                if (
                    sleep_task
                    and move_direction
                    and "bed" in str(blocker).strip().lower()
                    and current_menu_type in {"", "no menu"}
                ):
                    logger.write(
                        "[FastLLM] Preserving sleep move into bed tile instead of rewriting to interact: "
                        f"{action} | blocker={blocker}"
                    )
                    return decision
                if cls._should_preserve_navigation_anchor_move(
                    action_text=action,
                    game_state=game_state,
                    blocker=blocker,
                ):
                    logger.write(
                        "[FastLLM] Preserving navigation anchor move into doorway/entrance tile: "
                        f"{action} | blocker={blocker}"
                    )
                    return decision
                adjacent_interact = ""
                if current_menu_type in {"", "no menu"} and not (
                    sleep_task and "bed" in str(blocker).strip().lower()
                ):
                    normalized_selected_tool = cls._normalize_tool_name(selected_item_name)
                    normalized_blocker = str(blocker or "").strip().lower()
                    if (
                        normalized_selected_tool == "Watering Can"
                        and "pet bowl" in normalized_blocker
                    ):
                        logger.write(
                            "[FastLLM] Preserving pet-bowl approach move instead of rewriting to interact: "
                            f"{action} | blocker={blocker}"
                        )
                        return decision
                    adjacent_interact = cls._build_adjacent_blocker_interact(
                        action_text=action,
                        game_state=game_state,
                        blocker=blocker,
                    )
                if adjacent_interact:
                    logger.write(
                        "[FastLLM] Rewrote blocked adjacent actionable tile move into interact: "
                        f"{action} -> {adjacent_interact} | blocker={blocker}"
                    )
                    return VLLMDecision(
                        action=adjacent_interact,
                        reason="adjacent_blocker_interact_fix",
                        escalate=False,
                    )
                if cls._is_tilling_or_digging_context(game_state):
                    till_recovery = cls._build_local_tilling_recovery_action(
                        game_state=game_state,
                        selected_item_name=selected_item_name,
                        inventory=inventory,
                        toolbar_information=toolbar_information,
                    )
                    till_recovery_move = cls._parse_move_components(till_recovery)
                    till_recovery_is_local = bool(
                        till_recovery.startswith("use(")
                        or (
                            till_recovery_move is not None
                            and abs(int(till_recovery_move[0])) + abs(int(till_recovery_move[1])) <= 2
                        )
                    )
                    if till_recovery and till_recovery != action and till_recovery_is_local:
                        logger.write(
                            "[FastLLM] Rewrote blocked cultivation move into grounded till recovery: "
                            f"{action} -> {till_recovery} | blocker={blocker}"
                        )
                        return VLLMDecision(
                            action=till_recovery,
                            reason="blocked_tilling_recovery",
                            escalate=False,
                        )
                route_recovery = cls._build_local_route_recovery_action(
                    game_state=game_state,
                )
                if route_recovery and route_recovery != action:
                    logger.write(
                        "[FastLLM] Rewrote blocked move into route-aware recovery action: "
                        f"{action} -> {route_recovery} | blocker={blocker}"
                    )
                    return VLLMDecision(
                        action=route_recovery,
                        reason="blocked_route_recovery",
                        escalate=False,
                    )
                corrected = cls._build_structure_blocked_move_recovery(
                    action_text=action,
                    game_state=game_state,
                    blocker=blocker,
                )
                if corrected:
                    logger.write(
                        "[FastLLM] Rewrote blocked structure move into routed recovery move: "
                        f"{action} -> {corrected} | blocker={blocker}"
                    )
                    return VLLMDecision(
                        action=corrected,
                        reason="blocked_structure_reroute",
                        escalate=False,
                    )
                logger.write(
                    "[FastLLM] Escalating move() into explicit blocker: "
                    f"{action} | blocker={blocker}"
                )
                return VLLMDecision(
                    action="",
                    reason="move_target_blocked",
                    escalate=True,
                )

        directional_skill = cls._parse_directional_skill(action)
        if directional_skill and directional_skill[0] == "use" and cls._selected_item_requires_interact(selected_item_name):
            _, direction = directional_skill
            target_obj, required_tool = cls._get_directional_target(game_state, direction)
            if cls._selected_item_is_seed(selected_item_name) or cls._selected_item_is_fertilizer(selected_item_name):
                if target_obj and cls._is_valid_placeable_target(game_state, direction, selected_item_name, target_obj):
                    corrected = f'interact(direction="{direction}")'
                    logger.write(
                        "[FastLLM] Rewrote cultivation placeable item use() into interact(): "
                        f"{action} -> {corrected} | selected={selected_item_name or '(none)'} | target={target_obj}"
                    )
                    return VLLMDecision(
                        action=corrected,
                        reason="placeable_item_use_fix",
                        escalate=False,
                    )
                corrected = cls._build_local_placeable_recovery_action(
                    game_state=game_state,
                    item_name=selected_item_name,
                    invalid_direction=direction,
                )
                if corrected and corrected != action:
                    logger.write(
                        "[FastLLM] Recovered cultivation placeable item use() with local reroute: "
                        f"{action} -> {corrected} | selected={selected_item_name or '(none)'} | target={target_obj}"
                    )
                    return VLLMDecision(
                        action=corrected,
                        reason="cultivation_placeable_requires_interact",
                        escalate=False,
                    )
                logger.write(
                    "[FastLLM] Escalating cultivation placeable item use() with no grounded interact fix: "
                    f"{action} | selected={selected_item_name or '(none)'} | target={target_obj}"
                )
                return VLLMDecision(
                    action="",
                    reason="cultivation_placeable_requires_interact",
                    escalate=True,
                )
            if required_tool:
                tool_slot = cls._find_tool_slot(inventory, required_tool)
                if tool_slot is None:
                    tool_slot = cls._find_tool_slot_in_toolbar_text(toolbar_information, required_tool)
                if tool_slot is not None:
                    corrected = f"choose_item(slot_index={tool_slot})"
                    logger.write(
                        "[FastLLM] Corrected invalid use() with placeable item: "
                        f"{action} -> {corrected} | "
                        f"selected={selected_item_name or '(none)'}, "
                        f"target={target_obj}"
                    )
                    return VLLMDecision(
                        action=corrected,
                        reason=f"placeable_item_tool_fix:{required_tool}",
                        escalate=False,
                    )
                if suggested_action and suggested_action != action:
                    logger.write(
                        "[FastLLM] Reverted invalid use() with placeable item to suggestion: "
                        f"{action} -> {suggested_action} | "
                        f"selected={selected_item_name or '(none)'}"
                    )
                    return VLLMDecision(
                        action=suggested_action,
                        reason="placeable_item_use_revert",
                        escalate=False,
                    )

            corrected = f'interact(direction="{direction}")'
            logger.write(
                "[FastLLM] Rewrote invalid use() with placeable item: "
                f"{action} -> {corrected} | "
                f"selected={selected_item_name or '(none)'}"
            )
            return VLLMDecision(
                action=corrected,
                reason="placeable_item_use_fix",
                escalate=False,
            )

        if directional_skill and directional_skill[0] == "interact" and cls._selected_item_requires_interact(selected_item_name):
            _, direction = directional_skill
            target_obj, _ = cls._get_directional_target(game_state, direction)
            if (
                cls._selected_item_is_fertilizer(selected_item_name)
                and action == last_action_text
                and target_obj
                and cls._is_valid_placeable_target(game_state, direction, selected_item_name, target_obj)
                and execution_has_no_confirmation(last_exec_info)
                and zero_progress_streak >= 1
            ):
                corrected = cls._build_local_placeable_recovery_action(
                    game_state=game_state,
                    item_name=selected_item_name,
                    invalid_direction=direction,
                )
                if corrected and corrected != action:
                    logger.write(
                        "[FastLLM] Recovered repeated no-confirmation fertilizer interact with local reroute: "
                        f"{action} -> {corrected}"
                    )
                    return VLLMDecision(
                        action=corrected,
                        reason="fertilize_target_effective_recovery",
                        escalate=False,
                    )
                logger.write(
                    "[FastLLM] Escalating repeated no-confirmation fertilizer interact for explicit replan: "
                    f"{action}"
                )
                return VLLMDecision(
                    action="",
                    reason="fertilize_target_not_effective",
                    escalate=True,
                )
            if (
                cls._selected_item_is_fertilizer(selected_item_name)
                and action == last_action_text
                and target_obj
                and cls._is_valid_placeable_target(game_state, direction, selected_item_name, target_obj)
                and numeric_task_progress_quantity is not None
                and numeric_task_progress_quantity > 0
            ):
                corrected = cls._build_local_placeable_recovery_action(
                    game_state=game_state,
                    item_name=selected_item_name,
                    invalid_direction=direction,
                )
                if corrected and corrected != action:
                    logger.write(
                        "[FastLLM] Shifted fertilizer placement to a new nearby tile after confirmed progress on the current patch: "
                        f"{action} -> {corrected}"
                    )
                    return VLLMDecision(
                        action=corrected,
                        reason="fertilize_progressive_patch_sweep",
                        escalate=False,
                    )
            if target_obj and not cls._is_valid_placeable_target(game_state, direction, selected_item_name, target_obj):
                if (
                    zero_progress_streak >= 2
                    or repeated_action_streak >= 2
                    or position_issue_detected
                ):
                    logger.write(
                        "[FastLLM] Escalating invalid cultivation interact() target after repeated failure: "
                        f"{action} | selected={selected_item_name or '(none)'} | target={target_obj}"
                    )
                    return VLLMDecision(
                        action="",
                        reason="placeable_item_invalid_target",
                        escalate=True,
                    )
                alternative_directions = cls._collect_valid_placeable_directions(
                    game_state=game_state,
                    item_name=selected_item_name,
                    invalid_direction=direction,
                )
                if (
                    len(alternative_directions) == 1
                    and zero_progress_streak < 1
                    and repeated_action_streak < 1
                    and not position_issue_detected
                ):
                    alternative_direction = alternative_directions[0]
                    corrected = f'interact(direction="{alternative_direction}")'
                    logger.write(
                        "[FastLLM] Rewrote invalid interact() with placeable item: "
                        f"{action} -> {corrected} | "
                        f"selected={selected_item_name or '(none)'}, "
                        f"target={target_obj}"
                    )
                    return VLLMDecision(
                        action=corrected,
                        reason="placeable_item_interact_target_fix",
                        escalate=False,
                    )
                if suggested_action and suggested_action != action:
                    logger.write(
                        "[FastLLM] Reverted invalid interact() with placeable item to suggestion: "
                        f"{action} -> {suggested_action} | "
                        f"selected={selected_item_name or '(none)'}, "
                        f"target={target_obj}"
                    )
                    return VLLMDecision(
                        action=suggested_action,
                        reason="placeable_item_interact_revert",
                        escalate=False,
                    )
                logger.write(
                    "[FastLLM] Escalating invalid interact() with placeable item and no safe local rewrite: "
                    f"{action} | selected={selected_item_name or '(none)'}, target={target_obj}"
                )
                return VLLMDecision(
                    action="",
                    reason="placeable_item_invalid_target",
                    escalate=True,
                )

        if directional_skill and directional_skill[0] == "use":
            _, direction = directional_skill
            target_obj, required_tool = cls._get_directional_target(game_state, direction)
            selected_tool_name = cls._normalize_tool_name(selected_item_name)
            if (
                selected_tool_name == "Hoe"
            ):
                if cls._is_inside_house_for_cultivation(game_state):
                    corrected = cls._build_inside_house_exit_recovery_action(
                        game_state=game_state,
                    )
                    if corrected and corrected != action:
                        logger.write(
                            "[FastLLM] Rewrote indoor Hoe use() into house-exit recovery move: "
                            f"{action} -> {corrected}"
                        )
                        return VLLMDecision(
                            action=corrected,
                            reason="till_inside_house_exit_recovery",
                            escalate=False,
                        )
                if not target_obj:
                    return decision
                if cls._is_valid_hoe_target(target_obj):
                    return decision
                if cls._is_allowed_empty_hoe_target(game_state, target_obj, direction):
                    tilling_context = cls._is_tilling_or_digging_context(game_state)
                    if tilling_context and (
                        zero_progress_streak < 1
                        and repeated_action_streak < 2
                        and not position_issue_detected
                        and not execution_has_no_confirmation(last_exec_info)
                    ):
                        return decision
                    if (
                        action == suggested_action
                        and zero_progress_streak < 2
                        and repeated_action_streak < 2
                        and not execution_has_no_confirmation(last_exec_info)
                    ):
                        return decision
                    corrected = cls._build_local_tilling_recovery_action(
                        game_state=game_state,
                        selected_item_name=selected_item_name,
                        inventory=inventory,
                        toolbar_information=toolbar_information,
                        invalid_direction=direction,
                    )
                    if corrected and corrected != action:
                        logger.write(
                            "[FastLLM] Recovered repeated empty-tile Hoe use with local reposition: "
                            f"{action} -> {corrected} | target={target_obj}"
                        )
                        return VLLMDecision(
                            action=corrected,
                            reason="hoe_empty_target_recovery",
                            escalate=False,
                        )
                elif cls._is_empty_like_tile(target_obj) or cls._is_open_ground_tile(target_obj):
                    if (
                        not cls._task_context_text(game_state)
                        and zero_progress_streak < 1
                        and repeated_action_streak < 2
                        and not position_issue_detected
                        and not execution_has_no_confirmation(last_exec_info)
                    ):
                        return decision
                    corrected = cls._build_local_tilling_recovery_action(
                        game_state=game_state,
                        selected_item_name=selected_item_name,
                        inventory=inventory,
                        toolbar_information=toolbar_information,
                        invalid_direction=direction,
                    )
                    if corrected and corrected != action:
                        logger.write(
                            "[FastLLM] Recovered unsafe empty-tile Hoe use with local reposition: "
                            f"{action} -> {corrected} | target={target_obj}"
                        )
                        return VLLMDecision(
                            action=corrected,
                            reason="hoe_empty_target_recovery",
                            escalate=False,
                        )
                corrected = cls._build_local_tilling_recovery_action(
                    game_state=game_state,
                    selected_item_name=selected_item_name,
                    inventory=inventory,
                    toolbar_information=toolbar_information,
                    invalid_direction=direction,
                )
                if corrected and corrected != action:
                    logger.write(
                        "[FastLLM] Recovered invalid Hoe use() target with grounded local till route: "
                        f"{action} -> {corrected} | target={target_obj}"
                    )
                    return VLLMDecision(
                        action=corrected,
                        reason="hoe_invalid_target_recovery",
                        escalate=False,
                    )
                logger.write(
                    "[FastLLM] Escalating invalid Hoe use() target for explicit replan: "
                    f"{action} | target={target_obj}"
                )
                return VLLMDecision(
                    action="",
                    reason="hoe_invalid_target",
                    escalate=True,
                )
            elif (
                selected_tool_name == "Watering Can"
                and cls._is_watering_context(game_state)
                and target_obj
                and not cls._is_valid_watering_target(game_state, direction, target_obj)
            ):
                alternative_direction = cls._find_alternative_tool_use_direction(
                    game_state=game_state,
                    tool_name=selected_tool_name,
                    invalid_direction=direction,
                )
                if alternative_direction:
                    corrected = f'use(direction="{alternative_direction}")'
                    logger.write(
                        "[FastLLM] Rewrote invalid Watering Can use() target: "
                        f"{action} -> {corrected} | target={target_obj}"
                    )
                    return VLLMDecision(
                        action=corrected,
                        reason="watering_target_direction_fix",
                        escalate=False,
                    )
                logger.write(
                    "[FastLLM] Escalating invalid Watering Can use() target with no grounded fallback: "
                    f"{action} | target={target_obj}"
                )
                return VLLMDecision(
                    action="",
                    reason="watering_can_invalid_target",
                    escalate=True,
                )
            elif selected_tool_name in {"Pickaxe", "Axe", "Scythe"} and target_obj:
                clear_profile = build_clear_task_profile(task_text)
                target_is_clearable = bool(cls._classify_clearable_object(target_obj))
                if (
                    clear_profile
                    and target_is_clearable
                    and not clear_target_matches_profile(target_obj, clear_profile)
                ):
                    corrected = cls._build_local_clear_recovery_action(
                        game_state=game_state,
                        selected_item_name=selected_item_name,
                        inventory=inventory,
                        toolbar_information=toolbar_information,
                    )
                    if corrected and corrected != action:
                        logger.write(
                            "[FastLLM] Recovered clear-task tool use() that violates task target profile: "
                            f"{action} -> {corrected} | "
                            f"selected={selected_tool_name}, target={target_obj}"
                        )
                        return VLLMDecision(
                            action=corrected,
                            reason="clear_profile_target_recovery",
                            escalate=False,
                        )

                    if suggested_action and suggested_action != action:
                        suggested_directional_skill = cls._parse_directional_skill(suggested_action)
                        suggested_is_profile_aligned = True
                        if suggested_directional_skill and suggested_directional_skill[0] == "use":
                            suggested_target, _ = cls._get_directional_target(
                                game_state,
                                suggested_directional_skill[1],
                            )
                            if (
                                suggested_target
                                and cls._classify_clearable_object(suggested_target)
                                and not clear_target_matches_profile(suggested_target, clear_profile)
                            ):
                                suggested_is_profile_aligned = False

                        if suggested_is_profile_aligned and (
                            not suggested_directional_skill
                            or suggested_directional_skill[0] != "use"
                            or cls._use_suggestion_is_grounded_fallback(
                                game_state=game_state,
                                suggested_action=suggested_action,
                                selected_tool_name=selected_tool_name,
                            )
                        ):
                            logger.write(
                                "[FastLLM] Reverted clear-task tool use() that violates task target profile: "
                                f"{action} -> {suggested_action} | "
                                f"selected={selected_tool_name}, target={target_obj}"
                            )
                            return VLLMDecision(
                                action=suggested_action,
                                reason="clear_profile_target_revert",
                                escalate=False,
                            )

                    logger.write(
                        "[FastLLM] Escalating clear-task tool use() that violates task target profile with no grounded fallback: "
                        f"{action} | selected={selected_tool_name}, target={target_obj}"
                    )
                    return VLLMDecision(
                        action="",
                        reason="clear_profile_target_mismatch",
                        escalate=True,
                    )

                if not required_tool:
                    alternative_direction = cls._find_alternative_tool_use_direction(
                        game_state=game_state,
                        tool_name=selected_tool_name,
                        invalid_direction=direction,
                    )
                    if alternative_direction:
                        corrected = f'use(direction="{alternative_direction}")'
                        logger.write(
                            "[FastLLM] Rewrote invalid tool use() target: "
                            f"{action} -> {corrected} | "
                            f"selected={selected_tool_name}, "
                            f"target={target_obj}"
                        )
                        return VLLMDecision(
                            action=corrected,
                            reason="tool_target_direction_fix",
                            escalate=False,
                        )
                    if suggested_action and suggested_action != action:
                        if cls._use_suggestion_is_grounded_fallback(
                            game_state=game_state,
                            suggested_action=suggested_action,
                            selected_tool_name=selected_tool_name,
                        ):
                            logger.write(
                                "[FastLLM] Reverted invalid tool use() target to suggestion: "
                                f"{action} -> {suggested_action} | "
                                f"selected={selected_tool_name}, "
                                f"target={target_obj}"
                            )
                            return VLLMDecision(
                                action=suggested_action,
                                reason="tool_target_revert",
                                escalate=False,
                            )
                        logger.write(
                            "[FastLLM] Rejected invalid tool-use suggestion fallback: "
                            f"{suggested_action} | selected={selected_tool_name}, target={target_obj}"
                        )
                    logger.write(
                        "[FastLLM] Escalating invalid tool use() target with no grounded fallback: "
                        f"{action} | selected={selected_tool_name}, target={target_obj}"
                    )
                    return VLLMDecision(
                        action="",
                        reason=f"{selected_tool_name.lower()}_invalid_target",
                        escalate=True,
                    )
            if (
                target_obj
                and required_tool
                and selected_item_name.lower() != required_tool.lower()
            ):
                tool_slot = cls._find_tool_slot(inventory, required_tool)
                if tool_slot is None:
                    tool_slot = cls._find_tool_slot_in_toolbar_text(toolbar_information, required_tool)
                if tool_slot is not None:
                    corrected = f"choose_item(slot_index={tool_slot})"
                    logger.write(
                        "[FastLLM] Corrected use() tool mismatch: "
                        f"{action} -> {corrected} | "
                        f"target={target_obj} requires {required_tool}, "
                        f"selected={selected_item_name or '(none)'}"
                    )
                    return VLLMDecision(
                        action=corrected,
                        reason=f"tool_mismatch_fix:{required_tool}",
                        escalate=False,
                    )

        choose_slot = cls._extract_choose_item_slot(action)
        if choose_slot is None:
            return decision

        if (
            suggested_action.startswith("interact(")
            and cls._selected_item_requires_interact(selected_item_name)
            and action != suggested_action
        ):
            logger.write(
                "[FastLLM] Reverted unnecessary choose_item() override while a placeable item is "
                f"already selected: {action} -> {suggested_action} | "
                f"selected={selected_item_name or '(none)'}"
            )
            return VLLMDecision(
                action=suggested_action,
                reason="placeable_item_already_selected",
                escalate=False,
            )

        slot_visible_in_current_facts = choose_slot in inventory_slot_map
        observed_slot_item = inventory_slot_map.get(choose_slot)
        if not slot_visible_in_current_facts:
            if 0 <= choose_slot <= 11:
                logger.write(
                    "[FastLLM] Allowing choose_item() for toolbar slot without explicit contradictory slot facts: "
                    f"{action} | slot={choose_slot}"
                )
                return decision
            route_recovery = cls._build_local_route_recovery_action(game_state=game_state)
            if route_recovery and route_recovery != action and route_recovery != suggested_action:
                logger.write(
                    "[FastLLM] Recovered from choose_item() targeting a non-visible slot via route-aware action: "
                    f"{action} -> {route_recovery} | slot={choose_slot}"
                )
                return VLLMDecision(
                    action=route_recovery,
                    reason="choose_item_unknown_slot_route_recovery",
                    escalate=False,
                )
            if suggested_action and suggested_action != action:
                logger.write(
                    "[FastLLM] Reverted choose_item() targeting a slot that is not present in current Inventory/Toolbar facts: "
                    f"{action} -> {suggested_action} | slot={choose_slot}"
                )
                return cls._validate_decision_against_state(
                    VLLMDecision(
                        action=suggested_action,
                        reason="choose_item_unknown_slot_revert",
                        escalate=False,
                    ),
                    suggestion,
                    game_state,
                )
            logger.write(
                "[FastLLM] Escalating choose_item() targeting a slot that is not present in current Inventory/Toolbar facts: "
                f"{action} | slot={choose_slot}"
            )
            return VLLMDecision(
                action="",
                reason="choose_item_unknown_slot",
                escalate=True,
            )
        normalized_observed_tool = cls._normalize_tool_name(observed_slot_item)
        normalized_selected_tool = cls._normalize_tool_name(selected_item_name)
        if cls._slot_is_explicitly_empty(observed_slot_item):
            route_recovery = cls._build_local_route_recovery_action(game_state=game_state)
            if route_recovery and route_recovery != action and route_recovery != suggested_action:
                logger.write(
                    "[FastLLM] Recovered from choose_item() targeting an empty slot via route-aware action: "
                    f"{action} -> {route_recovery} | slot={choose_slot}"
                )
                return VLLMDecision(
                    action=route_recovery,
                    reason="choose_item_empty_slot_route_recovery",
                    escalate=False,
                )
            if suggested_action and suggested_action != action:
                logger.write(
                    "[FastLLM] Reverted choose_item() targeting an explicitly empty slot to suggestion: "
                    f"{action} -> {suggested_action} | slot={choose_slot}"
                )
                return cls._validate_decision_against_state(
                    VLLMDecision(
                        action=suggested_action,
                        reason="choose_item_empty_slot_revert",
                        escalate=False,
                    ),
                    suggestion,
                    game_state,
                )
            logger.write(
                "[FastLLM] Escalating choose_item() targeting an explicitly empty slot: "
                f"{action} | slot={choose_slot}"
            )
            return VLLMDecision(
                action="",
                reason="choose_item_empty_slot",
                escalate=True,
            )

        suggested_directional_skill = cls._parse_directional_skill(suggested_action)
        if (
            observed_slot_item
            and selected_item_name
            and str(observed_slot_item).strip().lower() != str(selected_item_name).strip().lower()
            and suggested_directional_skill is not None
            and (
                (
                    suggested_directional_skill[0] == "use"
                    and normalized_selected_tool in {"Hoe", "Watering Can"}
                )
                or (
                    suggested_directional_skill[0] == "interact"
                    and cls._selected_item_requires_interact(selected_item_name)
                )
            )
        ):
            corrected = cls._build_invalidated_suggestion_local_recovery(
                game_state=game_state,
                suggestion_action=suggested_action,
                selected_item_name=selected_item_name,
                inventory=inventory,
                toolbar_information=toolbar_information,
            )
            if corrected and corrected != action:
                logger.write(
                    "[FastLLM] Reverted choose_item() away from task-critical selected item: "
                    f"{action} -> {corrected} | selected={selected_item_name or '(none)'}"
                )
                return cls._validate_decision_against_state(
                    VLLMDecision(
                        action=corrected,
                        reason="task_critical_item_recovery",
                        escalate=False,
                    ),
                    suggestion,
                    game_state,
                )
            if suggested_action and suggested_action != action:
                logger.write(
                    "[FastLLM] Reverted choose_item() away from task-critical selected item to suggestion: "
                    f"{action} -> {suggested_action} | selected={selected_item_name or '(none)'}"
                )
                return cls._validate_decision_against_state(
                    VLLMDecision(
                        action=suggested_action,
                        reason="task_critical_item_revert",
                        escalate=False,
                    ),
                    suggestion,
                    game_state,
                )

        if (
            suggested_action.startswith("interact(")
            and cls._selected_item_requires_interact(selected_item_name)
            and action != suggested_action
        ):
            logger.write(
                "[FastLLM] Reverted unnecessary choose_item() override while a placeable item is "
                f"already selected: {action} -> {suggested_action} | "
                f"selected={selected_item_name or '(none)'}"
            )
            return VLLMDecision(
                action=suggested_action,
                reason="placeable_item_already_selected",
                escalate=False,
            )

        expected_tool = cls._extract_explicit_tool_name(decision.reason)
        if expected_tool:
            expected_slot = cls._find_tool_slot(inventory, expected_tool)
            if expected_slot is None:
                expected_slot = cls._find_tool_slot_in_toolbar_text(toolbar_information, expected_tool)
            if expected_slot is not None and expected_slot != choose_slot:
                corrected = f"choose_item(slot_index={expected_slot})"
                logger.write(
                    "[FastLLM] Corrected choose_item() slot mismatch: "
                    f"{action} -> {corrected} | "
                    f"reason expects {expected_tool}"
                )
                return VLLMDecision(
                    action=corrected,
                    reason=f"slot_fix:{expected_tool}",
                    escalate=False,
                )

        if selected_slot is not None and choose_slot == selected_slot:
            immediate_use = cls._build_immediate_use_if_aligned(game_state)
            if immediate_use:
                logger.write(
                    "[FastLLM] Replaced redundant choose_item() with immediate use: "
                    f"{action} -> {immediate_use}"
                )
                return VLLMDecision(
                    action=immediate_use,
                    reason="redundant_choose_item_fix",
                    escalate=False,
                )

            fallback_action = str(suggestion.get("action", "") or "").strip()
            if fallback_action and fallback_action != action:
                logger.write(
                    "[FastLLM] Replaced redundant choose_item() with suggestion fallback: "
                    f"{action} -> {fallback_action}"
                )
                return VLLMDecision(
                    action=fallback_action,
                    reason="redundant_choose_item_fallback",
                    escalate=False,
                )

        return decision

    @classmethod
    def _build_front_obstacle_context(
        cls,
        game_state: Optional[Dict[str, Any]],
        execution_log: List[Dict[str, Any]],
        hint_action: Any = None,
    ) -> Dict[str, str]:
        context = {
            "front_tile_summary": "(none)",
            "blocked_recovery_hint": "",
            "blocked_override_action": "",
            "current_blocker_signature": "(none)",
            "nearest_grounded_target_summary": "(none)",
        }
        if not isinstance(game_state, dict):
            return context

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            return context
        exits_text = str(gathered.get("exits") or game_state.get("exits") or "").strip()

        explicit_summary = (
            game_state.get("front_tile_summary")
            or gathered.get("front_tile_summary")
            or ""
        )
        explicit_hint = (
            game_state.get("blocked_recovery_hint")
            or gathered.get("blocked_recovery_hint")
            or ""
        )
        if explicit_summary:
            context["front_tile_summary"] = str(explicit_summary)
        if explicit_hint:
            context["blocked_recovery_hint"] = str(explicit_hint)
        explicit_signature = (
            game_state.get("current_blocker_signature")
            or gathered.get("current_blocker_signature")
            or ""
        )
        if explicit_signature:
            context["current_blocker_signature"] = str(explicit_signature)
        explicit_nearest = (
            game_state.get("nearest_grounded_target_summary")
            or gathered.get("nearest_grounded_target_summary")
            or ""
        )
        if explicit_nearest:
            context["nearest_grounded_target_summary"] = str(explicit_nearest)
        elif exits_text:
            context["nearest_grounded_target_summary"] = (
                f"Nearest grounded route context: {exits_text}."
            )

        surroundings_map, _ = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return context

        latest_entry = execution_log[-1] if execution_log else {}
        latest_action = str(latest_entry.get("action", "") or "").strip()
        latest_errors = str(latest_entry.get("errors_info", "") or "").lower()
        latest_blocked_move = (
            latest_action.startswith("move(")
            and (
                latest_entry.get("success") is False
                or "blocked by an obstacle" in latest_errors
                or "path is likely blocked" in latest_errors
            )
        )

        blocked_direction = cls._parse_move_direction(latest_action) if latest_blocked_move else None
        facing_direction = str(gathered.get("facing_direction", "") or "").strip().lower()
        hinted_direction = cls._parse_move_direction(hint_action)
        if hinted_direction is None:
            hinted_skill = cls._parse_directional_skill(hint_action)
            if hinted_skill is not None:
                hinted_direction = hinted_skill[1]
        direction = blocked_direction or facing_direction or hinted_direction
        relative_cell = cls._direction_to_relative(direction)
        if relative_cell is None:
            return context

        front_obj = surroundings_map.get(relative_cell)
        if front_obj is None:
            context["front_tile_summary"] = (
                f'Front tile {relative_cell} toward {direction}: no explicit object in current surroundings.'
            )
            return context

        clearable = cls._classify_clearable_object(front_obj)
        context["current_blocker_signature"] = (
            f"Immediate tile toward {direction} at {relative_cell}: {front_obj}."
        )
        if clearable:
            context["front_tile_summary"] = (
                f'Front tile {relative_cell} toward {direction}: {front_obj}. '
                f'Clearable with {clearable["tool"]}.'
            )
        else:
            context["front_tile_summary"] = (
                f'Front tile {relative_cell} toward {direction}: {front_obj}. '
                "Not an obvious clearable obstacle."
            )

        if not latest_blocked_move:
            return context

        reroute_action = cls._build_structure_blocked_move_recovery(
            action_text=latest_action,
            game_state=game_state,
            blocker=front_obj,
        )

        if not clearable:
            context["blocked_recovery_hint"] = (
                "Latest move was blocked by a grounded obstacle or building directly ahead. "
                "Do not repeat the same-direction move; route around it or pick a different nearby waypoint first."
            )
            if reroute_action and reroute_action != latest_action:
                context["blocked_override_action"] = reroute_action
            return context

        context["blocked_recovery_hint"] = (
            f'Latest move was blocked by {clearable["label"]} directly ahead. '
            "Prefer a grounded reroute or nearby waypoint first. "
            f'Only switch to {clearable["tool"]} and clear it if rerouting is not grounded or the reroute also fails.'
        )
        if reroute_action and reroute_action != latest_action:
            context["blocked_override_action"] = reroute_action
            return context

        if cls._is_menu_open(gathered.get("current_menu") or gathered.get("CurrentMenuData")):
            return context

        toolbar_information = game_state.get("toolbar_information") or gathered.get("toolbar_information") or ""
        inventory = gathered.get("inventory") or game_state.get("inventory", [])
        selected_item_name = (
            cls._extract_selected_item_name(gathered)
            or cls._extract_selected_item_name_from_toolbar(toolbar_information)
        ).lower()
        direction_call = f'use(direction="{direction}")'
        if selected_item_name == clearable["tool"].lower():
            context["blocked_override_action"] = direction_call
            return context

        tool_slot = cls._find_tool_slot(inventory, clearable["tool"])
        if tool_slot is None:
            tool_slot = cls._find_tool_slot_in_toolbar_text(toolbar_information, clearable["tool"])
        if tool_slot is not None:
            context["blocked_override_action"] = f"choose_item(slot_index={tool_slot})"

        return context

    def decide(
        self,
        context_summary: str,
        suggestion: Dict[str, str],
        execution_log: List[Dict[str, Any]],
        mem0_reference: str,
        step: int,
        total_steps: int,
        skill_list: str,
        game_state: Optional[Dict[str, Any]] = None,
    ) -> VLLMDecision:
        """Ask the fast LLM to make an autonomous action decision.

        Uses enable_thinking=false for direct output without CoT overhead.

        Args:
            context_summary: Summary from big brain.
            suggestion: Current step suggestion {"action": ..., "reason": ...}.
            execution_log: List of execution log entries.
            mem0_reference: Memory reference text.
            step: Current step index.
            total_steps: Total suggestion steps.
            skill_list: Available skills text.
            game_state: Full game state dict (for template-based prompts).

        Returns:
            VLLMDecision with action/escalate.
        """
        front_obstacle_context = self._build_front_obstacle_context(
            game_state,
            execution_log,
            hint_action=suggestion.get("action", ""),
        )
        blocked_override_action = front_obstacle_context.get("blocked_override_action", "").strip()
        if blocked_override_action:
            logger.write(
                "[FastLLM] Blocked-move override: "
                f"{blocked_override_action} | "
                f"{front_obstacle_context.get('blocked_recovery_hint', '')}"
            )
            return VLLMDecision(
                action=blocked_override_action,
                reason="blocked_recovery",
                escalate=False,
            )

        if self.template and game_state:
            prompt = self._build_prompt_from_template(
                game_state,
                suggestion,
                execution_log,
                context_summary,
                mem0_reference,
                step,
                total_steps,
                skill_list,
            )
        else:
            prompt = self._build_prompt_legacy(
                context_summary, suggestion, execution_log,
                mem0_reference, step, total_steps, skill_list, game_state,
            )

        logger.write(
            f"[LLM_DIAG] >>> REQUEST | model={self.model} | mode=fastllm/nothinking"
            f" | prompt_text={len(prompt)}chars (~{len(prompt)//3}tok)"
        )
        logger.write(f"[LLM_DIAG]   prompt: {prompt[:200].replace(chr(10), ' ')}...")
        overall_started_at = time.time()
        self.last_effective_duration_s = 0.0
        retry_cfg = self._load_retry_config()
        max_retries = max(1, int(retry_cfg.get("max_retries", 10)))
        retry_interval_s = max(0.0, float(retry_cfg.get("retry_interval_s", 8.0)))

        key_name, api_key = self._get_next_api_key()
        attempt = 0
        last_attempt_duration_s = 0.0
        while True:
            attempt += 1
            attempt_started_at = time.time()
            try:
                slot_wait_timeout_s = get_llm_endpoint_wait_timeout(
                    self.model,
                    total_timeout_s=self.request_timeout_s,
                )
                with acquire_llm_endpoint_slot(
                    model_name=self.model,
                    purpose="little_brain",
                    logger_obj=logger,
                    timeout_s=slot_wait_timeout_s,
                ) as slot_info:
                    request_timeout_s = resolve_remaining_llm_request_timeout(
                        self.request_timeout_s,
                        slot_info,
                    )
                    increment_llm_call_counter("little_brain")
                    resp = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": self.model,
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": self.max_tokens,
                            "temperature": 0.1,
                            "chat_template_kwargs": {"enable_thinking": False},
                        },
                        timeout=request_timeout_s,
                    )
                resp.raise_for_status()
                data = resp.json()
                text = data["choices"][0]["message"]["content"].strip()

                last_attempt_duration_s = max(0.0, time.time() - attempt_started_at)
                self.last_effective_duration_s = last_attempt_duration_s
                retry_overhead_s = max(
                    0.0,
                    (time.time() - overall_started_at) - last_attempt_duration_s,
                )
                if retry_overhead_s > 0:
                    add_llm_retry_overhead(retry_overhead_s)
                usage = data.get("usage", {})
                logger.write(
                    f"[LLM_DIAG] <<< RESPONSE | model={self.model} | mode=fastllm"
                    f" | key={key_name}"
                    f" | duration={last_attempt_duration_s:.1f}s"
                    f" | response={len(text)}chars"
                    f" | tokens(prompt={usage.get('prompt_tokens', '?')}"
                    f" comp={usage.get('completion_tokens', '?')}"
                    f" total={usage.get('total_tokens', '?')})"
                )
                logger.write(f"[FastLLM] Raw response: {text}")
                decision = self._parse_response(text, suggestion)
                if (
                    isinstance(game_state, dict)
                    and decision.escalate
                    and str(decision.reason or "").strip() == "parse_fallback_no_suggestion"
                ):
                    corrected = self._build_autonomous_local_recovery_action(
                        game_state=game_state,
                    )
                    if corrected:
                        logger.warn(
                            "[FastLLM] Could not parse response with no fallback suggestion; "
                            f"using grounded local recovery: {corrected}"
                        )
                        decision = VLLMDecision(
                            action=corrected,
                            reason="parse_fallback_no_suggestion_local_recovery",
                            escalate=False,
                        )
                return self._validate_decision_against_state(
                    decision=decision,
                    suggestion=suggestion,
                    game_state=game_state,
                )
            except LLMEndpointThrottleTimeout as e:
                self.last_effective_duration_s = max(0.0, time.time() - attempt_started_at)
                logger.warn(
                    "[FastLLM] Local queue wait exhausted the little-brain budget "
                    f"after {time.time() - overall_started_at:.1f}s: {e}"
                )
                return VLLMDecision(action="", reason="throttle_timeout", escalate=True)
            except requests.exceptions.Timeout:
                last_attempt_duration_s = max(0.0, time.time() - attempt_started_at)
                self.last_effective_duration_s = last_attempt_duration_s
                if attempt < max_retries:
                    logger.warn(
                        f"[FastLLM] Request timed out on attempt {attempt}/{max_retries}; "
                        f"retrying in {retry_interval_s:.1f}s"
                    )
                    time.sleep(retry_interval_s)
                    continue
                retry_overhead_s = max(
                    0.0,
                    (time.time() - overall_started_at) - last_attempt_duration_s,
                )
                if retry_overhead_s > 0:
                    add_llm_retry_overhead(retry_overhead_s)
                logger.warn(f"[FastLLM] Request timed out after {time.time() - overall_started_at:.1f}s, escalating")
                return VLLMDecision(action="", reason="timeout", escalate=True)
            except Exception as e:
                last_attempt_duration_s = max(0.0, time.time() - attempt_started_at)
                self.last_effective_duration_s = last_attempt_duration_s
                if attempt < max_retries and self._is_retryable_503_error(e):
                    logger.warn(
                        f"[FastLLM] Retryable 503 on attempt {attempt}/{max_retries}; "
                        f"retrying in {retry_interval_s:.1f}s"
                    )
                    time.sleep(retry_interval_s)
                    continue
                retry_overhead_s = max(
                    0.0,
                    (time.time() - overall_started_at) - last_attempt_duration_s,
                )
                if retry_overhead_s > 0:
                    add_llm_retry_overhead(retry_overhead_s)
                logger.warn(f"[FastLLM] Request failed after {time.time() - overall_started_at:.1f}s: {e}, escalating")
                return VLLMDecision(action="", reason=f"api_error: {e}", escalate=True)

    def _substitute_template(self, template: str, params: Dict[str, Any]) -> str:
        """Substitute <$variable_name$> placeholders in template.

        Follows the same pattern as stardojo's assemble_prompt:
        - Strings: direct replacement
        - Lists/Dicts: JSON serialized
        - None/empty: replaced with empty string
        """
        def _replace_match(match):
            var_name = match.group(1)
            value = params.get(var_name, "")
            if value is None:
                return ""
            if isinstance(value, (list, dict)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)

        return _PLACEHOLDER_RE.sub(_replace_match, template)

    @staticmethod
    def _compact_prompt_text(
        value: Any,
        *,
        max_lines: int = 6,
        max_chars: int = 320,
        per_line_chars: int = 140,
        fallback: str = "",
    ) -> str:
        """Compact large prompt fields without changing their core meaning."""
        if value is None:
            return fallback

        raw_lines: List[str] = []
        if isinstance(value, str):
            raw_lines = value.splitlines() or [value]
        elif isinstance(value, dict):
            for key, item in value.items():
                rendered = (
                    json.dumps(item, ensure_ascii=False)
                    if isinstance(item, (dict, list, tuple, set))
                    else str(item)
                )
                raw_lines.append(f"{key}: {rendered}")
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                rendered = (
                    json.dumps(item, ensure_ascii=False)
                    if isinstance(item, (dict, list, tuple, set))
                    else str(item)
                )
                raw_lines.append(rendered)
        else:
            raw_lines = [str(value)]

        cleaned_lines: List[str] = []
        for raw_line in raw_lines:
            line = re.sub(r"\s+", " ", str(raw_line or "")).strip()
            if not line:
                continue
            if per_line_chars > 0 and len(line) > per_line_chars:
                line = line[: per_line_chars - 3].rstrip() + "..."
            cleaned_lines.append(line)

        if not cleaned_lines:
            return fallback

        if max_lines > 0 and len(cleaned_lines) > max_lines:
            omitted = len(cleaned_lines) - max_lines
            cleaned_lines = cleaned_lines[:max_lines] + [f"... (+{omitted} more lines)"]

        text = "\n".join(cleaned_lines)
        if max_chars > 0 and len(text) > max_chars:
            text = text[: max_chars - 16].rstrip() + "... (truncated)"
        return text

    @classmethod
    def _build_prompt_inventory_summary(
        cls,
        inventory: Any,
        toolbar_information: Any,
        selected_slot: Optional[int],
    ) -> str:
        slot_map = cls._extract_inventory_slot_map(inventory, toolbar_information)
        if not slot_map:
            return cls._compact_prompt_text(
                inventory,
                max_lines=8,
                max_chars=320,
                per_line_chars=120,
                fallback="(none)",
            )

        lines: List[str] = []
        if selected_slot is not None and selected_slot in slot_map:
            selected_item = str(slot_map[selected_slot] or "").strip()
            if selected_item and not cls._slot_is_explicitly_empty(selected_item):
                lines.append(f"selected slot_index {selected_slot}: {selected_item}")

        for slot_index in sorted(slot_map):
            if selected_slot is not None and slot_index == selected_slot:
                continue
            item_name = str(slot_map[slot_index] or "").strip()
            if not item_name or cls._slot_is_explicitly_empty(item_name):
                continue
            lines.append(f"slot_index {slot_index}: {item_name}")

        if not lines:
            lines.append("(all visible slots empty)")

        return cls._compact_prompt_text(
            lines,
            max_lines=8,
            max_chars=360,
            per_line_chars=120,
            fallback="(none)",
        )

    @classmethod
    def _build_prompt_surroundings_summary(
        cls,
        game_state: Optional[Dict[str, Any]],
    ) -> str:
        surroundings_map, raw_text = cls._get_structured_surroundings_map(game_state)
        if not surroundings_map:
            return cls._compact_prompt_text(
                raw_text or cls._get_surroundings_summary_text(game_state),
                max_lines=12,
                max_chars=520,
                per_line_chars=140,
                fallback="(none)",
            )

        lines: List[str] = []
        seen_cells = set()

        for direction in cls._CARDINAL_DIRECTIONS:
            relative_cell = cls._direction_to_relative(direction)
            if relative_cell is None:
                continue
            seen_cells.add(relative_cell)
            target_obj = str(surroundings_map.get(relative_cell, "") or "").strip()
            lines.append(f"{direction} {relative_cell}: {target_obj or 'unknown'}")

        nearby_lines: List[str] = []
        nearby_empty_lines: List[str] = []
        farther_lines: List[str] = []
        for cell, raw_value in sorted(
            surroundings_map.items(),
            key=lambda item: (
                max(abs(item[0][0]), abs(item[0][1])),
                abs(item[0][0]) + abs(item[0][1]),
                item[0][1],
                item[0][0],
            ),
        ):
            if cell in seen_cells:
                continue
            target_obj = str(raw_value or "").strip()
            if not target_obj:
                continue
            rendered = f"[{cell[0]}, {cell[1]}]: {target_obj}"
            if cls._is_empty_like_tile(target_obj):
                if max(abs(cell[0]), abs(cell[1])) <= 2:
                    nearby_empty_lines.append(rendered)
                continue
            if max(abs(cell[0]), abs(cell[1])) <= 2:
                nearby_lines.append(rendered)
            else:
                farther_lines.append(rendered)

        lines.extend(nearby_lines[:6])
        lines.extend(nearby_empty_lines[:4])
        lines.extend(farther_lines[:4])

        return cls._compact_prompt_text(
            lines,
            max_lines=14,
            max_chars=560,
            per_line_chars=140,
            fallback="(none)",
        )

    def _build_prompt_from_template(
        self,
        game_state: Dict[str, Any],
        suggestion: Dict[str, str],
        execution_log: List[Dict[str, Any]],
        context_summary: str,
        mem0_reference: str,
        step: int,
        total_steps: int,
        skill_list: str,
    ) -> str:
        """Build prompt from stardojo-style template with game state variables."""
        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        toolbar_information = game_state.get("toolbar_information") or gathered.get("toolbar_information") or ""
        chosen_item = gathered.get("chosen_item", game_state.get("chosen_item", ""))
        if not chosen_item:
            chosen_item = self._extract_selected_item_name_from_toolbar(toolbar_information)
        current_menu = gathered.get("current_menu") or gathered.get("CurrentMenuData") or ""
        image_introduction = game_state.get("image_introduction", [])
        if not isinstance(image_introduction, list):
            image_introduction = []
        front_obstacle_context = self._build_front_obstacle_context(
            game_state,
            execution_log,
            hint_action=suggestion.get("action", ""),
        )

        inventory = gathered.get("inventory")
        if not inventory:
            inventory = game_state.get("inventory", [])
        selected_slot = self._extract_selected_slot_index(game_state, gathered)

        crops = gathered.get("crops") or game_state.get("crops", [])
        buildings = gathered.get("buildings") or game_state.get("buildings", [])
        furniture = gathered.get("furniture") or game_state.get("furniture", [])
        npcs = gathered.get("npcs") or game_state.get("npcs", [])
        exits = gathered.get("exits") or game_state.get("exits", [])

        # Format execution log
        log_lines = []
        for entry in execution_log:
            status = "success" if entry.get("success") else "failed"
            log_lines.append(
                f"  step {entry.get('step', '?')}: "
                f"{entry.get('action', '?')} -> {status} "
                f"({entry.get('note', '')})"
            )
        log_text = self._compact_prompt_text(
            log_lines,
            max_lines=5,
            max_chars=360,
            per_line_chars=120,
            fallback="(none)",
        )

        # Build params dict mapping template variable names to values
        main_task_description = (
            game_state.get("task")
            or game_state.get("main_task")
            or game_state.get("task_description")
            or gathered.get("task_description")
            or ""
        )
        current_subtask_description = (
            game_state.get("subtask_description")
            or gathered.get("subtask_description")
            or ""
        )
        latest_execution_summary = (
            game_state.get("latest_execution_summary")
            or game_state.get("action_feedback")
            or ""
        )
        recent_execution_feedback = game_state.get("recent_execution_feedback", [])
        task_progress_summary = (
            game_state.get("task_progress_summary")
            or latest_execution_summary
            or ""
        )
        failure_signals = game_state.get("failure_signals", "")

        params = {
            # Game context (from gathered_info, matching original stardojo template vars)
            "task_description": main_task_description,
            "subtask_description": current_subtask_description,
            "target_item": game_state.get("target_item", ""),
            "source_type": game_state.get("source_type", ""),
            "source_detail": game_state.get("source_detail", ""),
            "basic_knowledge": self._compact_prompt_text(
                game_state.get("basic_knowledge", ""),
                max_lines=4,
                max_chars=260,
                per_line_chars=120,
            ),
            "context_summary": context_summary,
            "initial_state": context_summary,
            "mem0_reference": mem0_reference,
            "memory_reference": mem0_reference,
            "location": self._compact_prompt_text(gathered.get("location", ""), max_lines=2, max_chars=80, per_line_chars=80),
            "time": self._compact_prompt_text(gathered.get("time", ""), max_lines=1, max_chars=40, per_line_chars=40),
            "season": self._compact_prompt_text(gathered.get("season", ""), max_lines=1, max_chars=40, per_line_chars=40),
            "health": self._compact_prompt_text(gathered.get("health", ""), max_lines=1, max_chars=40, per_line_chars=40),
            "energy": self._compact_prompt_text(gathered.get("energy", ""), max_lines=1, max_chars=40, per_line_chars=40),
            "money": self._compact_prompt_text(gathered.get("money", ""), max_lines=1, max_chars=40, per_line_chars=40),
            "current_position": self._compact_prompt_text(gathered.get("position", ""), max_lines=1, max_chars=60, per_line_chars=60),
            "facing_direction": self._compact_prompt_text(gathered.get("facing_direction", ""), max_lines=1, max_chars=24, per_line_chars=24),
            "facing_position": self._compact_prompt_text(gathered.get("facing_position", ""), max_lines=1, max_chars=80, per_line_chars=80),
            "surroundings": self._build_prompt_surroundings_summary(game_state),
            "front_tile_summary": front_obstacle_context.get("front_tile_summary", "(none)"),
            "blocked_recovery_hint": front_obstacle_context.get("blocked_recovery_hint", ""),
            "current_blocker_signature": front_obstacle_context.get("current_blocker_signature", "(none)"),
            "nearest_grounded_target_summary": front_obstacle_context.get("nearest_grounded_target_summary", "(none)"),
            "current_menu": self._compact_prompt_text(current_menu, max_lines=4, max_chars=220, per_line_chars=100),
            "CurrentMenuData": current_menu,
            "inventory": self._build_prompt_inventory_summary(inventory, toolbar_information, selected_slot),
            "chosen_item": self._compact_prompt_text(chosen_item, max_lines=1, max_chars=80, per_line_chars=80),
            "crops": self._compact_prompt_text(crops, max_lines=6, max_chars=420, per_line_chars=120, fallback="(none)"),
            "buildings": self._compact_prompt_text(buildings, max_lines=6, max_chars=360, per_line_chars=120, fallback="(none)"),
            "furniture": self._compact_prompt_text(furniture, max_lines=4, max_chars=240, per_line_chars=100, fallback="(none)"),
            "npcs": self._compact_prompt_text(npcs, max_lines=4, max_chars=240, per_line_chars=100, fallback="(none)"),
            "exits": self._compact_prompt_text(exits, max_lines=6, max_chars=260, per_line_chars=100, fallback="(none)"),
            "toolbar_information": self._compact_prompt_text(
                toolbar_information,
                max_lines=14,
                max_chars=560,
                per_line_chars=120,
                fallback="(none)",
            ),
            "skill_library": self._compact_prompt_text(
                skill_list,
                max_lines=16,
                max_chars=700,
                per_line_chars=120,
                fallback="(none)",
            ),
            "history_summary": game_state.get("history_summary") or game_state.get("summarization", ""),
            "action_feedback": self._compact_prompt_text(
                game_state.get("action_feedback") or latest_execution_summary,
                max_lines=4,
                max_chars=220,
                per_line_chars=120,
            ),
            "action": game_state.get("pre_action", ""),
            "action_planning_reasoning": game_state.get("pre_decision_making_reasoning") or game_state.get("decision_making_reasoning", ""),
            "self_reflection_reasoning": game_state.get("self_reflection_reasoning", ""),
            "latest_execution_summary": latest_execution_summary,
            "recent_execution_feedback": recent_execution_feedback,
            "task_progress_summary": task_progress_summary,
            "failure_signals": failure_signals,
            "image_introduction": self._compact_prompt_text(
                image_introduction,
                max_lines=4,
                max_chars=260,
                per_line_chars=100,
                fallback="(none)",
            ),
            # Last executed action
            "last_action": self._compact_prompt_text(
                game_state.get("pre_action", ""),
                max_lines=1,
                max_chars=120,
                per_line_chars=120,
            ),
            # LittleBrain-specific: BigBrain suggestion
            "suggested_action": self._compact_prompt_text(
                suggestion.get("action", "N/A"),
                max_lines=1,
                max_chars=120,
                per_line_chars=120,
            ),
            "suggested_reason": self._compact_prompt_text(
                suggestion.get("reason", "N/A"),
                max_lines=2,
                max_chars=120,
                per_line_chars=80,
            ),
            # Execution log
            "execution_log": log_text,
            # Progress
            "step_progress": f"{step + 1}/{total_steps}",
        }

        template = self.template
        if template is None:
            raise ValueError("Template must be set before building prompt from template")

        rendered = self._substitute_template(template, params)

        prefix_sections = []
        if context_summary:
            # Strip raw absolute coordinate listings (e.g. "[-3, -3]: Farmhouse")
            # that confuse LittleBrain into mixing absolute vs relative coords.
            # Keep only semantic/summary lines from the BigBrain analysis.
            filtered_summary = self._filter_context_summary_for_littlebrain(context_summary)
            if filtered_summary:
                prefix_sections.append(
                    "[BigBrain Plan Context] (strategic summary — do NOT treat coordinates below as relative to you)\n"
                    f"{self._compact_prompt_text(filtered_summary, max_lines=8, max_chars=600, per_line_chars=120)}"
                )
        if mem0_reference:
            prefix_sections.append(
                "[Memory Reference] (historical experience)\n"
                f"{self._compact_prompt_text(mem0_reference, max_lines=5, max_chars=320, per_line_chars=120)}"
            )

        if prefix_sections:
            result = "\n\n".join(prefix_sections + [rendered])
        else:
            result = rendered

        stuck_warning = self._detect_stuck_warning(execution_log, game_state)
        if stuck_warning:
            result += "\n" + stuck_warning
        return result

    @staticmethod
    def _filter_context_summary_for_littlebrain(summary: str) -> str:
        """Remove raw absolute-coordinate listings from BigBrain context summary.

        Lines like '[-3, -3]: Farmhouse' confuse LittleBrain because its
        Surroundings section uses *relative* coordinates.  We keep only
        semantic / non-coordinate lines so LittleBrain focuses on its own
        local observations.
        """
        if not summary:
            return ""
        # Pattern: lines that are purely "[int, int]: label" listings
        coord_line_re = re.compile(r"^\s*\[?\s*-?\d+\s*,\s*-?\d+\s*\]?\s*:\s*\S+")
        kept: list[str] = []
        for line in summary.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if coord_line_re.match(stripped):
                continue  # drop raw coordinate listing
            kept.append(stripped)
        return "\n".join(kept)

    def _build_prompt_legacy(
        self,
        context_summary: str,
        suggestion: Dict[str, str],
        execution_log: List[Dict[str, Any]],
        mem0_reference: str,
        step: int,
        total_steps: int,
        skill_list: str,
        game_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Build the decision prompt for the fast LLM (legacy fallback)."""
        # Format execution log
        log_lines = []
        for entry in execution_log:
            status = "success" if entry.get("success") else "FAILED"
            line = (
                f"  step {entry.get('step', '?')}: "
                f"{entry.get('action', '?')} -> {status} "
                f"({entry.get('note', '')})"
            )
            errors_info = entry.get("errors_info", "")
            if errors_info and not entry.get("success"):
                line += f" | error: {errors_info}"
            log_lines.append(line)
        log_text = "\n".join(log_lines) if log_lines else "  (none)"

        # Format mem0 reference
        mem0_text = mem0_reference if mem0_reference else "(none)"

        compact_observation = self._build_compact_game_observation(game_state)
        latest_execution_summary = ""
        zero_progress_streak = 0
        position_issue_detected = False
        if isinstance(game_state, dict):
            latest_execution_summary = str(
                game_state.get("latest_execution_summary", "") or ""
            ).strip()
            zero_progress_streak = int(
                game_state.get("zero_progress_streak", 0) or 0
            )
            position_issue_detected = bool(
                game_state.get("position_issue_detected", False)
            )

        prompt = (
            "You are a fast single-step game action decider. "
            "Decide exactly one next action using the latest observations.\n"
            "\n"
            f"[Task] {game_state.get('task') or game_state.get('main_task') or game_state.get('task_description') or 'unknown'}\n"
            "\n"
            f"[Initial State] (from big brain analysis)\n"
            f"{context_summary}\n"
            "\n"
            f"[Current Observations] (latest state, higher priority than stale history)\n"
            f"{compact_observation}\n"
            "\n"
            f"[Executed Steps]\n"
            f"{log_text}\n"
            "\n"
            f"[Latest Execution Feedback]\n"
            f"{latest_execution_summary or '(none)'}\n"
            "\n"
            f"[Big Brain Suggestion] (reference — follow for moves, override when you see a clearly better productive action)\n"
            f"  Suggested action: {suggestion.get('action', 'N/A')}\n"
            f"  Suggested reason: {suggestion.get('reason', 'N/A')}\n"
            "\n"
            f"[Memory Reference] (historical experience)\n"
            f"{mem0_text}\n"
            "\n"
            f"[Progress] Step {step + 1}/{total_steps}\n"
            "\n"
            f"[Available Skills]\n"
            f"{skill_list}\n"
            "\n"
            "Decide:\n"
            "- Always trust [Current Observations] over stale summary or stale suggestion.\n"
            "- First check whether a menu is open. If a menu is open, prefer choose_option(...) or menu(...). Do not move/use/interact blindly while a menu is blocking gameplay.\n"
            "- If the currently selected item/tool already matches the needed tool, do NOT output choose_item(...) again.\n"
            "- Only output choose_item(...) when the current tool is wrong and the switch is necessary for the immediate next real action.\n"
            "- choose_item(slot_index=...) uses the exact 0-based inventory slot_index (0-35). Prefer the Inventory list when it is available; use Toolbar information when the item is there. Do not guess, and do not convert it to 1-based numbering.\n"
            "- For direction-based skills, always use explicit syntax like use(direction=\"down\") or interact(direction=\"left\"). Never output use(down) or interact(left).\n"
            "- use(direction) is for TOOLS (Axe, Hoe, Watering Can, Pickaxe, Scythe). interact(direction) is for ITEMS and world objects (seeds, fertilizers, shipping bin, doors, NPCs, chests, beds, harvesting mature crops).\n"
            "- If the selected item is a seed or fertilizer, NEVER output use(direction=...). Those items must use interact(direction=...), or you should make a short local reposition to line up with the correct tile first.\n"
            "- Tool reminder: Pickaxe clears stone, Axe clears wood/twigs/logs, and Scythe clears weeds/grass/fiber.\n"
            "- Before outputting use(direction=...), verify the obstacle in that direction matches the selected tool. Never use Axe on weeds/grass/fiber, Scythe on stone, or Pickaxe on wood/twig/log.\n"
            "- If the same or a similar action already failed, do not repeat it blindly; change direction, tool, position, or menu state first unless the latest observations clearly justify a retry.\n"
            f"- Current zero-progress streak: {zero_progress_streak}. If it is 2 or more, do not start by repeating the same productive action.\n"
            f"- Position issue detected: {position_issue_detected}. If true, prefer move(...) before use()/interact(). If the current observations show a clearable obstacle directly in front, route around it first unless the reroute is not grounded.\n"
            "- If a move was blocked and the current observations suggest a clearable obstacle directly in front of you, prefer a grounded reroute, short sidestep, or different axis order first. Only choose the matching tool and clear it when rerouting is not grounded or repeated reroutes still fail.\n"
            "- If a move was blocked but there is no clearable obstacle directly in front, prefer a 3-5 tile reposition, the other axis first, or a direction change before retrying.\n"
            "- AXIS ORDER: If you need to move diagonally (both x and y) AND Surroundings show an obstacle on one axis, split into two single-axis moves with the clear axis first. Example: if stone is at [1,0] (right), do move(x=0, y=-2) first, then move(x=3, y=0). If no obstacle is visible, a single diagonal move is fine.\n"
            "- If the big brain suggestion is move(...), follow it by default. Only override a big brain move when current observations show a concrete contradiction: the path or destination axis is visibly blocked, a menu prevents movement, or a clearly reachable productive action is already in range right now. Do not override it just because another action feels plausible.\n"
            "- Movement is allowed and often correct. If no target is clearly in range from the current tile, prefer a sensible move(...) over guessing a use(...) direction.\n"
            "- CRITICAL: move(x, y) takes RELATIVE offsets, NOT absolute coordinates. If you are at (10, 20) and want to reach (13, 18), output move(x=3, y=-2). Do NOT output move(x=13, y=18). Typical safe range is -15 to 15 per axis. When both axes are needed, choose the axis order that avoids known obstacles.\n"
            "- Prefer a direct productive action over setup only when the target is clearly already in range; otherwise reposition first.\n"
            "- If executed steps or observations show the suggestion is outdated, override it with a better action.\n"
            "- NAVIGATION: If the task requires going to a different location, check the Exits in Current Observations. Move toward the exit that leads closer to the target. Farm connects east to Bus Stop and north to Backwoods via the pet bowl entrance path; Bus Stop connects east to Town; Town has shops and NPCs.\n"
            "- If uncertain, choose the safest grounded action supported by the current observations. Do NOT output ESCALATE as a normal answer just because the state is ambiguous.\n"
            "- Only use ESCALATE when the screen is temporarily non-interactable (for example loading, fade-to-black, or another transient state where no grounded menu/world action exists).\n"
            "\n"
            "Output format (strict):\n"
            "ACTION: skill_name(params)\n"
            "REASON: brief reason (within 20 chars)\n"
            "\n"
            "Emergency-only alternative:\n"
            "ESCALATE\n"
            "REASON: why big brain is needed (within 20 chars)"
        )

        stuck_warning = self._detect_stuck_warning(execution_log, game_state)
        if stuck_warning:
            prompt += "\n" + stuck_warning
        return prompt

    def _build_compact_game_observation(
        self,
        game_state: Optional[Dict[str, Any]],
        max_chars: int = 220,
    ) -> str:
        """Build a compact latest-observation block for legacy prompt mode."""
        if not isinstance(game_state, dict):
            return "(none)"

        gathered = game_state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}
        front_obstacle_context = self._build_front_obstacle_context(game_state, [])

        def _compact(value: Any, fallback: str = "(none)") -> str:
            if value is None:
                return fallback
            if isinstance(value, (dict, list)):
                text = json.dumps(value, ensure_ascii=False)
            else:
                text = str(value)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                return fallback
            if len(text) > max_chars:
                return text[: max_chars - 3] + "..."
            return text

        chosen_item = gathered.get("chosen_item")
        selected_item_name = (
            gathered.get("selected_item_name")
            or gathered.get("selected_tool")
            or gathered.get("current_tool")
        )
        if not selected_item_name and isinstance(chosen_item, dict):
            for key in ("currentitem", "current_item", "item_name", "name", "item"):
                candidate = chosen_item.get(key)
                if candidate:
                    selected_item_name = candidate
                    break

        observation_lines = [
            f"- Subtask: {_compact(game_state.get('subtask_description') or game_state.get('task'))}",
            f"- Menu: {_compact(gathered.get('current_menu') or gathered.get('CurrentMenuData'))}",
            f"- Location: {_compact(gathered.get('location'))}",
            f"- Position: {_compact(gathered.get('position'))}",
            f"- Facing: {_compact(gathered.get('facing_direction'))}",
            f"- Effect point: {_compact(gathered.get('facing_position'))}",
            f"- Selected item: {_compact(selected_item_name)}",
            f"- Toolbar: {_compact(game_state.get('toolbar_information') or gathered.get('toolbar_information'))}",
            f"- Surroundings: {_compact(self._get_surroundings_summary_text(game_state))}",
            f"- Front tile: {_compact(front_obstacle_context.get('front_tile_summary'))}",
            f"- Blocked recovery: {_compact(front_obstacle_context.get('blocked_recovery_hint'))}",
            f"- Exits: {_compact(gathered.get('exits') or game_state.get('exits'))}",
            f"- Last action: {_compact(game_state.get('pre_action'))}",
        ]
        return "\n".join(observation_lines)

    @staticmethod
    def _extract_canonical_action_candidate(action_text: Any) -> str:
        text = str(action_text or "").strip()
        if not text:
            return ""

        text = text.strip("`").strip()
        text = re.sub(r"^\d+\.\s*", "", text)
        text = re.sub(r"^(?:[-*]\s*)?(?:step\s*\d+\s*[:\-]\s*)?", "", text, flags=re.IGNORECASE)
        text = re.sub(
            r"\s*->\s*(?:success|succeeded|failed|failure|error|done|ok|true|false)\b.*$",
            "",
            text,
            flags=re.IGNORECASE,
        )
        text = re.sub(r"\s+#.*$", "", text)
        text = text.strip()
        if not text:
            return ""

        if not re.match(
            r"^(move|use|interact|choose_item|attach_item|unattach_item|craft|choose_option|menu|nop)\b",
            text,
            re.IGNORECASE,
        ):
            return ""

        move_named = re.match(
            r"^move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)$",
            text,
            re.IGNORECASE,
        )
        if move_named:
            return f"move(x={int(move_named.group(1))}, y={int(move_named.group(2))})"

        move_positional = re.match(
            r"^move\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$",
            text,
            re.IGNORECASE,
        )
        if move_positional:
            return f"move(x={int(move_positional.group(1))}, y={int(move_positional.group(2))})"

        directional_named = re.match(
            r"^(use|interact)\(\s*direction\s*=\s*[\"']?(up|down|left|right)[\"']?\s*\)$",
            text,
            re.IGNORECASE,
        )
        if directional_named:
            skill = directional_named.group(1).lower()
            direction = directional_named.group(2).lower()
            return f'{skill}(direction="{direction}")'

        directional_positional = re.match(
            r"^(use|interact)\(\s*(up|down|left|right)\s*\)$",
            text,
            re.IGNORECASE,
        )
        if directional_positional:
            skill = directional_positional.group(1).lower()
            direction = directional_positional.group(2).lower()
            return f'{skill}(direction="{direction}")'

        slot_named = re.match(
            r"^(choose_item|attach_item)\(\s*slot_index\s*=\s*(-?\d+)\s*\)$",
            text,
            re.IGNORECASE,
        )
        if slot_named:
            skill = slot_named.group(1).lower()
            slot_index = int(slot_named.group(2))
            return f"{skill}(slot_index={slot_index})"

        slot_positional = re.match(
            r"^(choose_item|attach_item)\(\s*(-?\d+)\s*\)$",
            text,
            re.IGNORECASE,
        )
        if slot_positional:
            skill = slot_positional.group(1).lower()
            slot_index = int(slot_positional.group(2))
            return f"{skill}(slot_index={slot_index})"

        bare_call = re.match(
            r"^(unattach_item|nop)\(\s*\)",
            text,
            re.IGNORECASE,
        )
        if bare_call:
            return f"{bare_call.group(1).lower()}()"

        structured_call = VLLMClient._extract_canonical_structured_skill_call(text)
        if structured_call:
            return structured_call

        return ""

    @staticmethod
    def _extract_int_literal(node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return int(node.value)
        return None

    @staticmethod
    def _extract_optional_int_literal(node: ast.AST) -> Optional[int]:
        if isinstance(node, ast.Constant) and node.value is None:
            return None
        return VLLMClient._extract_int_literal(node)

    @staticmethod
    def _extract_str_literal(node: ast.AST) -> Optional[str]:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return str(node.value)
        return None

    @classmethod
    def _extract_canonical_structured_skill_call(cls, text: str) -> str:
        try:
            parsed = ast.parse(text, mode="eval")
        except SyntaxError:
            return ""

        call = parsed.body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            return ""

        skill_name = call.func.id.lower()
        if skill_name not in {"craft", "choose_option", "menu"}:
            return ""

        if skill_name == "craft":
            if len(call.args) + len(call.keywords) != 1:
                return ""
            if call.args:
                item_name = cls._extract_str_literal(call.args[0])
            else:
                keyword = call.keywords[0]
                if keyword.arg != "item":
                    return ""
                item_name = cls._extract_str_literal(keyword.value)
            if not item_name:
                return ""
            return f"craft(item={json.dumps(item_name)})"

        if skill_name == "choose_option":
            if len(call.args) > 3:
                return ""
            ordered_keys = ("option_index", "quantity", "direction")
            values: Dict[str, ast.AST] = {}
            for key, node in zip(ordered_keys, call.args):
                values[key] = node
            for keyword in call.keywords:
                if keyword.arg not in ordered_keys or keyword.arg in values:
                    return ""
                values[keyword.arg] = keyword.value

            if "option_index" not in values:
                return ""
            option_index = cls._extract_int_literal(values["option_index"])
            if option_index is None:
                return ""

            rendered_parts = [f"option_index={option_index}"]
            if "quantity" in values:
                quantity_node = values["quantity"]
                quantity_value = cls._extract_optional_int_literal(quantity_node)
                if quantity_value is None:
                    if not (
                        isinstance(quantity_node, ast.Constant)
                        and quantity_node.value is None
                    ):
                        return ""
                else:
                    rendered_parts.append(f"quantity={quantity_value}")

            if "direction" in values:
                direction_node = values["direction"]
                if isinstance(direction_node, ast.Constant) and direction_node.value is None:
                    pass
                else:
                    direction_value = cls._extract_str_literal(direction_node)
                    if direction_value is None:
                        return ""
                    direction_value = direction_value.lower()
                    if direction_value not in {"in", "out"}:
                        return ""
                    rendered_parts.append(f'direction="{direction_value}"')

            return f"choose_option({', '.join(rendered_parts)})"

        if len(call.args) > 2:
            return ""
        ordered_keys = ("option", "menu_name")
        values = {}
        for key, node in zip(ordered_keys, call.args):
            values[key] = node
        for keyword in call.keywords:
            if keyword.arg not in ordered_keys or keyword.arg in values:
                return ""
            values[keyword.arg] = keyword.value
        if "option" not in values or "menu_name" not in values:
            return ""

        option = cls._extract_str_literal(values["option"])
        menu_name = cls._extract_str_literal(values["menu_name"])
        if not option or not menu_name:
            return ""
        return f'menu(option={json.dumps(option)}, menu_name={json.dumps(menu_name)})'

    @classmethod
    def _extract_action_candidates_from_text(cls, text: Any) -> List[str]:
        source = str(text or "")
        if not source:
            return []

        action_patterns = (
            r"move\(\s*(?:x\s*=\s*-?\d+\s*,\s*y\s*=\s*-?\d+|-?\d+\s*,\s*-?\d+)\s*\)",
            r"(?:use|interact)\(\s*(?:direction\s*=\s*[\"']?(?:up|down|left|right)[\"']?|up|down|left|right)\s*\)",
            r"(?:choose_item|attach_item)\(\s*(?:slot_index\s*=\s*-?\d+|-?\d+)\s*\)",
            r"(?:unattach_item|nop)\(\s*\)",
            r"(?:craft|choose_option|menu)\([^)\n\r`]*\)",
        )
        combined = re.compile(
            "|".join(f"(?:{pattern})" for pattern in action_patterns),
            re.IGNORECASE,
        )

        candidates: List[str] = []
        seen = set()
        for match in combined.finditer(source):
            candidate = cls._extract_canonical_action_candidate(match.group(0))
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
        return candidates

    def _parse_response(
        self, text: str, fallback_suggestion: Dict[str, str]
    ) -> VLLMDecision:
        """Parse LLM response into a VLLMDecision.

        Supports two output formats:

        1. Legacy format (ACTION/REASON):
            ACTION: skill_name(params)
            REASON: ...

        2. Stardojo format (Reasoning + Actions code block):
            Reasoning:
            1. ...
            Actions:
            ```python
                action(args=x)
            ```

        Or:
            ESCALATE
            REASON: ...
        """
        text = text.strip()

        # Check for ESCALATE at start of a line (anchored to avoid false positives)
        if re.search(r"(?m)^\s*ESCALATE\s*$", text, re.IGNORECASE):
            reason_match = re.search(
                r"(?m)^REASON:\s*(.+?)$", text, re.IGNORECASE
            )
            reason = (
                reason_match.group(1).strip() if reason_match else "unknown"
            )
            return VLLMDecision(action="", reason=reason, escalate=True)

        # Try stardojo format: extract action from ```python ... ``` block
        code_match = re.search(r"```(?:python)?\s*(.+?)```", text, re.DOTALL)
        if code_match:
            code_block = code_match.group(1).strip()
            # Extract the action line (skip comments)
            action_lines = [
                line.strip() for line in code_block.split("\n")
                if line.strip()
                and not line.strip().startswith("#")
                and not line.strip().startswith("Reasoning:")
            ]
            for line in action_lines:
                action = self._extract_canonical_action_candidate(line)
                if not action:
                    continue
                # Extract reasoning (text before Actions:)
                reasoning_match = re.search(
                    r"Reasoning:\s*\n(.+?)Actions:",
                    text,
                    re.DOTALL | re.IGNORECASE,
                )
                reason = ""
                if reasoning_match:
                    reason = reasoning_match.group(1).strip()[:60]
                return VLLMDecision(action=action, reason=reason, escalate=False)

        # Try legacy format: ACTION/REASON
        action_match = re.search(
            r"(?m)^ACTION:\s*(.+?)$", text, re.IGNORECASE
        )
        reason_match = re.search(
            r"(?m)^REASON:\s*(.+?)$", text, re.IGNORECASE
        )

        if action_match:
            action = self._extract_canonical_action_candidate(action_match.group(1))
            reason = (
                reason_match.group(1).strip() if reason_match else ""
            )
            if action:
                return VLLMDecision(action=action, reason=reason, escalate=False)

        actions_section_match = re.search(
            r"Actions:\s*(.+)$",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if actions_section_match:
            actions_section = actions_section_match.group(1)
            action_candidates = self._extract_action_candidates_from_text(actions_section)
            if action_candidates:
                action = action_candidates[-1]
                logger.write(f"[FastLLM] Parsed actions-section fallback: {action}")
                return VLLMDecision(
                    action=action,
                    reason="actions_section_fallback",
                    escalate=False,
                )

        for raw_line in reversed(text.splitlines()):
            action = self._extract_canonical_action_candidate(raw_line)
            if not action:
                continue
            logger.write(f"[FastLLM] Parsed inline action fallback: {action}")
            return VLLMDecision(
                action=action,
                reason="inline_action_fallback",
                escalate=False,
            )

        lowered = text.lower()
        invalidity_markers = (
            "do not have",
            "don't have",
            "missing from the hotbar",
            "missing from the inventory",
            "missing item",
            "inventory is empty",
            "slots 5-35 are empty",
            "does not contain any item",
            "slot is empty",
            "critical blocker is the wrong tool",
            "wrong tool",
            "invalid target",
            "path not found",
            "cannot perform",
            "cannot proceed",
            "cannot fertilize",
            "cannot plant",
            "cannot complete the task",
            "does not help achieve",
            "not helpful",
            "would be useless",
            "the item is missing",
            "the primary blocker is",
            "the suggestion is a no-op",
            "no-op",
        )
        if any(marker in lowered for marker in invalidity_markers):
            logger.warn(
                "[FastLLM] Could not parse response and the model text invalidated the suggestion; "
                "falling back to validator-side grounded recovery"
            )
            return VLLMDecision(
                action=str(fallback_suggestion.get("action", "") or "").strip(),
                reason="parse_fallback_invalidated_suggestion",
                escalate=False,
            )

        # Truncation recovery: if the response has Reasoning but no Actions block,
        # it likely hit the max_tokens limit. Adopt the BigBrain suggestion directly
        # instead of wasting the step.
        has_reasoning = bool(re.search(r"(?i)^Reasoning:", text, re.MULTILINE))
        has_actions_header = bool(re.search(r"(?i)^Actions:", text, re.MULTILINE))
        if has_reasoning and not has_actions_header:
            fallback_action = str(fallback_suggestion.get("action", "") or "").strip()
            if fallback_action:
                logger.warn(
                    "[FastLLM] Response appears truncated (Reasoning present but no Actions block); "
                    "adopting BigBrain suggestion to avoid wasting step"
                )
                return VLLMDecision(
                    action=fallback_action,
                    reason="truncation_recovery",
                    escalate=False,
                )

        fallback_action = str(fallback_suggestion.get("action", "") or "").strip()
        if not fallback_action:
            logger.warn(
                "[FastLLM] Could not parse response and no fallback suggestion exists; escalating"
            )
            return VLLMDecision(
                action="",
                reason="parse_fallback_no_suggestion",
                escalate=True,
            )
        # Fallback: adopt suggestion if response doesn't match format
        logger.warn(
            "[FastLLM] Could not parse response, adopting suggestion"
        )
        return VLLMDecision(
            action=fallback_action,
            reason=f"parse_fallback: "
                   f"{fallback_suggestion.get('reason', '')}",
            escalate=False,
        )
