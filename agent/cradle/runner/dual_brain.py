"""
Dual-brain controller for orchestrating big/little brain (Phase 3.4).

This is the main integration point. It wires together:
    - BrainScheduler (when to use which brain)
    - BigBrain (full LangGraph workflow)
    - LittleBrain (fast LLM path)
    - EnvironmentChangeDetector (scene change)
    - FailureDetector (execution quality)

Used by stardew_runner._run_with_langgraph() as an external controller.
"""
import os
import re
import time
from typing import Any, Dict, Optional

from cradle.log import Logger
from cradle.runner.big_brain import BrainPlanResult

logger = Logger()


_TRUE_ENV_VALUES = {"1", "true", "yes", "on"}


def _env_flag(name: str) -> bool:
    return str(os.getenv(name, "")).strip().lower() in _TRUE_ENV_VALUES


class DualBrainController:
    """Orchestrates dual-brain decision cycle.

    Usage in runner::

        controller = DualBrainController.from_config(
            workflow_app=self.workflow_app,
            gm=self.gm,
            skill_execute_provider=self.skill_execute,
            augment_provider=self.augment,
            embed_provider=self.embed_provider,
            mem0_provider=mem0_provider,
        )

        while step < max_steps:
            result_state = controller.step(state, workflow_config)
            state = prepare_next_state(result_state)
    """

    _MENU_CLOSED_TYPES = {"", "no menu", "none", "null"}
    _VLLM_OUTAGE_MARKERS = (
        "api_error: 503",
        "503 service unavailable",
        "service unavailable",
        "reach max requests",
        "service support max requests",
        "max requests",
        "too many requests",
        "overloaded",
    )
    _VLLM_TRANSPORT_FAILURE_MARKERS = (
        "vllm_escalate: api_error:",
        "vllm_escalate: timeout",
        "vllm_escalate: throttle_timeout",
        "api_error: 503",
        "503 service unavailable",
        "service unavailable",
        "reach max requests",
        "service support max requests",
        "max requests",
        "too many requests",
        "overloaded",
        "throttle_timeout",
    )

    def __init__(
        self,
        workflow_app: Any,
        scheduler: Any,
        big_brain: Any,
        little_brain: Any,
        env_detector: Any,
        failure_detector: Any,
        vllm_client: Any,
        vllm_available: bool = False,
        big_brain_backoff_seconds: float = 8.0,
        vllm_health_retry_seconds: float = 30.0,
        vllm_reenable_success_threshold: int = 2,
        vllm_reenable_probe_interval_seconds: float = 3.0,
        big_brain_only: bool = False,
    ):
        self.workflow_app = workflow_app
        self.scheduler = scheduler
        self.big_brain = big_brain
        self.little_brain = little_brain
        self.env_detector = env_detector
        self.failure_detector = failure_detector
        self.vllm_client = vllm_client
        self.vllm_available = vllm_available
        self.big_brain_backoff_seconds = max(0.0, float(big_brain_backoff_seconds))
        self.vllm_health_retry_seconds = max(5.0, float(vllm_health_retry_seconds))
        self.vllm_reenable_success_threshold = max(1, int(vllm_reenable_success_threshold))
        self.vllm_reenable_probe_interval_seconds = max(
            1.0,
            min(float(vllm_reenable_probe_interval_seconds), self.vllm_health_retry_seconds),
        )
        self.big_brain_only = bool(big_brain_only)
        self._big_brain_backoff_until: float = 0.0
        self._next_vllm_health_retry_ts: float = 0.0
        self._vllm_reenable_success_streak: int = 0
        if not self.big_brain_only and not self.vllm_available and self.vllm_client is not None:
            self._schedule_next_vllm_health_retry()

        # Phase 6.5: Forced escape state
        self._forced_escape_count: int = 0
        self._max_forced_escapes: int = 3

    @staticmethod
    def _build_image_embedder(image_cfg: Optional[Dict[str, Any]] = None):
        from cradle.runner.image_embedder import ImageEmbedder

        return ImageEmbedder.from_config(image_cfg)

    @classmethod
    def from_config(
        cls,
        workflow_app: Any,
        gm: Any,
        skill_execute_provider: Any,
        augment_provider: Any = None,
        embed_provider: Any = None,
        mem0_provider: Any = None,
    ) -> "DualBrainController":
        """Create controller from enhanced_config.yaml settings.

        Reads dual_brain section from config and instantiates all components.
        """
        import yaml
        import os
        from cradle.utils.file_utils import assemble_project_path

        # Load config. STARDOJO_ENHANCED_CONFIG lets experiment launchers use a
        # generated config without mutating the default repo config.
        config_path = os.getenv("STARDOJO_ENHANCED_CONFIG", "").strip()
        if config_path:
            config_path = assemble_project_path(config_path)
        else:
            config_path = assemble_project_path("./conf/enhanced_config.yaml")
        cfg: Dict = {}
        performance_cfg: Dict = {}
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
                cfg = raw.get("dual_brain", {})
                performance_cfg = raw.get("performance", {}) or {}

        # --- Scheduler ---
        from cradle.runner.scheduler import BrainScheduler

        sched_cfg = cfg.get("scheduler", {})
        mode = str(cfg.get("mode", cfg.get("run_mode", "")) or "").strip().lower()
        big_brain_only = _env_flag("STARDOJO_BIG_BRAIN_ONLY") or mode in {
            "big_brain_only",
            "bigbrain_only",
            "big_only",
            "only_big_brain",
        }
        single_action_big_brain = _env_flag("STARDOJO_BIG_BRAIN_SINGLE_ACTION") or bool(
            cfg.get("single_action_big_brain", False)
        )
        failure_cfg = (
            cfg.get("failure", {})
            or performance_cfg.get("failure", {})
            or {}
        )
        scheduler = BrainScheduler(
            cycle_size=int(sched_cfg.get("cycle_size", 4)),
            max_consecutive_failures=int(
                cfg.get("little_brain", {}).get("max_consecutive_failures", 2)
            ),
            stall_zero_progress_threshold=int(
                sched_cfg.get("stall_zero_progress_threshold", 4)
            ),
            stall_repeat_threshold=int(
                sched_cfg.get("stall_repeat_threshold", 5)
            ),
            stall_repeat_zero_progress_threshold=int(
                sched_cfg.get("stall_repeat_zero_progress_threshold", 2)
            ),
        )

        # --- Fast LLM client (nothinking mode) ---
        from cradle.runner.vllm_client import VLLMClient

        lb_cfg = cfg.get("little_brain", {})
        llm_cfg = lb_cfg.get("llm", lb_cfg.get("vllm", {}))
        model_override = llm_cfg.get("model") or None

        # Prefer from_openai_config for unified credential management
        try:
            vllm_client = VLLMClient.from_openai_config(
                model_override=model_override,
            )
            # Override max_tokens if specified in config
            if "max_tokens" in llm_cfg:
                vllm_client.max_tokens = int(llm_cfg["max_tokens"])
            if "request_timeout_s" in llm_cfg:
                vllm_client.request_timeout_s = max(1.0, float(llm_cfg["request_timeout_s"]))
            health_check_timeout = llm_cfg.get(
                "health_check_timeout_s",
                llm_cfg.get("health_check_timeout_seconds"),
            )
            if health_check_timeout is not None:
                vllm_client.health_check_timeout_s = max(1.0, float(health_check_timeout))
            elif "request_timeout_s" in llm_cfg:
                vllm_client.health_check_timeout_s = VLLMClient._default_health_check_timeout(
                    vllm_client.request_timeout_s
                )
        except Exception as e:
            logger.warn(f"[DualBrain] from_openai_config failed: {e}, using defaults")
            vllm_client = VLLMClient(
                api_key=llm_cfg.get("api_key", ""),
                model=llm_cfg.get("model", "qwen-plus"),
                max_tokens=int(llm_cfg.get("max_tokens", 100)),
                request_timeout_s=int(llm_cfg.get("request_timeout_s", 30)),
                health_check_timeout_s=llm_cfg.get(
                    "health_check_timeout_s",
                    llm_cfg.get("health_check_timeout_seconds"),
                ),
            )

        vllm_available = False if big_brain_only else vllm_client.health_check()
        if not vllm_available and not big_brain_only:
            logger.warn(
                "[DualBrain] Fast LLM API not available right now. "
                "Using big brain until the next health refresh. "
                "Check endpoint/API key in conf/openai_config.json if this persists."
            )

        # --- Image embedder + env detector ---
        from cradle.runner.env_detector import EnvironmentChangeDetector

        try:
            image_embedder = cls._build_image_embedder(cfg.get("image_embedding", {}))
        except Exception as e:
            logger.warn(f"[DualBrain] ImageEmbedder init failed: {e}, env detection disabled")
            image_embedder = None

        env_cfg = cfg.get("env_detector", {})
        env_detector = EnvironmentChangeDetector(
            image_embedder=image_embedder,
            threshold=float(env_cfg.get("change_threshold", 0.35)),
        )

        # --- Big brain ---
        from cradle.runner.big_brain import BigBrain

        bb_cfg = cfg.get("big_brain", {})
        output_plan_steps = int(bb_cfg.get("output_plan_steps", 4))
        if single_action_big_brain:
            output_plan_steps = 1
        big_brain = BigBrain(
            workflow_app=workflow_app,
            output_plan_steps=output_plan_steps,
            context_summary_max_chars=int(
                bb_cfg.get("context_summary_max_chars", 150)
            ),
        )

        # --- Little brain ---
        from cradle.runner.little_brain import LittleBrain

        little_brain = LittleBrain(
            vllm_client=vllm_client,
            mem0_provider=mem0_provider,
            skill_execute_provider=skill_execute_provider,
            gm=gm,
            augment_provider=augment_provider,
            max_relative_move=int(lb_cfg.get("max_relative_move", 20)),
        )

        # --- Failure detector ---
        from cradle.runner.failure_detector import FailureDetector

        failure_detector = FailureDetector(
            little_brain_timeout_ms=int(
                failure_cfg.get("little_brain_timeout_ms", 5000)
            ),
            big_brain_timeout_s=int(
                failure_cfg.get("big_brain_timeout_s", 75)
            ),
            max_consecutive_failures=int(
                cfg.get("little_brain", {}).get("max_consecutive_failures", 2)
            ),
        )

        logger.write("=" * 80)
        logger.write("[DualBrain] Controller initialized")
        logger.write(
            "  Scheduler: "
            f"cycle_size={scheduler.cycle_size}, "
            f"stall_z={scheduler.stall_zero_progress_threshold}, "
            f"stall_repeat={scheduler.stall_repeat_threshold}, "
            f"stall_repeat_z={scheduler.stall_repeat_zero_progress_threshold}"
        )
        logger.write(
            f"  Env change threshold: {env_detector.threshold}"
        )
        logger.write(
            f"  FastLLM API: {'available' if vllm_available else 'NOT available'}"
        )
        logger.write(
            f"  Mode: {'big-brain-only' if big_brain_only else 'dual-brain'}, "
            f"output_plan_steps={output_plan_steps}"
        )
        logger.write(
            f"  Env detector: threshold={env_detector.threshold}"
        )
        logger.write("=" * 80)

        return cls(
            workflow_app=workflow_app,
            scheduler=scheduler,
            big_brain=big_brain,
            little_brain=little_brain,
            env_detector=env_detector,
            failure_detector=failure_detector,
            vllm_client=vllm_client,
            vllm_available=vllm_available,
            big_brain_backoff_seconds=float(
                failure_cfg.get("big_brain_backoff_seconds", 8.0)
            ),
            vllm_health_retry_seconds=float(
                llm_cfg.get("health_retry_seconds", 30.0)
            ),
            vllm_reenable_success_threshold=int(
                llm_cfg.get("recovery_success_threshold", 2)
            ),
            vllm_reenable_probe_interval_seconds=float(
                llm_cfg.get("recovery_probe_interval_s", 3.0)
            ),
            big_brain_only=big_brain_only,
        )

    # -- Phase 6.5: Stuck detection and forced escape --

    def _is_definitively_stuck(self, state: dict) -> bool:
        if self._forced_escape_count >= self._max_forced_escapes:
            return False  # 已尝试够多次，让 BigBrain 再试
        zero = int(state.get("zero_progress_streak", 0) or 0)
        repeated = int(state.get("repeated_action_streak", 0) or 0)
        oscillation = int(state.get("oscillation_streak", 0) or 0)
        consecutive_failures = int(state.get("consecutive_failures", 0) or 0)
        # repeated_action_streak alone can still be a productive chain.
        repeated_stuck = repeated >= 6 and (zero >= 2 or consecutive_failures >= 2)
        return zero >= 4 or oscillation >= 3 or repeated_stuck

    def _generate_escape_action(self, state: dict) -> str:
        import re, random
        last = str(state.get("last_action", "") or "")
        move_m = re.match(r"^move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)$", last, re.IGNORECASE)
        strategy = self._forced_escape_count % 4
        if strategy == 0:  # 垂直方向
            if move_m:
                x, y = int(move_m.group(1)), int(move_m.group(2))
                return f"move(x={max(-5,min(5,-y if y else 3))}, y={max(-5,min(5,x if x else 3))})"
            return "move(x=3, y=0)"
        elif strategy == 1:  # 反方向
            if move_m:
                x, y = int(move_m.group(1)), int(move_m.group(2))
                return f"move(x={max(-5,min(5,-x)) if x else 0}, y={max(-5,min(5,-y)) if y else 0})"
            return "move(x=0, y=-3)"
        elif strategy == 2:  # 随机小 move
            dx, dy = random.choice([-5,-4,-3,-2,2,3,4,5]), random.choice([-5,-4,-3,-2,2,3,4,5])
            return f"move(x={dx}, y={dy})"
        else:  # 尝试 interact
            return 'interact(direction="down")'

    def _schedule_big_brain_backoff(self, reason: str) -> None:
        if self.big_brain_backoff_seconds <= 0:
            return
        self._big_brain_backoff_until = max(
            self._big_brain_backoff_until,
            time.time() + self.big_brain_backoff_seconds,
        )
        logger.warn(
            "[DualBrain] Scheduling big-brain backoff "
            f"({self.big_brain_backoff_seconds:.1f}s) due to {reason}"
        )

    def _apply_big_brain_backoff_if_needed(self) -> None:
        remaining_s = self._big_brain_backoff_until - time.time()
        if remaining_s <= 0:
            self._big_brain_backoff_until = 0.0
            return
        logger.warn(
            "[DualBrain] Backing off before next big-brain replan "
            f"for {remaining_s:.1f}s"
        )
        time.sleep(remaining_s)
        self._big_brain_backoff_until = 0.0

    def _maybe_refresh_vllm_availability(self) -> None:
        if self.vllm_available or self.vllm_client is None:
            return
        now = time.time()
        if now < self._next_vllm_health_retry_ts:
            return
        self._schedule_next_vllm_health_retry(now=now)
        try:
            if self.vllm_client.health_check(timeout_s=self._get_vllm_health_check_timeout()):
                self._vllm_reenable_success_streak += 1
                if self._vllm_reenable_success_streak >= self.vllm_reenable_success_threshold:
                    self.vllm_available = True
                    self._vllm_reenable_success_streak = 0
                    logger.write("[DualBrain] FastLLM API recovered; LittleBrain re-enabled")
                else:
                    self._next_vllm_health_retry_ts = now + self.vllm_reenable_probe_interval_seconds
                    logger.write(
                        "[DualBrain] FastLLM health probe passed "
                        f"({self._vllm_reenable_success_streak}/{self.vllm_reenable_success_threshold}); "
                        f"rechecking in {self.vllm_reenable_probe_interval_seconds:.1f}s before re-enabling"
                    )
            else:
                self._vllm_reenable_success_streak = 0
        except Exception as e:
            self._vllm_reenable_success_streak = 0
            logger.warn(f"[DualBrain] FastLLM availability refresh failed: {e}")

    def _schedule_next_vllm_health_retry(self, now: Optional[float] = None) -> None:
        if self.vllm_client is None:
            return
        current_time = time.time() if now is None else float(now)
        self._next_vllm_health_retry_ts = max(
            self._next_vllm_health_retry_ts,
            current_time + self.vllm_health_retry_seconds,
        )

    def _get_vllm_health_check_timeout(self) -> float:
        timeout_s = getattr(self.vllm_client, "health_check_timeout_s", None)
        try:
            return max(1.0, float(timeout_s))
        except (TypeError, ValueError):
            return 5.0

    @staticmethod
    def _normalize_menu_type(menu_value: Any) -> str:
        if isinstance(menu_value, dict):
            menu_type = menu_value.get("type", "")
        else:
            raw_text = str(menu_value or "").strip()
            match = re.search(
                r"type['\"]?\s*[:=]\s*['\"]?([A-Za-z ]+)",
                raw_text,
                re.IGNORECASE,
            )
            menu_type = match.group(1) if match else raw_text

        normalized = re.sub(r"[^a-z0-9]+", " ", str(menu_type or "").lower()).strip()
        if normalized in {"", "none", "null", "nomenu", "no menu"}:
            return "no menu"
        return normalized

    @classmethod
    def _is_menu_open(cls, menu_value: Any) -> bool:
        return cls._normalize_menu_type(menu_value) not in cls._MENU_CLOSED_TYPES

    @classmethod
    def _is_vllm_service_unavailable_reason(cls, reason: str) -> bool:
        if cls._is_vllm_transport_failure_reason(reason):
            return True
        normalized = re.sub(r"\s+", " ", str(reason or "").lower()).strip()
        if not normalized:
            return False
        return any(marker in normalized for marker in cls._VLLM_OUTAGE_MARKERS)

    @classmethod
    def _is_vllm_transport_failure_reason(cls, reason: str) -> bool:
        normalized = re.sub(r"\s+", " ", str(reason or "").lower()).strip()
        if not normalized:
            return False
        return any(marker in normalized for marker in cls._VLLM_TRANSPORT_FAILURE_MARKERS)

    def _maybe_mark_vllm_unavailable(self, *, reason: str, source: str) -> None:
        if not self.vllm_available:
            return
        if not self._is_vllm_transport_failure_reason(reason):
            return

        self.vllm_available = False
        self._vllm_reenable_success_streak = 0
        self._schedule_next_vllm_health_retry()
        reason_excerpt = str(reason or "").strip().replace("\n", " ")[:240]
        logger.warn(
            "[DualBrain] FastLLM temporarily marked unavailable due to upstream transport failure "
            f"(source={source}, reason={reason_excerpt}); "
            f"next health retry in {self.vllm_health_retry_seconds:.1f}s"
        )

    @classmethod
    def _suggestion_reuse_is_safe(cls, reason: str) -> bool:
        """Allow suggestion reuse for transport failures and non-semantic parse-style failures."""
        normalized = str(reason or "").strip().lower()
        if not normalized:
            return False

        if cls._is_vllm_transport_failure_reason(normalized):
            return True

        # Unknown vllm escalations should remain conservative.
        if "vllm_escalate:" in normalized:
            return False

        allow_markers = (
            "parse_fallback",
            "fast_llm_empty_action",
            "autonomous_fallback_failed",
        )
        if any(marker in normalized for marker in allow_markers):
            return True

        # Explicit state-grounded invalidity should never resurrect the same suggestion.
        deny_markers = (
            "move_target_blocked",
            "blocked_structure",
            "invalid_target",
            "tool_invalid_target",
            "hoe_invalid_target",
            "tool_mismatch",
            "unsupported_menu",
            "placeable_item_invalid_target",
            "fertilize_target_not_effective",
            "noop_move",
            "failure_detector:",
            "position_mismatch",
            "oscillation",
        )
        if any(marker in normalized for marker in deny_markers):
            return False

        return False

    def _reuse_active_suggestion_as_planned_action(
        self,
        source_state: dict,
        base_state: dict,
        *,
        reason: str,
    ) -> Optional[dict]:
        """Conservatively reuse the current BigBrain suggestion as an executable action."""
        if not self._suggestion_reuse_is_safe(reason):
            return None

        suggestions = source_state.get("suggestions", [])
        if not isinstance(suggestions, list) or not suggestions:
            return None

        try:
            current_step = int(source_state.get("current_step", 0) or 0)
        except (TypeError, ValueError):
            current_step = 0

        if not (0 <= current_step < len(suggestions)):
            return None
        completed_steps = source_state.get("completed_steps", [])
        if isinstance(completed_steps, list) and current_step in completed_steps:
            return None

        candidate_index = current_step
        candidate = suggestions[candidate_index]
        if isinstance(candidate, dict):
            raw_action = str(candidate.get("action", "") or "").strip()
            raw_reason = str(candidate.get("reason", "") or "").strip()
        else:
            raw_action = str(candidate or "").strip()
            raw_reason = ""

        action = self.little_brain._sanitize_action(raw_action)
        if not action:
            return None

        suggestion_reason = raw_reason or reason
        logger.warn(
            "[DualBrain] Reusing active BigBrain suggestion as executable fallback: "
            f"{action} | trigger={reason}"
        )

        execution_log = base_state.get("execution_log")
        if not isinstance(execution_log, list):
            execution_log = list(source_state.get("execution_log", []))

        return {
            **base_state,
            "suggestions": [{"action": action, "reason": suggestion_reason}],
            "planned_actions": [action],
            "context_summary": base_state.get("context_summary", source_state.get("context_summary", "")),
            "current_step": 1,
            "execution_log": execution_log,
            "brain_mode": "little",
            "action_source": "big_brain_suggestion_reuse",
            "escalation_reason": "",
            "completed_steps": [],
            "has_execution_feedback": False,
            "execution_pending": False,
            "pending_action": action,
            "pending_step_index": 0,
            "pending_suggested_action": action,
            "force_big_brain_replan": False,
            "allow_suggestion_execution_fallback": False,
            "fail_level": "",
            "fail_score": 0.0,
            "failure_reasons": [],
            "decision_trace": "",
            "success": None,
        }

    def _run_big_brain_only_step(self, state: dict, workflow_config: dict) -> dict:
        pre_execution_failure = self._evaluate_latest_execution_feedback(
            state=state,
            brain_mode="big",
        )
        if pre_execution_failure is not None:
            state["fail_level"] = pre_execution_failure.level
            state["fail_score"] = pre_execution_failure.score
            state["failure_reasons"] = pre_execution_failure.reasons
            state["decision_trace"] = pre_execution_failure.decision_trace
            if pre_execution_failure.level in ("F2", "F3"):
                action = str(state.get("last_action", "") or "")
                if action:
                    self.big_brain.record_failed_action(action)

        t0 = time.time()
        self._apply_big_brain_backoff_if_needed()
        result_state = self._run_big_brain(state, workflow_config)
        elapsed_ms = (time.time() - t0) * 1000.0

        result_state["brain_mode"] = "big"
        result_state["big_brain_only"] = True
        result_state["dual_brain_enabled"] = True
        result_state.setdefault("action_source", "big_brain_only")

        logger.write(
            "[DualBrain] Step done: mode=big-brain-only, "
            f"elapsed={elapsed_ms:.0f}ms, "
            f"actions={len(result_state.get('planned_actions', []) or [])}"
        )
        return result_state

    def step(self, state: dict, workflow_config: dict) -> dict:
        """Execute one dual-brain step.

        This is the main entry point called by the runner on each iteration.

        Args:
            state: Current GameState dict.
            workflow_config: LangGraph config (thread_id etc).

        Returns:
            Updated state dict after execution.
        """
        if not self.big_brain_only:
            self._maybe_refresh_vllm_availability()

        # 1. Detect environment change from screenshot
        screenshot_path = state.get("screenshot_path", "")
        env_changed = False
        env_change_score = 0.0
        if screenshot_path and isinstance(screenshot_path, str):
            try:
                env_changed, env_change_score = self.env_detector.detect_change(
                    screenshot_path
                )
            except Exception as e:
                logger.debug(f"[DualBrain] Env detection failed: {e}")

        # Inject detection results into state
        state["env_changed"] = env_changed
        state["env_change_score"] = env_change_score
        state["vllm_available"] = self.vllm_available
        state["dual_brain_enabled"] = True
        state["big_brain_only"] = self.big_brain_only

        # Phase 6.2/6.4: Reset failed action/plan buffers when state changed
        if state.get("last_state_changed", False):
            self.big_brain.clear_failed_actions()
            self.big_brain.clear_plan_failure()
            self._forced_escape_count = 0

        if self.big_brain_only:
            return self._run_big_brain_only_step(state, workflow_config)

        # Phase 6.5: Stuck detection - bypass all LLM planning
        if self._is_definitively_stuck(state):
            forced = self._generate_escape_action(state)
            self._forced_escape_count += 1
            logger.warn(f"[DualBrain] STUCK - forcing escape action: {forced} (attempt {self._forced_escape_count})")
            return {
                **{k: v for k, v in state.items() if k not in (
                    "planned_actions", "suggestions", "brain_mode",
                    "current_step", "escalation_reason",
                )},
                "planned_actions": [forced],
                "suggestions": [{"action": forced, "reason": "forced_escape"}],
                "brain_mode": "forced",
                "current_step": 0,
                "success": True,
                "escalation_reason": f"forced_escape_{self._forced_escape_count}",
                "execution_pending": False,
                "pending_action": "",
                "pending_step_index": None,
                "pending_suggested_action": "",
                "force_big_brain_replan": False,
                "has_execution_feedback": False,
            }

        # 2. Ask scheduler which brain to use
        brain_mode = self.scheduler.decide(state)
        pre_execution_failure = self._evaluate_latest_execution_feedback(
            state=state,
            brain_mode=brain_mode,
        )
        if pre_execution_failure is not None:
            state["fail_level"] = pre_execution_failure.level
            state["fail_score"] = pre_execution_failure.score
            state["failure_reasons"] = pre_execution_failure.reasons
            state["decision_trace"] = pre_execution_failure.decision_trace

            if brain_mode == "little" and pre_execution_failure.should_escalate:
                logger.write(
                    "[DualBrain] Escalating before little-brain execution "
                    f"based on latest execution feedback: {pre_execution_failure.decision_trace}"
                )
                brain_mode = "big"
                state["escalation_reason"] = (
                    f"failure_detector:{pre_execution_failure.level}"
                )

        t0 = time.time()

        if brain_mode == "big":
            self._apply_big_brain_backoff_if_needed()
            result_state = self._run_big_brain(state, workflow_config)
            result_state = self._handoff_big_plan_to_little_brain(result_state)
        else:
            result_state = self._run_little_brain(state)

        elapsed_ms = (time.time() - t0) * 1000.0

        if result_state.get("execution_pending", False):
            result_state["has_execution_feedback"] = False
            logger.write(
                f"[DualBrain] Step done: mode={brain_mode}, "
                f"elapsed={elapsed_ms:.0f}ms, success=pending, fail=NA"
            )
            logger.write(
                "[DualBrain] External action dispatched; awaiting real "
                "execution feedback before replanning"
            )
            return result_state

        # For little brain, use LLM-only latency for failure detection
        # to avoid counting game I/O (~7s) as timeout
        failure_elapsed_ms = elapsed_ms
        if brain_mode == "little":
            failure_elapsed_ms = result_state.get("llm_elapsed_ms", elapsed_ms)

        # 3. Failure detection (Phase 3.3 integration)
        if pre_execution_failure is not None and not getattr(self.little_brain, "execute_internally", False):
            failure_result = pre_execution_failure
        else:
            failure_result = self._evaluate_latest_execution_feedback(
                state=result_state,
                brain_mode=brain_mode,
                elapsed_ms=failure_elapsed_ms,
            )

        if failure_result is not None:
            result_state["fail_level"] = failure_result.level
            result_state["fail_score"] = failure_result.score
            result_state["failure_reasons"] = failure_result.reasons
            result_state["decision_trace"] = failure_result.decision_trace

            # Phase 6.2: Record failed action for BigBrain interception
            # Only record on hard failures (F2/F3). F1 is too common (every
            # repositioning move triggers it via no_progress) and would fill
            # the buffer with legitimate moves.
            if failure_result.level in ("F2", "F3"):
                action = str(state.get("last_action", "") or "")
                if action:
                    self.big_brain.record_failed_action(action)
            elif failure_result.level == "F1":
                # For F1, only record if there's actual error info (e.g. blocked move)
                errors_info = str(state.get("last_errors_info", "") or "").strip()
                if errors_info:
                    action = str(state.get("last_action", "") or "")
                    if action:
                        self.big_brain.record_failed_action(action)

            # Phase 6.4: Record failed plan for similarity detection
            if brain_mode == "big" and failure_result.level in ("F1", "F2", "F3"):
                suggestions = state.get("suggestions", [])
                if suggestions:
                    plan_actions = [s.get("action", "") for s in suggestions if isinstance(s, dict)]
                    if plan_actions:
                        self.big_brain.mark_plan_failed(plan_actions)

        # 4. Log step metrics
        actual_mode = result_state.get("brain_mode", brain_mode)
        logger.write(
            f"[DualBrain] Step done: mode={brain_mode}, "
            f"elapsed={elapsed_ms:.0f}ms, "
            f"success={result_state.get('success', '?')}, "
            f"fail={result_state.get('fail_level', 'NA')}"
        )

        # 5. Handle little brain escalation to big brain
        if (
            brain_mode == "little"
            and failure_result is not None
            and failure_result.should_escalate
            and result_state.get("brain_mode") != "big"
        ):
            result_state["brain_mode"] = "big"
            result_state["escalation_reason"] = (
                f"failure_detector:{failure_result.level}"
            )
            result_state["completed_steps"] = list(
                range(int(result_state.get("current_step", 0)))
            )

        if (
            brain_mode == "little"
            and result_state.get("brain_mode") == "big"
        ):
            escalation_reason = str(result_state.get("escalation_reason", "") or "")
            is_transport_failure = self._is_vllm_transport_failure_reason(escalation_reason)
            self._maybe_mark_vllm_unavailable(
                reason=escalation_reason,
                source="little_brain_escalation",
            )

            recovered_state = self._reuse_active_suggestion_as_planned_action(
                source_state=state,
                base_state=result_state,
                reason=escalation_reason or "little_brain_escalated",
            )
            if recovered_state is not None:
                if is_transport_failure:
                    logger.warn(
                        "[DualBrain] LittleBrain transport failure -> reusing current "
                        "BigBrain suggestion for execution"
                    )
                return recovered_state

            if is_transport_failure:
                logger.warn(
                    "[DualBrain] LittleBrain transport failure -> forcing BigBrain replan "
                    "(no reusable current suggestion)"
                )
            else:
                logger.warn("[DualBrain] LittleBrain semantic escalation -> forcing BigBrain replan")

            logger.write(
                f"[DualBrain] Little brain escalated: "
                f"{result_state.get('escalation_reason', '?')}"
            )
            # Merge little brain partial result into state, then run big brain
            merged = {**state, **result_state}
            replanned_state = self._run_big_brain(merged, workflow_config)
            result_state = self._handoff_big_plan_to_little_brain(replanned_state)

        return result_state

    def _handoff_big_plan_to_little_brain(self, state: dict) -> dict:
        """BigBrain only produces suggestions; LittleBrain chooses the action."""
        if not isinstance(state, dict):
            return state

        if state.get("action_source") in {
            "little_brain_autonomous_fallback",
            "big_brain_suggestion_reuse",
            "deterministic_fallback_no_fastllm",
            "deterministic_fallback",
        }:
            return state

        suggestions = state.get("suggestions", [])
        if not suggestions:
            return state

        logger.write(
            "[DualBrain] Big brain plan ready; handing off to LittleBrain "
            "for the immediate action"
        )
        handoff_state = {**state}
        handoff_state["brain_mode"] = "little"
        handoff_state["current_step"] = int(handoff_state.get("current_step", 0) or 0)
        return self._run_little_brain(handoff_state)

    def _evaluate_latest_execution_feedback(
        self,
        state: dict,
        brain_mode: str,
        elapsed_ms: Optional[float] = None,
    ):
        has_execution_feedback = bool(state.get("has_execution_feedback", False))
        if not has_execution_feedback:
            return None

        exec_info = state.get("last_exec_info", {})
        if not isinstance(exec_info, dict):
            exec_info = {}

        action = str(state.get("last_action", "") or "")
        previous_progress = state.get(
            "previous_task_progress",
            state.get("previous_task_progress_quantity"),
        )
        current_progress = state.get("task_progress")
        if current_progress is None:
            current_progress = state.get("task_progress_quantity")

        return self.failure_detector.evaluate(
            exec_info=exec_info,
            action=action,
            elapsed_ms=0.0 if elapsed_ms is None else elapsed_ms,
            brain_mode=brain_mode,
            previous_actions=state.get("previous_actions", []),
            state_changed=bool(state.get("last_state_changed", state.get("success", False))),
            previous_progress=previous_progress,
            current_progress=current_progress,
            consecutive_zero_progress=int(state.get("zero_progress_streak", 0) or 0),
            repeated_action_streak=int(state.get("repeated_action_streak", 0) or 0),
            position_issue_detected=bool(state.get("position_issue_detected", False)),
            oscillation_streak=int(state.get("oscillation_streak", 0) or 0),
        )

    def _run_big_brain(self, state: dict, workflow_config: dict) -> dict:
        """Run big brain (full LangGraph workflow)."""
        logger.write("[DualBrain] === Big Brain ===")

        # Sync latest screenshot into LocalMemory so that info_gathering
        # and self_reflection see the most recent frame, not a stale one
        # from a previous big brain cycle.
        from cradle.memory import LocalMemory
        from cradle import constants
        mem = LocalMemory()
        if state.get("screenshot_path"):
            mem.update_info_history({
                "screenshot_path": state["screenshot_path"],
                constants.IMAGES_MEM_BUCKET: state["screenshot_path"],
            })

        result_state, plan_result = self.big_brain.plan(
            state, workflow_config
        )

        # Check if BigBrain produced an empty plan (likely API timeout)
        is_empty_plan = (
            not plan_result.suggestions
            or (
                len(plan_result.suggestions) == 1
                and plan_result.suggestions[0].get("reason") == "no_plan_available"
            )
        )

        if is_empty_plan:
            self._schedule_big_brain_backoff("empty_plan")
            if self.big_brain_only:
                logger.warn(
                    "[DualBrain] BigBrain-only mode produced no executable plan; "
                    "returning no action for this step"
                )
                return {
                    **result_state,
                    "suggestions": [],
                    "planned_actions": [],
                    "context_summary": self.big_brain._build_context_summary(
                        result_state, state
                    ),
                    "current_step": 0,
                    "execution_log": [],
                    "brain_mode": "big",
                    "action_source": "big_brain_only_empty_plan",
                    "escalation_reason": "big_brain_empty_plan",
                    "completed_steps": [],
                    "has_execution_feedback": False,
                    "execution_pending": False,
                    "pending_action": "",
                    "pending_step_index": None,
                    "pending_suggested_action": "",
                    "force_big_brain_replan": False,
                }
            if self.vllm_available:
                logger.warn(
                    "[DualBrain] BigBrain produced empty plan "
                    "(likely API timeout), invoking LittleBrain autonomous fallback"
                )
                fallback = self._little_brain_autonomous_fallback(state, result_state)
                fallback["has_execution_feedback"] = False  # prevent stale feedback re-evaluation
                return fallback

            logger.warn(
                "[DualBrain] BigBrain produced empty plan and fast LLM is unavailable; "
                "using deterministic fallback instead of returning an empty plan"
            )
            deterministic_action = self._build_deterministic_fallback_action(state)
            if deterministic_action:
                return {
                    **result_state,
                    "suggestions": [{"action": deterministic_action, "reason": "deterministic_fallback_no_fastllm"}],
                    "planned_actions": [deterministic_action],
                    "context_summary": self.big_brain._build_context_summary(
                        result_state, state
                    ),
                    "current_step": 0,
                    "execution_log": [],
                    "brain_mode": "big",
                    "action_source": "deterministic_fallback_no_fastllm",
                    "escalation_reason": "",
                    "completed_steps": [],
                    "has_execution_feedback": False,
                    "execution_pending": False,
                    "pending_action": "",
                    "pending_step_index": None,
                    "pending_suggested_action": "",
                    "force_big_brain_replan": False,
                }
            return {
                **result_state,
                "suggestions": [],
                "context_summary": self.big_brain._build_context_summary(
                    result_state, state
                ),
                "current_step": 0,
                "execution_log": [],
                "brain_mode": "big",
                "escalation_reason": "big_brain_empty_plan",
                "completed_steps": [],
                "has_execution_feedback": False,  # prevent stale feedback re-evaluation
                "execution_pending": False,
                "pending_action": "",
                "pending_step_index": None,
                "pending_suggested_action": "",
                "force_big_brain_replan": False,
            }

        # Get state update from plan
        plan_update = self.big_brain.get_plan_as_state_update(plan_result)
        if self.big_brain_only:
            plan_update["brain_mode"] = "big"
            plan_update["action_source"] = "big_brain_only"

        # Merge: workflow result + plan metadata
        merged = {**result_state, **plan_update}

        # Reset scheduler counter
        self.scheduler.reset_counter()

        # Only clear failure history after a genuine execution success.
        # Replanning after no-progress / blocked steps must preserve counters,
        # or the detector can never escalate persistent livelocks.
        if self._latest_execution_counts_as_success(state):
            self.failure_detector.on_big_brain_success()

        # Reset env detector baseline
        self.env_detector.reset()

        if self.big_brain_only:
            return merged

        # Update little brain with new plan
        self.little_brain.load_plan(
            suggestions=plan_result.suggestions,
            context_summary=plan_result.context_summary,
            current_task=plan_result.current_task,
        )

        return merged

    @staticmethod
    def _latest_execution_counts_as_success(state: dict) -> bool:
        if not isinstance(state, dict):
            return False
        if not bool(state.get("has_execution_feedback", False)):
            return False
        if state.get("latest_task_eval", {}).get("completed") is True:
            return True
        progress_delta = state.get("task_progress_delta", None)
        if progress_delta not in (None, "", 0, 0.0):
            return True
        if bool(state.get("last_state_changed", False)):
            return True
        return False

    def _little_brain_autonomous_fallback(self, state: dict, result_state: dict) -> dict:
        """When BigBrain fails (timeout/empty plan), use LittleBrain to make autonomous decisions.

        LittleBrain decides purely based on current game state observations,
        without any stale BigBrain suggestions.
        """
        # Extract whatever context BigBrain managed to gather before timeout
        context_summary = self.big_brain._build_context_summary(result_state, state)
        if not context_summary:
            context_summary = state.get("context_summary", "")

        current_task = (
            result_state.get("subtask_description")
            or state.get("subtask_description")
            or state.get("task", "")
        )

        # Tell LittleBrain to decide on its own (empty suggestion = full autonomy)
        autonomous_suggestion = {
            "action": "",
            "reason": "big_brain_unavailable_decide_freely",
        }

        decision = self.little_brain.vllm_client.decide(
            context_summary=context_summary,
            suggestion=autonomous_suggestion,
            execution_log=self.little_brain.execution_log,
            mem0_reference="",
            step=0,
            total_steps=1,
            skill_list=state.get("skill_library", ""),
            game_state=state,
        )

        if decision.escalate or not decision.action:
            if decision.escalate:
                self._maybe_mark_vllm_unavailable(
                    reason=str(decision.reason or ""),
                    source="little_brain_autonomous_fallback",
                )

            failure_reason = str(decision.reason or "autonomous_fallback_failed").strip()
            recovered_state = self._reuse_active_suggestion_as_planned_action(
                source_state=state,
                base_state=result_state,
                reason=f"little_brain_autonomous_fallback_failed:{failure_reason}",
            )
            if recovered_state is not None:
                return recovered_state

            # Deterministic fallback: instead of wasting the step entirely,
            # generate a safe exploratory action based on task type and state.
            deterministic_action = self._build_deterministic_fallback_action(state)
            if deterministic_action:
                logger.warn(
                    "[DualBrain] LittleBrain autonomous also failed; "
                    f"using deterministic fallback: {deterministic_action}"
                )
                suggestions = [{"action": deterministic_action, "reason": "deterministic_fallback"}]
                self.little_brain.load_plan(suggestions, context_summary, current_task)
                plan_result = BrainPlanResult(
                    suggestions=suggestions,
                    context_summary=context_summary,
                    current_task=current_task,
                )
                plan_update = self.big_brain.get_plan_as_state_update(plan_result)
                merged = {**result_state, **plan_update}
                merged["planned_actions"] = [deterministic_action]
                merged["action_source"] = "deterministic_fallback"
                merged["big_brain_backoff_seconds"] = self.big_brain_backoff_seconds
                return merged

            logger.warn(
                "[DualBrain] LittleBrain autonomous also failed; "
                "returning control to BigBrain instead of issuing nop()"
            )
            return {
                **result_state,
                "suggestions": [],
                "planned_actions": [],
                "context_summary": context_summary,
                "current_step": 0,
                "execution_log": list(self.little_brain.execution_log),
                "brain_mode": "big",
                "action_source": "little_brain_autonomous_fallback_failed",
                "big_brain_backoff_seconds": self.big_brain_backoff_seconds,
                "escalation_reason": f"little_brain_autonomous_fallback_failed:{failure_reason}",
                "completed_steps": [],
                "has_execution_feedback": False,
                "execution_pending": False,
                "pending_action": "",
                "pending_step_index": None,
                "pending_suggested_action": "",
                "force_big_brain_replan": False,
                "success": False,
            }

        action = self.little_brain._sanitize_action(decision.action)
        reason = decision.reason
        logger.write(f"[DualBrain] LittleBrain autonomous decided: {action} ({reason})")

        suggestions = [{"action": action, "reason": reason}]
        self.little_brain.load_plan(suggestions, context_summary, current_task)

        plan_result = BrainPlanResult(
            suggestions=suggestions,
            context_summary=context_summary,
            current_task=current_task,
        )
        plan_update = self.big_brain.get_plan_as_state_update(plan_result)

        merged = {**result_state, **plan_update}
        merged["planned_actions"] = [action]
        merged["action_source"] = "little_brain_autonomous_fallback"
        merged["big_brain_backoff_seconds"] = self.big_brain_backoff_seconds

        return merged

    def _run_little_brain(self, state: dict) -> dict:
        """Run little brain (fast LLM path)."""
        logger.write("[DualBrain] === Little Brain ===")

        result = self.little_brain.execute(state)

        # Increment scheduler counter
        self.scheduler.increment_counter()

        return result

    @classmethod
    def _build_deterministic_fallback_action(cls, state: dict) -> str:
        """Generate a safe deterministic action when both BigBrain and LittleBrain fail.

        Uses task text and surroundings to pick a reasonable exploratory move
        rather than wasting the entire step.
        """
        if not isinstance(state, dict):
            return ""

        task_text = str(
            state.get("task", "")
            or state.get("subtask_description", "")
            or state.get("task_description", "")
        ).lower()

        gathered = state.get("gathered_info", {})
        if not isinstance(gathered, dict):
            gathered = {}

        # If a menu is open, try closing it first
        current_menu = (
            gathered.get("current_menu")
            or gathered.get("CurrentMenuData")
            or state.get("current_menu")
            or ""
        )
        if cls._is_menu_open(current_menu):
            return 'menu(option="close")'

        # Try to find an open ground tile in surroundings to move to
        surroundings = gathered.get("surroundings", state.get("surroundings", {}))
        open_moves: list[tuple[int, int, int]] = []
        if isinstance(surroundings, dict):
            _BLOCKED_KEYWORDS = ("farmhouse", "barn", "coop", "water", "fence", "wall", "rock", "stone", "tree")
            for key, val in surroundings.items():
                try:
                    if isinstance(key, (list, tuple)) and len(key) == 2:
                        cx, cy = int(key[0]), int(key[1])
                    else:
                        continue
                except (ValueError, TypeError):
                    continue
                if cx == 0 and cy == 0:
                    continue
                val_lower = str(val or "").strip().lower()
                if not val_lower or val_lower in ("empty", "grass", "dirt", "hoedirt", "floor"):
                    open_moves.append((abs(cx) + abs(cy), cx, cy))
                elif not any(kw in val_lower for kw in _BLOCKED_KEYWORDS):
                    open_moves.append((abs(cx) + abs(cy) + 10, cx, cy))

        if open_moves:
            open_moves.sort()
            _, bx, by = open_moves[0]
            return f"move(x={bx}, y={by})"

        # Last resort: try a small exploratory move based on task keywords
        if any(kw in task_text for kw in ("barn", "milk", "hay", "feed", "animal")):
            return "move(x=0, y=-3)"  # Barns are typically north
        if any(kw in task_text for kw in ("coop", "egg", "chicken", "incubat")):
            return "move(x=3, y=-3)"  # Coops are typically north-east
        if any(kw in task_text for kw in ("shop", "buy", "store", "pierre")):
            return "move(x=0, y=5)"  # Town is typically south
        if any(kw in task_text for kw in ("pet", "bowl", "cat", "dog")):
            return "move(x=2, y=1)"  # Pet bowl is typically near farmhouse entrance

        # Generic: move south (toward farm exit / center)
        return "move(x=0, y=3)"

    def get_status(self) -> dict:
        """Return controller status for debugging."""
        return {
            "vllm_available": self.vllm_available,
            "scheduler": self.scheduler.get_status(),
            "little_brain": self.little_brain.get_status(),
            "failure_detector": self.failure_detector.get_status(),
        }
