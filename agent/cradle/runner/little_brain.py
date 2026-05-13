"""
Little brain (System 1) for dual-brain architecture (Phase 3.2).

Autonomous fast-path decision maker using fast LLM (nothinking mode).
References big brain suggestions but can override them based on
execution history and current context.
"""
import re
import time
from typing import Any, Dict, List, Optional

from cradle.log import Logger
from cradle.runner.vllm_client import VLLMClient, VLLMDecision

logger = Logger()


class LittleBrain:
    """Autonomous little brain that executes fast decisions.

    Workflow per step:
        1. Environment change detection (via env_detector, done in controller)
        2. Mem0 retrieval (optional, as reference)
        3. Fast LLM autonomous decision (may override suggestion)
        4. Skill execution via SkillExecuteProvider
        5. Record execution log, update state
    """

    def __init__(
        self,
        vllm_client: VLLMClient,
        mem0_provider: Optional[Any] = None,
        skill_execute_provider: Optional[Any] = None,
        gm: Optional[Any] = None,
        augment_provider: Optional[Any] = None,
        execute_internally: bool = True,
        max_relative_move: int = 20,
    ):
        """
        Args:
            vllm_client: VLLMClient instance for decisions.
            mem0_provider: Mem0Provider for memory retrieval (optional).
            skill_execute_provider: SkillExecuteProvider for action execution.
            gm: GameManager for screenshots.
            augment_provider: AugmentProvider for image augmentation.
        """
        self.vllm_client = vllm_client
        self.mem0_provider = mem0_provider
        self.skill_execute_provider = skill_execute_provider
        self.gm = gm
        self.augment_provider = augment_provider
        self.execute_internally = execute_internally
        self.max_relative_move = max(1, int(max_relative_move))

        # Current plan state (loaded from big brain)
        self.suggestions: List[Dict[str, str]] = []
        self.context_summary: str = ""
        self.current_task: str = ""
        self.execution_log: List[Dict[str, Any]] = []

    def load_plan(
        self,
        suggestions: List[Dict[str, str]],
        context_summary: str,
        current_task: str,
    ):
        """Load a new plan from big brain.

        Called after big brain produces its output. Resets execution log
        for the new cycle.
        """
        self.suggestions = suggestions
        self.context_summary = context_summary
        self.current_task = current_task
        self.execution_log = []  # New cycle, clear log
        logger.write(
            f"[LittleBrain] Plan loaded: {len(suggestions)} suggestions, "
            f"task={current_task}"
        )

    def execute(self, state: dict) -> dict:
        """Execute one little-brain step.

        Args:
            state: Current GameState dict. Must contain:
                - current_step: int (0-based step index)
                - suggestions: List[Dict[str, str]]
                - context_summary: str
                - consecutive_failures: int
                - skill_library: str (brief skill list)

        Returns:
            Updated state dict with execution results.
        """
        step = state.get("current_step", 0)
        suggestions = state.get("suggestions", self.suggestions)
        context_summary = state.get("context_summary", self.context_summary)
        consecutive_failures = state.get("consecutive_failures", 0)

        if step >= len(suggestions):
            # No more suggestions -> escalate
            return self._escalate(
                state, step, "no_more_suggestions",
            )

        suggestion = suggestions[step]
        logger.write(
            f"[LittleBrain] Step {step + 1}/{len(suggestions)}: "
            f"suggestion={suggestion.get('action', '?')}"
        )

        # Short-circuit for padding no-op from big brain plan
        suggested_action = str(suggestion.get("action", "")).strip()
        if suggested_action.replace(" ", "").lower() == "nop()":
            logger.write("[LittleBrain] NOP suggestion -> short-circuit success")

            prev_actions = list(state.get("previous_actions", []))
            prev_actions.append("nop()")

            frame_ids = state.get("executed_frames", state.get("frame_ids", (0, 0)))
            if not isinstance(frame_ids, tuple) or len(frame_ids) != 2:
                frame_ids = (0, 0)

            self.execution_log.append({
                "step": step,
                "action": "nop()",
                "suggested_action": suggestion.get("action", ""),
                "success": True,
                "note": "short_circuit_nop",
            })

            updated = {
                "current_step": step + 1,
                "suggestions": suggestions,
                "context_summary": context_summary,
                "execution_log": list(self.execution_log),
                "consecutive_failures": 0,
                "success": True,
                "brain_mode": "little",
                "planned_actions": ["nop()"],
                "exec_info": {
                    "done": True,
                    "executed_skills": [],
                    "errors": False,
                    "errors_info": "",
                },
                "executed_frames": frame_ids,
                "execution_result": {
                    "success": True,
                    "frame_ids": frame_ids,
                    "error": None,
                },
                "previous_actions": prev_actions,
                "task": state.get("task", self.current_task),
                "llm_elapsed_ms": 0.0,
                "execution_pending": False,
                "pending_action": "",
                "pending_step_index": None,
                "pending_suggested_action": "",
                "force_big_brain_replan": False,
                "has_execution_feedback": False,
            }

            if step + 1 >= len(suggestions):
                logger.write("[LittleBrain] Cycle complete -> trigger big brain")
                updated["brain_mode"] = "big"
                updated["escalation_reason"] = "cycle_complete"
                updated["completed_steps"] = list(range(step + 1))

            return updated

        # 1. Mem0 retrieval (optional, as reference)
        mem0_ref = self._get_mem0_reference(state, context_summary, suggestion)

        # 2. Fast LLM autonomous decision
        llm_t0 = time.time()
        decision = self.vllm_client.decide(
            context_summary=context_summary,
            suggestion=suggestion,
            execution_log=self.execution_log,
            mem0_reference=mem0_ref,
            step=step,
            total_steps=len(suggestions),
            skill_list=state.get("skill_library", ""),
            game_state=state,
        )
        effective_duration_s = float(
            getattr(self.vllm_client, "last_effective_duration_s", 0.0) or 0.0
        )
        llm_elapsed_ms = (
            effective_duration_s * 1000.0
            if effective_duration_s > 0
            else (time.time() - llm_t0) * 1000.0
        )

        # 3. Handle ESCALATE
        if decision.escalate:
            logger.write(
                f"[LittleBrain] Fast LLM escalated: {decision.reason}"
            )
            esc_state = self._escalate(
                state, step, f"vllm_escalate: {decision.reason}",
            )
            esc_state["llm_elapsed_ms"] = llm_elapsed_ms
            return esc_state

        # 4. Execute action via SkillExecuteProvider (or skip if external execution)
        action = self._sanitize_action(decision.action)
        if not action:
            esc_state = self._escalate(state, step, "fast_llm_empty_action")
            esc_state["llm_elapsed_ms"] = llm_elapsed_ms
            return esc_state

        action = self._guard_opposite_move(action, suggested_action, decision.reason, state)
        action = self._preserve_placeable_reposition_move(state, action, suggested_action)
        action = self._rewrite_known_stall_move(state, action, decision.reason)
        action = self._sanitize_action(action)

        stall_guard_reason = self._guard_against_known_stalls(
            state=state,
            action=action,
        )
        if stall_guard_reason:
            esc_state = self._escalate(state, step, stall_guard_reason)
            esc_state["llm_elapsed_ms"] = llm_elapsed_ms
            return esc_state

        external_execution = not self.execute_internally
        if self.execute_internally:
            exec_result = self._execute_skill(action, state)
        else:
            exec_result = None

        if external_execution:
            logger.write(f"[LittleBrain] Step {step + 1} dispatched for external execution")

            updated = {
                "current_step": step + 1,
                "suggestions": suggestions,
                "context_summary": context_summary,
                "execution_log": list(self.execution_log),
                "consecutive_failures": state.get("consecutive_failures", 0),
                "success": None,
                "brain_mode": "little",
                "planned_actions": [action],
                "exec_info": {},
                "executed_frames": (0, 0),
                "execution_result": {
                    "success": None,
                    "frame_ids": (0, 0),
                    "error": None,
                    "pending": True,
                },
                "previous_actions": list(state.get("previous_actions", [])),
                "task": state.get("task", self.current_task),
                "llm_elapsed_ms": llm_elapsed_ms,
                "execution_pending": True,
                "pending_action": action,
                "pending_step_index": step,
                "pending_suggested_action": suggestion.get("action", ""),
                "force_big_brain_replan": False,
                "has_execution_feedback": False,
            }

            return updated

        # 5. Record execution log
        success = exec_result.get("success", False)
        exec_info_data = exec_result.get("exec_info", {})
        errors_info = ""
        if isinstance(exec_info_data, dict):
            errors_info = exec_info_data.get("errors_info", "")
        self.execution_log.append({
            "step": step,
            "action": action,
            "suggested_action": suggestion.get("action", ""),
            "success": success,
            "note": decision.reason,
            "errors_info": errors_info,
        })

        # 6. Update state
        if not success:
            consecutive_failures += 1
            logger.warn(
                f"[LittleBrain] Step {step + 1} failed "
                f"(consecutive: {consecutive_failures})"
            )
        else:
            consecutive_failures = 0
            logger.write(f"[LittleBrain] Step {step + 1} succeeded")
            # Clear position_issue_detected after successful move
            if action.startswith("move("):
                pass  # propagated via updated dict below

        # 7. Accumulate previous_actions history
        prev_actions = list(state.get("previous_actions", []))
        prev_actions.append(action)

        # 8. Build execution_result (matching skill_execute_node format)
        frame_ids = exec_result.get("frame_ids", (0, 0))
        execution_result = {
            "success": success,
            "frame_ids": frame_ids,
            "error": None if success else "execution_failed",
        }

        # Build updated state
        updated = {
            "current_step": step + 1,
            "suggestions": suggestions,  # Preserve for next iteration
            "context_summary": context_summary,
            "execution_log": list(self.execution_log),
            "consecutive_failures": consecutive_failures,
            "success": success,
            "brain_mode": "little",
            "planned_actions": [action],
            "exec_info": exec_result.get("exec_info", {}),
            "executed_frames": frame_ids,
            "execution_result": execution_result,
            "previous_actions": prev_actions,
            "task": state.get("task", self.current_task),
            "llm_elapsed_ms": llm_elapsed_ms,
            "execution_pending": False,
            "pending_action": "",
            "pending_step_index": None,
            "pending_suggested_action": "",
            "force_big_brain_replan": False,
            "has_execution_feedback": False,
        }

        # Clear position_issue_detected after a successful move
        if success and action.startswith("move("):
            updated["position_issue_detected"] = False

        # Update screenshot if captured
        screenshot_path = exec_result.get("screenshot_path")
        if screenshot_path:
            updated["screenshot_path"] = screenshot_path

        # Check if cycle is complete
        if step + 1 >= len(suggestions):
            logger.write("[LittleBrain] Cycle complete -> trigger big brain")
            updated["brain_mode"] = "big"
            updated["escalation_reason"] = "cycle_complete"
            updated["completed_steps"] = list(range(step + 1))

        return updated

    def _escalate(self, state: dict, step: int, reason: str) -> dict:
        """Build state update that escalates to big brain."""
        logger.write(f"[LittleBrain] Escalating: {reason}")
        return {
            "brain_mode": "big",
            "escalation_reason": reason,
            "completed_steps": list(range(step)),
            "suggestions": state.get("suggestions", self.suggestions),
            "context_summary": state.get("context_summary", self.context_summary),
            "execution_log": list(self.execution_log),
            "current_step": step,
            "consecutive_failures": state.get("consecutive_failures", 0),
            "execution_pending": False,
            "pending_action": "",
            "pending_step_index": None,
            "pending_suggested_action": "",
            "force_big_brain_replan": False,
            "has_execution_feedback": False,
        }

    def record_external_execution_feedback(
        self,
        *,
        action: str,
        success: bool,
        errors_info: str = "",
        step: Optional[int] = None,
        suggested_action: str = "",
        state_changed: Optional[bool] = None,
        uncertain_execution: Optional[bool] = None,
        heightened_failure_signal: Optional[bool] = None,
        progress_delta: Any = None,
        progress_quantity: Any = None,
    ) -> None:
        """Store real execution feedback when env-layer execution is external."""
        normalized_action = self._sanitize_action(str(action or "").strip())
        if not normalized_action:
            return

        normalized_suggestion = str(suggested_action or "").strip()
        normalized_errors = str(errors_info or "")
        last_entry = self.execution_log[-1] if self.execution_log else None
        same_action = (
            isinstance(last_entry, dict)
            and str(last_entry.get("action", "") or "").strip() == normalized_action
        )
        same_step = (
            step is None
            or (
                isinstance(last_entry, dict)
                and last_entry.get("step") == step
            )
        )

        if same_action and same_step:
            entry = last_entry
        else:
            entry = {
                "step": step if step is not None else len(self.execution_log),
                "action": normalized_action,
                "suggested_action": normalized_suggestion,
                "success": bool(success),
                "note": "external_execution_feedback",
                "errors_info": normalized_errors,
            }
            if state_changed is not None:
                entry["state_changed"] = bool(state_changed)
            if uncertain_execution is not None:
                entry["uncertain_execution"] = bool(uncertain_execution)
            if heightened_failure_signal is not None:
                entry["heightened_failure_signal"] = bool(heightened_failure_signal)
            if progress_delta is not None:
                entry["progress_delta"] = progress_delta
            if progress_quantity is not None:
                entry["progress_quantity"] = progress_quantity
            self.execution_log.append(entry)
            return

        entry["action"] = normalized_action
        if step is not None:
            entry["step"] = step
        if normalized_suggestion:
            entry["suggested_action"] = normalized_suggestion
        entry["success"] = bool(success)
        entry["errors_info"] = normalized_errors
        if state_changed is not None:
            entry["state_changed"] = bool(state_changed)
        if uncertain_execution is not None:
            entry["uncertain_execution"] = bool(uncertain_execution)
        if heightened_failure_signal is not None:
            entry["heightened_failure_signal"] = bool(heightened_failure_signal)
        if progress_delta is not None:
            entry["progress_delta"] = progress_delta
        if progress_quantity is not None:
            entry["progress_quantity"] = progress_quantity
        if not entry.get("note"):
            entry["note"] = "external_execution_feedback"

    @staticmethod
    def _should_use_mem0_reference(state: Dict[str, Any]) -> bool:
        return True

    @staticmethod
    def _clean_memory_actions(actions: Any) -> List[str]:
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
    def _is_setup_only_memory_actions(actions: List[str]) -> bool:
        if not actions:
            return False
        return all(
            action.lower().startswith(("choose_item(", "attach_item(", "unattach_item("))
            for action in actions
        )

    def _get_mem0_reference(
        self,
        state: Dict[str, Any],
        context_summary: str,
        suggestion: Dict[str, str],
    ) -> str:
        """Retrieve Mem0 memory as reference text."""
        if self.mem0_provider is None:
            return ""
        if not self._should_use_mem0_reference(state):
            return ""

        try:
            task_name = state.get("task") or self.current_task
            query = f"task: {task_name} | {context_summary} {suggestion.get('action', '')}"
            result = self.mem0_provider.retrieve(query)
            confidence = float(result.get("memory_confidence", 0.0) or 0.0)

            hits = result.get("memory_hits", [])
            hits = [
                hit for hit in (hits or [])
                if isinstance(hit, dict) and (
                    bool(hit.get("success", False))
                    or float(hit.get("successes", 0) or 0) > 0
                )
            ]
            if not hits:
                return ""

            top_hit = hits[0]
            state_desc = str(top_hit.get("state", "") or "").replace("\n", " ").strip()
            actions = self._clean_memory_actions(top_hit.get("actions", []))
            metadata = top_hit.get("metadata", {})
            progress = ""
            if isinstance(metadata, dict):
                progress = str(metadata.get("progress", "") or "").strip()
            if not actions or self._is_setup_only_memory_actions(actions):
                return ""
            confidence = max(confidence, 0.01)

            lines = [
                f"  - successful recovery (conf={confidence:.2f}): {actions}",
            ]
            if progress:
                lines.append(f"  - progress: {progress[:80]}")
            if state_desc:
                lines.append(f"  - state: {state_desc[:100]}")
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"[LittleBrain] Mem0 retrieval failed: {e}")
            return ""

    def _sanitize_action(self, action: str) -> str:
        """Normalize action text and clamp oversized relative moves."""
        chosen = str(action or "").strip()
        if not chosen:
            return ""

        limit = self.max_relative_move
        move_match = re.match(
            r"^move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)$",
            chosen,
            re.IGNORECASE,
        )
        if move_match:
            x = int(move_match.group(1))
            y = int(move_match.group(2))
            if abs(x) > limit or abs(y) > limit:
                logger.warn(
                    f"[LittleBrain] Oversized move({x}, {y}) from fast LLM, "
                    "clamping to safe relative move bounds"
                )
                x = max(-limit, min(limit, x))
                y = max(-limit, min(limit, y))
                return f"move(x={x}, y={y})"

        return chosen

    @staticmethod
    def _guard_opposite_move(
        action: str,
        suggested: str,
        reason: str,
        state: Optional[dict] = None,
    ) -> str:
        """Backstop guard for clearly unjustified move overrides.

        Use the same blocker-aware move validation as FastLLM so LittleBrain
        can keep local escape moves when the BigBrain suggestion clearly walks
        into a visible blocker such as a bed, wall, or building tile.
        """
        if not action.startswith("move(") or not suggested.startswith("move("):
            return action

        normalized_reason = str(reason or "").strip().lower()
        if normalized_reason in {
            "blocked_structure_reroute",
            "blocked_recovery",
            "blocked_route_recovery",
        }:
            return action
        try:
            move_allowed, move_reason = VLLMClient._move_override_is_justified(
                suggested_action=suggested,
                action=action,
                game_state=state if isinstance(state, dict) else None,
            )
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.debug(f"[LittleBrain] Override guard validation failed: {exc}")
            return action

        if move_allowed:
            return action

        logger.warn(
            f"[LittleBrain] Override guard: action {action} diverges from BigBrain move "
            f"{suggested} (reason='{reason}', validator='{move_reason}'). "
            "Reverting to BigBrain suggestion."
        )
        return suggested

    @staticmethod
    def _preserve_placeable_reposition_move(
        state: dict,
        action: str,
        suggested: str,
    ) -> str:
        normalized_action = str(action or "").strip()
        normalized_suggested = str(suggested or "").strip()
        if not normalized_suggested.startswith("move("):
            return normalized_action
        if not normalized_action.startswith("interact("):
            return normalized_action

        task_text = str(state.get("main_task", "") or state.get("task", "") or "").lower()
        if "fertilize" not in task_text and "sow" not in task_text:
            return normalized_action

        last_action = str(state.get("last_action", "") or "").strip()
        zero_progress_streak = int(state.get("zero_progress_streak", 0) or 0)
        repeated_action_streak = int(state.get("repeated_action_streak", 0) or 0)
        if last_action != normalized_action:
            return normalized_action
        if zero_progress_streak < 1 and repeated_action_streak < 1:
            return normalized_action

        logger.warn(
            "[LittleBrain] Preserving BigBrain reposition move for placeable task: "
            f"{normalized_action} -> {normalized_suggested}"
        )
        return normalized_suggested

    @staticmethod
    def _rewrite_known_stall_move(
        state: dict,
        action: str,
        reason: str = "",
    ) -> str:
        """Rewrite obviously stale blocked moves before execution.

        Keep the scope intentionally narrow: only rewrite when the immediately
        previous move was explicitly blocked and the new move repeats the same
        axis direction. This avoids mutating legitimate reroutes, especially
        the grounded recovery moves produced by FastLLM itself.
        """
        normalized_action = str(action or "").strip()
        if not normalized_action.startswith("move("):
            return normalized_action

        normalized_reason = str(reason or "").strip().lower()
        if normalized_reason in {
            "blocked_structure_reroute",
            "blocked_recovery",
            "blocked_route_recovery",
        }:
            return normalized_action

        move_match = re.match(
            r"^move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)$",
            normalized_action,
            re.IGNORECASE,
        )
        if not move_match:
            return normalized_action

        x = int(move_match.group(1))
        y = int(move_match.group(2))
        last_action = str(state.get("last_action", "") or "").strip()
        last_errors_info = str(state.get("last_errors_info", "") or "").strip().lower()
        blocked_recently = (
            "blocked by an obstacle" in last_errors_info
            or "path is likely blocked" in last_errors_info
            or "did not change player position" in last_errors_info
        )
        if not blocked_recently:
            return normalized_action

        same_move_as_last = bool(normalized_action == last_action)
        last_move_match = re.match(
            r"^move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)$",
            last_action,
            re.IGNORECASE,
        )
        if not last_move_match:
            return normalized_action

        last_x = int(last_move_match.group(1))
        last_y = int(last_move_match.group(2))

        def _axis_direction(dx: int, dy: int) -> str:
            if dx != 0 and dy == 0:
                return "+x" if dx > 0 else "-x"
            if dy != 0 and dx == 0:
                return "+y" if dy > 0 else "-y"
            if abs(dx) >= abs(dy):
                return "+x" if dx > 0 else "-x"
            return "+y" if dy > 0 else "-y"

        same_axis_direction = _axis_direction(x, y) == _axis_direction(last_x, last_y)
        if not (same_move_as_last or same_axis_direction):
            return normalized_action

        try:
            recovery = str(
                VLLMClient._build_local_route_recovery_action(game_state=state) or ""
            ).strip()
        except Exception as exc:  # pragma: no cover - defensive guardrail
            logger.debug(f"[LittleBrain] Stall route recovery failed: {exc}")
            recovery = ""
        if recovery and recovery != normalized_action:
            logger.warn(
                "[LittleBrain] Rewriting known blocked move into local route recovery: "
                f"{normalized_action} -> {recovery}"
            )
            return recovery

        clamp = 3 if (x == 0 or y == 0) else 2
        clamped_x = max(-clamp, min(clamp, x))
        clamped_y = max(-clamp, min(clamp, y))
        if (clamped_x, clamped_y) != (x, y):
            rewritten = f"move(x={clamped_x}, y={clamped_y})"
            logger.warn(
                "[LittleBrain] Shrinking move stride under stall context: "
                f"{normalized_action} -> {rewritten}"
            )
            return rewritten

        return normalized_action

    def _guard_against_known_stalls(self, state: dict, action: str) -> str:
        """Stop obviously stale suggestions before they reach the env."""
        normalized_action = str(action or "").strip()
        if not normalized_action:
            return "invalid_empty_action"

        zero_progress_streak = int(state.get("zero_progress_streak", 0) or 0)
        last_action = str(state.get("last_action", "") or "").strip()
        position_issue_detected = bool(state.get("position_issue_detected", False))

        if (
            position_issue_detected
            and zero_progress_streak >= 2
            and not normalized_action.startswith("move(")
        ):
            return "position_issue_requires_move"

        if (
            zero_progress_streak >= 2
            and normalized_action == last_action
            and normalized_action.startswith(("use(", "interact("))
        ):
            return "repeat_zero_progress_action"

        return ""

    def _execute_skill(self, action: str, state: dict) -> dict:
        """Execute a skill action using SkillExecuteProvider.

        Mirrors the pattern in langgraph_nodes.skill_execute_node:
        1. Sync skill_steps to LocalMemory.working_area
        2. Call SkillExecuteProvider()
        3. Capture screenshot and augment
        4. Return result
        """
        if self.skill_execute_provider is None:
            logger.warn("[LittleBrain] No skill_execute_provider, dry run")
            return {"success": False, "frame_ids": (0, 0)}

        try:
            from cradle.memory import LocalMemory
            from cradle import constants

            memory = LocalMemory()

            # Sync skill_steps to working_area
            memory.update_info_history({
                "skill_steps": [action],
                "screen_classification": state.get("screen_classification", ""),
                "pre_screen_classification": state.get(
                    "pre_screen_classification", ""
                ),
                "pre_action": state.get("pre_action", ""),
            })

            # Execute via provider
            result = self.skill_execute_provider()

            # Parse result
            exec_info = result.get("exec_info", {})
            executed_skills = exec_info.get("executed_skills", [])
            is_done = exec_info.get("done", False)
            errors_info = str(exec_info.get("errors_info", "") or "")

            # Mark as failed if it was a silent nop (empty action) or if
            # a non-move skill returned no confirmation
            is_empty_nop = "empty_action_nop" in errors_info
            success = (is_done or len(executed_skills) > 0) and not is_empty_nop

            start_frame = result.get("start_frame_id", 0)
            end_frame = result.get("end_frame_id", 0)

            # Capture screenshot after execution
            screenshot_path = ""
            try:
                if self.gm is not None:
                    screenshot_path = self.gm.capture_screen()
                    memory.update_info_history({
                        "screenshot_path": screenshot_path,
                        constants.IMAGES_MEM_BUCKET: screenshot_path,
                    })
                    if self.augment_provider is not None:
                        self.augment_provider()
            except Exception as cap_err:
                logger.warn(
                    f"[LittleBrain] Screenshot capture failed: {cap_err}"
                )

            return {
                "success": success,
                "frame_ids": (start_frame, end_frame),
                "exec_info": exec_info,
                "screenshot_path": screenshot_path,
            }

        except Exception as e:
            logger.error(f"[LittleBrain] Skill execution error: {e}")
            return {"success": False, "frame_ids": (0, 0)}

    def get_status(self) -> dict:
        """Return little brain status for logging."""
        return {
            "suggestions_count": len(self.suggestions),
            "execution_log_count": len(self.execution_log),
            "context_summary_len": len(self.context_summary),
            "current_task": self.current_task,
            "max_relative_move": self.max_relative_move,
        }
