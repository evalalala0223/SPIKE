"""
Big brain (System 2) wrapper for dual-brain architecture (Phase 3.3).

Wraps the existing LangGraph workflow and post-processes its output
into the structured BrainPlanResult format that the little brain consumes.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from cradle.log import Logger
from cradle.runner.langgraph_nodes import sanitize_for_checkpoint

logger = Logger()


@dataclass
class BrainPlanResult:
    """Structured output from big brain planning."""

    suggestions: List[Dict[str, str]]  # [{"action": ..., "reason": ...}]
    context_summary: str  # State summary for little brain (150 chars)
    current_task: str  # Current sub-task description
    completed_steps: List[int] = field(default_factory=list)


class BigBrain:
    """Big brain wraps full LangGraph workflow + post-processing.

    Responsibilities:
        1. Invoke LangGraph workflow (info → reflect → task_inf → action_plan)
        2. Convert planned_actions + planning_reasoning → suggestions[4]
        3. Build context_summary from gathered_info
        4. Handle completed_steps context from previous cycle
        5. Apply dynamic node skipping (skip reflection/task_inference)
    """

    def __init__(
        self,
        workflow_app: Any,
        output_plan_steps: int = 4,
        context_summary_max_chars: int = 150,
    ):
        """
        Args:
            workflow_app: Compiled LangGraph app (from build_game_workflow).
            output_plan_steps: Number of suggestion steps (default 4).
            context_summary_max_chars: Max length of context_summary.
        """
        self.workflow_app = workflow_app
        self.output_plan_steps = output_plan_steps
        self.context_summary_max_chars = context_summary_max_chars

        # Phase 6.2: Failed action ring buffer
        self._recent_failed_actions: List[str] = []
        self._failed_action_buffer_size = 5

        # Phase 6.4: Last failed plan for similarity detection
        self._last_failed_plan: List[str] = []

    # -- Phase 6.2: failed action tracking --

    def record_failed_action(self, action: str):
        normalized = str(action or "").strip()
        if not normalized:
            return
        self._recent_failed_actions.append(normalized)
        if len(self._recent_failed_actions) > self._failed_action_buffer_size:
            self._recent_failed_actions = self._recent_failed_actions[-self._failed_action_buffer_size:]

    def clear_failed_actions(self):
        self._recent_failed_actions.clear()

    def _diversify_action(self, failed_action: str) -> str:
        """将失败的 move 替换为垂直方向小 move，非 move 替换为 nop()"""
        move_match = re.match(
            r"^move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)$",
            failed_action, re.IGNORECASE,
        )
        if move_match:
            x, y = int(move_match.group(1)), int(move_match.group(2))
            # 垂直方向: 交换轴并限制到小值
            perp_x = max(-5, min(5, -y if y != 0 else (2 if x >= 0 else -2)))
            perp_y = max(-5, min(5, x if x != 0 else (2 if y >= 0 else -2)))
            return f"move(x={perp_x}, y={perp_y})"
        return failed_action

    # -- Phase 6.4: plan similarity detection --

    def mark_plan_failed(self, plan_actions: List[str]):
        self._last_failed_plan = list(plan_actions)

    def clear_plan_failure(self):
        self._last_failed_plan.clear()

    def _compute_plan_similarity(self, new_actions: List[str], old_actions: List[str]) -> float:
        if not old_actions or not new_actions:
            return 0.0
        matches = sum(1 for a, b in zip(new_actions, old_actions) if a == b)
        return matches / max(len(new_actions), len(old_actions))

    def plan(
        self,
        state: dict,
        workflow_config: Optional[dict] = None,
    ) -> tuple:
        """Run big brain planning.

        Args:
            state: Current GameState dict.
            workflow_config: LangGraph invoke config (thread_id etc).

        Returns:
            (result_state, plan_result) where result_state is the full
            LangGraph output and plan_result is the structured BrainPlanResult.
        """
        escalation_reason = state.get("escalation_reason", "")
        completed_steps = state.get("completed_steps", [])

        if escalation_reason:
            logger.write(
                f"[BigBrain] Planning triggered by: {escalation_reason}, "
                f"completed_steps={completed_steps}"
            )

        # 1. Invoke full LangGraph workflow
        try:
            result_state = self.workflow_app.invoke(
                sanitize_for_checkpoint(state), config=workflow_config
            )
        except Exception as e:
            logger.error(f"[BigBrain] LangGraph invoke failed: {e}")
            # Return empty plan on failure
            return state, BrainPlanResult(
                suggestions=[],
                context_summary="",
                current_task=state.get("task", ""),
            )

        # 2. Extract suggestions from action planning output
        suggestions = self._extract_suggestions(result_state)

        # 3. Build context summary from gathered info
        context_summary = self._build_context_summary(result_state, state)

        # 4. Get current task
        current_task = result_state.get("subtask_description") or result_state.get("task", state.get("subtask_description", state.get("task", "")))

        # 5. If suggestions are empty, try to salvage an action from the reasoning text
        if not suggestions:
            reasoning_text = result_state.get("planning_reasoning", "")
            salvaged = self._salvage_action_from_reasoning(reasoning_text)
            if salvaged and self._salvaged_action_is_safe(
                salvaged_action=salvaged,
                result_state=result_state,
                prior_state=state,
            ):
                logger.warn(
                    f"[BigBrain] Empty plan salvaged action from reasoning: {salvaged}"
                )
                suggestions = [{"action": salvaged, "reason": "salvaged_from_reasoning"}]

        plan_result = BrainPlanResult(
            suggestions=suggestions,
            context_summary=context_summary,
            current_task=current_task,
            completed_steps=completed_steps,
        )

        logger.write(
            f"[BigBrain] Plan produced: {len(suggestions)} suggestions, "
            f"task={current_task[:50]}, "
            f"context_summary={len(context_summary)} chars"
        )

        return result_state, plan_result

    def _extract_suggestions(self, result_state: dict) -> List[Dict[str, str]]:
        """Convert planned_actions + planning_reasoning into suggestions.

        Each suggestion has {"action": ..., "reason": ...}.
        Tries to parse per-action reasons from planning_reasoning.
        Falls back to generic reasons if parsing fails.
        """
        planned_actions = result_state.get("planned_actions", [])
        reasoning = result_state.get("planning_reasoning", "")

        if not planned_actions:
            return []

        cleaned_actions: List[str] = []
        for action in planned_actions:
            action_text = str(action).strip()
            if action_text:
                cleaned_actions.append(action_text)

        if not cleaned_actions:
            return []

        # Limit to output_plan_steps
        actions = cleaned_actions[: self.output_plan_steps]

        # Deduplicate consecutive identical actions:
        # - Non-move actions: collapse consecutive duplicates to 1
        # - Move actions: allow up to 2 consecutive identical moves (long distance),
        #   but 3+ identical moves is likely LLM degradation
        deduped = []
        for action in actions:
            if deduped and action == deduped[-1]:
                if action.startswith("move("):
                    # Count consecutive identical moves already in deduped
                    consecutive_count = 0
                    for prev in reversed(deduped):
                        if prev == action:
                            consecutive_count += 1
                        else:
                            break
                    if consecutive_count >= 2:
                        continue  # Skip 3rd+ identical move
                else:
                    continue  # Skip consecutive duplicate non-move action
            deduped.append(action)

        if len(deduped) < len(actions):
            logger.write(
                f"[BigBrain] Deduped plan: {len(actions)} -> {len(deduped)} actions"
            )
        actions = deduped

        # Phase 6.2: Intercept first action if it matches a recently failed action
        if actions and self._recent_failed_actions:
            first = actions[0]
            if first in self._recent_failed_actions:
                replacement = self._diversify_action(first)
                if replacement != first:
                    logger.warn(f"[BigBrain] Step 1 '{first}' matches recently failed action. Replacing with '{replacement}'")
                    actions[0] = replacement
                else:
                    logger.warn(
                        f"[BigBrain] Step 1 '{first}' matches recently failed action, "
                        "but no safe diversification is available; keeping the grounded action."
                    )

        # Phase 6.4: Plan similarity detection - mutate if new plan too similar to failed plan
        if self._last_failed_plan and actions:
            similarity = self._compute_plan_similarity(actions, self._last_failed_plan)
            # During navigation, repeated move-only plans in the same direction
            # are expected and correct.  Use a higher threshold so that only
            # nearly-identical plans are mutated.
            all_moves = all(a.strip().startswith("move(") for a in actions)
            threshold = 0.85 if all_moves else 0.5
            if similarity > threshold:
                logger.warn(
                    f"[BigBrain] New plan {similarity:.0%} similar to failed plan "
                    f"(threshold={threshold}, all_moves={all_moves}). Mutating step 1."
                )
                # 只替换第一个相同动作，保持计划剩余部分连贯
                for i in range(min(len(actions), len(self._last_failed_plan))):
                    if actions[i] == self._last_failed_plan[i]:
                        replacement = self._diversify_action(actions[i])
                        if replacement != actions[i]:
                            actions[i] = replacement
                        break  # 只改一步
            elif similarity > 0.5 and all_moves:
                logger.write(
                    f"[BigBrain] Plan {similarity:.0%} similar to failed plan but "
                    f"all-move navigation — keeping original plan."
                )

        # Try to extract per-action reasons from reasoning text
        per_action_reasons = self._parse_reasoning(reasoning, len(actions))

        suggestions = []
        for i, action in enumerate(actions):
            reason = (
                per_action_reasons[i]
                if i < len(per_action_reasons) and per_action_reasons[i]
                else self._generate_default_reason(action)
            )
            suggestions.append({"action": action, "reason": reason})

        return suggestions

    def _parse_reasoning(
        self, reasoning: str, expected_count: int
    ) -> List[str]:
        """Try to extract per-action reasons from reasoning text.

        Looks for numbered patterns like:
            1. reason text
            2. reason text
        Or:
            Step 1: reason text
            Step 2: reason text
        """
        if not reasoning:
            return []

        reasons: List[str] = []

        # Pattern 1: "1. reason" or "1) reason"
        numbered = re.findall(
            r"(?:^|\n)\s*\d+[\.\)]\s*(.+?)(?=\n\s*\d+[\.\)]|\n\n|$)",
            reasoning,
            re.DOTALL,
        )
        if numbered and len(numbered) >= expected_count:
            for r in numbered[:expected_count]:
                reasons.append(r.strip()[:60])
            return reasons

        # Pattern 2: "Step N:" or "step N:"
        step_pattern = re.findall(
            r"[Ss]tep\s*\d+[:\s]+(.+?)(?=[Ss]tep\s*\d+|$)",
            reasoning,
            re.DOTALL,
        )
        if step_pattern and len(step_pattern) >= expected_count:
            for r in step_pattern[:expected_count]:
                reasons.append(r.strip()[:60])
            return reasons

        # Pattern 3: split by sentence and assign to each action
        sentences = re.split(r"[。.;；]", reasoning)
        sentences = [s.strip() for s in sentences if s.strip()]
        if sentences:
            for i in range(expected_count):
                if i < len(sentences):
                    reasons.append(sentences[i][:60])
                else:
                    reasons.append("")
            return reasons

        return []

    def _generate_default_reason(self, action: str) -> str:
        """Generate a default reason from the action name."""
        # Extract skill name from action string
        match = re.match(r"(\w+)\s*\(", action)
        if match:
            skill_name = match.group(1)
            return f"execute {skill_name}"
        return "execute action"

    def _build_context_summary(self, result_state: dict, prior_state: Optional[dict] = None) -> str:
        """Build context_summary from gathered_info and other state.

        Targets ~150 chars: position, orientation, goal, key items, surroundings.
        """
        gathered = result_state.get("gathered_info", {})
        task = result_state.get("task", "")
        source_state = prior_state if isinstance(prior_state, dict) else result_state

        parts = []

        # Extract description from gathered_info
        if isinstance(gathered, dict):
            desc = gathered.get("description", "")
            if isinstance(desc, str) and desc:
                # Truncate description to fit
                parts.append(desc[:100])
        elif isinstance(gathered, str) and gathered:
            parts.append(gathered[:100])

        # Add task
        if task:
            parts.append(f"task: {task[:40]}")

        latest_execution_summary = str(source_state.get("latest_execution_summary", "") or "").strip()
        if latest_execution_summary:
            parts.append(latest_execution_summary[:90])

        zero_progress_streak = int(source_state.get("zero_progress_streak", 0) or 0)
        if zero_progress_streak >= 2:
            parts.append(f"zero-progress streak: {zero_progress_streak}")

        summary = " | ".join(parts)

        # Truncate to max length
        if len(summary) > self.context_summary_max_chars:
            summary = summary[: self.context_summary_max_chars - 3] + "..."

        return summary

    def get_plan_as_state_update(
        self, plan_result: BrainPlanResult
    ) -> dict:
        """Convert BrainPlanResult into a state dict update.

        Used by DualBrainController to merge plan into GameState.
        """
        return {
            "suggestions": plan_result.suggestions,
            "planned_actions": [
                suggestion.get("action", "")
                for suggestion in plan_result.suggestions
                if isinstance(suggestion, dict)
            ],
            "subtask_description": plan_result.current_task,
            "context_summary": plan_result.context_summary,
            "current_step": 0,  # LittleBrain will decide the immediate action from suggestion 0
            "execution_log": [],  # Clear for new cycle
            "brain_mode": "little",  # Next step will be little brain
            "escalation_reason": "",
            # Clear stale execution feedback so that the next failure evaluation
            # in DualBrainController.step() does not re-process the old feedback
            # that was already evaluated before this BigBrain call.
            "has_execution_feedback": False,
            "execution_pending": False,
            "pending_action": "",
            "pending_step_index": None,
            "pending_suggested_action": "",
            "force_big_brain_replan": False,
            "completed_steps": [],
        }

    @staticmethod
    def _salvage_action_from_reasoning(reasoning_text: str) -> str:
        """Try to extract an action from reasoning text when the normal action parser fails.

        Sometimes the LLM embeds valid actions inside its reasoning
        (e.g., "we should move(x=3, y=2) to reach the barn") but the
        structured parser fails. This salvage is intentionally strict:
        only extract from an explicit Actions/code-block region, never
        from free-form chain-of-thought text.
        """
        if not reasoning_text:
            return ""

        action_pattern = re.compile(
            r"\b(move|use|interact|choose_item|choose_option|menu|craft|attach_item|unattach_item|nop)"
            r"\s*\([^)]*\)",
            re.IGNORECASE,
        )

        candidate_regions: List[str] = []
        for code_block in re.finditer(
            r"```(?:python)?\s*(.*?)```",
            reasoning_text,
            re.IGNORECASE | re.DOTALL,
        ):
            region = str(code_block.group(1) or "").strip()
            if region:
                candidate_regions.append(region)

        actions_idx = reasoning_text.lower().find("actions:")
        if actions_idx >= 0:
            tail_region = reasoning_text[actions_idx:]
            if tail_region.strip():
                candidate_regions.append(tail_region)

        for region in candidate_regions:
            full_matches = list(action_pattern.finditer(region))
            if full_matches:
                return full_matches[0].group(0).strip()

        return ""

    @staticmethod
    def _salvaged_action_is_safe(
        salvaged_action: str,
        result_state: dict,
        prior_state: Optional[dict] = None,
    ) -> bool:
        action_text = str(salvaged_action or "").strip().lower()
        if not action_text:
            return False

        source_state = prior_state if isinstance(prior_state, dict) else result_state
        task_text = " ".join(
            str(source_state.get(key, "") or "")
            for key in ("task", "main_task", "task_description", "subtask_description")
        ).lower()

        cultivation_markers = (
            " till ",
            " sow ",
            " fertilize ",
            " water ",
            " harvest ",
            " cultivate ",
        )
        normalized_task = f" {re.sub(r'[^a-z0-9]+', ' ', task_text)} "
        is_cultivation = any(token in normalized_task for token in cultivation_markers)

        # Empty-plan salvage is too unreliable for cultivation tool-use/item-use
        # actions because directional reasoning often contradicts the structured
        # surroundings. Keep salvage only for safer navigation/setup actions.
        if is_cultivation and action_text.startswith(("use(", "interact(")):
            return False

        return True
