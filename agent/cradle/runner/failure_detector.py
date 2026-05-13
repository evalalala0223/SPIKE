"""
Failure detector for dual-brain architecture (Phase 3.3).

Evaluates execution results and determines failure levels (F0-F3).
Computes a fail_score and provides escalation recommendations.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from cradle.log import Logger
from stardojo.utils.execution_feedback_utils import execution_has_explicit_failure

logger = Logger()


@dataclass
class FailureResult:
    """Result of failure detection for a single step."""

    level: str  # F0 (pass), F1 (soft), F2 (hard), F3 (fatal)
    score: float  # Weighted fail score (0.0 = clean, 1.0+ = hard fail)
    reasons: List[str]  # List of triggered failure reasons
    should_escalate: bool  # Whether to escalate to big brain
    decision_trace: str  # Human-readable trace of the decision


# Failure signal weights
_SIGNAL_WEIGHTS = {
    "exec_error": 1.0,
    "invalid_action": 0.7,
    "timeout": 0.5,
    "no_progress": 0.4,
    "task_drift": 0.3,
    "repetitive_action": 0.45,
    "position_mismatch": 0.6,
    "oscillation": 0.5,
}


class FailureDetector:
    """Detect and classify execution failures.

    Failure levels:
        F0 (pass):  Action succeeded, state progressed toward goal
        F1 (soft):  Action succeeded but no progress / low progress
        F2 (hard):  Execution error, invalid action, timeout
        F3 (fatal): Consecutive hard failures beyond limit

    Escalation rules:
        - Little → Big: 3 consecutive failures (F1/F2) or single fail_score >= 1.5
        - Big stays:    Big brain F2 allows 1 replan; second F2 → F3
        - Big → Little: Big brain success once and no task_drift
    """

    def __init__(
        self,
        little_brain_timeout_ms: int = 5000,
        big_brain_timeout_s: int = 75,
        max_consecutive_failures: int = 4,
    ):
        self.little_brain_timeout_ms = little_brain_timeout_ms
        self.big_brain_timeout_s = big_brain_timeout_s
        self.max_consecutive_failures = max_consecutive_failures
        self._consecutive_failures = 0
        self._consecutive_compound_failures = 0  # Phase 8.2: F1 with multiple signals
        self._big_brain_replan_count = 0

    def evaluate(
        self,
        exec_info: Dict[str, Any],
        action: str,
        elapsed_ms: float,
        brain_mode: str,
        previous_actions: Optional[List[str]] = None,
        state_changed: bool = True,
        previous_progress: Optional[float] = None,
        current_progress: Optional[float] = None,
        consecutive_zero_progress: int = 0,
        repeated_action_streak: int = 0,
        position_issue_detected: bool = False,
        oscillation_streak: int = 0,
    ) -> FailureResult:
        """Evaluate an execution result and determine failure level.

        Args:
            exec_info: Result from gm.execute_actions() or SkillExecuteProvider.
            action: The action string that was executed.
            elapsed_ms: Elapsed time in milliseconds.
            brain_mode: "big" or "little".
            previous_actions: Recent action history (for repeat detection).
            state_changed: Whether the game state progressed.

        Returns:
            FailureResult with level, score, and escalation recommendation.
        """
        reasons: List[str] = []
        signals: Dict[str, bool] = {
            "exec_error": False,
            "invalid_action": False,
            "timeout": False,
            "no_progress": False,
            "task_drift": False,
            "repetitive_action": False,
            "position_mismatch": False,
            "oscillation": False,
        }

        # Signal 1: Execution error
        if execution_has_explicit_failure(exec_info):
            signals["exec_error"] = True
            error_info = exec_info.get("errors_info", "unknown")
            reasons.append(f"exec_error: {error_info}")

        # Signal 2: Invalid action (no skills executed)
        executed_skills = exec_info.get("executed_skills", [])
        if not executed_skills and not exec_info.get("done", False):
            signals["invalid_action"] = True
            reasons.append(f"invalid_action: {action}")

        # Signal 3: Timeout
        timeout_ms = (
            self.little_brain_timeout_ms
            if brain_mode == "little"
            else self.big_brain_timeout_s * 1000
        )
        if elapsed_ms > timeout_ms:
            signals["timeout"] = True
            reasons.append(f"timeout: {elapsed_ms:.0f}ms > {timeout_ms}ms")

        # Signal 4: No progress
        progress_unchanged = (
            current_progress is not None
            and previous_progress is not None
            and current_progress == previous_progress
        )
        if (not state_changed or progress_unchanged or consecutive_zero_progress > 0) and not signals["exec_error"]:
            signals["no_progress"] = True
            if consecutive_zero_progress > 0:
                reasons.append(
                    f"no_progress: productive action made no progress "
                    f"(streak={consecutive_zero_progress})"
                )
            elif progress_unchanged:
                reasons.append(
                    f"no_progress: task progress stuck at {current_progress}"
                )
            else:
                reasons.append("no_progress: state unchanged")

        # Signal 5: Task drift (repeated same action without progress)
        if previous_actions and not state_changed:
            if len(previous_actions) >= 2 and previous_actions[-1] == action:
                signals["task_drift"] = True
                reasons.append("task_drift: repeated action without progress")

        # Signal 6: Repeated same action without progress
        repeated_without_progress = bool(
            repeated_action_streak >= 4
            and (
                consecutive_zero_progress >= 2
                or not state_changed
            )
        )
        if repeated_without_progress:
            signals["repetitive_action"] = True
            reasons.append(
                "repetitive_action: same action repeated without progress "
                f"({repeated_action_streak} repeats, zero_progress={consecutive_zero_progress})"
            )

        # Signal: Oscillation between opposite moves
        if oscillation_streak >= 2:
            signals["oscillation"] = True
            reasons.append(f"oscillation: opposite-move oscillation for {oscillation_streak} pairs")

        # Signal 7: Task inference or feedback says current position is wrong
        if position_issue_detected:
            signals["position_mismatch"] = True
            reasons.append(
                "position_mismatch: latest reasoning says move closer / become adjacent first"
            )

        # Compute fail_score
        score = sum(
            weight
            for signal, weight in _SIGNAL_WEIGHTS.items()
            if signals.get(signal, False)
        )

        # Determine failure level
        level = self._classify_level(
            signals,
            score,
            state_changed,
            consecutive_zero_progress=consecutive_zero_progress,
            repeated_action_streak=repeated_action_streak,
        )

        # Update consecutive failure counter
        # Phase 8.2: F1 with only no_progress (score <= 0.4, single signal) is
        # common during normal navigation and should not count toward escalation.
        # Only compound F1 (multiple signals) or F2+ count.
        if level == "F2":
            self._consecutive_failures += 1
            self._consecutive_compound_failures += 1
        elif level == "F1":
            active_count = sum(1 for v in signals.values() if v)
            if active_count >= 2 or score > 0.5:
                # Compound F1: multiple signals firing together
                self._consecutive_failures += 1
                self._consecutive_compound_failures += 1
            else:
                # Simple F1 (e.g. just no_progress during move) — track but
                # don't count toward compound escalation threshold
                self._consecutive_failures += 1
        elif level == "F0":
            self._consecutive_failures = 0
            self._consecutive_compound_failures = 0

        # Determine escalation
        should_escalate = self._should_escalate(
            level,
            score,
            brain_mode,
            consecutive_zero_progress=consecutive_zero_progress,
            repeated_action_streak=repeated_action_streak,
            position_issue_detected=position_issue_detected,
        )

        # Build decision trace
        active_signals = [s for s, v in signals.items() if v]
        trace = (
            f"level={level} score={score:.2f} "
            f"signals={active_signals} "
            f"consecutive={self._consecutive_failures} "
            f"compound={self._consecutive_compound_failures} "
            f"zero_progress={consecutive_zero_progress} "
            f"repeat={repeated_action_streak} "
            f"escalate={should_escalate}"
        )

        result = FailureResult(
            level=level,
            score=score,
            reasons=reasons,
            should_escalate=should_escalate,
            decision_trace=trace,
        )

        # Log
        if level != "F0":
            logger.warn(f"[FailureDetector] {trace}")
        else:
            logger.write(f"[FailureDetector] {trace}")

        return result

    def _classify_level(
        self,
        signals: Dict[str, bool],
        score: float,
        state_changed: bool = False,
        consecutive_zero_progress: int = 0,
        repeated_action_streak: int = 0,
    ) -> str:
        """Classify failure into F0-F3 levels.

        For timeout-only (no exec_error/invalid_action), downgrade based on
        state_changed because elapsed_ms includes game I/O (skill execution
        ~6-9s + screenshot), not just LLM latency.
        """
        if signals["exec_error"] or signals["invalid_action"]:
            if self._consecutive_failures >= self.max_consecutive_failures:
                return "F3"
            return "F2"

        if signals["timeout"]:
            # Timeout without hard errors: downgrade based on outcome
            if state_changed:
                return "F0"  # Slow but successful, not a failure
            return "F1"  # Slow and no progress, soft warning

        if consecutive_zero_progress >= 4:
            return "F2"

        if signals["no_progress"] or signals["task_drift"]:
            return "F1"

        if signals.get("oscillation", False):
            return "F1"

        if signals["repetitive_action"] or signals["position_mismatch"]:
            return "F1"

        return "F0"

    def _should_escalate(
        self,
        level: str,
        score: float,
        brain_mode: str,
        consecutive_zero_progress: int = 0,
        repeated_action_streak: int = 0,
        position_issue_detected: bool = False,
    ) -> bool:
        """Determine whether to escalate to big brain.

        Little brain escalation:
            - 3+ consecutive F1/F2 failures
            - Single step with fail_score >= 1.5
            - F3 (fatal) always escalates

        Big brain escalation:
            - F2 allows 1 replan; second F2 escalates (F3 territory)
            - F3 always escalates
        """
        if brain_mode == "little":
            if level == "F3":
                return True
            if consecutive_zero_progress >= 4:
                return True
            if repeated_action_streak >= 5 and consecutive_zero_progress >= 2:
                return True
            if position_issue_detected and consecutive_zero_progress >= 3:
                return True
            # Phase 8.2: Use compound failures (multi-signal F1 or F2) for
            # escalation threshold, not simple no_progress-only F1.
            if self._consecutive_compound_failures >= self.max_consecutive_failures:
                return True
            # Fallback: even simple F1s should escalate eventually to avoid
            # infinite loops where no compound failures ever accumulate.
            if self._consecutive_failures >= self.max_consecutive_failures * 2:
                return True
            if score >= 1.5:
                return True
            return False

        # Big brain mode
        if level == "F3":
            return True
        if level == "F2":
            self._big_brain_replan_count += 1
            if self._big_brain_replan_count > 1:
                return True  # F3 territory
            return False  # Allow 1 replan

        return False

    def reset(self):
        """Reset counters (e.g. after successful big brain cycle)."""
        self._consecutive_failures = 0
        self._consecutive_compound_failures = 0
        self._big_brain_replan_count = 0

    def on_big_brain_success(self):
        """Called when big brain succeeds. Resets big brain replan counter."""
        self._big_brain_replan_count = 0
        self._consecutive_failures = 0
        self._consecutive_compound_failures = 0

    def get_status(self) -> dict:
        """Return detector status for logging."""
        return {
            "consecutive_failures": self._consecutive_failures,
            "consecutive_compound_failures": self._consecutive_compound_failures,
            "big_brain_replan_count": self._big_brain_replan_count,
        }
