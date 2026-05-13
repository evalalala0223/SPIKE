"""
Brain scheduler for dual-brain architecture (Phase 3.2).

Decides whether to route to big brain (full LangGraph workflow) or
little brain (lightweight fast-LLM path) on each step.
"""
from typing import Literal

from cradle.log import Logger

logger = Logger()

BrainMode = Literal["big", "little"]


class BrainScheduler:
    """Periodic scheduler: 1 big-brain call + N little-brain calls per cycle.

    Trigger big brain when ANY of:
        1. First step (is_first_step)
        2. Little brain steps >= cycle_size
        3. Consecutive failures >= max_consecutive_failures
        4. Environment change detected (env_changed)
        5. Fast LLM not available

    Continue little brain when ALL of:
        1. Has pending suggestions
        2. Environment stable
        3. Fast LLM available
        4. No excessive consecutive failures
    """

    def __init__(
        self,
        cycle_size: int = 4,
        max_consecutive_failures: int = 2,
        stall_zero_progress_threshold: int = 4,
        stall_repeat_threshold: int = 5,
        stall_repeat_zero_progress_threshold: int = 2,
    ):
        self.cycle_size = int(cycle_size)
        self.max_consecutive_failures = max_consecutive_failures
        self.stall_zero_progress_threshold = max(1, int(stall_zero_progress_threshold))
        self.stall_repeat_threshold = max(1, int(stall_repeat_threshold))
        self.stall_repeat_zero_progress_threshold = max(
            1,
            int(stall_repeat_zero_progress_threshold),
        )
        self.little_brain_steps = 0

    def decide(self, state: dict) -> BrainMode:
        """Decide which brain to use for the current step.

        Args:
            state: Current GameState dict.

        Returns:
            "big" or "little".
        """
        # Condition 1: First step always uses big brain
        if state.get("is_first_step", False):
            logger.write("[Scheduler] -> big brain (first step)")
            return "big"

        # Condition 1.5: External execution feedback requires a fresh big-brain plan
        if state.get("force_big_brain_replan", False):
            logger.write("[Scheduler] -> big brain (forced replan after external feedback)")
            return "big"

        # Condition 2: Cycle completed. Non-positive cycle_size disables
        # periodic refresh; adaptive triggers still route to big brain.
        if self.cycle_size > 0 and self.little_brain_steps >= self.cycle_size:
            logger.write(
                f"[Scheduler] -> big brain (cycle complete: "
                f"{self.little_brain_steps}/{self.cycle_size})"
            )
            return "big"

        # Condition 3: Fast LLM not available
        if not state.get("vllm_available", False):
            logger.write("[Scheduler] -> big brain (fast LLM unavailable)")
            return "big"

        # Condition 4: Environment change detected
        if state.get("env_changed", False):
            logger.write(
                f"[Scheduler] -> big brain (env change: "
                f"score={state.get('env_change_score', 0):.3f})"
            )
            return "big"

        # Condition 4.5: Stalled execution feedback.
        zero_progress = int(state.get("zero_progress_streak", 0) or 0)
        repeated_actions = int(state.get("repeated_action_streak", 0) or 0)
        if zero_progress >= self.stall_zero_progress_threshold:
            logger.write(
                "[Scheduler] -> big brain "
                f"(stall: zero_progress={zero_progress}/"
                f"{self.stall_zero_progress_threshold})"
            )
            return "big"
        if (
            repeated_actions >= self.stall_repeat_threshold
            and zero_progress >= self.stall_repeat_zero_progress_threshold
        ):
            logger.write(
                "[Scheduler] -> big brain "
                f"(stall: repeat={repeated_actions}/{self.stall_repeat_threshold}, "
                f"zero_progress={zero_progress}/"
                f"{self.stall_repeat_zero_progress_threshold})"
            )
            return "big"

        # Condition 5: Consecutive failures
        consecutive_failures = state.get("consecutive_failures", 0)
        if consecutive_failures >= self.max_consecutive_failures:
            logger.write(
                f"[Scheduler] -> big brain (consecutive failures: "
                f"{consecutive_failures}/{self.max_consecutive_failures})"
            )
            return "big"

        # Condition 6: No pending suggestions
        suggestions = state.get("suggestions", [])
        current_step = state.get("current_step", 0)
        if not suggestions or current_step >= len(suggestions):
            logger.write("[Scheduler] -> big brain (no pending suggestions)")
            return "big"

        # All checks passed: use little brain
        logger.write(
            f"[Scheduler] -> little brain "
            f"(step {current_step + 1}/{len(suggestions)}, "
            f"cycle {self.little_brain_steps + 1}/"
            f"{self.cycle_size if self.cycle_size > 0 else 'inf'})"
        )
        return "little"

    def reset_counter(self):
        """Reset the little brain step counter (after big brain runs)."""
        self.little_brain_steps = 0

    def increment_counter(self):
        """Increment the little brain step counter."""
        self.little_brain_steps += 1

    def get_status(self) -> dict:
        """Return scheduler status for logging/debugging."""
        return {
            "little_brain_steps": self.little_brain_steps,
            "cycle_size": self.cycle_size,
            "max_consecutive_failures": self.max_consecutive_failures,
            "stall_zero_progress_threshold": self.stall_zero_progress_threshold,
            "stall_repeat_threshold": self.stall_repeat_threshold,
            "stall_repeat_zero_progress_threshold": (
                self.stall_repeat_zero_progress_threshold
            ),
        }
