"""Verify the strengthened _build_task_progress_summary directives.

Re-implements a minimal copy of the function logic to test
without importing the full stardojo dependency tree.
"""
import unittest
from typing import List, Optional


def _build_task_progress_summary(
    action_text: str,
    previous_progress: Optional[float],
    current_progress: Optional[float],
    progress_delta: Optional[float],
    zero_progress_streak: int,
    repeated_action_streak: int,
    productive_action: bool,
    state_changed: bool,
    position_issue_detected: bool,
    completed: Optional[bool],
    errors_info: str = "",
    oscillation_streak: int = 0,
    missing_confirmation: bool = False,
) -> str:
    """Minimal reproduction of the production function for testing.

    When the production code changes, sync the logic here and re-run.
    """
    parts: List[str] = []

    if action_text:
        parts.append(f"Last action: {action_text}.")

    if current_progress is not None:
        if previous_progress is not None and progress_delta is not None:
            if progress_delta > 0:
                parts.append(
                    f"Task progress increased from {previous_progress:g} to {current_progress:g}."
                )
            elif progress_delta < 0:
                parts.append(
                    f"Task progress decreased from {previous_progress:g} to {current_progress:g}."
                )
            else:
                parts.append(f"Task progress stayed at {current_progress:g}.")
        else:
            parts.append(f"Recorded task progress is {current_progress:g}.")

    if productive_action and not state_changed:
        parts.append("The productive action had no observable effect.")
    elif state_changed:
        parts.append("The action changed the local state.")

    if zero_progress_streak >= 2:
        parts.append(
            f"There have been {zero_progress_streak} consecutive productive actions without progress."
        )

    if repeated_action_streak >= 3 and action_text:
        parts.append(
            f"FORBIDDEN: The action `{action_text}` has failed {repeated_action_streak} "
            f"consecutive times. Your next action MUST NOT be `{action_text}`. Choose a "
            f"different action — typically move() to reposition or a different direction."
        )
    elif repeated_action_streak >= 2:
        parts.append(
            f"The same action has been repeated {repeated_action_streak} times. "
            f"If it failed again, do not repeat it a third time — change action or position."
        )

    if position_issue_detected:
        parts.append(
            "Current task reasoning indicates the player must move closer or become adjacent before using the tool again."
        )

    if missing_confirmation:
        if state_changed:
            parts.append(
                "The action returned no explicit confirmation, but the observed state suggests it may have taken effect."
            )
        else:
            parts.append(
                "The action returned no explicit confirmation and there was no observed state change."
            )

    blocked_feedback = errors_info.lower() if isinstance(errors_info, str) else ""
    if "blocked by an obstacle" in blocked_feedback or "path is likely blocked" in blocked_feedback:
        parts.append(
            "The last move appears to be blocked by an obstacle. If the tile in front of the player or the effect point contains stone, twig, wood, weeds, grass, or fiber, clear it first with the matching tool before moving again. "
            "If no clearable obstacle is visible, sidestep 1-2 tiles on the perpendicular axis (if you were moving on x, try a small y move; if you were moving on y, try a small x move) before retrying. Drop magnitude to 1 tile."
        )

    if oscillation_streak >= 2:
        parts.append(
            f"FORBIDDEN: The agent has been oscillating between opposite moves for "
            f"{oscillation_streak} consecutive pairs. Your next action MUST NOT be a move that "
            f"reverses the previous direction. Commit to one direction for at least 3-5 tiles, "
            f"or switch to a perpendicular axis. If your last move was on x-axis, the next move "
            f"must be on y-axis (or vice versa)."
        )

    if errors_info:
        parts.append(f"Execution feedback: {errors_info}")

    if completed is True:
        parts.append("The task is completed.")
    elif completed is False and current_progress is not None:
        parts.append("The task is not completed yet.")

    return " ".join(parts).strip()


class TestProgressSummaryDirectives(unittest.TestCase):

    def _call(self, **kwargs):
        defaults = dict(
            action_text="",
            previous_progress=None,
            current_progress=None,
            progress_delta=None,
            zero_progress_streak=0,
            repeated_action_streak=0,
            productive_action=False,
            state_changed=False,
            position_issue_detected=False,
            completed=None,
            errors_info="",
            oscillation_streak=0,
            missing_confirmation=False,
        )
        defaults.update(kwargs)
        return _build_task_progress_summary(**defaults)

    # -- repeated-action directive --
    def test_repeated_3_says_forbidden_and_names_action(self):
        text = self._call(
            action_text='interact(direction="right")',
            repeated_action_streak=3,
        )
        self.assertIn("FORBIDDEN", text)
        self.assertIn('interact(direction="right")', text)

    def test_repeated_5_says_forbidden(self):
        text = self._call(
            action_text='use(direction="down")',
            repeated_action_streak=5,
        )
        self.assertIn("FORBIDDEN", text)
        self.assertIn('use(direction="down")', text)
        self.assertIn("5", text)

    def test_repeated_2_warns_but_no_forbidden(self):
        text = self._call(
            action_text='use(direction="down")',
            repeated_action_streak=2,
        )
        self.assertIn("repeated 2 times", text)
        self.assertNotIn("FORBIDDEN", text)

    def test_repeated_1_no_warning(self):
        text = self._call(
            action_text='move(x=1, y=0)',
            repeated_action_streak=1,
        )
        self.assertNotIn("repeated", text.lower())
        self.assertNotIn("FORBIDDEN", text)

    # -- oscillation directive --
    def test_oscillation_2_says_forbidden(self):
        text = self._call(oscillation_streak=2)
        self.assertIn("FORBIDDEN", text)
        self.assertIn("perpendicular", text)

    def test_oscillation_0_no_warning(self):
        text = self._call(oscillation_streak=0)
        self.assertNotIn("oscillat", text.lower())
        self.assertNotIn("FORBIDDEN", text)

    # -- blocked move sidestep hint --
    def test_blocked_move_contains_sidestep(self):
        text = self._call(
            errors_info="move(x=5, y=0) toward right FAILED - path is likely blocked by an obstacle"
        )
        self.assertIn("sidestep", text)
        self.assertIn("perpendicular", text)

    def test_no_blocked_no_sidestep(self):
        text = self._call(errors_info="")
        self.assertNotIn("sidestep", text)

    # -- combined --
    def test_combined_repeated_and_oscillation(self):
        text = self._call(
            action_text='move(x=-1, y=0)',
            repeated_action_streak=4,
            oscillation_streak=3,
        )
        self.assertEqual(text.count("FORBIDDEN"), 2)


if __name__ == "__main__":
    unittest.main()
