from __future__ import annotations

import unittest

from cradle.runner.failure_detector import FailureDetector


class TestFailureDetector(unittest.TestCase):
    def test_explicit_blocked_feedback_counts_as_exec_error(self) -> None:
        detector = FailureDetector(max_consecutive_failures=4)

        result = detector.evaluate(
            exec_info={
                "errors": False,
                "errors_info": (
                    'move(x=0, y=3) toward down FAILED - '
                    "player position did not change, path is likely blocked by an obstacle"
                ),
                "executed_skills": ['move(x=0, y=3)'],
                "done": True,
            },
            action='move(x=0, y=3)',
            elapsed_ms=2000,
            brain_mode="little",
            previous_actions=['move(x=0, y=3)'],
            state_changed=False,
            previous_progress=0,
            current_progress=0,
            consecutive_zero_progress=0,
            repeated_action_streak=1,
            position_issue_detected=False,
            oscillation_streak=0,
        )

        self.assertEqual(result.level, "F2")
        self.assertTrue(any("exec_error" in reason for reason in result.reasons))

    def test_no_confirmation_alone_remains_soft_signal(self) -> None:
        detector = FailureDetector(max_consecutive_failures=4)

        result = detector.evaluate(
            exec_info={
                "errors": False,
                "errors_info": "choose_item() returned no confirmation; action may not have taken effect.",
                "executed_skills": ["choose_item(slot_index=5)"],
                "done": True,
            },
            action="choose_item(slot_index=5)",
            elapsed_ms=2000,
            brain_mode="little",
            previous_actions=["choose_item(slot_index=5)"],
            state_changed=False,
            previous_progress=0,
            current_progress=0,
            consecutive_zero_progress=1,
            repeated_action_streak=1,
            position_issue_detected=False,
            oscillation_streak=0,
        )

        self.assertEqual(result.level, "F1")
        self.assertFalse(any("exec_error" in reason for reason in result.reasons))

    def test_repeated_productive_action_does_not_trigger_repetitive_action(self) -> None:
        detector = FailureDetector(max_consecutive_failures=4)

        result = detector.evaluate(
            exec_info={
                "errors": False,
                "errors_info": "",
                "executed_skills": ['use(direction="down")'],
                "done": True,
            },
            action='use(direction="down")',
            elapsed_ms=2000,
            brain_mode="little",
            previous_actions=['use(direction="down")'] * 4,
            state_changed=True,
            previous_progress=3,
            current_progress=4,
            consecutive_zero_progress=0,
            repeated_action_streak=4,
            position_issue_detected=False,
            oscillation_streak=0,
        )

        self.assertEqual(result.level, "F0")
        self.assertFalse(result.should_escalate)
        self.assertFalse(any("repetitive_action" in reason for reason in result.reasons))

    def test_repeated_unproductive_action_triggers_escalation(self) -> None:
        detector = FailureDetector(max_consecutive_failures=4)

        result = detector.evaluate(
            exec_info={
                "errors": False,
                "errors_info": "",
                "executed_skills": ['use(direction="down")'],
                "done": True,
            },
            action='use(direction="down")',
            elapsed_ms=2000,
            brain_mode="little",
            previous_actions=['use(direction="down")'] * 5,
            state_changed=False,
            previous_progress=3,
            current_progress=3,
            consecutive_zero_progress=2,
            repeated_action_streak=5,
            position_issue_detected=False,
            oscillation_streak=0,
        )

        self.assertIn(result.level, {"F1", "F2"})
        self.assertTrue(result.should_escalate)
        self.assertTrue(any("repetitive_action" in reason for reason in result.reasons))

    def test_zero_progress_alone_still_triggers_escalation(self) -> None:
        detector = FailureDetector(max_consecutive_failures=4)

        result = detector.evaluate(
            exec_info={
                "errors": False,
                "errors_info": "",
                "executed_skills": ['interact(direction="down")'],
                "done": True,
            },
            action='interact(direction="down")',
            elapsed_ms=2000,
            brain_mode="little",
            previous_actions=['interact(direction="down")'] * 2,
            state_changed=False,
            previous_progress=0,
            current_progress=0,
            consecutive_zero_progress=5,
            repeated_action_streak=2,
            position_issue_detected=False,
            oscillation_streak=0,
        )

        self.assertEqual(result.level, "F2")
        self.assertTrue(result.should_escalate)


if __name__ == "__main__":
    unittest.main()
