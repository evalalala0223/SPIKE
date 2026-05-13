from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from agent.stardojo.stardojo_react_agent import PipelineRunner


class TestExecutionFeedbackSummary(unittest.TestCase):
    def test_progress_increase_is_treated_as_observable_effect(self) -> None:
        summary = PipelineRunner._build_task_progress_summary(
            action_text='interact(direction="down")',
            previous_progress=1,
            current_progress=2,
            progress_delta=1,
            zero_progress_streak=0,
            repeated_action_streak=0,
            productive_action=True,
            state_changed=False,
            position_issue_detected=False,
            completed=False,
            errors_info="interact() returned no confirmation; action may not have taken effect.",
            oscillation_streak=0,
            missing_confirmation=True,
        )

        self.assertIn("Task progress increased from 1 to 2.", summary)
        self.assertIn("made measurable task progress", summary)
        self.assertNotIn("had no observable effect", summary)


if __name__ == "__main__":
    unittest.main()
