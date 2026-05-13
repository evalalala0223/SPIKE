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

from cradle.runner.langgraph_routing import should_skip_reflection


class TestLangGraphRouting(unittest.TestCase):
    def test_should_skip_reflection_after_clean_external_feedback_even_with_stale_pending_result(self) -> None:
        state = {
            "is_first_step": False,
            "has_execution_feedback": True,
            "execution_result": {"success": None, "pending": True},
            "success": True,
            "task_progress_delta": 1,
            "last_state_changed": False,
            "last_exec_info": {"errors": False, "errors_info": ""},
            "last_errors_info": "",
            "zero_progress_streak": 0,
            "consecutive_failures": 0,
        }

        self.assertEqual(should_skip_reflection(state), "skip")

    def test_should_skip_reflection_when_only_pending_execution_exists(self) -> None:
        state = {
            "is_first_step": False,
            "execution_result": {"success": None, "pending": True},
            "has_execution_feedback": False,
        }

        self.assertEqual(should_skip_reflection(state), "skip")


if __name__ == "__main__":
    unittest.main()
