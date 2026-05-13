from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
REACT_AGENT_SOURCE_PATH = ROOT / "agent" / "stardojo" / "stardojo_react_agent.py"


class TestRecoveryContextWiring(unittest.TestCase):
    def test_run_action_planning_sets_recovery_context_before_execute_actions(self) -> None:
        source = REACT_AGENT_SOURCE_PATH.read_text(encoding="utf-8")

        execute_actions_call = source.find("self.gm.execute_actions(")
        self.assertNotEqual(execute_actions_call, -1)

        anchor = source.rfind(
            'skill_steps = params.get("skill_steps", [])',
            0,
            execute_actions_call,
        )
        self.assertNotEqual(anchor, -1)

        recovery_context_call = source.rfind(
            "set_recovery_context(",
            anchor,
            execute_actions_call,
        )

        self.assertNotEqual(recovery_context_call, -1)
        self.assertNotEqual(execute_actions_call, -1)
        self.assertLess(recovery_context_call, execute_actions_call)


if __name__ == "__main__":
    unittest.main()
