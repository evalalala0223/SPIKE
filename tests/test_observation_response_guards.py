from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "env" / "actions.py"
SPEC = importlib.util.spec_from_file_location("env_actions_testable", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load env.actions from {MODULE_PATH}")
env_actions = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(env_actions)


class TestObservationResponseGuards(unittest.TestCase):
    def test_transient_observation_placeholder_matches_message_received(self) -> None:
        self.assertTrue(
            env_actions.ActionProxy._is_transient_observation_placeholder(
                "observe_v2%3",
                "Message received",
            )
        )

    def test_transient_observation_placeholder_ignores_normal_non_observe_messages(self) -> None:
        self.assertFalse(
            env_actions.ActionProxy._is_transient_observation_placeholder(
                "move_relative%1%0",
                "Message received",
            )
        )

    def test_transient_observation_placeholder_ignores_valid_json(self) -> None:
        self.assertFalse(
            env_actions.ActionProxy._is_transient_observation_placeholder(
                "observe_v2%3",
                '{"Player": {"Position": {"X": 1, "Y": 2}}}',
            )
        )


if __name__ == "__main__":
    unittest.main()
