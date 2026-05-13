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

from stardojo.provider.module.task_inference import StardewTaskInferenceProvider
from stardojo.provider.module.action_planning import StardewActionPlanningProvider


class _DummyPlanner:
    def __init__(self, name: str) -> None:
        self.name = name


class _DummyGM:
    pass


class TestProviderInstanceIsolation(unittest.TestCase):
    def test_task_inference_provider_does_not_reuse_previous_planner(self) -> None:
        first = StardewTaskInferenceProvider(planner=_DummyPlanner("first"), gm=_DummyGM())
        second = StardewTaskInferenceProvider(planner=_DummyPlanner("second"), gm=_DummyGM())

        self.assertIsNot(first, second)
        self.assertEqual(first.planner.name, "first")
        self.assertEqual(second.planner.name, "second")

    def test_action_planning_provider_does_not_reuse_previous_planner(self) -> None:
        first = StardewActionPlanningProvider(planner=_DummyPlanner("first"), gm=_DummyGM())
        second = StardewActionPlanningProvider(planner=_DummyPlanner("second"), gm=_DummyGM())

        self.assertIsNot(first, second)
        self.assertEqual(first.planner.name, "first")
        self.assertEqual(second.planner.name, "second")


if __name__ == "__main__":
    unittest.main()
