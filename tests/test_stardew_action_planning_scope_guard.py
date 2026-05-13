from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _load_stardew_action_planning_module():
    module_name = "_test_stardew_action_planning_process"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    module_path = (
        ROOT
        / "agent"
        / "stardojo"
        / "provider"
        / "process"
        / "action_planning.py"
    )
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ACTION_PLANNING_MODULE = _load_stardew_action_planning_module()
StardewActionPlanningPreprocessProvider = (
    _ACTION_PLANNING_MODULE.StardewActionPlanningPreprocessProvider
)


class _FakeMemory:
    def __init__(self, history: dict[str, list[object]]) -> None:
        self._history = history
        self.working_area: dict[str, object] = {}
        self._current_task_scope = ""

    def get_recent_history(self, key: str, k: int = 1) -> list[object]:
        values = self._history.get(key, [])
        return list(values[:k])


class TestStardewActionPlanningScopeGuard(unittest.TestCase):
    def test_cradle_working_area_is_ignored_when_task_scope_does_not_match(self) -> None:
        fake_memory = _FakeMemory(
            {
                "toolbar_information": [
                    "Currently selected item: slot_index 3: Scythe"
                ],
                "selected_position": [3],
                "summarization": ["Fresh summary"],
                "task_description": ["clear_10_weeds_with_scythe"],
                "subtask_description": [
                    "The current subtask is clear the weeds in front of the porch."
                ],
                "gathered_info": [
                    {
                        "current_menu": {"type": "No Menu"},
                        "inventory": ["Scythe"],
                        "surroundings": "[1, 0]: Weeds",
                        "crops": [],
                        "furniture": [],
                        "npcs": [],
                        "exits": [],
                    }
                ],
            }
        )
        fake_memory._current_task_scope = "stardew:clear_10_weeds_with_scythe"
        stale_cradle_memory = types.SimpleNamespace(
            _current_task_scope="stardew:go_to_coop",
            working_area={
                "subtask_description": "STALE: walk to the coop door.",
                "history_summary": "STALE: previous task context.",
                "action_feedback": "STALE: stale feedback.",
            },
        )
        provider = StardewActionPlanningPreprocessProvider(
            gm=None,
            toolbar_information="",
        )

        with patch.object(_ACTION_PLANNING_MODULE, "memory", fake_memory), patch.object(
            _ACTION_PLANNING_MODULE,
            "_cradle_memory",
            stale_cradle_memory,
        ):
            params = provider()

        self.assertEqual(
            params["subtask_description"],
            "The current subtask is clear the weeds in front of the porch.",
        )
        self.assertNotIn("STALE", params["history_summary"])
        self.assertNotIn("STALE", params["action_feedback"])


if __name__ == "__main__":
    unittest.main()
