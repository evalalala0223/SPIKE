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

from cradle.runner.langgraph_nodes import LangGraphNodes


class _StubProvider:
    def __init__(self, result: dict | None = None) -> None:
        self._result = result or {}

    def __call__(self, *_: object, **__: object) -> dict:
        return dict(self._result)


class _FakeRuntimeMemory:
    def __init__(self) -> None:
        self.working_area: dict[str, object] = {}

    def update_info_history(self, data: dict[str, object]) -> None:
        self.working_area.update(data)


def _make_nodes(runtime_memory: _FakeRuntimeMemory) -> LangGraphNodes:
    providers = {
        "video_clip": _StubProvider({}),
        "self_reflection": _StubProvider({"success": True}),
        "task_inference": _StubProvider({}),
        "action_planning": _StubProvider({}),
        "skill_execute": _StubProvider({}),
    }
    return LangGraphNodes(providers=providers, runtime_memory=runtime_memory)


class TestLangGraphSelfReflectionContext(unittest.TestCase):
    def test_self_reflection_node_syncs_latest_execution_context_into_runtime_memory(self) -> None:
        runtime_memory = _FakeRuntimeMemory()
        nodes = _make_nodes(runtime_memory)

        nodes.self_reflection_node(
            {
                "is_first_step": False,
                "gathered_info": {},
                "execution_result": {"success": True, "pending": False},
                "last_action": 'menu(option="open", menu_name="inventory")',
                "decision_making_reasoning": "Open the inventory to inspect missing seeds.",
                "last_exec_info": {"errors": False, "errors_info": ""},
                "toolbar_information": "slot_index 0: Axe",
                "previous_toolbar_information": "slot_index 0: Axe",
                "history_summary": "Seeds are not visible in the current facts.",
                "subtask_description": "The current subtask is check inventory for Potato Seeds.",
                "subtask_reasoning": "Need to verify whether the seeds are missing before routing to the shop.",
                "action_feedback": "Potato Seeds are not present in the visible hotbar.",
                "previous_actions": ['menu(option="open", menu_name="inventory")'],
                "previous_results": [{"success": True}],
            }
        )

        self.assertEqual(
            runtime_memory.working_area["pre_action"],
            'menu(option="open", menu_name="inventory")',
        )
        self.assertEqual(
            runtime_memory.working_area["action"],
            'menu(option="open", menu_name="inventory")',
        )
        self.assertEqual(
            runtime_memory.working_area["decision_making_reasoning"],
            "Open the inventory to inspect missing seeds.",
        )
        self.assertEqual(
            runtime_memory.working_area["exec_info"],
            {"errors": False, "errors_info": ""},
        )
        self.assertEqual(
            runtime_memory.working_area["action_feedback"],
            "Potato Seeds are not present in the visible hotbar.",
        )


if __name__ == "__main__":
    unittest.main()
