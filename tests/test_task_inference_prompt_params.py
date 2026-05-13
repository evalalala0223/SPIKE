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

from stardojo import constants


def _load_stardew_task_inference_module():
    module_name = "_test_stardojo_task_inference_process"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    provider_stub = types.ModuleType("stardojo.provider")

    class _BaseProvider:
        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

    provider_stub.BaseProvider = _BaseProvider
    previous_provider_module = sys.modules.get("stardojo.provider")
    sys.modules["stardojo.provider"] = provider_stub
    try:
        module_path = (
            Path(__file__).resolve().parents[1]
            / "agent"
            / "stardojo"
            / "provider"
            / "process"
            / "task_inference.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module spec from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if previous_provider_module is not None:
            sys.modules["stardojo.provider"] = previous_provider_module
        else:
            sys.modules.pop("stardojo.provider", None)


_TASK_INFERENCE_MODULE = _load_stardew_task_inference_module()
StardewTaskInferencePreprocessProvider = (
    _TASK_INFERENCE_MODULE.StardewTaskInferencePreprocessProvider
)


class _FakeMemory:
    def __init__(self, history: dict[str, list[object]], working_area: dict[str, object] | None = None) -> None:
        self._history = history
        self.working_area = working_area or {}

    def get_recent_history(self, key: str, k: int = 1) -> list[object]:
        values = self._history.get(key)
        if not values:
            return ["" for _ in range(k)]
        padded = list(values[:k])
        while len(padded) < k:
            padded.append("")
        return padded


class TestTaskInferencePromptParams(unittest.TestCase):
    def test_navigation_prompt_params_include_grounded_map_context(self) -> None:
        fake_memory = _FakeMemory(
            {
                "task_description": ["go_to_coop"],
                "summarization": ["The player is still outside on the farm."],
                "subtask_description": ["The current subtask is route to the coop entrance and enter it."],
                "subtask_reasoning": ["Bootstrap navigation subtask."],
                "toolbar_information": ["Items in toolbar:\nslot_index 0: Axe\nCurrently selected item: slot_index 0: Axe"],
                "position": [[64, 15]],
                "surroundings": ["[1, 0]: empty\n[0, 1]: empty"],
                "selected_position": [0],
                "chosen_item": [{"currentitem": "Axe", "index": 0}],
                constants.AUGMENTED_IMAGES_MEM_BUCKET: ["C:/tmp/fake_augmented.jpeg"],
                "decision_making_reasoning": [""],
                "self_reflection_reasoning": [""],
                "pre_action": [""],
                "image_description": [""],
            },
            working_area={
                "gathered_info": {
                    "location": "Farm",
                    "position": [64, 15],
                    "facing_direction": "left",
                    "facing_position": [63, 15],
                    "current_menu": {"type": "No Menu"},
                    "buildings": "Coop (door: 2 tiles left, relative offset: x=-2, y=0)",
                    "furniture": "(none)",
                    "npcs": "Cat (1 tile up, relative offset: x=0, y=-1)",
                    "exits": "Bus Stop (3 tiles right, relative offset: x=3, y=0)",
                },
                "location": "Farm",
                "position": [64, 15],
                "action_feedback": "The player is still outside the coop.",
            },
        )
        provider = StardewTaskInferencePreprocessProvider(gm=None)

        with patch.object(_TASK_INFERENCE_MODULE, "memory", fake_memory):
            params = provider()

        self.assertEqual(params["location"], "Farm")
        self.assertEqual(params["current_position"], [64, 15])
        self.assertEqual(params["facing_direction"], "left")
        self.assertEqual(params["facing_position"], [63, 15])
        self.assertEqual(params["current_menu"], {"type": "No Menu"})
        self.assertIn("Coop", params["buildings"])
        self.assertIn("Bus Stop", params["exits"])
        self.assertIn("Cat", params["npcs"])
        self.assertEqual(params["source_type"], "farm_building")
        self.assertIn("enter", params["source_detail"].lower())

    def test_clear_task_prompt_params_include_grounded_debris_summary(self) -> None:
        fake_memory = _FakeMemory(
            {
                "task_description": ["clear_10_weeds_with_scythe"],
                "summarization": ["Scythe selected, no confirmed weed progress yet."],
                "subtask_description": ["The current subtask is clear the nearby weeds."],
                "subtask_reasoning": ["Ground on the visible weed patch rather than the farmhouse."],
                "toolbar_information": ["Items in toolbar:\nslot_index 4: Scythe\nCurrently selected item: slot_index 4: Scythe"],
                "position": [[66, 15]],
                "surroundings": ["[3, 3]: Weeds\n[2, 1]: mailbox\n[0, 1]: Farmhouse"],
                "selected_position": [4],
                "chosen_item": [{"currentitem": "Scythe", "index": 4}],
                constants.AUGMENTED_IMAGES_MEM_BUCKET: ["C:/tmp/fake_augmented.jpeg"],
                "decision_making_reasoning": [""],
                "self_reflection_reasoning": [""],
                "pre_action": [""],
                "image_description": ["The farmhouse is visible with weeds farther out on the farm."],
            },
            working_area={
                "gathered_info": {
                    "location": "Farm",
                    "position": [66, 15],
                    "facing_direction": "right",
                    "facing_position": [67, 15],
                    "current_menu": {"type": "No Menu"},
                    "surroundings": "[3, 3]: Weeds\n[2, 1]: mailbox\n[0, 1]: Farmhouse",
                },
                "task_description": "clear_10_weeds_with_scythe",
                "subtask_description": "The current subtask is clear the nearby weeds.",
            },
        )
        provider = StardewTaskInferencePreprocessProvider(gm=None)

        with patch.object(_TASK_INFERENCE_MODULE, "memory", fake_memory):
            params = provider()

        self.assertIn("weeds", params["nearest_grounded_target_summary"].lower())
        self.assertIn("(3, 3)", params["nearest_grounded_target_summary"])
