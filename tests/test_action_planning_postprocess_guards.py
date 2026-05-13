from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, List
import re
import unittest

from cradle.runner.vllm_client import VLLMClient


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "agent" / "cradle" / "provider" / "process" / "action_planning.py"


def _load_postprocess_guard_class():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))

    class_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "StardewActionPlanningPostprocessProvider":
            class_node = node
            break

    if class_node is None:
        raise AssertionError("StardewActionPlanningPostprocessProvider not found")

    selected_methods = []
    for item in class_node.body:
        if isinstance(item, ast.FunctionDef) and item.name in {
            "_canonicalize_skill_steps",
            "_extract_safe_reasoning_action",
            "_strip_leading_noop_move",
        }:
            selected_methods.append(item)

    extracted_class = ast.ClassDef(
        name="_ExtractedPostprocessGuards",
        bases=[],
        keywords=[],
        body=selected_methods,
        decorator_list=[],
    )
    extracted_module = ast.Module(body=[extracted_class], type_ignores=[])
    ast.fix_missing_locations(extracted_module)

    namespace = {
        "Any": Any,
        "List": List,
        "re": re,
        "VLLMClient": VLLMClient,
        "logger": SimpleNamespace(warn=lambda *args, **kwargs: None),
    }
    exec(compile(extracted_module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["_ExtractedPostprocessGuards"]


_POSTPROCESS_GUARDS = _load_postprocess_guard_class()

def _load_parse_semi_formatted_text():
    module_name = "_test_cradle_json_utils_module"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing.parse_semi_formatted_text

    module_path = ROOT / "agent" / "cradle" / "utils" / "json_utils.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.parse_semi_formatted_text


parse_semi_formatted_text = _load_parse_semi_formatted_text()


class TestActionPlanningPostprocessGuards(unittest.TestCase):
    def test_reasoning_fallback_ignores_doc_examples(self) -> None:
        action = _POSTPROCESS_GUARDS._extract_safe_reasoning_action(
            "Reasoning:\nThe valid action description says call use(up) to use against (0,-1)."
        )

        self.assertEqual(action, "")

    def test_reasoning_fallback_extracts_only_explicit_action_sections(self) -> None:
        action = _POSTPROCESS_GUARDS._extract_safe_reasoning_action(
            "Reasoning:\nNeed to move.\nActions:\n```python\nmove(x=0, y=1)\n```"
        )

        self.assertEqual(action, "move(x=0, y=1)")

    def test_canonicalize_skill_steps_drops_placeholder_actions(self) -> None:
        actions = _POSTPROCESS_GUARDS._canonicalize_skill_steps(
            ["craft(item)", 'use(direction="down")']
        )

        self.assertEqual(actions, ['use(direction="down")'])

    def test_strip_leading_noop_move_drops_only_the_first_placeholder(self) -> None:
        actions = _POSTPROCESS_GUARDS._strip_leading_noop_move(
            ['move(x=0, y=0)', 'use(direction="down")', 'move(x=1, y=0)']
        )

        self.assertEqual(actions, ['use(direction="down")', 'move(x=1, y=0)'])

    def test_parse_semi_formatted_text_extracts_choose_item_from_freeform_reasoning(self) -> None:
        parsed = parse_semi_formatted_text(
            (
                "The seeds are already available. "
                "Let me start by selecting the seeds from slot 5 before planting."
            )
        )

        self.assertEqual(parsed.get("actions"), ["choose_item(slot_index=5)"])

    def test_parse_semi_formatted_text_extracts_directional_move_from_freeform_reasoning(self) -> None:
        parsed = parse_semi_formatted_text(
            "The farmhouse blocks the left tile, so I should move down by 1 tile first."
        )

        self.assertEqual(parsed.get("actions"), ["move(x=0, y=1)"])

    def test_parse_semi_formatted_text_prefers_final_decision_region(self) -> None:
        parsed = parse_semi_formatted_text(
            (
                "I could choose slot 4 to clear weeds first. "
                "However, the immediate next action is to select the Hoe from slot 1."
            )
        )

        self.assertEqual(parsed.get("actions"), ["choose_item(slot_index=1)"])

    def test_parse_semi_formatted_text_ignores_negated_move(self) -> None:
        parsed = parse_semi_formatted_text(
            "Do not move down into the farmhouse. Instead, the next action is move right by 2 tiles."
        )

        self.assertEqual(parsed.get("actions"), ["move(x=2, y=0)"])


if __name__ == "__main__":
    unittest.main()
