from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types
import unittest


def _load_cradle_task_inference_process():
    module_name = "_test_cradle_task_inference_process"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    fake_provider_module = types.ModuleType("cradle.provider")

    class _BaseProvider:
        def __init__(self, *args, **kwargs) -> None:
            pass

    fake_provider_module.BaseProvider = _BaseProvider

    original_provider_module = sys.modules.get("cradle.provider")
    sys.modules["cradle.provider"] = fake_provider_module
    try:
        module_path = (
            Path(__file__).resolve().parents[1]
            / "agent"
            / "cradle"
            / "provider"
            / "process"
            / "task_inference.py"
        )
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if original_provider_module is not None:
            sys.modules["cradle.provider"] = original_provider_module
        else:
            sys.modules.pop("cradle.provider", None)


def _load_stardojo_task_inference_process():
    module_name = "_test_stardojo_task_inference_process"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    fake_provider_module = types.ModuleType("stardojo.provider")

    class _BaseProvider:
        def __init__(self, *args, **kwargs) -> None:
            pass

    fake_provider_module.BaseProvider = _BaseProvider

    original_provider_module = sys.modules.get("stardojo.provider")
    sys.modules["stardojo.provider"] = fake_provider_module
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
            raise RuntimeError(f"Failed to load spec for {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if original_provider_module is not None:
            sys.modules["stardojo.provider"] = original_provider_module
        else:
            sys.modules.pop("stardojo.provider", None)


cradle_task_inference_process = _load_cradle_task_inference_process()
stardojo_task_inference_process = _load_stardojo_task_inference_process()


class TestStardewPromptAlignment(unittest.TestCase):
    def setUp(self) -> None:
        cradle_task_inference_process.memory.reset_runtime_state()
        stardojo_task_inference_process.memory.reset_runtime_state()

    def test_action_planning_template_aliases_have_explicit_producers(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        action_prompt = (
            repo_root / "agent" / "res" / "stardew" / "prompts" / "templates" / "action_planning_cortex.prompt"
        ).read_text(encoding="utf-8")
        cradle_preprocess = (
            repo_root / "agent" / "cradle" / "provider" / "process" / "action_planning.py"
        ).read_text(encoding="utf-8")
        stardojo_preprocess = (
            repo_root / "agent" / "stardojo" / "provider" / "process" / "action_planning.py"
        ).read_text(encoding="utf-8")
        langgraph_nodes = (
            repo_root / "agent" / "cradle" / "runner" / "langgraph_nodes.py"
        ).read_text(encoding="utf-8")

        self.assertIn("<$action$>", action_prompt)
        self.assertIn("<$action_planning_reasoning$>", action_prompt)
        self.assertIn('"action": previous_action', cradle_preprocess)
        self.assertIn('"action_planning_reasoning": action_planning_reasoning', cradle_preprocess)
        self.assertIn('"action_planning_reasoning": decision_making_reasoning', cradle_preprocess)
        self.assertIn('_safe_recent("action_feedback", "")', cradle_preprocess)
        self.assertIn('latest_execution_summary', cradle_preprocess)
        self.assertIn('"action": previous_action', stardojo_preprocess)
        self.assertIn('"action_planning_reasoning": action_planning_reasoning', stardojo_preprocess)
        self.assertIn('"action_planning_reasoning": decision_making_reasoning', stardojo_preprocess)
        self.assertIn('"action_planning_reasoning": (', langgraph_nodes)

    def test_task_inference_template_current_fact_fields_have_explicit_producers(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        task_prompt = (
            repo_root / "agent" / "res" / "stardew" / "prompts" / "templates" / "task_inference_cortex.prompt"
        ).read_text(encoding="utf-8")
        cradle_preprocess = (
            repo_root / "agent" / "cradle" / "provider" / "process" / "task_inference.py"
        ).read_text(encoding="utf-8")
        stardojo_preprocess = (
            repo_root / "agent" / "stardojo" / "provider" / "process" / "task_inference.py"
        ).read_text(encoding="utf-8")
        langgraph_nodes = (
            repo_root / "agent" / "cradle" / "runner" / "langgraph_nodes.py"
        ).read_text(encoding="utf-8")

        self.assertIn("<$current_toolbar_fact$>", task_prompt)
        self.assertIn("<$image_description$>", task_prompt)
        self.assertIn("<$surroundings$>", task_prompt)
        self.assertIn('"current_toolbar_fact": current_toolbar_fact', cradle_preprocess)
        self.assertIn('"image_description": image_description', cradle_preprocess)
        self.assertIn('"surroundings": surroundings', cradle_preprocess)
        self.assertIn('"image_description": image_description', stardojo_preprocess)
        self.assertIn("build_task_acquisition_context", cradle_preprocess)
        self.assertIn("extract_stardew_prompt_fact_fields", cradle_preprocess)
        self.assertIn('"image_description": (', langgraph_nodes)

    def test_cortex_prompts_define_unknown_surroundings_and_empty_hoedirt_rules(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        action_prompt = (
            repo_root / "agent" / "res" / "stardew" / "prompts" / "templates" / "action_planning_cortex.prompt"
        ).read_text(encoding="utf-8")
        littlebrain_prompt = (
            repo_root / "agent" / "res" / "stardew" / "prompts" / "templates" / "action_planning_littlebrain.prompt"
        ).read_text(encoding="utf-8")
        task_prompt = (
            repo_root / "agent" / "res" / "stardew" / "prompts" / "templates" / "task_inference_cortex.prompt"
        ).read_text(encoding="utf-8")

        self.assertIn("unknown, not empty", action_prompt)
        self.assertIn("empty HoeDirt / tilled soil", action_prompt)
        self.assertIn("unknown, not empty", littlebrain_prompt)
        self.assertIn("nearby valid target", task_prompt)
        self.assertIn("different item, menu, shop, or route", task_prompt)

    def test_profile_specific_big_brain_action_prompts_include_front_tile_grounding(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        for template_name in (
            "action_planning_cultivation.prompt",
            "action_planning_farm_clearup.prompt",
            "action_planning_shopping.prompt",
        ):
            prompt_text = (
                repo_root / "agent" / "res" / "stardew" / "prompts" / "templates" / template_name
            ).read_text(encoding="utf-8")
            self.assertIn("<$front_tile_summary$>", prompt_text, msg=template_name)
            self.assertIn("<$blocked_recovery_hint$>", prompt_text, msg=template_name)
            self.assertIn("<$current_blocker_signature$>", prompt_text, msg=template_name)
            self.assertIn("<$nearest_grounded_target_summary$>", prompt_text, msg=template_name)

    def test_task_inference_skip_has_current_fact_conflict_guard(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        langgraph_nodes = (
            repo_root / "agent" / "cradle" / "runner" / "langgraph_nodes.py"
        ).read_text(encoding="utf-8")

        self.assertIn("_subtask_conflicts_with_current_facts", langgraph_nodes)
        self.assertIn("previous subtask conflicts with current facts", langgraph_nodes)

    def test_langgraph_syncs_prompt_fact_fields_for_three_stardew_nodes(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        langgraph_nodes = (
            repo_root / "agent" / "cradle" / "runner" / "langgraph_nodes.py"
        ).read_text(encoding="utf-8")

        self.assertGreaterEqual(
            langgraph_nodes.count("prompt_fact_fields = extract_stardew_prompt_fact_fields("),
            3,
        )
        self.assertIn("is_first_step = bool(state.get('is_first_step', False))", langgraph_nodes)

    def test_task_inference_preprocess_keeps_normalized_current_facts(self) -> None:
        memory = cradle_task_inference_process.memory
        memory.update_info_history(
            {
                "task_description": "sow_1_dirt_with_potato_seeds",
                "summarization": "previous summary",
                "subtask_description": "locate and equip Potato Seeds",
                "subtask_reasoning": "seeds are needed for sowing",
                "toolbar_information": "CURRENT TOOLBAR",
                "decision_making_reasoning": "previous planner reasoning",
                "self_reflection_reasoning": "previous reflection",
                "self_reflection_progress": 0,
                "self_reflection_status_summary": "",
                "pre_action": "choose_item(slot_index=12)",
                "action": "choose_item(slot_index=12)",
                "position": [10, 10],
                "gathered_info": {
                    "chosen_item": {"currentitem": "Potato Seeds"},
                },
            }
        )
        memory.working_area["toolbar_information"] = "STALE TOOLBAR"
        memory.working_area["surroundings"] = [
            {"position": [11, 10], "terrain": "Grass", "object": "Twig"},
        ]
        memory.working_area["chosen_item"] = {"currentitem": "Potato Seeds"}
        memory.working_area["selected_item_name"] = ""

        provider = cradle_task_inference_process.StardewTaskInferencePreprocessProvider(gm=None)
        processed = provider()

        self.assertEqual(processed["toolbar_information"], "CURRENT TOOLBAR")
        self.assertEqual(processed["selected_item_name"], "Potato Seeds")
        self.assertIsInstance(processed["surroundings"], str)
        self.assertIn("[1, 0]: Grass, Twig", processed["surroundings"])

    def test_stardojo_task_inference_preprocess_keeps_scalar_reasoning_and_image_description(self) -> None:
        memory = stardojo_task_inference_process.memory
        memory.update_info_history(
            {
                "task_description": "buy_1_parsnip_seeds",
                "summarization": "previous summary",
                "subtask_description": "walk to Pierre's",
                "subtask_reasoning": "need to buy seeds",
                "toolbar_information": "CURRENT TOOLBAR",
                "decision_making_reasoning": "previous planner reasoning",
                "self_reflection_reasoning": "previous reflection",
                "pre_action": "move(x=1, y=0)",
                "position": [10, 10],
            }
        )
        memory.working_area["gathered_info"] = {
            "description": "current image desc",
        }
        memory.working_area["surroundings"] = [
            {"position": [11, 10], "terrain": "Grass", "object": "Twig"},
        ]

        provider = stardojo_task_inference_process.StardewTaskInferencePreprocessProvider(gm=None)
        processed = provider()

        self.assertEqual(processed["previous_reasoning"], "previous planner reasoning")
        self.assertEqual(processed["self_reflection_reasoning"], "previous reflection")
        self.assertEqual(processed["image_description"], "current image desc")
        self.assertIsInstance(processed["surroundings"], str)
        self.assertIn("[1, 0]: Grass, Twig", processed["surroundings"])


if __name__ == "__main__":
    unittest.main()
