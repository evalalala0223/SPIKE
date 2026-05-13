from __future__ import annotations

from collections import Counter
from pathlib import Path
import unittest

import yaml

from stardojo.utils.prompt_profile_utils import (
    build_task_specific_planner_params,
    infer_stardew_prompt_profile,
    resolve_prompt_profile_template_paths,
    resolve_dual_brain_bigbrain_template_paths,
)
from stardojo.utils.task_bootstrap import get_task_spec


ROOT = Path(__file__).resolve().parents[1]
TASK_SUITE_DIR = ROOT / "env" / "tasks" / "task_suite"


class TestStardewPromptRouting(unittest.TestCase):
    def test_lite100_prompt_profile_snapshot(self) -> None:
        expected = {
            "combat_lite.yaml": {"combat": 12},
            "crafting_lite.yaml": {"crafting": 14},
            "exploration_lite.yaml": {"navigation": 25, "farm_clearup": 2, "farm_ops": 1},
            "farming_lite.yaml": {"farm_clearup": 3, "cultivation": 8, "farm_ops": 10},
            "social_lite.yaml": {"shopping": 16, "social": 9},
        }

        actual = {}
        for path in sorted(TASK_SUITE_DIR.glob("*_lite.yaml")):
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            counter = Counter()
            for task_name in raw:
                spec = get_task_spec(task_name)
                profile = infer_stardew_prompt_profile(task_name, task_spec=spec)
                counter[profile] += 1
            actual[path.name] = dict(counter)

        self.assertEqual(actual, expected)

    def test_representative_tasks_route_to_expected_profiles(self) -> None:
        cases = {
            "go_to_bed": "navigation",
            "go_to_coop": "navigation",
            "go_to_the_mines_2nd_floor": "navigation",
            "chop_10_wood_with_axe": "farm_clearup",
            "purchase_1_chicken": "shopping",
            "ship_1_parsnip_with_shipping_bin": "shopping",
            "upgrade_to_copper_pickaxe": "shopping",
            "break_5_geode": "shopping",
            "forage_10_hay_with_scythe": "farm_ops",
            "open_1_deluxe_coop": "farm_ops",
            "produce_1_refined_quartz_with_furnace": "crafting",
            "earn_200_friendship_with_harvey": "social",
            "date_abigail": "social",
            "break_up_with_abigail": "social",
            "propose_to_abigail": "social",
            "craft_1_cherry_bomb": "crafting",
            "kill_1_green_slime_with_rusty_sword": "combat",
        }

        for task_name, expected_profile in cases.items():
            spec = get_task_spec(task_name)
            with self.subTest(task_name=task_name):
                self.assertEqual(
                    infer_stardew_prompt_profile(task_name, task_spec=spec),
                    expected_profile,
                )

    def test_free_text_fallback_prioritizes_action_intent_over_route_prefix(self) -> None:
        cases = {
            "go to marnie and buy hay": "shopping",
            "go to robin and build a coop": "shopping",
            "return home and ship parsnip": "shopping",
            "go to abigail and propose": "social",
            "gift cauliflower to abigail": "social",
            "earn friendship with abigail": "social",
            "go to bed": "navigation",
        }

        for task_text, expected_profile in cases.items():
            with self.subTest(task_text=task_text):
                self.assertEqual(infer_stardew_prompt_profile(task_text), expected_profile)

    def test_build_task_specific_planner_params_use_expected_template_mapping(self) -> None:
        base = {
            "prompt_paths": {
                "templates": {
                    "action_planning": "dummy",
                    "information_gathering": "dummy",
                    "self_reflection": "dummy",
                    "task_inference": "dummy",
                    "information_toolbar_gathering": "dummy",
                }
            }
        }

        cases = {
            "purchase_1_chicken": {
                "profile": "shopping",
                "action_planning": "./res/stardew/prompts/templates/action_planning_shopping.prompt",
                "information_gathering": "./res/stardew/prompts/templates/information_gathering_cultivation.prompt",
                "self_reflection": "./res/stardew/prompts/templates/self_reflection_general.prompt",
                "task_inference": "./res/stardew/prompts/templates/task_inference_shopping.prompt",
                "information_toolbar_gathering": "./res/stardew/prompts/templates/information_toolbar_gathering_cultivation.prompt",
            },
            "open_1_deluxe_coop": {
                "profile": "farm_ops",
                "action_planning": "./res/stardew/prompts/templates/action_planning_farm_ops.prompt",
                "information_gathering": "./res/stardew/prompts/templates/information_gathering_cultivation.prompt",
                "self_reflection": "./res/stardew/prompts/templates/self_reflection_general.prompt",
                "task_inference": "./res/stardew/prompts/templates/task_inference_farm_ops.prompt",
                "information_toolbar_gathering": "./res/stardew/prompts/templates/information_toolbar_gathering_cultivation.prompt",
            },
            "clear_10_weeds_with_scythe": {
                "profile": "farm_clearup",
                "action_planning": "./res/stardew/prompts/templates/action_planning_farm_clearup.prompt",
                "information_gathering": "./res/stardew/prompts/templates/information_gathering_farm_clearup.prompt",
                "self_reflection": "./res/stardew/prompts/templates/self_reflection_farm_clearup.prompt",
                "task_inference": "./res/stardew/prompts/templates/task_inference_farm_clearup.prompt",
                "information_toolbar_gathering": "./res/stardew/prompts/templates/information_toolbar_gathering_farm_clearup.prompt",
            },
        }

        for task_name, expected in cases.items():
            with self.subTest(task_name=task_name):
                params, profile = build_task_specific_planner_params(base, task_name)
                self.assertEqual(profile, expected["profile"])
                for key, value in expected.items():
                    if key == "profile":
                        continue
                    self.assertEqual(params["prompt_paths"]["templates"][key], value)

    def test_resolve_prompt_profile_template_paths_overwrites_stale_values(self) -> None:
        existing = {
            "action_planning": "./res/stardew/prompts/templates/action_planning_farm_clearup.prompt",
            "task_inference": "./res/stardew/prompts/templates/task_inference_farm_clearup.prompt",
        }

        resolved, audit, preserve = resolve_prompt_profile_template_paths(
            "farm_ops",
            existing_templates=existing,
            template_keys=("action_planning", "task_inference"),
        )

        self.assertTrue(preserve)
        self.assertEqual(
            resolved["action_planning"],
            "./res/stardew/prompts/templates/action_planning_farm_ops.prompt",
        )
        self.assertEqual(
            resolved["task_inference"],
            "./res/stardew/prompts/templates/task_inference_farm_ops.prompt",
        )
        self.assertFalse(audit["action_planning"]["fallback_used"])
        self.assertFalse(audit["task_inference"]["fallback_used"])

    def test_resolve_dual_brain_bigbrain_template_paths_preserves_only_expected_profiles(self) -> None:
        preserve_cases = {
            "cultivation": (
                "./res/stardew/prompts/templates/action_planning_cultivation.prompt",
                "./res/stardew/prompts/templates/task_inference_cultivation.prompt",
            ),
            "farm_clearup": (
                "./res/stardew/prompts/templates/action_planning_farm_clearup.prompt",
                "./res/stardew/prompts/templates/task_inference_farm_clearup.prompt",
            ),
            "farm_ops": (
                "./res/stardew/prompts/templates/action_planning_farm_ops.prompt",
                "./res/stardew/prompts/templates/task_inference_farm_ops.prompt",
            ),
            "shopping": (
                "./res/stardew/prompts/templates/action_planning_shopping.prompt",
                "./res/stardew/prompts/templates/task_inference_shopping.prompt",
            ),
        }
        swap_cases = {"navigation", "social", "crafting", "combat"}

        for prompt_profile, (action_path, task_path) in preserve_cases.items():
            with self.subTest(prompt_profile=prompt_profile):
                paths, preserve_profile_templates = resolve_dual_brain_bigbrain_template_paths(prompt_profile)
                self.assertTrue(preserve_profile_templates)
                self.assertEqual(paths["action_planning"], action_path)
                self.assertEqual(paths["task_inference"], task_path)

        for prompt_profile in swap_cases:
            with self.subTest(prompt_profile=prompt_profile):
                paths, preserve_profile_templates = resolve_dual_brain_bigbrain_template_paths(prompt_profile)
                self.assertFalse(preserve_profile_templates)
                self.assertEqual(
                    paths["action_planning"],
                    "./res/stardew/prompts/templates/action_planning_cortex.prompt",
                )
                self.assertEqual(
                    paths["task_inference"],
                    "./res/stardew/prompts/templates/task_inference_cortex.prompt",
                )

    def test_langgraph_dependency_declared_when_enabled(self) -> None:
        config_path = ROOT / "agent" / "conf" / "enhanced_config.yaml"
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        use_langgraph = bool(((config.get("features") or {}).get("use_langgraph")))

        requirements_path = ROOT / "requirements.txt"
        requirements_text = requirements_path.read_text(encoding="utf-8")

        if use_langgraph:
            self.assertTrue(
                "langgraph==" in requirements_text or "\nlanggraph\n" in requirements_text
            )

    def test_dual_brain_runners_use_profile_aware_template_helper(self) -> None:
        react_agent_text = (ROOT / "agent" / "stardojo" / "stardojo_react_agent.py").read_text(encoding="utf-8")
        cradle_runner_text = (ROOT / "agent" / "cradle" / "runner" / "stardew_runner.py").read_text(encoding="utf-8")

        for file_text in (react_agent_text, cradle_runner_text):
            self.assertIn("sync_planner_prompt_templates", file_text)

        self.assertIn("preserving profile-specific BigBrain templates", react_agent_text)
        self.assertIn("swapping BigBrain templates to cortex", react_agent_text)
        self.assertIn("preserving profile-specific BigBrain templates", cradle_runner_text)
        self.assertIn("swapping BigBrain templates to cortex", cradle_runner_text)


if __name__ == "__main__":
    unittest.main()
