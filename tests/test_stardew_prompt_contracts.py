from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / "agent" / "res" / "stardew" / "prompts" / "templates"


def _read_template(name: str) -> str:
    return (TEMPLATES_DIR / name).read_text(encoding="utf-8").lower()


class TestStardewPromptContracts(unittest.TestCase):
    def test_cultivation_prompts_keep_crop_lifecycle_and_exclude_animal_ops(self) -> None:
        for template_name in ("action_planning_cultivation.prompt", "task_inference_cultivation.prompt"):
            with self.subTest(template_name=template_name):
                text = _read_template(template_name)
                self.assertIn("till", text)
                self.assertIn("water", text)
                self.assertIn("harvest", text)
                self.assertNotIn("pet bowl", text)
                self.assertNotIn("feeding bench", text)
                self.assertNotIn("incubator", text)
                self.assertNotIn("animal door", text)

    def test_cortex_crafting_prompts_require_exact_missing_materials(self) -> None:
        task_inference_text = _read_template("task_inference_cortex.prompt")
        self.assertIn("sap is not fiber", task_inference_text)

        action_planning_text = _read_template("action_planning_cortex.prompt")
        self.assertIn("recipe is missing sap", action_planning_text)
        self.assertIn("fiber", action_planning_text)

    def test_cultivation_prompts_distinguish_sowing_from_unfertilized_fertilize_targets(self) -> None:
        for template_name in (
            "action_planning_cortex.prompt",
            "action_planning_littlebrain.prompt",
            "action_planning_cultivation.prompt",
            "task_inference_cultivation.prompt",
            "self_reflection_cultivation.prompt",
        ):
            with self.subTest(template_name=template_name):
                text = _read_template(template_name)
                self.assertIn("empty hoedirt", text)
                self.assertIn("unfertilized hoedirt", text)
                self.assertIn("fertiliz", text)
                self.assertIn("tilled soil", text)
                self.assertNotIn("fertilizing is only valid on explicit empty hoedirt", text)
                self.assertNotIn(
                    "for sowing and fertilizing, only treat explicit empty hoedirt",
                    text,
                )

    def test_cultivation_action_prompt_forbids_blind_inventory_search_and_slot5_anchor(self) -> None:
        text = (TEMPLATES_DIR / "action_planning_cultivation.prompt").read_text(encoding="utf-8").lower()

        self.assertIn("slots 0-35", text)
        self.assertIn("do not open inventory just to search for hidden slots", text)
        self.assertNotIn("choose_item(slot_index=5)", text)

    def test_cultivation_prompts_anchor_till_to_grounded_targets(self) -> None:
        action_text = _read_template("action_planning_cultivation.prompt")
        self.assertIn("nearest grounded target summary", action_text)
        self.assertIn('turn left/right', action_text)
        self.assertIn("grounded reposition move", action_text)

        inference_text = _read_template("task_inference_cultivation.prompt")
        self.assertIn("count it as existing progress", inference_text)
        self.assertIn('turn left" or "turn right', inference_text)

    def test_farm_ops_prompts_cover_required_animal_building_targets(self) -> None:
        required_terms = (
            "coop",
            "barn",
            "animal door hatch",
            "pet bowl",
            "feeding bench",
            "incubator",
            "hay",
            "silo",
        )

        for template_name in ("action_planning_farm_ops.prompt", "task_inference_farm_ops.prompt"):
            with self.subTest(template_name=template_name):
                text = _read_template(template_name)
                for term in required_terms:
                    self.assertIn(term, text)

    def test_farm_ops_action_prompt_forbids_standard_layout_blind_route(self) -> None:
        text = _read_template("action_planning_farm_ops.prompt")

        self.assertIn("do not invent a standard farm layout", text)
        self.assertIn("short grounded search move of 1-3 tiles", text)

    def test_farm_ops_prompts_allow_local_scytheable_fallback_for_hay_when_grass_is_missing(self) -> None:
        for template_name in ("action_planning_farm_ops.prompt", "task_inference_farm_ops.prompt"):
            with self.subTest(template_name=template_name):
                text = _read_template(template_name)
                self.assertIn("forage_10_hay_with_scythe", text)
                self.assertIn("grass", text)
                self.assertIn("weeds", text)

    def test_shopping_prompts_cover_counter_menu_service_and_shipping_bin(self) -> None:
        required_terms = ("counter", "menu", "service", "shipping bin")

        for template_name in ("action_planning_shopping.prompt", "task_inference_shopping.prompt"):
            with self.subTest(template_name=template_name):
                text = _read_template(template_name)
                for term in required_terms:
                    self.assertIn(term, text)

    def test_shopping_prompts_keep_sell_flow_local_once_counter_or_menu_is_reached(self) -> None:
        for template_name in ("action_planning_shopping.prompt", "task_inference_shopping.prompt"):
            with self.subTest(template_name=template_name):
                text = _read_template(template_name)
                self.assertIn("sell_*", text)
                self.assertIn("target item", text)
                self.assertIn("counter", text)
                self.assertIn("menu", text)

    def test_shopping_prompts_consume_acquisition_hint_placeholders(self) -> None:
        required_placeholders = ("<$target_item$>", "<$source_type$>", "<$source_detail$>")

        for template_name in ("action_planning_shopping.prompt", "task_inference_shopping.prompt"):
            with self.subTest(template_name=template_name):
                text = (TEMPLATES_DIR / template_name).read_text(encoding="utf-8")
                for placeholder in required_placeholders:
                    self.assertIn(placeholder, text)

    def test_profile_specific_action_prompts_keep_structured_current_facts(self) -> None:
        prompt_requirements = {
            "action_planning_cultivation.prompt": (
                "<$front_tile_summary$>",
                "<$blocked_recovery_hint$>",
                "<$current_blocker_signature$>",
                "<$nearest_grounded_target_summary$>",
            ),
            "action_planning_farm_clearup.prompt": (
                "<$current_position$>",
                "<$facing_direction$>",
                "<$surroundings$>",
                "<$buildings$>",
                "<$current_blocker_signature$>",
                "<$nearest_grounded_target_summary$>",
            ),
            "action_planning_shopping.prompt": (
                "<$current_menu$>",
                "<$inventory$>",
                "<$surroundings$>",
                "<$exits$>",
                "<$current_blocker_signature$>",
                "<$nearest_grounded_target_summary$>",
            ),
        }

        for template_name, placeholders in prompt_requirements.items():
            with self.subTest(template_name=template_name):
                text = (TEMPLATES_DIR / template_name).read_text(encoding="utf-8")
                for placeholder in placeholders:
                    self.assertIn(placeholder, text)

    def test_profile_specific_action_prompts_warn_against_blocked_building_directions(self) -> None:
        for template_name in (
            "action_planning_cultivation.prompt",
            "action_planning_farm_clearup.prompt",
            "action_planning_shopping.prompt",
        ):
            with self.subTest(template_name=template_name):
                text = _read_template(template_name)
                self.assertIn("farmhouse", text)
                self.assertIn("porch", text)
                self.assertIn("building", text)

    def test_farm_clearup_prompts_do_not_restore_old_rigid_grid_language(self) -> None:
        forbidden_terms = ("clearing area", "house spans")

        for template_name in ("action_planning_farm_clearup.prompt", "task_inference_farm_clearup.prompt"):
            with self.subTest(template_name=template_name):
                text = _read_template(template_name)
                for term in forbidden_terms:
                    self.assertNotIn(term, text)

    def test_cortex_prompt_consumes_blocker_context_placeholders(self) -> None:
        text = (TEMPLATES_DIR / "action_planning_cortex.prompt").read_text(encoding="utf-8")

        self.assertIn("<$front_tile_summary$>", text)
        self.assertIn("<$blocked_recovery_hint$>", text)
        self.assertIn("<$current_blocker_signature$>", text)
        self.assertIn("<$nearest_grounded_target_summary$>", text)

    def test_cortex_prompt_forbids_long_jump_through_visible_blockers(self) -> None:
        text = _read_template("action_planning_cortex.prompt")

        self.assertIn("do not keep the long move", text)
        self.assertIn("short grounded probe move of 1-3 tiles", text)

    def test_combat_prompts_force_local_attack_when_enemy_is_already_close(self) -> None:
        for template_name in ("action_planning_cortex.prompt", "task_inference_cortex.prompt"):
            with self.subTest(template_name=template_name):
                text = _read_template(template_name)
                self.assertIn("kill_*", text)
                self.assertIn("adjacent", text)
                self.assertIn("rusty sword", text)


if __name__ == "__main__":
    unittest.main()
