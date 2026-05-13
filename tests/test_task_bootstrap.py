from __future__ import annotations

import unittest

from stardojo.utils.task_bootstrap import (
    _infer_task_spec_from_text,
    build_initial_subtask,
    build_task_acquisition_context,
)


class TestTaskBootstrap(unittest.TestCase):
    def test_fertilize_preloaded_task_does_not_request_obtain_step(self) -> None:
        subtask = build_initial_subtask("fertilize_5_dirt_with_basic_retaining_soil")

        self.assertIn("select Basic Retaining Soil", subtask)
        self.assertNotIn("obtain Basic Retaining Soil", subtask)

    def test_sow_preloaded_task_does_not_request_obtain_step(self) -> None:
        subtask = build_initial_subtask("sow_5_dirt_with_cauliflower_seeds")

        self.assertIn("select Cauliflower Seeds", subtask)
        self.assertNotIn("obtain Cauliflower Seeds", subtask)

    def test_preloaded_fertilize_task_marks_inventory_source(self) -> None:
        context = build_task_acquisition_context("fertilize_5_dirt_with_basic_retaining_soil")

        self.assertEqual(context["source_type"], "inventory_preloaded")
        self.assertIn("preloaded", context["source_detail"].lower())

    def test_non_preloaded_speed_gro_uses_shop_route(self) -> None:
        subtask = build_initial_subtask("fertilize_1_dirt_with_speed_gro")
        context = build_task_acquisition_context("fertilize_1_dirt_with_speed_gro")

        self.assertIn("pierre", subtask.lower())
        self.assertEqual(context["source_type"], "inventory_or_shop")
        self.assertIn("pierre", context["source_detail"].lower())

    def test_non_preloaded_potato_seeds_use_shop_route(self) -> None:
        subtask = build_initial_subtask("sow_1_dirt_with_potato_seeds")
        context = build_task_acquisition_context("sow_1_dirt_with_potato_seeds")

        self.assertIn("pierre", subtask.lower())
        self.assertEqual(context["source_type"], "inventory_or_shop")
        self.assertIn("pierre", context["source_detail"].lower())

    def test_multi_tool_clear_bootstrap_avoids_generic_select_all_tools_fallback(self) -> None:
        subtask = build_initial_subtask("clear_30_debris_with_scythe_and_pickaxe_and_axe")

        self.assertIn("move off blocking structures", subtask.lower())
        self.assertIn("matching tool", subtask.lower())
        self.assertNotIn("select scythe, pickaxe, axe", subtask.lower())

    def test_sleep_bootstrap_mentions_confirming_prompt(self) -> None:
        subtask = build_initial_subtask("go_to_bed")

        self.assertIn("bed", subtask.lower())
        self.assertIn("confirm", subtask.lower())

    def test_cultivate_and_harvest_bootstrap_mentions_full_growth_cycle(self) -> None:
        subtask = build_initial_subtask("cultivate_and_harvest_1_garlic")

        self.assertIn("garlic seeds", subtask.lower())
        self.assertIn("till", subtask.lower())
        self.assertIn("water", subtask.lower())
        self.assertIn("sleep", subtask.lower())
        self.assertIn("harvest", subtask.lower())

    def test_cultivate_and_harvest_acquisition_context_uses_preloaded_seeds(self) -> None:
        context = build_task_acquisition_context("cultivate_and_harvest_1_garlic")

        self.assertEqual(context["target_item"], "Garlic Seeds")
        self.assertEqual(context["source_type"], "inventory_preloaded")
        self.assertIn("watering", context["source_detail"].lower())

    def test_pet_animal_bootstrap_routes_to_coop_then_barn(self) -> None:
        subtask = build_initial_subtask("pet_3_animal")

        self.assertIn("coop", subtask.lower())
        self.assertIn("barn", subtask.lower())
        self.assertIn("pet", subtask.lower())

    def test_pet_animal_acquisition_context_uses_animal_housing_route(self) -> None:
        context = build_task_acquisition_context("pet_3_animal")

        self.assertEqual(context["target_item"], "Animal")
        self.assertEqual(context["source_type"], "animal_housing")
        self.assertIn("coop", context["source_detail"].lower())
        self.assertIn("barn", context["source_detail"].lower())

    def test_go_to_coop_bootstrap_enters_building(self) -> None:
        subtask = build_initial_subtask("go_to_coop")
        context = build_task_acquisition_context("go_to_coop")

        self.assertIn("coop", subtask.lower())
        self.assertIn("enter", subtask.lower())
        self.assertEqual(context["source_type"], "farm_building")
        self.assertIn("enter", context["source_detail"].lower())

    def test_go_to_bus_stop_bootstrap_uses_east_exit_hint(self) -> None:
        subtask = build_initial_subtask("go_to_bus_stop")
        context = build_task_acquisition_context("go_to_bus_stop")

        self.assertIn("east exit", subtask.lower())
        self.assertEqual(context["source_type"], "navigation")
        self.assertIn("east exit", context["source_detail"].lower())

    def test_go_to_backwoods_bootstrap_uses_pet_bowl_route_hint(self) -> None:
        subtask = build_initial_subtask("go_to_backwoods")
        context = build_task_acquisition_context("go_to_backwoods")

        self.assertIn("pet bowl", subtask.lower())
        self.assertEqual(context["source_type"], "navigation")
        self.assertIn("pet bowl", context["source_detail"].lower())

    def test_open_deluxe_coop_bootstrap_targets_animal_door(self) -> None:
        subtask = build_initial_subtask("open_1_deluxe_coop")
        context = build_task_acquisition_context("open_1_deluxe_coop")

        self.assertIn("animal door", subtask.lower())
        self.assertEqual(context["source_type"], "animal_door")
        self.assertIn("entering the building does not", context["source_detail"].lower())

    def test_fill_pet_bowl_bootstrap_routes_to_farmhouse_pet_area(self) -> None:
        subtask = build_initial_subtask("fill_1_pet_bowl_with_watering_can")
        context = build_task_acquisition_context("fill_1_pet_bowl_with_watering_can")

        self.assertIn("pet bowl", subtask.lower())
        self.assertIn("farmhouse", subtask.lower())
        self.assertEqual(context["source_type"], "pet_area")
        self.assertIn("farmhouse", context["source_detail"].lower())

    def test_fill_feeding_bench_bootstrap_routes_inside_animal_building(self) -> None:
        subtask = build_initial_subtask("fill_1_feeding_bench_with_hay")
        context = build_task_acquisition_context("fill_1_feeding_bench_with_hay")

        self.assertIn("feeding bench", subtask.lower())
        self.assertIn("coop or barn", subtask.lower())
        self.assertEqual(context["source_type"], "animal_housing")
        self.assertIn("hopper", context["source_detail"].lower())

    def test_egg_and_milk_bootstrap_route_to_correct_buildings(self) -> None:
        egg_subtask = build_initial_subtask("harvest_1_egg")
        egg_context = build_task_acquisition_context("harvest_1_egg")
        milk_subtask = build_initial_subtask("harvest_1_milk_with_milk_pail")
        milk_context = build_task_acquisition_context("harvest_1_milk_with_milk_pail")

        self.assertIn("coop", egg_subtask.lower())
        self.assertEqual(egg_context["source_type"], "animal_housing")
        self.assertIn("coop", egg_context["source_detail"].lower())
        self.assertIn("milk pail", milk_subtask.lower())
        self.assertIn("barn", milk_subtask.lower())
        self.assertEqual(milk_context["target_item"], "Milk Pail")
        self.assertEqual(milk_context["source_type"], "inventory_preloaded")
        self.assertIn("preloaded", milk_context["source_detail"].lower())
        self.assertIn("barn", milk_context["source_detail"].lower())

    def test_tool_required_harvest_bootstrap_uses_required_tool(self) -> None:
        subtask = build_initial_subtask("harvest_1_wool_with_shears")
        context = build_task_acquisition_context("harvest_1_wool_with_shears")

        self.assertIn("shears", subtask.lower())
        self.assertIn("collect", subtask.lower())
        self.assertEqual(context["target_item"], "Shears")
        self.assertEqual(context["source_type"], "inventory_or_tool")

    def test_incubate_and_pet_friendship_bootstrap_are_multi_step(self) -> None:
        incubate_subtask = build_initial_subtask("incubate_1_chicken_with_incubator")
        incubate_context = build_task_acquisition_context("incubate_1_chicken_with_incubator")
        friendship_subtask = build_initial_subtask("earn_50_friendship_with_1_cat")
        friendship_context = build_task_acquisition_context("earn_50_friendship_with_1_cat")

        self.assertIn("incubator", incubate_subtask.lower())
        self.assertIn("coop", incubate_subtask.lower())
        self.assertEqual(incubate_context["source_type"], "animal_housing")
        self.assertIn("incubator", incubate_context["source_detail"].lower())
        self.assertIn("sleep", friendship_subtask.lower())
        self.assertEqual(friendship_context["source_type"], "pet_routine")
        self.assertIn("multiple days", friendship_context["source_detail"].lower())

    def test_service_bootstrap_covers_animal_shop_blacksmith_carpenter_and_shipping_bin(self) -> None:
        cases = {
            "purchase_1_chicken": ("marnie", "animal_shop"),
            "sell_1_chicken": ("marnie", "animal_shop"),
            "break_5_geode": ("clint", "blacksmith"),
            "build_1_big_coop": ("robin", "carpenter"),
            "move_1_coop": ("robin", "carpenter"),
            "upgrade_farmhouse": ("robin", "carpenter"),
            "demolish_1_shipping_bin": ("robin", "carpenter"),
            "upgrade_to_large_pack": ("pierre", "seed_shop"),
            "ship_1_parsnip_with_shipping_bin": ("shipping bin", "shipping_bin"),
            "forage_10_hay_with_scythe": ("grassy farm area", "grass_patch_or_farm_area"),
        }

        for task_name, (expected_phrase, expected_source_type) in cases.items():
            with self.subTest(task_name=task_name):
                subtask = build_initial_subtask(task_name)
                context = build_task_acquisition_context(task_name)
                self.assertIn(expected_phrase, subtask.lower())
                self.assertEqual(context["source_type"], expected_source_type)

    def test_break_geode_bootstrap_mentions_processing_station_not_counter_shop(self) -> None:
        subtask = build_initial_subtask("break_5_geode")

        self.assertIn("upper-right", subtask.lower())
        self.assertIn("furnace", subtask.lower())
        self.assertIn("anvil", subtask.lower())
        self.assertIn("not the ore shop counter", subtask.lower())

    def test_combat_bootstrap_mentions_weapon_search_and_attack_range(self) -> None:
        subtask = build_initial_subtask("kill_10_green_slime_with_rusty_sword")

        self.assertIn("rusty sword", subtask.lower())
        self.assertIn("search for", subtask.lower())
        self.assertIn("attack range", subtask.lower())

    def test_combat_acquisition_context_mentions_enemy_search(self) -> None:
        context = build_task_acquisition_context("kill_10_green_slime_with_rusty_sword")

        self.assertEqual(context["source_type"], "enemy_search")
        self.assertIn("visible target enemy", context["source_detail"].lower())

    def test_forage_acquisition_context_mentions_explicit_visibility(self) -> None:
        context = build_task_acquisition_context("forage_1_wild_horseradish")

        self.assertEqual(context["source_type"], "forage_search")
        self.assertIn("explicitly visible", context["source_detail"].lower())

    def test_relationship_bootstrap_uses_required_items(self) -> None:
        cases = {
            "date_abigail": "bouquet",
            "break_up_with_abigail": "wilted bouquet",
            "propose_to_abigail": "mermaid's pendant",
        }

        for task_name, expected_item in cases.items():
            with self.subTest(task_name=task_name):
                subtask = build_initial_subtask(task_name)
                context = build_task_acquisition_context(task_name)
                self.assertIn(expected_item, subtask.lower())
                self.assertIn(expected_item, context["target_item"].lower())
                self.assertIn("abigail", context["source_detail"].lower())

    def test_sell_and_shipping_bin_sources_are_distinct(self) -> None:
        store_context = build_task_acquisition_context("sell_1_parsnip_to_pierre")
        shipping_context = build_task_acquisition_context("ship_1_parsnip_with_shipping_bin")

        self.assertEqual(store_context["target_item"], "Parsnip")
        self.assertEqual(store_context["source_type"], "seller")
        self.assertEqual(shipping_context["target_item"], "Parsnip")
        self.assertEqual(shipping_context["source_type"], "shipping_bin")
        self.assertNotEqual(store_context["source_detail"], shipping_context["source_detail"])

    def test_sell_sell_animal_and_silo_always_populate_target_item(self) -> None:
        contexts = {
            "sell_1_parsnip_to_pierre": build_task_acquisition_context("sell_1_parsnip_to_pierre"),
            "sell_1_chicken": build_task_acquisition_context("sell_1_chicken"),
            "forage_10_hay_with_scythe": build_task_acquisition_context("forage_10_hay_with_scythe"),
        }

        self.assertEqual(contexts["sell_1_parsnip_to_pierre"]["target_item"], "Parsnip")
        self.assertEqual(contexts["sell_1_chicken"]["target_item"], "Chicken")
        self.assertEqual(contexts["forage_10_hay_with_scythe"]["target_item"], "Hay")
        self.assertEqual(contexts["sell_1_chicken"]["source_type"], "animal_shop")
        self.assertEqual(contexts["forage_10_hay_with_scythe"]["source_type"], "grass_patch_or_farm_area")

    def test_no_spec_parser_prefers_action_intent_over_route_prefix(self) -> None:
        cases = {
            "go to marnie and buy hay": {"evaluator": "purchase", "object": "Hay"},
            "go to robin and build a coop": {"evaluator": "build", "object": "Coop"},
            "return home and ship parsnip": {"evaluator": "sell", "object": "Parsnip", "tool": "Shipping Bin"},
            "buy a chicken from marnie": {"evaluator": "purchase_animal", "object": "Chicken"},
            "sell chicken to marnie": {"evaluator": "sell_animal", "object": "Chicken"},
            "break geode at clint": {"evaluator": "break", "object": "Geode"},
            "date abigail": {"evaluator": "date", "object": "Abigail", "tool": "Bouquet"},
            "break up with abigail": {"evaluator": "breakup", "object": "Abigail", "tool": "Wilted Bouquet"},
            "propose to abigail": {"evaluator": "propose", "object": "Abigail", "tool": "Mermaid's Pendant"},
            "go to abigail and propose": {"evaluator": "propose", "object": "Abigail", "tool": "Mermaid's Pendant"},
            "forage hay with scythe": {"evaluator": "silo", "object": "Hay", "tool": "Scythe"},
        }

        for task_text, expected_spec in cases.items():
            with self.subTest(task_text=task_text):
                actual_spec = _infer_task_spec_from_text(task_text)
                for key, expected_value in expected_spec.items():
                    self.assertEqual(actual_spec.get(key), expected_value)

    def test_no_spec_bootstrap_and_acquisition_cover_trade_hay_and_relationship_tasks(self) -> None:
        cases = {
            "go to marnie and buy hay": ("buy hay", "shop"),
            "go to robin and build a coop": ("robin", "carpenter"),
            "return home and ship parsnip": ("shipping bin", "shipping_bin"),
            "buy a chicken from marnie": ("marnie", "animal_shop"),
            "sell chicken to marnie": ("animal sale", "animal_shop"),
            "break geode at clint": ("clint", "blacksmith"),
            "date abigail": ("bouquet", "inventory_or_source"),
            "break up with abigail": ("wilted bouquet", "inventory_or_source"),
            "propose to abigail": ("mermaid's pendant", "inventory_or_source"),
            "forage hay with scythe": ("grassy farm area", "grass_patch_or_farm_area"),
        }

        for task_text, (expected_phrase, expected_source_type) in cases.items():
            with self.subTest(task_text=task_text):
                subtask = build_initial_subtask(task_text)
                context = build_task_acquisition_context(task_text)
                self.assertIn(expected_phrase, subtask.lower())
                self.assertEqual(context["source_type"], expected_source_type)


if __name__ == "__main__":
    unittest.main()
