from __future__ import annotations

import unittest

from env.tasks.exploration import Exploration


class TestExplorationSleepEvaluator(unittest.TestCase):
    def _build_task(self) -> Exploration:
        return Exploration(
            llm_description="go_to_bed",
            object="Bed",
            quantity=1,
            tool=None,
            save="save_new",
            init_commands=None,
            evaluator="sleep",
            difficulty="easy",
        )

    def test_sleep_evaluator_completes_after_day_rollover(self) -> None:
        task = self._build_task()
        first_obs = {
            "player": {"location": "FarmHouse", "position": {"x": 10, "y": 10}},
            "callbackdata": {"ondaystarted": 1},
            "furnitures": [],
        }
        second_obs = {
            "player": {"location": "FarmHouse", "position": {"x": 10, "y": 10}},
            "callbackdata": {"ondaystarted": 2},
            "furnitures": [],
        }

        self.assertEqual(task.evaluate(first_obs, None), {"completed": False, "quantity": 0})
        self.assertEqual(task.evaluate(second_obs, None), {"completed": True, "quantity": 1})

    def test_sleep_evaluator_completes_when_player_is_on_bed(self) -> None:
        task = self._build_task()
        first_obs = {
            "player": {"location": "FarmHouse", "position": {"x": 10, "y": 10}},
            "callbackdata": {"ondaystarted": 1},
            "furnitures": [],
        }
        second_obs = {
            "player": {"location": "FarmHouse", "position": {"x": 12, "y": 14}},
            "callbackdata": {"ondaystarted": 1},
            "furnitures": [
                {
                    "name": "Bed",
                    "position": {"x": 12, "y": 14},
                    "boundingbox": {"left": 11, "right": 14, "top": 13, "bottom": 15},
                }
            ],
        }

        self.assertEqual(task.evaluate(first_obs, None), {"completed": False, "quantity": 0})
        self.assertEqual(task.evaluate(second_obs, None), {"completed": True, "quantity": 1})

    def test_sleep_evaluator_does_not_complete_when_near_bed_only(self) -> None:
        task = self._build_task()
        first_obs = {
            "player": {"location": "FarmHouse", "position": {"x": 10, "y": 10}},
            "callbackdata": {"ondaystarted": 1},
            "furnitures": [],
        }
        second_obs = {
            "player": {"location": "FarmHouse", "position": {"x": 8, "y": 14}},
            "callbackdata": {"ondaystarted": 1},
            "furnitures": [
                {
                    "name": "Bed",
                    "position": {"x": 12, "y": 14},
                    "boundingbox": {"left": 11, "right": 14, "top": 13, "bottom": 15},
                }
            ],
        }

        self.assertEqual(task.evaluate(first_obs, None), {"completed": False, "quantity": 0})
        self.assertEqual(task.evaluate(second_obs, None), {"completed": False, "quantity": 0})


if __name__ == "__main__":
    unittest.main()
