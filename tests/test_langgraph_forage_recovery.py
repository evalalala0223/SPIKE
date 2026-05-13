import os
import sys
import unittest


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "agent"))

from cradle.runner.langgraph_nodes import LangGraphNodes


def _make_nodes():
    return LangGraphNodes.__new__(LangGraphNodes)


class TestLangGraphForageRecovery(unittest.TestCase):
    def test_forage_nearby_target_ignores_image_description_false_positive(self):
        nodes = _make_nodes()
        state = {
            "main_task": "forage_1_clam",
            "gathered_info": {
                "image_description": "The player is holding a clam near the beach.",
                "description": "Holding a clam.",
                "surroundings": "[1, 0]: sand\n[0, 1]: driftwood\n[-1, 0]: open ground",
            },
        }

        self.assertFalse(nodes._current_task_has_nearby_target("forage_1_clam", state))

    def test_forage_recovery_builds_local_pickup_subtask(self):
        nodes = _make_nodes()
        state = {
            "main_task": "forage_1_clam",
            "gathered_info": {
                "surroundings": "[1, 0]: Clam\n[0, 1]: sand",
                "front_tile_summary": "Front tile (1, 0) toward right: Clam.",
                "nearest_grounded_target_summary": "Nearest grounded interaction target: Clam at (1, 0).",
            },
        }

        subtask, reasoning = nodes._build_current_fact_recovery_subtask("forage_1_clam", state)

        self.assertIn("clam", subtask.lower())
        self.assertIn("interact", subtask.lower())
        self.assertNotIn("search", subtask.lower())
        self.assertIn("ground", reasoning.lower())

    def test_forage_far_grounded_summary_does_not_count_as_nearby_target(self):
        nodes = _make_nodes()
        state = {
            "main_task": "forage_1_clam",
            "gathered_info": {
                "surroundings": "[1, 0]: sand\n[0, 1]: driftwood",
                "nearest_grounded_target_summary": "Nearest grounded interaction target: Clam at (5, 0).",
            },
        }

        self.assertFalse(nodes._current_task_has_nearby_target("forage_1_clam", state))
        subtask, reasoning = nodes._build_current_fact_recovery_subtask("forage_1_clam", state)
        self.assertEqual(subtask, "")
        self.assertEqual(reasoning, "")

    def test_completion_claim_guard_requires_actual_progress(self):
        nodes = _make_nodes()

        no_progress_state = {
            "latest_task_eval": {"completed": False},
            "task_progress_quantity": 0,
            "previous_task_progress_quantity": 0,
        }
        progress_state = {
            "latest_task_eval": {"completed": False},
            "task_progress_quantity": 1,
            "previous_task_progress_quantity": 0,
        }

        self.assertTrue(
            nodes._subtask_claims_completion_without_progress(
                "The task is completed.",
                no_progress_state,
            )
        )
        self.assertFalse(
            nodes._subtask_claims_completion_without_progress(
                "The task is completed.",
                progress_state,
            )
        )


if __name__ == "__main__":
    unittest.main()
