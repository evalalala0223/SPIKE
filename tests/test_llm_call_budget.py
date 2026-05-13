from __future__ import annotations

import importlib
import unittest

from cradle.utils.llm_call_budget import (
    get_llm_call_breakdown,
    get_llm_call_count,
    increment_llm_call_counter,
    reset_llm_call_counter,
)


class TestLLMCallBudget(unittest.TestCase):
    def setUp(self) -> None:
        reset_llm_call_counter()

    def test_dual_import_paths_share_same_module_and_counter(self) -> None:
        canonical_module = importlib.import_module("cradle.utils.llm_call_budget")
        legacy_module = importlib.import_module("agent.cradle.utils.llm_call_budget")

        self.assertIs(canonical_module, legacy_module)

        canonical_module.reset_llm_call_counter()
        canonical_module.increment_llm_call_counter("little_brain")

        self.assertEqual(legacy_module.get_llm_call_count(), 1)
        self.assertEqual(
            legacy_module.get_llm_call_breakdown(),
            {"little_brain": 1},
        )

    def test_counter_resets_cleanly(self) -> None:
        increment_llm_call_counter("little_brain")
        increment_llm_call_counter("big_brain:action_planning")

        reset_llm_call_counter()

        self.assertEqual(get_llm_call_count(), 0)
        self.assertEqual(get_llm_call_breakdown(), {})

    def test_counter_tracks_total_and_breakdown(self) -> None:
        increment_llm_call_counter("little_brain")
        increment_llm_call_counter("big_brain:action_planning")
        increment_llm_call_counter("big_brain:action_planning")

        self.assertEqual(get_llm_call_count(), 3)
        self.assertEqual(
            get_llm_call_breakdown(),
            {
                "little_brain": 1,
                "big_brain:action_planning": 2,
            },
        )


if __name__ == "__main__":
    unittest.main()
