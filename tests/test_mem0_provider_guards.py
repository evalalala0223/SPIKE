from __future__ import annotations

import unittest

from cradle.memory.mem0_provider import Mem0Provider


class TestMem0ProviderGuards(unittest.TestCase):
    def test_setup_only_actions_only_match_inventory_setup(self) -> None:
        self.assertTrue(
            Mem0Provider._are_setup_only_actions(["choose_item(slot_index=4)"])
        )
        self.assertFalse(
            Mem0Provider._are_setup_only_actions(['move(x=0, y=1)'])
        )

    def test_move_only_actions_are_tracked_separately(self) -> None:
        self.assertTrue(
            Mem0Provider._is_move_only_actions(['move(x=0, y=1)'])
        )
        self.assertFalse(
            Mem0Provider._is_move_only_actions(['use(direction="down")'])
        )

    def test_explicit_no_progress_text_rejects_false_positive_reasoning(self) -> None:
        self.assertTrue(
            Mem0Provider._is_explicit_no_progress_text(
                "The executed action is none and the task is not completed yet."
            )
        )
        provider = Mem0Provider(enabled=False)
        self.assertFalse(
            provider._is_meaningful_progress_text(
                "The executed action is none and the task is not completed yet."
            )
        )


if __name__ == "__main__":
    unittest.main()
