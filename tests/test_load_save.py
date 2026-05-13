from __future__ import annotations

import unittest
from unittest import mock

from env.tasks.utils import load_save


class TestLoadSave(unittest.TestCase):
    def test_load_save_waits_for_game_start_before_init_commands(self) -> None:
        proxy = mock.Mock()
        call_order: list[str] = []

        def record(name: str):
            def _inner(*args, **kwargs):
                call_order.append(name)
                return None

            return _inner

        proxy.port = 10783
        proxy.reset_pause_state.side_effect = record("reset_pause_state")
        proxy.load_game_record.side_effect = record("load_game_record")
        proxy.wait_game_start.side_effect = lambda: call_order.append("wait_game_start") or True
        proxy.warp_mine.side_effect = record("warp_mine")

        with (
            mock.patch.object(load_save, "copy_save_folder_as", return_value="save_instance"),
            mock.patch.object(load_save.time, "sleep", return_value=None),
        ):
            load_save.load_save(proxy, "save_new", ['warp_mine(2)'])

        self.assertEqual(
            call_order,
            ["reset_pause_state", "load_game_record", "wait_game_start", "warp_mine"],
        )

    def test_load_save_skips_inner_wait_without_init_commands(self) -> None:
        proxy = mock.Mock()
        proxy.port = 10783

        with (
            mock.patch.object(load_save, "copy_save_folder_as", return_value="save_instance"),
            mock.patch.object(load_save.time, "sleep", return_value=None),
        ):
            load_save.load_save(proxy, "save_new", None)

        proxy.wait_game_start.assert_not_called()


if __name__ == "__main__":
    unittest.main()
