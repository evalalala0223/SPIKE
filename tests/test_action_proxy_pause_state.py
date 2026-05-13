from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from env.actions import ActionProxy


class TestActionProxyPauseState(unittest.TestCase):
    def test_pause_game_short_circuits_when_already_paused(self) -> None:
        proxy = ActionProxy(port=10783)
        with (
            mock.patch.object(proxy, "is_paused", return_value=True),
            mock.patch.object(proxy, "_post_message") as post_message,
        ):
            self.assertTrue(proxy.pause_game())
        post_message.assert_not_called()

    def test_resume_game_short_circuits_when_already_resumed(self) -> None:
        proxy = ActionProxy(port=10783)
        with (
            mock.patch.object(proxy, "is_paused", return_value=False),
            mock.patch.object(proxy, "_post_message") as post_message,
        ):
            self.assertTrue(proxy.resume_game())
        post_message.assert_not_called()

    def test_wait_game_start_accepts_day_started_callback_without_valid_obs(self) -> None:
        proxy = ActionProxy(port=10783)
        callback_only_obs = '{"CallBackData": {"OnDayStarted": 1}}'
        with (
            mock.patch.object(proxy, "_post_message", return_value=callback_only_obs) as post_message,
            mock.patch("env.actions.time.sleep", return_value=None),
        ):
            self.assertTrue(proxy.wait_game_start())
        post_message.assert_called()

    def test_wait_game_start_keeps_valid_obs_fast_path(self) -> None:
        proxy = ActionProxy(port=10783)
        valid_obs = '{"Player": {"Position": {"X": 1, "Y": 2}}}'
        with (
            mock.patch.object(proxy, "_post_message", return_value=valid_obs) as post_message,
            mock.patch("env.actions.time.sleep", return_value=None),
        ):
            self.assertTrue(proxy.wait_game_start())
        post_message.assert_called()


if __name__ == "__main__":
    unittest.main()
