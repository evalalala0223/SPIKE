from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


class TestEnvConfigPathGuards(unittest.TestCase):
    def test_load_env_config_accepts_agent_prefixed_path(self) -> None:
        from stardojo.config import Config

        fake_gameio = types.ModuleType("stardojo.gameio")
        fake_gui_utils = types.ModuleType("stardojo.gameio.gui_utils")
        fake_gui_utils.get_named_windows = lambda *args, **kwargs: []
        fake_gui_utils.get_named_windows_fallback = lambda *args, **kwargs: []
        fake_gui_utils.get_screen_size = lambda: (1920, 1080)
        fake_gui_utils.mouse_button_down = lambda *args, **kwargs: None
        fake_gui_utils.mouse_button_up = lambda *args, **kwargs: None
        fake_gui_utils.key_down = lambda *args, **kwargs: None
        fake_gui_utils.key_up = lambda *args, **kwargs: None
        fake_gui_utils.mouse_wheel_scroll = lambda *args, **kwargs: None
        fake_gui_utils.type_keys = lambda *args, **kwargs: None
        fake_gui_utils.mouse_click = lambda *args, **kwargs: None
        fake_gui_utils.get_mouse_location = lambda *args, **kwargs: (0, 0)
        fake_gui_utils.mouse_move_to = lambda *args, **kwargs: None
        fake_gameio.gui_utils = fake_gui_utils
        sys.modules["stardojo.gameio"] = fake_gameio
        sys.modules["stardojo.gameio.gui_utils"] = fake_gui_utils

        config = Config()
        config.load_env_config("agent/conf/env_config_stardew.json")

        self.assertEqual(config.env_short_name, "stardew")
        self.assertEqual(config.env_name, "Stardew Valley")


if __name__ == "__main__":
    unittest.main()
