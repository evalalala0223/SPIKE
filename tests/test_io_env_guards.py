from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


def _install_gui_stubs() -> None:
    class _DummyWindow:
        left = 0
        top = 0
        width = 1920
        height = 1080

    class _DummyPyAutoGUI(types.SimpleNamespace):
        def __getattr__(self, _name: str):
            return lambda *args, **kwargs: None

    pyautogui_stub = _DummyPyAutoGUI(
        keyDown=lambda *args, **kwargs: None,
        keyUp=lambda *args, **kwargs: None,
        mouseDown=lambda *args, **kwargs: None,
        mouseUp=lambda *args, **kwargs: None,
        click=lambda *args, **kwargs: None,
        move=lambda *args, **kwargs: None,
        moveTo=lambda *args, **kwargs: None,
        moveRel=lambda *args, **kwargs: None,
        scroll=lambda *args, **kwargs: None,
        typewrite=lambda *args, **kwargs: None,
        size=lambda: (1920, 1080),
        position=lambda: types.SimpleNamespace(x=0, y=0),
        getActiveWindow=lambda: _DummyWindow(),
        getWindowsWithTitle=lambda _title: [],
    )
    sys.modules.setdefault("pyautogui", pyautogui_stub)

    pydirectinput_stub = types.SimpleNamespace(
        FAILSAFE=False,
        keyDown=lambda *args, **kwargs: None,
        keyUp=lambda *args, **kwargs: None,
    )
    sys.modules.setdefault("pydirectinput", pydirectinput_stub)

    class _DummyAHK:
        def get_mouse_position(self):
            return (0, 0)

    sys.modules.setdefault("ahk", types.SimpleNamespace(AHK=_DummyAHK))


_install_gui_stubs()


class TestIOEnvironmentGuards(unittest.TestCase):
    MODULE_SPECS = (
        ("stardojo.gameio.io_env", ROOT / "agent" / "stardojo" / "gameio" / "io_env.py", "stardojo.gameio"),
        ("cradle.gameio.io_env", ROOT / "agent" / "cradle" / "gameio" / "io_env.py", "cradle.gameio"),
    )

    def _reset_io_state(self, io_env: object) -> None:
        io_env.held_keys = []
        io_env.held_buttons = []
        io_env.backup_held_keys = []
        io_env.backup_held_buttons = []

    def _reset_module_config(self, module: object) -> None:
        module.config.is_game = False
        module.config.env_name = "-"
        module.config.win_name_pattern = ""
        module.config.env_window = None

    def _load_io_module(self, module_name: str, module_path: Path, package_name: str):
        fake_package = types.ModuleType(package_name)
        fake_package.__path__ = [str(module_path.parent)]

        fake_gui_utils = types.ModuleType(f"{package_name}.gui_utils")
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
        fake_gui_utils.get_active_window = lambda: None

        sys.modules[package_name] = fake_package
        sys.modules[f"{package_name}.gui_utils"] = fake_gui_utils
        setattr(fake_package, "gui_utils", fake_gui_utils)

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load module spec for {module_name}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def test_pop_held_keys_removes_matching_key_not_stack_top(self) -> None:
        for module_name, module_path, package_name in self.MODULE_SPECS:
            with self.subTest(module=module_name):
                module = self._load_io_module(module_name, module_path, package_name)
                self._reset_module_config(module)
                io_env = module.IOEnvironment()
                self._reset_io_state(io_env)
                io_env.held_keys = [
                    {io_env.KEY_KEY: "w", io_env.EXPIRATION_KEY: 3},
                    {io_env.KEY_KEY: "e", io_env.EXPIRATION_KEY: 3},
                ]

                with mock.patch.object(module, "key_up") as key_up_mock:
                    with mock.patch.object(module.time, "sleep", return_value=None):
                        io_env.pop_held_keys("w")

                self.assertEqual(
                    [entry[io_env.KEY_KEY] for entry in io_env.held_keys],
                    ["e"],
                )
                key_up_mock.assert_called_once_with("w")
                self._reset_io_state(io_env)

    def test_key_press_reactivates_game_window_before_input_when_focus_is_lost(self) -> None:
        for module_name, module_path, package_name in self.MODULE_SPECS:
            with self.subTest(module=module_name):
                module = self._load_io_module(module_name, module_path, package_name)
                self._reset_module_config(module)
                io_env = module.IOEnvironment()
                self._reset_io_state(io_env)

                events: list[object] = []
                fake_window = types.SimpleNamespace(
                    is_active=lambda: False,
                    activate=lambda: events.append("activate"),
                )
                module.config.is_game = True
                module.config.env_name = "Stardew Valley"
                module.config.env_window = fake_window

                with mock.patch.object(module, "key_down", side_effect=lambda key: events.append(("down", key))):
                    with mock.patch.object(module, "key_up", side_effect=lambda key: events.append(("up", key))):
                        with mock.patch.object(module.time, "sleep", return_value=None):
                            io_env.key_press("e")

                self.assertEqual(events[0], "activate")
                self.assertIn(("down", "e"), events)
                self.assertIn(("up", "e"), events)
                self._reset_io_state(io_env)

    def test_key_press_releases_key_when_sleep_raises(self) -> None:
        for module_name, module_path, package_name in self.MODULE_SPECS:
            with self.subTest(module=module_name):
                module = self._load_io_module(module_name, module_path, package_name)
                self._reset_module_config(module)
                io_env = module.IOEnvironment()
                self._reset_io_state(io_env)

                with mock.patch.object(module, "key_down") as key_down_mock:
                    with mock.patch.object(module, "key_up") as key_up_mock:
                        with mock.patch.object(module.time, "sleep", side_effect=RuntimeError("boom")):
                            with self.assertRaises(RuntimeError):
                                io_env.key_press("e")

                key_down_mock.assert_called_once_with("e")
                key_up_mock.assert_called_once_with("e")
                self._reset_io_state(io_env)

    def test_key_press_returns_false_when_focus_guard_blocks_input(self) -> None:
        for module_name, module_path, package_name in self.MODULE_SPECS:
            with self.subTest(module=module_name):
                module = self._load_io_module(module_name, module_path, package_name)
                self._reset_module_config(module)
                io_env = module.IOEnvironment()
                self._reset_io_state(io_env)
                module.config.is_game = True
                module.config.env_window = None

                with mock.patch.object(module, "key_down") as key_down_mock:
                    with mock.patch.object(module, "key_up") as key_up_mock:
                        result = io_env.key_press("e")

                self.assertFalse(result)
                key_down_mock.assert_not_called()
                key_up_mock.assert_not_called()
                self._reset_io_state(io_env)

    def test_resolve_env_window_uses_matching_active_window_fallback(self) -> None:
        for module_name, module_path, package_name in self.MODULE_SPECS:
            with self.subTest(module=module_name):
                module = self._load_io_module(module_name, module_path, package_name)
                self._reset_module_config(module)
                io_env = module.IOEnvironment()
                self._reset_io_state(io_env)
                module.config.env_name = "Stardew Valley"
                module.config.win_name_pattern = ""

                active_window = types.SimpleNamespace(
                    title="Stardew Valley",
                    activate=lambda: None,
                    is_active=lambda: True,
                )

                with mock.patch.object(io_env, "get_windows_by_config", return_value=[]):
                    with mock.patch.object(io_env, "get_active_window", return_value=active_window):
                        resolved = io_env._resolve_env_window()

                self.assertIs(resolved, active_window)
                self.assertIs(module.config.env_window, active_window)
                self._reset_io_state(io_env)

    def test_key_hold_with_duration_releases_key_when_sleep_raises(self) -> None:
        for module_name, module_path, package_name in self.MODULE_SPECS:
            with self.subTest(module=module_name):
                module = self._load_io_module(module_name, module_path, package_name)
                self._reset_module_config(module)
                io_env = module.IOEnvironment()
                self._reset_io_state(io_env)

                with mock.patch.object(module, "key_down") as key_down_mock:
                    with mock.patch.object(module, "key_up") as key_up_mock:
                        with mock.patch.object(module.time, "sleep", side_effect=RuntimeError("boom")):
                            with self.assertRaises(RuntimeError):
                                io_env.key_hold("e", duration=0.5)

                key_down_mock.assert_called_once_with("e")
                key_up_mock.assert_called_once_with("e")
                self._reset_io_state(io_env)

    def test_mouse_click_button_releases_button_when_sleep_raises(self) -> None:
        for module_name, module_path, package_name in self.MODULE_SPECS:
            with self.subTest(module=module_name):
                module = self._load_io_module(module_name, module_path, package_name)
                self._reset_module_config(module)
                io_env = module.IOEnvironment()
                self._reset_io_state(io_env)

                with mock.patch.object(io_env, "_mouse_button_down") as down_mock:
                    with mock.patch.object(io_env, "_mouse_button_up") as up_mock:
                        with mock.patch.object(module.time, "sleep", side_effect=RuntimeError("boom")):
                            with self.assertRaises(RuntimeError):
                                io_env.mouse_click_button("left", duration=0.5)

                down_mock.assert_called_once_with(io_env.LEFT_MOUSE_BUTTON)
                up_mock.assert_called_once_with(io_env.LEFT_MOUSE_BUTTON)
                self._reset_io_state(io_env)


if __name__ == "__main__":
    unittest.main()
