from __future__ import annotations

import importlib.util
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


class TestGuiUtilsGuards(unittest.TestCase):
    def _load_gui_utils_with_failing_ahk(self):
        module_name = "test_stardojo_gameio_gui_utils"
        module_path = ROOT / "agent" / "stardojo" / "gameio" / "gui_utils.py"

        class _DummyPyAutoGUI(types.SimpleNamespace):
            def __getattr__(self, _name: str):
                return lambda *args, **kwargs: None

        class _FailingAHK:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("AHK constructor should not run at import time")

        sys.modules["pyautogui"] = _DummyPyAutoGUI(
            size=lambda: (1920, 1080),
            position=lambda: types.SimpleNamespace(x=0, y=0),
            getWindowsWithTitle=lambda _title: [],
            getActiveWindow=lambda: None,
            mouseDown=lambda *args, **kwargs: None,
            mouseUp=lambda *args, **kwargs: None,
            move=lambda *args, **kwargs: None,
            moveTo=lambda *args, **kwargs: None,
            moveRel=lambda *args, **kwargs: None,
            keyDown=lambda *args, **kwargs: None,
            keyUp=lambda *args, **kwargs: None,
            typewrite=lambda *args, **kwargs: None,
            scroll=lambda *args, **kwargs: None,
        )
        sys.modules["pydirectinput"] = types.SimpleNamespace(
            FAILSAFE=False,
            keyDown=lambda *args, **kwargs: None,
            keyUp=lambda *args, **kwargs: None,
        )
        sys.modules["ahk"] = types.SimpleNamespace(AHK=_FailingAHK)

        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            raise RuntimeError("Could not build module spec for gui_utils")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        return module

    def test_gui_utils_import_does_not_instantiate_ahk(self) -> None:
        module = self._load_gui_utils_with_failing_ahk()

        # Import should succeed because AHK is now lazy.
        self.assertTrue(hasattr(module, "get_ahk"))
        self.assertEqual(module.get_screen_size(), (1920, 1080))

        # The failure should only surface when AHK-backed functionality is used.
        with self.assertRaises(RuntimeError):
            module.get_ahk()


if __name__ == "__main__":
    unittest.main()
