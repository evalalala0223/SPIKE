from __future__ import annotations

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


def _install_go_through_door_stubs():
    stubs = {}

    class _FakeMSSContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def grab(self, _region):
            return types.SimpleNamespace(size=(1, 1), bgra=b"\x00" * 4)

    mss_mod = types.ModuleType("mss")
    mss_mod.mss = lambda: _FakeMSSContext()
    stubs["mss"] = mss_mod

    cv2_mod = types.ModuleType("cv2")
    cv2_mod.imread = lambda *_args, **_kwargs: None
    cv2_mod.resize = lambda img, *_args, **_kwargs: img
    stubs["cv2"] = cv2_mod

    numpy_mod = types.ModuleType("numpy")
    numpy_mod.sqrt = lambda value: value ** 0.5
    numpy_mod.random = types.SimpleNamespace(rand=lambda: 0.5)
    stubs["numpy"] = numpy_mod

    mtm_mod = types.ModuleType("MTM")
    mtm_mod.matchTemplates = lambda *_args, **_kwargs: None
    stubs["MTM"] = mtm_mod

    pil_mod = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.frombytes = lambda *_args, **_kwargs: types.SimpleNamespace(save=lambda *_a, **_k: None)
    image_mod.open = lambda *_args, **_kwargs: types.SimpleNamespace(save=lambda *_a, **_k: None)
    image_draw_mod = types.ModuleType("PIL.ImageDraw")
    image_draw_mod.Draw = lambda *_args, **_kwargs: types.SimpleNamespace(rectangle=lambda *_a, **_k: None)
    pil_mod.Image = image_mod
    pil_mod.ImageDraw = image_draw_mod
    stubs["PIL"] = pil_mod
    stubs["PIL.Image"] = image_mod
    stubs["PIL.ImageDraw"] = image_draw_mod

    config_mod = types.ModuleType("stardojo.config")

    class _FakeConfig:
        env_sub_path = "stardew"
        work_dir = "."
        env_region = (0, 0, 1, 1)
        resolution_ratio = 1
        ocr_different_previous_text = False
        ocr_enabled = False

    config_mod.Config = lambda: _FakeConfig()
    stubs["stardojo.config"] = config_mod

    log_mod = types.ModuleType("stardojo.log")

    class _FakeLogger:
        def write(self, *args, **kwargs):
            pass

        def warn(self, *args, **kwargs):
            pass

        def debug(self, *args, **kwargs):
            pass

    log_mod.Logger = lambda: _FakeLogger()
    stubs["stardojo.log"] = log_mod

    file_utils_mod = types.ModuleType("stardojo.utils.file_utils")
    file_utils_mod.assemble_project_path = lambda path: path
    stubs["stardojo.utils.file_utils"] = file_utils_mod

    basic_skills_mod = types.ModuleType("stardojo.environment.stardew.atomic_skills.basic_skills")
    basic_skills_mod.use_tool = lambda *_args, **_kwargs: None
    basic_skills_mod.do_action = lambda *_args, **_kwargs: None
    basic_skills_mod.move_up = lambda *_args, **_kwargs: None
    basic_skills_mod.move_down = lambda *_args, **_kwargs: None
    basic_skills_mod.move_left = lambda *_args, **_kwargs: None
    basic_skills_mod.move_right = lambda *_args, **_kwargs: None
    basic_skills_mod.select_tool = lambda *_args, **_kwargs: None
    basic_skills_mod.mouse_check_do_action = lambda *_args, **_kwargs: None
    stubs["stardojo.environment.stardew.atomic_skills.basic_skills"] = basic_skills_mod

    skill_registry_mod = types.ModuleType("stardojo.environment.stardew.skill_registry")
    skill_registry_mod.register_skill = lambda _name: (lambda func: func)
    stubs["stardojo.environment.stardew.skill_registry"] = skill_registry_mod

    go_home_mod = types.ModuleType("stardojo.environment.stardew.composite_skills.go_home")
    go_home_mod.go_home = lambda *_args, **_kwargs: None
    stubs["stardojo.environment.stardew.composite_skills.go_home"] = go_home_mod

    previous_modules = {name: sys.modules.get(name) for name in stubs}
    for name, module in stubs.items():
        sys.modules[name] = module
    return previous_modules


_PREVIOUS_MODULES = _install_go_through_door_stubs()
_MODULE_PATH = ROOT / "agent" / "stardojo" / "environment" / "stardew" / "composite_skills" / "go_through_door.py"
_SPEC = importlib.util.spec_from_file_location("_go_through_door_under_test", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)

for _name, _previous in _PREVIOUS_MODULES.items():
    if _previous is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _previous


class TestGoThroughDoor(unittest.TestCase):
    def test_missing_template_at_close_range_does_not_report_success(self) -> None:
        with mock.patch.object(_MODULE, "match_template", return_value=(None, None)), mock.patch.object(
            _MODULE,
            "do_action",
        ), mock.patch.object(_MODULE, "time") as mock_time:
            mock_time.sleep.return_value = None
            self.assertFalse(
                _MODULE.cv_go_to_icon(
                    iterations=1,
                    template_file="door.jpg",
                    terminal_threshold=95,
                    character=None,
                )
            )

    def test_terminal_interaction_requires_template_to_disappear(self) -> None:
        with mock.patch.object(
            _MODULE,
            "match_template",
            side_effect=[((0, 0), {}), ((0, 0), {})],
        ), mock.patch.object(_MODULE, "take_screenshot", side_effect=["before.jpg", "after.jpg"]), mock.patch.object(
            _MODULE,
            "do_action",
        ), mock.patch.object(_MODULE, "time") as mock_time:
            mock_time.sleep.return_value = None
            self.assertFalse(
                _MODULE.cv_go_to_icon(
                    iterations=1,
                    template_file="door.jpg",
                    terminal_threshold=95,
                    character=None,
                )
            )

    def test_terminal_interaction_reports_success_when_template_disappears(self) -> None:
        with mock.patch.object(
            _MODULE,
            "match_template",
            side_effect=[((0, 0), {}), (None, None)],
        ), mock.patch.object(_MODULE, "take_screenshot", side_effect=["before.jpg", "after.jpg"]), mock.patch.object(
            _MODULE,
            "do_action",
        ), mock.patch.object(_MODULE, "time") as mock_time:
            mock_time.sleep.return_value = None
            self.assertTrue(
                _MODULE.cv_go_to_icon(
                    iterations=1,
                    template_file="door.jpg",
                    terminal_threshold=95,
                    character=None,
                )
            )


if __name__ == "__main__":
    unittest.main()
