"""Verify GameManager._pick_recovery_action logic using the REAL production code.

Previously this file mirrored the function; that made it possible for production
and tests to diverge. We now bind to the real GameManager._pick_recovery_action.
"""
import sys
import types
import unittest
from typing import Optional
from unittest.mock import MagicMock


def _install_fake_stardojo_modules():
    """Install lightweight stand-ins so importing game_manager does not pull the
    full cradle/agent runtime (which would require models, configs, etc.).
    """
    stubs = {}

    # stardojo.constants
    constants_mod = types.ModuleType("stardojo.constants")
    constants_mod.EXECUTED_SKILLS = "executed_skills"
    constants_mod.LAST_SKILL = "last_skill"
    constants_mod.ERRORS = "errors"
    constants_mod.ERRORS_INFO = "errors_info"
    constants_mod.INVALID_BBOX = "invalid_bbox"
    constants_mod.GENERAL_GAME_INTERFACE = "general_game_interface"
    constants_mod.PAUSE_INTERFACE = "pause_interface"
    constants_mod.SKILL_CONFIG_ALLOW_GENERATED_REGISTRATION = "allow_generated_registration"
    stubs["stardojo.constants"] = constants_mod

    # stardojo.config
    config_mod = types.ModuleType("stardojo.config")
    class _FakeConfig:
        env_name = "stardew"
        ide_name = ""
        ocr_check_composite_skill_names = set()
        ocr_fully_ban = True
        ocr_different_previous_text = False
        enable_ocr = False
        is_game = False
        composite_terminal_skill_names = set()
        composite_skill_wait_seconds = {}
        skill_configs = {}
        work_dir = "."
        max_recent_steps = 5
    config_mod.Config = lambda: _FakeConfig()
    stubs["stardojo.config"] = config_mod

    # stardojo.environment.ui_control (type stub only)
    ui_control_mod = types.ModuleType("stardojo.environment.ui_control")
    class _UIControl: ...
    ui_control_mod.UIControl = _UIControl
    stubs["stardojo.environment.ui_control"] = ui_control_mod

    # stardojo.log
    log_mod = types.ModuleType("stardojo.log")
    class _FakeLogger:
        def write(self, *a, **k): pass
        def warn(self, *a, **k): pass
        def error(self, *a, **k): pass
    log_mod.Logger = lambda: _FakeLogger()
    stubs["stardojo.log"] = log_mod

    # stardojo.gameio
    gameio_pkg = types.ModuleType("stardojo.gameio")
    class _FakeIOEnv:
        def update_timeouts(self): pass
        def release_held_keys(self): pass
        def release_held_buttons(self): pass
        def get_mouse_position(self, absolute=False): return (0, 0)
        def list_session_screenshots(self, session_dir): return []
        llm_provider = None
    _io_env_singleton = _FakeIOEnv()
    gameio_pkg.IOEnvironment = lambda: _io_env_singleton
    stubs["stardojo.gameio"] = gameio_pkg

    # stardojo.gameio.lifecycle.ui_control
    lifecycle_pkg = types.ModuleType("stardojo.gameio.lifecycle")
    lifecycle_ui_control = types.ModuleType("stardojo.gameio.lifecycle.ui_control")
    lifecycle_ui_control.check_active_window = lambda: True
    stubs["stardojo.gameio.lifecycle"] = lifecycle_pkg
    stubs["stardojo.gameio.lifecycle.ui_control"] = lifecycle_ui_control

    # stardojo.utils.file_utils
    utils_pkg = types.ModuleType("stardojo.utils")
    file_utils_mod = types.ModuleType("stardojo.utils.file_utils")
    file_utils_mod.assemble_project_path = lambda p: p
    stubs["stardojo.utils"] = utils_pkg
    stubs["stardojo.utils.file_utils"] = file_utils_mod

    # Root stardojo package
    stardojo_pkg = types.ModuleType("stardojo")
    stubs["stardojo"] = stardojo_pkg

    previous_modules = {name: sys.modules.get(name) for name in stubs}
    for name, mod in stubs.items():
        sys.modules[name] = mod
    return previous_modules


_PREVIOUS_MODULES = _install_fake_stardojo_modules()

# Now we can import the real production GameManager
import importlib.util
import os

_GAME_MANAGER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "agent", "stardojo", "gameio", "game_manager.py",
)
_spec = importlib.util.spec_from_file_location("_gm_under_test", _GAME_MANAGER_PATH)
_gm_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_gm_module)
GameManager = _gm_module.GameManager

for _name, _previous in _PREVIOUS_MODULES.items():
    if _previous is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _previous


def _make_gm():
    """Build a minimally-initialized GameManager instance (no ui_control)."""
    gm = GameManager.__new__(GameManager)
    gm.default_executer = None
    gm._last_problem_action_text = None
    gm._consecutive_same_action_count = 0
    gm.SAME_ACTION_REFUSE_THRESHOLD = 3
    gm._consecutive_failed_move_axis_count = 0
    gm._last_failed_move_axis_direction = None
    gm._recovery_inventory = None
    gm._recovery_task_description = None
    gm._recovery_variation_toggle = 0
    return gm


class _FakeSkillRegistry:
    def __init__(self) -> None:
        self.executed: list[tuple[str, dict, object]] = []
        self.move_should_fail = False

    def convert_expression_to_skill(self, expression: str):
        expression = str(expression or "").strip()
        if expression.startswith("move("):
            import re

            match = re.match(r'move\(x=(-?\d+), y=(-?\d+)\)', expression)
            return "move", {"x": int(match.group(1)), "y": int(match.group(2)), "other_params": []}
        return "nop", {"other_params": []}

    def execute_skill(self, executer, skill_name: str = "move", skill_params: Optional[dict] = None):
        params = dict(skill_params or {})
        self.executed.append((skill_name, params, executer))
        if skill_name == "move":
            return not self.move_should_fail
        return True

    def execute_nop_skill(self):
        self.executed.append(("nop", {}, None))


class _BusyTimeoutMoveRegistry(_FakeSkillRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.move_busy_timeout = False

    def execute_skill(self, executer, skill_name: str = "move", skill_params: Optional[dict] = None):
        params = dict(skill_params or {})
        self.executed.append((skill_name, params, executer))
        if skill_name == "move":
            if self.move_busy_timeout:
                return "busy_timeout:move"
            return not self.move_should_fail
        return True


class _CraftTimeoutRegistry:
    def __init__(self) -> None:
        self.executed: list[tuple[str, dict, object]] = []

    def convert_expression_to_skill(self, expression: str):
        expression = str(expression or "").strip()
        if expression.startswith("craft("):
            return "craft", {"item": "Torch", "other_params": []}
        return "nop", {"other_params": []}

    def execute_skill(self, executer, skill_name: str = "craft", skill_params: Optional[dict] = None):
        params = dict(skill_params or {})
        self.executed.append((skill_name, params, executer))
        if skill_name == "craft":
            return "busy_timeout:craft"
        return True

    def execute_nop_skill(self):
        self.executed.append(("nop", {}, None))


class TestCircuitBreakerRecovery(unittest.TestCase):
    """Exercise the real GameManager._pick_recovery_action."""

    def test_menu_open_inventory_without_context_closes(self):
        """With no recovery context, menu-open loop falls back to menu-close."""
        gm = _make_gm()
        self.assertEqual(
            gm._pick_recovery_action('menu(option="open", menu_name="inventory")'),
            'menu(option="close", menu_name="inventory")',
        )

    def test_menu_open_crafting_without_context_closes(self):
        gm = _make_gm()
        self.assertEqual(
            gm._pick_recovery_action('menu(option="open", menu_name="crafting")'),
            'menu(option="close", menu_name="crafting")',
        )

    def test_menu_open_with_matching_inventory_still_closes(self):
        """Rollback behavior: menu-open recovery always closes the menu."""
        gm = _make_gm()
        gm.set_recovery_context(
            inventory=[
                "slot_index 0: Axe (quantity: 1)",
                "slot_index 1: Hoe (quantity: 1)",
                "slot_index 6: Speed-Gro (quantity: 1)",
                "slot_index 7: No item",
            ],
            task_description="fertilize_1_dirt_with_speed_gro",
        )
        result = gm._pick_recovery_action('menu(option="open", menu_name="inventory")')
        self.assertEqual(result, 'menu(option="close", menu_name="inventory")')

    def test_menu_open_with_matching_dict_inventory_still_closes(self):
        """Rollback behavior also applies to dict-format inventory."""
        gm = _make_gm()
        gm.set_recovery_context(
            inventory=[
                {"Name": "Axe", "Quantity": 1},
                {"Name": "Hoe", "Quantity": 1},
                {"Name": "Speed-Gro", "Quantity": 2},
            ],
            task_description="fertilize_5_dirt_with_speed_gro",
        )
        result = gm._pick_recovery_action('menu(option="open", menu_name="inventory")')
        self.assertEqual(result, 'menu(option="close", menu_name="inventory")')

    def test_menu_open_with_no_matching_item_falls_back_to_close(self):
        """If the task item is not in the inventory, fall back to menu(close)."""
        gm = _make_gm()
        gm.set_recovery_context(
            inventory=[
                "slot_index 0: Axe (quantity: 1)",
                "slot_index 1: Hoe (quantity: 1)",
            ],
            task_description="fertilize_1_dirt_with_speed_gro",
        )
        result = gm._pick_recovery_action('menu(option="open", menu_name="inventory")')
        self.assertEqual(result, 'menu(option="close", menu_name="inventory")')

    def test_use_down_recovers_to_move_down(self):
        gm = _make_gm()
        self.assertEqual(
            gm._pick_recovery_action('use(direction="down")'),
            'move(x=1, y=0)',
        )

    def test_interact_right_recovers_to_move_right(self):
        gm = _make_gm()
        self.assertEqual(
            gm._pick_recovery_action('interact(direction="right")'),
            'move(x=0, y=1)',
        )

    def test_interact_up_recovers_to_move_up(self):
        gm = _make_gm()
        self.assertEqual(
            gm._pick_recovery_action('interact(direction="up")'),
            'move(x=1, y=0)',
        )

    def test_choose_item_recovers_to_move(self):
        gm = _make_gm()
        self.assertEqual(
            gm._pick_recovery_action('choose_item(slot_index=5)'),
            'move(x=1, y=0)',
        )

    def test_stuck_x_move_recovers_to_y_move(self):
        gm = _make_gm()
        self.assertEqual(
            gm._pick_recovery_action('move(x=5, y=0)'),
            'move(x=0, y=1)',
        )

    def test_stuck_y_move_recovers_to_x_move(self):
        gm = _make_gm()
        self.assertEqual(
            gm._pick_recovery_action('move(x=0, y=-3)'),
            'move(x=1, y=0)',
        )

    def test_repeated_move_recovery_alternates_sidestep_direction(self):
        gm = _make_gm()
        self.assertEqual(gm._pick_recovery_action('move(x=5, y=0)'), 'move(x=0, y=1)')
        self.assertEqual(gm._pick_recovery_action('move(x=5, y=0)'), 'move(x=0, y=-1)')

    def test_unknown_action_falls_back_to_move(self):
        gm = _make_gm()
        self.assertEqual(
            gm._pick_recovery_action('craft(item="Torch")'),
            'move(x=1, y=0)',
        )

    def test_find_target_item_slot_parses_string_inventory(self):
        gm = _make_gm()
        gm.set_recovery_context(
            inventory=[
                "slot_index 0: Axe (quantity: 1)",
                "slot_index 6: Speed-Gro (quantity: 1)",
            ],
            task_description="fertilize_1_dirt_with_speed_gro",
        )
        self.assertEqual(gm._find_target_item_slot(), 6)

    def test_find_target_item_slot_no_task_returns_none(self):
        gm = _make_gm()
        gm.set_recovery_context(
            inventory=["slot_index 0: Axe (quantity: 1)"],
            task_description="",
        )
        self.assertIsNone(gm._find_target_item_slot())

    def test_find_target_item_slot_empty_inventory_returns_none(self):
        gm = _make_gm()
        gm.set_recovery_context(
            inventory=[],
            task_description="fertilize_1_dirt_with_speed_gro",
        )
        self.assertIsNone(gm._find_target_item_slot())

    def test_find_target_item_slot_non_matching_task_returns_none(self):
        """Tasks not matching the fertilize/sow pattern should return None."""
        gm = _make_gm()
        gm.set_recovery_context(
            inventory=["slot_index 0: Axe (quantity: 1)"],
            task_description="clear_10_weeds_with_scythe",
        )
        self.assertIsNone(gm._find_target_item_slot())

    def test_repeated_successful_moves_do_not_trigger_axis_breaker(self):
        gm = _make_gm()
        gm.skill_registry = _FakeSkillRegistry()
        gm.post_action_wait = lambda *_: None

        results = [gm.execute_actions(['move(x=1, y=0)']) for _ in range(4)]

        self.assertTrue(all(str(result["last_skill"]).startswith('move(x=1, y=0)') for result in results))
        self.assertTrue(all("AXIS-CIRCUIT-BREAKER" not in result["errors_info"] for result in results))

    def test_repeated_blocked_moves_trigger_axis_breaker_on_next_attempt(self):
        gm = _make_gm()
        registry = _FakeSkillRegistry()
        registry.move_should_fail = True
        gm.skill_registry = registry
        gm.post_action_wait = lambda *_: None

        for _ in range(3):
            result = gm.execute_actions(['move(x=1, y=0)'])
            self.assertIn("path is likely blocked", result["errors_info"])

        registry.move_should_fail = False
        blocked = gm.execute_actions(['move(x=2, y=0)'])

        self.assertIn("AXIS-CIRCUIT-BREAKER", blocked["errors_info"])
        self.assertEqual(blocked.get("refusal_type"), "axis_circuit_breaker")
        self.assertEqual(blocked.get("refused_action"), 'move(x=2, y=0)')
        self.assertEqual(blocked["last_skill"], 'move(x=0, y=1)')
        self.assertIn('move(x=0, y=1) # axis-breaker', blocked["executed_skills"])

    def test_axis_breaker_failed_sidestep_falls_back_to_nop(self):
        gm = _make_gm()
        registry = _FakeSkillRegistry()
        registry.move_should_fail = True
        gm.skill_registry = registry
        gm.post_action_wait = lambda *_: None

        for _ in range(3):
            gm.execute_actions(['move(x=1, y=0)'])

        blocked = gm.execute_actions(['move(x=2, y=0)'])

        self.assertEqual(blocked["last_skill"], "nop()")
        self.assertIn("nop() # axis-breaker-failed", blocked["executed_skills"])
        self.assertIn("Recovery move failed", blocked["errors_info"])

    def test_repeated_same_problem_action_triggers_same_action_circuit_breaker(self):
        gm = _make_gm()
        registry = _FakeSkillRegistry()
        registry.move_should_fail = True
        gm.skill_registry = registry
        gm.post_action_wait = lambda *_: None

        for _ in range(3):
            result = gm.execute_actions(['move(x=1, y=0)'])
            self.assertIn("path is likely blocked", result["errors_info"])

        refused = gm.execute_actions(['move(x=1, y=0)'])

        self.assertIn("CIRCUIT-BREAKER", refused["errors_info"])
        self.assertEqual(refused.get("refusal_type"), "same_action_circuit_breaker")
        self.assertEqual(refused.get("refused_action"), 'move(x=1, y=0)')
        self.assertEqual(refused["last_skill"], "nop()")
        self.assertIn("nop() # circuit-breaker", refused["executed_skills"])

    def test_same_action_circuit_breaker_prefers_recovery_action_when_available(self):
        gm = _make_gm()
        registry = _FakeSkillRegistry()
        registry.move_should_fail = True
        gm.skill_registry = registry
        gm.post_action_wait = lambda *_: None

        for _ in range(3):
            result = gm.execute_actions(['move(x=1, y=0)'])
            self.assertIn("path is likely blocked", result["errors_info"])

        registry.move_should_fail = False
        refused = gm.execute_actions(['move(x=1, y=0)'])

        self.assertIn("CIRCUIT-BREAKER", refused["errors_info"])
        self.assertEqual(refused.get("refusal_type"), "same_action_circuit_breaker")
        self.assertEqual(refused.get("refused_action"), 'move(x=1, y=0)')
        self.assertEqual(refused["last_skill"], 'move(x=0, y=1)')
        self.assertIn('move(x=0, y=1) # circuit-breaker-recovery', refused["executed_skills"])

    def test_same_action_circuit_breaker_busy_timeout_recovery_falls_back_to_nop(self):
        gm = _make_gm()
        registry = _BusyTimeoutMoveRegistry()
        registry.move_should_fail = True
        gm.skill_registry = registry
        gm.post_action_wait = lambda *_: None

        for _ in range(3):
            gm.execute_actions(['move(x=1, y=0)'])

        registry.move_should_fail = False
        registry.move_busy_timeout = True
        refused = gm.execute_actions(['move(x=1, y=0)'])

        self.assertEqual(refused["last_skill"], "nop()")
        self.assertIn("nop() # circuit-breaker", refused["executed_skills"])
        self.assertIn("busy timeout", refused["errors_info"].lower())

    def test_busy_timeout_craft_is_treated_as_failure_and_stops_remaining_actions(self):
        gm = _make_gm()
        gm.skill_registry = _CraftTimeoutRegistry()
        gm.post_action_wait = lambda *_: None

        result = gm.execute_actions(['craft(item="Torch")', 'move(x=1, y=0)'])

        self.assertTrue(result["errors"])
        self.assertIn("busy timeout", result["errors_info"].lower())
        self.assertEqual(result["last_skill"], 'craft(item="Torch")')
        self.assertEqual(result["executed_skills"], ['craft(item="Torch")'])


if __name__ == "__main__":
    unittest.main()
