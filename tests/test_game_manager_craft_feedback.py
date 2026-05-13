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


sys.modules.setdefault("pyautogui", types.SimpleNamespace())
sys.modules.setdefault("pydirectinput", types.SimpleNamespace())
sys.modules.setdefault("ahk", types.SimpleNamespace(AHK=object))
sys.modules.setdefault("dill", types.SimpleNamespace(loads=lambda *args, **kwargs: None, dumps=lambda *args, **kwargs: b""))

dataclass_wizard_module = types.ModuleType("dataclass_wizard")
dataclass_wizard_module.JSONWizard = object
dataclass_wizard_abstractions = types.ModuleType("dataclass_wizard.abstractions")
dataclass_wizard_abstractions.W = object
dataclass_wizard_type_def = types.ModuleType("dataclass_wizard.type_def")
dataclass_wizard_type_def.JSONObject = dict
dataclass_wizard_type_def.Encoder = object
sys.modules.setdefault("dataclass_wizard", dataclass_wizard_module)
sys.modules.setdefault("dataclass_wizard.abstractions", dataclass_wizard_abstractions)
sys.modules.setdefault("dataclass_wizard.type_def", dataclass_wizard_type_def)


from stardojo.gameio.game_manager import GameManager


class TestGameManagerCraftFeedback(unittest.TestCase):
    def test_craft_ok_response_is_treated_as_success(self) -> None:
        self.assertTrue(GameManager._is_successful_craft_response("craft_ok:Scarecrow"))
        self.assertFalse(GameManager._is_successful_craft_response("Message received"))

    def test_craft_fail_response_is_formatted_as_explicit_error(self) -> None:
        message = GameManager._format_craft_failure_error(
            skill_params={"item": "Scarecrow"},
            skill_response="craft_fail:missing_materials",
        )

        self.assertIn('craft(item="Scarecrow") FAILED', message)
        self.assertIn("missing materials", message)

    def test_generic_message_received_is_not_treated_as_craft_success(self) -> None:
        message = GameManager._format_craft_failure_error(
            skill_params={"item": "Scarecrow"},
            skill_response="Message received",
        )

        self.assertIn("returned no confirmation", message)


if __name__ == "__main__":
    unittest.main()
