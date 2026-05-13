from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


class TestTaskInferenceTemplateFields(unittest.TestCase):
    def test_specialized_task_inference_templates_include_map_context_fields(self) -> None:
        template_root = (
            Path(__file__).resolve().parents[1]
            / "agent"
            / "res"
            / "stardew"
            / "prompts"
            / "templates"
        )

        required_markers = (
            "Location:",
            "Current position (coordinate):",
            "Current Menu:",
            "Buildings on current map:",
            "Exits on current map:",
        )

        for name in (
            "task_inference_cultivation.prompt",
            "task_inference_farm_clearup.prompt",
            "task_inference_farm_ops.prompt",
            "task_inference_shopping.prompt",
        ):
            text = (template_root / name).read_text(encoding="utf-8")
            for marker in required_markers:
                self.assertIn(marker, text, f"{name} missing marker {marker}")
