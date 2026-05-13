from __future__ import annotations

import ast
import logging
import re
import types
import unittest
from pathlib import Path
from typing import Any, Optional


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "agent" / "stardojo" / "stardojo_react_agent.py"


def _load_position_helpers():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))

    wanted = {"_normalize_position", "_format_buildings", "_format_exits"}
    body = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PipelineRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in wanted:
                    body.append(item)
            break

    extracted_module = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(extracted_module)

    namespace = {
        "Any": Any,
        "Optional": Optional,
        "re": re,
        "logging": logging,
    }
    exec(compile(extracted_module, str(SOURCE_PATH), "exec"), namespace)
    namespace["PipelineRunner"] = types.SimpleNamespace(
        _normalize_position=namespace["_normalize_position"],
    )
    return namespace["_normalize_position"], namespace["_format_buildings"], namespace["_format_exits"]


_NORMALIZE_POSITION, _FORMAT_BUILDINGS, _FORMAT_EXITS = _load_position_helpers()


class TestStardojoPositionFormatting(unittest.TestCase):
    def test_normalize_position_parses_string_coordinates(self) -> None:
        self.assertEqual(_NORMALIZE_POSITION("(75, 18)"), (75, 18))
        self.assertEqual(_NORMALIZE_POSITION("x=82, y=14"), (82, 14))

    def test_format_buildings_uses_relative_offsets_from_string_player_position(self) -> None:
        text = _FORMAT_BUILDINGS(
            [
                {
                    "name": "Deluxe Barn",
                    "doorPosition": {"X": 80, "Y": 20},
                }
            ],
            "(75, 18)",
        )

        self.assertIn("relative offset: x=5, y=2", text)
        self.assertIn("2 tiles down", text)
        self.assertIn("5 tiles right", text)

    def test_format_exits_uses_relative_offsets_from_string_player_position(self) -> None:
        text = _FORMAT_EXITS(
            [
                {
                    "target": "Bus Stop",
                    "position": {"X": 72, "Y": 18},
                }
            ],
            "x=75, y=18",
        )

        self.assertIn("relative offset: x=-3, y=0", text)
        self.assertIn("3 tiles left", text)


if __name__ == "__main__":
    unittest.main()
