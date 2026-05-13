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

from stardojo import constants
from stardojo.provider.llm.openai import OpenAIProvider


class TestOpenAIPromptTripartite(unittest.TestCase):
    def test_tripartite_serializes_dict_placeholder_in_part2(self) -> None:
        provider = OpenAIProvider()
        template = "\n\n".join(
            [
                "System directive.",
                "User context before images.",
                constants.IMAGES_INPUT_TAG,
                "Current Menu: <$current_menu$>",
            ]
        )
        params = {
            constants.IMAGES_INPUT_TAG_NAME: [],
            "current_menu": {
                "type": "No Menu",
                "responses": [],
            },
        }

        messages = provider.assemble_prompt_tripartite(template_str=template, params=params)

        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[1]["role"], "user")
        user_text = "\n".join(
            str(item.get("text", ""))
            for item in messages[1].get("content", [])
            if item.get("type") == "text"
        )
        self.assertIn('"type": "No Menu"', user_text)
        self.assertNotIn("'type': 'No Menu'", user_text)
