from __future__ import annotations

import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestLLMProviderCallBudgetGuards(unittest.TestCase):
    def _assert_counter_calls_present(self, relative_path: str, expected_sources: list[str]) -> None:
        file_path = ROOT / relative_path
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))

        found_sources: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            if node.func.id != "increment_llm_call_counter":
                continue
            if not node.args:
                continue
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                found_sources.append(first_arg.value)

        self.assertEqual(found_sources, expected_sources, msg=relative_path)

    def test_stardojo_provider_counter_hooks_exist(self) -> None:
        self._assert_counter_calls_present(
            "agent/stardojo/provider/llm/claude.py",
            ["big_brain:claude", "big_brain:claude_async"],
        )
        self._assert_counter_calls_present(
            "agent/stardojo/provider/llm/gemini.py",
            ["big_brain:gemini", "big_brain:gemini_async"],
        )
        self._assert_counter_calls_present(
            "agent/stardojo/provider/llm/restful_claude.py",
            ["big_brain:restful_claude", "big_brain:restful_claude_async"],
        )

    def test_cradle_provider_counter_hooks_exist(self) -> None:
        self._assert_counter_calls_present(
            "agent/cradle/provider/llm/claude.py",
            ["big_brain:claude", "big_brain:claude_async"],
        )
        self._assert_counter_calls_present(
            "agent/cradle/provider/llm/restful_claude.py",
            ["big_brain:restful_claude", "big_brain:restful_claude_async"],
        )


if __name__ == "__main__":
    unittest.main()
