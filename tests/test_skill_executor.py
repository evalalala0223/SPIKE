from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from env.skill_executor import SkillExecutor


class TestSkillExecutor(unittest.TestCase):
    def test_load_module_reuses_canonical_sys_modules_entry(self) -> None:
        module_name = "stardojo.fake.skill_module"
        fake_module = types.ModuleType(module_name)
        fake_module.actionproxy = None

        executor = SkillExecutor.__new__(SkillExecutor)
        executor.actionproxy = object()

        with mock.patch.dict(sys.modules, {module_name: fake_module}, clear=False):
            with mock.patch("env.skill_executor.importlib.import_module") as import_module:
                loaded = SkillExecutor._load_module(executor, module_name)

        self.assertIs(loaded, fake_module)
        self.assertIs(fake_module.actionproxy, executor.actionproxy)
        import_module.assert_not_called()
        self.assertNotIn(f"_skill_executor_{module_name}", sys.modules)

    def test_load_module_imports_canonical_module_name(self) -> None:
        module_name = "stardojo.fake.skill_module"
        fake_module = types.ModuleType(module_name)
        fake_module.actionproxy = None

        executor = SkillExecutor.__new__(SkillExecutor)
        executor.actionproxy = object()

        fake_spec = mock.Mock()
        fake_spec.loader = object()

        with mock.patch.dict(sys.modules, {}, clear=False):
            with mock.patch(
                "env.skill_executor.importlib.util.find_spec",
                return_value=fake_spec,
            ), mock.patch(
                "env.skill_executor.importlib.import_module",
                return_value=fake_module,
            ) as import_module:
                loaded = SkillExecutor._load_module(executor, module_name)

        self.assertIs(loaded, fake_module)
        self.assertIs(fake_module.actionproxy, executor.actionproxy)
        import_module.assert_called_once_with(module_name)
        self.assertNotIn(f"_skill_executor_{module_name}", sys.modules)


if __name__ == "__main__":
    unittest.main()
