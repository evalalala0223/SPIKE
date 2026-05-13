from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _LoggerStub:
    def debug(self, *args, **kwargs) -> None:
        return None

    def write(self, *args, **kwargs) -> None:
        return None

    def warn(self, *args, **kwargs) -> None:
        return None

    def error(self, *args, **kwargs) -> None:
        return None


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []
    return module


@contextlib.contextmanager
def _patched_modules(replacements: dict[str, types.ModuleType]):
    saved = {name: sys.modules.get(name) for name in replacements}
    try:
        sys.modules.update(replacements)
        yield
    finally:
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def _load_module(
    alias: str,
    relative_path: str,
    replacements: dict[str, types.ModuleType],
):
    module_path = ROOT / relative_path
    with _patched_modules(replacements):
        spec = importlib.util.spec_from_file_location(alias, module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[alias] = module
        try:
            spec.loader.exec_module(module)
        finally:
            sys.modules.pop(alias, None)
    return module


def _build_game_manager_stubs(
    prefix: str,
    *,
    allow_generated_registration: bool,
) -> dict[str, types.ModuleType]:
    package = _package(prefix)
    constants_mod = types.ModuleType(f"{prefix}.constants")
    constants_mod.GENERAL_GAME_INTERFACE = "general"
    constants_mod.PAUSE_INTERFACE = "pause"
    constants_mod.SKILL_CONFIG_ALLOW_GENERATED_REGISTRATION = "allow_generated_skill_registration"
    constants_mod.EXECUTED_SKILLS = "executed_skills"
    constants_mod.LAST_SKILL = "last_skill"
    constants_mod.ERRORS = "errors"
    constants_mod.ERRORS_INFO = "errors_info"

    config_mod = types.ModuleType(f"{prefix}.config")

    class Config:
        def __init__(self) -> None:
            self.env_name = "env"
            self.ide_name = "ide"
            self.work_dir = "."
            self.skill_configs = {
                constants_mod.SKILL_CONFIG_ALLOW_GENERATED_REGISTRATION: allow_generated_registration,
            }

    config_mod.Config = Config

    log_mod = types.ModuleType(f"{prefix}.log")
    log_mod.Logger = _LoggerStub

    environment_pkg = _package(f"{prefix}.environment")
    ui_control_mod = types.ModuleType(f"{prefix}.environment.ui_control")
    ui_control_mod.UIControl = type("UIControl", (), {})

    gameio_pkg = _package(f"{prefix}.gameio")

    class IOEnvironment:
        def __init__(self) -> None:
            self.llm_provider = None

        def update_timeouts(self) -> None:
            return None

    gameio_pkg.IOEnvironment = IOEnvironment
    lifecycle_pkg = _package(f"{prefix}.gameio.lifecycle")
    lifecycle_mod = types.ModuleType(f"{prefix}.gameio.lifecycle.ui_control")
    lifecycle_mod.check_active_window = lambda: True

    utils_pkg = _package(f"{prefix}.utils")
    file_utils_mod = types.ModuleType(f"{prefix}.utils.file_utils")
    file_utils_mod.assemble_project_path = lambda path: str(path)

    package.constants = constants_mod
    package.config = config_mod
    package.log = log_mod
    package.environment = environment_pkg
    package.gameio = gameio_pkg
    package.utils = utils_pkg
    environment_pkg.ui_control = ui_control_mod
    gameio_pkg.lifecycle = lifecycle_pkg
    lifecycle_pkg.ui_control = lifecycle_mod
    utils_pkg.file_utils = file_utils_mod

    return {
        prefix: package,
        f"{prefix}.constants": constants_mod,
        f"{prefix}.config": config_mod,
        f"{prefix}.log": log_mod,
        f"{prefix}.environment": environment_pkg,
        f"{prefix}.environment.ui_control": ui_control_mod,
        f"{prefix}.gameio": gameio_pkg,
        f"{prefix}.gameio.lifecycle": lifecycle_pkg,
        f"{prefix}.gameio.lifecycle.ui_control": lifecycle_mod,
        f"{prefix}.utils": utils_pkg,
        f"{prefix}.utils.file_utils": file_utils_mod,
    }


def _build_skill_registry_stubs(prefix: str) -> dict[str, types.ModuleType]:
    package = _package(prefix)
    constants_mod = types.ModuleType(f"{prefix}.constants")
    constants_mod.SKILL_CONFIG_FROM_DEFAULT = "skill_from_default"
    constants_mod.SKILL_CONFIG_MODE = "skill_mode"
    constants_mod.SKILL_CONFIG_NAMES_BASIC = "skill_names_basic"
    constants_mod.SKILL_CONFIG_NAMES_ALLOW = "skill_names_allow"
    constants_mod.SKILL_CONFIG_NAMES_DENY = "skill_names_deny"
    constants_mod.SKILL_CONFIG_NAMES_OTHERS = "skill_names_others"
    constants_mod.SKILL_CONFIG_REGISTERED_SKILLS = "skills_registered"

    config_mod = types.ModuleType(f"{prefix}.config")

    class Config:
        def __init__(self) -> None:
            self.skill_configs = {
                constants_mod.SKILL_CONFIG_FROM_DEFAULT: False,
                constants_mod.SKILL_CONFIG_MODE: None,
                constants_mod.SKILL_CONFIG_NAMES_BASIC: [],
                constants_mod.SKILL_CONFIG_NAMES_ALLOW: [],
                constants_mod.SKILL_CONFIG_NAMES_DENY: [],
                constants_mod.SKILL_CONFIG_NAMES_OTHERS: None,
                constants_mod.SKILL_CONFIG_REGISTERED_SKILLS: None,
            }

    config_mod.Config = Config

    log_mod = types.ModuleType(f"{prefix}.log")
    log_mod.Logger = _LoggerStub

    json_utils_mod = types.ModuleType(f"{prefix}.utils.json_utils")
    json_utils_mod.load_json = lambda path: {}
    json_utils_mod.save_json = lambda *args, **kwargs: None

    dict_utils_mod = types.ModuleType(f"{prefix}.utils.dict_utils")
    dict_utils_mod.kget = lambda mapping, key, default=None: mapping.get(key, default) if isinstance(mapping, dict) else default

    check_mod = types.ModuleType(f"{prefix}.utils.check")
    check_mod.is_valid_value = lambda value: True

    skill_mod = types.ModuleType(f"{prefix}.environment.skill")
    skill_mod.Skill = type("Skill", (), {})

    env_utils_mod = types.ModuleType(f"{prefix}.environment.utils")
    env_utils_mod.serialize_skills = lambda skills: skills
    env_utils_mod.deserialize_skills = lambda skills: skills

    io_env_mod = types.ModuleType(f"{prefix}.gameio.io_env")
    io_env_mod.IOEnvironment = type("IOEnvironment", (), {})

    environment_pkg = _package(f"{prefix}.environment")
    gameio_pkg = _package(f"{prefix}.gameio")
    utils_pkg = _package(f"{prefix}.utils")

    package.constants = constants_mod
    package.config = config_mod
    package.log = log_mod
    package.environment = environment_pkg
    package.gameio = gameio_pkg
    package.utils = utils_pkg
    environment_pkg.skill = skill_mod
    environment_pkg.utils = env_utils_mod
    gameio_pkg.io_env = io_env_mod
    utils_pkg.json_utils = json_utils_mod
    utils_pkg.dict_utils = dict_utils_mod
    utils_pkg.check = check_mod

    return {
        prefix: package,
        f"{prefix}.constants": constants_mod,
        f"{prefix}.config": config_mod,
        f"{prefix}.log": log_mod,
        f"{prefix}.environment": environment_pkg,
        f"{prefix}.environment.skill": skill_mod,
        f"{prefix}.environment.utils": env_utils_mod,
        f"{prefix}.gameio": gameio_pkg,
        f"{prefix}.gameio.io_env": io_env_mod,
        f"{prefix}.utils": utils_pkg,
        f"{prefix}.utils.json_utils": json_utils_mod,
        f"{prefix}.utils.dict_utils": dict_utils_mod,
        f"{prefix}.utils.check": check_mod,
    }


class TestGeneratedSkillRegistrationGuards(unittest.TestCase):
    def _load_game_manager(self, prefix: str, *, allow_generated_registration: bool):
        return _load_module(
            alias=f"test_{prefix}_game_manager",
            relative_path=f"agent/{prefix}/gameio/game_manager.py",
            replacements=_build_game_manager_stubs(
                prefix,
                allow_generated_registration=allow_generated_registration,
            ),
        )

    def _load_skill_registry(self, prefix: str):
        return _load_module(
            alias=f"test_{prefix}_skill_registry",
            relative_path=f"agent/{prefix}/environment/skill_registry.py",
            replacements=_build_skill_registry_stubs(prefix),
        )

    def test_generated_skill_registration_is_blocked_by_default(self) -> None:
        sample_actions = [{"values": [{"code": "def generated_skill():\n    return 1"}]}]

        for prefix in ("stardojo", "cradle"):
            with self.subTest(prefix=prefix):
                module = self._load_game_manager(
                    prefix,
                    allow_generated_registration=False,
                )
                manager = module.GameManager.__new__(module.GameManager)
                calls: list[dict[str, object]] = []

                def _unexpected_add(**kwargs):
                    calls.append(kwargs)
                    return True, "unexpected"

                manager.add_new_skill = _unexpected_add

                attempted = module.GameManager.register_generated_skills(
                    manager,
                    sample_actions,
                )

                self.assertEqual(attempted, 0)
                self.assertEqual(calls, [])

    def test_generated_skill_registration_marks_code_as_trusted_when_explicitly_enabled(self) -> None:
        sample_actions = [
            {"values": [{"code": "def alpha():\n    return 1"}, {"code": "   "}]},
            {"values": [{"code": "def beta():\n    return 2"}]},
            {"values": ["ignore-me"]},
        ]

        for prefix in ("stardojo", "cradle"):
            with self.subTest(prefix=prefix):
                module = self._load_game_manager(
                    prefix,
                    allow_generated_registration=True,
                )
                manager = module.GameManager.__new__(module.GameManager)
                calls: list[dict[str, object]] = []

                def _record_add(**kwargs):
                    calls.append(kwargs)
                    return True, "ok"

                manager.add_new_skill = _record_add

                attempted = module.GameManager.register_generated_skills(
                    manager,
                    sample_actions,
                )

                self.assertEqual(attempted, 2)
                self.assertEqual(len(calls), 2)
                self.assertTrue(all(call["trusted_source"] is True for call in calls))

    def test_skill_registry_rejects_untrusted_dynamic_code(self) -> None:
        for prefix in ("stardojo", "cradle"):
            with self.subTest(prefix=prefix):
                module = self._load_skill_registry(prefix)
                registry = object.__new__(module.SkillRegistry)

                ok, info = module.SkillRegistry.register_skill_from_code(
                    registry,
                    "def generated_skill():\n    return 1",
                    trusted_source=False,
                )

                self.assertFalse(ok)
                self.assertIn("untrusted source", info.lower())

    def test_call_sites_use_register_generated_skills_helper(self) -> None:
        targets = [
            ROOT / "agent/stardojo/provider/module/skill_curation.py",
            ROOT / "agent/cradle/provider/module/skill_curation.py",
            ROOT / "agent/stardojo/stardojo_react_agent.py",
        ]

        for path in targets:
            with self.subTest(path=path.name):
                text = path.read_text(encoding="utf-8")
                self.assertIn("register_generated_skills(all_generated_actions)", text)
                self.assertNotIn(
                    "add_new_skill(skill_code=extracted_skill['code'])",
                    text,
                )


if __name__ == "__main__":
    unittest.main()
