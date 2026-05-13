import sys
import importlib
import importlib.util
from typing import Any, Optional, Sequence


# Composite skills disabled: only load original StarDojo atomic skills.
# Re-enable by uncommenting and passing as extra_module_names.
# _COMPOSITE_MODULES = (
#     "stardojo.environment.stardew.composite_skills.go_home",
#     "stardojo.environment.stardew.composite_skills.shopping",
#     "stardojo.environment.stardew.composite_skills.go_through_door",
#     "stardojo.environment.stardew.composite_skills.buy_item",
#     "stardojo.environment.stardew.composite_skills.farm",
# )


class SkillExecutor:
    def __init__(
        self,
        actionproxy: Any,
        module_name: str = "stardojo.environment.stardew.atomic_skills.basic_skills",
        extra_module_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.actionproxy = actionproxy
        self.module_names = [module_name, *(extra_module_names or ())]
        self._load_skills()

    def _load_module(self, module_name: str):
        # Reuse the canonical module entry so registered skill functions keep
        # matching their importable module identity.
        skill_module = sys.modules.get(module_name)
        if skill_module is None:
            spec = importlib.util.find_spec(module_name)
            if spec is None or spec.loader is None:
                raise ImportError(f"Module {module_name} not found.")
            skill_module = importlib.import_module(module_name)

        if hasattr(skill_module, "actionproxy"):
            setattr(skill_module, "actionproxy", self.actionproxy)
        return skill_module

    def _load_skills(self):
        for module_name in self.module_names:
            skill_module = self._load_module(module_name)
            exported_names = set(getattr(skill_module, "__all__", []) or [])

            for func_name in dir(skill_module):
                func = getattr(skill_module, func_name)
                if not callable(func) or func_name.startswith("__"):
                    continue
                if exported_names and func_name not in exported_names:
                    continue
                setattr(self, func_name, func)
