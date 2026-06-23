"""Task schema for the SPIKE task suite.

Mirrors the structure consumed by ``env/tasks/base.py`` and the YAML files under
``env/tasks/task_suite/``.  Every task entry is a mapping::

    <task_name>:
      id: <int>            # index within its suite
      object: <str>        # the target object/entity (e.g. "Green Slime")
      quantity: <int>      # how many / how much
      tool: <str|null>     # required tool, or null
      save: <str>          # which save slot to load
      init_commands: <list|null>   # console commands run on reset
      evaluator: <str>     # which evaluator branch validates completion
      difficulty: <str>    # easy | medium | hard

The vocabularies below are derived from the *actual* task suites and the
evaluator branches implemented in ``env/tasks/*.py`` so that generated tasks are
runnable without touching the engine.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# --- Valid vocabularies (source of truth = existing suites + evaluators) ------

# evaluator -> the suite(s) that implement it. Used for routing + validation.
EVALUATOR_TO_SUITE: dict[str, str] = {
    # farming.py
    "clear": "farming", "till": "farming", "fertilize": "farming",
    "sow": "farming", "water": "farming", "harvest": "farming",
    # combat.py
    "kill": "combat",
    # crafting.py
    "craft": "crafting", "purchase": "crafting", "sell": "crafting",
    "build": "crafting", "demolish": "crafting", "upgrade_tool": "crafting",
    "upgrade_farmhouse": "crafting", "fill": "crafting", "break": "crafting",
    "open": "crafting", "close": "crafting", "backpack": "crafting",
    "incubate": "crafting", "jojamart": "crafting",
    # social.py
    "talk": "social", "gift": "social", "friendship": "social", "pet": "social",
    "propose": "social", "date": "social", "breakup": "social", "mood": "social",
    "human": "social", "purchase_animal": "social", "sell_animal": "social",
    # exploration.py
    "location": "exploration", "silo": "exploration", "skill": "exploration",
    "profession": "exploration", "bundle": "exploration", "museum": "exploration",
    "repair": "exploration", "accept": "exploration", "quit": "exploration",
    "reward": "exploration", "complete_help": "exploration",
    "exchange": "exploration", "complete_story": "exploration",
    "watch": "exploration", "read": "exploration", "sleep": "exploration",
    "move": "exploration",
}

VALID_EVALUATORS: set[str] = set(EVALUATOR_TO_SUITE)

# Atomic skills the game agent can actually execute (env_config_stardew.json).
VALID_SKILLS: list[str] = [
    "move", "use", "interact", "choose_item", "attach_item", "unattach_item",
    "craft", "choose_option", "menu", "navigate", "descend_mine",
]

VALID_SAVES: set[str] = {"save_new", "save_farming", "save_quests"}
VALID_DIFFICULTIES: set[str] = {"easy", "medium", "hard"}

# init_command grammar seen in the suites, e.g. set_time(time=900), warp("Mine"),
# warp_mine(2). We validate loosely: "<name>(<args>)".
_INIT_CMD_RE = re.compile(r"^[a-zA-Z_]\w*\(.*\)$")


@dataclass
class TaskSpec:
    """A single, validated SPIKE task definition."""

    name: str
    object: str
    quantity: int
    evaluator: str
    difficulty: str = "easy"
    tool: Optional[str] = None
    save: str = "save_new"
    init_commands: Optional[list[str]] = None
    id: Optional[int] = None
    # free-form provenance kept out of the YAML the engine reads
    rationale: str = ""

    # -- serialization -------------------------------------------------------
    def to_yaml_entry(self) -> dict[str, Any]:
        """Return the dict that goes under ``<name>:`` in a suite YAML."""
        return {
            "id": int(self.id) if self.id is not None else 0,
            "object": self.object,
            "quantity": int(self.quantity),
            "tool": self.tool,
            "save": self.save,
            "init_commands": self.init_commands,
            "evaluator": self.evaluator,
            "difficulty": self.difficulty,
        }

    def suite(self) -> Optional[str]:
        return EVALUATOR_TO_SUITE.get(self.evaluator)

    # -- validation ----------------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of human-readable problems (empty == valid)."""
        problems: list[str] = []
        if not self.name or not re.match(r"^[a-z0-9][a-z0-9_'+]*$", self.name):
            problems.append(
                f"name '{self.name}' must be snake_case (lowercase, digits, _ ' +)"
            )
        if not self.object or not str(self.object).strip():
            problems.append("object must be a non-empty string")
        if not isinstance(self.quantity, int) or self.quantity < 1:
            problems.append(f"quantity must be a positive int, got {self.quantity!r}")
        if self.evaluator not in VALID_EVALUATORS:
            problems.append(
                f"evaluator '{self.evaluator}' is not one of the implemented "
                f"evaluators ({sorted(VALID_EVALUATORS)})"
            )
        if self.difficulty not in VALID_DIFFICULTIES:
            problems.append(f"difficulty '{self.difficulty}' not in {VALID_DIFFICULTIES}")
        if self.save not in VALID_SAVES:
            problems.append(f"save '{self.save}' not in {VALID_SAVES}")
        if self.init_commands is not None:
            if not isinstance(self.init_commands, list):
                problems.append("init_commands must be a list or null")
            else:
                for cmd in self.init_commands:
                    if not isinstance(cmd, str) or not _INIT_CMD_RE.match(cmd.strip()):
                        problems.append(f"init_command '{cmd}' is not 'name(args)'")
        return problems

    def is_valid(self) -> bool:
        return not self.validate()

    @classmethod
    def from_dict(cls, name: str, d: dict[str, Any]) -> "TaskSpec":
        return cls(
            name=name,
            object=d.get("object", ""),
            quantity=int(d.get("quantity", 1) or 1),
            evaluator=d.get("evaluator", ""),
            difficulty=d.get("difficulty", "easy"),
            tool=d.get("tool"),
            save=d.get("save", "save_new"),
            init_commands=d.get("init_commands"),
            id=d.get("id"),
            rationale=d.get("rationale", ""),
        )


def validate_many(tasks: list[TaskSpec]) -> dict[str, list[str]]:
    """Validate a batch; returns {task_name: [problems]} for invalid ones only."""
    report: dict[str, list[str]] = {}
    for t in tasks:
        problems = t.validate()
        if problems:
            report[t.name] = problems
    return report


def schema_help() -> str:
    """A compact, LLM-friendly description of the schema + vocabularies."""
    return (
        "Each task is a YAML mapping keyed by a snake_case task name with fields:\n"
        "  object (str), quantity (int>=1), tool (str|null), save (str), "
        "init_commands (list[str]|null), evaluator (str), difficulty (easy|medium|hard).\n"
        f"Valid evaluators by category:\n"
        + "\n".join(
            f"  - {suite}: "
            + ", ".join(sorted(e for e, s in EVALUATOR_TO_SUITE.items() if s == suite))
            for suite in ("farming", "combat", "crafting", "social", "exploration")
        )
        + f"\nValid saves: {sorted(VALID_SAVES)}\n"
        f"Atomic skills the game agent can run: {VALID_SKILLS}\n"
        "init_commands examples: ['set_time(time=900)'], ['warp(\"Mine\")'], "
        "['warp_mine(2)'].  Use null when no setup is needed."
    )
