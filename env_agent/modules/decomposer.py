"""Task decomposition: the env agent's "brain".

Given a high-level goal plus context (task-pool view, grounded wiki knowledge,
and the game agent's pitfalls), the decomposer asks the LLM to produce a list of
SPIKE-compatible sub-tasks, then validates each against the real task schema and
repairs/drops invalid ones.  The result is a list of :class:`TaskSpec`.

Design notes (borrowed from OpenHands SDK):
  * Strong typing at the boundary (TaskSpec + validate) == the Action/Observation
    contract idea: the LLM's free-form output is coerced into a validated,
    serialisable structure before anything downstream uses it.
  * The decomposer is pure: it never writes files. Persisting is the toolkit's job.
"""

from __future__ import annotations

from typing import Any, Optional

from ..core.llm_client import LLMClient, extract_json
from ..core.schema import TaskSpec, schema_help, validate_many


DECOMPOSE_SYSTEM = """You are the Environment Agent for SPIKE, a benchmark of \
Stardew Valley game-playing agents. Your job is to decompose a high-level goal \
into a curriculum of concrete, executable sub-tasks for a downstream game agent.

Hard requirements:
- Output STRICT JSON: a list of task objects, nothing else.
- Each task object has keys: name, object, quantity, tool, save, init_commands, \
evaluator, difficulty, rationale.
- `name` is snake_case and MUST NOT collide with existing task names.
- Only use evaluators that exist in the engine (listed below). The evaluator must \
match what the task actually checks.
- Respect the game agent's known weaknesses: when it repeatedly fails a kind of \
task, scaffold it (reduce quantity, add prerequisite steps, pick the right tool, \
provide init_commands that set up the scene) and order tasks easy -> hard.
- Prefer grounding numbers/tools/locations in the provided wiki knowledge.

{schema}
"""

DECOMPOSE_USER = """GOAL:
{goal}

TASK-POOL CONTEXT (existing tasks; do not duplicate):
{pool}

GROUNDED GAME KNOWLEDGE (from offline wiki / web):
{knowledge}

GAME-AGENT PITFALLS (where it repeatedly fails — design around these):
{pitfalls}

Produce {n} sub-tasks as a JSON list. Order them as a curriculum (easy first). \
Each task's `rationale` should briefly justify it and reference a pitfall or \
knowledge fact when relevant.
"""


class TaskDecomposer:
    def __init__(self, llm: Optional[LLMClient]) -> None:
        self.llm = llm

    def decompose(
        self,
        goal: str,
        *,
        pool_summary: str,
        knowledge: str,
        pitfalls: str,
        existing_names: set[str],
        n: int = 6,
    ) -> tuple[list[TaskSpec], dict[str, Any]]:
        """Return (valid_tasks, meta). meta carries raw output + validation info."""
        if self.llm is None or not self.llm.available:
            # offline heuristic fallback so the pipeline still produces *something*
            tasks = self._heuristic(goal, n=n, existing_names=existing_names)
            return tasks, {"mode": "heuristic", "raw": None, "invalid": {}}

        system = DECOMPOSE_SYSTEM.format(schema=schema_help())
        user = DECOMPOSE_USER.format(
            goal=goal,
            pool=pool_summary or "(none)",
            knowledge=knowledge or "(none)",
            pitfalls=pitfalls or "(none)",
            n=n,
        )
        try:
            raw = self.llm.chat(
                [{"role": "system", "content": system}, {"role": "user", "content": user}]
            )
        except Exception as e:
            # LLM reachable but failed at runtime (rate limit, budget, network...).
            # Degrade gracefully to the heuristic scaffold so the run still
            # produces a valid (placeholder) curriculum instead of crashing.
            tasks = self._heuristic(goal, n=n, existing_names=existing_names)
            return tasks, {"mode": "heuristic_fallback", "raw": None, "error": str(e), "invalid": {}}
        try:
            data = extract_json(raw)
        except Exception as e:
            tasks = self._heuristic(goal, n=n, existing_names=existing_names)
            return tasks, {"mode": "heuristic_fallback", "raw": raw, "error": str(e), "invalid": {}}

        if isinstance(data, dict):
            data = data.get("tasks", [data])
        candidates: list[TaskSpec] = []
        for i, item in enumerate(data if isinstance(data, list) else []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip().lower().replace(" ", "_")
            spec = TaskSpec(
                name=name,
                object=str(item.get("object", "")).strip(),
                quantity=int(item.get("quantity", 1) or 1),
                evaluator=str(item.get("evaluator", "")).strip(),
                difficulty=str(item.get("difficulty", "easy")).strip().lower(),
                tool=item.get("tool") or None,
                save=str(item.get("save", "save_new")).strip() or "save_new",
                init_commands=item.get("init_commands") or None,
                id=i,
                rationale=str(item.get("rationale", "")).strip(),
            )
            candidates.append(spec)

        valid, invalid = self._filter_valid(candidates, existing_names)
        return valid, {"mode": "llm", "raw": raw, "invalid": invalid}

    # -- validation / repair -------------------------------------------------
    @staticmethod
    def _filter_valid(
        candidates: list[TaskSpec], existing_names: set[str]
    ) -> tuple[list[TaskSpec], dict[str, list[str]]]:
        valid: list[TaskSpec] = []
        invalid: dict[str, list[str]] = {}
        seen = set(existing_names)
        for spec in candidates:
            problems = spec.validate()
            if spec.name in seen:
                problems.append("name collides with an existing/earlier task")
            if problems:
                invalid[spec.name or "<unnamed>"] = problems
                continue
            spec.id = len(valid)
            valid.append(spec)
            seen.add(spec.name)
        return valid, invalid

    # -- offline fallback ----------------------------------------------------
    @staticmethod
    def _heuristic(goal: str, *, n: int, existing_names: set[str]) -> list[TaskSpec]:
        """A trivial, schema-valid scaffold used only when no LLM is available.

        It produces a graduated `location`/`kill`-style curriculum so the rest of
        the pipeline (validation, writing, events) can be exercised end-to-end.
        """
        base = "".join(c if c.isalnum() else "_" for c in goal.lower())[:24].strip("_") or "goal"
        out: list[TaskSpec] = []
        for i in range(max(1, n)):
            name = f"{base}_step_{i}"
            if name in existing_names:
                continue
            out.append(
                TaskSpec(
                    name=name,
                    object="BusStop",
                    quantity=1,
                    evaluator="location",
                    difficulty="easy" if i < n // 2 else "medium",
                    tool=None,
                    save="save_new",
                    init_commands=None,
                    id=i,
                    rationale="offline heuristic placeholder (no LLM configured)",
                )
            )
        return out
