"""Task-pool cognition.

Loads every suite under ``env/tasks/task_suite/*.yaml`` and builds a structured
view the env agent can reason over:

  * the full set of existing tasks (so it never duplicates),
  * which objects / tools / evaluators / difficulties are already used,
  * simple "relatedness" lookup (tasks that share object/evaluator/keywords),
  * a compact textual summary for prompting.

It is read-only with respect to the existing suites; new tasks are written to a
separate file by the toolkit, never mixed into the benchmark suites by default.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional

import yaml

from ..core.schema import TaskSpec


class TaskPool:
    def __init__(self, task_suite_dir: str | Path) -> None:
        self.dir = Path(task_suite_dir)
        self.tasks: dict[str, TaskSpec] = {}        # name -> spec
        self.by_suite: dict[str, list[str]] = defaultdict(list)
        self.suite_files: dict[str, Path] = {}
        self._load()

    def _load(self) -> None:
        for path in sorted(self.dir.glob("*.yaml")):
            suite = path.stem
            self.suite_files[suite] = path
            try:
                data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if not isinstance(data, dict):
                continue
            for name, info in data.items():
                if not isinstance(info, dict):
                    continue
                spec = TaskSpec.from_dict(name, info)
                self.tasks[name] = spec
                self.by_suite[suite].append(name)

    # -- aggregate stats -----------------------------------------------------
    def stats(self) -> dict[str, Any]:
        objs = Counter(t.object for t in self.tasks.values())
        evs = Counter(t.evaluator for t in self.tasks.values())
        tools = Counter((t.tool or "null") for t in self.tasks.values())
        diff = Counter(t.difficulty for t in self.tasks.values())
        return {
            "total": len(self.tasks),
            "suites": {s: len(n) for s, n in self.by_suite.items()},
            "top_objects": objs.most_common(20),
            "evaluators": dict(evs),
            "tools": dict(tools),
            "difficulty": dict(diff),
        }

    # -- relatedness ---------------------------------------------------------
    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))

    def find_related(self, query: str, *, k: int = 12) -> list[TaskSpec]:
        """Return tasks most related to a free-text query (token overlap +
        object/evaluator hits)."""
        q = self._tokens(query)
        scored: list[tuple[float, TaskSpec]] = []
        for t in self.tasks.values():
            hay = f"{t.name} {t.object} {t.evaluator} {t.tool or ''}"
            overlap = len(q & self._tokens(hay))
            if overlap:
                scored.append((overlap, t))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [t for _, t in scored[:k]]

    def exists(self, name: str) -> bool:
        return name in self.tasks

    # -- prompting -----------------------------------------------------------
    def summary_for_prompt(self, *, related_to: Optional[str] = None, k: int = 15) -> str:
        st = self.stats()
        lines = [
            f"Task pool: {st['total']} tasks across suites "
            + ", ".join(f"{s}({n})" for s, n in st["suites"].items()),
            "Existing difficulty mix: " + ", ".join(f"{d}={c}" for d, c in st["difficulty"].items()),
        ]
        if related_to:
            rel = self.find_related(related_to, k=k)
            if rel:
                lines.append(f"\nExisting tasks related to '{related_to}':")
                for t in rel:
                    lines.append(
                        f"  - {t.name}: object={t.object!r}, qty={t.quantity}, "
                        f"tool={t.tool!r}, evaluator={t.evaluator}, "
                        f"difficulty={t.difficulty}"
                    )
        return "\n".join(lines)
