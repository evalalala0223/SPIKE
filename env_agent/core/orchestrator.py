"""EnvAgent orchestrator: ties the modules into a single reasoning loop.

Pipeline (one decomposition run):

    1. EXPERIENCE   -> scan game-agent run history, refresh pitfalls.md
    2. COGNITION    -> pull related existing tasks from the pool
    3. KNOWLEDGE    -> grounded QA over the offline wiki (+ optional web)
    4. DECOMPOSE    -> LLM turns goal+context into validated TaskSpecs
    5. PERSIST      -> write a SPIKE-compatible suite YAML via the toolkit
                       (optionally through the OpenHands SDK)

Every stage emits a structured event (events.jsonl) so a run is fully
reconstructable. State objects are passed explicitly (no global singletons),
mirroring the OpenHands SDK's stateless-agent / explicit-state design.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import yaml

from .events import EventLog
from .llm_client import LLMClient
from .schema import TaskSpec
from ..modules.task_pool import TaskPool
from ..modules.knowledge_base import KnowledgeBase
from ..modules.experience import ExperienceAnalyzer
from ..modules.decomposer import TaskDecomposer
from ..tools.sdk_tools import Toolkit


def _resolve(root: Path, p: str | None) -> Optional[Path]:
    if p is None:
        return None
    pp = Path(p)
    return pp if pp.is_absolute() else (root / pp)


@dataclass
class DecomposeResult:
    goal: str
    tasks: list[TaskSpec]
    output_path: Optional[str]
    knowledge_sources: list[str] = field(default_factory=list)
    invalid: dict[str, list[str]] = field(default_factory=dict)
    events_path: Optional[str] = None


class EnvAgent:
    def __init__(self, config_path: str | Path, *, repo_root: Optional[str | Path] = None) -> None:
        self.config_path = Path(config_path)
        self.cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        # repo root defaults to two levels up from this file (SPIKE/)
        self.root = Path(repo_root).resolve() if repo_root else self.config_path.resolve().parents[1]

        # --- LLM ---
        llm_cfg = self.cfg.get("llm", {})
        llm_config = _resolve(self.root, llm_cfg.get("config"))
        dotenv = _resolve(self.root, llm_cfg.get("dotenv"))
        self.llm: Optional[LLMClient] = None
        if llm_config and llm_config.exists():
            try:
                self.llm = LLMClient(
                    llm_config,
                    dotenv_path=dotenv,
                    temperature=llm_cfg.get("temperature", 0.4),
                )
            except Exception as e:
                print(f"[env-agent] LLM init failed ({e}); running in offline mode.")

        # --- modules ---
        self.pool = TaskPool(_resolve(self.root, self.cfg["task_pool"]["suite_dir"]))

        kb_cfg = self.cfg.get("knowledge_base", {})
        self.kb = KnowledgeBase(
            _resolve(self.root, kb_cfg.get("cache_dir", "env_agent/knowledge/wiki_cache")),
            llm=self.llm,
            web_search=None,  # wire a search backend here to enable web fallback
        )

        exp_cfg = self.cfg.get("experience", {})
        self.experience = ExperienceAnalyzer(_resolve(self.root, exp_cfg["results_dir"]))
        self.pitfalls_file = _resolve(self.root, exp_cfg.get("pitfalls_file"))
        self._exp_limit = exp_cfg.get("limit_runs")
        self._exp_top_n = exp_cfg.get("top_n", 20)

        self.decomposer = TaskDecomposer(self.llm)

        tools_cfg = self.cfg.get("tools", {})
        self.toolkit = Toolkit(
            self.root,
            backend=tools_cfg.get("backend", "native"),
            llm_model=self.llm.model if self.llm else None,
            llm_base_url=self.llm.base_url if self.llm else None,
        )

        out_cfg = self.cfg.get("output", {})
        self.generated_dir = _resolve(self.root, out_cfg.get("generated_dir", "env_agent/generated"))
        self.events_dir = _resolve(self.root, out_cfg.get("events_dir", "env_agent/runs"))

    # -- helpers -------------------------------------------------------------
    def _new_eventlog(self, tag: str) -> EventLog:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.events_dir / f"{stamp}_{tag}" / "events.jsonl"
        return EventLog(path)

    # -- individual capabilities --------------------------------------------
    def refresh_pitfalls(self) -> dict[str, Any]:
        """Stage 1 on its own: rebuild the pitfalls markdown from run history."""
        report = self.experience.update_pitfalls_file(
            self.pitfalls_file, limit_runs=self._exp_limit, top_n=self._exp_top_n
        )
        return report

    def sync_wiki(self) -> str:
        return self.kb.sync()

    def ask(self, question: str, *, k: int = 4) -> dict[str, Any]:
        """Knowledge-base QA passthrough."""
        return self.kb.answer(question, k=k)

    # -- main pipeline -------------------------------------------------------
    def decompose(
        self,
        goal: str,
        *,
        n: Optional[int] = None,
        write: bool = True,
        knowledge_query: Optional[str] = None,
    ) -> DecomposeResult:
        n = n or self.cfg.get("decompose", {}).get("num_tasks", 6)
        log = self._new_eventlog("decompose")
        log.emit("run_start", f"goal={goal!r}", goal=goal, n=n, llm=bool(self.llm and self.llm.available))

        # 1. experience -> pitfalls
        report = self.experience.update_pitfalls_file(
            self.pitfalls_file, limit_runs=self._exp_limit, top_n=self._exp_top_n
        )
        pitfalls_summary = self.experience.summary_for_prompt(report, top_n=10)
        log.emit(
            "experience_analyzed",
            f"{report['runs_scanned']} runs, {len(report['problem_tasks'])} problem tasks",
            runs_scanned=report["runs_scanned"],
            overall_success_rate=report["overall_success_rate"],
            problem_tasks=[d["task"] for d in report["problem_tasks"][:10]],
        )

        # 2. cognition -> related tasks
        pool_summary = self.pool.summary_for_prompt(related_to=goal, k=15)
        log.emit("pool_cognition", f"{len(self.pool.tasks)} existing tasks indexed",
                 total=len(self.pool.tasks))

        # 3. knowledge QA
        kq = knowledge_query or goal
        qa = self.kb.answer(kq, k=4)
        knowledge_text = qa.get("answer", "")
        log.emit("knowledge_retrieved", f"sources={qa.get('sources')}",
                 query=kq, sources=qa.get("sources", []))

        # 4. decompose
        tasks, meta = self.decomposer.decompose(
            goal,
            pool_summary=pool_summary,
            knowledge=knowledge_text,
            pitfalls=pitfalls_summary,
            existing_names=set(self.pool.tasks),
            n=n,
        )
        log.emit(
            "decomposed",
            f"{len(tasks)} valid tasks (mode={meta.get('mode')})",
            mode=meta.get("mode"),
            valid=[t.name for t in tasks],
            invalid=meta.get("invalid", {}),
        )

        # 5. persist
        output_path: Optional[str] = None
        if write and tasks:
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = "".join(c if c.isalnum() else "_" for c in goal.lower())[:30].strip("_")
            out_file = self.generated_dir / f"{slug or 'goal'}_{stamp}.yaml"
            header = (
                f"generated by env_agent from goal: {goal} "
                f"| {len(tasks)} tasks | {stamp}"
            )
            output_path = self.toolkit.write_tasks_yaml(out_file, tasks, header=header)
            # also drop a sidecar with rationales (kept out of the engine YAML)
            self._write_rationale_sidecar(out_file, goal, tasks, qa)
            log.emit("persisted", f"wrote {len(tasks)} tasks -> {output_path}",
                     path=output_path, backend=self.toolkit.backend)

        events_path = str(log.path)
        log.emit("run_end", "done")
        log.close()

        return DecomposeResult(
            goal=goal,
            tasks=tasks,
            output_path=output_path,
            knowledge_sources=qa.get("sources", []),
            invalid=meta.get("invalid", {}),
            events_path=events_path,
        )

    def _write_rationale_sidecar(
        self, yaml_path: Path, goal: str, tasks: list[TaskSpec], qa: dict[str, Any]
    ) -> None:
        lines = [
            f"# Curriculum for: {goal}",
            "",
            f"_Generated {datetime.now().isoformat(timespec='seconds')}; "
            f"knowledge sources: {qa.get('sources') or 'none'}_",
            "",
        ]
        for i, t in enumerate(tasks, 1):
            lines.append(
                f"{i}. **{t.name}** "
                f"(`{t.evaluator}`, {t.difficulty}, object={t.object}, qty={t.quantity}, "
                f"tool={t.tool})"
            )
            if t.rationale:
                lines.append(f"   - {t.rationale}")
        md_path = yaml_path.with_suffix(".md")
        self.toolkit.write_text(md_path, "\n".join(lines))
