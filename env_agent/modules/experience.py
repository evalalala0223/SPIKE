"""Experience analysis: turn the game agent's run history into actionable
"pitfalls" the env agent reads before decomposing tasks.

Inputs:
  * ``runs/results/**/result.json`` produced by the SPIKE benchmark runner.
    Each file has top-level outcome fields plus a ``steps[]`` array where every
    step records ``action`` and ``records.{executed_skills,errors,errors_info}``
    and ``task_eval``.
  * (optional) ``agent/memory.json`` style memory snapshots.

Outputs:
  * an aggregated failure report (in-memory dict), and
  * a maintained markdown file (``knowledge/agent_pitfalls.md``) summarising the
    situations where the game agent repeatedly fails, grouped by task / suite /
    error pattern, so it can be fed straight into the decomposition prompt.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def _iter_result_files(results_dir: Path):
    yield from results_dir.rglob("result.json")


def _norm_error(text: str) -> str:
    """Collapse a raw error string into a coarse signature for grouping."""
    if not text:
        return ""
    t = re.sub(r"\d+", "N", text.strip().lower())
    t = re.sub(r"\s+", " ", t)
    return t[:160]


class ExperienceAnalyzer:
    def __init__(self, results_dir: str | Path) -> None:
        self.results_dir = Path(results_dir)

    # -- aggregation ---------------------------------------------------------
    def analyze(self, *, limit_runs: Optional[int] = None) -> dict[str, Any]:
        files = sorted(_iter_result_files(self.results_dir), reverse=True)
        if limit_runs:
            files = files[:limit_runs]

        total = 0
        completed = 0
        per_task: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "attempts": 0,
                "completed": 0,
                "suite": "",
                "difficulty": "",
                "exit_reasons": Counter(),
                "error_sigs": Counter(),
                "failing_skills": Counter(),
            }
        )
        global_errors: Counter = Counter()
        global_exit: Counter = Counter()

        for fp in files:
            try:
                data = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                continue
            total += 1
            name = data.get("task_name") or data.get("task_description") or fp.parent.name
            rec = per_task[name]
            rec["attempts"] += 1
            rec["suite"] = data.get("runner_task_name", rec["suite"])
            rec["difficulty"] = data.get("difficulty", rec["difficulty"])
            ok = bool(data.get("completed"))
            completed += int(ok)
            rec["completed"] += int(ok)

            exit_reason = data.get("end_reason") or data.get("budget_exit_reason") or "unknown"
            rec["exit_reasons"][exit_reason] += 1
            global_exit[exit_reason] += 1

            for step in data.get("steps", []) or []:
                records = step.get("records") or {}
                if records.get("errors"):
                    sig = _norm_error(str(records.get("errors_info", "")))
                    if sig:
                        rec["error_sigs"][sig] += 1
                        global_errors[sig] += 1
                    last_skill = records.get("last_skill") or step.get("action")
                    if last_skill:
                        skill_name = str(last_skill).split("(")[0]
                        rec["failing_skills"][skill_name] += 1

        # finalise: focus on tasks that fail repeatedly
        problem_tasks = []
        for name, rec in per_task.items():
            fail = rec["attempts"] - rec["completed"]
            if rec["attempts"] >= 1 and rec["completed"] < rec["attempts"]:
                rate = rec["completed"] / rec["attempts"] if rec["attempts"] else 0.0
                problem_tasks.append(
                    {
                        "task": name,
                        "suite": rec["suite"],
                        "difficulty": rec["difficulty"],
                        "attempts": rec["attempts"],
                        "completed": rec["completed"],
                        "fail": fail,
                        "success_rate": round(rate, 2),
                        "exit_reasons": rec["exit_reasons"].most_common(3),
                        "top_errors": rec["error_sigs"].most_common(3),
                        "failing_skills": rec["failing_skills"].most_common(3),
                    }
                )
        # worst first: most failures, then lowest success rate
        problem_tasks.sort(key=lambda d: (d["fail"], -d["success_rate"]), reverse=True)

        return {
            "runs_scanned": total,
            "overall_success_rate": round(completed / total, 3) if total else 0.0,
            "global_exit_reasons": global_exit.most_common(8),
            "global_error_signatures": global_errors.most_common(12),
            "problem_tasks": problem_tasks,
        }

    # -- markdown maintenance ------------------------------------------------
    def render_markdown(self, report: dict[str, Any], *, top_n: int = 20) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        out: list[str] = [
            "# Game Agent Pitfalls (auto-generated)",
            "",
            f"_Updated: {ts}; runs scanned: {report['runs_scanned']}; "
            f"overall success rate: {report['overall_success_rate']:.0%}_",
            "",
            "> Read by the env agent before task decomposition. It summarises where "
            "the SPIKE game agent repeatedly struggles so new tasks can scaffold "
            "around these weaknesses (smaller quantities, prerequisite steps, "
            "better tool setup, clearer init_commands).",
            "",
            "## Global failure modes",
            "",
            "Most common exit reasons:",
        ]
        for reason, cnt in report["global_exit_reasons"]:
            out.append(f"- `{reason}` x{cnt}")
        out += ["", "Most common error signatures:"]
        if report["global_error_signatures"]:
            for sig, cnt in report["global_error_signatures"]:
                out.append(f"- (x{cnt}) {sig}")
        else:
            out.append("- _(no step-level errors recorded)_")

        out += ["", f"## Top {top_n} problem tasks", ""]
        for d in report["problem_tasks"][:top_n]:
            out.append(
                f"### {d['task']}  "
                f"[{d['suite']}/{d['difficulty']}] "
                f"— {d['completed']}/{d['attempts']} solved "
                f"(success {d['success_rate']:.0%})"
            )
            if d["exit_reasons"]:
                out.append(
                    "  - exit reasons: "
                    + ", ".join(f"{r}×{c}" for r, c in d["exit_reasons"])
                )
            if d["failing_skills"]:
                out.append(
                    "  - skills present at errors: "
                    + ", ".join(f"{s}×{c}" for s, c in d["failing_skills"])
                )
            if d["top_errors"]:
                for sig, c in d["top_errors"]:
                    out.append(f"  - error (×{c}): {sig}")
            out.append("")
        return "\n".join(out)

    def update_pitfalls_file(
        self, path: str | Path, *, limit_runs: Optional[int] = None, top_n: int = 20
    ) -> dict[str, Any]:
        report = self.analyze(limit_runs=limit_runs)
        md = self.render_markdown(report, top_n=top_n)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")
        return report

    # -- prompting -----------------------------------------------------------
    @staticmethod
    def summary_for_prompt(report: dict[str, Any], *, top_n: int = 10) -> str:
        lines = [
            f"Game-agent overall success rate: {report['overall_success_rate']:.0%} "
            f"over {report['runs_scanned']} runs.",
            "Recurring failure modes (task — why):",
        ]
        for d in report["problem_tasks"][:top_n]:
            why = []
            if d["exit_reasons"]:
                why.append("exits=" + "/".join(r for r, _ in d["exit_reasons"]))
            if d["top_errors"]:
                why.append("err=" + (d["top_errors"][0][0][:80]))
            lines.append(
                f"  - {d['task']} [{d['suite']}/{d['difficulty']}] "
                f"{d['completed']}/{d['attempts']} solved; " + "; ".join(why)
            )
        return "\n".join(lines)
