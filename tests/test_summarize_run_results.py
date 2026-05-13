from __future__ import annotations

import sys
import types
import shutil
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

if "gymnasium" not in sys.modules:
    gymnasium_stub = types.ModuleType("gymnasium")
    gymnasium_stub.register = lambda *args, **kwargs: None
    sys.modules["gymnasium"] = gymnasium_stub

from summarize_run_results import _build_markdown_report, _summarize_tasks


class TestSummarizeRunResults(unittest.TestCase):
    def test_summary_prefers_derived_run_status_over_stale_index_status(self) -> None:
        root_dir = ROOT / "tests" / ".tmp_summary_case"
        shutil.rmtree(root_dir, ignore_errors=True)
        run_dir = root_dir / "runs" / "results" / "run_001"
        run_dir.mkdir(parents=True, exist_ok=True)

        try:
            (run_dir / "index.json").write_text(
                (
                    "{"
                    "\"run_status\": \"running\", "
                    "\"benchmark_status\": \"invalid\", "
                    "\"invalid_reason\": \"stale_index\", "
                    "\"expected_tasks\": 2, "
                    "\"actual_tasks\": 2"
                    "}"
                ),
                encoding="utf-8",
            )

            task_results = [
                {
                    "task_index": 1,
                    "task_name": "alpha",
                    "completed": True,
                    "end_reason": "completed",
                    "duration_sec": 1.0,
                    "exit_step": 1,
                },
                {
                    "task_index": 2,
                    "task_name": "beta",
                    "completed": False,
                    "end_reason": "max_steps",
                    "budget_exit_reason": "max_steps",
                    "duration_sec": 2.0,
                    "exit_step": 2,
                },
            ]

            summary, enriched_results = _summarize_tasks(
                run_dir,
                task_results,
                root_dir=root_dir,
                prompt_cost_per_1k=None,
                completion_cost_per_1k=None,
            )
            markdown = _build_markdown_report(summary, enriched_results)

            self.assertEqual(summary["derived_run_status"], "completed")
            self.assertTrue(summary["index_run_status_stale"])
            self.assertIn(
                "Run lifecycle status (derived from task results): completed",
                markdown,
            )
            self.assertIn(
                "stale stored run status (index.json): running",
                markdown.lower(),
            )
            self.assertIn(
                "`index.json` still says `running`, but this regenerated summary uses on-disk task results as the source of truth.",
                markdown,
            )
        finally:
            shutil.rmtree(root_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
