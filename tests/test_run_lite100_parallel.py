from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from env.parallel_worker_guard import resolve_parallel_worker_limit
from run_lite100_parallel import (
    cleanup_workspace_benchmark_processes,
    cleanup_workspace_fastllm_health_cache,
    cleanup_workspace_llm_endpoint_slots,
)


class TestRunLite100Parallel(unittest.TestCase):
    def test_cleanup_workspace_llm_endpoint_slots_removes_only_lock_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            slot_dir = root / "agent" / "cache" / "locks" / "llm_endpoint_slots"
            slot_dir.mkdir(parents=True, exist_ok=True)
            (root / "agent" / "conf").mkdir(parents=True, exist_ok=True)

            with (root / "agent" / "conf" / "enhanced_config.yaml").open("w", encoding="utf-8") as fd:
                yaml.safe_dump(
                    {
                        "performance": {
                            "llm_endpoint_throttle": {
                                "slot_dir": "./cache/locks/llm_endpoint_slots",
                            }
                        }
                    },
                    fd,
                    allow_unicode=True,
                    sort_keys=False,
                )

            (slot_dir / "slot_0.lock").write_text("lock-0", encoding="utf-8")
            (slot_dir / "slot_4.lock").write_text("lock-4", encoding="utf-8")
            marker = slot_dir / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            cleaned = cleanup_workspace_llm_endpoint_slots(root)

            self.assertEqual(cleaned, 2)
            self.assertFalse((slot_dir / "slot_0.lock").exists())
            self.assertFalse((slot_dir / "slot_4.lock").exists())
            self.assertTrue(marker.exists())

    def test_cleanup_workspace_llm_endpoint_slots_returns_zero_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "agent" / "conf").mkdir(parents=True, exist_ok=True)

            with (root / "agent" / "conf" / "enhanced_config.yaml").open("w", encoding="utf-8") as fd:
                yaml.safe_dump(
                    {
                        "performance": {
                            "llm_endpoint_throttle": {
                                "slot_dir": "./cache/locks/llm_endpoint_slots",
                            }
                        }
                    },
                    fd,
                    allow_unicode=True,
                    sort_keys=False,
                )

            cleaned = cleanup_workspace_llm_endpoint_slots(root)

            self.assertEqual(cleaned, 0)

    def test_cleanup_workspace_benchmark_processes_matches_relative_script_by_cwd(self) -> None:
        root = Path(r"C:\code_all\stardojo")
        fake_proc = mock.Mock()
        fake_proc.info = {
            "pid": 4242,
            "cmdline": ["python.exe", "run_lite100_parallel.py", "--parallel_numb", "4"],
        }
        fake_proc.cwd.return_value = str(root)

        with mock.patch("run_lite100_parallel.os.getpid", return_value=999999):
            with mock.patch("run_lite100_parallel.psutil.process_iter", return_value=[fake_proc]):
                with mock.patch("run_lite100_parallel._terminate_process_tree") as terminate:
                    cleaned = cleanup_workspace_benchmark_processes(root)

        self.assertEqual(cleaned, 1)
        terminate.assert_called_once_with(fake_proc)

    def test_cleanup_workspace_fastllm_health_cache_removes_json_and_lock_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cache_dir = root / "agent" / "cache" / "locks" / "fastllm_health"
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / "status.json").write_text("{}", encoding="utf-8")
            (cache_dir / "probe.lock").write_text("lock", encoding="utf-8")
            marker = cache_dir / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            cleaned = cleanup_workspace_fastllm_health_cache(root)

            self.assertEqual(cleaned, 2)
            self.assertFalse((cache_dir / "status.json").exists())
            self.assertFalse((cache_dir / "probe.lock").exists())
            self.assertTrue(marker.exists())

    def test_parallel_worker_guard_keeps_requested_workers_when_queue_is_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "agent" / "conf").mkdir(parents=True, exist_ok=True)

            with (root / "agent" / "conf" / "enhanced_config.yaml").open("w", encoding="utf-8") as fd:
                yaml.safe_dump(
                    {
                        "performance": {
                            "llm_endpoint_throttle": {
                                "enabled": True,
                                "max_concurrency": 5,
                                "model_name_substrings": ["qwen"],
                            }
                        }
                    },
                    fd,
                    allow_unicode=True,
                    sort_keys=False,
                )

            llm_config = root / "agent" / "conf" / "openai_config.json"
            llm_config.write_text('{"comp_model":"Qwen/Qwen3.5-397B-A17B-FP8"}', encoding="utf-8")

            decision = resolve_parallel_worker_limit(
                8,
                root_dir=root,
                llm_config_path="agent/conf/openai_config.json",
            )

            self.assertEqual(decision.requested_workers, 8)
            self.assertEqual(decision.effective_workers, 8)
            self.assertTrue(decision.queue_enforced)
            self.assertFalse(decision.limited)
            self.assertEqual(decision.throttle_max_concurrency, 5)


if __name__ == "__main__":
    unittest.main()
