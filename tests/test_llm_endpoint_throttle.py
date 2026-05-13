from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from cradle.utils import llm_endpoint_throttle as throttle


class TestLLMEndpointThrottle(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        throttle._CONFIG_CACHE = {
            "enabled": True,
            "max_concurrency": 8,
            "model_name_substrings": ["qwen"],
            "slot_dir": self._tmpdir.name,
            "timeout_s": 30,
            "wait_budget_ratio": 0.5,
            "min_request_window_s": 20.0,
            "poll_interval_s": 0.05,
            "stale_after_s": 60,
        }

    def tearDown(self) -> None:
        throttle._CONFIG_CACHE = None
        self._tmpdir.cleanup()

    def test_timeout_raises_instead_of_proceeding_without_slot(self) -> None:
        for slot_idx in range(8):
            slot_path = Path(self._tmpdir.name) / f"slot_{slot_idx}.lock"
            slot_path.write_text("busy", encoding="utf-8")

        with self.assertRaises(throttle.LLMEndpointThrottleTimeout):
            with throttle.acquire_llm_endpoint_slot(
                model_name="qwen-plus",
                purpose="unit_test",
                timeout_s=1.0,
            ):
                self.fail("expected throttle timeout before entering context")

    def test_wait_budget_uses_total_timeout(self) -> None:
        wait_timeout = throttle.get_llm_endpoint_wait_timeout(
            "qwen-plus",
            total_timeout_s=30.0,
        )
        self.assertEqual(wait_timeout, 10.0)

    def test_slot_metadata_reports_wait_and_limit(self) -> None:
        with throttle.acquire_llm_endpoint_slot(
            model_name="qwen-plus",
            purpose="unit_test",
        ) as slot_info:
            self.assertTrue(slot_info["throttled"])
            self.assertEqual(slot_info["max_concurrency"], 8)
            self.assertGreaterEqual(slot_info["waited_s"], 0.0)


if __name__ == "__main__":
    unittest.main()
