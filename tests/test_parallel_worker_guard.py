from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env.parallel_worker_guard import resolve_parallel_worker_limit


class TestParallelWorkerGuard(unittest.TestCase):
    def test_throttle_disabled_keeps_requested_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            enhanced_path = root / "agent" / "conf"
            enhanced_path.mkdir(parents=True, exist_ok=True)
            (enhanced_path / "enhanced_config.yaml").write_text(
                yaml.safe_dump(
                    {
                        "performance": {
                            "llm_endpoint_throttle": {
                                "enabled": False,
                                "max_concurrency": 5,
                            }
                        }
                    },
                    allow_unicode=False,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            llm_path = root / "agent" / "conf" / "openai_config.json"
            llm_path.write_text(json.dumps({"comp_model": "Qwen/Qwen3.5-397B-A17B-FP8"}), encoding="utf-8")

            decision = resolve_parallel_worker_limit(
                8,
                root_dir=root,
                llm_config_path="agent/conf/openai_config.json",
            )

            self.assertFalse(decision.throttle_enabled)
            self.assertEqual(decision.requested_workers, 8)
            self.assertEqual(decision.effective_workers, 8)
            self.assertFalse(decision.limited)

    def test_matching_model_keeps_requested_workers_and_enables_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            enhanced_path = root / "agent" / "conf"
            enhanced_path.mkdir(parents=True, exist_ok=True)
            (enhanced_path / "enhanced_config.yaml").write_text(
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
                    allow_unicode=False,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            llm_path = root / "agent" / "conf" / "openai_config.json"
            llm_path.write_text(json.dumps({"comp_model": "Qwen/Qwen3.5-397B-A17B-FP8"}), encoding="utf-8")

            decision = resolve_parallel_worker_limit(
                8,
                root_dir=root,
                llm_config_path="agent/conf/openai_config.json",
            )

            self.assertTrue(decision.throttle_enabled)
            self.assertTrue(decision.model_matched)
            self.assertEqual(decision.throttle_max_concurrency, 5)
            self.assertEqual(decision.requested_workers, 8)
            self.assertEqual(decision.effective_workers, 8)
            self.assertFalse(decision.limited)
            self.assertTrue(decision.queue_enforced)

    def test_non_matching_model_does_not_cap_workers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            enhanced_path = root / "agent" / "conf"
            enhanced_path.mkdir(parents=True, exist_ok=True)
            (enhanced_path / "enhanced_config.yaml").write_text(
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
                    allow_unicode=False,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            llm_path = root / "agent" / "conf" / "openai_config.json"
            llm_path.write_text(json.dumps({"comp_model": "gpt-4o"}), encoding="utf-8")

            decision = resolve_parallel_worker_limit(
                8,
                root_dir=root,
                llm_config_path="agent/conf/openai_config.json",
            )

            self.assertTrue(decision.throttle_enabled)
            self.assertFalse(decision.model_matched)
            self.assertEqual(decision.effective_workers, 8)
            self.assertFalse(decision.limited)

    def test_unknown_model_keeps_requested_workers_and_enables_queue(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            enhanced_path = root / "agent" / "conf"
            enhanced_path.mkdir(parents=True, exist_ok=True)
            (enhanced_path / "enhanced_config.yaml").write_text(
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
                    allow_unicode=False,
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            llm_path = root / "agent" / "conf" / "openai_config.json"
            llm_path.write_text(json.dumps({}), encoding="utf-8")

            decision = resolve_parallel_worker_limit(
                8,
                root_dir=root,
                llm_config_path="agent/conf/openai_config.json",
            )

            self.assertTrue(decision.throttle_enabled)
            self.assertFalse(decision.model_matched)
            self.assertEqual(decision.model_name, "")
            self.assertEqual(decision.effective_workers, 8)
            self.assertFalse(decision.limited)
            self.assertTrue(decision.queue_enforced)
            self.assertTrue(decision.queue_enforced)


if __name__ == "__main__":
    unittest.main()
