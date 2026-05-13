from __future__ import annotations

from collections import deque
import unittest

from stardojo import constants
from stardojo.memory.local_memory import LocalMemory


class TestStardojoLocalMemory(unittest.TestCase):
    @staticmethod
    def _make_memory() -> LocalMemory:
        memory = object.__new__(LocalMemory)
        memory.max_recent_steps = 5
        memory.recent_history = {}
        memory.working_area = {}
        memory._memory_debug_enabled = False
        memory._memory_debug_max_len = 240
        return memory

    def test_similarity_search_handles_deque_history_bucket(self) -> None:
        memory = self._make_memory()
        memory.recent_history = {
            "history_summary": deque(["a", "b", "c"], maxlen=5),
            constants.IMAGES_MEM_BUCKET: deque([], maxlen=5),
        }

        results = memory.similarity_search("history_summary", top_k=2)

        self.assertEqual(results, ["b", "c"])

    def test_add_recent_history_kv_normalizes_legacy_scalar_bucket(self) -> None:
        memory = self._make_memory()
        memory.recent_history = {
            constants.LAST_TASK_DURATION: 3,
        }

        memory.add_recent_history_kv(constants.LAST_TASK_DURATION, 2)

        bucket = memory.recent_history[constants.LAST_TASK_DURATION]
        self.assertIsInstance(bucket, deque)
        self.assertEqual(list(bucket), [3, 2])

    def test_load_normalizes_recent_history_buckets(self) -> None:
        memory = self._make_memory()
        memory.recent_history = {
            "history_summary": "legacy summary",
            constants.LAST_TASK_DURATION: [3],
        }

        memory._normalize_recent_history_buckets()

        self.assertIsInstance(memory.recent_history["history_summary"], deque)
        self.assertEqual(list(memory.recent_history["history_summary"]), ["legacy summary"])
        self.assertIsInstance(memory.recent_history[constants.LAST_TASK_DURATION], deque)


if __name__ == "__main__":
    unittest.main()
