from __future__ import annotations

import sys
import threading
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)


from cradle.memory.local_memory import LocalMemory
from cradle.runner.langgraph_nodes import LangGraphNodes
from cradle.utils.singleton import Singleton


def _make_memory(max_recent_steps: int = 5) -> LocalMemory:
    memory = object.__new__(LocalMemory)
    memory._state_lock = threading.RLock()
    memory._scope_local = threading.local()
    memory.max_recent_steps = max_recent_steps
    memory.memory_path = "."
    memory.sa_kg = None
    memory.task_duration = 3
    memory._working_area_global = {}
    memory._current_task_scope = ""
    memory._memory_debug_enabled = False
    memory._memory_debug_max_len = 240
    memory._debug_read_cache = {}
    memory._recent_history_global = memory._build_default_recent_history()
    memory._normalize_recent_history_buckets()
    return memory


class _StubProvider:
    def __init__(self, result: dict | None = None) -> None:
        self._result = result or {}

    def __call__(self, *_: object, **__: object) -> dict:
        return dict(self._result)


class _CountingRuntimeMemory:
    def __init__(self) -> None:
        self.scope_calls = 0

    def run_with_isolated_scope(self, func, *args, **kwargs):
        self.scope_calls += 1
        return func(*args, **kwargs)


class TestCortexLocalMemory(unittest.TestCase):
    def test_get_latest_uses_default_for_missing_and_none_values(self) -> None:
        memory = _make_memory()

        self.assertEqual(memory.get_latest("missing_key", {}), {})

        memory.update_info_history({"gathered_info": None})
        self.assertEqual(memory.get_latest("gathered_info", {}), {})

        memory.update_info_history({"gathered_info": {"description": "visible crop"}})
        self.assertEqual(
            memory.get_latest("gathered_info", {}),
            {"description": "visible crop"},
        )

    def test_run_with_isolated_scope_does_not_leak_thread_updates(self) -> None:
        memory = _make_memory()
        memory.update_info_history({"task_description": "global_task"})

        thread_result: dict[str, object] = {}

        def _worker() -> None:
            def _scoped_update():
                memory.update_info_history(
                    {
                        "task_description": "thread_task",
                        "subtask_description": "thread_subtask",
                    }
                )
                return {
                    "working_area": memory.get_working_area_snapshot(),
                    "history": memory.get_recent_history("task_description", k=2),
                }

            thread_result.update(memory.run_with_isolated_scope(_scoped_update))

        thread = threading.Thread(target=_worker)
        thread.start()
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive())

        self.assertEqual(memory.get_working_area_value("task_description"), "global_task")
        self.assertIsNone(memory.get_working_area_value("subtask_description"))
        self.assertEqual(memory.get_recent_history("task_description", k=2), ["global_task"])

        scoped_working_area = thread_result["working_area"]
        self.assertIsInstance(scoped_working_area, dict)
        self.assertEqual(scoped_working_area["task_description"], "thread_task")
        self.assertEqual(scoped_working_area["subtask_description"], "thread_subtask")
        self.assertEqual(thread_result["history"], ["global_task", "thread_task"])

    def test_explicit_singleton_reconfigure_updates_existing_local_memory(self) -> None:
        memory = LocalMemory(memory_path="singleton_A", max_recent_steps=5)
        saved_state = {
            "memory_path": memory.memory_path,
            "max_recent_steps": memory.max_recent_steps,
            "working_area": deepcopy(memory.get_working_area_snapshot()),
            "recent_history": deepcopy(memory.get_recent_history_snapshot()),
            "task_scope": getattr(memory, "_current_task_scope", ""),
        }

        try:
            memory.reset_runtime_state(task_scope="unit:test", work_dir="singleton_A")
            memory.update_info_history({"action": "a1"})
            memory.update_info_history({"action": "a2"})
            memory.update_info_history({"action": "a3"})

            same_memory = LocalMemory(memory_path="singleton_B", max_recent_steps=2)

            self.assertIs(memory, same_memory)
            self.assertEqual(memory.memory_path, "singleton_B")
            self.assertEqual(memory.max_recent_steps, 2)
            self.assertEqual(memory.get_recent_history("action", k=5), ["a2", "a3"])

            noarg_memory = LocalMemory()
            self.assertIs(memory, noarg_memory)
            self.assertEqual(memory.memory_path, "singleton_B")
            self.assertEqual(memory.max_recent_steps, 2)
        finally:
            with memory._state_lock:
                memory.memory_path = saved_state["memory_path"]
                memory.max_recent_steps = saved_state["max_recent_steps"]
                memory._working_area_global = saved_state["working_area"]
                memory._recent_history_global = saved_state["recent_history"]
                memory._current_task_scope = saved_state["task_scope"]

    def test_local_memory_init_survives_sakg_initialization_failure(self) -> None:
        existing = Singleton._instances.pop(LocalMemory, None)

        class _BrokenSAKG:
            def initialize(self, *args, **kwargs) -> None:
                raise RuntimeError("boom")

        try:
            with mock.patch("cradle.memory.sa_kg.SAKG", _BrokenSAKG):
                memory = LocalMemory(memory_path="sakg_fail", max_recent_steps=3)
                self.assertIsNone(memory.sa_kg)
                memory.update_info_history({"task_description": "still_usable"})
                self.assertEqual(memory.get_latest("task_description", ""), "still_usable")
        finally:
            Singleton._instances.pop(LocalMemory, None)
            if existing is not None:
                Singleton._instances[LocalMemory] = existing

    def test_parallel_langgraph_uses_memory_isolation_hook(self) -> None:
        runtime_memory = _CountingRuntimeMemory()
        providers = {
            "video_clip": _StubProvider({}),
            "self_reflection": _StubProvider({}),
            "task_inference": _StubProvider({}),
            "action_planning": _StubProvider({}),
            "skill_execute": _StubProvider({}),
        }
        nodes = LangGraphNodes(providers=providers, runtime_memory=runtime_memory)
        nodes.info_gathering_node = lambda state: {"gathered_info": {"source": "info"}}
        nodes.self_reflection_node = lambda state: {"reflection_result": {"source": "reflect"}}

        result = nodes.parallel_info_and_reflect_node(
            {
                "is_first_step": False,
                "step_id": 0,
                "step_count": 0,
            }
        )

        self.assertEqual(runtime_memory.scope_calls, 2)
        self.assertEqual(result["gathered_info"], {"source": "info"})
        self.assertEqual(result["reflection_result"], {"source": "reflect"})


if __name__ == "__main__":
    unittest.main()
