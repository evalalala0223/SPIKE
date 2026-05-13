from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "env" / "llm_env_multi_tasks_parallel.py"


def _load_parallel_error_helpers():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))

    methods = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "StarDojoLLM":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in {
                    "_build_assigned_task_meta",
                    "_safe_build_task_meta",
                    "_build_terminal_error_info",
                    "_build_task_transition_info",
                    "_has_pending_claimed_task",
                    "_clear_runtime_state_for_claimed_task",
                }:
                    methods[item.name] = item
            break

    if set(methods) != {
        "_build_assigned_task_meta",
        "_safe_build_task_meta",
        "_build_terminal_error_info",
        "_build_task_transition_info",
        "_has_pending_claimed_task",
        "_clear_runtime_state_for_claimed_task",
    }:
        raise AssertionError("Expected StarDojoLLM error-helper methods were not found")

    extracted_module = ast.Module(
        body=[
            methods["_build_assigned_task_meta"],
            methods["_safe_build_task_meta"],
            methods["_build_terminal_error_info"],
            methods["_build_task_transition_info"],
            methods["_has_pending_claimed_task"],
            methods["_clear_runtime_state_for_claimed_task"],
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(extracted_module)

    namespace = {
        "Any": Any,
        "Dict": Dict,
        "Optional": Optional,
        "get_llm_call_count": lambda: 7,
        "get_llm_call_breakdown": lambda: {"big_brain": 4, "little_brain": 3},
        "time": SimpleNamespace(time=lambda: 123.456),
    }
    exec(compile(extracted_module, str(SOURCE_PATH), "exec"), namespace)
    return (
        namespace["_build_assigned_task_meta"],
        namespace["_safe_build_task_meta"],
        namespace["_build_terminal_error_info"],
        namespace["_build_task_transition_info"],
        namespace["_has_pending_claimed_task"],
        namespace["_clear_runtime_state_for_claimed_task"],
    )


(
    _BUILD_ASSIGNED_TASK_META,
    _SAFE_BUILD_TASK_META,
    _BUILD_TERMINAL_ERROR_INFO,
    _BUILD_TASK_TRANSITION_INFO,
    _HAS_PENDING_CLAIMED_TASK,
    _CLEAR_RUNTIME_STATE_FOR_CLAIMED_TASK,
) = _load_parallel_error_helpers()


class _Harness:
    _build_assigned_task_meta = _BUILD_ASSIGNED_TASK_META
    _safe_build_task_meta = _SAFE_BUILD_TASK_META
    _build_terminal_error_info = _BUILD_TERMINAL_ERROR_INFO
    _build_task_transition_info = _BUILD_TASK_TRANSITION_INFO
    _has_pending_claimed_task = _HAS_PENDING_CLAIMED_TASK
    _clear_runtime_state_for_claimed_task = _CLEAR_RUNTIME_STATE_FOR_CLAIMED_TASK


class TestParallelEnvErrorPayloads(unittest.TestCase):
    def test_parallel_runner_keeps_active_task_identity_for_worker_failures(self) -> None:
        source = SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn("env_active_task_index", source)
        self.assertIn("worker_recv_error", source)
        self.assertIn("worker_protocol_error", source)
        self.assertIn("startup_timeout_quarantine_threshold", source)
        self.assertIn("_is_worker_startup_timeout_failure", source)
        self.assertIn("_update_worker_startup_timeout_streak", source)
        self.assertIn("startup_timeout_failure_streak", source)
        self.assertIn("worker_quarantined", source)
        self.assertIn("task reset retry pending", source)
        self.assertIn('runtime_exit_reason="reset_error"', source)
        self.assertIn('info_obj.get("runtime_exit_reason")', source)
        self.assertIn("_build_perf_summary", source)
        self.assertIn('result["perf_summary"] = _build_perf_summary(steps)', source)
        self.assertIn('"set_agent_sec"', source)
        self.assertIn('"reset_sec"', source)
        self.assertIn('"freeze_before_plan_sec"', source)
        self.assertIn("ensure_stardew_window_preferences", source)
        self.assertIn("self._clear_runtime_state_for_claimed_task()", source)
        self.assertIn("existing_server_ready", source)
        self.assertIn("timeout_s=1.0", source)

    def test_safe_build_task_meta_falls_back_to_local_fields_when_budget_builder_raises(self) -> None:
        env = _Harness()
        env.env_id = 2
        env.port = 10784
        env.task_name = "farming_lite"
        env.task_id = 6
        env.task_queue_index = 13
        env.task = SimpleNamespace(llm_description="clear_5_stone_with_pickaxe", difficulty="easy")
        env.experiment_budget_mode = "benchmark_steps"
        env.current_task_started_at = 99.0
        env.agent_run_dir_name = "10784_farming_lite_6_123.0"
        env.planner_comp_model = "Qwen/Qwen3.5-397B-A17B-FP8"
        env.embedding_model = "BAAI/bge-base-en-v1.5"
        env.prompt_profile = "farm_clearup"
        env.resolved_action_planning_template = "./res/stardew/prompts/templates/action_planning_farm_clearup.prompt"
        env.resolved_task_inference_template = "./res/stardew/prompts/templates/task_inference_farm_clearup.prompt"
        env.max_turn_count = 30
        env.max_llm_calls = 80
        env.agent = SimpleNamespace(get_runtime_task_metrics=lambda: {"planning_attempt_count": 5})
        env._build_task_meta = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        env._get_effective_max_turn_count = lambda: 30
        env._get_effective_max_llm_calls = lambda: 80

        task_meta = env._safe_build_task_meta()

        self.assertEqual(task_meta["queue_index"], 13)
        self.assertEqual(task_meta["task_name"], "farming_lite")
        self.assertEqual(task_meta["task_id"], 6)
        self.assertEqual(task_meta["task_description"], "clear_5_stone_with_pickaxe")
        self.assertEqual(task_meta["llm_call_count"], 7)
        self.assertEqual(task_meta["planning_attempt_count"], 5)
        self.assertEqual(task_meta["planner_comp_model"], "Qwen/Qwen3.5-397B-A17B-FP8")

    def test_build_terminal_error_info_includes_task_identity_and_non_completed_eval(self) -> None:
        env = _Harness()
        env.last_action = 'move(x=0, y=1)'
        env._safe_build_task_meta = lambda: {
            "queue_index": 13,
            "task_name": "farming_lite",
            "task_id": 6,
        }

        info = env._build_terminal_error_info(
            error="worker crashed",
            runtime_exit_reason="step_exception",
            step_started=120.0,
            recovered=False,
        )

        self.assertEqual(info["error"], "worker crashed")
        self.assertEqual(info["runtime_exit_reason"], "step_exception")
        self.assertEqual(info["task_eval"], {"completed": False})
        self.assertEqual(info["task_meta"]["queue_index"], 13)
        self.assertEqual(info["task_meta"]["runtime_exit_reason"], "step_exception")
        self.assertFalse(info["recovered"])

    def test_build_task_transition_info_keeps_task_meta_and_no_execution_flags(self) -> None:
        env = _Harness()
        env._safe_build_task_meta = lambda: {
            "queue_index": 16,
            "task_name": "exploration_lite",
            "task_id": 4,
        }

        info = env._build_task_transition_info(
            "task reset retry pending: wait_game_start timeout",
            error="wait_game_start timeout",
        )

        self.assertEqual(info["task_meta"]["queue_index"], 16)
        self.assertTrue(info["no_execution"])
        self.assertTrue(info["task_transition"])
        self.assertIn("task reset retry pending", info["warning"])
        self.assertEqual(info["error"], "wait_game_start timeout")

    def test_has_pending_claimed_task_requires_queue_index_and_unfinished_state(self) -> None:
        env = _Harness()
        env.task_config = {"queue_index": 19}
        env.task_queue_index = 19
        env.current_task_finsh = False
        self.assertTrue(env._has_pending_claimed_task())

        env.current_task_finsh = True
        self.assertFalse(env._has_pending_claimed_task())

    def test_clear_runtime_state_for_claimed_task_resets_terminal_flags_before_retry(self) -> None:
        env = _Harness()
        env.agent = object()
        env.task = object()
        env.config = object()
        env.logger = object()
        env.skill_steps = ["move(x=1, y=0)"]
        env.obs = {"dummy": True}
        env.last_action = 'move(x=1, y=0)'
        env.terminated = True
        env.truncated = True
        env.step_num = 12
        env.consecutive_step_errors = 2
        env.task_budget = object()
        env.max_turn_count = 30
        env.max_llm_calls = 80
        env.current_task_started_at = None
        env.last_set_agent_duration_sec = 3.14
        env.agent_run_dir_name = "old_run"
        env.prompt_profile = "shopping"
        env.resolved_action_planning_template = "old_action"
        env.resolved_task_inference_template = "old_task"

        env._clear_runtime_state_for_claimed_task()

        self.assertIsNone(env.agent)
        self.assertIsNone(env.task)
        self.assertIsNone(env.config)
        self.assertIsNone(env.logger)
        self.assertFalse(env.terminated)
        self.assertFalse(env.truncated)
        self.assertEqual(env.step_num, 0)
        self.assertIsNone(env.agent_run_dir_name)
        self.assertIsNotNone(env.current_task_started_at)


if __name__ == "__main__":
    unittest.main()
