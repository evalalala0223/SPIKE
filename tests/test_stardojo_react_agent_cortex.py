from __future__ import annotations

import ast
from copy import deepcopy
import os
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, cast
import unittest
from unittest import mock

from stardojo.utils.cortex_runtime_utils import (
    build_runtime_local_recovery_action,
    record_cortex_planning_latency,
    record_cortex_no_execution,
    reset_cortex_no_execution_watchdog,
    resolve_cortex_executable_actions,
    should_treat_cortex_attempt_as_first_step,
    should_initialize_cortex_state,
    validate_runtime_pre_execution_action,
)
from stardojo.utils.execution_feedback_utils import (
    execution_has_no_confirmation,
    infer_execution_success_raw,
    stable_snapshot_text,
)
from stardojo.utils.llm_timing_utils import (
    consume_llm_retry_overhead_s,
    reset_llm_retry_timing_accounting,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "agent" / "stardojo" / "stardojo_react_agent.py"


def _load_run_cortex_planning():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))

    function_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PipelineRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_run_cortex_planning":
                    function_node = item
                    break
            break

    if function_node is None:
        raise AssertionError("_run_cortex_planning not found in PipelineRunner")

    extracted_module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(extracted_module)

    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "cast": cast,
        "deepcopy": deepcopy,
        "create_initial_state": lambda **kwargs: {},
        "config": SimpleNamespace(
            work_dir=".",
            env_short_name="stardew",
            env_config={},
            number_of_execute_skills=1,
        ),
        "c_constants": None,
        "logger": SimpleNamespace(write=lambda *args, **kwargs: None, warn=lambda *args, **kwargs: None),
        "should_initialize_cortex_state": should_initialize_cortex_state,
        "should_treat_cortex_attempt_as_first_step": should_treat_cortex_attempt_as_first_step,
        "record_cortex_planning_latency": record_cortex_planning_latency,
        "reset_cortex_no_execution_watchdog": reset_cortex_no_execution_watchdog,
        "resolve_cortex_executable_actions": resolve_cortex_executable_actions,
        "record_cortex_no_execution": record_cortex_no_execution,
        "validate_runtime_pre_execution_action": validate_runtime_pre_execution_action,
        "validate_cultivation_pre_execution_action": lambda **kwargs: {"is_valid": True},
        "reset_llm_retry_timing_accounting": reset_llm_retry_timing_accounting,
        "consume_llm_retry_overhead_s": consume_llm_retry_overhead_s,
        "time": __import__("time"),
    }
    exec(compile(extracted_module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["_run_cortex_planning"]


def _load_update_execution_feedback():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))

    function_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PipelineRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "update_execution_feedback":
                    function_node = item
                    break
            break

    if function_node is None:
        raise AssertionError("update_execution_feedback not found in PipelineRunner")

    extracted_module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(extracted_module)

    namespace = {
        "Any": Any,
        "Dict": Dict,
        "List": List,
        "Optional": Optional,
        "cast": cast,
        "config": SimpleNamespace(max_recent_steps=8),
        "stable_snapshot_text": stable_snapshot_text,
        "execution_has_no_confirmation": execution_has_no_confirmation,
        "execution_observation_confirms_change": lambda *_args, **_kwargs: False,
        "extract_stardew_prompt_fact_fields": lambda **_kwargs: {},
        "build_runtime_local_recovery_action": build_runtime_local_recovery_action,
        "infer_execution_success_raw": infer_execution_success_raw,
        "reset_cortex_no_execution_watchdog": reset_cortex_no_execution_watchdog,
        "logger": SimpleNamespace(write=lambda *args, **kwargs: None, warn=lambda *args, **kwargs: None),
    }
    exec(compile(extracted_module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["update_execution_feedback"]


def _load_prepare_big_brain_template_for_call():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))

    function_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PipelineRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_prepare_big_brain_template_for_call":
                    function_node = item
                    break
            break

    if function_node is None:
        raise AssertionError("_prepare_big_brain_template_for_call not found in PipelineRunner")

    extracted_module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(extracted_module)

    namespace = {
        "logger": SimpleNamespace(write=lambda *args, **kwargs: None, warn=lambda *args, **kwargs: None),
    }
    exec(compile(extracted_module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["_prepare_big_brain_template_for_call"]


def _load_pipeline_shutdown():
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(SOURCE_PATH))

    function_node = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "PipelineRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "pipeline_shutdown":
                    function_node = item
                    break
            break

    if function_node is None:
        raise AssertionError("pipeline_shutdown not found in PipelineRunner")

    extracted_module = ast.Module(body=[function_node], type_ignores=[])
    ast.fix_missing_locations(extracted_module)

    namespace = {
        "os": os,
        "re": re,
        "config": SimpleNamespace(work_dir="."),
        "logger": SimpleNamespace(
            write=lambda *args, **kwargs: None,
            warn=lambda *args, **kwargs: None,
            work_dir=None,
            log_dir="",
        ),
        "process_log_messages": lambda work_dir: "hello",
        "replace_unsupported_chars": lambda text: text,
    }
    exec(compile(extracted_module, str(SOURCE_PATH), "exec"), namespace)
    return namespace["pipeline_shutdown"], namespace["config"], namespace["logger"]


_RUN_CORTEX_PLANNING = _load_run_cortex_planning()
_UPDATE_EXECUTION_FEEDBACK = _load_update_execution_feedback()
_PREPARE_BIG_BRAIN_TEMPLATE_FOR_CALL = _load_prepare_big_brain_template_for_call()
_PIPELINE_SHUTDOWN, _PIPELINE_SHUTDOWN_CONFIG, _PIPELINE_SHUTDOWN_LOGGER = _load_pipeline_shutdown()


class _BlockedNoExecutionController:
    def step(self, state: dict, workflow_config: dict | None = None) -> dict:
        return {
            "brain_mode": "little",
            "suggestions": [],
            "planned_actions": [],
            "escalation_reason": "little_brain_autonomous_fallback_failed:api_error:503",
            "current_step": 0,
        }


class _SuggestionOnlyController:
    def step(self, state: dict, workflow_config: dict | None = None) -> dict:
        return {
            "brain_mode": "big",
            "suggestions": [{"action": 'move(x=0, y=1)', "reason": "advance"}],
            "planned_actions": [],
            "allow_suggestion_execution_fallback": False,
            "current_step": 0,
        }


class _PendingExternalActionController:
    def step(self, state: dict, workflow_config: dict | None = None) -> dict:
        return {
            "brain_mode": "little",
            "suggestions": [{"action": "choose_item(slot_index=4)", "reason": "equip scythe"}],
            "planned_actions": ["choose_item(slot_index=4)"],
            "execution_pending": True,
            "pending_action": "choose_item(slot_index=4)",
            "pending_step_index": 0,
            "pending_suggested_action": "choose_item(slot_index=4)",
            "has_execution_feedback": False,
            "current_step": 1,
        }


class _Harness:
    _run_cortex_planning = _RUN_CORTEX_PLANNING
    update_execution_feedback = _UPDATE_EXECUTION_FEEDBACK
    _prepare_big_brain_template_for_call = _PREPARE_BIG_BRAIN_TEMPLATE_FOR_CALL
    pipeline_shutdown = _PIPELINE_SHUTDOWN
    _sync_big_brain_templates = staticmethod(lambda reason="": None)
    _ensure_big_brain_template_integrity = staticmethod(lambda key: None)


class TestStardojoReactAgentCortex(unittest.TestCase):
    def test_prepare_big_brain_template_for_call_refreshes_and_verifies_supported_keys(self) -> None:
        agent = _Harness()
        call_order: list[str] = []
        agent._sync_big_brain_templates = lambda reason="": call_order.append(f"sync:{reason}")
        agent._ensure_big_brain_template_integrity = lambda key: call_order.append(f"ensure:{key}")

        agent._prepare_big_brain_template_for_call("task_inference")
        agent._prepare_big_brain_template_for_call("action_planning")
        agent._prepare_big_brain_template_for_call("self_reflection")

        self.assertEqual(
            call_order,
            [
                "sync:",
                "ensure:task_inference",
                "sync:",
                "ensure:action_planning",
            ],
        )

    def test_cradle_planner_provider_uses_template_prepare_hook(self) -> None:
        source = SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn(
            'self.runner._prepare_big_brain_template_for_call(self.method_name)',
            source,
        )

    def test_stardew_task_inference_provider_uses_template_prepare_hook(self) -> None:
        source = SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn(
            'self.runner._prepare_big_brain_template_for_call("task_inference")',
            source,
        )

    def test_cortex_workflow_wires_runner_runtime_memory(self) -> None:
        source = SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "runtime_memory=self._get_cortex_runtime_memory()",
            source,
        )

    def test_decision_only_skill_execute_provider_accepts_injected_memory(self) -> None:
        source = SOURCE_PATH.read_text(encoding="utf-8")

        self.assertIn(
            "DecisionOnlySkillExecuteProvider(\n                memory=self._get_cortex_runtime_memory(),",
            source,
        )

    @staticmethod
    def _make_update_feedback_harness() -> _Harness:
        agent = _Harness()
        agent.dual_brain_controller = SimpleNamespace(
            little_brain=SimpleNamespace(execute_internally=False, execution_log=[])
        )
        agent._runtime_stop_signal = None
        agent.task_description = "go_to_bed"
        agent.initial_subtask_description = ""
        agent.initial_subtask_reasoning = ""
        agent.memory = SimpleNamespace(update_info_history=lambda payload: None)
        agent.cortex_memory = None
        agent._normalize_action_text = lambda action: str(action or "").strip()
        agent._normalize_position = lambda position: tuple(position) if isinstance(position, list) else position
        agent._extract_item_name = lambda item: str(item or "").strip()
        agent._infer_facing_position = lambda position, direction: ""
        agent._sync_external_little_brain_feedback = lambda **kwargs: None
        agent._extract_progress_quantity = (
            lambda task_eval: task_eval.get("quantity") if isinstance(task_eval, dict) else None
        )
        agent._classify_cultivation_failure = lambda **kwargs: {
            "failure_root_cause": "",
            "required_change_type": "",
            "failure_signature": "",
        }
        agent._build_action_feedback_text = lambda **kwargs: kwargs.get("latest_execution_summary", "")
        agent._is_productive_action = (
            lambda action_text: str(action_text or "").strip().lower().startswith(
                ("use(", "interact(", "choose_option(", "craft(")
            )
        )
        agent._obs_to_gathered_info = lambda obs: dict(obs)
        agent._format_toolbar_text = lambda inventory, chosen_item: ""
        agent._infer_selected_position = lambda inventory, chosen_item: None
        agent._resolve_current_subtask_values = lambda: {
            "subtask_description": "",
            "subtask_reasoning": "",
        }
        agent._sanitize_subtask_memory = lambda *args, **kwargs: {
            "sanitized_subtask_hint": "",
            "redundant_tool_selection": False,
            "selected_item_already_correct": False,
        }
        agent._count_same_action_tail = lambda actions, action: 1
        agent._detect_oscillation = lambda actions: 0
        agent._detect_position_issue = lambda *args, **kwargs: False
        agent._build_lightweight_execution_summary = lambda **kwargs: ""
        agent._build_task_progress_summary = lambda **kwargs: ""
        agent._get_recent_or_default = lambda key, default=None: default
        return agent

    def test_run_cortex_planning_keeps_no_execution_as_non_executable_and_updates_counters(self) -> None:
        agent = _Harness()
        agent.dual_brain_controller = _BlockedNoExecutionController()
        agent._cortex_workflow_config = {}
        agent._latest_obs = None
        agent._cortex_state = {
            "task": "clear_5_stone_with_pickaxe",
            "main_task": "clear_5_stone_with_pickaxe",
            "subtask_description": "step off the porch and align with the nearest stone",
            "subtask_reasoning": "blocked directly below the porch",
        }
        agent.task_description = "clear_5_stone_with_pickaxe"
        agent.initial_subtask_description = "step off the porch"
        agent.initial_subtask_reasoning = "align with the nearest stone"
        agent._skill_library_json = "[]"
        agent.task_acquisition_context = {}
        agent._runtime_stop_signal = None
        agent.cortex_memory = None
        agent.memory = object()
        agent._pick_latest_available_image_path = (
            lambda obs, preferred_path="", step_num=0: "shot_0.png"
        )
        agent._sync_stardew_memory_from_obs = (
            lambda obs, step_num, latest_image="": {}
        )
        agent._obs_to_gathered_info = lambda obs: {}
        agent._normalize_cortex_action = lambda action: str(action or "").strip()
        agent._resolve_current_subtask_values = lambda: {
            "subtask_description": "step off the porch and align with the nearest stone",
            "subtask_reasoning": "blocked directly below the porch",
        }
        agent._select_cortex_suggestion_for_logging = lambda **kwargs: None

        planned_actions = agent._run_cortex_planning(obs={}, step_num=0)

        self.assertEqual(planned_actions, [])
        self.assertEqual(agent._cortex_state["planning_attempt_count"], 1)
        self.assertEqual(agent._cortex_state["no_execution_return_count"], 1)
        self.assertEqual(agent._cortex_state["blocked_replan_count"], 0)
        self.assertTrue(agent._cortex_state["force_big_brain_replan"])

    def test_run_cortex_planning_executes_validated_current_step_suggestion_fallback(self) -> None:
        agent = _Harness()
        agent.dual_brain_controller = _SuggestionOnlyController()
        agent._cortex_workflow_config = {}
        agent._latest_obs = None
        agent._cortex_state = {
            "task": "go_to_bus_stop",
            "main_task": "go_to_bus_stop",
            "subtask_description": "walk toward the farm exit",
            "subtask_reasoning": "the next grounded step is movement",
        }
        agent.task_description = "go_to_bus_stop"
        agent.initial_subtask_description = "walk toward the farm exit"
        agent.initial_subtask_reasoning = "the next grounded step is movement"
        agent._skill_library_json = "[]"
        agent.task_acquisition_context = {}
        agent._runtime_stop_signal = None
        agent.cortex_memory = None
        agent.memory = object()
        agent._pick_latest_available_image_path = (
            lambda obs, preferred_path="", step_num=0: "shot_0.png"
        )
        agent._sync_stardew_memory_from_obs = (
            lambda obs, step_num, latest_image="": {}
        )
        agent._obs_to_gathered_info = lambda obs: {"surroundings": ""}
        agent._normalize_cortex_action = lambda action: str(action or "").strip()
        agent._resolve_current_subtask_values = lambda: {
            "subtask_description": "walk toward the farm exit",
            "subtask_reasoning": "the next grounded step is movement",
        }
        agent._select_cortex_suggestion_for_logging = lambda **kwargs: None

        planned_actions = agent._run_cortex_planning(obs={}, step_num=0)

        self.assertEqual(planned_actions, ['move(x=0, y=1)'])
        self.assertEqual(agent._cortex_state["planning_attempt_count"], 1)
        self.assertEqual(agent._cortex_state["no_execution_return_count"], 0)

    def test_run_cortex_planning_surfaces_pending_external_action_for_env_execution(self) -> None:
        agent = _Harness()
        agent.dual_brain_controller = _PendingExternalActionController()
        agent._cortex_workflow_config = {}
        agent._latest_obs = None
        agent._cortex_state = {
            "task": "clear_10_weeds_with_scythe",
            "main_task": "clear_10_weeds_with_scythe",
            "subtask_description": "switch to the scythe before moving toward the nearby weeds",
            "subtask_reasoning": "the axe is still selected",
        }
        agent.task_description = "clear_10_weeds_with_scythe"
        agent.initial_subtask_description = "switch to the scythe before moving toward the nearby weeds"
        agent.initial_subtask_reasoning = "the axe is still selected"
        agent._skill_library_json = "[]"
        agent.task_acquisition_context = {}
        agent._runtime_stop_signal = None
        agent.cortex_memory = None
        agent.memory = object()
        agent._pick_latest_available_image_path = (
            lambda obs, preferred_path="", step_num=0: "shot_0.png"
        )
        agent._sync_stardew_memory_from_obs = (
            lambda obs, step_num, latest_image="": {}
        )
        agent._obs_to_gathered_info = lambda obs: {"surroundings": ""}
        agent._normalize_cortex_action = lambda action: str(action or "").strip()
        agent._resolve_current_subtask_values = lambda: {
            "subtask_description": "switch to the scythe before moving toward the nearby weeds",
            "subtask_reasoning": "the axe is still selected",
        }
        agent._select_cortex_suggestion_for_logging = lambda **kwargs: None

        planned_actions = agent._run_cortex_planning(obs={}, step_num=0)

        self.assertEqual(planned_actions, ["choose_item(slot_index=4)"])
        self.assertTrue(agent._cortex_state["execution_pending"])
        self.assertEqual(agent._cortex_state["pending_action"], "choose_item(slot_index=4)")
        self.assertEqual(agent._cortex_state["planning_attempt_count"], 1)
        self.assertEqual(agent._cortex_state["no_execution_return_count"], 0)

    def test_update_execution_feedback_keeps_remaining_plan_after_successful_external_step(self) -> None:
        agent = self._make_update_feedback_harness()
        agent._cortex_state = {
            "gathered_info": {"position": [0, 0], "current_menu": "No Menu", "inventory": []},
            "task_progress_quantity": 0,
            "execution_pending": True,
            "pending_action": "move(x=1, y=0)",
            "pending_step_index": 0,
            "pending_suggested_action": "move(x=1, y=0)",
            "suggestions": [
                {"action": "move(x=1, y=0)", "reason": "advance"},
                {"action": 'use(direction="down")', "reason": "clear"},
            ],
            "planned_actions": ["move(x=1, y=0)"],
            "current_step": 1,
            "completed_steps": [],
            "brain_mode": "little",
            "consecutive_failures": 0,
            "previous_actions": [],
            "previous_results": [],
            "history_summary": "",
        }

        agent.update_execution_feedback(
            exec_info={
                "errors": False,
                "errors_info": "",
                "executed_skills": ["move(x=1, y=0)"],
                "last_skill": "move(x=1, y=0)",
            },
            action="move(x=1, y=0)",
            obs={"position": [1, 0], "inventory": [], "chosen_item": "", "current_menu": "No Menu"},
            task_eval={"quantity": 0, "completed": False},
        )

        self.assertFalse(agent._cortex_state["execution_pending"])
        self.assertFalse(agent._cortex_state["force_big_brain_replan"])
        self.assertEqual(len(agent._cortex_state["suggestions"]), 2)
        self.assertEqual(agent._cortex_state["current_step"], 1)
        self.assertEqual(agent._cortex_state["completed_steps"], [0])
        self.assertEqual(agent._cortex_state["brain_mode"], "little")
        self.assertEqual(agent._cortex_state["planned_actions"], [])
        self.assertEqual(agent._cortex_state["executed_step_count"], 1)

    def test_update_execution_feedback_forces_replan_after_failed_external_step(self) -> None:
        agent = self._make_update_feedback_harness()
        agent._cortex_state = {
            "gathered_info": {"position": [0, 0], "current_menu": "No Menu", "inventory": []},
            "task_progress_quantity": 0,
            "execution_pending": True,
            "pending_action": "move(x=1, y=0)",
            "pending_step_index": 0,
            "pending_suggested_action": "move(x=1, y=0)",
            "suggestions": [
                {"action": "move(x=1, y=0)", "reason": "advance"},
                {"action": 'use(direction="down")', "reason": "clear"},
            ],
            "planned_actions": ["move(x=1, y=0)"],
            "current_step": 1,
            "completed_steps": [],
            "brain_mode": "little",
            "consecutive_failures": 0,
            "previous_actions": [],
            "previous_results": [],
            "history_summary": "",
        }

        agent.update_execution_feedback(
            exec_info={
                "errors": True,
                "errors_info": "path is likely blocked by an obstacle",
                "executed_skills": ["move(x=1, y=0)"],
                "last_skill": "move(x=1, y=0)",
            },
            action="move(x=1, y=0)",
            obs={"position": [0, 0], "inventory": [], "chosen_item": "", "current_menu": "No Menu"},
            task_eval={"quantity": 0, "completed": False},
        )

        self.assertFalse(agent._cortex_state["execution_pending"])
        self.assertTrue(agent._cortex_state["force_big_brain_replan"])
        self.assertEqual(agent._cortex_state["current_step"], 0)
        self.assertEqual(agent._cortex_state["completed_steps"], [])
        self.assertEqual(agent._cortex_state["brain_mode"], "big")
        self.assertEqual(agent._cortex_state["planned_actions"], [])

    def test_update_execution_feedback_escalates_repeated_choose_item_no_confirmation(self) -> None:
        agent = self._make_update_feedback_harness()
        agent._cortex_state = {
            "gathered_info": {"position": [0, 0], "current_menu": "No Menu", "inventory": []},
            "task_progress_quantity": None,
            "execution_pending": True,
            "pending_action": "choose_item(slot_index=3)",
            "pending_step_index": 0,
            "pending_suggested_action": "choose_item(slot_index=3)",
            "suggestions": [{"action": "choose_item(slot_index=3)", "reason": "equip hoe"}],
            "planned_actions": ["choose_item(slot_index=3)"],
            "current_step": 1,
            "completed_steps": [],
            "brain_mode": "little",
            "consecutive_failures": 1,
            "previous_actions": ["choose_item(slot_index=3)"],
            "previous_results": [
                {
                    "action": "choose_item(slot_index=3)",
                    "state_changed": False,
                    "progress_delta": None,
                    "progress_quantity": None,
                    "uncertain_execution": True,
                    "inventory_changed": False,
                }
            ],
            "history_summary": "",
        }

        agent.update_execution_feedback(
            exec_info={
                "errors": False,
                "errors_info": "returned no confirmation",
                "executed_skills": ["choose_item(slot_index=3)"],
                "last_skill": "choose_item(slot_index=3)",
            },
            action="choose_item(slot_index=3)",
            obs={"position": [0, 0], "inventory": [], "chosen_item": "", "current_menu": "No Menu"},
            task_eval={"quantity": None, "completed": False},
        )

        self.assertTrue(agent._cortex_state["heightened_failure_signal"])
        self.assertEqual(agent._cortex_state["consecutive_failures"], 3)
        self.assertTrue(agent._cortex_state["force_big_brain_replan"])

    def test_update_execution_feedback_escalates_repeated_use_no_confirmation_without_progress(self) -> None:
        agent = self._make_update_feedback_harness()
        agent._cortex_state = {
            "gathered_info": {"position": [0, 0], "current_menu": "No Menu", "inventory": []},
            "task_progress_quantity": 0,
            "execution_pending": True,
            "pending_action": 'use(direction="right")',
            "pending_step_index": 0,
            "pending_suggested_action": 'use(direction="right")',
            "suggestions": [{"action": 'use(direction="right")', "reason": "till"}],
            "planned_actions": ['use(direction="right")'],
            "current_step": 1,
            "completed_steps": [],
            "brain_mode": "little",
            "consecutive_failures": 1,
            "previous_actions": ['use(direction="right")'],
            "previous_results": [
                {
                    "action": 'use(direction="right")',
                    "state_changed": False,
                    "inventory_changed": False,
                    "progress_delta": 0,
                    "progress_quantity": 0,
                    "uncertain_execution": False,
                }
            ],
            "history_summary": "",
        }

        agent.update_execution_feedback(
            exec_info={
                "errors": False,
                "errors_info": "returned no confirmation",
                "executed_skills": ['use(direction="right")'],
                "last_skill": 'use(direction="right")',
            },
            action='use(direction="right")',
            obs={"position": [0, 0], "inventory": [], "chosen_item": "", "current_menu": "No Menu"},
            task_eval={"quantity": 0, "completed": False},
        )

        self.assertTrue(agent._cortex_state["heightened_failure_signal"])
        self.assertEqual(agent._cortex_state["consecutive_failures"], 3)
        self.assertEqual(agent._cortex_state["pending_local_recovery_action"], 'move(x=0, y=1)')
        self.assertEqual(
            agent._cortex_state["pending_local_recovery_reason"],
            "repeated_interaction_no_confirmation",
        )
        self.assertFalse(agent._cortex_state["force_big_brain_replan"])
        self.assertEqual(agent._cortex_state["brain_mode"], "little")

    def test_update_execution_feedback_does_not_raise_repeated_use_signal_after_progress(self) -> None:
        agent = self._make_update_feedback_harness()
        agent._cortex_state = {
            "gathered_info": {"position": [0, 0], "current_menu": "No Menu", "inventory": []},
            "task_progress_quantity": 0,
            "execution_pending": True,
            "pending_action": 'use(direction="right")',
            "pending_step_index": 0,
            "pending_suggested_action": 'use(direction="right")',
            "suggestions": [{"action": 'use(direction="right")', "reason": "till"}],
            "planned_actions": ['use(direction="right")'],
            "current_step": 1,
            "completed_steps": [],
            "brain_mode": "little",
            "consecutive_failures": 1,
            "previous_actions": ['use(direction="right")'],
            "previous_results": [
                {
                    "action": 'use(direction="right")',
                    "state_changed": False,
                    "inventory_changed": False,
                    "progress_delta": 0,
                    "progress_quantity": 0,
                    "uncertain_execution": False,
                }
            ],
            "history_summary": "",
        }

        agent.update_execution_feedback(
            exec_info={
                "errors": False,
                "errors_info": "returned no confirmation",
                "executed_skills": ['use(direction="right")'],
                "last_skill": 'use(direction="right")',
            },
            action='use(direction="right")',
            obs={"position": [1, 0], "inventory": [], "chosen_item": "", "current_menu": "No Menu"},
            task_eval={"quantity": 1, "completed": False},
        )

        self.assertFalse(agent._cortex_state["heightened_failure_signal"])
        self.assertEqual(agent._cortex_state["consecutive_failures"], 0)
        self.assertFalse(agent._cortex_state["force_big_brain_replan"])
        self.assertEqual(agent._cortex_state["pending_local_recovery_action"], "")

    def test_update_execution_feedback_syncs_final_failure_state_to_little_brain(self) -> None:
        agent = self._make_update_feedback_harness()
        synced = {}
        agent._sync_external_little_brain_feedback = lambda **kwargs: synced.update(kwargs)
        agent._cortex_state = {
            "gathered_info": {"position": [0, 0], "current_menu": "No Menu", "inventory": []},
            "task_progress_quantity": 0,
            "execution_pending": True,
            "pending_action": 'use(direction="right")',
            "pending_step_index": 0,
            "pending_suggested_action": 'use(direction="right")',
            "suggestions": [{"action": 'use(direction="right")', "reason": "till"}],
            "planned_actions": ['use(direction="right")'],
            "current_step": 1,
            "completed_steps": [],
            "brain_mode": "little",
            "consecutive_failures": 1,
            "previous_actions": ['use(direction="right")'],
            "previous_results": [
                {
                    "action": 'use(direction="right")',
                    "state_changed": False,
                    "inventory_changed": False,
                    "progress_delta": 0,
                    "progress_quantity": 0,
                    "uncertain_execution": False,
                }
            ],
            "history_summary": "",
        }

        agent.update_execution_feedback(
            exec_info={
                "errors": False,
                "errors_info": "returned no confirmation",
                "executed_skills": ['use(direction="right")'],
                "last_skill": 'use(direction="right")',
            },
            action='use(direction="right")',
            obs={"position": [0, 0], "inventory": [], "chosen_item": "", "current_menu": "No Menu"},
            task_eval={"quantity": 0, "completed": False},
        )

        self.assertFalse(synced["success"])
        self.assertFalse(synced["state_changed"])
        self.assertTrue(synced["uncertain_execution"])
        self.assertTrue(synced["heightened_failure_signal"])
        self.assertEqual(synced["progress_delta"], 0)
        self.assertEqual(synced["progress_quantity"], 0)

    def test_update_execution_feedback_treats_first_choose_item_as_state_change(self) -> None:
        agent = self._make_update_feedback_harness()
        synced = {}
        agent._sync_external_little_brain_feedback = lambda **kwargs: synced.update(kwargs)
        agent._cortex_state = {
            "gathered_info": {
                "position": [0, 0],
                "current_menu": "No Menu",
                "inventory": [],
                "selected_item_name": "",
                "chosen_item": "",
            },
            "selected_position": None,
            "task_progress_quantity": None,
            "execution_pending": True,
            "pending_action": "choose_item(slot_index=3)",
            "pending_step_index": 0,
            "pending_suggested_action": "choose_item(slot_index=3)",
            "suggestions": [{"action": "choose_item(slot_index=3)", "reason": "equip hoe"}],
            "planned_actions": ["choose_item(slot_index=3)"],
            "current_step": 1,
            "completed_steps": [],
            "brain_mode": "little",
            "consecutive_failures": 0,
            "previous_actions": [],
            "previous_results": [],
            "history_summary": "",
        }

        agent.update_execution_feedback(
            exec_info={
                "errors": False,
                "errors_info": "returned no confirmation",
                "executed_skills": ["choose_item(slot_index=3)"],
                "last_skill": "choose_item(slot_index=3)",
            },
            action="choose_item(slot_index=3)",
            obs={
                "position": [0, 0],
                "inventory": [],
                "chosen_item": "Hoe",
                "current_menu": "No Menu",
            },
            task_eval={"quantity": None, "completed": False},
        )

        self.assertTrue(agent._cortex_state["last_state_changed"])
        self.assertFalse(agent._cortex_state["uncertain_execution"])
        self.assertEqual(agent._cortex_state["zero_progress_streak"], 0)
        self.assertTrue(synced["state_changed"])
        self.assertTrue(synced["success"])

    def test_pipeline_shutdown_sanitizes_task_name_for_markdown_log(self) -> None:
        agent = _Harness()
        agent._shutdown_done = False
        agent.gm = SimpleNamespace(cleanup_io=lambda: None)
        agent.task_description = 'complete_the_story_quest_"introductions"'
        fake_work_dir = str(ROOT / "fake_run_dir")
        fake_log_path = os.path.join(fake_work_dir, "logs", "stardojo.log")
        expected_md_path = os.path.join(
            fake_work_dir,
            "logs",
            "complete_the_story_quest__introductions_log.md",
        )

        _PIPELINE_SHUTDOWN_CONFIG.work_dir = fake_work_dir
        _PIPELINE_SHUTDOWN_LOGGER.work_dir = fake_work_dir
        _PIPELINE_SHUTDOWN_LOGGER.log_dir = fake_log_path

        with mock.patch("os.path.exists", side_effect=lambda path: path == fake_log_path):
            with mock.patch("os.makedirs") as mocked_makedirs:
                with mock.patch("builtins.open", mock.mock_open()) as mocked_open:
                    agent.pipeline_shutdown()

        mocked_makedirs.assert_called_once()
        mocked_open.assert_called_once()
        self.assertEqual(mocked_open.call_args.args[0], expected_md_path)


if __name__ == "__main__":
    unittest.main()
