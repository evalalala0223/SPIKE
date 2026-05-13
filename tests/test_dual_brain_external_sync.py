from __future__ import annotations

import time
import unittest
from unittest import mock

from cradle.runner.big_brain import BigBrain, BrainPlanResult
from cradle.runner.dual_brain import DualBrainController
from cradle.runner.little_brain import LittleBrain
from cradle.runner.scheduler import BrainScheduler
from cradle.runner.vllm_client import VLLMDecision


class _SuggestionFollowingVLLMClient:
    def decide(self, **kwargs: object) -> VLLMDecision:
        suggestion = kwargs.get("suggestion", {}) or {}
        action = str(getattr(suggestion, "get", lambda *_: "")("action", "") or "")
        return VLLMDecision(action=action, reason="follow_suggestion", escalate=False)


class _EscalatingVLLMClient:
    def decide(self, **kwargs: object) -> VLLMDecision:
        return VLLMDecision(action="", reason="timeout", escalate=True)


class _Escalating503VLLMClient:
    def decide(self, **kwargs: object) -> VLLMDecision:
        return VLLMDecision(
            action="",
            reason=(
                "api_error: 503 Service Unavailable: reach max requests. "
                "current requests: 6 service support max requests: 5"
            ),
            escalate=True,
        )


class _HealthCheckProbeVLLMClient:
    def __init__(self, results: list[bool], health_check_timeout_s: float = 12.0) -> None:
        self.results = list(results)
        self.health_check_timeout_s = health_check_timeout_s
        self.calls: list[float | None] = []

    def health_check(self, timeout_s: float | None = None) -> bool:
        self.calls.append(timeout_s)
        return self.results.pop(0)


class _PendingLittleBrain:
    def __init__(self) -> None:
        self.calls = 0

    def execute(self, state: dict) -> dict:
        self.calls += 1
        return {
            "brain_mode": "little",
            "planned_actions": ['move(x=1, y=0)'],
            "execution_pending": True,
            "success": None,
            "has_execution_feedback": False,
        }

    def get_status(self) -> dict:
        return {}


class _EscalatingLittleBrain:
    def __init__(self, escalation_reason: str) -> None:
        self.escalation_reason = escalation_reason
        self.calls = 0

    @staticmethod
    def _sanitize_action(action: str) -> str:
        return str(action or "").strip()

    def execute(self, state: dict) -> dict:
        self.calls += 1
        return {
            "brain_mode": "big",
            "escalation_reason": self.escalation_reason,
            "suggestions": list(state.get("suggestions", [])),
            "current_step": int(state.get("current_step", 0) or 0),
            "completed_steps": list(state.get("completed_steps", [])),
            "execution_log": list(state.get("execution_log", [])),
            "has_execution_feedback": False,
            "execution_pending": False,
            "pending_action": "",
            "pending_step_index": None,
            "pending_suggested_action": "",
            "force_big_brain_replan": False,
        }

    def get_status(self) -> dict:
        return {}


class _FixedScheduler:
    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.increment_calls = 0

    def decide(self, state: dict) -> str:
        return self.mode

    def reset_counter(self) -> None:
        return None

    def increment_counter(self) -> None:
        self.increment_calls += 1

    def get_status(self) -> dict:
        return {}


class _NoOpEnvDetector:
    threshold = 0.35

    def detect_change(self, screenshot_path: str) -> tuple[bool, float]:
        return False, 0.0

    def reset(self) -> None:
        return None


class _NoOpFailureDetector:
    def __init__(self) -> None:
        self.calls = 0

    def evaluate(self, **kwargs: object):
        self.calls += 1
        return None

    def on_big_brain_success(self) -> None:
        return None

    def get_status(self) -> dict:
        return {}


class _QueuedBigBrain(BigBrain):
    def __init__(self, plan_results: list[BrainPlanResult]) -> None:
        super().__init__(workflow_app=None)
        self._plan_results = list(plan_results)
        self.plan_calls = 0

    def plan(self, state: dict, workflow_config: dict | None = None) -> tuple[dict, BrainPlanResult]:
        self.plan_calls += 1
        return {}, self._plan_results.pop(0)


class TestDualBrainExternalSync(unittest.TestCase):
    def test_definitively_stuck_does_not_trigger_on_productive_repeat_alone(self) -> None:
        controller = DualBrainController(
            workflow_app=None,
            scheduler=_FixedScheduler("big"),
            big_brain=_QueuedBigBrain([]),
            little_brain=_PendingLittleBrain(),
            env_detector=_NoOpEnvDetector(),
            failure_detector=_NoOpFailureDetector(),
            vllm_client=None,
            vllm_available=True,
        )

        self.assertFalse(
            controller._is_definitively_stuck(
                {
                    "zero_progress_streak": 0,
                    "repeated_action_streak": 4,
                    "oscillation_streak": 0,
                    "consecutive_failures": 0,
                }
            )
        )

    def test_scheduler_forces_big_brain_replan_after_external_feedback(self) -> None:
        scheduler = BrainScheduler(cycle_size=4, max_consecutive_failures=2)

        decision = scheduler.decide(
            {
                "is_first_step": False,
                "force_big_brain_replan": True,
                "vllm_available": True,
                "env_changed": False,
                "consecutive_failures": 0,
                "suggestions": [{"action": 'move(x=1, y=0)', "reason": "advance"}],
                "current_step": 0,
            }
        )

        self.assertEqual(decision, "big")

    def test_dual_brain_returns_pending_external_action_without_replanning(self) -> None:
        scheduler = _FixedScheduler("little")
        big_brain = _QueuedBigBrain([])
        little_brain = _PendingLittleBrain()
        failure_detector = _NoOpFailureDetector()
        controller = DualBrainController(
            workflow_app=None,
            scheduler=scheduler,
            big_brain=big_brain,
            little_brain=little_brain,
            env_detector=_NoOpEnvDetector(),
            failure_detector=failure_detector,
            vllm_client=None,
            vllm_available=True,
        )

        result = controller.step(
            {
                "suggestions": [{"action": 'move(x=1, y=0)', "reason": "advance"}],
                "current_step": 0,
                "is_first_step": False,
                "vllm_available": True,
                "has_execution_feedback": False,
                "consecutive_failures": 0,
                "last_state_changed": False,
            },
            workflow_config={},
        )

        self.assertTrue(result["execution_pending"])
        self.assertEqual(result["planned_actions"], ['move(x=1, y=0)'])
        self.assertEqual(big_brain.plan_calls, 0)
        self.assertEqual(little_brain.calls, 1)
        self.assertEqual(scheduler.increment_calls, 1)
        self.assertEqual(failure_detector.calls, 0)

    def test_external_execution_continues_remaining_plan_steps_before_replanning(self) -> None:
        first_plan = BrainPlanResult(
            suggestions=[
                {"action": "choose_item(slot_index=4)", "reason": "equip"},
                {"action": 'use(direction="down")', "reason": "clear"},
            ],
            context_summary="ctx1",
            current_task="task",
        )
        second_plan = BrainPlanResult(
            suggestions=[
                {"action": 'use(direction="down")', "reason": "clear"},
            ],
            context_summary="ctx2",
            current_task="task",
        )
        controller = DualBrainController(
            workflow_app=None,
            scheduler=BrainScheduler(cycle_size=4, max_consecutive_failures=2),
            big_brain=_QueuedBigBrain([first_plan, second_plan]),
            little_brain=LittleBrain(
                vllm_client=_SuggestionFollowingVLLMClient(),
                execute_internally=False,
            ),
            env_detector=_NoOpEnvDetector(),
            failure_detector=_NoOpFailureDetector(),
            vllm_client=None,
            vllm_available=True,
        )

        state = {
            "task": "task",
            "main_task": "task",
            "skill_library": "",
            "suggestions": [],
            "current_step": 0,
            "is_first_step": False,
            "vllm_available": True,
            "has_execution_feedback": False,
            "consecutive_failures": 0,
            "last_state_changed": False,
            "previous_actions": [],
            "previous_results": [],
            "zero_progress_streak": 0,
            "repeated_action_streak": 0,
            "position_issue_detected": False,
            "oscillation_streak": 0,
        }

        first_result = controller.step(state, workflow_config={})

        self.assertEqual(first_result["planned_actions"], ["choose_item(slot_index=4)"])
        self.assertTrue(first_result["execution_pending"])
        self.assertEqual(first_result["current_step"], 1)
        self.assertEqual(first_result["suggestions"], first_plan.suggestions)
        self.assertFalse(first_result["force_big_brain_replan"])

        feedback_state = {
            **state,
            **first_result,
            "execution_pending": False,
            "pending_action": "",
            "pending_step_index": None,
            "pending_suggested_action": "",
            "force_big_brain_replan": False,
            "planned_actions": [],
            "completed_steps": [0],
            "has_execution_feedback": True,
            "last_action": "choose_item(slot_index=4)",
            "last_exec_info": {
                "errors": False,
                "errors_info": "",
                "executed_skills": ["choose_item(slot_index=4)"],
                "last_skill": "choose_item(slot_index=4)",
            },
            "last_state_changed": True,
            "previous_actions": ["choose_item(slot_index=4)"],
            "previous_task_progress": 0,
            "task_progress": 0,
            "previous_task_progress_quantity": 0,
            "task_progress_quantity": 0,
        }

        second_result = controller.step(feedback_state, workflow_config={})

        self.assertEqual(second_result["planned_actions"], ['use(direction="down")'])
        self.assertTrue(second_result["execution_pending"])
        self.assertEqual(second_result["suggestions"], first_plan.suggestions)
        self.assertEqual(second_result["current_step"], 2)
        self.assertEqual(controller.big_brain.plan_calls, 1)
        self.assertFalse(second_result["force_big_brain_replan"])

        final_feedback_state = {
            **state,
            **second_result,
            "execution_pending": False,
            "pending_action": "",
            "pending_step_index": None,
            "pending_suggested_action": "",
            "force_big_brain_replan": False,
            "planned_actions": [],
            "completed_steps": [0, 1],
            "has_execution_feedback": True,
            "last_action": 'use(direction="down")',
            "last_exec_info": {
                "errors": False,
                "errors_info": "",
                "executed_skills": ['use(direction="down")'],
                "last_skill": 'use(direction="down")',
            },
            "last_state_changed": True,
            "previous_actions": ["choose_item(slot_index=4)", 'use(direction="down")'],
            "current_step": 2,
        }

        third_result = controller.step(final_feedback_state, workflow_config={})

        self.assertEqual(third_result["planned_actions"], ['use(direction="down")'])
        self.assertTrue(third_result["execution_pending"])
        self.assertEqual(third_result["suggestions"], second_plan.suggestions)
        self.assertEqual(controller.big_brain.plan_calls, 2)
        self.assertFalse(third_result["force_big_brain_replan"])

    def test_empty_big_brain_plan_failure_uses_deterministic_fallback_instead_of_nop(self) -> None:
        controller = DualBrainController(
            workflow_app=None,
            scheduler=_FixedScheduler("big"),
            big_brain=_QueuedBigBrain(
                [BrainPlanResult(suggestions=[], context_summary="ctx", current_task="task")]
            ),
            little_brain=LittleBrain(
                vllm_client=_EscalatingVLLMClient(),
                execute_internally=False,
            ),
            env_detector=_NoOpEnvDetector(),
            failure_detector=_NoOpFailureDetector(),
            vllm_client=None,
            vllm_available=True,
        )

        result = controller.step(
            {
                "task": "task",
                "main_task": "task",
                "skill_library": "",
                "suggestions": [],
                "current_step": 0,
                "is_first_step": False,
                "vllm_available": True,
                "has_execution_feedback": False,
                "consecutive_failures": 0,
                "last_state_changed": False,
                "previous_actions": [],
                "previous_results": [],
                "zero_progress_streak": 0,
                "repeated_action_streak": 0,
                "position_issue_detected": False,
                "oscillation_streak": 0,
            },
            workflow_config={},
        )

        self.assertEqual(
            result["suggestions"],
            [{"action": "move(x=0, y=3)", "reason": "deterministic_fallback"}],
        )
        self.assertEqual(result["planned_actions"], ["move(x=0, y=3)"])
        self.assertEqual(result["action_source"], "deterministic_fallback")
        self.assertEqual(result["brain_mode"], "little")
        self.assertEqual(result["escalation_reason"], "")

    def test_reuse_active_suggestion_does_not_fall_back_to_stale_first_step(self) -> None:
        controller = DualBrainController(
            workflow_app=None,
            scheduler=_FixedScheduler("big"),
            big_brain=_QueuedBigBrain([]),
            little_brain=LittleBrain(
                vllm_client=_SuggestionFollowingVLLMClient(),
                execute_internally=False,
            ),
            env_detector=_NoOpEnvDetector(),
            failure_detector=_NoOpFailureDetector(),
            vllm_client=None,
            vllm_available=True,
        )

        reused = controller._reuse_active_suggestion_as_planned_action(
            source_state={
                "suggestions": [
                    {"action": "choose_item(slot_index=4)", "reason": "equip"},
                    {'action': 'use(direction="down")', "reason": "clear"},
                ],
                "current_step": 2,
                "completed_steps": [0, 1],
            },
            base_state={},
            reason="fallback",
        )

        self.assertIsNone(reused)

    def test_reuse_active_suggestion_is_disabled_for_explicit_blocked_reason(self) -> None:
        controller = DualBrainController(
            workflow_app=None,
            scheduler=_FixedScheduler("big"),
            big_brain=_QueuedBigBrain([]),
            little_brain=LittleBrain(
                vllm_client=_SuggestionFollowingVLLMClient(),
                execute_internally=False,
            ),
            env_detector=_NoOpEnvDetector(),
            failure_detector=_NoOpFailureDetector(),
            vllm_client=None,
            vllm_available=True,
        )

        reused = controller._reuse_active_suggestion_as_planned_action(
            source_state={
                "suggestions": [
                    {"action": 'move(x=0, y=1)', "reason": "advance"},
                ],
                "current_step": 0,
                "completed_steps": [],
            },
            base_state={},
            reason="vllm_escalate: move_target_blocked",
        )

        self.assertIsNone(reused)

    def test_reuse_active_suggestion_is_disabled_for_failure_detector_reason(self) -> None:
        controller = DualBrainController(
            workflow_app=None,
            scheduler=_FixedScheduler("big"),
            big_brain=_QueuedBigBrain([]),
            little_brain=LittleBrain(
                vllm_client=_SuggestionFollowingVLLMClient(),
                execute_internally=False,
            ),
            env_detector=_NoOpEnvDetector(),
            failure_detector=_NoOpFailureDetector(),
            vllm_client=None,
            vllm_available=True,
        )

        reused = controller._reuse_active_suggestion_as_planned_action(
            source_state={
                "suggestions": [
                    {"action": 'move(x=0, y=1)', "reason": "advance"},
                ],
                "current_step": 0,
                "completed_steps": [],
            },
            base_state={},
            reason="failure_detector:F2",
        )

        self.assertIsNone(reused)

    def test_little_brain_503_transport_failure_reuses_current_big_brain_step(self) -> None:
        scheduler = _FixedScheduler("little")
        big_brain = _QueuedBigBrain([])
        little_brain = _EscalatingLittleBrain(
            "vllm_escalate: api_error: 503 Service Temporarily Unavailable"
        )
        vllm_client = _HealthCheckProbeVLLMClient([True], health_check_timeout_s=12.0)
        controller = DualBrainController(
            workflow_app=None,
            scheduler=scheduler,
            big_brain=big_brain,
            little_brain=little_brain,
            env_detector=_NoOpEnvDetector(),
            failure_detector=_NoOpFailureDetector(),
            vllm_client=vllm_client,
            vllm_available=True,
        )

        result = controller.step(
            {
                "suggestions": [
                    {"action": 'move(x=0, y=1)', "reason": "advance"},
                    {"action": 'use(direction="down")', "reason": "clear"},
                ],
                "current_step": 0,
                "completed_steps": [],
                "is_first_step": False,
                "vllm_available": True,
                "has_execution_feedback": False,
                "consecutive_failures": 0,
                "last_state_changed": False,
                "previous_actions": [],
                "zero_progress_streak": 0,
                "repeated_action_streak": 0,
                "position_issue_detected": False,
                "oscillation_streak": 0,
            },
            workflow_config={},
        )

        self.assertEqual(result["action_source"], "big_brain_suggestion_reuse")
        self.assertEqual(result["planned_actions"], ['move(x=0, y=1)'])
        self.assertEqual(result["current_step"], 1)
        self.assertEqual(big_brain.plan_calls, 0)
        self.assertEqual(little_brain.calls, 1)
        self.assertFalse(controller.vllm_available)

    def test_little_brain_timeout_reuses_only_current_step_suggestion(self) -> None:
        scheduler = _FixedScheduler("little")
        big_brain = _QueuedBigBrain([])
        little_brain = _EscalatingLittleBrain("vllm_escalate: timeout")
        vllm_client = _HealthCheckProbeVLLMClient([True], health_check_timeout_s=12.0)
        controller = DualBrainController(
            workflow_app=None,
            scheduler=scheduler,
            big_brain=big_brain,
            little_brain=little_brain,
            env_detector=_NoOpEnvDetector(),
            failure_detector=_NoOpFailureDetector(),
            vllm_client=vllm_client,
            vllm_available=True,
        )

        result = controller.step(
            {
                "suggestions": [
                    {"action": "choose_item(slot_index=4)", "reason": "equip"},
                    {"action": 'use(direction="down")', "reason": "clear"},
                ],
                "current_step": 1,
                "completed_steps": [0],
                "is_first_step": False,
                "vllm_available": True,
                "has_execution_feedback": False,
                "consecutive_failures": 0,
                "last_state_changed": False,
                "previous_actions": [],
                "zero_progress_streak": 0,
                "repeated_action_streak": 0,
                "position_issue_detected": False,
                "oscillation_streak": 0,
            },
            workflow_config={},
        )

        self.assertEqual(result["action_source"], "big_brain_suggestion_reuse")
        self.assertEqual(result["planned_actions"], ['use(direction="down")'])
        self.assertEqual(result["suggestions"], [{"action": 'use(direction="down")', "reason": "clear"}])
        self.assertEqual(big_brain.plan_calls, 0)
        self.assertFalse(controller.vllm_available)

    def test_little_brain_throttle_timeout_transport_failure_reuses_current_step(self) -> None:
        scheduler = _FixedScheduler("little")
        big_brain = _QueuedBigBrain([])
        little_brain = _EscalatingLittleBrain("vllm_escalate: throttle_timeout")
        vllm_client = _HealthCheckProbeVLLMClient([True], health_check_timeout_s=12.0)
        controller = DualBrainController(
            workflow_app=None,
            scheduler=scheduler,
            big_brain=big_brain,
            little_brain=little_brain,
            env_detector=_NoOpEnvDetector(),
            failure_detector=_NoOpFailureDetector(),
            vllm_client=vllm_client,
            vllm_available=True,
        )

        result = controller.step(
            {
                "suggestions": [
                    {"action": 'move(x=1, y=0)', "reason": "advance"},
                ],
                "current_step": 0,
                "completed_steps": [],
                "is_first_step": False,
                "vllm_available": True,
                "has_execution_feedback": False,
                "consecutive_failures": 0,
                "last_state_changed": False,
                "previous_actions": [],
                "zero_progress_streak": 0,
                "repeated_action_streak": 0,
                "position_issue_detected": False,
                "oscillation_streak": 0,
            },
            workflow_config={},
        )

        self.assertEqual(result["action_source"], "big_brain_suggestion_reuse")
        self.assertEqual(result["planned_actions"], ['move(x=1, y=0)'])
        self.assertEqual(big_brain.plan_calls, 0)
        self.assertFalse(controller.vllm_available)

    def test_big_brain_empty_plan_without_fastllm_uses_deterministic_fallback(self) -> None:
        controller = DualBrainController(
            workflow_app=None,
            scheduler=_FixedScheduler("big"),
            big_brain=_QueuedBigBrain(
                [BrainPlanResult(suggestions=[], context_summary="ctx", current_task="task")]
            ),
            little_brain=LittleBrain(
                vllm_client=_SuggestionFollowingVLLMClient(),
                execute_internally=False,
            ),
            env_detector=_NoOpEnvDetector(),
            failure_detector=_NoOpFailureDetector(),
            vllm_client=None,
            vllm_available=False,
        )

        result = controller.step(
            {
                "task": "fill_1_pet_bowl_with_watering_can",
                "main_task": "fill_1_pet_bowl_with_watering_can",
                "skill_library": "",
                "suggestions": [],
                "current_step": 0,
                "is_first_step": False,
                "vllm_available": False,
                "has_execution_feedback": False,
                "consecutive_failures": 0,
                "last_state_changed": False,
                "previous_actions": [],
                "previous_results": [],
                "zero_progress_streak": 0,
                "repeated_action_streak": 0,
                "position_issue_detected": False,
                "oscillation_streak": 0,
            },
            workflow_config={},
        )

        self.assertEqual(
            result["suggestions"],
            [{"action": "move(x=2, y=1)", "reason": "deterministic_fallback_no_fastllm"}],
        )
        self.assertEqual(result["planned_actions"], ["move(x=2, y=1)"])
        self.assertEqual(result["action_source"], "deterministic_fallback_no_fastllm")

    def test_handoff_skips_little_brain_for_generic_deterministic_fallback(self) -> None:
        little_brain = _PendingLittleBrain()
        controller = DualBrainController(
            workflow_app=None,
            scheduler=_FixedScheduler("big"),
            big_brain=_QueuedBigBrain([]),
            little_brain=little_brain,
            env_detector=_NoOpEnvDetector(),
            failure_detector=_NoOpFailureDetector(),
            vllm_client=None,
            vllm_available=True,
        )

        state = {
            "action_source": "deterministic_fallback",
            "suggestions": [{"action": "move(x=0, y=3)", "reason": "deterministic_fallback"}],
            "planned_actions": ["move(x=0, y=3)"],
        }

        result = controller._handoff_big_plan_to_little_brain(state)

        self.assertEqual(result, state)
        self.assertEqual(little_brain.calls, 0)

    def test_deterministic_fallback_menu_detection_handles_nomenu_type(self) -> None:
        action = DualBrainController._build_deterministic_fallback_action(
            {
                "task": "",
                "gathered_info": {
                    "current_menu": {"type": "NoMenu"},
                },
            }
        )

        self.assertEqual(action, "move(x=0, y=3)")

    def test_deterministic_fallback_menu_detection_closes_open_menu(self) -> None:
        action = DualBrainController._build_deterministic_fallback_action(
            {
                "task": "",
                "gathered_info": {
                    "current_menu": {"type": "DialogueBox"},
                },
            }
        )

        self.assertEqual(action, 'menu(option="close")')

    def test_autonomous_fallback_503_marks_vllm_temporarily_unavailable(self) -> None:
        vllm_client = _Escalating503VLLMClient()
        controller = DualBrainController(
            workflow_app=None,
            scheduler=_FixedScheduler("big"),
            big_brain=_QueuedBigBrain([]),
            little_brain=LittleBrain(
                vllm_client=vllm_client,
                execute_internally=False,
            ),
            env_detector=_NoOpEnvDetector(),
            failure_detector=_NoOpFailureDetector(),
            vllm_client=vllm_client,
            vllm_available=True,
            vllm_health_retry_seconds=30.0,
        )

        retry_before = controller._next_vllm_health_retry_ts
        result = controller._little_brain_autonomous_fallback(
            state={
                "task": "collect_wood",
                "main_task": "collect_wood",
                "subtask_description": "collect_wood",
                "skill_library": "",
                "gathered_info": {"current_menu": {"type": "NoMenu"}},
            },
            result_state={},
        )

        self.assertEqual(result["action_source"], "deterministic_fallback")
        self.assertFalse(controller.vllm_available)
        self.assertGreater(controller._next_vllm_health_retry_ts, retry_before)
        self.assertGreater(controller._next_vllm_health_retry_ts, time.time())

    def test_fastllm_recovery_requires_confirming_probe_after_503(self) -> None:
        vllm_client = _HealthCheckProbeVLLMClient([True, True], health_check_timeout_s=12.0)

        with mock.patch("cradle.runner.dual_brain.time.time", return_value=100.0):
            controller = DualBrainController(
                workflow_app=None,
                scheduler=_FixedScheduler("big"),
                big_brain=_QueuedBigBrain([]),
                little_brain=LittleBrain(
                    vllm_client=vllm_client,
                    execute_internally=False,
                ),
                env_detector=_NoOpEnvDetector(),
                failure_detector=_NoOpFailureDetector(),
                vllm_client=vllm_client,
                vllm_available=True,
                vllm_health_retry_seconds=30.0,
                vllm_reenable_success_threshold=2,
                vllm_reenable_probe_interval_seconds=3.0,
            )
            controller._maybe_mark_vllm_unavailable(
                reason=(
                    "api_error: 503 Service Unavailable: reach max requests. "
                    "current requests: 6 service support max requests: 5"
                ),
                source="test",
            )

        self.assertFalse(controller.vllm_available)
        self.assertEqual(controller._next_vllm_health_retry_ts, 130.0)

        with mock.patch("cradle.runner.dual_brain.time.time", return_value=130.0):
            controller._maybe_refresh_vllm_availability()

        self.assertFalse(controller.vllm_available)
        self.assertEqual(controller._next_vllm_health_retry_ts, 133.0)
        self.assertEqual(vllm_client.calls, [12.0])

        with mock.patch("cradle.runner.dual_brain.time.time", return_value=133.0):
            controller._maybe_refresh_vllm_availability()

        self.assertTrue(controller.vllm_available)
        self.assertEqual(vllm_client.calls, [12.0, 12.0])

    def test_stuck_detection_triggers_after_zero_progress_or_repeated_unproductive_steps(self) -> None:
        controller = DualBrainController(
            workflow_app=None,
            scheduler=_FixedScheduler("little"),
            big_brain=_QueuedBigBrain([]),
            little_brain=_PendingLittleBrain(),
            env_detector=_NoOpEnvDetector(),
            failure_detector=_NoOpFailureDetector(),
            vllm_client=None,
            vllm_available=True,
        )

        self.assertTrue(
            controller._is_definitively_stuck(
                {"zero_progress_streak": 4, "repeated_action_streak": 0, "oscillation_streak": 0}
            )
        )
        self.assertFalse(
            controller._is_definitively_stuck(
                {
                    "zero_progress_streak": 0,
                    "repeated_action_streak": 4,
                    "oscillation_streak": 0,
                    "consecutive_failures": 0,
                }
            )
        )
        self.assertTrue(
            controller._is_definitively_stuck(
                {
                    "zero_progress_streak": 2,
                    "repeated_action_streak": 6,
                    "oscillation_streak": 0,
                    "consecutive_failures": 0,
                }
            )
        )
        self.assertFalse(
            controller._is_definitively_stuck(
                {
                    "zero_progress_streak": 3,
                    "repeated_action_streak": 3,
                    "oscillation_streak": 0,
                    "consecutive_failures": 0,
                }
            )
        )


if __name__ == "__main__":
    unittest.main()
