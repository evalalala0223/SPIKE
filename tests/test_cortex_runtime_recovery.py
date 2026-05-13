from __future__ import annotations

import sys
import unittest
from unittest import mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from stardojo.utils.cortex_runtime_utils import (
    is_recoverable_no_execution_reason,
    resolve_cortex_executable_actions,
)


class _StubDecision:
    def __init__(self, *, action: str = "", reason: str = "", escalate: bool = False) -> None:
        self.action = action
        self.reason = reason
        self.escalate = escalate


class _StubRuntimeRecoveryVLLM:
    @staticmethod
    def _extract_selected_item_name(gathered: object) -> str:
        if isinstance(gathered, dict):
            return str(gathered.get("selected_item_name", "") or "").strip()
        return ""

    @staticmethod
    def _extract_selected_item_name_from_toolbar(toolbar_information: object) -> str:
        return ""

    @staticmethod
    def _build_invalidated_suggestion_local_recovery(**_kwargs) -> str:
        return ""

    @staticmethod
    def _build_autonomous_local_recovery_action(*, game_state: object) -> str:
        if isinstance(game_state, dict):
            return str(game_state.get("autonomous_local_recovery_action", "") or "").strip()
        return ""

    @staticmethod
    def _validate_decision_against_state(
        decision: _StubDecision,
        suggestion: dict[str, str],
        game_state: object,
    ) -> _StubDecision:
        return decision


class TestCortexRuntimeRecovery(unittest.TestCase):
    def test_timeout_escalation_counts_as_recoverable_no_execution(self) -> None:
        self.assertTrue(
            is_recoverable_no_execution_reason("escalation_reason:vllm_escalate: timeout")
        )
        self.assertTrue(
            is_recoverable_no_execution_reason("escalation_reason:vllm_escalate: api_error: 503")
        )

    def test_current_step_suggestion_is_reused_after_timeout_escalation(self) -> None:
        result = resolve_cortex_executable_actions(
            result_state={
                "brain_mode": "big",
                "escalation_reason": "vllm_escalate: timeout",
                "current_step": 0,
                "main_task": "fertilize_1_dirt_with_speed_gro",
                "task": "fertilize_1_dirt_with_speed_gro",
                "toolbar_information": "\n".join(
                    [
                        "slot_index 0: Axe (quantity: 1)",
                        "slot_index 1: Hoe (quantity: 1)",
                        "Currently selected item: slot_index 0: Axe",
                    ]
                ),
                "gathered_info": {
                    "selected_item_name": "Axe",
                    "current_menu": {"type": "No Menu"},
                    "surroundings": "\n".join(
                        [
                            "[0, 1]: empty",
                            "[1, 0]: empty",
                            "[-1, 0]: empty",
                        ]
                    ),
                },
            },
            normalized_actions=[],
            normalized_suggestion_actions=['move(x=0, y=1)'],
        )

        self.assertEqual(result["actions"], ['move(x=0, y=1)'])
        self.assertEqual(result["execution_source"], "suggestions")
        self.assertTrue(result["used_suggestion_fallback"])

    def test_pending_local_recovery_overrides_new_planned_action(self) -> None:
        with mock.patch(
            "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
            return_value=(_StubRuntimeRecoveryVLLM, _StubDecision),
        ):
            result = resolve_cortex_executable_actions(
                result_state={
                    "pending_local_recovery_action": 'move(x=0, y=1)',
                    "pending_local_recovery_reason": "repeated_interaction_no_confirmation",
                },
                normalized_actions=['move(x=-3, y=0)'],
                normalized_suggestion_actions=[],
            )

        self.assertEqual(result["actions"], ['move(x=0, y=1)'])
        self.assertEqual(result["execution_source"], "runtime_recovery")
        self.assertFalse(result["used_suggestion_fallback"])

    def test_repeated_same_signature_no_execution_uses_local_recovery(self) -> None:
        with mock.patch(
            "stardojo.utils.cortex_runtime_utils._load_fastllm_runtime_classes",
            return_value=(_StubRuntimeRecoveryVLLM, _StubDecision),
        ):
            result = resolve_cortex_executable_actions(
                result_state={
                    "brain_mode": "big",
                    "step_count": 7,
                    "last_no_execution_step": 7,
                    "same_step_no_execution_streak": 1,
                    "last_no_execution_signature": (
                        "awaiting_big_brain_replan | shot_7.png | walk toward the farm exit"
                    ),
                    "screenshot_path": "shot_7.png",
                    "subtask_description": "walk toward the farm exit",
                    "autonomous_local_recovery_action": "move(x=1, y=0)",
                },
                normalized_actions=[],
                normalized_suggestion_actions=[],
            )

        self.assertEqual(result["actions"], ["move(x=1, y=0)"])
        self.assertEqual(result["execution_source"], "runtime_recovery")
        self.assertFalse(result["used_suggestion_fallback"])


if __name__ == "__main__":
    unittest.main()
