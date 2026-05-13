from __future__ import annotations

import contextlib
import unittest
from unittest import mock

from cradle.runner.dual_brain import DualBrainController
from cradle.runner.vllm_client import VLLMClient


class _NoOpScheduler:
    def decide(self, state: dict) -> str:
        return "big"

    def reset_counter(self) -> None:
        return None

    def increment_counter(self) -> None:
        return None

    def get_status(self) -> dict:
        return {}


class _NoOpBigBrain:
    def clear_failed_actions(self) -> None:
        return None

    def clear_plan_failure(self) -> None:
        return None

    def get_status(self) -> dict:
        return {}


class _NoOpLittleBrain:
    def get_status(self) -> dict:
        return {}


class _NoOpEnvDetector:
    threshold = 0.35

    def detect_change(self, screenshot_path: str) -> tuple[bool, float]:
        return False, 0.0


class _NoOpFailureDetector:
    def evaluate(self, **kwargs: object):
        return None

    def get_status(self) -> dict:
        return {}


class _ProbeVLLMClient:
    def __init__(self, results: list[bool], health_check_timeout_s: float = 12.0) -> None:
        self.results = list(results)
        self.health_check_timeout_s = health_check_timeout_s
        self.calls: list[float | None] = []

    def health_check(self, timeout_s: float | None = None) -> bool:
        self.calls.append(timeout_s)
        return self.results.pop(0)


class TestFastLLMHealthChecks(unittest.TestCase):
    def test_vllm_client_health_check_uses_safer_default_timeout(self) -> None:
        client = VLLMClient(api_key="dummy", request_timeout_s=30)
        response = mock.Mock(status_code=200)

        with mock.patch(
            "cradle.runner.vllm_client.acquire_llm_endpoint_slot",
            return_value=contextlib.nullcontext(),
        ), mock.patch(
            "cradle.runner.vllm_client.requests.post",
            return_value=response,
        ) as mock_post:
            self.assertTrue(client.health_check())

        self.assertEqual(mock_post.call_args.kwargs["timeout"], 15.0)

    def test_vllm_client_health_check_uses_configured_timeout(self) -> None:
        client = VLLMClient(
            api_key="dummy",
            request_timeout_s=30,
            health_check_timeout_s=9.5,
        )
        response = mock.Mock(status_code=200)

        with mock.patch(
            "cradle.runner.vllm_client.acquire_llm_endpoint_slot",
            return_value=contextlib.nullcontext(),
        ), mock.patch(
            "cradle.runner.vllm_client.requests.post",
            return_value=response,
        ) as mock_post:
            self.assertTrue(client.health_check())

        self.assertEqual(mock_post.call_args.kwargs["timeout"], 9.5)

    def test_vllm_client_health_check_subtracts_slot_wait_from_http_timeout(self) -> None:
        client = VLLMClient(
            api_key="dummy",
            request_timeout_s=30,
            health_check_timeout_s=9.5,
        )
        response = mock.Mock(status_code=200)

        with mock.patch(
            "cradle.runner.vllm_client.acquire_llm_endpoint_slot",
            return_value=contextlib.nullcontext({"waited_s": 4.0}),
        ), mock.patch(
            "cradle.runner.vllm_client.requests.post",
            return_value=response,
        ) as mock_post:
            self.assertTrue(client.health_check())

        self.assertEqual(mock_post.call_args.kwargs["timeout"], 5.5)

    def test_dual_brain_defers_retry_after_startup_failure(self) -> None:
        probe_client = _ProbeVLLMClient([True, True], health_check_timeout_s=12.0)

        with mock.patch("cradle.runner.dual_brain.time.time", return_value=100.0):
            controller = DualBrainController(
                workflow_app=None,
                scheduler=_NoOpScheduler(),
                big_brain=_NoOpBigBrain(),
                little_brain=_NoOpLittleBrain(),
                env_detector=_NoOpEnvDetector(),
                failure_detector=_NoOpFailureDetector(),
                vllm_client=probe_client,
                vllm_available=False,
                vllm_health_retry_seconds=30.0,
                vllm_reenable_success_threshold=2,
                vllm_reenable_probe_interval_seconds=3.0,
            )

        self.assertEqual(controller._next_vllm_health_retry_ts, 130.0)

        with mock.patch("cradle.runner.dual_brain.time.time", return_value=110.0):
            controller._maybe_refresh_vllm_availability()

        self.assertEqual(probe_client.calls, [])
        self.assertFalse(controller.vllm_available)

        with mock.patch("cradle.runner.dual_brain.time.time", return_value=130.0):
            controller._maybe_refresh_vllm_availability()

        self.assertEqual(probe_client.calls, [12.0])
        self.assertFalse(controller.vllm_available)
        self.assertEqual(controller._next_vllm_health_retry_ts, 133.0)

        with mock.patch("cradle.runner.dual_brain.time.time", return_value=133.0):
            controller._maybe_refresh_vllm_availability()

        self.assertEqual(probe_client.calls, [12.0, 12.0])
        self.assertTrue(controller.vllm_available)


if __name__ == "__main__":
    unittest.main()
