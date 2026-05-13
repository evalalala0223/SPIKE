import json
from pathlib import Path
import shutil
import unittest
import uuid

from cradle.memory.mem0_provider import Mem0Provider
from cradle.runner.langgraph_nodes import LangGraphNodes


class DummyMem0Provider(Mem0Provider):
    def _init_sa_kg(self) -> None:
        self.sa_kg = None
        self.sa_kg_enabled = False


class CaptureMem0Provider:
    def __init__(self) -> None:
        self.stored = []
        self.skips = []

    def store(self, **kwargs) -> None:
        self.stored.append(kwargs)

    def record_store_skip(self, reason: str) -> None:
        self.skips.append(reason)


def _write_records(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _make_nodes(provider: CaptureMem0Provider) -> LangGraphNodes:
    nodes = object.__new__(LangGraphNodes)
    nodes.mem0_provider = provider
    nodes.mem0_store_require_meaningful_progress = True
    nodes.mem0_store_progress_min_chars = 8
    return nodes


def test_mem0_compacts_low_value_history_on_init(tmp_path):
    storage_root = tmp_path / "mem0_store.jsonl"
    actual_storage = tmp_path / "stardew_valley" / "mem0_store.jsonl"
    metrics_path = tmp_path / "mem0_metrics.json"

    _write_records(
        actual_storage,
        [
            {
                "id": "low",
                "key": "low",
                "state": "task=demo | progress= | execution=",
                "actions": ["move(x=1, y=0)"],
                "success": True,
                "reward": 1.0,
                "reward_ema": 1.0,
                "attempts": 1,
                "successes": 1,
                "created_at": 1.0,
                "last_seen": 1.0,
                "metadata": {"progress": "", "store_source": "execution_feedback"},
            },
            {
                "id": "good",
                "key": "good",
                "state": "task=demo | progress=cleared weeds near porch | execution=used scythe",
                "actions": ['use(direction="left")'],
                "success": True,
                "reward": 1.0,
                "reward_ema": 1.0,
                "attempts": 1,
                "successes": 1,
                "created_at": 2.0,
                "last_seen": 2.0,
                "metadata": {"progress": "cleared weeds near porch", "store_source": "reflection"},
            },
        ],
    )

    provider = DummyMem0Provider(
        enabled=True,
        namespace="stardew_valley",
        storage_path=str(storage_root),
        metrics_path=str(metrics_path),
        require_meaningful_progress=True,
        progress_min_chars=8,
    )

    assert len(provider.records) == 1
    assert provider.records[0]["id"] == "good"
    persisted = [json.loads(line) for line in actual_storage.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [record["id"] for record in persisted] == ["good"]
    backups = list(actual_storage.parent.glob("mem0_store.jsonl.compact.*.bak"))
    assert backups


def test_mem0_store_rejects_low_value_setup_record(tmp_path):
    provider = DummyMem0Provider(
        enabled=True,
        namespace="stardew_valley",
        storage_path=str(tmp_path / "mem0_store.jsonl"),
        metrics_path=str(tmp_path / "mem0_metrics.json"),
        require_meaningful_progress=True,
        progress_min_chars=8,
    )

    provider.store(
        state="task=demo | progress= | execution=",
        actions=["choose_item(slot_index=3)"],
        success=True,
        reward=1.0,
        metadata={"progress": "", "store_source": "skill_execute_success_fallback_first_step"},
    )

    assert provider.records == []
    assert provider.metrics["store_skip_reasons"]["low_value_record"] == 1


def test_fallback_store_uses_same_progress_gate():
    provider = CaptureMem0Provider()
    nodes = object.__new__(LangGraphNodes)
    nodes.mem0_provider = provider
    nodes.mem0_store_require_meaningful_progress = True
    nodes.mem0_store_progress_min_chars = 8

    stored = nodes._commit_mem0_store(
        {
            "task": "demo",
            "planned_actions": ["move(x=1, y=0)"],
            "previous_actions": [],
            "gathered_info": "",
            "latest_execution_summary": "",
            "step_count": 1,
            "task_changed": False,
            "memory_quick_path": False,
        },
        reflection_result=None,
        execution_success=True,
        reflection_confirmed_success=False,
        reflection_status="",
        store_source="skill_execute_success_fallback_first_step",
    )

    assert stored is False
    assert provider.stored == []
    assert provider.skips == ["move_only_no_progress"]


def test_long_no_progress_reasoning_is_still_low_value_for_mem0():
    provider = CaptureMem0Provider()
    nodes = object.__new__(LangGraphNodes)
    nodes.mem0_provider = provider
    nodes.mem0_store_require_meaningful_progress = True
    nodes.mem0_store_progress_min_chars = 8

    stored = nodes._commit_mem0_store(
        {
            "task": "fertilize_5_dirt_with_basic_retaining_soil",
            "planned_actions": ["move(x=2, y=1)"],
            "previous_actions": [],
            "gathered_info": "",
            "latest_execution_summary": (
                "Last action: move(x=2, y=1). Task progress stayed at 0. "
                "The productive action had no observable effect. The task is not completed yet."
            ),
            "step_count": 3,
            "task_changed": False,
            "memory_quick_path": False,
        },
        reflection_result={
            "reasoning": (
                "The executed action was None, as no fertilizer application has been performed yet. "
                "The target task is not completed (0/5 tiles fertilized)."
            )
        },
        execution_success=True,
        reflection_confirmed_success=False,
        reflection_status="",
        store_source="execution_feedback",
    )

    assert stored is False
    assert provider.stored == []
    assert provider.skips == ["move_only_no_progress"]


class TestMem0Governance(unittest.TestCase):
    def test_mem0_compacts_low_value_history_on_init(self) -> None:
        cache_root = Path("cache")
        cache_root.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_root / f"tmp_mem0_governance_{uuid.uuid4().hex[:8]}"
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            storage_root = tmp_path / "mem0_store.jsonl"
            actual_storage = tmp_path / "stardew_valley" / "mem0_store.jsonl"
            metrics_path = tmp_path / "mem0_metrics.json"

            _write_records(
                actual_storage,
                [
                    {
                        "id": "move_dirty",
                        "key": "move_dirty",
                        "state": "task=demo | progress=The executed action is none and the task is not completed yet. | execution=",
                        "actions": ["move(x=1, y=0)"],
                        "success": True,
                        "reward": 1.0,
                        "reward_ema": 1.0,
                        "attempts": 1,
                        "successes": 1,
                        "created_at": 1.0,
                        "last_seen": 1.0,
                        "metadata": {"progress": "The executed action is none and the task is not completed yet."},
                    },
                    {
                        "id": "setup_dirty",
                        "key": "setup_dirty",
                        "state": "task=demo | progress=The Hoe is selected. | execution=",
                        "actions": ["choose_item(slot_index=1)"],
                        "success": True,
                        "reward": 1.0,
                        "reward_ema": 1.0,
                        "attempts": 1,
                        "successes": 1,
                        "created_at": 2.0,
                        "last_seen": 2.0,
                        "metadata": {"progress": "The Hoe is selected."},
                    },
                    {
                        "id": "good",
                        "key": "good",
                        "state": "task=demo | progress=cleared weeds near porch | execution=used scythe",
                        "actions": ['use(direction="left")'],
                        "success": True,
                        "reward": 1.0,
                        "reward_ema": 1.0,
                        "attempts": 1,
                        "successes": 1,
                        "created_at": 3.0,
                        "last_seen": 3.0,
                        "metadata": {"progress": "cleared weeds near porch", "store_source": "reflection"},
                    },
                ],
            )

            provider = DummyMem0Provider(
                enabled=True,
                namespace="stardew_valley",
                storage_path=str(storage_root),
                metrics_path=str(metrics_path),
                require_meaningful_progress=True,
                progress_min_chars=8,
            )

            self.assertEqual([record["id"] for record in provider.records], ["good"])
            persisted = [
                json.loads(line)
                for line in actual_storage.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([record["id"] for record in persisted], ["good"])
        finally:
            shutil.rmtree(tmp_path, ignore_errors=True)

    def test_move_only_store_without_hard_progress_is_rejected(self) -> None:
        provider = CaptureMem0Provider()
        nodes = _make_nodes(provider)

        stored = nodes._commit_mem0_store(
            {
                "task": "demo",
                "planned_actions": ["move(x=1, y=0)"],
                "previous_actions": [],
                "gathered_info": "",
                "latest_execution_summary": "The executed action is none and the task is not completed yet.",
                "step_count": 1,
                "task_changed": False,
                "memory_quick_path": False,
                "task_progress_delta": 0,
                "task_progress_quantity": 0,
                "previous_task_progress_quantity": 0,
            },
            reflection_result={
                "reasoning": "The executed action is none and no progress was made.",
            },
            execution_success=True,
            reflection_confirmed_success=False,
            reflection_status="",
            store_source="execution_feedback",
        )

        self.assertFalse(stored)
        self.assertEqual(provider.stored, [])
        self.assertEqual(provider.skips, ["move_only_no_progress"])

    def test_textual_progress_without_hard_progress_is_rejected(self) -> None:
        provider = CaptureMem0Provider()
        nodes = _make_nodes(provider)

        stored = nodes._commit_mem0_store(
            {
                "task": "demo",
                "planned_actions": ['use(direction="down")'],
                "previous_actions": [],
                "gathered_info": "",
                "latest_execution_summary": "The action looked productive but the task is still not completed.",
                "step_count": 2,
                "task_changed": False,
                "memory_quick_path": False,
                "task_progress_delta": 0,
                "task_progress_quantity": 0,
                "previous_task_progress_quantity": 0,
            },
            reflection_result={
                "reasoning": "Cleared weeds near the porch and should continue locally.",
            },
            execution_success=True,
            reflection_confirmed_success=False,
            reflection_status="",
            store_source="execution_feedback",
        )

        self.assertFalse(stored)
        self.assertEqual(provider.stored, [])
        self.assertEqual(provider.skips, ["no_hard_progress"])

    def test_hard_progress_store_includes_progress_metadata(self) -> None:
        provider = CaptureMem0Provider()
        nodes = _make_nodes(provider)

        stored = nodes._commit_mem0_store(
            {
                "task": "clear_10_weeds_with_scythe",
                "planned_actions": ['use(direction="left")'],
                "previous_actions": [],
                "gathered_info": "",
                "latest_execution_summary": "Task progress changed from 3 to 4.",
                "step_count": 4,
                "task_changed": False,
                "memory_quick_path": False,
                "task_progress_delta": 1,
                "task_progress_quantity": 4,
                "previous_task_progress_quantity": 3,
                "last_state_changed": True,
            },
            reflection_result={"reasoning": "cleared one more weed"},
            execution_success=True,
            reflection_confirmed_success=True,
            reflection_status="success",
            store_source="reflection",
        )

        self.assertTrue(stored)
        self.assertEqual(len(provider.stored), 1)
        stored_record = provider.stored[0]
        self.assertEqual(stored_record["reward"], 1.0)
        self.assertEqual(stored_record["metadata"]["progress_delta"], 1.0)
        self.assertEqual(stored_record["metadata"]["previous_progress_quantity"], 3.0)
        self.assertEqual(stored_record["metadata"]["progress_quantity"], 4.0)
        self.assertTrue(stored_record["metadata"]["hard_progress"])


if __name__ == "__main__":
    unittest.main()
