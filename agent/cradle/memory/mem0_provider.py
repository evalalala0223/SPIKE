import json
import math
import os
import re
import shutil
import tempfile
import time
from hashlib import sha1
from typing import Any, Dict, List, Optional, Tuple

from cradle.log import Logger
from cradle.utils.file_utils import assemble_project_path

logger = Logger()


class Mem0Provider:
    """Hybrid memory provider: Mem0-like primary memory + SA-KG augmentation.

    Design goals:
    - Keep a robust, local, dependency-light persistent memory as primary path
    - Fuse SA-KG retrieval features into confidence/ranking
    - Support safe quick-path routing with explainable confidence
    - Provide deterministic behavior without external services
    """

    def __init__(
        self,
        enabled: bool = False,
        embedding_provider: Optional[Any] = None,
        namespace: Optional[str] = None,
        storage_path: Optional[str] = None,
        metrics_path: Optional[str] = None,
        quick_path_threshold: float = 0.85,
        max_results: int = 3,
        max_records: int = 5000,
        recency_decay_hours: float = 24.0,
        sa_kg_weight: float = 0.25,
        reward_weight: float = 0.15,
        success_weight: float = 0.10,
        require_meaningful_progress: bool = True,
        progress_min_chars: int = 8,
    ) -> None:
        self.enabled = enabled
        self.quick_path_threshold = quick_path_threshold
        self.max_results = max_results
        self.max_records = max(100, int(max_records))
        self.recency_decay_hours = max(1.0, recency_decay_hours)
        self.sa_kg_weight = max(0.0, min(0.6, sa_kg_weight))
        self.reward_weight = max(0.0, min(0.6, reward_weight))
        self.success_weight = max(0.0, min(0.6, success_weight))
        self.require_meaningful_progress = bool(require_meaningful_progress)
        self.progress_min_chars = max(1, int(progress_min_chars))
        self.namespace = self._sanitize_namespace(namespace)
        default_storage_path = os.path.normpath(assemble_project_path(f"./cache/mem0/{self.namespace}/mem0_store.jsonl"))
        default_metrics_path = os.path.normpath(assemble_project_path(f"./cache/mem0/{self.namespace}/mem0_metrics.json"))
        if storage_path:
            configured_path = storage_path.format(namespace=self.namespace) if "{namespace}" in storage_path else storage_path
            configured_dir = os.path.dirname(configured_path)
            configured_name = os.path.basename(configured_path)
            self.storage_path = os.path.join(configured_dir, self.namespace, configured_name)
        else:
            self.storage_path = default_storage_path
        self.metrics_path = os.path.normpath(metrics_path) if metrics_path else default_metrics_path

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)

        # In-memory cache
        self.records: List[Dict[str, Any]] = self._load_records()

        # Persistent metrics
        self.metrics: Dict[str, Any] = self._load_metrics()
        if not self.records:
            self._bootstrap_records_if_empty()
        self._refresh_result_backfill_records_if_needed()
        self._compact_low_value_records(trigger="init")
        self._enforce_success_experience_cap(trigger="init")
        self._enforce_storage_governance(trigger="init")

        # Optional SA-KG augmenter
        self.sa_kg = None
        self.sa_kg_enabled = False
        self.embedding_provider = embedding_provider
        self._init_sa_kg()

    @staticmethod
    def _sanitize_namespace(namespace: Optional[str]) -> str:
        raw = (namespace or "default").strip().lower()
        sanitized = re.sub(r"[^a-z0-9._-]+", "_", raw)
        sanitized = sanitized.strip("._-")
        return sanitized or "default"

    def _init_sa_kg(self) -> None:
        try:
            from cradle.memory import SAKG

            self.sa_kg = SAKG()
            self.sa_kg.initialize(embedding_provider=self.embedding_provider, namespace=self.namespace)
            self.sa_kg_enabled = bool(self.sa_kg and self.sa_kg.enabled)
            if self.sa_kg_enabled:
                logger.write(f"[Mem0] SA-KG augmenter enabled (namespace={self.namespace})")
            else:
                logger.write("[Mem0] SA-KG augmenter disabled (config/dependency)")
        except Exception as e:
            self.sa_kg = None
            self.sa_kg_enabled = False
            logger.warn(f"[Mem0] Failed to initialize SA-KG augmenter: {e}")

    def _bootstrap_records_if_empty(self) -> None:
        records = self._load_success_result_records()
        if not records:
            return

        records_by_key: Dict[str, Dict[str, Any]] = {}
        for rec in records:
            normalized = self._normalize_record(rec)
            if normalized is None or self._is_low_value_record(normalized):
                continue
            key = str(normalized.get("key", ""))
            if not key:
                continue
            existing = records_by_key.get(key)
            if existing is None or float(normalized.get("last_seen", 0.0)) >= float(existing.get("last_seen", 0.0)):
                records_by_key[key] = normalized

        if not records_by_key:
            return

        self.records = list(records_by_key.values())
        self._enforce_success_experience_cap(trigger="bootstrap")
        self._enforce_storage_governance(trigger="bootstrap")
        self._persist_records()
        self.metrics["bootstrap_records_total"] = int(self.metrics.get("bootstrap_records_total", 0)) + len(self.records)
        self.metrics["last_bootstrap_records"] = len(self.records)
        self.metrics["last_bootstrap_source"] = "runs/results"
        self._persist_metrics()
        logger.write(f"[Mem0] Bootstrapped {len(self.records)} successful records from runs/results")

    def _load_success_result_records(self) -> List[Dict[str, Any]]:
        results_root = self._runs_results_root()
        if not os.path.isdir(results_root):
            return []

        result_paths: List[str] = []
        for dirpath, _dirnames, filenames in os.walk(results_root):
            if "result.json" in filenames:
                result_paths.append(os.path.join(dirpath, "result.json"))
        result_paths.sort()

        records: List[Dict[str, Any]] = []
        now = time.time()
        for path in result_paths:
            record = self._record_from_success_result(path, now)
            if record is not None:
                records.append(record)
        return records

    @staticmethod
    def _runs_results_root() -> str:
        agent_root = os.path.normpath(assemble_project_path("."))
        repo_root = os.path.dirname(agent_root)
        return os.path.join(repo_root, "runs", "results")

    def _record_from_success_result(self, path: str, now: float) -> Optional[Dict[str, Any]]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None

        if not isinstance(data, dict) or data.get("completed") is not True:
            return None
        if data.get("is_valid_benchmark") is False:
            return None

        steps = data.get("steps", [])
        if not isinstance(steps, list) or not steps:
            return None

        actions, progress_quantity, previous_progress_quantity = self._extract_success_result_actions(steps)
        if not actions:
            return None
        if self._are_setup_only_actions(actions):
            return None

        task_name = str(data.get("task_name") or data.get("runner_task_name") or "").strip()
        if not task_name:
            parent_name = os.path.basename(os.path.dirname(path))
            task_name = re.sub(r"^task_\d+_", "", parent_name)
            task_name = re.sub(r"_\d+$", "", task_name)

        final_quantity = self._coerce_numeric(data.get("final_quantity"))
        if final_quantity is not None:
            progress_quantity = final_quantity
        progress_delta = None
        if progress_quantity is not None and previous_progress_quantity is not None:
            progress_delta = progress_quantity - previous_progress_quantity
        step_count = self._result_step_count(data, steps)

        progress = f"Task is completed: {task_name or 'historical task'}."
        if progress_quantity is not None:
            progress += f" final_quantity={progress_quantity:g}."

        run_id = str(data.get("run_id") or os.path.basename(os.path.dirname(os.path.dirname(path))) or "").strip()
        state = (
            f"task={task_name} | progress={progress} "
            f"| source=result_backfill | run_id={run_id}"
        )
        metadata = {
            "task": task_name,
            "progress": progress,
            "store_source": "result_backfill",
            "source_run_id": run_id,
            "source_task_index": data.get("task_index"),
            "step_count": step_count,
            "completed": True,
            "progress_delta": progress_delta,
            "progress_quantity": progress_quantity,
            "previous_progress_quantity": previous_progress_quantity,
            "hard_progress": True,
            "setup_only": self._are_setup_only_actions(actions),
            "move_only": self._is_move_only_actions(actions),
            "action_scope": "full_route",
            "action_count": len(actions),
        }
        key = self._build_memory_key(state, actions)
        return {
            "id": f"mem_{sha1(f'{key}:{now}'.encode('utf-8')).hexdigest()[:12]}",
            "key": key,
            "state": state,
            "actions": actions,
            "success": True,
            "reward": 1.0,
            "reward_ema": 1.0,
            "attempts": 1,
            "successes": 1,
            "created_at": now,
            "last_seen": now,
            "metadata": metadata,
        }

    def _extract_success_result_actions(self, steps: List[Any]) -> Tuple[List[str], Optional[float], Optional[float]]:
        cleaned_steps: List[Tuple[str, Optional[float], bool]] = []
        previous_quantity: Optional[float] = None
        hard_progress_index: Optional[int] = None
        hard_progress_previous_quantity: Optional[float] = None
        hard_progress_quantity: Optional[float] = None

        for step in steps:
            if not isinstance(step, dict):
                continue
            action = str(step.get("action", "") or "").strip()
            if not action or action.lower() == "nop()":
                continue
            task_eval = step.get("task_eval", {})
            task_eval = task_eval if isinstance(task_eval, dict) else {}
            quantity = self._coerce_numeric(task_eval.get("quantity"))
            completed = task_eval.get("completed") is True
            progressed = bool(
                completed
                or (
                    quantity is not None
                    and previous_quantity is not None
                    and quantity != previous_quantity
                )
            )
            cleaned_steps.append((action, quantity, completed))
            if progressed:
                hard_progress_index = len(cleaned_steps) - 1
                hard_progress_previous_quantity = previous_quantity
                hard_progress_quantity = quantity
            if quantity is not None:
                previous_quantity = quantity

        if not cleaned_steps:
            return [], None, None

        final_index = hard_progress_index if hard_progress_index is not None else len(cleaned_steps) - 1
        actions = [action for action, _quantity, _completed in cleaned_steps[: final_index + 1]]
        return actions, hard_progress_quantity, hard_progress_previous_quantity

    def _refresh_result_backfill_records_if_needed(self) -> None:
        if not self.records:
            return

        stale_backfill = False
        for rec in self.records:
            if not self._is_result_backfill_record(rec):
                continue
            metadata = rec.get("metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            if metadata.get("action_scope") != "full_route":
                stale_backfill = True
                break
        if not stale_backfill:
            return

        refreshed_by_key: Dict[str, Dict[str, Any]] = {}
        for rec in self._load_success_result_records():
            normalized = self._normalize_record(rec)
            if normalized is None or self._is_low_value_record(normalized):
                continue
            key = str(normalized.get("key", ""))
            if not key:
                continue
            refreshed_by_key[key] = normalized

        if not refreshed_by_key:
            return

        retained = [rec for rec in self.records if not self._is_result_backfill_record(rec)]
        before = len(self.records)
        self.records = retained + list(refreshed_by_key.values())
        self._enforce_success_experience_cap(trigger="full_route_refresh")
        self._enforce_storage_governance(trigger="full_route_refresh")
        self._persist_records()
        self.metrics["result_backfill_full_route_refresh_total"] = (
            int(self.metrics.get("result_backfill_full_route_refresh_total", 0)) + 1
        )
        self.metrics["last_result_backfill_full_route_refresh_before"] = before
        self.metrics["last_result_backfill_full_route_refresh_after"] = len(self.records)
        self._persist_metrics()
        logger.write(
            "[Mem0] Refreshed result_backfill records with full successful routes "
            f"(before={before}, after={len(self.records)})"
        )

    def _result_step_count(self, data: Dict[str, Any], steps: List[Any]) -> int:
        for key in ("executed_step_count", "exit_step", "step_count"):
            value = self._coerce_numeric(data.get(key))
            if value is not None and value > 0:
                return int(value)
        return len(steps)

    def retrieve(self, query: str) -> Dict[str, Any]:
        if not self.enabled:
            self._record_memory_source("disabled")
            return {
                "memory_hits": [],
                "memory_confidence": 0.0,
                "memory_actions": [],
                "memory_source": "disabled",
            }

        if not self.records:
            target_task = self._extract_query_task(query)
            sakg_result = self._retrieve_sakg_only(query, target_task=target_task)
            if sakg_result is not None:
                return sakg_result
            self._record_memory_source("empty")
            return {
                "memory_hits": [],
                "memory_confidence": 0.0,
                "memory_actions": [],
                "memory_source": "empty",
            }

        eligible_records = [rec for rec in self.records if not self._is_low_value_record(rec)]
        target_task = self._extract_query_task(query)
        if target_task:
            eligible_records = [
                rec for rec in eligible_records
                if self._record_task_key(rec) == target_task
            ]
        if not eligible_records:
            sakg_result = self._retrieve_sakg_only(query, target_task=target_task)
            if sakg_result is not None:
                return sakg_result
            self._record_memory_source("no_task_records" if target_task else "no_high_value_records")
            return {
                "memory_hits": [],
                "memory_confidence": 0.0,
                "memory_actions": [],
                "memory_source": "no_task_records" if target_task else "no_high_value_records",
            }

        ranked = self._rank_primary_memory(query, records=eligible_records)
        successful_ranked = [
            (score, rec)
            for score, rec in ranked
            if self._safe_success_rate(rec) > 0.0 or bool(rec.get("success", False))
        ]
        if not successful_ranked:
            sakg_result = self._retrieve_sakg_only(query, target_task=target_task)
            if sakg_result is not None:
                return sakg_result
            self._record_memory_source("no_successful_records")
            return {
                "memory_hits": [],
                "memory_confidence": 0.0,
                "memory_actions": [],
                "memory_source": "no_successful_records",
            }
        if target_task:
            successful_ranked.sort(
                key=lambda item: (
                    self._success_experience_rank(item[1]),
                    -float(item[0]),
                )
            )
        ranked = successful_ranked

        fused = self._fuse_sakg_signal(query, ranked)
        if target_task:
            fused.sort(
                key=lambda item: (
                    self._success_experience_rank(item[1]),
                    -float(item[0]),
                )
            )
        top = fused[: self.max_results]

        hits = [item[1] for item in top]
        confidence = float(top[0][0]) if top else 0.0
        actions = hits[0].get("actions", []) if hits else []

        memory_source = "hybrid" if self.sa_kg_enabled else "primary"
        self._record_memory_source(memory_source)
        return {
            "memory_hits": hits,
            "memory_confidence": max(0.0, min(1.0, confidence)),
            "memory_actions": actions,
            "memory_source": memory_source,
        }

    def store(self, state: str, actions: List[str], success: bool, reward: float, metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self.enabled:
            self.record_store_skip("mem0_disabled")
            return

        cleaned_actions = [
            a.strip()
            for a in actions
            if isinstance(a, str)
            and a.strip()
            and a.strip().lower() != "nop()"
        ]
        if not cleaned_actions:
            self.record_store_skip("empty_actions")
            return

        candidate = {
            "state": state,
            "actions": cleaned_actions,
            "success": bool(success),
            "reward": float(reward),
            "reward_ema": float(reward),
            "attempts": 1,
            "successes": 1 if success else 0,
            "metadata": metadata or {},
        }
        if self._is_low_value_record(candidate):
            self.record_store_skip("low_value_record")
            return

        now = time.time()
        mem_key = self._build_memory_key(state, cleaned_actions)
        existing = self._find_by_key(mem_key)

        if existing is not None:
            existing["attempts"] = int(existing.get("attempts", 1)) + 1
            existing["successes"] = int(existing.get("successes", 1 if existing.get("success") else 0)) + (1 if success else 0)
            existing["success"] = bool(existing["successes"] > 0)
            existing["last_seen"] = now
            previous_reward_ema = float(existing.get("reward_ema", existing.get("reward", 0.0)))
            existing["reward_ema"] = 0.7 * previous_reward_ema + 0.3 * float(reward)
            if metadata:
                existing.setdefault("metadata", {}).update(metadata)
            record = existing
        else:
            record = {
                "id": f"mem_{sha1(f'{mem_key}:{now}'.encode('utf-8')).hexdigest()[:12]}",
                "key": mem_key,
                "state": state,
                "actions": cleaned_actions,
                "success": bool(success),
                "reward": float(reward),
                "reward_ema": float(reward),
                "attempts": 1,
                "successes": 1 if success else 0,
                "created_at": now,
                "last_seen": now,
                "metadata": metadata or {},
            }
            self.records.append(record)

        self._enforce_storage_governance(trigger="store")
        self._enforce_success_experience_cap(trigger="store")

        self._persist_records()
        self._record_store_committed()

        # Mirror to SA-KG feature layer (best-effort)
        if self.sa_kg_enabled and self.sa_kg is not None:
            try:
                for action in cleaned_actions:
                    self.sa_kg.add_experience(
                        state_description=state,
                        screenshot_path="",
                        action=action,
                        action_params={},
                        success=success,
                        metadata=metadata or {},
                    )
            except Exception as e:
                logger.warn(f"[Mem0] Failed to mirror record into SA-KG: {e}")

    def record_quick_path_decision(self, hit: bool, confidence: float = 0.0, memory_source: str = "unknown") -> None:
        qp = self.metrics.setdefault("quick_path", {})
        qp["total"] = int(qp.get("total", 0)) + 1
        if hit:
            qp["hits"] = int(qp.get("hits", 0)) + 1
        qp["last_confidence"] = float(confidence)
        qp["last_memory_source"] = memory_source
        total = max(1, int(qp.get("total", 1)))
        hits = int(qp.get("hits", 0))
        qp["hit_rate"] = round(hits / total, 6)
        self._persist_metrics()

    def record_store_skip(self, reason: str) -> None:
        self.metrics["store_attempts"] = int(self.metrics.get("store_attempts", 0)) + 1
        skips = self.metrics.setdefault("store_skip_reasons", {})
        skips[reason] = int(skips.get(reason, 0)) + 1
        self._persist_metrics()

    def _record_store_committed(self) -> None:
        self.metrics["store_attempts"] = int(self.metrics.get("store_attempts", 0)) + 1
        self.metrics["store_committed"] = int(self.metrics.get("store_committed", 0)) + 1
        self._persist_metrics()

    def _record_memory_source(self, source: str) -> None:
        source_dist = self.metrics.setdefault("memory_source_distribution", {})
        source_dist[source] = int(source_dist.get(source, 0)) + 1
        self.metrics["retrieval_total"] = int(self.metrics.get("retrieval_total", 0)) + 1
        self._persist_metrics()

    @staticmethod
    def _clean_actions(actions: Any) -> List[str]:
        if isinstance(actions, str):
            raw_actions = [actions]
        elif isinstance(actions, list):
            raw_actions = actions
        else:
            raw_actions = []

        cleaned: List[str] = []
        for action in raw_actions:
            text = str(action).strip()
            if not text or text.lower() == "nop()":
                continue
            cleaned.append(text)
        return cleaned

    @staticmethod
    def _extract_progress_from_state(state: str) -> str:
        match = re.search(r"(?:^|\|)\s*progress=(.*?)\s*(?:\||$)", str(state or ""))
        if not match:
            return ""
        return str(match.group(1) or "").strip()

    def _extract_progress_text(self, rec: Dict[str, Any]) -> str:
        metadata = rec.get("metadata", {})
        if isinstance(metadata, dict):
            progress = str(metadata.get("progress", "") or "").strip()
            if progress:
                return progress
        return self._extract_progress_from_state(str(rec.get("state", "") or ""))

    @staticmethod
    def _coerce_numeric(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

    @classmethod
    def _is_move_only_actions(cls, actions: Any) -> bool:
        cleaned = cls._clean_actions(actions)
        if not cleaned:
            return False
        return all(action.lower().startswith("move(") for action in cleaned)

    @classmethod
    def _extract_progress_snapshot(cls, rec: Dict[str, Any]) -> Dict[str, Any]:
        metadata = rec.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}

        completed_value = metadata.get("completed")
        completed = completed_value if isinstance(completed_value, bool) else None

        progress_delta = cls._coerce_numeric(metadata.get("progress_delta"))
        progress_quantity = cls._coerce_numeric(metadata.get("progress_quantity"))
        previous_progress_quantity = cls._coerce_numeric(
            metadata.get("previous_progress_quantity")
        )
        if (
            progress_delta is None
            and progress_quantity is not None
            and previous_progress_quantity is not None
        ):
            progress_delta = progress_quantity - previous_progress_quantity

        explicit_hard_progress = metadata.get("hard_progress")
        hard_progress = bool(explicit_hard_progress is True)
        if not hard_progress:
            hard_progress = bool(
                completed is True
                or (
                    progress_delta is not None
                    and progress_delta != 0.0
                )
                or (
                    progress_quantity is not None
                    and previous_progress_quantity is not None
                    and progress_quantity != previous_progress_quantity
                )
            )

        return {
            "completed": completed,
            "progress_delta": progress_delta,
            "progress_quantity": progress_quantity,
            "previous_progress_quantity": previous_progress_quantity,
            "hard_progress": hard_progress,
        }

    @staticmethod
    def _is_explicit_no_progress_text(progress_text: Any) -> bool:
        lowered = str(progress_text or "").strip().lower()
        if not lowered:
            return True

        negative_markers = (
            "executed action is none",
            "executed action was none",
            "no action was taken",
            "no action was executed",
            "no new action was performed",
            "no planting action has been performed yet",
            "no tilling action has been performed yet",
            "no fertilizer application has been performed yet",
            "no seeds have been planted yet",
            "no tiles have been tilled yet",
            "no tiles were tilled",
            "task remains incomplete",
            "task is not completed yet",
            "the task is not completed yet",
            "not completed yet",
            "not completed",
            "progress remains 0",
            "progress is 0",
            "progress = 0",
            "progress stayed at 0",
            "stayed at 0",
            "no progress",
            "did not make progress",
            "did not advance",
            "no observable effect",
            "without progress",
            "unsuccessful",
            "failed",
        )
        if any(marker in lowered for marker in negative_markers):
            return True

        if re.search(r"(?<!\d)0\s*/\s*\d+", lowered):
            return True
        if re.search(r"progress\s*(?:=|is|stayed at|remains?)\s*0(?:\D|$)", lowered):
            return True
        return False

    def _is_meaningful_progress_text(self, progress_text: Any) -> bool:
        if progress_text is None:
            return False
        normalized = str(progress_text).strip()
        if len(normalized) < self.progress_min_chars:
            return False

        lowered = normalized.lower()
        fraction_match = re.search(r"(?<!\d)(\d+)\s*/\s*(\d+)", lowered)
        if fraction_match:
            return int(fraction_match.group(1)) > 0

        delta_match = re.search(r"delta\s*=\s*(-?\d+(?:\.\d+)?)", lowered)
        if delta_match:
            return float(delta_match.group(1)) != 0.0

        change_match = re.search(
            r"(?:increased|changed)\s+from\s+(-?\d+(?:\.\d+)?)\s+to\s+(-?\d+(?:\.\d+)?)",
            lowered,
        )
        if change_match:
            return float(change_match.group(1)) != float(change_match.group(2))

        if "task is completed" in lowered:
            return True

        weak_patterns = {
            "none",
            "n/a",
            "unknown",
            "no progress",
            "no significant progress",
            "nothing changed",
            "same as before",
        }
        if lowered in weak_patterns:
            return False

        negative_markers = (
            "0/",
            "stayed at 0",
            "recorded task progress is 0",
            "task progress stayed at",
            "task is not completed yet",
            "not completed",
            "not yet completed",
            "no observable effect",
            "without progress",
            "no action was executed",
            "no new action was performed",
            "no fertilizer application has been performed yet",
            "no such action was executed",
            "no interaction",
            "did not advance",
            "did not make progress",
            "did not change",
            "still need",
            "still needs",
            "still missing",
            "still unavailable",
            "unsuccessful",
            "failed",
            "pending",
        )
        if any(marker in lowered for marker in negative_markers):
            return False

        positive_markers = (
            "task progress increased",
            "cleared ",
            "removed ",
            "chopped ",
            "cut down",
            "broken ",
            "harvested ",
            "collected ",
            "fertilized ",
            "planted ",
            "watered ",
            "tilled ",
            "mined ",
            "filled ",
            "petted ",
            "deposited ",
        )
        return any(marker in lowered for marker in positive_markers)

    @classmethod
    def _are_setup_only_actions(cls, actions: Any) -> bool:
        cleaned = cls._clean_actions(actions)
        if not cleaned:
            return False
        return all(
            action.lower().startswith(("choose_item(", "attach_item(", "unattach_item("))
            for action in cleaned
        )

    def _is_low_value_record(self, rec: Dict[str, Any]) -> bool:
        actions = self._clean_actions(rec.get("actions", []))
        if not actions:
            return True

        if self._are_setup_only_actions(actions):
            return True

        progress_snapshot = self._extract_progress_snapshot(rec)
        hard_progress = bool(progress_snapshot.get("hard_progress", False))
        move_only = self._is_move_only_actions(actions)
        if move_only and not hard_progress:
            return True

        reward_ema = float(rec.get("reward_ema", rec.get("reward", 0.0)) or 0.0)
        if reward_ema <= 0.0:
            return True

        progress = self._extract_progress_text(rec)
        if self._is_explicit_no_progress_text(progress):
            return True

        if hard_progress:
            return False

        if self._is_meaningful_progress_text(progress):
            return False

        success_like = bool(rec.get("success", False)) or self._safe_success_rate(rec) > 0.0
        if self.require_meaningful_progress and success_like:
            return True

        return True

    def _backup_storage_file(self, suffix: str) -> Optional[str]:
        if not os.path.exists(self.storage_path):
            return None
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = f"{self.storage_path}.{suffix}.{timestamp}.bak"
        try:
            shutil.copy2(self.storage_path, backup_path)
            return backup_path
        except Exception as e:
            logger.warn(f"[Mem0] Failed to create backup before compaction: {e}")
            return None

    def _compact_low_value_records(self, trigger: str = "runtime") -> None:
        if not self.records:
            return

        retained = [rec for rec in self.records if not self._is_low_value_record(rec)]
        removed = len(self.records) - len(retained)
        if removed <= 0:
            return

        backup_path = self._backup_storage_file("compact")
        self.records = retained
        self.metrics["records_compacted_total"] = int(self.metrics.get("records_compacted_total", 0)) + removed
        self.metrics["last_compaction_removed"] = removed
        if backup_path:
            self.metrics["last_compaction_backup"] = backup_path
        self._persist_records()
        self._persist_metrics()
        logger.write(
            f"[Mem0] Compacted {removed} low-value records (trigger={trigger}, backup={backup_path or 'none'})"
        )

    def _enforce_success_experience_cap(self, trigger: str = "runtime") -> None:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        updated_step_counts = False
        for rec in self.records:
            if not (bool(rec.get("success", False)) or self._safe_success_rate(rec) > 0.0):
                continue
            metadata = rec.get("metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            had_step_count = self._coerce_numeric(metadata.get("step_count")) is not None
            self._record_step_count(rec)
            metadata = rec.get("metadata", {})
            metadata = metadata if isinstance(metadata, dict) else {}
            has_step_count = self._coerce_numeric(metadata.get("step_count")) is not None
            updated_step_counts = updated_step_counts or (not had_step_count and has_step_count)
            key = self._success_experience_group_key(rec)
            if not key:
                continue
            groups.setdefault(key, []).append(rec)

        remove_ids = set()
        for records in groups.values():
            if len(records) <= 5:
                continue
            records.sort(key=self._success_experience_rank)
            for rec in records[5:]:
                remove_ids.add(id(rec))

        if not remove_ids:
            if updated_step_counts:
                self._persist_records()
            return

        backup_path = self._backup_storage_file("success_cap")
        before = len(self.records)
        self.records = [rec for rec in self.records if id(rec) not in remove_ids]
        removed = before - len(self.records)
        self.metrics["success_cap_removed_total"] = int(self.metrics.get("success_cap_removed_total", 0)) + removed
        self.metrics["last_success_cap_removed"] = removed
        if backup_path:
            self.metrics["last_success_cap_backup"] = backup_path
        self._persist_records()
        self._persist_metrics()
        logger.write(
            f"[Mem0] Enforced success experience cap: removed {removed} records "
            f"(max_per_task=5, trigger={trigger}, backup={backup_path or 'none'})"
        )

    def _success_experience_group_key(self, rec: Dict[str, Any]) -> str:
        return self._record_task_key(rec)

    @staticmethod
    def _is_result_backfill_record(rec: Dict[str, Any]) -> bool:
        metadata = rec.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        return str(metadata.get("store_source", "") or "") == "result_backfill"

    def _success_experience_rank(self, rec: Dict[str, Any]) -> Tuple[float, int, float]:
        step_count = self._record_step_count(rec)
        action_count = len(self._clean_actions(rec.get("actions", [])))
        last_seen = float(rec.get("last_seen", rec.get("created_at", 0.0)) or 0.0)
        return (step_count, action_count, -last_seen)

    def _record_step_count(self, rec: Dict[str, Any]) -> float:
        metadata = rec.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        for key in ("step_count", "executed_step_count", "source_step_count", "exit_step"):
            value = self._coerce_numeric(metadata.get(key))
            if value is not None and value > 0:
                if key == "step_count":
                    metadata["step_count"] = int(value)
                return float(value)

        step_count = self._lookup_result_step_count(metadata)
        if step_count is not None:
            metadata["step_count"] = int(step_count)
            rec["metadata"] = metadata
            return float(step_count)

        return float(len(self._clean_actions(rec.get("actions", []))) or 999999)

    def _lookup_result_step_count(self, metadata: Dict[str, Any]) -> Optional[int]:
        if str(metadata.get("store_source", "") or "") != "result_backfill":
            return None
        run_id = str(metadata.get("source_run_id", "") or "").strip()
        task_index = self._coerce_numeric(metadata.get("source_task_index"))
        if not run_id or task_index is None:
            return None

        run_dir = os.path.join(self._runs_results_root(), run_id)
        if not os.path.isdir(run_dir):
            return None

        prefix = f"task_{int(task_index):03d}_"
        try:
            for name in os.listdir(run_dir):
                result_path = os.path.join(run_dir, name, "result.json")
                if not name.startswith(prefix) or not os.path.exists(result_path):
                    continue
                with open(result_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                steps = data.get("steps", [])
                steps = steps if isinstance(steps, list) else []
                return self._result_step_count(data, steps)
        except Exception:
            return None
        return None

    def _enforce_storage_governance(self, trigger: str = "unknown") -> None:
        if len(self.records) <= self.max_records:
            return

        now = time.time()

        def record_priority(rec: Dict[str, Any]) -> float:
            success_rate = self._safe_success_rate(rec)
            reward_signal = max(-1.0, min(1.0, float(rec.get("reward_ema", rec.get("reward", 0.0)))))
            reward_norm = (reward_signal + 1.0) / 2.0
            last_seen = float(rec.get("last_seen", rec.get("created_at", now)))
            age_hours = max(0.0, (now - last_seen) / 3600.0)
            recency = math.exp(-age_hours / (self.recency_decay_hours * 2.0))
            return 0.45 * success_rate + 0.30 * reward_norm + 0.25 * recency

        before = len(self.records)
        self.records.sort(key=record_priority, reverse=True)
        self.records = self.records[: self.max_records]
        pruned = max(0, before - len(self.records))
        if pruned > 0:
            logger.write(f"[Mem0] Storage governance pruned {pruned} records (trigger={trigger})")
            self.metrics["records_pruned_total"] = int(self.metrics.get("records_pruned_total", 0)) + pruned
            self._persist_metrics()

    def _persist_records(self) -> None:
        tmp_path = None
        try:
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix=f"{os.path.basename(self.storage_path)}.",
                suffix=".tmp",
                dir=os.path.dirname(self.storage_path),
                text=True,
            )
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                for rec in self.records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(tmp_path, self.storage_path)
        except Exception as e:
            logger.warn(f"[Mem0] Failed to persist records: {e}")
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    def _rank_primary_memory(
        self,
        query: str,
        records: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for rec in records if records is not None else self.records:
            state_text = str(rec.get("state", ""))
            semantic = self._hybrid_similarity(query, state_text)
            reward_ema = float(rec.get("reward_ema", rec.get("reward", 0.0)))
            reward_signal = max(-1.0, min(1.0, reward_ema))
            success_rate = self._safe_success_rate(rec)
            recency = self._recency_factor(rec)

            confidence = semantic
            confidence += self.reward_weight * ((reward_signal + 1.0) / 2.0)
            confidence += self.success_weight * success_rate
            confidence *= recency

            rec["_score_primary"] = round(confidence, 6)
            rec["_score_semantic"] = round(semantic, 6)
            rec["_score_success_rate"] = round(success_rate, 6)
            rec["_score_recency"] = round(recency, 6)
            scored.append((confidence, rec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored

    def _fuse_sakg_signal(self, query: str, ranked: List[Tuple[float, Dict[str, Any]]]) -> List[Tuple[float, Dict[str, Any]]]:
        if not self.sa_kg_enabled or self.sa_kg is None:
            return ranked

        try:
            sakg_hits = self.sa_kg.retrieve_similar_states(query, top_k=self.max_results)
        except Exception as e:
            logger.warn(f"[Mem0] SA-KG retrieve failed: {e}")
            return ranked

        if not sakg_hits:
            return ranked

        action_to_boost: Dict[str, float] = {}
        for hit in sakg_hits:
            action_obj = hit.get("action")
            if not action_obj:
                continue
            action_name = getattr(action_obj, "action", "")
            if not action_name:
                continue
            similarity = float(hit.get("similarity", 0.0))
            success_rate = float(hit.get("success_rate", 0.0))
            boost = 0.6 * similarity + 0.4 * success_rate
            action_to_boost[action_name] = max(action_to_boost.get(action_name, 0.0), boost)

        fused: List[Tuple[float, Dict[str, Any]]] = []
        for base_score, rec in ranked:
            actions = rec.get("actions", [])
            sakg_boost = 0.0
            for action in actions:
                sakg_boost = max(sakg_boost, action_to_boost.get(action, 0.0))

            final_score = (1.0 - self.sa_kg_weight) * base_score + self.sa_kg_weight * sakg_boost
            rec["_score_sakg"] = round(sakg_boost, 6)
            rec["_score_final"] = round(final_score, 6)
            rec["_score_explain"] = (
                f"final={(1.0 - self.sa_kg_weight):.2f}*primary + {self.sa_kg_weight:.2f}*sakg"
            )
            fused.append((final_score, rec))

        fused.sort(key=lambda x: x[0], reverse=True)
        return fused

    def _retrieve_sakg_only(self, query: str, target_task: str = "") -> Optional[Dict[str, Any]]:
        hits = self._build_sakg_memory_records(query, target_task=target_task)
        if not hits:
            return None

        confidence = float(hits[0].get("_score_final", hits[0].get("_score_sakg", 0.0)) or 0.0)
        self._record_memory_source("sakg")
        return {
            "memory_hits": hits[: self.max_results],
            "memory_confidence": max(0.0, min(1.0, confidence)),
            "memory_actions": hits[0].get("actions", []),
            "memory_source": "sakg",
        }

    def _build_sakg_memory_records(self, query: str, target_task: str = "") -> List[Dict[str, Any]]:
        if not self.sa_kg_enabled or self.sa_kg is None:
            return []

        top_k = max(self.max_results * 3, self.max_results)
        try:
            sakg_hits = self.sa_kg.retrieve_similar_states(query, top_k=top_k)
        except Exception as e:
            logger.warn(f"[Mem0] SA-KG fallback retrieve failed: {e}")
            return []
        scan_hits = self._scan_sakg_states(query, top_k=top_k, target_task=target_task) if target_task or not sakg_hits else []
        if scan_hits:
            seen = set()
            merged_hits = []
            for hit in list(sakg_hits or []) + scan_hits:
                action_obj = hit.get("action") if isinstance(hit, dict) else None
                state_obj = hit.get("state") if isinstance(hit, dict) else None
                key = (
                    str(getattr(state_obj, "state_id", "") or ""),
                    str(getattr(action_obj, "edge_id", "") or getattr(action_obj, "action", "") or ""),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged_hits.append(hit)
            sakg_hits = merged_hits

        records: List[Dict[str, Any]] = []
        for hit in sakg_hits or []:
            if not isinstance(hit, dict):
                continue
            action_obj = hit.get("action")
            state_obj = hit.get("state")
            action_text = str(getattr(action_obj, "action", "") or "").strip()
            actions = self._clean_actions([action_text])
            if not actions:
                continue

            success_rate = float(hit.get("success_rate", getattr(action_obj, "success_rate", 0.0)) or 0.0)
            success_count = int(getattr(action_obj, "success_count", 0) or 0)
            action_success = bool(getattr(action_obj, "success", False)) or success_count > 0 or success_rate > 0.0
            if not action_success:
                continue

            metadata = getattr(state_obj, "metadata", {}) if state_obj is not None else {}
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            if target_task and self._normalize_text(str(metadata.get("task", "") or "")) != target_task:
                continue
            progress = str(metadata.get("progress", "") or "").strip()
            if progress and self._is_explicit_no_progress_text(progress):
                continue
            if self._are_setup_only_actions(actions):
                continue
            if self._is_move_only_actions(actions) and not (
                metadata.get("completed") is True
                or metadata.get("hard_progress") is True
                or self._is_meaningful_progress_text(progress)
            ):
                continue

            task = str(metadata.get("task", "") or "").strip()
            if not progress:
                progress = f"SA-KG recorded successful action for task: {task or 'historical task'}."

            state_desc = str(getattr(state_obj, "description", "") or "").strip()
            if not state_desc:
                state_desc = f"task={task} | progress={progress} | source=sakg"

            similarity = float(hit.get("similarity", 0.0) or 0.0)
            score = max(0.01, min(1.0, 0.6 * similarity + 0.4 * max(success_rate, 1.0 if action_success else 0.0)))
            rec_metadata = {
                **metadata,
                "progress": progress,
                "memory_source": "sakg",
                "store_source": metadata.get("store_source", "sakg"),
                "completed": metadata.get("completed"),
                "hard_progress": bool(metadata.get("hard_progress", False)),
                "setup_only": False,
                "move_only": self._is_move_only_actions(actions),
            }
            record = {
                "id": str(getattr(action_obj, "edge_id", "") or f"sakg_{sha1(f'{state_desc}:{action_text}'.encode('utf-8')).hexdigest()[:12]}"),
                "key": self._build_memory_key(state_desc, actions),
                "state": state_desc,
                "actions": actions,
                "success": True,
                "reward": float(getattr(action_obj, "reward", 1.0) or 1.0),
                "reward_ema": float(getattr(action_obj, "reward", 1.0) or 1.0),
                "attempts": max(1, int(getattr(action_obj, "execution_count", 1) or 1)),
                "successes": max(1, success_count),
                "created_at": float(getattr(action_obj, "timestamp", time.time()) or time.time()),
                "last_seen": float(getattr(action_obj, "timestamp", time.time()) or time.time()),
                "metadata": rec_metadata,
                "_score_sakg": round(score, 6),
                "_score_final": round(score, 6),
                "_score_explain": "sakg_fallback_success_reference",
            }
            records.append(record)

        records.sort(key=lambda rec: float(rec.get("_score_final", 0.0) or 0.0), reverse=True)
        return records

    def _scan_sakg_states(self, query: str, top_k: int, target_task: str = "") -> List[Dict[str, Any]]:
        states = getattr(self.sa_kg, "states", {}) if self.sa_kg is not None else {}
        actions = getattr(self.sa_kg, "actions", {}) if self.sa_kg is not None else {}
        if not isinstance(states, dict) or not isinstance(actions, dict):
            return []

        best_action_by_state: Dict[str, Any] = {}
        for action_obj in actions.values():
            if action_obj is None:
                continue
            state_id = str(getattr(action_obj, "from_state_id", "") or "")
            if not state_id:
                continue
            success_rate = float(getattr(action_obj, "success_rate", 0.0) or 0.0)
            success_count = int(getattr(action_obj, "success_count", 0) or 0)
            if not (bool(getattr(action_obj, "success", False)) or success_count > 0 or success_rate > 0.0):
                continue
            current = best_action_by_state.get(state_id)
            if current is None:
                best_action_by_state[state_id] = action_obj
                continue
            current_rate = float(getattr(current, "success_rate", 0.0) or 0.0)
            current_count = int(getattr(current, "execution_count", 0) or 0)
            action_count = int(getattr(action_obj, "execution_count", 0) or 0)
            if (success_rate, action_count) > (current_rate, current_count):
                best_action_by_state[state_id] = action_obj

        scored: List[Tuple[float, Dict[str, Any]]] = []
        for state_id, state_obj in states.items():
            action_obj = best_action_by_state.get(str(state_id))
            if action_obj is None:
                continue
            state_desc = str(getattr(state_obj, "description", "") or "")
            metadata = getattr(state_obj, "metadata", {})
            if isinstance(metadata, dict):
                if target_task and self._normalize_text(str(metadata.get("task", "") or "")) != target_task:
                    continue
                state_desc = f"{state_desc} task={metadata.get('task', '')} progress={metadata.get('progress', '')}"
            similarity = self._hybrid_similarity(query, state_desc)
            if similarity <= 0.0:
                continue
            scored.append((
                similarity,
                {
                    "state": state_obj,
                    "action": action_obj,
                    "similarity": similarity,
                    "success_rate": float(getattr(action_obj, "success_rate", 0.0) or 0.0),
                },
            ))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [hit for _score, hit in scored[:top_k]]

    def _find_by_key(self, key: str) -> Optional[Dict[str, Any]]:
        for rec in self.records:
            if rec.get("key") == key:
                return rec
        return None

    def _build_memory_key(self, state: str, actions: List[str]) -> str:
        normalized_state = self._normalize_text(state)
        normalized_actions = "|".join(sorted([self._normalize_text(a) for a in actions]))
        return sha1(f"{normalized_state}::{normalized_actions}".encode("utf-8")).hexdigest()

    def _safe_success_rate(self, rec: Dict[str, Any]) -> float:
        attempts = max(1, int(rec.get("attempts", 1)))
        successes = max(0, int(rec.get("successes", 1 if rec.get("success") else 0)))
        return min(1.0, successes / attempts)

    def _recency_factor(self, rec: Dict[str, Any]) -> float:
        now = time.time()
        last_seen = float(rec.get("last_seen", rec.get("created_at", now)))
        age_hours = max(0.0, (now - last_seen) / 3600.0)
        return math.exp(-age_hours / self.recency_decay_hours)

    def _hybrid_similarity(self, a: str, b: str) -> float:
        jaccard = self._jaccard_similarity(a, b)
        cosine = self._cosine_token_similarity(a, b)
        return 0.55 * cosine + 0.45 * jaccard

    def _cosine_token_similarity(self, a: str, b: str) -> float:
        ta = self._tokenize(a)
        tb = self._tokenize(b)
        if not ta or not tb:
            return 0.0

        fa: Dict[str, float] = {}
        fb: Dict[str, float] = {}
        for token in ta:
            fa[token] = fa.get(token, 0.0) + 1.0
        for token in tb:
            fb[token] = fb.get(token, 0.0) + 1.0

        dot = 0.0
        for token, v in fa.items():
            dot += v * fb.get(token, 0.0)
        na = math.sqrt(sum(v * v for v in fa.values()))
        nb = math.sqrt(sum(v * v for v in fb.values()))
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / (na * nb)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        normalized = Mem0Provider._normalize_text(text)
        if not normalized:
            return []
        return normalized.split()

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s\u4e00-\u9fff]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text

    @classmethod
    def _extract_query_task(cls, query: str) -> str:
        match = re.search(r"(?:^|\|)\s*task\s*[:=]\s*([^|\n\r]+)", str(query or ""), re.IGNORECASE)
        if not match:
            return ""
        return cls._normalize_text(str(match.group(1) or ""))

    @classmethod
    def _record_task_key(cls, rec: Dict[str, Any]) -> str:
        metadata = rec.get("metadata", {})
        metadata = metadata if isinstance(metadata, dict) else {}
        task = str(metadata.get("task", "") or "").strip()
        if not task:
            state = str(rec.get("state", "") or "")
            match = re.search(r"(?:^|\|)\s*task=(.*?)\s*(?:\||$)", state)
            if match:
                task = str(match.group(1) or "").strip()
        return cls._normalize_text(task)

    def _load_records(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.storage_path):
            return []

        records_by_key: Dict[str, Dict[str, Any]] = {}
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict):
                            normalized = self._normalize_record(rec)
                            if normalized is None:
                                continue
                            key = str(normalized.get("key") or normalized.get("id") or "")
                            if not key:
                                continue
                            existing = records_by_key.get(key)
                            if existing is None or float(normalized.get("last_seen", 0.0)) >= float(existing.get("last_seen", 0.0)):
                                records_by_key[key] = normalized
                    except Exception:
                        continue
        except Exception as e:
            logger.warn(f"[Mem0] Failed to load records: {e}")
        return list(records_by_key.values())

    def _normalize_record(self, rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            state = str(rec.get("state", "")).strip()
            actions_raw = rec.get("actions", [])
            actions = [str(a).strip() for a in actions_raw if str(a).strip()]
            if not actions:
                return None

            now = time.time()
            attempts = max(1, int(rec.get("attempts", 1)))
            successes = max(0, int(rec.get("successes", 1 if rec.get("success") else 0)))
            reward = float(rec.get("reward", 0.0))
            reward_ema = float(rec.get("reward_ema", reward))
            created_at = float(rec.get("created_at", now))
            last_seen = float(rec.get("last_seen", created_at))

            key = str(rec.get("key") or self._build_memory_key(state, actions))
            rec_id = str(rec.get("id") or f"mem_{sha1(f'{key}:{created_at}'.encode('utf-8')).hexdigest()[:12]}")

            return {
                "id": rec_id,
                "key": key,
                "state": state,
                "actions": actions,
                "success": bool(successes > 0),
                "reward": reward,
                "reward_ema": reward_ema,
                "attempts": attempts,
                "successes": successes,
                "created_at": created_at,
                "last_seen": last_seen,
                "metadata": rec.get("metadata", {}) if isinstance(rec.get("metadata", {}), dict) else {},
            }
        except Exception:
            return None

    def _load_metrics(self) -> Dict[str, Any]:
        default = {
            "retrieval_total": 0,
            "memory_source_distribution": {},
            "quick_path": {"total": 0, "hits": 0, "hit_rate": 0.0},
            "store_attempts": 0,
            "store_committed": 0,
            "store_skip_reasons": {},
            "records_compacted_total": 0,
            "last_compaction_removed": 0,
            "success_cap_removed_total": 0,
            "last_success_cap_removed": 0,
            "records_pruned_total": 0,
            "updated_at": time.time(),
        }

        if not os.path.exists(self.metrics_path):
            return default

        try:
            with open(self.metrics_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if not isinstance(loaded, dict):
                return default
            default.update(loaded)
            return default
        except Exception as e:
            logger.warn(f"[Mem0] Failed to load metrics: {e}")
            return default

    def _persist_metrics(self) -> None:
        tmp_path = None
        try:
            self.metrics["updated_at"] = time.time()
            tmp_fd, tmp_path = tempfile.mkstemp(
                prefix=f"{os.path.basename(self.metrics_path)}.",
                suffix=".tmp",
                dir=os.path.dirname(self.metrics_path),
                text=True,
            )
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                json.dump(self.metrics, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.metrics_path)
        except Exception as e:
            logger.warn(f"[Mem0] Failed to persist metrics: {e}")
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        a_tokens = set(a.lower().split())
        b_tokens = set(b.lower().split())
        if not a_tokens or not b_tokens:
            return 0.0
        intersection = a_tokens.intersection(b_tokens)
        union = a_tokens.union(b_tokens)
        return len(intersection) / max(1, len(union))
