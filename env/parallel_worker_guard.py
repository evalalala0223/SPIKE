from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass(frozen=True)
class ParallelWorkerLimitDecision:
    requested_workers: int
    effective_workers: int
    throttle_enabled: bool
    throttle_max_concurrency: Optional[int]
    model_name: str
    model_matched: bool
    queue_enforced: bool

    @property
    def limited(self) -> bool:
        return self.effective_workers < self.requested_workers


def _coerce_positive_int(value: Any, fallback: int) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return max(1, int(fallback))


def _resolve_path(root_dir: Path, raw_path: str) -> Path:
    candidate = Path(str(raw_path or "").strip())
    if not candidate.is_absolute():
        candidate = (root_dir / candidate).resolve()
    return candidate


def _load_yaml(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fd:
            payload = yaml.safe_load(fd) or {}
            if isinstance(payload, dict):
                return payload
    except Exception:
        pass
    return {}


def _load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fd:
            payload = json.load(fd) or {}
            if isinstance(payload, dict):
                return payload
    except Exception:
        pass
    return {}


def _extract_model_name(llm_payload: dict) -> str:
    if not isinstance(llm_payload, dict):
        return ""
    for key in ("comp_model", "model", "deployment_name"):
        value = str(llm_payload.get(key, "") or "").strip()
        if value:
            return value
    return ""


def resolve_parallel_worker_limit(
    requested_workers: int,
    *,
    root_dir: Path,
    llm_config_path: str,
    enhanced_config_path: str = "agent/conf/enhanced_config.yaml",
) -> ParallelWorkerLimitDecision:
    requested = _coerce_positive_int(requested_workers, fallback=1)
    resolved_root = Path(root_dir).resolve()

    enhanced_config_path = (
        os.getenv("STARDOJO_ENHANCED_CONFIG", "").strip()
        or enhanced_config_path
    )
    enhanced_cfg = _load_yaml(_resolve_path(resolved_root, enhanced_config_path))
    throttle_cfg = (
        ((enhanced_cfg.get("performance") or {}).get("llm_endpoint_throttle") or {})
        if isinstance(enhanced_cfg, dict)
        else {}
    )
    throttle_enabled = bool(throttle_cfg.get("enabled", False))
    if not throttle_enabled:
        return ParallelWorkerLimitDecision(
            requested_workers=requested,
            effective_workers=requested,
            throttle_enabled=False,
            throttle_max_concurrency=None,
            model_name="",
            model_matched=False,
            queue_enforced=False,
        )

    max_concurrency = _coerce_positive_int(
        throttle_cfg.get("max_concurrency", 1),
        fallback=1,
    )
    raw_patterns = throttle_cfg.get("model_name_substrings", [])
    if not isinstance(raw_patterns, list):
        raw_patterns = []
    patterns = [str(item or "").strip().lower() for item in raw_patterns if str(item or "").strip()]

    llm_cfg = _load_json(_resolve_path(resolved_root, llm_config_path))
    model_name = _extract_model_name(llm_cfg)
    normalized_model = model_name.lower()

    if not patterns:
        model_matched = True
    else:
        model_matched = bool(normalized_model and any(pattern in normalized_model for pattern in patterns))

    model_unknown = not bool(normalized_model)
    queue_enforced = bool(model_matched or model_unknown)
    effective_workers = requested

    return ParallelWorkerLimitDecision(
        requested_workers=requested,
        effective_workers=effective_workers,
        throttle_enabled=True,
        throttle_max_concurrency=max_concurrency,
        model_name=model_name,
        model_matched=model_matched,
        queue_enforced=queue_enforced,
    )
