from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, Optional, Tuple

import yaml

from cradle.utils.file_utils import assemble_project_path


_CONFIG_CACHE: Optional[Dict[str, Any]] = None


class LLMEndpointThrottleTimeout(TimeoutError):
    """Raised when a throttled endpoint cannot obtain a shared slot in time."""

    def __init__(
        self,
        *,
        model_name: str,
        purpose: str,
        waited_s: float,
        max_concurrency: int,
    ) -> None:
        self.model_name = str(model_name or "")
        self.purpose = str(purpose or "")
        self.waited_s = max(0.0, float(waited_s))
        self.max_concurrency = max(1, int(max_concurrency))
        super().__init__(
            f"shared slot timeout after {self.waited_s:.1f}s "
            f"for {self.model_name} ({self.purpose}), max_concurrency={self.max_concurrency}"
        )


def _load_throttle_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    defaults: Dict[str, Any] = {
        "enabled": False,
        "max_concurrency": 1,
        "model_name_substrings": ["qwen3.5-plus", "qwen-plus", "qwen"],
        "slot_dir": assemble_project_path("./cache/locks/llm_endpoint_slots"),
        "timeout_s": 180,
        "wait_budget_ratio": 0.5,
        "min_request_window_s": 20.0,
        "poll_interval_s": 0.25,
        "stale_after_s": 600,
        "release_linger_s": 0.0,
    }

    try:
        config_path = os.getenv("STARDOJO_ENHANCED_CONFIG", "").strip()
        config_path = assemble_project_path(config_path) if config_path else assemble_project_path("./conf/enhanced_config.yaml")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            throttle_cfg = (((cfg.get("performance") or {}).get("llm_endpoint_throttle") or {}))
            defaults.update({
                "enabled": bool(throttle_cfg.get("enabled", defaults["enabled"])),
                "max_concurrency": max(1, int(throttle_cfg.get("max_concurrency", defaults["max_concurrency"]))),
                "model_name_substrings": list(throttle_cfg.get("model_name_substrings", defaults["model_name_substrings"])) or defaults["model_name_substrings"],
                "slot_dir": assemble_project_path(throttle_cfg.get("slot_dir", "./cache/locks/llm_endpoint_slots")),
                "timeout_s": max(1, int(throttle_cfg.get("timeout_s", defaults["timeout_s"]))),
                "wait_budget_ratio": min(0.9, max(0.05, float(throttle_cfg.get("wait_budget_ratio", defaults["wait_budget_ratio"])))),
                "min_request_window_s": max(1.0, float(throttle_cfg.get("min_request_window_s", defaults["min_request_window_s"]))),
                "poll_interval_s": max(0.05, float(throttle_cfg.get("poll_interval_s", defaults["poll_interval_s"]))),
                "stale_after_s": max(30, int(throttle_cfg.get("stale_after_s", defaults["stale_after_s"]))),
                "release_linger_s": max(0.0, float(throttle_cfg.get("release_linger_s", defaults["release_linger_s"]))),
            })
    except Exception:
        pass

    _CONFIG_CACHE = defaults
    return _CONFIG_CACHE


def _log(logger_obj: Any, level: str, message: str) -> None:
    try:
        if logger_obj is None:
            return
        if level == "warn" and hasattr(logger_obj, "warn"):
            logger_obj.warn(message)
            return
        if hasattr(logger_obj, "write"):
            logger_obj.write(message)
    except Exception:
        pass


def _model_is_throttled(model_name: str, cfg: Dict[str, Any]) -> bool:
    name = str(model_name or "").strip().lower()
    if not name:
        return False
    for pattern in cfg.get("model_name_substrings", []):
        if str(pattern).strip().lower() in name:
            return True
    return False


def get_llm_endpoint_max_concurrency(model_name: str = "") -> Optional[int]:
    cfg = _load_throttle_config()
    if not cfg.get("enabled"):
        return None
    if model_name and not _model_is_throttled(model_name, cfg):
        return None
    return max(1, int(cfg.get("max_concurrency", 1)))


def get_llm_endpoint_wait_timeout(
    model_name: str,
    *,
    total_timeout_s: Optional[float] = None,
) -> Optional[float]:
    cfg = _load_throttle_config()
    if not cfg.get("enabled") or not _model_is_throttled(model_name, cfg):
        return None

    throttle_timeout_s = max(1.0, float(cfg.get("timeout_s", 180)))
    if total_timeout_s is None:
        return throttle_timeout_s

    total_timeout_s = max(1.0, float(total_timeout_s))
    wait_budget_ratio = min(0.9, max(0.05, float(cfg.get("wait_budget_ratio", 0.5))))
    min_request_window_s = max(1.0, float(cfg.get("min_request_window_s", 20.0)))
    max_wait_from_ratio = total_timeout_s * wait_budget_ratio
    max_wait_after_reserving_request = total_timeout_s - min_request_window_s

    if max_wait_after_reserving_request > 0:
        effective_wait_budget = min(max_wait_from_ratio, max_wait_after_reserving_request)
    else:
        effective_wait_budget = total_timeout_s * min(wait_budget_ratio, 0.25)

    return max(1.0, min(throttle_timeout_s, effective_wait_budget))


def resolve_remaining_llm_request_timeout(
    total_timeout_s: float,
    slot_metadata: Optional[Dict[str, Any]],
    *,
    minimum_timeout_s: float = 1.0,
) -> float:
    total_timeout = max(minimum_timeout_s, float(total_timeout_s))
    waited_s = 0.0
    if isinstance(slot_metadata, dict):
        try:
            waited_s = max(0.0, float(slot_metadata.get("waited_s", 0.0)))
        except (TypeError, ValueError):
            waited_s = 0.0
    return max(minimum_timeout_s, total_timeout - waited_s)


def _acquire_slot(
    model_name: str,
    purpose: str,
    logger_obj: Any = None,
    *,
    timeout_s: Optional[float] = None,
) -> Tuple[Optional[int], Optional[str], float, Optional[int]]:
    cfg = _load_throttle_config()
    if not cfg.get("enabled") or not _model_is_throttled(model_name, cfg):
        return None, None, 0.0, None

    slot_dir = str(cfg.get("slot_dir") or "")
    if not slot_dir:
        return None, None, 0.0, None
    os.makedirs(slot_dir, exist_ok=True)

    max_concurrency = max(1, int(cfg.get("max_concurrency", 1)))
    effective_timeout_s = (
        max(1.0, float(timeout_s))
        if timeout_s is not None
        else float(cfg.get("timeout_s", 180))
    )
    poll_interval_s = float(cfg.get("poll_interval_s", 0.25))
    stale_after_s = float(cfg.get("stale_after_s", 600))

    start = time.time()
    wait_logged = False
    while True:
        now = time.time()
        for slot_idx in range(max_concurrency):
            slot_path = os.path.join(slot_dir, f"slot_{slot_idx}.lock")
            try:
                fd = os.open(slot_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                payload = {
                    "pid": os.getpid(),
                    "model": model_name,
                    "purpose": purpose,
                    "time": now,
                }
                os.write(fd, json.dumps(payload, ensure_ascii=False).encode("utf-8"))
                if wait_logged:
                    _log(logger_obj, "write", f"[LLMThrottle] Acquired shared slot for {model_name} ({purpose}) after {time.time() - start:.1f}s")
                return fd, slot_path, max(0.0, time.time() - start), max_concurrency
            except FileExistsError:
                try:
                    age = now - os.path.getmtime(slot_path)
                    if age > stale_after_s:
                        _log(logger_obj, "warn", f"[LLMThrottle] Removing stale slot {slot_idx} for {model_name} (age={age:.1f}s)")
                        os.remove(slot_path)
                        continue
                except FileNotFoundError:
                    continue

        waited = time.time() - start
        if waited >= effective_timeout_s:
            _log(
                logger_obj,
                "warn",
                f"[LLMThrottle] Waited {waited:.1f}s for {model_name} ({purpose}); refusing to exceed shared limit {max_concurrency}",
            )
            raise LLMEndpointThrottleTimeout(
                model_name=model_name,
                purpose=purpose,
                waited_s=waited,
                max_concurrency=max_concurrency,
            )

        if not wait_logged:
            _log(logger_obj, "write", f"[LLMThrottle] Waiting for shared slot: model={model_name}, purpose={purpose}")
            wait_logged = True
        time.sleep(poll_interval_s)


def _release_slot(
    lock_fd: Optional[int],
    lock_path: Optional[str],
    *,
    linger_s: float = 0.0,
) -> None:
    if lock_fd is not None and lock_path and linger_s > 0:
        try:
            time.sleep(max(0.0, float(linger_s)))
        except Exception:
            pass
    if lock_fd is not None:
        try:
            os.close(lock_fd)
        except Exception:
            pass
    if lock_path:
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except FileNotFoundError:
            pass


@contextmanager
def acquire_llm_endpoint_slot(
    model_name: str,
    purpose: str,
    logger_obj: Any = None,
    *,
    timeout_s: Optional[float] = None,
) -> Iterator[Dict[str, Any]]:
    lock_fd: Optional[int] = None
    lock_path: Optional[str] = None
    slot_metadata: Dict[str, Any] = {
        "waited_s": 0.0,
        "throttled": False,
        "max_concurrency": get_llm_endpoint_max_concurrency(model_name),
    }
    try:
        lock_fd, lock_path, waited_s, max_concurrency = _acquire_slot(
            model_name=model_name,
            purpose=purpose,
            logger_obj=logger_obj,
            timeout_s=timeout_s,
        )
        slot_metadata = {
            "waited_s": waited_s,
            "throttled": bool(max_concurrency),
            "max_concurrency": max_concurrency,
        }
        yield slot_metadata
    finally:
        linger_s = 0.0
        if lock_fd is not None and lock_path:
            cfg = _load_throttle_config()
            try:
                linger_s = max(0.0, float(cfg.get("release_linger_s", 0.0)))
            except (TypeError, ValueError):
                linger_s = 0.0
        _release_slot(lock_fd, lock_path, linger_s=linger_s)
