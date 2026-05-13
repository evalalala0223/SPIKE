from __future__ import annotations

import threading


_TIMING_LOCAL = threading.local()


def reset_llm_retry_timing_accounting() -> None:
    _TIMING_LOCAL.retry_overhead_s = 0.0


def add_llm_retry_overhead(retry_overhead_s: float) -> None:
    current = float(getattr(_TIMING_LOCAL, "retry_overhead_s", 0.0) or 0.0)
    _TIMING_LOCAL.retry_overhead_s = current + max(0.0, float(retry_overhead_s or 0.0))


def consume_llm_retry_overhead_s() -> float:
    retry_overhead_s = float(getattr(_TIMING_LOCAL, "retry_overhead_s", 0.0) or 0.0)
    _TIMING_LOCAL.retry_overhead_s = 0.0
    return max(0.0, retry_overhead_s)
