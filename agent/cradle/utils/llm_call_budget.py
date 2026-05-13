from __future__ import annotations

import sys
from collections import defaultdict
from threading import Lock
from typing import Dict


_COUNTER_LOCK = Lock()
_LLM_CALL_COUNT = 0
_LLM_CALL_BREAKDOWN = defaultdict(int)

sys.modules.setdefault("cradle.utils.llm_call_budget", sys.modules[__name__])
sys.modules.setdefault("agent.cradle.utils.llm_call_budget", sys.modules[__name__])


def reset_llm_call_counter() -> None:
    global _LLM_CALL_COUNT, _LLM_CALL_BREAKDOWN
    with _COUNTER_LOCK:
        _LLM_CALL_COUNT = 0
        _LLM_CALL_BREAKDOWN = defaultdict(int)


def increment_llm_call_counter(source: str) -> int:
    global _LLM_CALL_COUNT
    normalized_source = str(source or "unknown").strip() or "unknown"
    with _COUNTER_LOCK:
        _LLM_CALL_COUNT += 1
        _LLM_CALL_BREAKDOWN[normalized_source] += 1
        return _LLM_CALL_COUNT


def get_llm_call_count() -> int:
    with _COUNTER_LOCK:
        return int(_LLM_CALL_COUNT)


def get_llm_call_breakdown() -> Dict[str, int]:
    with _COUNTER_LOCK:
        return dict(_LLM_CALL_BREAKDOWN)
