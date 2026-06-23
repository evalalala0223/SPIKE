"""Structured, replayable event log for the env agent.

Borrowed from the OpenHands SDK idea: every meaningful step emits an *immutable,
serialisable* event appended to a ``events.jsonl``.  This makes an env-agent run
fully reconstructable (what knowledge was retrieved, which pitfalls were read,
why each task was generated) instead of having to grep free-form logs.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class Event:
    seq: int
    ts: float
    type: str
    summary: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class EventLog:
    """Append-only event sink. Thread-unsafe by design (one run = one log)."""

    def __init__(self, path: str | Path, *, echo: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._echo = echo
        self._fh = self.path.open("a", encoding="utf-8")

    def emit(self, type: str, summary: str = "", **payload: Any) -> Event:
        ev = Event(seq=self._seq, ts=time.time(), type=type, summary=summary, payload=payload)
        self._seq += 1
        self._fh.write(ev.to_json() + "\n")
        self._fh.flush()
        if self._echo:
            print(f"[env-agent] {ev.type}: {ev.summary}")
        return ev

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def replay(path: str | Path) -> list[Event]:
    """Load an events.jsonl back into Event objects (for inspection/debugging)."""
    out: list[Event] = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        out.append(Event(**d))
    return out
