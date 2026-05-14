from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agent"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))
