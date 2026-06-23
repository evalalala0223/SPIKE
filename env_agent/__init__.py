"""SPIKE Environment Agent (env_agent).

A task-decomposition / curriculum agent that sits *above* the SPIKE game agent.

Responsibilities:
  1. Game cognition   - understands the existing SPIKE task pool and its relations.
  2. Knowledge QA     - retrieves background knowledge from an offline Stardew
                        Valley wiki and (optionally) the web.
  3. Experience loop  - reads the game agent's run logs / memory, summarises the
                        situations where it repeatedly fails into a markdown
                        "pitfalls" file, and feeds that back into decomposition.
  4. Task decomposition - turns a high level goal into SPIKE-compatible sub-task
                        YAML, validated against the real task schema.
  5. Tooling          - performs file/skill operations, optionally executed
                        through the OpenHands Software Agent SDK.
"""

from .core.orchestrator import EnvAgent

__all__ = ["EnvAgent"]
__version__ = "0.1.0"
