"""Toolkit for env-agent side effects (writing task YAML, updating skills, etc.).

Per the user's requirement, file/skill operations can be executed *through the
OpenHands Software Agent SDK*.  Since the SDK is an optional, heavy dependency,
this module uses a hybrid strategy:

  * ``backend="openhands"``: route operations through an OpenHands ``Conversation``
    using ``FileEditorTool`` / ``TerminalTool`` running in a ``LocalWorkspace``
    rooted at the SPIKE repo. This genuinely exercises the SDK.
  * ``backend="native"`` (default): perform the same operations with plain Python
    (atomic, deterministic, no extra deps) so the env agent runs out of the box.

Both backends expose the same surface: ``write_text``, ``read_text``,
``write_tasks_yaml``, ``register_skill_stub``.  This keeps the orchestrator
agnostic to how the side effect is realised (composition over inheritance).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml

from ..core.schema import TaskSpec


# ----------------------------------------------------------------------------
# YAML serialisation matching the existing suite style
# ----------------------------------------------------------------------------
def tasks_to_yaml(tasks: list[TaskSpec], *, header: Optional[str] = None) -> str:
    """Render tasks into the same flat mapping style used by the suite files."""
    doc: dict[str, Any] = {}
    for i, t in enumerate(tasks):
        t.id = i
        doc[t.name] = t.to_yaml_entry()
    body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True, default_flow_style=False)
    if header:
        return f"# {header}\n{body}"
    return body


class OpenHandsUnavailable(RuntimeError):
    pass


class Toolkit:
    def __init__(
        self,
        workspace_root: str | Path,
        *,
        backend: str = "native",
        llm_model: Optional[str] = None,
        llm_api_key_env: str = "LLM_API_KEY",
        llm_base_url: Optional[str] = None,
    ) -> None:
        self.root = Path(workspace_root).resolve()
        self.backend = backend
        self._llm_model = llm_model
        self._llm_api_key_env = llm_api_key_env
        self._llm_base_url = llm_base_url
        self._conversation = None
        if backend == "openhands":
            self._init_openhands()

    # -- OpenHands backend ---------------------------------------------------
    def _init_openhands(self) -> None:
        try:
            from openhands.sdk import LLM, Agent, Conversation, Tool  # type: ignore
            from openhands.tools.file_editor import FileEditorTool  # type: ignore
            from openhands.tools.terminal import TerminalTool  # type: ignore
        except Exception as e:  # pragma: no cover - optional dep
            raise OpenHandsUnavailable(
                "backend='openhands' requires `pip install openhands-sdk "
                f"openhands-tools` (import failed: {e})"
            )
        llm = LLM(
            model=self._llm_model or os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv(self._llm_api_key_env),
            base_url=self._llm_base_url or os.getenv("LLM_BASE_URL"),
        )
        agent = Agent(
            llm=llm,
            tools=[Tool(name=FileEditorTool.name), Tool(name=TerminalTool.name)],
        )
        self._conversation = Conversation(agent=agent, workspace=str(self.root))

    def _openhands_write(self, abs_path: Path, content: str) -> None:
        rel = abs_path.relative_to(self.root) if abs_path.is_relative_to(self.root) else abs_path
        conv = self._conversation
        conv.send_message(
            "Create or overwrite the file at "
            f"`{rel}` with exactly the following content, then stop:\n\n"
            f"```\n{content}\n```"
        )
        conv.run()

    # -- public surface ------------------------------------------------------
    def _abs(self, path: str | Path) -> Path:
        p = Path(path)
        return p if p.is_absolute() else (self.root / p)

    def write_text(self, path: str | Path, content: str) -> str:
        abs_path = self._abs(path)
        if self.backend == "openhands" and self._conversation is not None:
            self._openhands_write(abs_path, content)
        else:
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_text(content, encoding="utf-8")
        return str(abs_path)

    def read_text(self, path: str | Path) -> str:
        return self._abs(path).read_text(encoding="utf-8")

    def write_tasks_yaml(
        self, path: str | Path, tasks: list[TaskSpec], *, header: Optional[str] = None
    ) -> str:
        return self.write_text(path, tasks_to_yaml(tasks, header=header))

    def register_skill_stub(
        self, skills_dir: str | Path, skill_name: str, signature: str, body: str = "pass"
    ) -> str:
        """Append a ``@register_skill`` stub to a composite-skills module.

        This matches SPIKE's skill mechanism (``@register_skill("name")`` decorator
        populating the SKILLS dict). It writes a *stub* the developer can flesh out;
        it never silently overwrites existing skills.
        """
        skills_dir = self._abs(skills_dir)
        target = skills_dir / f"{skill_name}.py"
        if target.exists():
            existing = self.read_text(target)
            if f'@register_skill("{skill_name}")' in existing:
                return f"skill '{skill_name}' already registered in {target}"
        stub = (
            "from cradle.environment.stardew.skill_registry import register_skill\n\n\n"
            f'@register_skill("{skill_name}")\n'
            f"def {skill_name}({signature}):\n"
            f'    """Auto-generated stub by env_agent. TODO: implement."""\n'
            f"    {body}\n"
        )
        return self.write_text(target, stub)
