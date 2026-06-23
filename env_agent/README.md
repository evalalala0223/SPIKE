# SPIKE Environment Agent (`env_agent`)

A **task-decomposition / curriculum agent** that sits *above* the SPIKE game
agent. It understands the game and the existing task pool, retrieves background
knowledge from an offline Stardew Valley wiki (with optional web fallback),
reads the game agent's run history to learn where it repeatedly fails, and then
generates **SPIKE-compatible sub-task curricula** — optionally performing the
file/skill operations through the **OpenHands Software Agent SDK**.

```
                ┌──────────────────────────── EnvAgent ────────────────────────────┐
   goal ─────▶  │ 1.Experience → 2.Cognition → 3.Knowledge → 4.Decompose → 5.Persist │ ─▶ tasks.yaml
                └───────────────────────────────────────────────────────────────────┘
                     │             │              │             │            │
              result.json     task_suite/*    offline wiki    LLM       Toolkit
              (run history)   (task pool)     (+ web QA)   (validated)  (native│OpenHands SDK)
```

## Why this design

It borrows three ideas from the OpenHands SDK we reviewed:

| OpenHands idea | How it shows up here |
|---|---|
| **Typed Action/Observation contract** | LLM output is coerced into validated `TaskSpec` objects (`core/schema.py`) before anything uses it. Invalid tasks are dropped, not run. |
| **Replayable Event stream** | every stage emits an immutable event to `runs/<stamp>/events.jsonl` so a decomposition run is fully reconstructable. |
| **Explicit state, no singletons** | modules receive their state explicitly (unlike SPIKE's `LocalMemory` singleton), so multiple env-agent runs never cross-contaminate. |

## Layout

```
env_agent/
├── config.yaml              # all paths & options
├── run_env_agent.py         # CLI
├── core/
│   ├── schema.py            # TaskSpec + valid evaluators/skills/saves + validation
│   ├── llm_client.py        # OpenAI-compatible client reusing SPIKE's gemini proxy
│   ├── events.py            # replayable events.jsonl
│   └── orchestrator.py      # EnvAgent pipeline
├── modules/
│   ├── task_pool.py         # cognition over env/tasks/task_suite/*.yaml
│   ├── knowledge_base.py    # offline wiki retrieval + grounded QA (+ web hook)
│   ├── experience.py        # result.json → failure aggregation → pitfalls.md
│   └── decomposer.py        # LLM → validated TaskSpec curriculum
├── tools/
│   └── sdk_tools.py         # Toolkit: native OR OpenHands-SDK-backed file/skill ops
├── knowledge/
│   ├── agent_pitfalls.md    # auto-maintained failure summary
│   └── wiki_cache/          # offline wiki (after `sync-wiki`)
├── generated/               # generated sub-task suites (*.yaml + *.md rationale)
└── runs/                    # per-run events.jsonl
```

## Quick start

All commands run from the SPIKE repo root using the project venv.

```bash
# 0. (optional) pull the offline wiki into the cache (uses git clone)
./.venv/bin/python -m env_agent.run_env_agent sync-wiki

# 1. learn from history: build/refresh the pitfalls summary
./.venv/bin/python -m env_agent.run_env_agent analyze

# 2. ask the knowledge base something
./.venv/bin/python -m env_agent.run_env_agent ask "How long does a parsnip take to grow?"

# 3. inspect the existing task pool
./.venv/bin/python -m env_agent.run_env_agent pool

# 4. decompose a high-level goal into a runnable curriculum
./.venv/bin/python -m env_agent.run_env_agent decompose \
    --goal "Teach the agent to grow and harvest parsnips" --n 6
```

The generated `env_agent/generated/<goal>_<stamp>.yaml` uses **exactly** the
schema of the existing suites (`object/quantity/tool/save/init_commands/
evaluator/difficulty`), so you can point the SPIKE runner at it.

## Configuration (`config.yaml`)

- `llm.config` — reuses `agent/conf/gemini_config.json` (model + LiteLLM proxy
  `base_url`). The API key is read from the process env or `env/.env` using the
  `key_var` in that json. **Without a key the agent still runs** (experience
  analysis, retrieval, validation, file writing all work); only LLM-driven QA
  and decomposition fall back to a heuristic / raw-context mode.
- `experience.results_dir` — where the SPIKE benchmark writes `result.json`.
- `tools.backend` — `native` (default) or `openhands`.

## Using the OpenHands SDK backend

Set `tools.backend: openhands` in `config.yaml` and install the SDK into the
venv:

```bash
./.venv/bin/pip install -U openhands-sdk openhands-tools
export LLM_MODEL=...   LLM_API_KEY=...   LLM_BASE_URL=...   # for the SDK's own LLM
```

File writes (generated YAML, skill stubs) are then executed through an OpenHands
`Conversation` using `FileEditorTool`/`TerminalTool` in a `LocalWorkspace`
rooted at the repo. If the SDK is not installed, `backend: native` performs the
identical operations with plain Python.

> Note: `pip install` and `git clone` are explicit, user-initiated steps. This
> tool never runs them automatically.

## Programmatic use

```python
from env_agent import EnvAgent

agent = EnvAgent("env_agent/config.yaml")
agent.refresh_pitfalls()
res = agent.decompose("Reach the mines and kill some slimes safely", n=5)
for t in res.tasks:
    print(t.name, t.evaluator, t.to_yaml_entry())
```

## Extending

- **Web search QA**: pass a `web_search=callable` into `KnowledgeBase` (wired in
  `orchestrator.py`). The callable takes `(query, k)` and returns
  `[{"title","url","snippet"}]`.
- **New evaluators**: add them to `EVALUATOR_TO_SUITE` in `core/schema.py` after
  implementing the branch in `env/tasks/<suite>.py`.
- **Skill generation**: `Toolkit.register_skill_stub()` writes
  `@register_skill("name")` stubs matching SPIKE's skill registry.
