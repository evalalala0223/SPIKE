#!/usr/bin/env python
"""CLI entry point for the SPIKE Environment Agent.

Run from the SPIKE repo root with the project venv, e.g.::

    ./.venv/bin/python -m env_agent.run_env_agent decompose \
        --goal "Teach the agent to grow and harvest parsnips" --n 6

Subcommands:
    decompose   Full pipeline: analyze history -> retrieve knowledge -> generate
                a validated sub-task curriculum -> write YAML.
    analyze     Only refresh the pitfalls markdown from run history.
    ask         Knowledge-base QA over the offline wiki.
    sync-wiki   Shallow-clone the offline Stardew Valley wiki into the cache.
    pool        Print task-pool statistics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow running as a script (python env_agent/run_env_agent.py ...)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from env_agent.core.orchestrator import EnvAgent  # noqa: E402


def _default_config() -> Path:
    return Path(__file__).resolve().parent / "config.yaml"


def cmd_decompose(agent: EnvAgent, args: argparse.Namespace) -> int:
    res = agent.decompose(
        args.goal, n=args.n, write=not args.dry_run, knowledge_query=args.knowledge_query
    )
    print(f"\n=== Decomposition for: {res.goal} ===")
    print(f"Generated {len(res.tasks)} valid task(s).")
    for i, t in enumerate(res.tasks, 1):
        print(f"  {i}. {t.name}  [{t.evaluator}/{t.difficulty}] "
              f"object={t.object} qty={t.quantity} tool={t.tool}")
    if res.invalid:
        print(f"\nDropped {len(res.invalid)} invalid candidate(s):")
        for name, problems in res.invalid.items():
            print(f"  - {name}: {problems}")
    if res.output_path:
        print(f"\nWritten to: {res.output_path}")
    print(f"Knowledge sources: {res.knowledge_sources or 'none'}")
    print(f"Event log: {res.events_path}")
    return 0


def cmd_analyze(agent: EnvAgent, args: argparse.Namespace) -> int:
    report = agent.refresh_pitfalls()
    print(f"Scanned {report['runs_scanned']} runs; "
          f"overall success rate {report['overall_success_rate']:.0%}.")
    print(f"Problem tasks: {len(report['problem_tasks'])}")
    for d in report["problem_tasks"][: args.top_n]:
        print(f"  - {d['task']} [{d['suite']}/{d['difficulty']}] "
              f"{d['completed']}/{d['attempts']} solved")
    print(f"\nPitfalls written to: {agent.pitfalls_file}")
    return 0


def cmd_ask(agent: EnvAgent, args: argparse.Namespace) -> int:
    qa = agent.ask(args.question, k=args.k)
    print("=== Answer ===")
    print(qa.get("answer") or "(no relevant knowledge found)")
    print("\nSources:", qa.get("sources") or "none")
    return 0


def cmd_sync_wiki(agent: EnvAgent, args: argparse.Namespace) -> int:
    print(agent.sync_wiki())
    return 0


def cmd_pool(agent: EnvAgent, args: argparse.Namespace) -> int:
    print(json.dumps(agent.pool.stats(), ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="SPIKE Environment Agent")
    parser.add_argument("--config", default=str(_default_config()),
                        help="path to env_agent config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("decompose", help="generate a sub-task curriculum")
    p.add_argument("--goal", required=True, help="high-level goal to decompose")
    p.add_argument("--n", type=int, default=None, help="number of sub-tasks")
    p.add_argument("--knowledge-query", default=None,
                   help="override the wiki QA query (defaults to the goal)")
    p.add_argument("--dry-run", action="store_true", help="do not write YAML")
    p.set_defaults(func=cmd_decompose)

    p = sub.add_parser("analyze", help="refresh pitfalls.md from run history")
    p.add_argument("--top-n", type=int, default=20)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("ask", help="QA over the offline wiki")
    p.add_argument("question")
    p.add_argument("--k", type=int, default=4)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("sync-wiki", help="clone the offline Stardew wiki")
    p.set_defaults(func=cmd_sync_wiki)

    p = sub.add_parser("pool", help="print task-pool stats")
    p.set_defaults(func=cmd_pool)

    args = parser.parse_args()
    agent = EnvAgent(args.config)
    return args.func(agent, args)


if __name__ == "__main__":
    raise SystemExit(main())
