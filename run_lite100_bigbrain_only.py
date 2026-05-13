import argparse
import json
import os
import subprocess
from pathlib import Path

from env.parallel_worker_guard import resolve_parallel_worker_limit
from run_lite100_parallel import (
    VALID_EXPERIMENT_BUDGET_MODES,
    build_command,
    cleanup_workspace_benchmark_processes,
    cleanup_workspace_fastllm_health_cache,
    cleanup_workspace_llm_endpoint_slots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run StarDojo Lite100 with Cortex BigBrain-only single-action planning."
    )
    parser.add_argument(
        "--python_exec",
        default=None,
        help="Python executable to use. Defaults to the current interpreter.",
    )
    parser.add_argument(
        "--runner",
        default="env/llm_env_multi_tasks_parallel.py",
        help="Path to the parallel benchmark runner, relative to workspace root.",
    )
    parser.add_argument(
        "--llm_config",
        default="agent/conf/openai_config.json",
        help="LLM config path, relative to workspace root.",
    )
    parser.add_argument(
        "--embed_config",
        default="agent/conf/openai_config.json",
        help="Embedding config path, relative to workspace root.",
    )
    parser.add_argument(
        "--env_config",
        default="agent/conf/env_config_stardew.json",
        help="Environment config path, relative to workspace root.",
    )
    parser.add_argument(
        "--parallel_numb",
        type=int,
        default=8,
        help="Parallel worker count. Default is 8.",
    )
    parser.add_argument(
        "--start_port",
        type=int,
        default=10783,
        help="Starting port for the parallel runner.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the generated command and runtime overrides without executing it.",
    )
    parser.add_argument(
        "--experiment_budget_mode",
        default="benchmark_steps",
        choices=VALID_EXPERIMENT_BUDGET_MODES,
        help=(
            "Benchmark exit budget mode. Use benchmark_steps for step-capped success runs, "
            "or benchmark_llm_calls for cost-equivalent runs."
        ),
    )
    return parser.parse_args()


def _format_command(command: list[str]) -> str:
    return " ".join(
        json.dumps(part, ensure_ascii=False) if " " in part else part
        for part in command
    )


def _apply_bigbrain_only_env(env: dict[str, str]) -> dict[str, str]:
    env = dict(env)
    env["FIXED_SEED"] = "true"
    env["PYTHONHASHSEED"] = "42"
    env["STARDOJO_BIG_BRAIN_ONLY"] = "true"
    env["STARDOJO_BIG_BRAIN_SINGLE_ACTION"] = "true"
    env["NUMBER_OF_EXECUTE_SKILLS"] = "1"
    return env


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent

    args.parallel_numb = max(1, int(args.parallel_numb))
    parallel_limit = resolve_parallel_worker_limit(
        args.parallel_numb,
        root_dir=root_dir,
        llm_config_path=args.llm_config,
    )
    args.parallel_numb = parallel_limit.effective_workers
    if parallel_limit.limited:
        model_label = parallel_limit.model_name or "unknown"
        print(
            "[ParallelGuard] Clamped parallel workers "
            f"from {parallel_limit.requested_workers} to {parallel_limit.effective_workers} "
            f"(max_concurrency={parallel_limit.throttle_max_concurrency}, model={model_label})."
        )
    if parallel_limit.queue_enforced and not parallel_limit.limited:
        model_label = parallel_limit.model_name or "unknown"
        print(
            "[ParallelGuard] Keeping requested parallel workers "
            f"at {parallel_limit.requested_workers} while the shared LLM queue caps active requests "
            f"at {parallel_limit.throttle_max_concurrency} (model={model_label})."
        )

    command, task_count = build_command(args, root_dir)

    print(f"Loaded {task_count} lite benchmark tasks.")
    if parallel_limit.queue_enforced:
        print(
            "Parallel workers: "
            f"requested={parallel_limit.requested_workers}, effective={args.parallel_numb}, "
            f"queue_max_concurrency={parallel_limit.throttle_max_concurrency}"
        )
    else:
        print(f"Parallel workers: {args.parallel_numb}")
    print(f"Experiment budget mode: {args.experiment_budget_mode}")
    print("Cortex mode: BigBrain-only, single-action per step, memory config unchanged.")
    print("Runtime overrides:")
    print("  STARDOJO_BIG_BRAIN_ONLY=true")
    print("  STARDOJO_BIG_BRAIN_SINGLE_ACTION=true")
    print("  NUMBER_OF_EXECUTE_SKILLS=1")
    print("Command:")
    print(_format_command(command))

    if args.dry_run:
        return 0

    cleaned = cleanup_workspace_benchmark_processes(root_dir)
    if cleaned:
        print(f"Cleaned {cleaned} existing workspace benchmark process(es).")
    cleaned_slots = cleanup_workspace_llm_endpoint_slots(root_dir)
    if cleaned_slots:
        print(f"Cleaned {cleaned_slots} stale LLM endpoint slot file(s).")
    cleaned_health_cache = cleanup_workspace_fastllm_health_cache(root_dir)
    if cleaned_health_cache:
        print(f"Cleaned {cleaned_health_cache} FastLLM health cache file(s).")

    env = _apply_bigbrain_only_env(os.environ)
    completed = subprocess.run(command, cwd=str(root_dir), env=env)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
