import argparse
import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml

from env.parallel_worker_guard import resolve_parallel_worker_limit


DIAGNOSTIC_TASKS = {
    "farming_lite": [
        "clear_10_weeds_with_scythe",
        "clear_30_debris_with_scythe_and_pickaxe_and_axe",
        "till_5_tile_with_hoe",
        "sow_5_dirt_with_cauliflower_seeds",
        "fertilize_1_dirt_with_speed_gro",
        "harvest_1_milk_with_milk_pail",
    ],
    "exploration_lite": [
        "go_to_bed",
        "go_to_coop",
        "go_to_bus_stop",
        "go_to_pierre's_general_store",
        "go_to_the_mines_2nd_floor",
        "forage_1_daffodil",
        "mine_1_coal_with_pickaxe",
    ],
    "social_lite": [
        "purchase_5_beer",
        "sell_1_parsnip_to_pierre",
        "break_5_geode",
        "move_1_coop",
        "talk_to_alex",
    ],
    "crafting_lite": [
        "craft_1_wood_fence",
        "craft_1_basic_retaining_soil",
        "craft_1_furnace",
        "produce_1_copper_bar_with_furnace",
        "craft_1_sprinkler",
    ],
    "combat_lite": [
        "kill_1_green_slime_with_rusty_sword",
        "kill_5_green_slime_with_rusty_sword",
        "kill_1_bug_with_rusty_sword",
        "kill_1_duggy_with_rusty_sword",
        "kill_5_grub_with_rusty_sword",
    ],
}
VALID_EXPERIMENT_BUDGET_MODES = (
    "benchmark_steps",
    "benchmark_llm_calls",
)


def load_named_task_params(task_suite_dir: Path) -> tuple[list[dict], list[str]]:
    task_params: list[dict] = []
    selected_labels: list[str] = []

    for suite_name, wanted_names in DIAGNOSTIC_TASKS.items():
        suite_path = task_suite_dir / f"{suite_name}.yaml"
        if not suite_path.exists():
            raise FileNotFoundError(f"Missing lite suite file: {suite_path}")

        with suite_path.open("r", encoding="utf-8") as fd:
            suite_data = yaml.safe_load(fd) or {}

        if not isinstance(suite_data, dict):
            raise ValueError(f"Invalid suite format: {suite_path}")

        ordered_task_names = list(suite_data.keys())
        name_to_index = {task_name: idx for idx, task_name in enumerate(ordered_task_names)}
        missing = [task_name for task_name in wanted_names if task_name not in name_to_index]
        if missing:
            raise ValueError(
                f"Missing task names in {suite_path}: {', '.join(missing)}"
            )

        for task_name in wanted_names:
            task_params.append({
                "type": suite_name,
                "id": int(name_to_index[task_name]),
            })
            selected_labels.append(f"{suite_name}:{task_name}")

    expected_count = sum(len(task_names) for task_names in DIAGNOSTIC_TASKS.values())
    if len(task_params) != expected_count:
        raise ValueError(f"Expected {expected_count} diagnostic tasks, got {len(task_params)}")

    return task_params, selected_labels


def build_command(args: argparse.Namespace, root_dir: Path) -> tuple[list[str], list[str]]:
    task_suite_dir = root_dir / "env" / "tasks" / "task_suite"
    task_params, selected_labels = load_named_task_params(task_suite_dir)

    runner = (root_dir / args.runner).resolve()
    llm_config = (root_dir / args.llm_config).resolve()
    embed_config = (root_dir / args.embed_config).resolve()
    env_config = (root_dir / args.env_config).resolve()

    if not runner.exists():
        raise FileNotFoundError(f"Runner not found: {runner}")
    if not llm_config.exists():
        raise FileNotFoundError(f"LLM config not found: {llm_config}")
    if not embed_config.exists():
        raise FileNotFoundError(f"Embed config not found: {embed_config}")
    if not env_config.exists():
        raise FileNotFoundError(f"Env config not found: {env_config}")

    task_params_dir = root_dir / "runs" / "task_params"
    task_params_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=task_params_dir,
        prefix="lite_diagnostic_task_params_",
        suffix=".json",
        delete=False,
    ) as fd:
        json.dump(task_params, fd, ensure_ascii=False, separators=(",", ":"))
        task_params_path = Path(fd.name)

    command = [
        args.python_exec or sys.executable,
        str(runner),
        "--llm_config",
        str(llm_config),
        "--embed_config",
        str(embed_config),
        "--env_config",
        str(env_config),
        "--parallel_numb",
        str(args.parallel_numb),
        "--start_port",
        str(args.start_port),
        "--task_params",
        str(task_params_path),
        "--experiment_budget_mode",
        str(args.experiment_budget_mode),
    ]

    return command, selected_labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the diagnostic lite subset with the same parallel pipeline as lite-100."
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
        "--show_tasks",
        action="store_true",
        help="Print the selected task names before running.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print the generated command without executing it.",
    )
    parser.add_argument(
        "--experiment_budget_mode",
        default="benchmark_steps",
        choices=VALID_EXPERIMENT_BUDGET_MODES,
        help="Benchmark exit budget mode. Use benchmark_steps for step-capped success runs, or benchmark_llm_calls for cost-equivalent runs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
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

    command, selected_labels = build_command(args, root_dir)

    suite_counts = Counter(label.split(":", 1)[0] for label in selected_labels)

    print(f"Loaded {len(selected_labels)} diagnostic lite tasks.")
    print("Suite counts:")
    for suite_name, count in suite_counts.items():
        print(f"  - {suite_name}: {count}")
    if parallel_limit.limited:
        print(
            "Parallel workers: "
            f"requested={parallel_limit.requested_workers}, effective={args.parallel_numb}"
        )
    else:
        print(f"Parallel workers: {args.parallel_numb}")
    print(f"Experiment budget mode: {args.experiment_budget_mode}")
    print("Using the same parallel runner and configs as run_lite100_parallel.py.")

    if args.show_tasks:
        print("Selected tasks:")
        for label in selected_labels:
            print(f"  - {label}")

    print("Command:")
    print(" ".join(json.dumps(part, ensure_ascii=False) if " " in part else part for part in command))

    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=str(root_dir))
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
