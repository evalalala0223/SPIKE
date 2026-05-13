"""Focused regression run: 33 tasks mixing historical wins and stubborn failures.

Covers:
- Historical wins: all lite tasks that have at least one recorded success in
  runs/results across farming, exploration, social, and crafting.
- Persistent failures: representative tasks that still fail frequently, including
  backwoods navigation, hay foraging, speed-gro fertilizing, potato sowing,
  animal interaction, coal mining, geode breaking, and simple combat smoke checks.

Usage:
    python run_regression_focused.py --parallel_numb=4 --show_tasks
    python run_regression_focused.py --dry_run
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import yaml

from env.parallel_worker_guard import resolve_parallel_worker_limit


REGRESSION_TASKS = {
    "farming_lite": [
        # Historical wins.
        "clear_10_weeds_with_scythe",
        "clear_5_stone_with_pickaxe",
        "clear_30_debris_with_scythe_and_pickaxe_and_axe",
        "till_5_tile_with_hoe",
        "fertilize_5_dirt_with_basic_retaining_soil",
        "sow_5_dirt_with_cauliflower_seeds",
        "water_5_crop_with_watering_can",
        "harvest_5_parsnip",
        "fill_1_pet_bowl_with_watering_can",
        # Representative stubborn failures.
        "harvest_1_egg",
        "harvest_1_milk_with_milk_pail",
        "pet_3_animal",
    ],
    "exploration_lite": [
        # Historical wins.
        "go_to_bed",
        "go_to_coop",
        "go_to_bus_stop",
        "forage_1_clam",
        "forage_1_daffodil",
        "mine_1_copper_ore_with_pickaxe",
        # Representative stubborn failures.
        "go_to_backwoods",
        "forage_10_hay_with_scythe",
        "mine_1_coal_with_pickaxe",
    ],
    "social_lite": [
        # Historical wins.
        "ship_1_parsnip_with_shipping_bin",
        "purchase_5_beer",
        "sell_5_parsnip_to_pierre",
        "sell_1_parsnip_to_pierre",
        # Representative stubborn failures.
        "break_5_geode",
    ],
    "crafting_lite": [
        # Historical wins.
        "craft_1_wood_fence",
        "craft_1_scarecrow",
        "craft_1_basic_retaining_soil",
        "craft_1_field_snack",
    ],
    "combat_lite": [
        # Simple combat smoke checks.
        "kill_1_green_slime_with_rusty_sword",
        "kill_1_bug_with_rusty_sword",
        "kill_1_grub_with_rusty_sword",
    ],
}


def load_named_task_params(task_suite_dir: Path):
    task_params = []
    selected_labels = []

    for suite_name, wanted_names in REGRESSION_TASKS.items():
        suite_path = task_suite_dir / f"{suite_name}.yaml"
        if not suite_path.exists():
            raise FileNotFoundError(f"Missing lite suite file: {suite_path}")

        with suite_path.open("r", encoding="utf-8") as fd:
            suite_data = yaml.safe_load(fd) or {}

        ordered_task_names = list(suite_data.keys())
        name_to_index = {t: i for i, t in enumerate(ordered_task_names)}
        missing = [t for t in wanted_names if t not in name_to_index]
        if missing:
            raise ValueError(f"Missing tasks in {suite_path}: {', '.join(missing)}")

        for task_name in wanted_names:
            task_params.append({
                "type": suite_name,
                "id": int(name_to_index[task_name]),
            })
            selected_labels.append(f"{suite_name}:{task_name}")

    return task_params, selected_labels


def build_command(args, root_dir):
    task_suite_dir = root_dir / "env" / "tasks" / "task_suite"
    task_params, selected_labels = load_named_task_params(task_suite_dir)

    runner = (root_dir / args.runner).resolve()
    llm_config = (root_dir / args.llm_config).resolve()
    embed_config = (root_dir / args.embed_config).resolve()
    env_config = (root_dir / args.env_config).resolve()

    for path, name in [(runner, "Runner"), (llm_config, "LLM config"), (embed_config, "Embed config"), (env_config, "Env config")]:
        if not path.exists():
            raise FileNotFoundError(f"{name} not found: {path}")

    task_params_dir = root_dir / "runs" / "task_params"
    task_params_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=task_params_dir,
        prefix="regression_focused_", suffix=".json", delete=False,
    ) as fd:
        json.dump(task_params, fd, ensure_ascii=False, separators=(",", ":"))
        task_params_path = Path(fd.name)

    command = [
        args.python_exec or sys.executable,
        str(runner),
        "--llm_config", str(llm_config),
        "--embed_config", str(embed_config),
        "--env_config", str(env_config),
        "--parallel_numb", str(args.parallel_numb),
        "--start_port", str(args.start_port),
        "--task_params", str(task_params_path),
        "--experiment_budget_mode", str(args.experiment_budget_mode),
    ]
    if args.output_video:
        command.append("--output_video")

    return command, selected_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Run focused regression on 33 tasks covering historical wins, common failure cases, and simple combat smoke checks.")
    parser.add_argument("--python_exec", default=None)
    parser.add_argument("--runner", default="env/llm_env_multi_tasks_parallel.py")
    parser.add_argument("--llm_config", default="agent/conf/openai_config.json")
    parser.add_argument("--embed_config", default="agent/conf/openai_config.json")
    parser.add_argument("--env_config", default="agent/conf/env_config_stardew.json")
    parser.add_argument("--parallel_numb", type=int, default=8)
    parser.add_argument("--start_port", type=int, default=10783)
    parser.add_argument("--show_tasks", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--output_video", action="store_true")
    parser.add_argument("--experiment_budget_mode", default="benchmark_steps",
                        choices=("benchmark_steps", "benchmark_llm_calls"))
    return parser.parse_args()


def main():
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

    print(f"Loaded {len(selected_labels)} focused regression tasks.")
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

    if args.show_tasks:
        print("\nSelected tasks:")
        for label in selected_labels:
            print(f"  - {label}")

    print("\nCommand:")
    print(" ".join(
        json.dumps(part, ensure_ascii=False) if " " in part else part
        for part in command
    ))

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["FIXED_SEED"] = "true"
    env["PYTHONHASHSEED"] = "42"
    completed = subprocess.run(command, cwd=str(root_dir), env=env)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
