import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import psutil
import yaml

from env.parallel_worker_guard import resolve_parallel_worker_limit


LITE_SUITES = [
	"farming_lite",
	"exploration_lite",
	"social_lite",
	"crafting_lite",
	"combat_lite",
]
VALID_EXPERIMENT_BUDGET_MODES = (
	"benchmark_steps",
	"benchmark_llm_calls",
)


def _normalize_cmd_text(value: str) -> str:
	return str(value or "").replace("/", "\\").lower()


def _iter_workspace_benchmark_processes(root_dir: Path):
	current_pid = os.getpid()
	root_text = _normalize_cmd_text(str(root_dir.resolve()))
	patterns = (
		"run_lite100_parallel.py",
		"llm_env_multi_tasks_parallel.py",
	)

	for proc in psutil.process_iter(["pid", "name", "cmdline"]):
		try:
			pid = int(proc.info.get("pid") or -1)
			if pid <= 0 or pid == current_pid:
				continue
			cmdline = proc.info.get("cmdline") or []
			cmd_text = _normalize_cmd_text(" ".join(str(part) for part in cmdline))
			cwd_text = ""
			try:
				cwd_text = _normalize_cmd_text(proc.cwd() or "")
			except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, OSError):
				cwd_text = ""
			within_workspace = bool(root_text and (root_text in cmd_text or root_text in cwd_text))
			if within_workspace and any(pattern in cmd_text for pattern in patterns):
				yield proc
		except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
			continue


def _terminate_process_tree(proc: psutil.Process) -> None:
	try:
		children = proc.children(recursive=True)
	except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
		children = []

	for child in reversed(children):
		try:
			child.kill()
		except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
			pass

	try:
		proc.kill()
	except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
		pass


def cleanup_workspace_benchmark_processes(root_dir: Path) -> int:
	procs = list(_iter_workspace_benchmark_processes(root_dir))
	for proc in procs:
		_terminate_process_tree(proc)
	return len(procs)


def _resolve_agent_relative_path(root_dir: Path, raw_path: str) -> Path:
	candidate = Path(str(raw_path or "").strip())
	if candidate.is_absolute():
		return candidate
	return (root_dir / "agent" / candidate).resolve()


def cleanup_workspace_llm_endpoint_slots(
	root_dir: Path,
	*,
	enhanced_config_path: str = "agent/conf/enhanced_config.yaml",
) -> int:
	slot_dir = (root_dir / "agent" / "cache" / "locks" / "llm_endpoint_slots").resolve()
	env_config_path = os.environ.get("STARDOJO_ENHANCED_CONFIG", "").strip()
	if env_config_path:
		config_path = Path(env_config_path)
		if not config_path.is_absolute():
			config_path = (root_dir / env_config_path).resolve()
	else:
		config_path = (root_dir / enhanced_config_path).resolve()

	try:
		with config_path.open("r", encoding="utf-8") as fd:
			cfg = yaml.safe_load(fd) or {}
		throttle_cfg = (((cfg.get("performance") or {}).get("llm_endpoint_throttle") or {}))
		raw_slot_dir = str(throttle_cfg.get("slot_dir", "") or "").strip()
		if raw_slot_dir:
			slot_dir = _resolve_agent_relative_path(root_dir, raw_slot_dir)
	except Exception:
		pass

	if not slot_dir.exists():
		return 0

	cleaned = 0
	for lock_path in slot_dir.glob("slot_*.lock"):
		for attempt in range(4):
			try:
				lock_path.unlink()
				cleaned += 1
				break
			except FileNotFoundError:
				break
			except PermissionError:
				if attempt >= 3:
					break
				time.sleep(0.25)
	return cleaned


def cleanup_workspace_fastllm_health_cache(root_dir: Path) -> int:
	cache_dir = (root_dir / "agent" / "cache" / "locks" / "fastllm_health").resolve()
	if not cache_dir.exists():
		return 0

	cleaned = 0
	for pattern in ("*.json", "*.lock"):
		for cache_path in cache_dir.glob(pattern):
			try:
				cache_path.unlink()
				cleaned += 1
			except FileNotFoundError:
				continue
			except PermissionError:
				continue
	return cleaned


def load_lite_task_params(task_suite_dir: Path) -> list[dict]:
	task_params: list[dict] = []

	for suite_name in LITE_SUITES:
		suite_path = task_suite_dir / f"{suite_name}.yaml"
		if not suite_path.exists():
			raise FileNotFoundError(f"Missing lite suite file: {suite_path}")

		with suite_path.open("r", encoding="utf-8") as fd:
			suite_data = yaml.safe_load(fd) or {}

		if not isinstance(suite_data, dict):
			raise ValueError(f"Invalid suite format: {suite_path}")

		for idx, (_task_name, task_info) in enumerate(suite_data.items()):
			if not isinstance(task_info, dict):
				raise ValueError(f"Invalid task entry in {suite_path}: {_task_name}")

			task_params.append({
				"type": suite_name,
				"id": int(idx),
			})

	if len(task_params) != 100:
		raise ValueError(f"Expected 100 lite tasks, got {len(task_params)}")

	return task_params


def build_command(args: argparse.Namespace, root_dir: Path) -> tuple[list[str], int]:
	task_suite_dir = root_dir / "env" / "tasks" / "task_suite"
	task_params = load_lite_task_params(task_suite_dir)

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
		prefix="lite100_task_params_",
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

	return command, len(task_params)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Run the full 100-task StarDojo Lite benchmark with parallel workers."
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
	print(f"Using dual-brain pipeline from current StarDojo agent integration.")
	print("Command:")
	print(" ".join(json.dumps(part, ensure_ascii=False) if " " in part else part for part in command))

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

	env = os.environ.copy()
	env["FIXED_SEED"] = "true"
	env["PYTHONHASHSEED"] = "42"
	completed = subprocess.run(command, cwd=str(root_dir), env=env)
	return int(completed.returncode)


if __name__ == "__main__":
	raise SystemExit(main())
