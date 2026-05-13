from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def _load_result_validity_helpers():
    module_path = Path(__file__).resolve().parent / "env" / "result_validity_utils.py"
    spec = importlib.util.spec_from_file_location(
        "_summary_result_validity_utils",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load result validity helpers from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.annotate_run_summary_validity, module.annotate_task_result_validity


annotate_run_summary_validity, annotate_task_result_validity = _load_result_validity_helpers()


_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_AGENT_RUN_DIR_RE = re.compile(
    r"^(?P<port>\d+)_(?P<task_name>.+)_(?P<task_id>\d+)_(?P<timestamp>\d+(?:\.\d+)?)$"
)
_LLM_RESPONSE_RE = re.compile(
    r"\[LLM_DIAG\] <<< RESPONSE \| model=(?P<model>[^|]+?) \| mode=(?P<mode>[^|]+?)"
    r"(?: \| key=(?P<key>[^|]+?))? \| duration=(?P<duration>[0-9.]+)s"
    r" \| response=(?P<response_chars>\d+)chars \| tokens\(prompt=(?P<prompt>\d+)"
    r" comp=(?P<completion>\d+) total=(?P<total>\d+)\)"
)
_REPLAN_RE = re.compile(r"\[BigBrain\] Planning triggered by: (?P<reason>[^,\r\n]+)")
_MEMORY_HIT_RE = re.compile(r"\[Routing\] ⚡ Using memory quick path actions")
_MEMORY_NORMAL_RE = re.compile(r"\[Routing\] ▶ Proceeding with normal planning")
_MEMORY_DISABLED_RE = re.compile(
    r"\[Routing\] ▶ Dual-brain mode: Mem0 quick path disabled, proceeding with normal planning"
)
_MEMORY_GUARDED_RE = re.compile(r"\[Mem0\] Quick path guarded: (?P<reason>[^\r\n]+)")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_mean(values: list[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None


def _safe_median(values: list[float]) -> float | None:
    return round(statistics.median(values), 3) if values else None


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def _round_cost(value: float | None) -> float | None:
    return round(value, 6) if value is not None else None


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fd:
        data = json.load(fd)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return data


def _find_latest_run(results_root: Path) -> Path:
    run_dirs = [p for p in results_root.iterdir() if p.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No run directories found in {results_root}")
    return max(run_dirs, key=lambda p: p.stat().st_mtime)


def _resolve_run_dir(arg_run: str | None, root_dir: Path) -> Path:
    results_root = root_dir / "runs" / "results"
    if arg_run is None:
        return _find_latest_run(results_root)

    run_path = Path(arg_run)
    if run_path.is_absolute():
        return run_path

    direct = root_dir / arg_run
    if direct.exists():
        return direct

    nested = results_root / arg_run
    if nested.exists():
        return nested

    raise FileNotFoundError(f"Run directory not found: {arg_run}")


def _collect_task_results(run_dir: Path) -> list[dict[str, Any]]:
    index_path = run_dir / "index.json"
    results: list[dict[str, Any]] = []

    if index_path.exists():
        run_index = _load_json(index_path)
        tasks = run_index.get("tasks", [])
        if isinstance(tasks, list):
            for item in tasks:
                if not isinstance(item, dict):
                    continue
                result_file = item.get("result_file")
                if isinstance(result_file, str) and result_file:
                    candidate = run_dir / result_file
                    if candidate.exists():
                        results.append(_load_json(candidate))
                        continue

                task_index = item.get("task_index")
                if isinstance(task_index, int):
                    pattern = f"task_{task_index:03d}_*"
                    matches = sorted(run_dir.glob(pattern))
                    for match in matches:
                        result_path = match / "result.json"
                        if result_path.exists():
                            results.append(_load_json(result_path))
                            break

    if results:
        return results

    for task_dir in sorted(run_dir.glob("task_*")):
        result_path = task_dir / "result.json"
        if result_path.exists():
            results.append(_load_json(result_path))

    return results


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE_RE.sub("", text)


def _parse_iso_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _parse_agent_run_dir(path: Path) -> dict[str, Any] | None:
    match = _AGENT_RUN_DIR_RE.match(path.name)
    if not match:
        return None
    return {
        "path": path,
        "port": int(match.group("port")),
        "task_name": match.group("task_name"),
        "task_id": int(match.group("task_id")),
        "timestamp": float(match.group("timestamp")),
    }


def _task_port(task_result: dict[str, Any]) -> int | None:
    steps = task_result.get("steps", [])
    if not isinstance(steps, list):
        return None
    for step in steps:
        if not isinstance(step, dict):
            continue
        port = _safe_int(step.get("port"))
        if port is not None:
            return port
    return None


def _find_agent_run_dir(task_result: dict[str, Any], root_dir: Path) -> Path | None:
    runs_root = root_dir / "agent" / "runs"
    if not runs_root.exists():
        return None

    agent_run_dir_name = str(task_result.get("agent_run_dir_name") or "").strip()
    if agent_run_dir_name:
        exact_match = runs_root / agent_run_dir_name
        if exact_match.exists():
            return exact_match

    port = _task_port(task_result)
    task_name = str(task_result.get("task_name") or "").strip()
    runner_task_name = str(task_result.get("runner_task_name") or "").strip()
    task_id = _safe_int(task_result.get("task_id"))
    if port is None or task_id is None:
        return None

    candidates: list[dict[str, Any]] = []
    for preferred_name in (runner_task_name, task_name):
        if not preferred_name:
            continue
        pattern = f"{port}_{preferred_name}_{task_id}_*"
        for candidate in runs_root.glob(pattern):
            parsed = _parse_agent_run_dir(candidate)
            if parsed is not None:
                candidates.append(parsed)

    if not candidates:
        for candidate in runs_root.glob(f"{port}_*_{task_id}_*"):
            parsed = _parse_agent_run_dir(candidate)
            if parsed is not None:
                candidates.append(parsed)
        if not candidates:
            return None

    preferred_name = runner_task_name or task_name
    if preferred_name:
        exact_name_matches = [
            item for item in candidates if str(item.get("task_name") or "") == preferred_name
        ]
        if exact_name_matches:
            candidates = exact_name_matches

    start_ts = _parse_iso_timestamp(task_result.get("start_time"))
    if start_ts is None:
        return max(candidates, key=lambda item: item["timestamp"])["path"]

    return min(candidates, key=lambda item: abs(item["timestamp"] - start_ts))["path"]


def _summarize_step_metrics(task_result: dict[str, Any]) -> dict[str, Any]:
    steps = task_result.get("steps", [])
    if not isinstance(steps, list):
        return {
            "decision_count": 0,
            "avg_decision_latency_sec": None,
            "median_decision_latency_sec": None,
        }

    planning_latencies: list[float] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        perf = step.get("perf", {})
        if not isinstance(perf, dict):
            continue
        planning_sec = _safe_float(perf.get("planning_sec"))
        if planning_sec is not None:
            planning_latencies.append(planning_sec)

    return {
        "decision_count": len(planning_latencies),
        "avg_decision_latency_sec": _safe_mean(planning_latencies),
        "median_decision_latency_sec": _safe_median(planning_latencies),
        "decision_latency_samples": planning_latencies,
    }


def _estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    prompt_cost_per_1k: float | None,
    completion_cost_per_1k: float | None,
) -> float | None:
    if prompt_cost_per_1k is None and completion_cost_per_1k is None:
        return None

    prompt_cost = 0.0
    completion_cost = 0.0
    if prompt_cost_per_1k is not None:
        prompt_cost = (prompt_tokens / 1000.0) * prompt_cost_per_1k
    if completion_cost_per_1k is not None:
        completion_cost = (completion_tokens / 1000.0) * completion_cost_per_1k
    return prompt_cost + completion_cost


def _parse_agent_log_metrics(
    log_path: Path,
    prompt_cost_per_1k: float | None,
    completion_cost_per_1k: float | None,
) -> dict[str, Any]:
    text = _strip_ansi(log_path.read_text(encoding="utf-8", errors="ignore"))

    llm_prompt_tokens = 0
    llm_completion_tokens = 0
    llm_total_tokens = 0
    llm_latencies: list[float] = []
    llm_call_count = 0
    llm_models = Counter()

    for match in _LLM_RESPONSE_RE.finditer(text):
        prompt_tokens = int(match.group("prompt"))
        completion_tokens = int(match.group("completion"))
        total_tokens = int(match.group("total"))
        duration = float(match.group("duration"))
        model = match.group("model").strip()

        llm_prompt_tokens += prompt_tokens
        llm_completion_tokens += completion_tokens
        llm_total_tokens += total_tokens
        llm_latencies.append(duration)
        llm_call_count += 1
        llm_models[model] += 1

    replan_by_reason = Counter(
        match.group("reason").strip() for match in _REPLAN_RE.finditer(text)
    )
    memory_hit_count = len(_MEMORY_HIT_RE.findall(text))
    memory_normal_count = len(_MEMORY_NORMAL_RE.findall(text))
    memory_disabled_count = len(_MEMORY_DISABLED_RE.findall(text))
    memory_guarded_by_reason = Counter(
        match.group("reason").strip() for match in _MEMORY_GUARDED_RE.finditer(text)
    )

    memory_route_decisions = (
        memory_hit_count + memory_normal_count + memory_disabled_count
    )
    estimated_cost_usd = _estimate_cost_usd(
        llm_prompt_tokens,
        llm_completion_tokens,
        prompt_cost_per_1k,
        completion_cost_per_1k,
    )

    return {
        "log_path": str(log_path),
        "logged_llm_calls": llm_call_count,
        "logged_prompt_tokens": llm_prompt_tokens,
        "logged_completion_tokens": llm_completion_tokens,
        "logged_total_tokens": llm_total_tokens,
        "avg_logged_llm_latency_sec": _safe_mean(llm_latencies),
        "median_logged_llm_latency_sec": _safe_median(llm_latencies),
        "estimated_logged_cost_usd": _round_cost(estimated_cost_usd),
        "logged_llm_models": dict(sorted(llm_models.items())),
        "replan_count": sum(replan_by_reason.values()),
        "replan_by_reason": dict(sorted(replan_by_reason.items())),
        "memory_quick_path_hits": memory_hit_count,
        "memory_quick_path_total": memory_route_decisions,
        "memory_quick_path_hit_rate": _safe_ratio(memory_hit_count, memory_route_decisions),
        "memory_quick_path_disabled_count": memory_disabled_count,
        "memory_quick_path_guarded_count": sum(memory_guarded_by_reason.values()),
        "memory_quick_path_guarded_by_reason": dict(
            sorted(memory_guarded_by_reason.items())
        ),
    }


def _enrich_task_result(
    task_result: dict[str, Any],
    root_dir: Path,
    prompt_cost_per_1k: float | None,
    completion_cost_per_1k: float | None,
) -> dict[str, Any]:
    enriched = dict(task_result)
    step_metrics = _summarize_step_metrics(task_result)
    enriched.update(
        {
            "decision_count": step_metrics["decision_count"],
            "avg_decision_latency_sec": step_metrics["avg_decision_latency_sec"],
            "median_decision_latency_sec": step_metrics["median_decision_latency_sec"],
        }
    )

    agent_run_dir = _find_agent_run_dir(task_result, root_dir)
    enriched["agent_run_dir"] = str(agent_run_dir) if agent_run_dir else None

    if not agent_run_dir:
        fallback_llm_calls = _safe_int(task_result.get("llm_call_count"))
        fallback_model = str(task_result.get("planner_comp_model") or "").strip()
        enriched.update(
            {
                "agent_run_match_failed": True,
                "logged_llm_calls": fallback_llm_calls,
                "logged_prompt_tokens": None,
                "logged_completion_tokens": None,
                "logged_total_tokens": None,
                "avg_logged_llm_latency_sec": None,
                "median_logged_llm_latency_sec": None,
                "estimated_logged_cost_usd": None,
                "logged_llm_models": (
                    {fallback_model: fallback_llm_calls}
                    if fallback_model and fallback_llm_calls is not None
                    else ({fallback_model: 1} if fallback_model else {})
                ),
                "replan_count": None,
                "replan_by_reason": {},
                "memory_quick_path_hits": None,
                "memory_quick_path_total": None,
                "memory_quick_path_hit_rate": None,
                "memory_quick_path_disabled_count": None,
                "memory_quick_path_guarded_count": None,
                "memory_quick_path_guarded_by_reason": {},
            }
        )
        return enriched

    log_path = agent_run_dir / "logs" / "stardojo.log"
    if not log_path.exists():
        fallback_llm_calls = _safe_int(task_result.get("llm_call_count"))
        fallback_model = str(task_result.get("planner_comp_model") or "").strip()
        enriched.update(
            {
                "agent_run_match_failed": True,
                "logged_llm_calls": fallback_llm_calls,
                "logged_prompt_tokens": None,
                "logged_completion_tokens": None,
                "logged_total_tokens": None,
                "avg_logged_llm_latency_sec": None,
                "median_logged_llm_latency_sec": None,
                "estimated_logged_cost_usd": None,
                "logged_llm_models": (
                    {fallback_model: fallback_llm_calls}
                    if fallback_model and fallback_llm_calls is not None
                    else ({fallback_model: 1} if fallback_model else {})
                ),
                "replan_count": None,
                "replan_by_reason": {},
                "memory_quick_path_hits": None,
                "memory_quick_path_total": None,
                "memory_quick_path_hit_rate": None,
                "memory_quick_path_disabled_count": None,
                "memory_quick_path_guarded_count": None,
                "memory_quick_path_guarded_by_reason": {},
            }
        )
        return enriched

    enriched["agent_run_match_failed"] = False
    enriched.update(
        _parse_agent_log_metrics(
            log_path,
            prompt_cost_per_1k=prompt_cost_per_1k,
            completion_cost_per_1k=completion_cost_per_1k,
        )
    )
    return enriched


def _summarize_tasks(
    run_dir: Path,
    task_results: list[dict[str, Any]],
    root_dir: Path,
    prompt_cost_per_1k: float | None,
    completion_cost_per_1k: float | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_index_path = run_dir / "index.json"
    run_index = _load_json(run_index_path) if run_index_path.exists() else {}
    enriched_results = [
        _enrich_task_result(
            task_result,
            root_dir=root_dir,
            prompt_cost_per_1k=prompt_cost_per_1k,
            completion_cost_per_1k=completion_cost_per_1k,
        )
        for task_result in task_results
    ]
    for result in enriched_results:
        annotate_task_result_validity(result)

    derived_run_validity = annotate_run_summary_validity(
        {
            "expected_tasks": _safe_int(run_index.get("expected_tasks")),
            "tasks": [dict(result) for result in enriched_results],
        }
    )
    index_run_status = str(run_index.get("run_status") or "").strip()
    index_benchmark_status = str(run_index.get("benchmark_status") or "").strip()
    index_invalid_reason = str(run_index.get("invalid_reason") or "").strip()
    derived_run_status = str(derived_run_validity.get("run_status") or "").strip()
    derived_benchmark_status = str(
        derived_run_validity.get("benchmark_status") or ""
    ).strip()
    derived_invalid_reason = str(
        derived_run_validity.get("invalid_reason") or ""
    ).strip()

    completed = [r for r in enriched_results if bool(r.get("completed"))]
    errored = [r for r in enriched_results if r.get("error")]
    durations = [_safe_float(r.get("duration_sec")) for r in enriched_results]
    durations = [d for d in durations if d is not None]
    exit_steps = [_safe_int(r.get("exit_step")) for r in enriched_results]
    exit_steps = [s for s in exit_steps if s is not None]
    decision_latencies = [
        _safe_float(r.get("avg_decision_latency_sec")) for r in enriched_results
    ]
    decision_latencies = [d for d in decision_latencies if d is not None]

    decision_latency_samples: list[float] = []
    for result in task_results:
        step_metrics = _summarize_step_metrics(result)
        decision_latency_samples.extend(step_metrics.get("decision_latency_samples", []))

    by_difficulty: dict[str, dict[str, Any]] = {}
    difficulty_groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in enriched_results:
        difficulty_groups[str(result.get("difficulty") or "unknown")].append(result)

    for difficulty, items in sorted(difficulty_groups.items()):
        done = sum(1 for item in items if item.get("completed"))
        difficulty_durations = [
            duration
            for duration in (_safe_float(i.get("duration_sec")) for i in items)
            if duration is not None
        ]
        difficulty_decision_latencies = [
            latency
            for latency in (_safe_float(i.get("avg_decision_latency_sec")) for i in items)
            if latency is not None
        ]
        by_difficulty[difficulty] = {
            "total": len(items),
            "completed": done,
            "success_rate": round(done / len(items), 4) if items else 0.0,
            "avg_duration_sec": round(statistics.mean(difficulty_durations), 3)
            if difficulty_durations
            else None,
            "avg_decision_latency_sec": round(
                statistics.mean(difficulty_decision_latencies), 3
            )
            if difficulty_decision_latencies
            else None,
        }

    end_reason_counter = Counter(
        str(r.get("end_reason") or "unknown") for r in enriched_results
    )
    task_type_counter = Counter(
        str(r.get("task_name") or "unknown") for r in enriched_results
    )
    replan_reason_counter = Counter()
    memory_guard_reason_counter = Counter()
    logged_llm_model_counter = Counter()
    planner_model_counter = Counter()
    embedding_model_counter = Counter()
    prompt_profile_counter = Counter()
    step_budget_counter = Counter()
    experiment_budget_mode_counter = Counter()

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_logged_tokens = 0
    logged_cost_values: list[float] = []
    replan_total = 0
    replan_observed = False
    memory_hits = 0
    memory_total = 0
    memory_disabled_total = 0
    memory_guarded_total = 0
    logged_llm_calls = 0
    logged_llm_latencies: list[float] = []
    agent_run_match_failures = 0

    for result in enriched_results:
        prompt_tokens = _safe_int(result.get("logged_prompt_tokens")) or 0
        completion_tokens = _safe_int(result.get("logged_completion_tokens")) or 0
        total_tokens = _safe_int(result.get("logged_total_tokens")) or 0
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_logged_tokens += total_tokens

        cost = _safe_float(result.get("estimated_logged_cost_usd"))
        if cost is not None:
            logged_cost_values.append(cost)

        replan_count = _safe_int(result.get("replan_count"))
        if replan_count is not None:
            replan_total += replan_count
            replan_observed = True
        replan_reason_counter.update(result.get("replan_by_reason", {}))
        memory_guard_reason_counter.update(
            result.get("memory_quick_path_guarded_by_reason", {})
        )

        memory_hits += _safe_int(result.get("memory_quick_path_hits")) or 0
        memory_total += _safe_int(result.get("memory_quick_path_total")) or 0
        memory_disabled_total += (
            _safe_int(result.get("memory_quick_path_disabled_count")) or 0
        )
        memory_guarded_total += (
            _safe_int(result.get("memory_quick_path_guarded_count")) or 0
        )

        logged_llm_calls += _safe_int(result.get("logged_llm_calls")) or 0
        latency = _safe_float(result.get("avg_logged_llm_latency_sec"))
        if latency is not None:
            logged_llm_latencies.append(latency)
        logged_llm_model_counter.update(result.get("logged_llm_models", {}))
        if bool(result.get("agent_run_match_failed", False)):
            agent_run_match_failures += 1

        planner_comp_model = str(result.get("planner_comp_model") or "").strip()
        if planner_comp_model:
            planner_model_counter.update([planner_comp_model])
        embedding_model = str(result.get("embedding_model") or "").strip()
        if embedding_model:
            embedding_model_counter.update([embedding_model])
        prompt_profile = str(result.get("prompt_profile") or "").strip()
        if prompt_profile:
            prompt_profile_counter.update([prompt_profile])
        step_budget = _safe_int(result.get("step_budget"))
        if step_budget is not None:
            step_budget_counter.update([str(step_budget)])
        experiment_budget_mode = str(result.get("experiment_budget_mode") or "").strip()
        if experiment_budget_mode:
            experiment_budget_mode_counter.update([experiment_budget_mode])

    slowest = sorted(
        enriched_results,
        key=lambda r: _safe_float(r.get("duration_sec")) or -1.0,
        reverse=True,
    )[:10]
    task_dir_count = len(
        [
            item
            for item in run_dir.iterdir()
            if item.is_dir() and item.name.startswith("task_")
        ]
    )
    result_json_count = len(list(run_dir.glob("task_*/result.json")))

    summary = {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "index_run_status": index_run_status,
        "index_benchmark_status": index_benchmark_status,
        "index_invalid_reason": index_invalid_reason,
        "derived_run_status": derived_run_status,
        "derived_benchmark_status": derived_benchmark_status,
        "derived_invalid_reason": derived_invalid_reason,
        "index_run_status_stale": bool(
            index_run_status and derived_run_status and index_run_status != derived_run_status
        ),
        "index_benchmark_status_stale": bool(
            index_benchmark_status
            and derived_benchmark_status
            and index_benchmark_status != derived_benchmark_status
        ),
        "index_expected_tasks": _safe_int(run_index.get("expected_tasks")),
        "index_actual_tasks": _safe_int(run_index.get("actual_tasks")),
        "task_dir_count": task_dir_count,
        "result_json_count": result_json_count,
        "index_parallel_numb": _safe_int(run_index.get("parallel_numb")),
        "index_experiment_budget_mode": str(run_index.get("experiment_budget_mode") or ""),
        "total_tasks": len(enriched_results),
        "completed_tasks": len(completed),
        "failed_tasks": len(enriched_results) - len(completed),
        "success_rate": round(len(completed) / len(enriched_results), 4)
        if enriched_results
        else 0.0,
        "tasks_with_error": len(errored),
        "avg_duration_sec": _safe_mean(durations),
        "median_duration_sec": _safe_median(durations),
        "avg_exit_step": _safe_mean([float(s) for s in exit_steps]),
        "median_exit_step": _safe_median([float(s) for s in exit_steps]),
        "avg_decision_latency_sec": _safe_mean(decision_latency_samples),
        "median_decision_latency_sec": _safe_median(decision_latency_samples),
        "logged_prompt_tokens": total_prompt_tokens,
        "logged_completion_tokens": total_completion_tokens,
        "logged_total_tokens": total_logged_tokens,
        "logged_llm_calls": logged_llm_calls,
        "avg_logged_llm_latency_sec": _safe_mean(logged_llm_latencies),
        "estimated_logged_cost_usd": _round_cost(sum(logged_cost_values))
        if logged_cost_values
        else None,
        "logged_llm_models": dict(sorted(logged_llm_model_counter.items())),
        "planner_comp_models": dict(sorted(planner_model_counter.items())),
        "embedding_models": dict(sorted(embedding_model_counter.items())),
        "prompt_profiles": dict(sorted(prompt_profile_counter.items())),
        "step_budgets": dict(sorted(step_budget_counter.items())),
        "experiment_budget_modes": dict(sorted(experiment_budget_mode_counter.items())),
        "agent_run_match_failures": agent_run_match_failures,
        "replan_count": replan_total if replan_observed else None,
        "replan_by_reason": dict(sorted(replan_reason_counter.items())),
        "memory_quick_path_hits": memory_hits,
        "memory_quick_path_total": memory_total,
        "memory_quick_path_hit_rate": _safe_ratio(memory_hits, memory_total),
        "memory_quick_path_disabled_count": memory_disabled_total,
        "memory_quick_path_guarded_count": memory_guarded_total,
        "memory_quick_path_guarded_by_reason": dict(
            sorted(memory_guard_reason_counter.items())
        ),
        "end_reasons": dict(sorted(end_reason_counter.items())),
        "task_types": dict(sorted(task_type_counter.items())),
        "by_difficulty": by_difficulty,
        "slowest_tasks": [
            {
                "task_index": item.get("task_index"),
                "task_name": item.get("task_name"),
                "task_id": item.get("task_id"),
                "task_description": item.get("task_description"),
                "duration_sec": item.get("duration_sec"),
                "completed": item.get("completed"),
                "end_reason": item.get("end_reason"),
                "avg_decision_latency_sec": item.get("avg_decision_latency_sec"),
                "logged_total_tokens": item.get("logged_total_tokens"),
                "replan_count": item.get("replan_count"),
                "memory_quick_path_hit_rate": item.get("memory_quick_path_hit_rate"),
            }
            for item in slowest
        ],
        "errored_tasks": [
            {
                "task_index": item.get("task_index"),
                "task_name": item.get("task_name"),
                "task_id": item.get("task_id"),
                "task_description": item.get("task_description"),
                "error": item.get("error"),
            }
            for item in errored
        ],
    }
    return summary, enriched_results


def _build_markdown_report(summary: dict[str, Any], task_results: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    lines.append(f"# Run Summary: {summary['run_id']}")
    lines.append("")
    lines.append(f"- Run directory: {summary['run_dir']}")
    if summary.get("derived_run_status"):
        lines.append(
            f"- Run lifecycle status (derived from task results): {summary['derived_run_status']}"
        )
    if summary.get("derived_benchmark_status"):
        lines.append(
            f"- Benchmark validity (derived from task results): {summary['derived_benchmark_status']}"
        )
    if summary.get("derived_invalid_reason"):
        lines.append(
            f"- Benchmark invalid reason (derived from task results): {summary['derived_invalid_reason']}"
        )
    if summary.get("index_run_status"):
        label = "stale stored run status (index.json)" if summary.get("index_run_status_stale") else "Stored run status (index.json)"
        lines.append(f"- {label}: {summary['index_run_status']}")
        if (
            str(summary.get("index_run_status") or "").strip().lower() == "running"
            and str(summary.get("derived_run_status") or "").strip().lower() != "running"
        ):
            lines.append(
                "- `index.json` still says `running`, but this regenerated summary uses on-disk task results as the source of truth."
            )
    if summary.get("index_benchmark_status"):
        label = (
            "Stale stored benchmark validity (index.json)"
            if summary.get("index_benchmark_status_stale")
            else "Stored benchmark validity (index.json)"
        )
        lines.append(f"- {label}: {summary['index_benchmark_status']}")
    if summary.get("index_invalid_reason"):
        lines.append(f"- Stored benchmark invalid reason (index.json): {summary['index_invalid_reason']}")
    if summary.get("index_expected_tasks") is not None:
        lines.append(f"- Expected tasks (index.json): {summary['index_expected_tasks']}")
    if summary.get("index_actual_tasks") is not None:
        lines.append(f"- Indexed tasks (index.json): {summary['index_actual_tasks']}")
    if summary.get("task_dir_count") is not None:
        lines.append(f"- Task directories on disk: {summary['task_dir_count']}")
    if summary.get("result_json_count") is not None:
        lines.append(f"- Result.json files on disk: {summary['result_json_count']}")
    if summary.get("index_parallel_numb") is not None:
        lines.append(f"- Parallel workers (index.json): {summary['index_parallel_numb']}")
    if summary.get("index_experiment_budget_mode"):
        lines.append(f"- Experiment budget mode (index.json): {summary['index_experiment_budget_mode']}")
    lines.append(f"- Total tasks: {summary['total_tasks']}")
    lines.append(f"- Completed: {summary['completed_tasks']}")
    lines.append(f"- Failed/unfinished: {summary['failed_tasks']}")
    lines.append(f"- Success rate: {summary['success_rate']:.2%}")
    lines.append(f"- Tasks with explicit error: {summary['tasks_with_error']}")
    lines.append(f"- Avg duration: {summary['avg_duration_sec']}")
    lines.append(f"- Median duration: {summary['median_duration_sec']}")
    lines.append(f"- Avg exit step: {summary['avg_exit_step']}")
    lines.append(f"- Median exit step: {summary['median_exit_step']}")
    lines.append("")

    lines.append("## Result.json Snapshot")
    lines.append(
        "| Index | Task | Completed | Final Quantity | Exit Step | End Reason | Run Status |"
    )
    lines.append("| --- | --- | --- | ---: | ---: | --- | --- |")
    for item in sorted(task_results, key=lambda r: (_safe_int(r.get("task_index")) or 10**9)):
        lines.append(
            f"| {item.get('task_index')} | {item.get('task_name')} | {item.get('completed')} | "
            f"{item.get('final_quantity')} | {item.get('exit_step')} | {item.get('end_reason')} | "
            f"{item.get('run_status')} |"
        )
    lines.append("")

    lines.append("## Extra Metrics")
    lines.append(
        f"- Avg decision latency: {summary['avg_decision_latency_sec']} sec "
        f"(derived from per-step `perf.planning_sec`)"
    )
    lines.append(
        f"- Median decision latency: {summary['median_decision_latency_sec']} sec"
    )
    lines.append(
        f"- Logged token usage: prompt={summary['logged_prompt_tokens']}, "
        f"completion={summary['logged_completion_tokens']}, total={summary['logged_total_tokens']}"
    )
    lines.append(
        f"- Estimated logged cost (USD): {summary['estimated_logged_cost_usd']}"
    )
    lines.append(
        f"- Logged LLM calls: {summary['logged_llm_calls']}, "
        f"avg logged LLM latency: {summary['avg_logged_llm_latency_sec']} sec"
    )
    if summary["logged_llm_models"]:
        lines.append(f"- Logged LLM models: {summary['logged_llm_models']}")
    if summary.get("planner_comp_models"):
        lines.append(f"- Planner comp models: {summary['planner_comp_models']}")
    if summary.get("embedding_models"):
        lines.append(f"- Embedding models: {summary['embedding_models']}")
    if summary.get("prompt_profiles"):
        lines.append(f"- Prompt profiles: {summary['prompt_profiles']}")
    if summary.get("step_budgets"):
        lines.append(f"- Step budgets: {summary['step_budgets']}")
    if summary.get("experiment_budget_modes"):
        lines.append(f"- Experiment budget modes: {summary['experiment_budget_modes']}")
    if summary.get("agent_run_match_failures"):
        lines.append(
            f"- Agent run match failures: {summary['agent_run_match_failures']} "
            "(fallback to result.json metadata when available)"
        )
    lines.append(
        f"- Replan count: {summary['replan_count']} "
        f"({summary['replan_by_reason'] or 'no replan reasons logged'})"
    )
    lines.append(
        f"- Memory quick-path hit rate: {summary['memory_quick_path_hit_rate']} "
        f"({summary['memory_quick_path_hits']}/{summary['memory_quick_path_total']})"
    )
    lines.append(
        f"- Memory quick-path disabled: {summary['memory_quick_path_disabled_count']}, "
        f"guarded: {summary['memory_quick_path_guarded_count']}"
    )
    lines.append("")

    lines.append("## End Reasons")
    for reason, count in summary["end_reasons"].items():
        lines.append(f"- {reason}: {count}")
    lines.append("")

    lines.append("## Difficulty Breakdown")
    lines.append(
        "| Difficulty | Total | Completed | Success Rate | Avg Duration Sec | Avg Decision Latency Sec |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for difficulty, info in summary["by_difficulty"].items():
        lines.append(
            f"| {difficulty} | {info['total']} | {info['completed']} | "
            f"{info['success_rate']:.2%} | {info['avg_duration_sec']} | "
            f"{info['avg_decision_latency_sec']} |"
        )
    lines.append("")

    lines.append("## Slowest Tasks")
    lines.append(
        "| Index | Task | ID | Description | Duration Sec | Avg Decision Latency Sec | Tokens | Replans | Mem Hit Rate | Completed | End Reason |"
    )
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for item in summary["slowest_tasks"]:
        lines.append(
            f"| {item['task_index']} | {item['task_name']} | {item['task_id']} | "
            f"{item['task_description']} | {item['duration_sec']} | "
            f"{item['avg_decision_latency_sec']} | {item['logged_total_tokens']} | "
            f"{item['replan_count']} | {item['memory_quick_path_hit_rate']} | "
            f"{item['completed']} | {item['end_reason']} |"
        )
    lines.append("")

    errored = summary.get("errored_tasks", [])
    if errored:
        lines.append("## Errored Tasks")
        for item in errored:
            lines.append(
                f"- task_index={item['task_index']} {item['task_name']}#{item['task_id']} "
                f"{item['task_description']}: {item['error']}"
            )
        lines.append("")

    lines.append("## Per-task Results")
    lines.append(
        "| Index | Task | ID | Description | Difficulty | Completed | Exit Step | Duration Sec | Avg Decision Latency Sec | Tokens | Replans | Mem Hit Rate | End Reason |"
    )
    lines.append(
        "| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    for item in sorted(task_results, key=lambda r: (_safe_int(r.get("task_index")) or 10**9)):
        lines.append(
            f"| {item.get('task_index')} | {item.get('task_name')} | {item.get('task_id')} | "
            f"{item.get('task_description')} | {item.get('difficulty')} | "
            f"{item.get('completed')} | {item.get('exit_step')} | {item.get('duration_sec')} | "
            f"{item.get('avg_decision_latency_sec')} | {item.get('logged_total_tokens')} | "
            f"{item.get('replan_count')} | {item.get('memory_quick_path_hit_rate')} | "
            f"{item.get('end_reason')} |"
        )

    lines.append("")
    lines.append("> Notes")
    lines.append(
        "> `Avg decision latency` comes from `result.json -> steps[*].perf.planning_sec`."
    )
    lines.append(
        "> `Tokens` and `estimated logged cost` come from `[LLM_DIAG] <<< RESPONSE` lines in matched `agent/runs/*/logs/stardojo.log`; if a provider does not log token usage there, these values undercount total model usage."
    )
    lines.append(
        "> `Memory hit rate` is based on quick-path routing logs (`Using memory quick path actions`) over total logged memory-route decisions."
    )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a StarDojo benchmark result directory."
    )
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Run directory path or run_id under runs/results. Defaults to the latest run.",
    )
    parser.add_argument(
        "--write_files",
        action="store_true",
        help="Deprecated: summary files are now written into the run directory by default.",
    )
    parser.add_argument(
        "--print_only",
        action="store_true",
        help="Only print the summary and do not write files into the run directory.",
    )
    parser.add_argument(
        "--prompt_cost_per_1k",
        type=float,
        default=None,
        help="Optional USD cost per 1K prompt tokens for estimated logged cost.",
    )
    parser.add_argument(
        "--completion_cost_per_1k",
        type=float,
        default=None,
        help="Optional USD cost per 1K completion tokens for estimated logged cost.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root_dir = Path(__file__).resolve().parent
    run_dir = _resolve_run_dir(args.run_dir, root_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    task_results = _collect_task_results(run_dir)
    if not task_results:
        raise FileNotFoundError(f"No task result files found under {run_dir}")

    summary, enriched_results = _summarize_tasks(
        run_dir,
        task_results,
        root_dir=root_dir,
        prompt_cost_per_1k=args.prompt_cost_per_1k,
        completion_cost_per_1k=args.completion_cost_per_1k,
    )
    markdown = _build_markdown_report(summary, enriched_results)

    print(markdown)

    if not args.print_only:
        summary_json_path = run_dir / "summary_stats.json"
        summary_md_path = run_dir / "summary_report.md"
        with summary_json_path.open("w", encoding="utf-8") as fd:
            json.dump(summary, fd, ensure_ascii=False, indent=2)
        summary_md_path.write_text(markdown, encoding="utf-8")
        print(f"Saved summary JSON: {summary_json_path}")
        print(f"Saved summary report: {summary_md_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
