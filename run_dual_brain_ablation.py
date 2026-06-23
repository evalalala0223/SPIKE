"""
SPIKE Dual-Brain ON / OFF Ablation Experiment
=============================================

只跑 3 个有代表性的 case（easy / medium / hard），分别用 dual-brain ON 和 OFF
两套 enhanced_config.yaml 跑一遍，自动比较：

  - 任务成功率 (completed)
  - 每任务步数与 LLM 调用数
  - 平均规划延迟 (planning_sec) / 总耗时
  - replan / memory quick-path 是否被触发（dual-brain 是否真的生效的硬证据）
  - 由 runner 写出的 budget_exit_reason / end_reason

代表性 case（仅这 3 个）：
  1. EASY    farming_lite     id=8     water_5_crop_with_watering_can
  2. MEDIUM  exploration_lite id=12    chop_20_wood_with_axe
  3. HARD    farming_lite     id=10    cultivate_and_harvest_1_garlic

用法（mac，单实例，串行）：
    cd /Users/sharonxrzhu/CodeBuddy/projects/腾讯/SPIKE
    PYTHONPATH="$PWD:$PWD/agent:$PWD/env" \
        .venv/bin/python run_dual_brain_ablation.py \
            --llm_config agent/conf/gemini_config.json \
            --embed_config agent/conf/gemini_config.json

可选参数：
    --conditions on,off       只跑某一组 (默认两组都跑)
    --repeats 1               每个 case 重复跑几次（默认 1，论文是 3）
    --start_port 10783
    --dry_run                 只打印命令，不真跑
    --skip_existing           检测到 run_dir 里已有 3 个 task 结果就跳过该条件
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import datetime as dt
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = ROOT / "runs" / "results"
PARAM_DIR = ROOT / "runs" / "task_params"

# --------------------------------------------------------------------------- #
# 3 个 representative cases（与之前 case study 完全对齐）
# --------------------------------------------------------------------------- #
CASES: list[dict[str, Any]] = [
    {"label": "easy",   "type": "farming_lite",     "id": 8,   "name": "water_5_crop_with_watering_can"},
    {"label": "medium", "type": "exploration_lite", "id": 12,  "name": "chop_20_wood_with_axe"},
    {"label": "hard",   "type": "farming_lite",     "id": 10,  "name": "cultivate_and_harvest_1_garlic"},
]

CONDITIONS = {
    "on":  ROOT / "agent" / "conf" / "enhanced_config_dual_brain_on.yaml",
    "off": ROOT / "agent" / "conf" / "enhanced_config_dual_brain_off.yaml",
}


# --------------------------------------------------------------------------- #
# 1. 启动前自检：把这两个开关读出来打印，杜绝"以为开了其实没开"
# --------------------------------------------------------------------------- #
def inspect_config(yaml_path: Path) -> dict[str, Any]:
    if not yaml_path.exists():
        return {"_error": f"missing {yaml_path}"}
    with yaml_path.open("r", encoding="utf-8") as fd:
        raw = yaml.safe_load(fd) or {}
    feats = raw.get("features") or {}
    db = raw.get("dual_brain") or {}
    mem0 = raw.get("mem0") or {}
    sa_kg = raw.get("sa_kg") or {}
    return {
        "path": str(yaml_path),
        "features.use_dual_brain": bool(feats.get("use_dual_brain")),
        "features.use_mem0":       bool(feats.get("use_mem0")),
        "dual_brain.enabled":      bool(db.get("enabled")),
        "mem0.enabled":            bool(mem0.get("enabled")),
        "sa_kg.enabled":           bool(sa_kg.get("enabled")),
        "dual_brain_effective":    bool(feats.get("use_dual_brain")) and bool(db.get("enabled")),
        "mem0_effective":          bool(feats.get("use_mem0")) and bool(mem0.get("enabled")),
    }


# --------------------------------------------------------------------------- #
# 2. 准备 task_params JSON（给 runner 喂 3 个 case）
# --------------------------------------------------------------------------- #
def write_task_params() -> Path:
    PARAM_DIR.mkdir(parents=True, exist_ok=True)
    payload = [{"type": c["type"], "id": int(c["id"])} for c in CASES]
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = PARAM_DIR / f"ablation_3case_{ts}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 3. 跑 runner
# --------------------------------------------------------------------------- #
def run_condition(
    condition: str,
    *,
    llm_config: str,
    embed_config: str,
    env_config: str,
    parallel_numb: int,
    start_port: int,
    task_params_path: Path,
    dry_run: bool,
    repeat_idx: int,
) -> dict[str, Any]:
    """跑一次 (condition × repeat_idx)，返回这次产生的 run_dir 摘要。"""
    yaml_path = CONDITIONS[condition].resolve()
    inspection = inspect_config(yaml_path)

    runner = (ROOT / "env" / "llm_env_multi_tasks_parallel.py").resolve()
    if not runner.exists():
        raise FileNotFoundError(runner)

    # runner 内部会用 datetime.now() 生成 run_id，我们记录起始 / 结束时间，
    # 跑完后用时间窗找出新生成的 run_dir。
    started_at = time.time()
    started_iso = dt.datetime.fromtimestamp(started_at).strftime("%Y-%m-%d %H:%M:%S")

    cmd = [
        sys.executable,
        str(runner),
        "--llm_config",  str((ROOT / llm_config).resolve()),
        "--embed_config", str((ROOT / embed_config).resolve()),
        "--env_config",   str((ROOT / env_config).resolve()),
        "--parallel_numb", str(parallel_numb),
        "--start_port",    str(start_port),
        "--task_params",   str(task_params_path.resolve()),
        "--experiment_budget_mode", "benchmark_steps",
    ]

    env = os.environ.copy()
    # 关键：通过这个环境变量切配置文件，不需要改原 enhanced_config.yaml
    env["STARDOJO_ENHANCED_CONFIG"] = str(yaml_path)
    env["FIXED_SEED"] = "true"
    env["PYTHONHASHSEED"] = "42"
    pp = env.get("PYTHONPATH", "")
    extra = [str(ROOT), str(ROOT / "agent"), str(ROOT / "env")]
    env["PYTHONPATH"] = os.pathsep.join([*extra, pp]) if pp else os.pathsep.join(extra)

    print()
    print("=" * 80)
    print(f"  CONDITION = {condition.upper()}   (repeat #{repeat_idx})")
    print(f"  Started   = {started_iso}")
    print(f"  YAML      = {yaml_path}")
    for k, v in inspection.items():
        print(f"    {k:30s} = {v}")
    print(f"  STARDOJO_ENHANCED_CONFIG = {env['STARDOJO_ENHANCED_CONFIG']}")
    print(f"  Command:")
    print("    " + " ".join(shlex.quote(p) for p in cmd))
    print("=" * 80)

    if dry_run:
        return {
            "condition": condition,
            "repeat": repeat_idx,
            "yaml": str(yaml_path),
            "inspection": inspection,
            "run_dir": None,
            "returncode": 0,
            "dry_run": True,
        }

    completed = subprocess.run(cmd, cwd=str(ROOT), env=env)
    ended_at = time.time()

    run_dir = _find_new_run_dir(started_at, ended_at)
    return {
        "condition": condition,
        "repeat": repeat_idx,
        "yaml": str(yaml_path),
        "inspection": inspection,
        "run_dir": str(run_dir) if run_dir else None,
        "returncode": int(completed.returncode),
        "dry_run": False,
    }


def _find_new_run_dir(started_at: float, ended_at: float) -> Path | None:
    if not RESULTS_ROOT.exists():
        return None
    # 时间窗内 mtime 最新的 run dir
    candidates = []
    for child in RESULTS_ROOT.iterdir():
        if not child.is_dir():
            continue
        try:
            mtime = child.stat().st_mtime
        except FileNotFoundError:
            continue
        if started_at - 5 <= mtime <= ended_at + 5:
            candidates.append((mtime, child))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


# --------------------------------------------------------------------------- #
# 4. 解析每次 run 的结果
# --------------------------------------------------------------------------- #
def _safe_load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fd:
            return json.load(fd)
    except Exception:
        return {}


def _grep_count(log_path: Path, needle: str) -> int:
    if not log_path.exists():
        return 0
    n = 0
    try:
        with log_path.open("r", encoding="utf-8", errors="ignore") as fd:
            for line in fd:
                if needle in line:
                    n += 1
    except Exception:
        return 0
    return n


def parse_run_dir(run_dir: Path) -> list[dict[str, Any]]:
    """把一个 run_dir 解析成 N 条 (case_label, ...) 记录。"""
    rows: list[dict[str, Any]] = []
    if not run_dir or not run_dir.exists():
        return rows

    index = _safe_load_json(run_dir / "index.json")
    tasks_meta = {}
    for item in index.get("tasks", []) or []:
        if isinstance(item, dict):
            key = (str(item.get("runner_task_name", "")), int(item.get("task_id", -1)))
            tasks_meta[key] = item

    agent_runs_root = ROOT / "agent" / "runs"

    for case in CASES:
        # 在 run_dir 里找 task_NNN_{name}_{id}/result.json
        match = None
        prefix = f"_{case['name']}_{case['id']}"
        for d in sorted(run_dir.glob("task_*")):
            if d.name.endswith(prefix):
                match = d / "result.json"
                break
        if match is None or not match.exists():
            rows.append({
                "case": case["label"], "task_name": case["name"],
                "result_found": False,
            })
            continue

        result = _safe_load_json(match)
        # 在 agent/runs/ 下找日志，统计 dual-brain 触发次数
        agent_run_name = result.get("agent_run_dir_name") or ""
        log_path = agent_runs_root / agent_run_name / "logs" / "stardojo.log"

        replans  = _grep_count(log_path, "[BigBrain] Planning triggered by")
        mem_hit  = _grep_count(log_path, "[Routing] ⚡ Using memory quick path")
        mem_norm = _grep_count(log_path, "[Routing] ▶ Proceeding with normal planning")
        mem_dis  = _grep_count(log_path, "[Routing] ▶ Dual-brain mode: Mem0 quick path disabled")
        cb_route = _grep_count(log_path, "CIRCUIT-BREAKER")

        rows.append({
            "case":          case["label"],
            "task_name":     case["name"],
            "result_found":  True,
            "completed":     bool(result.get("completed")),
            "final_quantity": result.get("final_quantity"),
            "exit_step":     result.get("exit_step"),
            "step_budget":   result.get("step_budget"),
            "duration_sec":  round(float(result.get("duration_sec") or 0.0), 2),
            "llm_calls":     int(result.get("llm_call_count") or 0),
            "llm_breakdown": result.get("llm_call_breakdown") or {},
            "end_reason":    result.get("end_reason"),
            "budget_exit_reason": result.get("budget_exit_reason"),
            "avg_planning_sec": round(float(((result.get("perf_summary") or {}).get("avg") or {}).get("planning_sec") or 0.0), 2),
            "max_planning_sec": round(float(((result.get("perf_summary") or {}).get("max") or {}).get("planning_sec") or 0.0), 2),
            "agent_run":     agent_run_name,
            "log_exists":    log_path.exists(),
            # ↓ dual-brain 是否真的生效的硬证据
            "log_replans":   replans,
            "log_mem_quick_path_hits":     mem_hit,
            "log_mem_quick_path_normal":   mem_norm,
            "log_mem_quick_path_disabled": mem_dis,
            "log_circuit_breaker":         cb_route,
        })
    return rows


# --------------------------------------------------------------------------- #
# 5. 汇总报告
# --------------------------------------------------------------------------- #
def build_report(by_run: list[dict[str, Any]]) -> str:
    """by_run = [{condition, repeat, run_dir, rows: [...]}]"""
    lines: list[str] = []
    lines.append("# Dual-Brain Ablation Report")
    lines.append("")
    lines.append(f"- Generated: {dt.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Cases: {[c['label'] + ':' + c['name'] for c in CASES]}")
    lines.append("")

    lines.append("## 0. Config inspection")
    lines.append("| Condition | use_dual_brain | dual_brain.enabled | use_mem0 | mem0.enabled | dual_brain_effective | mem0_effective |")
    lines.append("|---|---|---|---|---|---|---|")
    seen = set()
    for entry in by_run:
        cond = entry["condition"]
        if cond in seen:
            continue
        seen.add(cond)
        ins = entry["inspection"]
        lines.append(
            f"| {cond.upper()} | {ins.get('features.use_dual_brain')} | {ins.get('dual_brain.enabled')} "
            f"| {ins.get('features.use_mem0')} | {ins.get('mem0.enabled')} "
            f"| **{ins.get('dual_brain_effective')}** | {ins.get('mem0_effective')} |"
        )
    lines.append("")

    lines.append("## 1. Per-case comparison")
    lines.append("")
    # 按 case 分组打表
    for case in CASES:
        lines.append(f"### Case: `{case['label']}` — {case['name']}  (suite={case['type']}, id={case['id']})")
        lines.append("")
        lines.append("| Condition | Repeat | Completed | Steps / Budget | LLM calls | Duration (s) | Avg plan (s) | Max plan (s) | End reason | Replans | Mem quick-hit | Circuit-breaker |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for entry in by_run:
            for row in entry["rows"]:
                if row.get("case") != case["label"]:
                    continue
                if not row.get("result_found"):
                    lines.append(
                        f"| {entry['condition'].upper()} | {entry['repeat']} | ❓ no result | – | – | – | – | – | – | – | – | – |"
                    )
                    continue
                lines.append(
                    f"| {entry['condition'].upper()} | {entry['repeat']} "
                    f"| {'✅' if row['completed'] else '❌'} "
                    f"| {row['exit_step']} / {row['step_budget']} "
                    f"| {row['llm_calls']} "
                    f"| {row['duration_sec']} "
                    f"| {row['avg_planning_sec']} "
                    f"| {row['max_planning_sec']} "
                    f"| {row['end_reason']} "
                    f"| {row['log_replans']} "
                    f"| {row['log_mem_quick_path_hits']} "
                    f"| {row['log_circuit_breaker']} |"
                )
        lines.append("")

    lines.append("## 2. Insight checklist")
    lines.append("")
    lines.append("- 如果 ON 组的 `Replans` 仍然 = 0，说明 `[BigBrain] Planning triggered by` 这条日志没出现，dual-brain 实际可能仍然没启用——核对 enhanced_config 路径与代码 import 顺序。")
    lines.append("- 如果 ON 组 `Mem quick-hit` 仍然 = 0，说明 SA-MB / mem0 quick path 没命中，可能是 embedding provider 没就绪或阈值太严。")
    lines.append("- 期望对比：HARD 任务 OFF 组通常会跑满 150 步 0 完成；ON 组若真生效，应在 ~step 10 触发 replan，明显减少 `Circuit-breaker` 次数。")
    lines.append("- 期望对比：EASY / MEDIUM 在 ON 组下 LLM calls 应明显下降（quick path 复用），但 SR 不应下降。")
    lines.append("")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# 6. main
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm_config",   default="agent/conf/gemini_config.json")
    p.add_argument("--embed_config", default="agent/conf/gemini_config.json")
    p.add_argument("--env_config",   default="agent/conf/env_config_stardew.json")
    p.add_argument("--parallel_numb", type=int, default=1, help="Mac 建议 1")
    p.add_argument("--start_port",    type=int, default=10783)
    p.add_argument("--repeats",       type=int, default=1, help="每个 condition 重复几次（论文 3）")
    p.add_argument("--conditions",    default="on,off", help="逗号分隔，可选 on/off 或两者")
    p.add_argument("--dry_run",       action="store_true")
    p.add_argument("--report_only",   action="store_true",
                   help="不跑实验，只解析参数中已有的 run_dirs（通过 --existing_runs 提供）")
    p.add_argument("--existing_runs", default="",
                   help="逗号分隔，每个形如 `on:runs/results/20260608_xxx`，配合 --report_only 用")
    p.add_argument("--out_dir",       default="runs/ablation",
                   help="对照实验报告输出根目录")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    out_root = (ROOT / args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = out_root / f"session_{stamp}"
    session_dir.mkdir(parents=True, exist_ok=True)

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip() in CONDITIONS]
    if not conditions:
        print(f"[FATAL] no valid conditions; choose from {list(CONDITIONS)}")
        return 2

    print()
    print("####  Pre-flight config inspection  ####")
    for cond in conditions:
        print(f"\n--- {cond.upper()} ---")
        for k, v in inspect_config(CONDITIONS[cond]).items():
            print(f"  {k:30s} = {v}")
    print()

    by_run: list[dict[str, Any]] = []

    if args.report_only:
        # 直接解析用户给出的 run_dir 列表
        for spec in [s.strip() for s in args.existing_runs.split(",") if s.strip()]:
            cond, _, raw = spec.partition(":")
            run_dir = (ROOT / raw).resolve()
            if not run_dir.exists():
                print(f"[WARN] missing run_dir: {run_dir}")
                continue
            inspection = inspect_config(CONDITIONS.get(cond, CONDITIONS["off"]))
            by_run.append({
                "condition": cond, "repeat": 0,
                "yaml": str(CONDITIONS.get(cond, CONDITIONS["off"])),
                "inspection": inspection,
                "run_dir": str(run_dir),
                "returncode": 0,
                "rows": parse_run_dir(run_dir),
            })
    else:
        task_params_path = write_task_params()
        print(f"[INFO] task_params written to {task_params_path}")
        for cond in conditions:
            for rep in range(1, args.repeats + 1):
                outcome = run_condition(
                    cond,
                    llm_config=args.llm_config,
                    embed_config=args.embed_config,
                    env_config=args.env_config,
                    parallel_numb=args.parallel_numb,
                    start_port=args.start_port,
                    task_params_path=task_params_path,
                    dry_run=args.dry_run,
                    repeat_idx=rep,
                )
                if outcome["run_dir"]:
                    outcome["rows"] = parse_run_dir(Path(outcome["run_dir"]))
                else:
                    outcome["rows"] = []
                by_run.append(outcome)

                # 实时写一份临时报告（防止跑到一半被打断）
                (session_dir / "ablation_partial.json").write_text(
                    json.dumps(by_run, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8",
                )

    # 最终落盘
    (session_dir / "ablation_full.json").write_text(
        json.dumps(by_run, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    report_md = build_report(by_run)
    report_path = session_dir / "ablation_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    print("\n========== ABLATION REPORT ==========")
    print(report_md)
    print(f"\n[OK] JSON  → {session_dir / 'ablation_full.json'}")
    print(f"[OK] Markdown → {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
