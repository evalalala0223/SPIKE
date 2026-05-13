"""
LangGraph 条件路由 (Phase 1)

决定工作流中的分支和跳转逻辑。

设计原则：
1. 路由函数必须是纯函数（无副作用）
2. 返回值必须与 StateGraph 的条件边定义匹配
3. 添加详细日志，方便调试流程
4. 考虑边界情况和默认行为

作者: AI Development Team
日期: 2026-02-01
版本: 1.0.0
"""
from typing import Literal, Optional

import os

from cradle.runner.game_state import GameState
from cradle.log import Logger

logger = Logger()


def _load_enhanced_config() -> dict:
    try:
        import yaml
        from cradle.utils.file_utils import assemble_project_path

        config_path = assemble_project_path('./conf/enhanced_config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as file:
                return yaml.safe_load(file) or {}
    except Exception as e:
        logger.debug(f"[Routing] Failed to load enhanced config: {e}")
    return {}


# ========== 类型定义 ==========

ReflectionRoute = Literal["skip", "run"]
ContinuationRoute = Literal["continue", "retry", "end"]
MemoryRoute = Literal["use_memory", "run"]
ActionPlanRoute = Literal["execute", "plan_only"]


# ========== 路由函数 ==========

def _last_execution_was_clean(state: GameState) -> bool:
    """Check whether the last execution truly succeeded without any soft failures.

    LittleBrain marks ``success=True`` whenever skills were executed, even for
    blocked moves that only write ``errors_info``.  So checking ``success``
    alone is insufficient — we also inspect ``errors_info`` and the state-level
    failure signals.
    """
    if bool(state.get("uncertain_execution", False)):
        return False
    if bool(state.get("heightened_failure_signal", False)):
        return False

    exec_result = state.get('execution_result', {})
    has_feedback = bool(state.get("has_execution_feedback", False))

    # Prefer the synchronized external feedback when available.
    if has_feedback:
        exec_info = state.get('last_exec_info') or state.get('exec_info') or {}
        if isinstance(exec_info, dict):
            errors_info = str(exec_info.get('errors_info', '') or '').strip()
            if errors_info:
                return False

        state_errors = str(state.get('last_errors_info', '') or '').strip()
        if state_errors:
            return False

        latest_task_eval = state.get("latest_task_eval", {})
        if isinstance(latest_task_eval, dict) and latest_task_eval.get("completed") is True:
            return True

        progress_delta = state.get("task_progress_delta", None)
        if progress_delta not in (None, "", 0, 0.0):
            return True

        if bool(state.get("last_state_changed", False)):
            return True

        success = state.get("success", None)
        if success is not None:
            return bool(success)

        return False

    if not isinstance(exec_result, dict):
        return False
    if bool(exec_result.get("pending", False)):
        return False
    if not bool(exec_result.get('success', False)):
        return False
    if bool(exec_result.get('error')):
        return False

    exec_info = exec_result.get('exec_info') or state.get('last_exec_info') or state.get('exec_info') or {}
    if isinstance(exec_info, dict):
        errors_info = str(exec_info.get('errors_info', '') or '').strip()
        if errors_info:
            return False

    state_errors = str(state.get('last_errors_info', '') or '').strip()
    if state_errors:
        return False

    return True


def _has_reflectable_execution_feedback(state: GameState) -> bool:
    if bool(state.get("has_execution_feedback", False)):
        return True

    exec_result = state.get("execution_result", {})
    if not isinstance(exec_result, dict):
        return bool(exec_result)
    if bool(exec_result.get("pending", False)):
        return False
    return bool(exec_result)


# ========== 路由函数 ==========

def should_skip_reflection(state: GameState) -> ReflectionRoute:
    """
    决定是否跳过 self_reflection 节点
    
    跳过条件：
    - 首步（无历史数据可反思）
    
    Args:
        state: 当前游戏状态
    
    Returns:
        "skip": 跳过反思节点，直接进入 task_inference
        "run": 执行反思节点
    
    Example:
        >>> state: GameState = {"is_first_step": True}
        >>> should_skip_reflection(state)
        'skip'
    """
    # 检查是否首步
    if bool(state.get("blocker_replan_only", False)):
        logger.write("[Routing] Skipping self_reflection (blocker_replan_only)")
        return "skip"

    if state.get('is_first_step', False):
        return "skip"

    # 检查是否有执行结果可反思
    if not _has_reflectable_execution_feedback(state):
        return "skip"

    # Skip self_reflection when BigBrain was triggered by FailureDetector escalation
    # (the failure info is already captured in failure_signals, reflection adds little value)
    escalation_reason = str(state.get('escalation_reason', '') or '')
    if escalation_reason.startswith('failure_detector:'):
        logger.write(f"[Routing] Skipping self_reflection (escalation: {escalation_reason})")
        return "skip"

    # Phase 8.5: Skip self_reflection on routine cycle_complete re-planning,
    # but only when the last execution had no issues at all. Blocked moves
    # and other soft failures set errors_info but still mark success=True
    # (because the skill was executed), so we must also check errors_info.
    if escalation_reason == 'cycle_complete':
        if _last_execution_was_clean(state):
            logger.write("[Routing] Skipping self_reflection (cycle_complete, clean execution)")
            return "skip"

    enhanced_cfg = _load_enhanced_config()
    skip_cfg = (((enhanced_cfg.get('dual_brain', {}) or {}).get('big_brain', {}) or {}).get('skip_reflection', {}) or {})
    if bool(skip_cfg.get('on_previous_success', True)):
        if _last_execution_was_clean(state):
            # Only skip if there is no accumulated zero-progress streak.
            # When the agent has been making no progress, reflection is
            # valuable even after a "clean" execution step.
            zero_progress = int(state.get('zero_progress_streak', 0) or 0)
            consecutive_failures = int(state.get('consecutive_failures', 0) or 0)
            if zero_progress < 2 and consecutive_failures < 2:
                return "skip"
            else:
                logger.write(
                    f"[Routing] Keeping self_reflection despite clean execution "
                    f"(zero_progress={zero_progress}, consecutive_failures={consecutive_failures})"
                )

    logger.write("[Routing] Running self_reflection")
    return "run"


def should_use_memory_action(state: GameState) -> MemoryRoute:
    """
    决定是否使用记忆快速路径

    条件：
    - memory_quick_path 为 True
    - planned_actions 非空
    - Phase 3: 双脑模式下禁用独立快速路径 (小脑自行参考 Mem0)
    """
    # Phase 3: dual-brain mode disables independent quick path
    if state.get('dual_brain_enabled', False):
        logger.write("[Routing] ▶ Dual-brain mode: Mem0 quick path disabled, proceeding with normal planning")
        return "run"

    if state.get('memory_quick_path', False) and state.get('planned_actions'):
        logger.write("[Routing] ⚡ Using memory quick path actions")
        return "use_memory"

    logger.write("[Routing] ▶ Proceeding with normal planning")
    return "run"


def should_execute_after_planning(state: GameState) -> ActionPlanRoute:
    """
    决定 action_planning 后是否执行 skill_execute

    双脑模式下跳过 skill_execute（大脑只规划，小脑负责执行）。
    标准模式下正常执行。

    Args:
        state: 当前游戏状态

    Returns:
        "execute": 正常执行 skill_execute
        "plan_only": 跳过执行，直接结束（双脑模式）
    """
    if state.get('dual_brain_enabled', False):
        logger.write("[Routing] Plan-only mode (dual-brain): skipping skill_execute")
        return "plan_only"

    return "execute"


def should_continue_or_retry(state: GameState) -> ContinuationRoute:
    """
    决定执行后的流程：继续 / 重试 / 结束
    
    决策逻辑：
    - 有错误 + 未达重试上限 → 重试
    - 有错误 + 达到重试上限 → 结束
    - 成功 → 继续
    - 默认 → 结束
    
    Args:
        state: 当前游戏状态
    
    Returns:
        "continue": 成功，继续下一步循环
        "retry": 失败，重新规划动作
        "end": 达到重试上限或其他原因，结束
    
    Example:
        >>> state: GameState = {"success": True}
        >>> should_continue_or_retry(state)
        'continue'
    """
    # 检查是否有错误
    error = state.get('error')
    success = state.get('success', False)
    retry_count = state.get('retry_count', 0)
    
    # 情况1：有明确错误
    if error:
        if retry_count >= 3:
            logger.warn(f"[Routing] 🛑 Max retries reached ({retry_count}/3), ending workflow")
            return "end"
        else:
            logger.warn(f"[Routing] ⚠️ Error detected: {error}")
            logger.write(f"[Routing] 🔄 Retrying (attempt {retry_count + 1}/3)")
            # Bug #23修复：递增retry_count（注意：这里只是日志，实际递增在节点里）
            return "retry"
    
    # 情况2：执行成功
    if success:
        logger.write("[Routing] ✅ Execution successful, continuing")
        return "continue"
    
    # 情况3：执行失败但没有错误（例如skill执行失败）
    if not success:
        if retry_count >= 3:
            logger.warn(f"[Routing] 🛑 Max retries reached ({retry_count}/3), ending workflow")
            return "end"
        else:
            logger.warn("[Routing] ⚠️ Execution failed (no explicit error)")
            logger.write(f"[Routing] 🔄 Retrying (attempt {retry_count + 1}/3)")
            # Bug #23修复：递增retry_count（注意：这里只是日志，实际递增在节点里）
            return "retry"
    
    # 默认：继续（理论上不应该到达这里）
    logger.warn("[Routing] ⚠️ Unexpected state (no success/error info), continuing")
    return "continue"


def check_parallel_completion(state: GameState) -> Literal["ready", "wait"]:
    """
    检查并行节点是否都完成
    
    用于同步 info_gathering 和 self_reflection 的并行执行。
    （注：Phase 1.0 暂不启用并行，此函数为未来预留）
    
    Args:
        state: 当前游戏状态
    
    Returns:
        "ready": 两个节点都完成，可以继续
        "wait": 等待另一个节点完成
    
    Note:
        这个函数在 Phase 1.0 中不会被使用，因为我们还没实现异步 Provider。
        预留给 Phase 1.5 并行优化。
    """
    has_info = 'gathered_info' in state and state['gathered_info']
    has_reflection = 'reflection_result' in state and state['reflection_result']
    
    if has_info and has_reflection:
        logger.write("[Routing] ✅ Both info_gathering and self_reflection completed")
        return "ready"
    
    logger.write("[Routing] ⏳ Waiting for parallel nodes to complete...")
    return "wait"


# ========== 辅助函数 ==========

def log_routing_decision(
    node_name: str,
    decision: str,
    reason: str,
    state_snapshot: Optional[dict] = None
):
    """
    记录路由决策（用于调试）
    
    Args:
        node_name: 节点名称
        decision: 决策结果
        reason: 决策原因
        state_snapshot: 状态快照（可选）
    """
    logger.write(f"[Routing] Node: {node_name} | Decision: {decision} | Reason: {reason}")
    
    if state_snapshot is not None:
        logger.debug(f"[Routing] State snapshot: {state_snapshot}")


def get_routing_statistics(state: GameState) -> dict:
    """
    获取路由统计信息（用于性能分析）
    
    Args:
        state: 当前游戏状态
    
    Returns:
        dict: 统计信息
            - skipped_reflections: 跳过的反思次数
            - skipped_task_inferences: 跳过的任务推理次数
            - retry_count: 重试次数
            - step_count: 总步数
    """
    # 这些统计可以从状态或日志中提取
    # 当前仅返回基础信息
    return {
        "step_count": state.get("step_count", 0),
        "retry_count": state.get("retry_count", 0),
        "consecutive_failures": state.get("consecutive_failures", 0),
        "is_first_step": state.get("is_first_step", False)
    }
