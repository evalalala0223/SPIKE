"""
LangGraph 状态定义 (Phase 1)

这个状态会在所有节点之间流转，替代原来的 working_area 字典传递。

设计原则：
1. 只包含必要字段，避免过度膨胀
2. 使用 total=False 允许部分字段可选
3. 清晰区分输入、中间状态、输出
4. 保持与现有 Provider 接口的兼容性

作者: AI Development Team
日期: 2026-02-01
版本: 1.0.0
"""
from typing import TypedDict, Dict, List, Tuple, Optional, Any


class GameState(TypedDict, total=False):
    """
    游戏 Agent 状态机
    
    这个 TypedDict 定义了 LangGraph 工作流中流转的状态对象。
    所有节点（Provider）都会接收这个状态，并返回更新的字段。
    
    使用示例：
        >>> state: GameState = {
        ...     "frame_ids": (0, 10),
        ...     "screenshot_path": "/path/to/screen.jpg",
        ...     "step_count": 0,
        ...     "is_first_step": True
        ... }
        >>> # 在节点中更新
        >>> return {"gathered_info": {...}, "step_count": state["step_count"] + 1}
    
    注意事项：
        - total=False 表示所有字段都是可选的
        - 节点只需返回需要更新的字段，不需要返回完整状态
        - LangGraph 会自动合并新旧状态
    """
    
    # ========== 输入字段 (由 Runner 初始化) ==========
    
    step_id: int
    """步骤计数器，从1开始递增，用于日志追踪和并发控制（Phase 2.1新增）"""
    
    frame_ids: Tuple[int, int]
    """视频帧范围 (start_frame_id, end_frame_id)"""
    
    screenshot_path: str
    """当前截图的文件路径（带坐标轴增强）"""
    
    video_clip: Dict[str, Any]
    """视频片段数据，包含多帧截图"""

    info_gathering_mode: str
    """信息收集输入模式，如 'stardew_original'"""
    
    work_dir: str
    """工作目录路径，如 runs/1769935376.7449024/"""
    
    env_name: str
    """游戏环境名称，如 'stardew', 'rdr2', 'skylines'"""
    
    env_config: Dict[str, Any]
    """环境配置，从 env_config_*.json 加载"""
    
    # ========== 中间状态 (由各个节点更新) ==========
    
    # Information Gathering 节点输出
    gathered_info: Dict[str, Any]
    """
    信息收集结果，包含：
    - description: str - 场景描述
    - target_object: str - 目标物体
    - minimap_info: Dict - 小地图信息（如有）
    - ui_elements: List - UI 元素列表（如有）
    """
    
    # Self Reflection 节点输出
    reflection_result: Dict[str, Any]
    """
    自我反思结果，包含：
    - reasoning: str - 推理过程
    - success: bool - 上一步是否成功
    - skipped: bool - 是否跳过反思（首步）
    """
    
    # Task Inference 节点输出
    task: str
    """当前短期任务描述"""

    main_task: str
    """主线任务描述（不应被 task inference 的子任务文本覆盖）"""

    subtask_description: str
    """当前子任务描述（供 little brain 与 planning 使用）"""
    
    long_horizon_task: str
    """长期目标描述"""
    
    task_changed: bool
    """任务是否发生变化（用于优化，避免重复推理）"""

    # ========== Phase 2.2: Mem0 记忆节点输出 ==========

    memory_hits: List[Dict[str, Any]]
    """Mem0 检索命中记录"""

    memory_confidence: float
    """Mem0 检索置信度"""

    memory_actions: List[str]
    """Mem0 建议动作列表（用于快速路径）"""

    memory_reference: str
    """格式化后的长期记忆参考文本，提供给规划提示词使用"""

    memory_quick_path: bool
    """是否启用记忆快速路径"""

    memory_retrieval_mode: str
    """记忆利用模式：execute（直执行）或 hint（仅提示给规划）"""

    quick_path_consecutive_hits: int
    """连续命中quick path次数（用于防振荡）"""

    quick_path_guard_reason: str
    """quick path被守卫拦截的原因（用于日志观测）"""
    
    # Action Planning 节点输出
    planned_actions: List[str]
    """
    计划的技能调用列表，如：
    ['move(x=1, y=0)', 'use(direction="down")', 'nop()']
    """
    
    planning_reasoning: str
    """LLM 的推理过程（action planning）"""
    
    skill_library: str
    """序列化的技能库字符串（JSON格式），由skill_curation生成"""
    
    # Skill Execution 节点输出
    execution_result: Dict[str, Any]
    """
    技能执行结果，包含：
    - success: bool - 是否成功
    - frame_ids: Tuple[int, int] - 执行期间的帧范围
    - error: Optional[str] - 错误信息
    """
    
    executed_frames: Tuple[int, int]
    """技能执行期间的帧范围"""
    
    # ========== 控制流字段 ==========
    
    step_count: int
    """当前步数计数器（从 0 开始）"""
    
    is_first_step: bool
    """
    是否为首步
    
    用途：首步跳过 self_reflection（因为没有历史数据）
    """
    
    success: bool
    """上一步是否成功（用于决定是否重试）"""

    execution_success_raw: bool
    """技能执行原始成功结果（未应用重复动作防护前）"""
    
    error: Optional[str]
    """错误信息（如果有）"""
    
    retry_count: int
    """当前重试次数（最大 3 次）"""
    
    # ========== 历史记录 (用于 memory) ==========
    
    previous_actions: List[str]
    """最近 K 步的动作历史（用于 context）"""
    
    previous_results: List[Dict[str, Any]]
    """最近 K 步的结果历史"""
    
    # ========== 优化标志 ==========

    ui_changed: bool
    """UI 界面是否变化（用于决定是否重新推理任务）"""

    consecutive_failures: int
    """连续失败次数（超过阈值时需要重新推理任务）"""

    # ========== Phase 3: 大小脑决策架构 ==========

    dual_brain_enabled: bool
    """是否启用双脑模式"""

    brain_mode: str
    """当前脑模式: "big" | "little" | "" (非双脑)"""

    suggestions: List[Dict[str, str]]
    """大脑的 4 步操作建议 [{"action": "...", "reason": "..."}]"""

    context_summary: str
    """大脑生成的状态摘要 (150字内, 给小脑参考)"""

    current_step: int
    """小脑当前执行步骤索引 (0-3)"""

    execution_log: List[Dict[str, Any]]
    """小脑累积执行日志 [{step, action, success, note}]"""

    escalation_reason: str
    """小脑升级到大脑的原因"""

    completed_steps: List[int]
    """已完成的步骤索引 (失败恢复时传给大脑)"""

    vllm_available: bool
    """vLLM 服务是否可用"""

    env_changed: bool
    """环境是否发生突变 (CLIP change_score > threshold)"""

    env_change_score: float
    """环境变化分数 (0=相同, 1=完全不同)"""

    fail_level: str
    """失败分级: F0 | F1 | F2 | F3"""

    fail_score: float
    """失败评分（FailureDetector 加权得分）"""

    failure_reasons: List[str]
    """失败原因列表（信号明细）"""

    decision_trace: str
    """失败检测决策轨迹（用于日志与复盘）"""


# ========== 类型别名 ==========

ProviderOutput = Dict[str, Any]
"""Provider 节点的返回类型（部分状态更新）"""


# ========== 验证函数 ==========

def validate_initial_state(state: GameState) -> bool:
    """
    验证初始状态是否包含所有必需字段
    
    Args:
        state: 待验证的状态对象
    
    Returns:
        bool: 是否有效
    
    Raises:
        ValueError: 如果缺少必需字段
    """
    required_fields = [
        "frame_ids",
        "screenshot_path",
        "work_dir",
        "step_count",
        "is_first_step"
    ]
    
    missing = [field for field in required_fields if field not in state]
    
    if missing:
        raise ValueError(f"Initial state missing required fields: {missing}")
    
    return True


def create_initial_state(
    frame_ids: Tuple[int, int],
    screenshot_path: str,
    work_dir: str,
    env_name: str,
    env_config: Dict[str, Any]
) -> GameState:
    """
    创建初始状态对象（便捷函数）
    
    Args:
        frame_ids: 帧范围
        screenshot_path: 截图路径
        work_dir: 工作目录
        env_name: 环境名称
        env_config: 环境配置
    
    Returns:
        GameState: 初始化的状态对象
    
    Example:
        >>> state = create_initial_state(
        ...     frame_ids=(-1, 19),
        ...     screenshot_path="./runs/xxx/screen_xxx.jpg",
        ...     work_dir="./runs/xxx/",
        ...     env_name="stardew",
        ...     env_config={"game": "Stardew Valley"}
        ... )
    """
    state: GameState = {
        # 输入
        "step_id": 0,  # Phase 2.1: 初始为0，由workflow递增
        "frame_ids": frame_ids,
        "screenshot_path": screenshot_path,
        "work_dir": work_dir,
        "env_name": env_name,
        "env_config": env_config,

        # 控制流
        "step_count": 0,
        "is_first_step": True,
        "retry_count": 0,
        "consecutive_failures": 0,

        # 历史
        "previous_actions": [],
        "previous_results": [],
        "main_task": "",
        "subtask_description": "",

        # 默认值
        "success": True,
        "execution_success_raw": True,
        "ui_changed": False,

        # Phase 3: 双脑架构默认值
        "dual_brain_enabled": False,
        "brain_mode": "",
        "suggestions": [],
        "context_summary": "",
        "current_step": 0,
        "execution_log": [],
        "escalation_reason": "",
        "completed_steps": [],
        "vllm_available": False,
        "env_changed": False,
        "env_change_score": 0.0,
        "fail_level": "F0",
        "fail_score": 0.0,
        "failure_reasons": [],
        "decision_trace": "",
    }
    
    validate_initial_state(state)
    
    return state


if __name__ == "__main__":
    # 测试
    print("Testing GameState...")
    
    # 测试创建
    state = create_initial_state(
        frame_ids=(0, 10),
        screenshot_path="/test/screen.jpg",
        work_dir="/test/runs/123/",
        env_name="stardew",
        env_config={"game": "test"}
    )
    
    print(f"✅ Initial state created: {list(state.keys())}")
    
    # 测试更新
    state["gathered_info"] = {"description": "test"}
    state["step_count"] = 1
    
    print(f"✅ State updated: step_count={state['step_count']}")
    
    print("\n✅ All tests passed!")
