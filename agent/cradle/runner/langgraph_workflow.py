"""
LangGraph 工作流构建 (Phase 1)

将节点、路由、检查点组装成完整的 StateGraph。

设计原则：
1. 清晰的图结构定义（节点 + 边）
2. 条件路由优化性能
3. Checkpoint 支持故障恢复
4. 详细日志记录流程

作者: AI Development Team
日期: 2026-02-01
版本: 1.0.0
"""
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from cradle.runner.game_state import GameState
from cradle.runner.langgraph_nodes import LangGraphNodes
from cradle.runner.langgraph_routing import (
    should_skip_reflection,
    should_continue_or_retry,
    should_use_memory_action,
    should_execute_after_planning
)
from cradle.log import Logger

logger = Logger()


def _log_graph_structure(mem0_enabled: bool, parallel_mode: bool) -> None:
    logger.write("[LangGraph] Graph structure:")

    if parallel_mode:
        if mem0_enabled:
            lines = [
                "[LangGraph]   START -> memory_retrieve",
                "[LangGraph]            [conditional]",
                "[LangGraph]            |- use_memory -> skill_execute",
                "[LangGraph]            '- run -> parallel_info_and_reflect -> memory_store -> task_inference",
                "[LangGraph]                                                              -> action_planning",
                "[LangGraph]                                                              -> skill_execute",
            ]
        else:
            lines = [
                "[LangGraph]   START -> parallel_info_and_reflect -> task_inference",
                "[LangGraph]                                          -> action_planning",
                "[LangGraph]                                          -> skill_execute",
            ]
    else:
        if mem0_enabled:
            lines = [
                "[LangGraph]   START -> memory_retrieve",
                "[LangGraph]            [conditional]",
                "[LangGraph]            |- use_memory -> skill_execute",
                "[LangGraph]            '- run -> info_gathering",
                "[LangGraph]                       [conditional]",
                "[LangGraph]                       |- skip -> memory_store -> task_inference",
                "[LangGraph]                       '- run -> self_reflection -> memory_store -> task_inference",
                "[LangGraph]                                                              -> action_planning",
                "[LangGraph]                                                              -> skill_execute",
            ]
        else:
            lines = [
                "[LangGraph]   START -> info_gathering",
                "[LangGraph]            [conditional]",
                "[LangGraph]            |- skip -> task_inference",
                "[LangGraph]            '- run -> self_reflection -> task_inference",
                "[LangGraph]                                           -> action_planning",
                "[LangGraph]                                           -> skill_execute",
            ]

    for line in lines:
        logger.write(line)

    logger.write("[LangGraph]                                          [conditional]")
    logger.write("[LangGraph]                                          |- continue -> END")
    logger.write("[LangGraph]                                          |- retry -> action_planning")
    logger.write("[LangGraph]                                          '- end -> END")


def build_game_workflow(
    providers: Dict[str, Any],
    enable_checkpoint: bool = True,
    gm: Any = None,
    augment_provider: Any = None,
    parallel_mode: bool = False,
    runtime_memory: Any = None,
):
    """
    构建游戏 Agent 工作流
    
    工作流结构（串行模式）：
        START
          ↓
        info_gathering
          ↓
        [条件] should_skip_reflection?
          ├─ skip → task_inference
          └─ run → self_reflection → task_inference
                                        ↓
                                                ↓
                                                skill_execute
                                                      ↓
                                                [条件] should_continue_or_retry?
                                                      ├─ continue → END
                                                      ├─ retry → action_planning (重新规划)
                                                      └─ end → END
    
    工作流结构（并行模式，Phase 2.1）：
        START
          ↓
        parallel_info_and_reflect  (info_gathering || self_reflection)
          ↓
        task_inference
          ↓
        action_planning
          ↓
        skill_execute
          ↓
        [retry or END]
    
    Args:
        providers: Provider 实例字典
            {
                'video_clip': VideoClipProvider,
                'self_reflection': SelfReflectionProvider,
                'task_inference': TaskInferenceProvider,
                'action_planning': ActionPlanningProvider,
                'skill_execute': SkillExecuteProvider
            }
        enable_checkpoint: 是否启用 Checkpoint（默认 True）
        gm: GameManager 实例（用于每步截图）
        augment_provider: AugmentProvider 实例（用于每步增强）
        parallel_mode: 🚀 Phase 2.1 - 是否启用并行执行（默认False保持兼容）
    
    Returns:
        CompiledGraph: 编译后的 LangGraph 应用
    
    Example:
        >>> providers = {...}  # Provider 实例
        >>> app = build_game_workflow(providers, parallel_mode=True)  # 启用并行
        >>> initial_state = create_initial_state(...)
        >>> result = app.invoke(initial_state, {"configurable": {"thread_id": "session_1"}})
    """
    logger.write("=" * 80)
    logger.write("[LangGraph] Building game workflow (Phase 1)...")
    logger.write("=" * 80)
    
    # ========== 步骤 1: 创建节点适配器 ==========
    nodes = LangGraphNodes(
        providers,
        gm=gm,
        augment_provider=augment_provider,
        action_planning_preprocess=providers.get('action_planning_preprocess') if isinstance(providers, dict) else None,
        action_planning_postprocess=providers.get('action_planning_postprocess') if isinstance(providers, dict) else None,
        self_reflection_preprocess=providers.get('self_reflection_preprocess') if isinstance(providers, dict) else None,
        self_reflection_postprocess=providers.get('self_reflection_postprocess') if isinstance(providers, dict) else None,
        info_gathering_preprocess=providers.get('information_gathering_preprocess') if isinstance(providers, dict) else None,
        info_gathering_postprocess=providers.get('information_gathering_postprocess') if isinstance(providers, dict) else None,
        task_inference_preprocess=providers.get('task_inference_preprocess') if isinstance(providers, dict) else None,
        task_inference_postprocess=providers.get('task_inference_postprocess') if isinstance(providers, dict) else None,
        runtime_memory=runtime_memory,
    )
    logger.write("[LangGraph] ✅ Node adapters created")
    
    # ========== Phase 2.1: step_id 计数器与注入机制 ==========
    step_counter = {"count": 0}
    
    def inject_step_id(node_func):
        """
        包装节点函数，自动注入step_id到state
        
        Phase 2.1优化：统一管理step_id，用于：
        1. 日志追踪：[step_id=N][node=X] 格式
        2. 并发控制：LocalMemory按step_id合并
        3. 调试支持：按step_id过滤回放
        """
        def wrapped(state: GameState):
            step_counter["count"] += 1
            state_dict = dict(state)  # 复制state避免修改原对象
            state_dict["step_id"] = step_counter["count"]
            logger.write(f"[step_id={step_counter['count']}] → Entering node")
            return node_func(state_dict)
        return wrapped
    
    logger.write("[LangGraph] ✅ step_id injection mechanism ready")
    
    # ========== 步骤 2: 创建 StateGraph ==========
    workflow = StateGraph(GameState)
    logger.write("[LangGraph] ✅ StateGraph initialized")
    
    # ========== Phase 2.1: 并行模式门控（仅在外层已通过安全校验时启用） ==========
    if parallel_mode:
        logger.write("[LangGraph] ✅ Parallel mode enabled by gate (info gathering scope)")
    else:
        logger.write("[LangGraph] Serial mode active")

    # ========== Phase 2.2: 检查 Mem0 配置 ==========
    mem0_enabled = False
    try:
        import yaml
        import os
        from cradle.utils.file_utils import assemble_project_path
        config_path = assemble_project_path('./conf/enhanced_config.yaml')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f)
                features_cfg = cfg.get('features', {})
                mem0_cfg = cfg.get('mem0', {})
                mem0_enabled = bool(features_cfg.get('use_mem0', False)) and bool(mem0_cfg.get('enabled', False))
                if mem0_enabled:
                    logger.write("[LangGraph] ✅ Mem0 enabled (from enhanced_config.yaml)")
    except Exception as e:
        logger.debug(f"[LangGraph] Could not read mem0 config: {e}")
    
    # ========== 步骤 3: 添加节点（根据模式） ==========
    if parallel_mode:
        # 🚀 并行模式：使用parallel_info_and_reflect节点
        if mem0_enabled:
            workflow.add_node("memory_retrieve", inject_step_id(nodes.memory_retrieve_node))
            workflow.add_node("memory_store", inject_step_id(nodes.memory_store_node))
        workflow.add_node("parallel_info_and_reflect", inject_step_id(nodes.parallel_info_and_reflect_node))
        workflow.add_node("task_inference", inject_step_id(nodes.task_inference_node))
        workflow.add_node("action_planning", inject_step_id(nodes.action_planning_node))
        workflow.add_node("skill_execute", inject_step_id(nodes.skill_execute_node))
        
        logger.write("[LangGraph] ✅ Added 4 nodes (PARALLEL mode): parallel_info_and_reflect, task_inference, action_planning, skill_execute")
        
        # 设置入口点
        if mem0_enabled:
            workflow.set_entry_point("memory_retrieve")
            logger.write("[LangGraph] ✅ Entry point set to 'memory_retrieve'")
        else:
            workflow.set_entry_point("parallel_info_and_reflect")
            logger.write("[LangGraph] ✅ Entry point set to 'parallel_info_and_reflect'")
        
    else:
        # 串行模式（原逻辑）
        if mem0_enabled:
            workflow.add_node("memory_retrieve", inject_step_id(nodes.memory_retrieve_node))
            workflow.add_node("memory_store", inject_step_id(nodes.memory_store_node))
        workflow.add_node("info_gathering", inject_step_id(nodes.info_gathering_node))
        workflow.add_node("self_reflection", inject_step_id(nodes.self_reflection_node))
        workflow.add_node("task_inference", inject_step_id(nodes.task_inference_node))
        workflow.add_node("action_planning", inject_step_id(nodes.action_planning_node))
        workflow.add_node("skill_execute", inject_step_id(nodes.skill_execute_node))
        
        logger.write("[LangGraph] ✅ Added 5 nodes (SERIAL mode): info_gathering, self_reflection, task_inference, action_planning, skill_execute")
        
        # 设置入口点
        if mem0_enabled:
            workflow.set_entry_point("memory_retrieve")
            logger.write("[LangGraph] ✅ Entry point set to 'memory_retrieve'")
        else:
            workflow.set_entry_point("info_gathering")
            logger.write("[LangGraph] ✅ Entry point set to 'info_gathering'")
    
    # ========== 步骤 5: 定义边（数据流） ==========
    
    if parallel_mode:
        # 🚀 并行模式边定义（简化流程）
        if mem0_enabled:
            workflow.add_conditional_edges(
                "memory_retrieve",
                should_use_memory_action,
                {
                    "use_memory": "skill_execute",
                    "run": "parallel_info_and_reflect"
                }
            )
            logger.write("[LangGraph] ✅ Conditional edge: memory_retrieve → [use_memory/run]")

        if mem0_enabled:
            workflow.add_edge("parallel_info_and_reflect", "memory_store")
            logger.write("[LangGraph] ✅ Edge (PARALLEL): parallel_info_and_reflect → memory_store")
            workflow.add_edge("memory_store", "task_inference")
            logger.write("[LangGraph] ✅ Edge (PARALLEL): memory_store → task_inference")
        else:
            workflow.add_edge("parallel_info_and_reflect", "task_inference")
            logger.write("[LangGraph] ✅ Edge (PARALLEL): parallel_info_and_reflect → task_inference")
        
        workflow.add_edge("task_inference", "action_planning")
        logger.write("[LangGraph] ✅ Edge (PARALLEL): task_inference → action_planning")

        workflow.add_conditional_edges(
            "action_planning",
            should_execute_after_planning,
            {
                "execute": "skill_execute",
                "plan_only": END
            }
        )
        logger.write("[LangGraph] ✅ Conditional edge (PARALLEL): action_planning → [execute/plan_only]")
        
        workflow.add_conditional_edges(
            "skill_execute",
            should_continue_or_retry,
            {
                "continue": END,
                "retry": "action_planning",
                "end": END
            }
        )
        logger.write("[LangGraph] ✅ Conditional edge (PARALLEL): skill_execute → [continue/retry/end]")
        
    else:
        # 串行模式边定义（原逻辑）
        if mem0_enabled:
            workflow.add_conditional_edges(
                "memory_retrieve",
                should_use_memory_action,
                {
                    "use_memory": "skill_execute",
                    "run": "info_gathering"
                }
            )
            logger.write("[LangGraph] ✅ Conditional edge: memory_retrieve → [use_memory/run]")

        # Edge 1: info_gathering → conditional (skip or run self_reflection)
        workflow.add_conditional_edges(
            "info_gathering",
            should_skip_reflection,
            {
                "run": "self_reflection",  # 执行反思
                "skip": "memory_store" if mem0_enabled else "task_inference",  # 跳过反思，直接任务推理
            }
        )
        if mem0_enabled:
            logger.write("[LangGraph] ✅ Conditional edge: info_gathering → [skip->memory_store / run->self_reflection]")
        else:
            logger.write("[LangGraph] ✅ Conditional edge: info_gathering → [skip/run] self_reflection")
        
        # Edge 2: self_reflection → task_inference (always)
        if mem0_enabled:
            workflow.add_edge("self_reflection", "memory_store")
            logger.write("[LangGraph] ✅ Edge: self_reflection → memory_store")
            workflow.add_edge("memory_store", "task_inference")
            logger.write("[LangGraph] ✅ Edge: memory_store → task_inference")
        else:
            workflow.add_edge("self_reflection", "task_inference")
            logger.write("[LangGraph] ✅ Edge: self_reflection → task_inference")
        
        # Edge 3: task_inference → action_planning (always, even if skipped)
        # 注意：即使跳过任务推理，也要执行 action planning
        workflow.add_edge("task_inference", "action_planning")
        logger.write("[LangGraph] ✅ Edge: task_inference → action_planning")

        # Edge 4: action_planning → conditional (execute or plan_only)
        # Phase 3: 双脑模式下跳过 skill_execute，大脑只规划不执行
        workflow.add_conditional_edges(
            "action_planning",
            should_execute_after_planning,
            {
                "execute": "skill_execute",
                "plan_only": END
            }
        )
        logger.write("[LangGraph] ✅ Conditional edge: action_planning → [execute/plan_only]")
        
        # Edge 5: skill_execute → conditional (continue/retry/end)
        workflow.add_conditional_edges(
            "skill_execute",
            should_continue_or_retry,
            {
                "continue": END,                 # 成功，结束本次循环
                "retry": "action_planning",      # 失败，重新规划动作
                "end": END                       # 达到最大重试次数或其他原因
            }
        )
        logger.write("[LangGraph] ✅ Conditional edge: skill_execute → [continue/retry/end]")
    
    logger.write("[LangGraph] ✅ Workflow graph constructed")
    
    # ========== 步骤 6: 添加 Checkpoint ==========
    # GameState 使用 TypedDict 定义，所有字段都是可序列化的基本类型
    # 使用 InMemorySaver 实现内存中的 checkpoint（不持久化到磁盘）
    if enable_checkpoint:
        memory = InMemorySaver()
        logger.write("[LangGraph] ✅ Checkpoint enabled (in-memory)")
    else:
        memory = None
        logger.write("[LangGraph] ⚠️ Checkpoint disabled by config")
    
    # ========== 步骤 7: 编译工作流 ==========
    app = workflow.compile(checkpointer=memory)
    
    logger.write("=" * 80)
    logger.write("[LangGraph] 🚀 Workflow compiled and ready!")
    _log_graph_structure(mem0_enabled=mem0_enabled, parallel_mode=parallel_mode)
    logger.write("=" * 80)
    
    return app


# ========== 并行版本（未来优化，Phase 1.5） ==========

def build_parallel_workflow(providers: Dict[str, Any], enable_checkpoint: bool = True):
    """
    并行优化版本（Phase 1.5 - 未来实现）
    
    info_gathering 和 self_reflection 可以并行执行。
    
    注意：
        需要 Provider 支持异步（Phase 0 还未实现）
        这个函数暂时不会被使用，预留给未来优化。
    
    工作流结构（并行）：
        START
          ↓
        info_gathering
          ├──────────────────┐
          ↓            ↓            ↓
        self_reflection  (parallel)  (直接到同步点)
          ↓            ↓            ↓
          └──────────────────┘
                        ↓
                  sync_point (同步等待)
                        ↓
                  task_inference
                        ↓
                     (后续流程同上)
    
    Args:
        providers: Provider 实例字典
        enable_checkpoint: 是否启用 Checkpoint
    
    Returns:
        CompiledGraph: 编译后的并行工作流
    """
    logger.warn("[LangGraph] Parallel workflow is not yet implemented (Phase 1.5)")
    logger.warn("[LangGraph] Falling back to sequential workflow")
    
    # 暂时返回顺序版本
    return build_game_workflow(providers, enable_checkpoint)


# ========== 可视化辅助 ==========

def visualize_workflow(app):
    """
    可视化工作流图（需要安装 graphviz）
    
    Args:
        app: 编译后的 LangGraph 应用
    
    Returns:
        str: Mermaid 图语法（可在 markdown 中渲染）
    
    Example:
        >>> app = build_game_workflow(providers)
        >>> mermaid = visualize_workflow(app)
        >>> print(mermaid)
    """
    try:
        # LangGraph 内置的可视化功能
        mermaid = app.get_graph().draw_mermaid()
        return mermaid
    except Exception as e:
        logger.warn(f"[LangGraph] Failed to visualize workflow: {e}")
        return None


def get_workflow_statistics(app) -> Dict[str, Any]:
    """
    获取工作流统计信息
    
    Args:
        app: 编译后的 LangGraph 应用
    
    Returns:
        dict: 统计信息
            - node_count: 节点数量
            - edge_count: 边数量
            - conditional_edges: 条件边数量
    """
    graph = app.get_graph()
    
    # 获取节点和边的信息
    nodes = list(graph.nodes.keys()) if hasattr(graph, 'nodes') else []
    edges = list(graph.edges) if hasattr(graph, 'edges') else []
    
    return {
        "node_count": len(nodes),
        "nodes": nodes,
        "edge_count": len(edges),
        "has_checkpoint": app.checkpointer is not None
    }


if __name__ == "__main__":
    # 测试工作流构建
    print("Testing workflow construction...")
    
    # Mock providers
    class MockProvider:
        def gather_information(self, **kwargs):
            return {"description": "test"}
        
        def reflect(self, **kwargs):
            return {"reasoning": "test", "success": True}
        
        def infer_task(self, **kwargs):
            return {"task": "test task", "long_horizon_task": "test"}
        
        def plan_action(self, **kwargs):
            return {"actions": ["action1()"], "reasoning": "test"}
        
        def execute(self, **kwargs):
            return {"success": True, "frame_ids": (0, 10)}
    
    mock = MockProvider()
    providers = {
        'video_clip': mock,
        'self_reflection': mock,
        'task_inference': mock,
        'action_planning': mock,
        'skill_execute': mock
    }
    
    # 构建工作流
    app = build_game_workflow(providers, enable_checkpoint=True)
    print("✅ Workflow built successfully")
    
    # 获取统计
    stats = get_workflow_statistics(app)
    print(f"✅ Workflow statistics: {stats}")
    
    # 测试执行（简单状态）
    from cradle.runner.game_state import create_initial_state
    
    initial_state = create_initial_state(
        frame_ids=(0, 10),
        screenshot_path="/test.jpg",
        work_dir="/test/",
        env_name="test",
        env_config={}
    )
    
    print("✅ Initial state created")
    
    # 注意：这里不实际执行，因为 mock provider 接口不完整
    # 实际执行需要在集成测试中进行
    
    print("\n✅ All workflow construction tests passed!")
