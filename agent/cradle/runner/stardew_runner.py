import os
import atexit
import time
from typing import Dict, Any

from cradle.utils.dict_utils import kget
from cradle.utils.string_utils import replace_unsupported_chars
from cradle import constants
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.memory.base import BaseMemory
from cradle.provider.llm.llm_factory import LLMFactory
from cradle.environment.skill_registry_factory import SkillRegistryFactory
from cradle.environment.ui_control_factory import UIControlFactory
from cradle.gameio.io_env import IOEnvironment
from cradle.gameio.game_manager import GameManager
from cradle.provider import VideoRecordProvider
from cradle.provider import VideoClipProvider
from cradle.provider import StardewInformationGatheringPreprocessProvider
from cradle.provider import StardewInformationGatheringPostprocessProvider
from cradle.provider import StardewSelfReflectionPreprocessProvider
from cradle.provider import StardewSelfReflectionPostprocessProvider
from cradle.provider import StardewTaskInferencePreprocessProvider
from cradle.provider import StardewTaskInferencePostprocessProvider
from cradle.provider import StardewActionPlanningPreprocessProvider
from cradle.provider import StardewActionPlanningPostprocessProvider
from cradle.provider import StardewInformationGatheringProvider
from cradle.provider import StardewSelfReflectionProvider
from cradle.provider import StardewActionPlanningProvider
from cradle.provider import StardewTaskInferenceProvider
from cradle.provider import SkillCurationProvider
from cradle.provider import SkillExecuteProvider
from cradle.provider import AugmentProvider
from cradle.planner.stardew_planner import StardewPlanner
from log_processor import process_log_messages
from stardojo.utils.cortex_runtime_utils import (
    LEGACY_COMPACT_PROMPT_SOURCE,
    CortexConfigurationError,
    resolve_little_brain_prompt_source,
)
from stardojo.utils.prompt_profile_utils import sync_planner_prompt_templates

config = Config()
logger = Logger()
io_env = IOEnvironment()

# ========== Phase 1: LangGraph imports ==========
try:
    from cradle.runner.langgraph_workflow import build_game_workflow
    from cradle.runner.game_state import create_initial_state, GameState
    LANGGRAPH_AVAILABLE = True
except ImportError as e:
    message = f"[Phase 1] LangGraph not available: {e}"
    if isinstance(e, ModuleNotFoundError) and getattr(e, "name", "") == "langgraph":
        message += " (missing dependency 'langgraph'; install with 'pip install -r requirements.txt')"
    logger.warn(message)
    LANGGRAPH_AVAILABLE = False


class PipelineRunner():

    def __init__(self,
                 llm_provider_config_path: str,
                 embed_provider_config_path: str,
                 task_description: str,
                 use_self_reflection: bool = False,
                 use_task_inference: bool = False):

        self.llm_provider_config_path = llm_provider_config_path
        self.embed_provider_config_path = embed_provider_config_path

        self.task_description = task_description
        self.use_self_reflection = use_self_reflection
        self.use_task_inference = use_task_inference

        # Init internal params
        self.set_internal_params()


    def set_internal_params(self, *args, **kwargs):

        self.provider_configs = config.provider_configs

        # Init LLM and embedding provider(s)
        lf = LLMFactory()
        self.llm_provider, self.embed_provider = lf.create(self.llm_provider_config_path,
                                                           self.embed_provider_config_path)

        srf = SkillRegistryFactory()
        srf.register_builder(config.env_short_name, config.skill_registry_name)
        self.skill_registry = srf.create(config.env_short_name, skill_configs=config.skill_configs,
                                         embedding_provider=self.embed_provider)

        self.video_recorder = VideoRecordProvider()
        self._shutdown_done = False

        ucf = UIControlFactory()
        ucf.register_builder(config.env_short_name, config.ui_control_name)
        self.env_ui_control = ucf.create(config.env_short_name)

        # Init game manager
        self.gm = GameManager(env_name=config.env_name,
                              embedding_provider=self.embed_provider,
                              llm_provider=self.llm_provider,
                              skill_registry=self.skill_registry,
                              ui_control=self.env_ui_control,
                              video_recorder=self.video_recorder,
                              )

        self.memory = LocalMemory()

        # Init planner
        from stardojo.utils.prompt_profile_utils import build_task_specific_planner_params

        planner_params, self.prompt_profile = build_task_specific_planner_params(
            config.planner_params,
            self.task_description,
        )
        self.planner = StardewPlanner(llm_provider=self.llm_provider,
                                   planner_params=planner_params,
                                   frame_extractor=None,
                                   icon_replacer=None,
                                   object_detector=None,
                                   use_self_reflection=True,
                                   use_task_inference=True)
        logger.write(
            f"[PromptProfile] Using '{self.prompt_profile}' templates for task '{self.task_description}'"
        )

        # Init skill library
        skills = self.gm.retrieve_skills(query_task=self.task_description,
                                         skill_num=config.skill_configs[constants.SKILL_CONFIG_MAX_COUNT],
                                         screen_type=constants.GENERAL_GAME_INTERFACE)

        self.skill_library = self.gm.get_skill_information(skills, config.skill_library_with_code)

        # 修复Bug：将skill_library转换为JSON字符串后存入memory，避免Skill对象序列化问题
        import json
        skill_library_str = json.dumps(self.skill_library, ensure_ascii=False)
        self.memory.update_info_history({"skill_library": skill_library_str})

        # Init video provider (inject shared recorder to avoid recorder instance split)
        self.video_clip = VideoClipProvider(gm=self.gm, video_recorder=self.video_recorder)

        self.provider_configs = config.provider_configs

        # Init augment providers
        augment_config = getattr(self.provider_configs, 'augment_provider', {})
        self.augment = AugmentProvider(**augment_config)
        self.augment_methods = [
            self.augment
        ]

        # Init module providers
        self.information_gathering_preprocess = StardewInformationGatheringPreprocessProvider(gm=self.gm)  # type: ignore
        self.information_gathering = StardewInformationGatheringProvider(planner=self.planner, gm=self.gm)
        info_gathering_postprocess_config = getattr(self.provider_configs, 'information_gathering_postprocess_provider', {})
        self.information_gathering_postprocess = StardewInformationGatheringPostprocessProvider(
            gm=self.gm,
            **info_gathering_postprocess_config
        )

        self.self_reflection_preprocess = StardewSelfReflectionPreprocessProvider(gm=self.gm, augment_methods=self.augment_methods)
        self.self_reflection = StardewSelfReflectionProvider(planner=self.planner, gm=self.gm)
        self.self_reflection_postprocess = StardewSelfReflectionPostprocessProvider(gm=self.gm)

        self.task_inference_preprocess = StardewTaskInferencePreprocessProvider(gm=self.gm)
        self.task_inference = StardewTaskInferenceProvider(planner=self.planner, gm=self.gm)
        self.task_inference_postprocess = StardewTaskInferencePostprocessProvider(gm=self.gm)

        action_planning_preprocess_config = getattr(self.provider_configs, 'action_planning_preprocess_provider', {})
        self.action_planning_preprocess = StardewActionPlanningPreprocessProvider(
            gm=self.gm,
            **action_planning_preprocess_config
        )
        self.action_planning = StardewActionPlanningProvider(planner=self.planner, gm=self.gm)
        self.action_planning_postprocess = StardewActionPlanningPostprocessProvider(gm=self.gm)

        self.skill_curation = SkillCurationProvider(gm=self.gm, video_recorder=self.video_recorder)

        # Init skill execute provider (inject shared recorder)
        self.skill_execute = SkillExecuteProvider(gm=self.gm, video_recorder=self.video_recorder)

        # Init checkpoint path
        self.checkpoint_path = os.path.join(config.work_dir, 'checkpoints')
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        # 🚀 Phase 2.1: 显示性能优化配置
        try:
            from cradle.config.enhanced_config import EnhancedConfig
            enhanced_cfg = EnhancedConfig()
            perf_cfg = enhanced_cfg._raw_config.get('performance', {})
            
            vision_cfg = perf_cfg.get('vision', {})
            streaming_cfg = perf_cfg.get('streaming', {})
            parallel_img_cfg = perf_cfg.get('parallel_image_encoding', {})  # Phase 2.1
            parallel_cfg = enhanced_cfg._raw_config.get('langgraph', {}).get('parallel', {})
            
            logger.write("=" * 80)
            logger.write("[Phase 2.1] Performance Optimizations Status:")
            logger.write(f"  - Parallel Image Encoding: {'✅ ENABLED' if parallel_img_cfg.get('enabled') else '❌ DISABLED'}")
            logger.write(f"  - Streaming LLM: {'✅ ENABLED' if streaming_cfg.get('enabled') else '❌ DISABLED'}")
            logger.write(f"  - Parallel Execution: {'✅ ENABLED' if parallel_cfg.get('enabled') else '❌ DISABLED'}")
            logger.write(f"  - Dynamic Frame Count: {'✅ ENABLED' if vision_cfg.get('dynamic_frame_count') else '❌ DISABLED'}")
            logger.write("=" * 80)
        except Exception as e:
            logger.debug(f"[Phase 2.1] Could not display performance config: {e}")

        # ========== Phase 1: Initialize LangGraph workflow ==========
        from cradle.config.enhanced_config import EnhancedConfig
        enhanced_cfg = EnhancedConfig()
        use_langgraph_enabled = enhanced_cfg._raw_config.get('features', {}).get('use_langgraph', False)
        self.use_langgraph = use_langgraph_enabled and LANGGRAPH_AVAILABLE
        self.workflow_app = None  # Type: CompiledGraph | None
        
        if self.use_langgraph:
            logger.write("=" * 80)
            logger.write("[Phase 1] Using LangGraph workflow engine")
            logger.write("=" * 80)
            
            self.langgraph_providers = {
                'video_clip': self.information_gathering,
                'information_gathering_preprocess': self.information_gathering_preprocess,
                'information_gathering_postprocess': self.information_gathering_postprocess,
                'self_reflection': self.self_reflection,
                'self_reflection_preprocess': self.self_reflection_preprocess,
                'self_reflection_postprocess': self.self_reflection_postprocess,
                'task_inference': self.task_inference,
                'task_inference_preprocess': self.task_inference_preprocess,
                'task_inference_postprocess': self.task_inference_postprocess,
                'action_planning': self.action_planning,
                'action_planning_preprocess': self.action_planning_preprocess,
                'action_planning_postprocess': self.action_planning_postprocess,
                'skill_execute': self.skill_execute
            }
            
            try:
                # 串行回退策略：并行路径已停用（当前并行为假并行）
                parallel_cfg = enhanced_cfg._raw_config.get('langgraph', {}).get('parallel', {})
                requested_parallel = bool(parallel_cfg.get('enabled', False))

                required_parallel_providers = [
                    'information_gathering_preprocess',
                    'information_gathering_postprocess',
                    'self_reflection_preprocess',
                    'self_reflection_postprocess',
                    'task_inference_preprocess',
                    'task_inference_postprocess',
                ]
                missing_parallel_deps = [
                    name for name in required_parallel_providers
                    if self.langgraph_providers.get(name) is None
                ]

                parallel_mode = False
                if requested_parallel:
                    logger.warn(
                        "[Phase 2.1] Parallel requested but forcibly disabled (serial rollback)."
                    )
                if missing_parallel_deps:
                    logger.warn(
                        "[Phase 2.1] Parallel deps check (informational): "
                        + ", ".join(missing_parallel_deps)
                    )
                else:
                    logger.write("[Phase 2.1] Serial mode active (parallel rollback)")
                
                self.workflow_app = build_game_workflow(
                    providers=self.langgraph_providers,
                    enable_checkpoint=True,
                    gm=self.gm,
                    augment_provider=self.augment,
                    parallel_mode=parallel_mode
                )
                logger.write("[Phase 1] LangGraph workflow compiled and ready")

                # ========== Phase 3: Initialize Dual Brain Controller ==========
                self.dual_brain_controller = None
                dual_brain_enabled = enhanced_cfg._raw_config.get('features', {}).get('use_dual_brain', False)
                if dual_brain_enabled and self.workflow_app is not None:
                    try:
                        from cradle.runner.dual_brain import DualBrainController

                        # Resolve Mem0 provider (if enabled)
                        mem0_provider = None
                        try:
                            from cradle.runner.langgraph_nodes import LangGraphNodes
                            # Check if Mem0 was initialized in the nodes
                            import yaml as _yaml
                            _cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'conf', 'enhanced_config.yaml')
                            from cradle.utils.file_utils import assemble_project_path
                            _cfg_path = assemble_project_path('./conf/enhanced_config.yaml')
                            if os.path.exists(_cfg_path):
                                with open(_cfg_path, 'r', encoding='utf-8') as _f:
                                    _raw = _yaml.safe_load(_f) or {}
                                    _mem0_on = bool(_raw.get('features', {}).get('use_mem0', False)) and bool(_raw.get('mem0', {}).get('enabled', False))
                                    if _mem0_on:
                                        from cradle.memory.mem0_provider import Mem0Provider
                                        mem0_provider = Mem0Provider(
                                            enabled=True,
                                            embedding_provider=self.embed_provider,
                                            storage_path=os.path.join(config.work_dir, 'mem0_records.json'),
                                            metrics_path=os.path.join(config.work_dir, 'mem0_metrics.json'),
                                            require_meaningful_progress=bool(_raw.get('mem0', {}).get('store_require_meaningful_progress', True)),
                                            progress_min_chars=int(_raw.get('mem0', {}).get('store_progress_min_chars', 8)),
                                        )
                        except Exception as mem0_err:
                            logger.debug(f"[Phase 3] Mem0 for dual brain not available: {mem0_err}")

                        self.dual_brain_controller = DualBrainController.from_config(
                            workflow_app=self.workflow_app,
                            gm=self.gm,
                            skill_execute_provider=self.skill_execute,
                            augment_provider=self.augment,
                            embed_provider=self.embed_provider,
                            mem0_provider=mem0_provider,
                        )

                        _resolved_paths, _template_audit, preserve_profile_templates = sync_planner_prompt_templates(
                            self.planner,
                            self.prompt_profile,
                            template_keys=("action_planning", "task_inference"),
                        )

                        if preserve_profile_templates:
                            logger.write(
                                f"[Phase 3] preserving profile-specific BigBrain templates for prompt_profile={self.prompt_profile}"
                            )
                        else:
                            logger.write(
                                f"[Phase 3] swapping BigBrain templates to cortex for prompt_profile={self.prompt_profile}"
                            )

                        little_brain_cfg = enhanced_cfg._raw_config.get('dual_brain', {}).get('little_brain', {})
                        use_stardew_template = bool(little_brain_cfg.get('use_stardew_template', False))
                        lb_template_path = assemble_project_path(
                            "./res/stardew/prompts/templates/action_planning_littlebrain.prompt"
                        )
                        little_brain_prompt_source = resolve_little_brain_prompt_source(
                            use_stardew_template=use_stardew_template,
                            little_brain_available=hasattr(self.dual_brain_controller, 'little_brain'),
                            template_path=lb_template_path,
                        )
                        if little_brain_prompt_source != LEGACY_COMPACT_PROMPT_SOURCE:
                            lb_template = read_resource_file(little_brain_prompt_source)
                            self.dual_brain_controller.little_brain.vllm_client.template = lb_template
                        logger.write(f"[Phase 3] LittleBrain prompt source: {little_brain_prompt_source}")

                        logger.write("[Phase 3] Dual brain controller initialized")
                    except Exception as db_err:
                        if isinstance(db_err, CortexConfigurationError):
                            raise
                        logger.warn(f"[Phase 3] Dual brain init failed: {db_err}, falling back to standard LangGraph")
                        self.dual_brain_controller = None

            except Exception as e:
                if isinstance(e, CortexConfigurationError):
                    raise
                logger.error(f"[Phase 1] Failed to build LangGraph workflow: {e}")
                logger.error("[Phase 1] Falling back to legacy runner")
                self.use_langgraph = False
                self.workflow_app = None
        else:
            if not LANGGRAPH_AVAILABLE:
                logger.warn("[Phase 1] LangGraph not available (missing langgraph package)")
            else:
                logger.write("[Phase 1] Using legacy sequential runner")
            self.workflow_app = None


    def pipeline_shutdown(self):

        if self._shutdown_done:
            logger.debug('[Shutdown] pipeline_shutdown already executed, skipping duplicate call')
            return
        self._shutdown_done = True

        self.gm.cleanup_io()
        self.video_recorder.finish_capture()

        # Merge all video slices into one complete replay video
        videos_dir = os.path.join(config.work_dir, 'videos')
        merged_video_path = os.path.join(config.work_dir, 'full_replay.mp4')
        try:
            self._merge_video_slices(videos_dir, merged_video_path)
        except Exception as e:
            logger.warn(f'Failed to merge video slices: {e}')

        log = process_log_messages(config.work_dir)

        with open(config.work_dir + '/logs/log.md', 'w', encoding='utf-8') as f:  # 修复: 指定 UTF-8 编码
            log = replace_unsupported_chars(log)
            f.write(log)

        logger.write('>>> Markdown generated.')
        logger.write('================================================================================')
        logger.write(f'>>> Video replay saved to: {merged_video_path}')
        logger.write(f'>>> Video slices in: {videos_dir}')
        logger.write(f'>>> Per-step clips in: {os.path.join(config.work_dir, "video_splits")}')
        logger.write(f'>>> Run directory: {config.work_dir}')
        logger.write('================================================================================')
        logger.write('>>> Bye.')

    def _merge_video_slices(self, videos_dir, output_path):
        """Merge all video slice files into one complete replay video."""
        import cv2
        import glob

        slice_files = sorted(glob.glob(os.path.join(videos_dir, '*.mp4')))
        if not slice_files:
            logger.warn('No video slices found to merge.')
            return

        # Read first slice to get video properties
        cap = cv2.VideoCapture(slice_files[0])
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if fps <= 0:
            fps = config.video_fps

        writer = cv2.VideoWriter(output_path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (width, height))

        total_frames = 0
        for slice_path in slice_files:
            cap = cv2.VideoCapture(slice_path)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                writer.write(frame)
                total_frames += 1
            cap.release()

        writer.release()
        logger.write(f'>>> Merged {len(slice_files)} slices into full_replay.mp4 ({total_frames} frames)')


    def run(self):

        # 1. Initiate the parameters
        success = False
        # 修复Bug：使用序列化的skill_library（从memory获取）
        skill_library_str = self.memory.get_latest("skill_library", "")
        init_params = {
            "task_description": self.task_description,
            "skill_library": skill_library_str,
            "exec_info": {
                "errors": False,
                "errors_info": ""
            },
            "pre_action": "",
            "pre_decision_making_reasoning": "",
            "pre_self_reflection_reasoning": "",
            "summarization": "",
            "toolbar_information": None,
            "subtask_description": "",
            "subtask_reasoning": "",
        }

        self.memory.update_info_history(init_params)

        # 2. Switch to game
        self.gm.switch_to_game()

        # 3. Start video recording
        self.video_recorder.start_capture()

        # 4. Initiate screen shot path and video clip path
        self.video_clip(init = True)

        self.gm.pause_game()

        # 5. Initial augment - this creates the first augmented image for the first loop iteration
        self.augment()

        # ========== Phase 1: Route to LangGraph or Legacy ==========
        if self.use_langgraph and self.workflow_app is not None:
            logger.write("[Phase 1] Starting LangGraph workflow execution")
            return self._run_with_langgraph()
        else:
            logger.write("[Phase 0] Starting legacy sequential execution")
            return self._run_legacy()

    def _run_with_langgraph(self):
        """
        
        - 路由函数必须是纯函数（无副作用）
        - Checkpoint 记录保存
        - 测试运行
        - 详细日志
        """
        logger.write("=" * 80)
        logger.write("[LangGraph] Starting workflow execution")
        logger.write("=" * 80)
        
        # 初始化状态
        screenshot_path_list = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET)
        screenshot_path = screenshot_path_list[-1] if screenshot_path_list else ""
        initial_frame_id = 0  # Use a default frame ID or retrieve from game state
        
        initial_state = create_initial_state(
            frame_ids=(-1, initial_frame_id),
            screenshot_path=screenshot_path,
            work_dir=config.work_dir,
            env_name=config.env_short_name,
            env_config=config.env_config
        )
        
        # Ensure initial_state is a dictionary
        if not isinstance(initial_state, dict):
            initial_state = {}
        
        # 修复Bug：使用memory中已序列化的skill_library
        initial_state['skill_library'] = self.memory.get_latest("skill_library", "")
        
        # 修复Bug：清理initial_state中可能存在的Skill对象
        from cradle.runner.langgraph_nodes import sanitize_for_checkpoint
        initial_state = sanitize_for_checkpoint(initial_state)
        
        # Type assertion to ensure initial_state is a dict for type checking
        initial_state = initial_state if isinstance(initial_state, dict) else {}
        frame_ids_info = initial_state.get('frame_ids', 'N/A')
        logger.write(f"[LangGraph] Initial state: frame_ids={frame_ids_info}, step_count=0")
        
        thread_id = f"session_{int(time.time())}"
        workflow_config = {"configurable": {"thread_id": thread_id}}
        
        logger.write(f"[LangGraph] Thread ID: {thread_id}")
        
        step = 0
        max_steps = config.max_turn_count
        
        try:
            while step < max_steps:
                logger.write("=" * 80)
                logger.write(f"[LangGraph] Step {step + 1}/{max_steps}")
                logger.write("=" * 80)
                
                # 执行操作（这里是调用 LangGraph 工作流或双脑控制器）
                try:
                    if self.workflow_app is None:
                        logger.error("[LangGraph] Workflow app is None, aborting")
                        break

                    # Phase 3: 双脑调度路径
                    dual_brain_controller = getattr(self, 'dual_brain_controller', None)
                    if dual_brain_controller is not None:
                        result_state = dual_brain_controller.step(
                            initial_state, workflow_config
                        )
                    else:
                        result_state = self.workflow_app.invoke(
                            initial_state,
                            config=workflow_config  # type: ignore
                        )
                    
                    success = result_state.get('success', False)
                    error = result_state.get('error')
                    
                    logger.write(f"[LangGraph] Step {step + 1} completed:")
                    logger.write(f"  - Success: {success}")
                    logger.write(f"  - Error: {error or 'None'}")
                    logger.write(f"  - Task: {result_state.get('task', 'N/A')}")
                    logger.write(f"  - Actions executed: {len(result_state.get('planned_actions', []))}")
                    
                    # 如果有错误，尝试重试
                    if error:
                        retry_count = result_state.get('retry_count', 0)
                        if retry_count >= 3:
                            logger.error(f"[LangGraph] Max retries reached, stopping")
                            break
                    
                    # Checkpoint 保存
                    step += 1
                    if step % config.checkpoint_interval == 0:
                        checkpoint_path = os.path.join(self.checkpoint_path, f'langgraph_checkpoint_{step:06d}.json')
                        logger.write(f"[LangGraph] Saving checkpoint to {checkpoint_path}")
                        # Note: LangGraph checkpoint 目前未实现，这里只是预留
                    
                    # 更新初始化状态为下一步
                    # 合并策略：保留上一轮的所有字段，用新结果覆盖
                    # 这确保小脑未返回的字段（如 task, gathered_info）不会丢失
                    screenshot_path_list = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET)
                    current_screenshot = screenshot_path_list[-1] if screenshot_path_list else ""
                    executed_frames = result_state.get('executed_frames', result_state.get('frame_ids', (-1, -1)))

                    initial_state = {
                        **initial_state,    # 保留上轮所有字段
                        **result_state,     # 新结果覆盖
                        "frame_ids": executed_frames,
                        "screenshot_path": current_screenshot,
                        "step_count": step,
                        "is_first_step": False
                    }
                    
                    logger.write(f"[LangGraph] Prepared state for next iteration: frame_ids={initial_state.get('frame_ids', 'N/A')}")
                    
                except KeyboardInterrupt:
                    logger.write('[LangGraph] KeyboardInterrupt Ctrl+C detected, exiting.')
                    break
                    
                except Exception as e:
                    logger.error(f"[LangGraph] Workflow execution failed at step {step + 1}: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    
                    # 鍐冲畾鏄惁缁х画
                    if step < 3:  # 鍓?3 姝ュけ璐ワ紝fallback 鍒?legacy
                        logger.error("[LangGraph] Early failure, falling back to legacy runner")
                        return self._run_legacy()
                    else:  # 鍚庣画澶辫触锛屽皾璇曠户缁?
                        logger.warn("[LangGraph] Continuing despite error...")
                        step += 1
                        continue
            
            logger.write("=" * 80)
            logger.write(f"[LangGraph] Workflow completed: {step} steps executed")
            logger.write("=" * 80)
            
        except Exception as e:
            logger.error(f"[LangGraph] Fatal error: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        finally:
            self.pipeline_shutdown()

    def _run_legacy(self):
        """
        鍘熸湁鐨勫浐瀹氭祦绋嬫墽琛?(Phase 0)
        
        淇濈暀浣滀负 fallback锛岀‘淇濆悜鍚庡吋瀹广€?
        """
        logger.write("[Legacy] Starting sequential pipeline")
        
        success = False
        step = 0

        while not success:
            try:
                # Note: Screenshot from previous iteration (or initial screenshot) is already available
                # No need to capture here - we use the result from last action execution

                # 7.1. Information gathering
                self.run_information_gathering()

                # 7.2. Self reflection
                self.run_self_reflection()

                # 7.3. Task inference
                self.run_task_inference()

                # 7.4. Skill curation
                self.run_skill_curation()

                # 7.5. Action planning
                self.run_action_planning()

                step += 1

                if step % config.checkpoint_interval == 0:
                    checkpoint_path = os.path.join(self.checkpoint_path, 'checkpoint_{:06d}.json'.format(step))
                    self.memory.save(checkpoint_path)

                if step > config.max_turn_count:
                    logger.write('Max steps reached, exiting.')
                    break

            except KeyboardInterrupt:
                logger.write('KeyboardInterrupt Ctrl+C detected, exiting.')
                self.pipeline_shutdown()
                break

        self.pipeline_shutdown()


    def run_information_gathering(self):

        # 1. Prepare the parameters to call llm api
        self.information_gathering_preprocess()

        # 2. Call llm api for information gathering
        response = self.information_gathering()

        # 3. Postprocess the response
        self.information_gathering_postprocess(response)


    def run_self_reflection(self):

        # 1. Prepare the parameters to call llm api
        self.self_reflection_preprocess()

        # 2. Call llm api for self reflection
        response = self.self_reflection()

        # 3. Postprocess the response
        self.self_reflection_postprocess(response)


    def run_task_inference(self):

        # 1. Prepare the parameters to call llm api
        self.task_inference_preprocess()

        # 2. Call llm api for task inference
        response = self.task_inference()

        # 3. Postprocess the response
        self.task_inference_postprocess(response)


    def run_action_planning(self):

        # 1. Prepare the parameters to call llm api
        self.action_planning_preprocess()

        # 2. Call llm api for action planning
        response = self.action_planning()

        # 3. Postprocess the response
        self.action_planning_postprocess(response)

        # 4. Execute the actions
        self.skill_execute()

        # 5. Capture screenshot after action execution to record the direct result
        # This screenshot will be used in the next loop iteration
        screenshot_path = self.gm.capture_screen()
        self.memory.update_info_history({
            "screenshot_path": screenshot_path,
            constants.IMAGES_MEM_BUCKET: screenshot_path
        })

        # 6. Augment the screenshot immediately so it's ready for next iteration
        self.augment()


    def run_skill_curation(self):

        # 1. Call skill curation
        self.skill_curation()


def exit_cleanup(runner):
    logger.write("Exiting pipeline.")
    runner.pipeline_shutdown()


def entry(args):

    task_description = "No Task"

    task_id, subtask_id = 1, 0
    try:
        # Read end to end task description from config file
        task_list = kget(config.env_config, constants.TASK_DESCRIPTION_LIST, default=[])
        if task_list and len(task_list) >= task_id:
            task_description = task_list[task_id-1][constants.TASK_DESCRIPTION]
            if subtask_id > 0:
                subtask_list = task_list[task_id-1].get(constants.SUB_TASK_DESCRIPTION_LIST, [])
                if subtask_list and len(subtask_list) >= subtask_id:
                    task_description = subtask_list[subtask_id-1]
    except Exception:
        logger.warn(f"Task description is not found for task_id: {task_id} and/or subtask_id: {subtask_id}")
        logger.warn(f"Using default input value: {task_description}")

    pipelineRunner = PipelineRunner(llm_provider_config_path=args.llmProviderConfig,
                                    embed_provider_config_path=args.embedProviderConfig,
                                    task_description=task_description,
                                    use_self_reflection = True,
                                    use_task_inference = True)

    atexit.register(exit_cleanup, pipelineRunner)

    pipelineRunner.run()


