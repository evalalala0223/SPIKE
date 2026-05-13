import os
import atexit
import sys
import json
import time
import uuid
import re
from collections.abc import Iterable
from typing import Dict, Any, Optional, List, cast
from copy import deepcopy


from stardojo.utils.dict_utils import kget
from stardojo.utils.string_utils import replace_unsupported_chars
from stardojo.utils.cortex_runtime_utils import (
    LEGACY_COMPACT_PROMPT_SOURCE,
    CortexConfigurationError,
    build_runtime_local_recovery_action,
    build_sanitized_subtask_hints,
    record_cortex_planning_latency,
    record_cortex_no_execution,
    reset_cortex_no_execution_watchdog,
    is_redundant_tool_selection_subtask,
    resolve_cortex_executable_actions,
    resolve_workflow_subtask_values,
    resolve_little_brain_prompt_source,
    select_cortex_suggestion_for_logging,
    should_initialize_cortex_state,
    should_treat_cortex_attempt_as_first_step,
    validate_cultivation_pre_execution_action,
    validate_runtime_pre_execution_action,
)
from stardojo.utils.prompt_profile_utils import (
    apply_runtime_action_planning_template_overrides,
    build_task_specific_planner_params,
    sync_planner_prompt_templates,
)
from stardojo.utils.file_utils import assemble_project_path, read_resource_file
from stardojo.utils.task_bootstrap import (
    build_task_acquisition_context,
    build_initial_subtask,
    build_initial_subtask_reasoning,
)
from stardojo.utils.execution_feedback_utils import (
    detect_position_issue,
    execution_observation_confirms_change,
    execution_has_no_confirmation,
    infer_execution_success_raw,
    stable_snapshot_text,
)
from stardojo.utils.stardew_prompt_state import extract_stardew_prompt_fact_fields
from stardojo import constants
from stardojo.log import Logger
from stardojo.config import Config
from stardojo.memory import LocalMemory as LegacyLocalMemory
from stardojo.provider.llm.llm_factory import LLMFactory
from stardojo.environment.skill_registry_factory import SkillRegistryFactory
from stardojo.environment.ui_control_factory import UIControlFactory
from stardojo.gameio.io_env import IOEnvironment
from stardojo.gameio.game_manager import GameManager
from stardojo.planner.stardew_planner import StardewPlanner, TaskInference, SelfReflection
from agent.log_processor import process_log_messages
from env.stardew_env import *
import logging
from env.env_constants import *

from stardojo.provider import (
    StardewInformationGatheringPreprocessProvider,
    StardewInformationGatheringPostprocessProvider,
    StardewInformationGatheringProvider,
    StardewActionPlanningPreprocessProvider,
    StardewActionPlanningPostprocessProvider,
    StardewActionPlanningProvider,
    StardewSelfReflectionPostprocessProvider,
    StardewSelfReflectionProvider,
    StardewSelfReflectionPreprocessProvider,
    StardewTaskInferencePostprocessProvider,
    StardewTaskInferencePreprocessProvider,
    StardewTaskInferenceProvider,
    SkillExecuteProvider,
    SkillCurationProvider,
    AugmentProvider
)

CORTEX_IMPORT_ERROR = ""

try:
    agent_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if agent_root not in sys.path:
        sys.path.insert(0, agent_root)

    from cradle import constants as c_constants
    from cradle.memory import LocalMemory as CradleLocalMemory
    from cradle.memory.mem0_provider import Mem0Provider
    from cradle.runner.langgraph_workflow import build_game_workflow
    from cradle.runner.game_state import create_initial_state
    from cradle.runner.dual_brain import DualBrainController
    CORTEX_AVAILABLE = True
except Exception as e:
    c_constants = None
    CradleLocalMemory = None
    Mem0Provider = None
    build_game_workflow = None
    create_initial_state = None
    DualBrainController = None
    CORTEX_AVAILABLE = False
    CORTEX_IMPORT_ERROR = f"{type(e).__name__}: {e}"
    if isinstance(e, ModuleNotFoundError) and getattr(e, "name", "") == "langgraph":
        CORTEX_IMPORT_ERROR += " (missing dependency 'langgraph'; install with 'pip install -r requirements.txt')"

config = Config()
logger = Logger()
logger.work_dir = config.work_dir
logger._configure_root_logger()
io_env = IOEnvironment()

COMPOSITE_SKILL_LIBRARY_OVERRIDES: Dict[str, Dict[str, Any]] = {}

RESOURCE_DEPENDENT_COMPOSITE_SKILLS: set[str] = set()

_CULTIVATION_FAILURE_ROOT_CAUSES = {
    "invalid_target_tile",
    "wrong_facing_direction",
    "wrong_tile_alignment",
    "item_missing",
    "menu_stuck",
    "movement_blocked",
    "stale_subtask",
    "unknown",
}
_CULTIVATION_REQUIRED_CHANGE_TYPES = {
    "change_position",
    "change_facing",
    "change_target_tile",
    "change_selected_item",
    "close_menu",
    "switch_to_retrieval_subtask",
    "rebuild_subtask",
}


def _truncate_for_log(value: Any, max_len: int = 2000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = repr(value)
    if len(text) > max_len:
        return text[:max_len] + "...<truncated>"
    return text


def _get_cradle_memory() -> Any:
    if CradleLocalMemory is None:
        raise RuntimeError("CradleLocalMemory is not available")
    return CradleLocalMemory()


def _load_enhanced_config() -> Dict[str, Any]:
    try:
        import yaml
        from stardojo.utils.file_utils import assemble_project_path

        config_path = os.getenv("STARDOJO_ENHANCED_CONFIG", "").strip()
        if config_path:
            config_path = assemble_project_path(config_path)
        else:
            config_path = assemble_project_path("./conf/enhanced_config.yaml")
        if not os.path.exists(config_path):
            return {}
        with open(config_path, "r", encoding="utf-8") as config_file:
            data = yaml.safe_load(config_file) or {}
        return data if isinstance(data, dict) else {}
    except Exception as e:
        logging.warning(f"[Cortex] Failed to load enhanced config: {e}")
        return {}


def _create_runtime_memory() -> Any:
    """Keep runner-side state on legacy Stardew memory.

    Cortex/LangGraph keeps its own cradle memory via ``self.cortex_memory``.
    Using legacy memory here avoids split-brain state between the runner and
    the Stardew preprocess/providers that already depend on stardojo.memory.
    """
    return LegacyLocalMemory()


def _build_cortex_workflow(*args, **kwargs):
    if build_game_workflow is None:
        raise RuntimeError("build_game_workflow is not available")
    return build_game_workflow(*args, **kwargs)


class DecisionOnlySkillExecuteProvider:
    def __init__(self, memory: Optional[Any] = None):
        if memory is not None:
            self.memory = memory
        else:
            if CradleLocalMemory is None:
                raise RuntimeError("CradleLocalMemory is not available")
            self.memory = CradleLocalMemory()

    def __call__(self, *args, **kwargs):
        params = deepcopy(self.memory.working_area)
        skill_steps = params.get("skill_steps", [])
        if not isinstance(skill_steps, list):
            skill_steps = [skill_steps] if skill_steps else []
        last_skill = skill_steps[-1] if skill_steps else ""
        return {
            "start_frame_id": params.get("frame_ids", (0, 0))[0] if isinstance(params.get("frame_ids"), tuple) else 0,
            "end_frame_id": params.get("frame_ids", (0, 0))[1] if isinstance(params.get("frame_ids"), tuple) else 0,
            "screenshot_path": params.get("screenshot_path", ""),
            "pre_action": last_skill,
            "pre_screen_classification": params.get("screen_classification", ""),
            "exec_info": {
                "done": True,
                "executed_skills": skill_steps,
                "last_skill": last_skill,
                "errors": False,
                "errors_info": "",
            }
        }

class _NoOpProvider:
    """No-op provider: passes through the first positional arg unchanged.

    Used as both preprocess (called with no args, return value ignored) and
    postprocess (called with result as first arg, return value replaces result).
    """
    def __call__(self, *args, **kwargs):
        if args:
            return args[0]
        return None


class _CradleInfoGathering:
    """Info gathering: return gathered_info already set in cradle.memory."""
    def __init__(self, runner: Optional["PipelineRunner"] = None):
        self.runner = runner

    def __call__(self, *args, **kwargs):
        memory = self.runner._get_cortex_runtime_memory() if self.runner is not None else _get_cradle_memory()
        gathered = memory.working_area.get("gathered_info", {})
        if not gathered:
            gathered = {"description": "text_obs_mode"}
        return gathered


class _CradleStardewInfoGatheringProvider:
    use_stardew_original_input = True

    def __init__(self, runner: "PipelineRunner"):
        self.runner = runner

    def __call__(self, *args, **kwargs):
        return self.runner._run_cortex_information_gathering()


class _CradleStardewTaskInferenceProvider:
    def __init__(self, runner: "PipelineRunner"):
        self.runner = runner

    def __call__(self, *args, **kwargs):
        if hasattr(self.runner, "_prepare_big_brain_template_for_call"):
            self.runner._prepare_big_brain_template_for_call("task_inference")
        return self.runner._run_cortex_task_inference()


class _CradlePreprocess:
    """Ensure cradle.memory has required default fields before a provider runs."""
    def __init__(self, defaults: dict, runner: Optional["PipelineRunner"] = None):
        self.defaults = defaults
        self.runner = runner

    def __call__(self, *args, **kwargs):
        memory = self.runner._get_cortex_runtime_memory() if self.runner is not None else _get_cradle_memory()
        for key, val in self.defaults.items():
            if key not in memory.working_area or memory.working_area[key] is None:
                memory.working_area[key] = val


class _CradlePlannerProvider:
    """Call a planner method with data from cradle.memory."""
    def __init__(self, planner, method_name: str, runner: Optional["PipelineRunner"] = None):
        self.planner = planner
        self.method_name = method_name
        self.runner = runner

    def __call__(self, *args, **kwargs):
        if self.runner is not None:
            if hasattr(self.runner, "_prepare_big_brain_template_for_call"):
                self.runner._prepare_big_brain_template_for_call(self.method_name)
            elif hasattr(self.runner, "_ensure_big_brain_template_integrity"):
                self.runner._ensure_big_brain_template_integrity(self.method_name)
        memory = self.runner._get_cortex_runtime_memory() if self.runner is not None else _get_cradle_memory()
        params = deepcopy(memory.working_area)
        method = getattr(self.planner, self.method_name)
        data = method(input=params)
        response = data.get('res_dict') or {}
        return response


class _CradleActionPlanningPostprocess:
    """Normalize raw planner actions for dual-brain workflow."""

    _action_pattern = re.compile(r"^[A-Za-z_]\w*\s*\(.*\)$")

    def _normalize_actions(self, raw_actions: Any) -> List[str]:
        action_pattern = getattr(self, "_action_pattern", re.compile(r"^[A-Za-z_]\w*\s*\(.*\)$"))
        if isinstance(raw_actions, str):
            candidates = raw_actions.replace('```python', '').replace('```', '').splitlines()
        elif isinstance(raw_actions, (list, tuple)):
            candidates = []
            for item in raw_actions:
                if isinstance(item, str):
                    candidates.extend(item.splitlines())
                elif item is not None:
                    candidates.append(str(item))
        elif raw_actions is None:
            candidates = []
        else:
            candidates = [str(raw_actions)]

        normalized: List[str] = []
        for action in candidates:
            action_text = re.sub(r"^(?:[-*]\s*|\d+[\.)]\s*)", "", action.split('#', 1)[0].strip())
            if action_text and action_pattern.match(action_text):
                normalized.append(action_text)
        return normalized

    def __call__(self, response: Dict[str, Any]):
        processed_response = deepcopy(response) if isinstance(response, dict) else {}
        raw_actions = processed_response.get("skill_steps", processed_response.get("actions", []))
        skill_steps = self._normalize_actions(raw_actions)
        processed_response["actions"] = skill_steps
        processed_response["skill_steps"] = skill_steps
        if skill_steps:
            processed_response["action"] = "[" + ",".join(skill_steps) + "]" if len(skill_steps) > 1 else skill_steps[0]
        else:
            processed_response["action"] = ""
        return processed_response


class _CradleTaskInferencePostprocess:
    """Normalize task inference output for dual-brain workflow."""

    def __call__(self, response: Dict[str, Any]):
        processed_response = deepcopy(response) if isinstance(response, dict) else {}

        history_summary = processed_response.get('history_summary') or processed_response.get('summarization', '')
        subtask_description = processed_response.get('subtask') or processed_response.get('subtask_description', '')
        subtask_reasoning = processed_response.get('subtask_reasoning', '')

        processed_response.update({
            'summarization': history_summary,
            'history_summary': history_summary,
            'subtask_description': subtask_description,
            'subtask_reasoning': subtask_reasoning,
        })

        return processed_response


class PipelineRunner():

    @staticmethod
    def _resolve_runtime_config_path(path_value: str) -> str:
        raw_path = str(path_value or "").strip()
        if not raw_path:
            return raw_path

        repo_root = os.path.dirname(agent_root)
        candidate_paths: List[str] = []

        def _append_candidate(candidate: str) -> None:
            if candidate and candidate not in candidate_paths:
                candidate_paths.append(candidate)

        if os.path.isabs(raw_path):
            _append_candidate(raw_path)
        else:
            _append_candidate(os.path.abspath(raw_path))
            _append_candidate(os.path.abspath(os.path.join(repo_root, raw_path)))
            _append_candidate(os.path.abspath(os.path.join(agent_root, raw_path)))

            normalized = raw_path.replace("\\", "/")
            if normalized.startswith("agent/"):
                trimmed = normalized.split("/", 1)[1]
                _append_candidate(os.path.abspath(os.path.join(agent_root, trimmed)))
                _append_candidate(os.path.abspath(os.path.join(repo_root, trimmed)))

        for candidate in candidate_paths:
            if os.path.exists(candidate):
                return candidate

        return candidate_paths[0] if candidate_paths else raw_path

    def __init__(self,
                 llm_provider_config_path: str,
                 embed_provider_config_path: str,
                 task_description: str,
                 use_self_reflection: bool = False,
                 use_task_inference: bool = False,
                 envConfig = None,
                 max_turn_count = None,
                 log_dir_name = None):

        self.llm_provider_config_path = self._resolve_runtime_config_path(llm_provider_config_path)
        self.embed_provider_config_path = self._resolve_runtime_config_path(embed_provider_config_path)

        self.task_description = task_description
        self.use_self_reflection = use_self_reflection
        self.use_task_inference = use_task_inference
        self.initial_subtask_description = build_initial_subtask(task_description)
        self.initial_subtask_reasoning = build_initial_subtask_reasoning(task_description)

        if envConfig is not None:
            envConfig = self._resolve_runtime_config_path(envConfig)
            config.load_env_config(envConfig)
        if max_turn_count is not None:
            config.max_turn_count = max_turn_count
        config.set_work_dir_base(log_dir_name)
        config._set_dirs()
        config.set_logger_dirs()
        logger.work_dir = config.work_dir
        # Refresh the root logger context as soon as the task switches so early
        # init logs do not inherit the previous task label from a reused worker.
        logger._configure_root_logger(task=self.task_description)

        # Init internal params
        self.set_internal_params()

    def get_config(self):
        return config

    @staticmethod
    def _load_provider_model_metadata(
        llm_provider_config_path: str,
        embed_provider_config_path: str,
    ) -> Dict[str, str]:
        def _read_json(path: str) -> Dict[str, Any]:
            try:
                with open(path, "r", encoding="utf-8") as fd:
                    data = json.load(fd)
                return data if isinstance(data, dict) else {}
            except Exception:
                return {}

        llm_config = _read_json(llm_provider_config_path)
        embed_config = _read_json(embed_provider_config_path)
        return {
            "planner_comp_model": str(
                llm_config.get("comp_model")
                or llm_config.get("model")
                or llm_config.get("deployment_name")
                or ""
            ).strip(),
            "embedding_model": str(
                embed_config.get("emb_model")
                or llm_config.get("emb_model")
                or embed_config.get("model")
                or ""
            ).strip(),
        }

    def _sync_big_brain_templates(self, *, reason: str = "sync") -> None:
        if getattr(self, "planner", None) is None:
            return

        resolved_paths, template_audit, preserve_profile_templates = sync_planner_prompt_templates(
            self.planner,
            self.prompt_profile,
            template_keys=("action_planning", "task_inference"),
        )
        self._resolved_big_brain_template_paths = dict(resolved_paths)
        self._big_brain_template_audit = dict(template_audit)
        self._preserve_profile_big_brain_templates = bool(preserve_profile_templates)

        if reason:
            logging.info(
                "[Cortex] BigBrain template sync (%s): profile=%s action_planning=%s task_inference=%s",
                reason,
                self.prompt_profile,
                resolved_paths.get("action_planning", ""),
                resolved_paths.get("task_inference", ""),
            )

    def _ensure_big_brain_template_integrity(self, template_key: str) -> None:
        planner = getattr(self, "planner", None)
        if planner is None or template_key not in {"action_planning", "task_inference"}:
            return

        expected_path = str(self._resolved_big_brain_template_paths.get(template_key, "") or "").strip()
        if not expected_path:
            self._sync_big_brain_templates(reason=f"bootstrap:{template_key}")
            expected_path = str(self._resolved_big_brain_template_paths.get(template_key, "") or "").strip()
            if not expected_path:
                return

        expected_text = read_resource_file(assemble_project_path(expected_path))
        if template_key == "action_planning":
            expected_text = apply_runtime_action_planning_template_overrides(expected_text)
        current_text = str(getattr(planner, "templates", {}).get(template_key, "") or "")
        target_obj = None
        if template_key == "action_planning":
            target_obj = getattr(planner, "action_planning_", None)
        elif template_key == "task_inference":
            target_obj = getattr(planner, "task_inference_", None)
        current_runtime_text = str(getattr(target_obj, "template", "") or "")

        if current_text == expected_text and current_runtime_text == expected_text:
            return

        logging.warning(
            "[Cortex] Detected stale %s template for profile=%s; repairing to %s",
            template_key,
            self.prompt_profile,
            expected_path,
        )
        self._sync_big_brain_templates(reason=f"integrity_repair:{template_key}")

    def _prepare_big_brain_template_for_call(self, template_key: str) -> None:
        if template_key not in {"action_planning", "task_inference"}:
            return
        # Refresh the current task's BigBrain templates at the point of use.
        # This avoids prompt-profile leakage when a worker has just switched tasks
        # but a downstream provider still holds stale template text in memory.
        self._sync_big_brain_templates(reason="")
        self._ensure_big_brain_template_integrity(template_key)

    def _get_cortex_runtime_memory(self) -> Any:
        if self.cortex_memory is not None:
            return self.cortex_memory
        return _get_cradle_memory()

    def reconfigure_root_logger(self, port=None, task=None):
        try:
            from cradle.log import Logger as CradleLogger

            cradle_logger = CradleLogger()
            cradle_logger.work_dir = config.work_dir
            CradleLogger.work_dir = config.work_dir
            cradle_logger._configure_root_logger(work_dir=config.work_dir, port=port, task=task)
        except Exception as e:
            logging.warning(f"[Cortex] Failed to sync cradle logger context: {e}")

        logger.work_dir = config.work_dir
        logger._configure_root_logger(port=port, task=task)
        return logger

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

        ucf = UIControlFactory()
        ucf.register_builder(config.env_short_name, config.ui_control_name)
        self.env_ui_control = ucf.create(config.env_short_name)

        # Init game manager
        self.gm = GameManager(env_name=config.env_name,
                              embedding_provider=self.embed_provider,
                              llm_provider=self.llm_provider,
                              skill_registry=self.skill_registry,
                              ui_control=self.env_ui_control,
                              )

        self.memory = _create_runtime_memory()
        task_scope = f"{config.env_short_name}:{self.task_description}"
        if hasattr(self.memory, "reset_runtime_state"):
            self.memory.reset_runtime_state(task_scope=task_scope, work_dir=config.work_dir)
        self.legacy_memory = LegacyLocalMemory()
        if self.legacy_memory is not self.memory and hasattr(self.legacy_memory, "reset_runtime_state"):
            self.legacy_memory.reset_runtime_state(task_scope=task_scope, work_dir=config.work_dir)
        self._skill_library_json = ""
        self._enhanced_cfg = _load_enhanced_config()
        features_cfg = self._enhanced_cfg.get("features", {}) or {}
        dual_brain_cfg = self._enhanced_cfg.get("dual_brain", {}) or {}
        self._dual_brain_enabled = bool(features_cfg.get("use_dual_brain", False)) and bool(
            dual_brain_cfg.get("enabled", False)
        )
        self.task_acquisition_context = build_task_acquisition_context(self.task_description)

        # Init planner
        planner_params, self.prompt_profile = build_task_specific_planner_params(
            config.planner_params,
            self.task_description,
        )
        self._planner_template_audit = dict(planner_params.pop("_prompt_template_audit", {}) or {})
        self._resolved_prompt_template_paths = dict(
            ((planner_params.get("prompt_paths", {}) or {}).get("templates", {}) or {})
        )
        self._resolved_big_brain_template_paths: Dict[str, str] = {}
        self._big_brain_template_audit: Dict[str, Dict[str, Any]] = {}
        self._preserve_profile_big_brain_templates = False
        self._provider_model_metadata = self._load_provider_model_metadata(
            self.llm_provider_config_path,
            self.embed_provider_config_path,
        )
        self.agent_run_dir_name = os.path.basename(os.path.normpath(config.work_dir))
        self.planner = StardewPlanner(llm_provider=self.llm_provider,
                                    planner_params=planner_params,
                                    frame_extractor=None,
                                    icon_replacer=None,
                                    object_detector=None,
                                   use_self_reflection=self.use_self_reflection,
                                   use_task_inference=self.use_task_inference)
        logger.write(
            f"[PromptProfile] Using '{self.prompt_profile}' templates for task '{self.task_description}'"
        )

        # Init skill library
        self._refresh_skill_library()

        self._update_memory_histories({"skill_library": self.skill_library})

        self.provider_configs = config.provider_configs

        # Init checkpoint path
        self.checkpoint_path = os.path.join(config.work_dir, 'checkpoints')
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.augment = AugmentProvider()
        self.augment_methods = [
            self.augment
        ]

        self.self_reflection_preprocess = StardewSelfReflectionPreprocessProvider(gm=self.gm,
                                                                                  augment_methods=self.augment_methods)
        self.self_reflection = StardewSelfReflectionProvider(planner=self.planner, gm=self.gm)
        self.self_reflection_postprocess = StardewSelfReflectionPostprocessProvider(gm=self.gm)

        info_gathering_postprocess_config = getattr(self.provider_configs, 'information_gathering_postprocess_provider', {})
        self.information_gathering_preprocess = StardewInformationGatheringPreprocessProvider(gm=self.gm)
        self.information_gathering = StardewInformationGatheringProvider(planner=self.planner, gm=self.gm)
        self.information_gathering_postprocess = StardewInformationGatheringPostprocessProvider(
            gm=self.gm,
            **info_gathering_postprocess_config,
        )

        self.task_inference_preprocess = StardewTaskInferencePreprocessProvider(gm=self.gm)
        self.task_inference = StardewTaskInferenceProvider(planner=self.planner, gm=self.gm)
        self.task_inference_postprocess = StardewTaskInferencePostprocessProvider(gm=self.gm)

        init_params = {
            "task_description": self.task_description,
            "skill_library": self.skill_library,
            "exec_info": {
                "errors": False,
                "errors_info": ""
            },
            "action": "",
            "action_planning_reasoning": "",
            "self_reflection_reasoning": "",
            "summarization": "",
            # "toolbar_information": None,
            "subtask_description": self.initial_subtask_description,
            "subtask_reasoning": self.initial_subtask_reasoning,
            **self.task_acquisition_context,
        }

        self._update_memory_histories(init_params)

        self.cortex_memory = None
        self.dual_brain_controller = None
        self.workflow_app = None
        self._cortex_state = None
        self._latest_obs: Dict[str, Any] = {}
        self._cortex_workflow_config = {"configurable": {"thread_id": f"stardojo_{uuid.uuid4().hex[:8]}"}}
        self._decision_only_skill_execute = None
        self._skill_library_json = self._skill_library_json or ""
        self._little_brain_prompt_source = LEGACY_COMPACT_PROMPT_SOURCE
        self._cortex_init_failure_reason = ""
        self._cortex_fallback_warned = False
        self._runtime_stop_signal: Optional[Dict[str, Any]] = None
        if self._dual_brain_enabled:
            self._init_cortex_internal()
        else:
            self._cortex_init_failure_reason = "dual brain disabled by config"
            logger.write("[Cortex] Dual brain disabled by config; using prompt-profile planner path")

    @staticmethod
    def _shop_item_icon_name(item_name: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(item_name or "").strip().lower()).strip("_")
        return f"{normalized}_icon" if normalized else ""

    @staticmethod
    def _extract_buy_item_name(action_expr: str) -> str:
        action = str(action_expr or "")
        named_match = re.search(r"item_name\s*=\s*['\"]([^'\"]+)['\"]", action)
        if named_match:
            return named_match.group(1).strip()

        positional_match = re.search(r"buy_item\(\s*['\"]([^'\"]+)['\"]", action)
        if positional_match:
            return positional_match.group(1).strip()
        return ""

    def _composite_skill_requirement_paths(self, skill_name: str, action_expr: str = "") -> List[str]:
        env_sub_path = config.env_sub_path
        character_icons = [
            f"./res/{env_sub_path}/icons/up.jpg",
            f"./res/{env_sub_path}/icons/down.jpg",
            f"./res/{env_sub_path}/icons/left.jpg",
            f"./res/{env_sub_path}/icons/right.jpg",
        ]

        if skill_name == "go_home":
            return [f"./res/{env_sub_path}/icons/house_door.jpg", *character_icons]

        if skill_name == "go_to_store":
            return [f"./res/{env_sub_path}/icons/store_door.jpg", *character_icons]

        if skill_name == "go_through_door":
            return [
                f"./res/{env_sub_path}/icons/home_entrance.jpg",
                f"./res/{env_sub_path}/icons/home_exit.jpg",
                f"./res/{env_sub_path}/icons/store_entrance.jpg",
                f"./res/{env_sub_path}/icons/store_exit.jpg",
                *character_icons,
            ]

        if skill_name == "buy_item":
            item_name = self._extract_buy_item_name(action_expr)
            if item_name:
                item_icon = self._shop_item_icon_name(item_name)
                if item_icon:
                    return [f"./res/{env_sub_path}/icons/shop_interface/{item_icon}.png"]
            return [f"./res/{env_sub_path}/icons/shop_interface"]

        return []

    def _composite_skill_is_available(self, skill_name: str, action_expr: str = "") -> bool:
        if skill_name not in RESOURCE_DEPENDENT_COMPOSITE_SKILLS:
            return True

        from stardojo.utils.file_utils import assemble_project_path

        for requirement in self._composite_skill_requirement_paths(skill_name, action_expr):
            resolved = assemble_project_path(requirement)
            if skill_name == "buy_item" and requirement.endswith("/shop_interface"):
                if not os.path.isdir(resolved):
                    return False
                try:
                    has_png = any(
                        entry.is_file() and entry.name.lower().endswith(".png")
                        for entry in os.scandir(resolved)
                    )
                except OSError:
                    return False
                if not has_png:
                    return False
                continue

            if not os.path.exists(resolved):
                return False

        return True

    def _current_skill_library_names(self) -> set[str]:
        return {
            str(item.get("function_expression", "")).split("(", 1)[0].strip()
            for item in (self.skill_library or [])
            if isinstance(item, dict) and item.get("function_expression")
        }

    @staticmethod
    def _normalize_skill_library_description(description: Any) -> str:
        text = str(description or "")
        if not text:
            return ""

        replacements = (
            (
                r"\bcall use\((up|right|down|left)\)",
                lambda m: f'call use(direction="{m.group(1).lower()}")',
            ),
            (
                r"\bcall interact\((up|right|down|left)\)",
                lambda m: f'call interact(direction="{m.group(1).lower()}")',
            ),
            (
                r"\bcall choose_item\((\-?\d+)\)",
                lambda m: f"call choose_item(slot_index={int(m.group(1))})",
            ),
            (
                r"\bcall attach_item\((\-?\d+)\)",
                lambda m: f"call attach_item(slot_index={int(m.group(1))})",
            ),
        )

        normalized = text
        for pattern, repl in replacements:
            normalized = re.sub(pattern, repl, normalized, flags=re.IGNORECASE)
        return normalized

    def _normalize_skill_library_entry(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return item

        normalized = dict(item)
        normalized["description"] = self._normalize_skill_library_description(
            normalized.get("description", "")
        )
        return normalized

    def _refresh_skill_library(self) -> None:
        skills = self.gm.retrieve_skills(
            query_task=self.task_description,
            skill_num=config.skill_configs[constants.SKILL_CONFIG_MAX_COUNT],
            screen_type=constants.GENERAL_GAME_INTERFACE,
        )

        skill_library = []
        filtered_out = []
        for item in self.gm.get_skill_information(skills, config.skill_library_with_code):
            item = self._normalize_skill_library_entry(item)
            if not isinstance(item, dict):
                skill_library.append(item)
                continue

            skill_name = str(item.get("function_expression", "")).split("(", 1)[0].strip()
            if (
                skill_name in COMPOSITE_SKILL_LIBRARY_OVERRIDES
                and not self._composite_skill_is_available(skill_name)
            ):
                filtered_out.append(skill_name)
                continue
            skill_library.append(item)

        if filtered_out:
            logger.write(
                "[SkillLibrary] Withheld unavailable composite skills: "
                + ", ".join(sorted(set(filtered_out)))
            )

        existing_names = {
            str(item.get("function_expression", "")).split("(", 1)[0]
            for item in skill_library
            if isinstance(item, dict)
        }

        self.skill_library = skill_library
        self._skill_library_json = (
            json.dumps(self.skill_library, ensure_ascii=False)
            if self.skill_library
            else ""
        )

    def _init_cortex_internal(self):
        if not CORTEX_AVAILABLE:
            self._cortex_init_failure_reason = CORTEX_IMPORT_ERROR or "cortex imports unavailable"
            logging.warning(
                f"Cortex modules are not available, fallback to legacy planning: {self._cortex_init_failure_reason}"
            )
            return

        try:
            # CRITICAL: Force-reset the cradle singleton memory BEFORE any
            # LangGraph component reads from it.  CradleLocalMemory uses
            # Singleton metaclass, so _get_cradle_memory() returns the SAME
            # instance across task switches within the same worker process.
            # Without this early reset, the new task's task_inference will
            # read stale subtask/history from the previous task (e.g.
            # go_to_coop reads "clear debris" from clear_10_weeds).
            try:
                task_scope = f"{config.env_short_name}:{self.task_description}"
                _singleton_mem = _get_cradle_memory()
                if hasattr(_singleton_mem, "reset_runtime_state"):
                    _singleton_mem.reset_runtime_state(
                        task_scope=task_scope,
                        work_dir=config.work_dir,
                    )
                    logging.info("[Cortex] Force-reset cradle singleton memory before cortex init")
                # Also clear working_area explicitly in case reset_runtime_state
                # doesn't cover all mutable state.
                if hasattr(_singleton_mem, "working_area"):
                    _singleton_mem.working_area = {}
            except Exception as e:
                logging.warning(f"[Cortex] Failed to force-reset cradle singleton memory: {e}")
                _singleton_mem = _get_cradle_memory()

            if CradleLocalMemory is not None and isinstance(self.memory, CradleLocalMemory):
                self.cortex_memory = self.memory
            else:
                self.cortex_memory = _singleton_mem
            self._decision_only_skill_execute = DecisionOnlySkillExecuteProvider(
                memory=self._get_cortex_runtime_memory(),
            )

            # Ensure planner callables are initialized even when
            # use_task_inference / use_self_reflection were False at construction.
            if self.planner.task_inference_ is None:
                self.planner.task_inference_ = TaskInference(
                    input_map=self.planner.inputs["task_inference"],
                    template=self.planner.templates["task_inference"],
                    llm_provider=self.planner.llm_provider,
                )
            if self.planner.self_reflection_ is None:
                self.planner.self_reflection_ = SelfReflection(
                    input_map=self.planner.inputs["self_reflection"],
                    template=self.planner.templates["self_reflection"],
                    llm_provider=self.planner.llm_provider,
                )

            noop = _NoOpProvider()
            action_postprocess = _CradleActionPlanningPostprocess()
            action_postprocess._action_pattern = getattr(action_postprocess, '_action_pattern', re.compile(r"^[A-Za-z_]\w*\s*\(.*\)$"))
            stardew_info_provider = _CradleStardewInfoGatheringProvider(self)
            stardew_task_provider = _CradleStardewTaskInferenceProvider(self)

            sr_defaults = {
                "image_introduction": [],
                "pre_action": "",
                "pre_decision_making_reasoning": "",
                "pre_energy": None,
                "pre_money": None,
                "pre_health": None,
                "exec_info": {},
                "history_summary": "",
                "subtask_description": "",
                "subtask_reasoning": "",
                "previous_toolbar_information": [],
            }

            ap_defaults = {
                "image_introduction": [],
            }

            ti_defaults = {
                "image_introduction": [],
                "subtask_description": "",
                "subtask_reasoning": "",
            }

            langgraph_providers = {
                'video_clip': stardew_info_provider,
                'information_gathering_preprocess': noop,
                'information_gathering_postprocess': noop,
                'self_reflection': _CradlePlannerProvider(self.planner, 'self_reflection', runner=self),
                'self_reflection_preprocess': _CradlePreprocess(sr_defaults, runner=self),
                'self_reflection_postprocess': noop,
                'task_inference': stardew_task_provider,
                'task_inference_preprocess': noop,
                'task_inference_postprocess': noop,
                'action_planning': _CradlePlannerProvider(self.planner, 'action_planning', runner=self),
                'action_planning_preprocess': _CradlePreprocess(ap_defaults, runner=self),
                'action_planning_postprocess': action_postprocess,
                'skill_execute': self._decision_only_skill_execute,
            }

            self.workflow_app = _build_cortex_workflow(
                providers=langgraph_providers,
                enable_checkpoint=True,
                gm=self.gm,
                augment_provider=self.augment,
                parallel_mode=False,
                runtime_memory=self._get_cortex_runtime_memory(),
            )

            if self.workflow_app is not None and DualBrainController is not None:
                enhanced_cfg = dict(getattr(self, "_enhanced_cfg", {}) or {})
                dual_brain_cfg = enhanced_cfg.get('dual_brain', {}) or {}
                little_brain_cfg = dual_brain_cfg.get('little_brain', {}) or {}
                mem0_provider = None

                try:
                    features_cfg = enhanced_cfg.get('features', {}) or {}
                    mem0_cfg = enhanced_cfg.get('mem0', {}) or {}
                    mem0_enabled = bool(features_cfg.get('use_mem0', False)) and bool(mem0_cfg.get('enabled', False))

                    if mem0_enabled and Mem0Provider is not None:
                        embedding_provider = self.embed_provider
                        namespace = getattr(self.gm, 'env_name', None) if self.gm is not None else None
                        mem0_provider = Mem0Provider(
                            enabled=True,
                            embedding_provider=embedding_provider,
                            namespace=namespace,
                            storage_path=mem0_cfg.get('storage_path'),
                            quick_path_threshold=float(mem0_cfg.get('quick_path_threshold', 0.85)),
                            max_results=int(mem0_cfg.get('max_results', 3)),
                            require_meaningful_progress=bool(mem0_cfg.get('store_require_meaningful_progress', True)),
                            progress_min_chars=int(mem0_cfg.get('store_progress_min_chars', 8)),
                        )
                        logging.info("[Cortex] LittleBrain Mem0 reference enabled")
                    elif mem0_enabled:
                        logging.warning("[Cortex] Mem0 is enabled in config but Mem0Provider is unavailable")
                except Exception as e:
                    logging.warning(f"[Cortex] Failed to initialize LittleBrain Mem0 provider: {e}")

                self.dual_brain_controller = DualBrainController.from_config(
                    workflow_app=self.workflow_app,
                    gm=self.gm,
                    skill_execute_provider=self._decision_only_skill_execute,
                    augment_provider=self.augment,
                    embed_provider=self.embed_provider,
                    mem0_provider=mem0_provider,
                )

                self._sync_big_brain_templates(reason="init")

                if self._preserve_profile_big_brain_templates:
                    logging.info(
                        "[Cortex] preserving profile-specific BigBrain templates for prompt_profile=%s",
                        self.prompt_profile,
                    )
                else:
                    logging.info(
                        "[Cortex] swapping BigBrain templates to cortex for prompt_profile=%s",
                        self.prompt_profile,
                    )

                # Disable LittleBrain internal skill execution to avoid double execution
                # In stardojo integration, env layer handles execution via execute_actions()
                if hasattr(self.dual_brain_controller, 'little_brain'):
                    self.dual_brain_controller.little_brain.execute_internally = False

                use_stardew_template = False
                if little_brain_cfg:
                    use_stardew_template = bool(little_brain_cfg.get('use_stardew_template', False))

                lb_template_path = assemble_project_path(
                    "./res/stardew/prompts/templates/action_planning_littlebrain.prompt"
                )
                self._little_brain_prompt_source = resolve_little_brain_prompt_source(
                    use_stardew_template=use_stardew_template,
                    little_brain_available=hasattr(self.dual_brain_controller, 'little_brain'),
                    template_path=lb_template_path,
                )
                if self._little_brain_prompt_source != LEGACY_COMPACT_PROMPT_SOURCE:
                    lb_template = read_resource_file(self._little_brain_prompt_source)
                    self.dual_brain_controller.little_brain.vllm_client.template = lb_template
                logging.info("[Cortex] LittleBrain prompt source: %s", self._little_brain_prompt_source)

                logging.info("Cortex dual-brain controller initialized")

        except Exception as e:
            self.dual_brain_controller = None
            self.workflow_app = None
            self._cortex_state = None
            self._cortex_init_failure_reason = f"{type(e).__name__}: {e}"
            logging.error(f"Failed to initialize cortex internal pipeline: {self._cortex_init_failure_reason}")
            if isinstance(e, CortexConfigurationError):
                raise

    @staticmethod
    def _extract_item_name(item: Any) -> str:
        if isinstance(item, dict):
            for key in ("name", "Name", "item_name", "ItemName", "currentitem", "current_item", "item"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        if isinstance(item, str):
            return item.strip()
        return ""

    @staticmethod
    def _extract_item_index(item: Any) -> Optional[int]:
        if not isinstance(item, dict):
            return None

        for key in ("index", "slot_index", "selected_position", "slot"):
            value = item.get(key)
            if isinstance(value, int):
                return value
            if isinstance(value, str) and value.strip().isdigit():
                return int(value.strip())
        return None

    @staticmethod
    def _normalize_position(position: Any) -> Optional[tuple]:
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            x, y = position[0], position[1]
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return int(x), int(y)
        if isinstance(position, dict):
            x = position.get("x", position.get("X"))
            y = position.get("y", position.get("Y"))
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                return int(x), int(y)
        if isinstance(position, str):
            numeric_parts = re.findall(r"-?\d+(?:\.\d+)?", position)
            if len(numeric_parts) >= 2:
                try:
                    return int(float(numeric_parts[0])), int(float(numeric_parts[1]))
                except (TypeError, ValueError):
                    return None
        return None

    @classmethod
    def _flatten_surrounding_value(cls, value: Any) -> List[str]:
        if value in (None, "", []):
            return []

        if isinstance(value, dict):
            for key in ("seed_name", "name", "Name", "item_name", "id"):
                candidate = value.get(key)
                if candidate not in (None, "", []):
                    return [str(candidate).strip()]
            return [str(value).strip()]

        if isinstance(value, (list, tuple, set)):
            parts: List[str] = []
            for item in value:
                parts.extend(cls._flatten_surrounding_value(item))
            return parts

        return [str(value).strip()]

    @classmethod
    def _canonicalize_surroundings_text(cls, text: Any) -> str:
        stripped = str(text or "").strip()
        if not stripped:
            return ""

        normalized_lines: List[str] = []
        matched_any = False
        for raw_line in stripped.splitlines():
            line = raw_line.strip()
            if not line:
                continue

            match = re.match(
                r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\](?:\([^)]*\))?\s*:\s*(.*)$",
                line,
            )
            if match:
                rel_x = int(match.group(1))
                rel_y = int(match.group(2))
                label = match.group(3).strip() or "empty"
                normalized_lines.append(f"[{rel_x}, {rel_y}]: {label}")
                matched_any = True
            else:
                normalized_lines.append(line)

        if matched_any:
            return "\n".join(normalized_lines)
        return stripped

    @staticmethod
    def _format_list_field(data: Any) -> str:
        """Format a list/dict obs field into a readable string for the prompt."""
        if not data:
            return ""
        if isinstance(data, str):
            return data
        if isinstance(data, list):
            if len(data) == 0:
                return ""
            parts = []
            for item in data:
                if isinstance(item, dict):
                    name = item.get("name") or item.get("Name") or item.get("type") or str(item)
                    pos = item.get("position") or item.get("Position") or item.get("doorPosition")
                    if pos and isinstance(pos, dict):
                        parts.append(f"{name} at ({pos.get('X', '?')}, {pos.get('Y', '?')})")
                    elif pos:
                        parts.append(f"{name} at {pos}")
                    else:
                        parts.append(str(name))
                else:
                    parts.append(str(item))
            return "\n".join(parts)
        return str(data)

    @staticmethod
    def _format_buildings(buildings: Any, player_position: Any = None) -> str:
        """Format building data with door positions and relative directions."""
        if not buildings or not isinstance(buildings, list):
            return ""
        normalized_player = PipelineRunner._normalize_position(player_position)
        player_x, player_y = normalized_player if normalized_player is not None else (None, None)

        if (player_x is None or player_y is None) and player_position not in (None, "", []):
            logging.debug("_format_buildings: could not parse player coords from %r", player_position)

        parts = []
        for item in buildings:
            if not isinstance(item, dict):
                parts.append(str(item))
                continue
            name = item.get("name") or item.get("Name") or "building"
            door = item.get("doorPosition") or item.get("position") or item.get("Position")
            normalized_door = PipelineRunner._normalize_position(door)
            if normalized_door is not None and player_x is not None and player_y is not None:
                try:
                    door_x, door_y = normalized_door
                    dx = int(door_x) - int(player_x)
                    dy = int(door_y) - int(player_y)
                    dir_parts = []
                    if dy < 0:
                        dir_parts.append(f"{abs(dy)} tiles up")
                    elif dy > 0:
                        dir_parts.append(f"{abs(dy)} tiles down")
                    if dx > 0:
                        dir_parts.append(f"{abs(dx)} tiles right")
                    elif dx < 0:
                        dir_parts.append(f"{abs(dx)} tiles left")
                    direction = ", ".join(dir_parts) if dir_parts else "at player"
                    parts.append(f"{name} (door: {direction}, relative offset: x={dx}, y={dy})")
                except (TypeError, ValueError):
                    parts.append(f"{name} (door at {door})")
            elif normalized_door is not None:
                door_x, door_y = normalized_door
                parts.append(f"{name} (door at ({door_x}, {door_y}))")
            else:
                parts.append(str(name))
        return "\n".join(parts)

    @staticmethod
    def _format_exits(exits: Any, player_position: Any = None) -> str:
        """Format exit data with relative directions from player position."""
        if not exits or not isinstance(exits, list):
            return ""
        normalized_player = PipelineRunner._normalize_position(player_position)
        player_x, player_y = normalized_player if normalized_player is not None else (None, None)

        if (player_x is None or player_y is None) and player_position not in (None, "", []):
            logging.debug("_format_exits: could not parse player coords from %r", player_position)

        parts = []
        for ex in exits:
            name = ex.get("target") or ex.get("name") or ex.get("Name") or "exit"
            pos = ex.get("position") or ex.get("Position") or {}
            normalized_exit = PipelineRunner._normalize_position(pos)
            if player_x is not None and player_y is not None and normalized_exit is not None:
                try:
                    ex_x, ex_y = normalized_exit
                    dx = int(ex_x) - int(player_x)
                    dy = int(ex_y) - int(player_y)
                    direction_parts = []
                    if dy < 0:
                        direction_parts.append(f"{abs(dy)} tiles up")
                    elif dy > 0:
                        direction_parts.append(f"{abs(dy)} tiles down")
                    if dx > 0:
                        direction_parts.append(f"{abs(dx)} tiles right")
                    elif dx < 0:
                        direction_parts.append(f"{abs(dx)} tiles left")
                    direction = ", ".join(direction_parts) if direction_parts else "at player position"
                    parts.append(f"{name} ({direction}, relative offset: x={dx}, y={dy})")
                except (TypeError, ValueError):
                    parts.append(f"{name} at ({ex_x}, {ex_y})")
            elif normalized_exit is not None:
                ex_x, ex_y = normalized_exit
                parts.append(f"{name} at ({ex_x}, {ex_y})")
            else:
                parts.append(str(name))
        return "\n".join(parts)

    @classmethod
    def _format_surroundings_text(
        cls,
        surroundings: Any,
        player_position: Any = None,
    ) -> str:
        if isinstance(surroundings, str):
            return cls._canonicalize_surroundings_text(surroundings)
        if not isinstance(surroundings, list):
            return cls._canonicalize_surroundings_text(surroundings)

        normalized_player = cls._normalize_position(player_position)
        formatted_lines: List[tuple[tuple[int, int, int], str]] = []

        for tile in surroundings:
            if not isinstance(tile, dict):
                continue

            tile_position = cls._normalize_position(tile.get("position"))
            if tile_position is None:
                continue

            rel_x, rel_y = tile_position
            if normalized_player is not None:
                rel_x -= normalized_player[0]
                rel_y -= normalized_player[1]

            descriptors: List[str] = []
            for key in (
                "debris_at_tile",
                "object_at_tile",
                "terrain_at_tile",
                "furniture_at_tile",
                "building_info",
                "exit_info",
                "npc_info",
                "crop_at_tile",
                "tile_properties",
            ):
                descriptors.extend(cls._flatten_surrounding_value(tile.get(key)))

            deduped_descriptors: List[str] = []
            seen = set()
            for descriptor in descriptors:
                normalized_descriptor = str(descriptor).strip()
                if not normalized_descriptor:
                    continue
                lowered = normalized_descriptor.lower()
                if lowered in seen:
                    continue
                seen.add(lowered)
                deduped_descriptors.append(normalized_descriptor)

            summary = ", ".join(deduped_descriptors) if deduped_descriptors else "empty"
            sort_key = (abs(rel_x) + abs(rel_y), rel_y, rel_x)
            formatted_lines.append((sort_key, f"[{rel_x}, {rel_y}]: {summary}"))

        formatted_lines.sort(key=lambda item: item[0])
        return "\n".join(line for _, line in formatted_lines)

    @classmethod
    def _infer_facing_position(cls, position: Any, facing_direction: Any) -> Any:
        normalized_position = cls._normalize_position(position)
        if normalized_position is None or not isinstance(facing_direction, str):
            return ""

        x, y = normalized_position
        direction = facing_direction.strip().lower()
        if direction == "up":
            return [x, y - 1]
        if direction == "right":
            return [x + 1, y]
        if direction == "down":
            return [x, y + 1]
        if direction == "left":
            return [x - 1, y]
        return ""

    @classmethod
    def _infer_selected_position(cls, inv: list, chosen_item: Any) -> Optional[int]:
        chosen_index = cls._extract_item_index(chosen_item)
        if chosen_index is not None:
            return chosen_index

        chosen_name = cls._extract_item_name(chosen_item)
        if not chosen_name or not isinstance(inv, list):
            return None

        chosen_name_lower = chosen_name.lower()
        for idx, item in enumerate(inv):
            item_text = str(item).lower()
            if f": {chosen_name_lower} " in item_text or f": {chosen_name_lower}(" in item_text or item_text.endswith(f": {chosen_name_lower}"):
                return idx
        return None

    @classmethod
    def _format_toolbar_text(cls, inv: list, chosen_item: Any = None) -> str:
        if inv:
            toolbar_text = "Items in toolbar:\n" + "\n".join(str(item) for item in inv)
            chosen_name = cls._extract_item_name(chosen_item)
            chosen_index = cls._extract_item_index(chosen_item)
            if chosen_name:
                if chosen_index is not None:
                    toolbar_text += f"\nCurrently selected item: slot_index {chosen_index}: {chosen_name}"
                else:
                    toolbar_text += f"\nCurrently selected item: {chosen_name}"
            return toolbar_text
        return "Empty toolbar"

    @staticmethod
    def _is_redundant_tool_selection_subtask(text: str, selected_item_name: str) -> bool:
        return is_redundant_tool_selection_subtask(text, selected_item_name)

    @classmethod
    def _sanitize_subtask_memory(
        cls,
        subtask_description: Any,
        subtask_reasoning: Any,
        chosen_item: Any,
    ) -> Dict[str, Any]:
        selected_item_name = cls._extract_item_name(chosen_item)
        hints = build_sanitized_subtask_hints(
            subtask_description=subtask_description,
            subtask_reasoning=subtask_reasoning,
            selected_item_name=selected_item_name,
        )
        if hints.get("redundant_tool_selection", False):
            logger.write(
                f"[Cortex] Detected redundant tool-selection subtask because {selected_item_name} is already selected."
            )
        return hints

    def _resolve_current_subtask_values(self) -> Dict[str, str]:
        return resolve_workflow_subtask_values(
            state=self._cortex_state if isinstance(self._cortex_state, dict) else None,
            initial_subtask_description=self.initial_subtask_description,
            initial_subtask_reasoning=self.initial_subtask_reasoning,
        )

    def consume_runtime_stop_signal(self) -> Optional[Dict[str, Any]]:
        signal = self._runtime_stop_signal
        self._runtime_stop_signal = None
        if isinstance(signal, dict):
            return deepcopy(signal)
        return None

    def get_runtime_task_metrics(self) -> Dict[str, Any]:
        state = self._cortex_state if isinstance(self._cortex_state, dict) else {}
        scheduler_status: Dict[str, Any] = {}
        dual_brain_status: Dict[str, Any] = {}
        if self.dual_brain_controller is not None:
            try:
                dual_brain_status = self.dual_brain_controller.get_status()
                scheduler_status = dict(dual_brain_status.get("scheduler", {}) or {})
            except Exception:
                scheduler_status = {}
                dual_brain_status = {}
        return {
            "planning_attempt_count": int(state.get("planning_attempt_count", 0) or 0),
            "blocked_replan_count": int(state.get("blocked_replan_count", 0) or 0),
            "no_execution_return_count": int(state.get("no_execution_return_count", 0) or 0),
            "executed_step_count": int(state.get("executed_step_count", 0) or 0),
            "watchdog_triggered": bool(state.get("watchdog_triggered", False)),
            "watchdog_reason": str(state.get("watchdog_reason", "") or ""),
            "blocker_replan_only": bool(state.get("blocker_replan_only", False)),
            "uncertain_execution": bool(state.get("uncertain_execution", False)),
            "last_planning_sec": state.get("last_planning_sec"),
            "planning_sec_median": state.get("planning_sec_median"),
            "last_no_execution_planning_sec": state.get("last_no_execution_planning_sec"),
            "no_execution_planning_sec_median": state.get("no_execution_planning_sec_median"),
            "no_execution_planning_sample_count": int(state.get("no_execution_planning_sample_count", 0) or 0),
            "watchdog_dynamic_timeout_sec": state.get("watchdog_dynamic_timeout_sec"),
            "same_step_elapsed_sec": state.get("same_step_elapsed_sec"),
            "agent_run_dir_name": getattr(self, "agent_run_dir_name", ""),
            "planner_comp_model": self._provider_model_metadata.get("planner_comp_model", ""),
            "embedding_model": self._provider_model_metadata.get("embedding_model", ""),
            "prompt_profile": getattr(self, "prompt_profile", ""),
            "resolved_action_planning_template": self._resolved_big_brain_template_paths.get("action_planning", ""),
            "resolved_task_inference_template": self._resolved_big_brain_template_paths.get("task_inference", ""),
            "scheduler_status": scheduler_status,
            "dual_brain_status": dual_brain_status,
        }

    @staticmethod
    def _extract_progress_quantity(task_eval: Optional[Dict[str, Any]]) -> Optional[float]:
        if not isinstance(task_eval, dict):
            return None

        for key in ("quantity", "current_quantity", "final_quantity"):
            value = task_eval.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                stripped = value.strip()
                if stripped and re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
                    return float(stripped)
        return None

    @staticmethod
    def _is_productive_action(action_text: str) -> bool:
        normalized = str(action_text or "").strip().lower()
        return normalized.startswith(("use(", "interact(", "choose_option(", "craft("))

    @staticmethod
    def _normalize_action_text(action: Any) -> str:
        """Normalize action text for comparison: collapse whitespace, unify quotes."""
        s = str(action or "").strip()
        s = s.replace("'", '"')           # unify quotes
        s = re.sub(r'\s+', ' ', s)        # collapse whitespace
        s = re.sub(r'\s*=\s*', '=', s)    # remove spaces around =
        s = re.sub(r'\s*,\s*', ', ', s)   # normalize comma spacing
        s = re.sub(r'\(\s+', '(', s)      # no space after (
        s = re.sub(r'\s+\)', ')', s)      # no space before )
        return s

    @staticmethod
    def _count_same_action_tail(actions: List[str], current_action: str) -> int:
        if not current_action:
            return 0

        norm_current = PipelineRunner._normalize_action_text(current_action)
        streak = 0
        for item in reversed(actions):
            if PipelineRunner._normalize_action_text(item) != norm_current:
                break
            streak += 1
        return streak

    @staticmethod
    def _detect_oscillation(actions: List[str], min_pairs: int = 2) -> int:
        """检测反向 move 振荡。返回振荡对数（0 = 无振荡）"""
        if len(actions) < min_pairs * 2:
            return 0
        def _parse_move(a):
            m = re.match(r"^move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)$", str(a or ""), re.IGNORECASE)
            return (int(m.group(1)), int(m.group(2))) if m else None

        pairs = 0
        for i in range(len(actions) - 1, 0, -1):
            curr, prev = _parse_move(actions[i]), _parse_move(actions[i-1])
            if curr is None or prev is None:
                break
            # 反向: 同轴取反 (x=-x,y=-y) 或 单轴取反 (x=-x,y=y=0) 等
            is_opposite = (
                (curr[0] == -prev[0] and curr[1] == -prev[1]) or
                (curr[0] == -prev[0] and curr[1] == prev[1] == 0) or
                (curr[1] == -prev[1] and curr[0] == prev[0] == 0)
            )
            if is_opposite:
                pairs += 1
            else:
                break
        return pairs

    @staticmethod
    def _detect_position_issue(*texts: Any) -> bool:
        return detect_position_issue(*texts)

    @staticmethod
    def _build_task_progress_summary(
        action_text: str,
        previous_progress: Optional[float],
        current_progress: Optional[float],
        progress_delta: Optional[float],
        zero_progress_streak: int,
        repeated_action_streak: int,
        productive_action: bool,
        state_changed: bool,
        position_issue_detected: bool,
        completed: Optional[bool],
        errors_info: str = "",
        oscillation_streak: int = 0,
        missing_confirmation: bool = False,
    ) -> str:
        parts: List[str] = []

        if action_text:
            parts.append(f"Last action: {action_text}.")

        if current_progress is not None:
            if previous_progress is not None and progress_delta is not None:
                if progress_delta > 0:
                    parts.append(
                        f"Task progress increased from {previous_progress:g} to {current_progress:g}."
                    )
                elif progress_delta < 0:
                    parts.append(
                        f"Task progress decreased from {previous_progress:g} to {current_progress:g}."
                    )
                else:
                    parts.append(f"Task progress stayed at {current_progress:g}.")
            else:
                parts.append(f"Recorded task progress is {current_progress:g}.")

        if productive_action and progress_delta is not None and progress_delta > 0:
            parts.append("The productive action made measurable task progress.")
        elif productive_action and not state_changed:
            parts.append("The productive action had no observable effect.")
        elif state_changed:
            parts.append("The action changed the local state.")

        if zero_progress_streak >= 2:
            parts.append(
                f"There have been {zero_progress_streak} consecutive productive actions without progress."
            )

        # Strengthen repeated-action feedback into a directive: name the exact
        # forbidden action so the LLM cannot ignore it as background warning.
        if repeated_action_streak >= 3 and action_text:
            parts.append(
                f"FORBIDDEN: The action `{action_text}` has failed {repeated_action_streak} "
                f"consecutive times. Your next action MUST NOT be `{action_text}`. Choose a "
                f"different action — typically move() to reposition or a different direction."
            )
        elif repeated_action_streak >= 2:
            parts.append(
                f"The same action has been repeated {repeated_action_streak} times. "
                f"If it failed again, do not repeat it a third time — change action or position."
            )

        if position_issue_detected:
            parts.append(
                "Current task reasoning indicates the player must move closer or become adjacent before using the tool again."
            )

        if missing_confirmation:
            if state_changed:
                parts.append(
                    "The action returned no explicit confirmation, but the observed state suggests it may have taken effect."
                )
            else:
                parts.append(
                    "The action returned no explicit confirmation and there was no observed state change."
                )

        blocked_feedback = errors_info.lower() if isinstance(errors_info, str) else ""
        if "blocked by an obstacle" in blocked_feedback or "path is likely blocked" in blocked_feedback:
            parts.append(
                "The last move appears to be blocked by an obstacle. If the tile in front of the player or the effect point contains stone, twig, wood, weeds, grass, or fiber, clear it first with the matching tool before moving again. "
                "If no clearable obstacle is visible, sidestep 1-2 tiles on the perpendicular axis (if you were moving on x, try a small y move; if you were moving on y, try a small x move) before retrying. Drop magnitude to 1 tile."
            )

        if oscillation_streak >= 2:
            parts.append(
                f"FORBIDDEN: The agent has been oscillating between opposite moves for "
                f"{oscillation_streak} consecutive pairs. Your next action MUST NOT be a move that "
                f"reverses the previous direction. Commit to one direction for at least 3-5 tiles, "
                f"or switch to a perpendicular axis. If your last move was on x-axis, the next move "
                f"must be on y-axis (or vice versa)."
            )

        if errors_info:
            parts.append(f"Execution feedback: {errors_info}")

        if completed is True:
            parts.append("The task is completed.")
        elif completed is False and current_progress is not None:
            parts.append("The task is not completed yet.")

        return " ".join(parts).strip()

    @staticmethod
    def _infer_cultivation_task_kind(task_text: Any) -> str:
        normalized = str(task_text or "").strip().lower()
        if normalized.startswith(("till ", "till_")):
            return "till"
        if normalized.startswith(("fertilize ", "fertilize_")):
            return "fertilize"
        if normalized.startswith(("sow ", "sow_", "plant ", "plant_")):
            return "sow"
        if normalized.startswith(("water ", "water_")):
            return "water"
        if normalized.startswith(("harvest ", "harvest_")):
            return "harvest"
        if normalized.startswith(("cultivate and harvest ", "cultivate_and_harvest_")):
            return "cultivate_and_harvest"
        return ""

    @staticmethod
    def _normalize_cultivation_failure_root_cause(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in _CULTIVATION_FAILURE_ROOT_CAUSES:
            return text
        return "unknown" if text else ""

    @staticmethod
    def _normalize_cultivation_required_change_type(value: Any) -> str:
        text = str(value or "").strip().lower()
        if text in _CULTIVATION_REQUIRED_CHANGE_TYPES:
            return text
        return ""

    @staticmethod
    def _build_cultivation_failure_signature(
        task_kind: str,
        action_text: str,
        failure_root_cause: str,
        required_change_type: str,
        blocker_signature: str = "",
    ) -> str:
        parts = [
            str(task_kind or "").strip(),
            str(action_text or "").strip(),
            str(failure_root_cause or "").strip(),
            str(required_change_type or "").strip(),
            str(blocker_signature or "").strip(),
        ]
        return " | ".join(part for part in parts if part)

    def _classify_cultivation_failure(
        self,
        *,
        task_text: str,
        action_text: str,
        state_changed: bool,
        menu_changed: bool,
        meaningful_failure: bool,
        position_issue_detected: bool,
        errors_info_text: str,
    ) -> Dict[str, str]:
        task_kind = self._infer_cultivation_task_kind(task_text)
        if not task_kind or not meaningful_failure:
            return {
                "failure_root_cause": "",
                "required_change_type": "",
                "failure_signature": "",
            }

        normalized_action = str(action_text or "").strip().lower()
        lowered_errors = str(errors_info_text or "").strip().lower()

        if normalized_action.startswith("menu(") and not menu_changed:
            failure_root_cause = "menu_stuck"
            required_change_type = "close_menu"
        elif "blocked by an obstacle" in lowered_errors or "path is likely blocked" in lowered_errors:
            failure_root_cause = "movement_blocked"
            required_change_type = "change_position"
        else:
            validation = validate_cultivation_pre_execution_action(
                state=self._cortex_state,
                action_text=action_text,
            )
            failure_root_cause = self._normalize_cultivation_failure_root_cause(
                validation.get("failure_root_cause", "")
            )
            required_change_type = self._normalize_cultivation_required_change_type(
                validation.get("required_change_type", "")
            )
            if not failure_root_cause:
                if task_kind == "till" and normalized_action.startswith("use(") and not state_changed:
                    failure_root_cause = "wrong_tile_alignment" if position_issue_detected else "invalid_target_tile"
                    required_change_type = "change_position" if position_issue_detected else "change_target_tile"
                elif task_kind == "fertilize" and normalized_action.startswith("interact(") and not state_changed:
                    failure_root_cause = "wrong_facing_direction" if position_issue_detected else "invalid_target_tile"
                    required_change_type = "change_facing" if position_issue_detected else "change_target_tile"
                else:
                    failure_root_cause = "unknown"
        state = self._cortex_state if isinstance(self._cortex_state, dict) else {}
        blocker_signature = str(state.get("current_blocker_signature", "") or "").strip()
        failure_signature = self._build_cultivation_failure_signature(
            task_kind=task_kind,
            action_text=action_text,
            failure_root_cause=failure_root_cause,
            required_change_type=required_change_type,
            blocker_signature=blocker_signature,
        )
        return {
            "failure_root_cause": failure_root_cause,
            "required_change_type": required_change_type,
            "failure_signature": failure_signature,
        }

    @staticmethod
    def _build_action_feedback_text(
        *,
        latest_execution_summary: str,
        failure_root_cause: str,
        required_change_type: str,
        failure_signature: str,
    ) -> str:
        parts: List[str] = []
        if latest_execution_summary:
            parts.append(latest_execution_summary.strip())
        if failure_root_cause:
            parts.append(f"Failure root cause: {failure_root_cause}.")
        if required_change_type:
            parts.append(f"Required change type: {required_change_type}.")
        if failure_signature:
            parts.append(f"Failure signature: {failure_signature}.")
        return " ".join(part for part in parts if part).strip()

    def _obs_to_gathered_info(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        chosen_item = obs.get("chosen_item", None)
        facing_direction = obs.get("facing_direction", "")
        position = obs.get("position", "")
        surroundings_text = self._format_surroundings_text(obs.get("surroundings", ""), position)
        toolbar_information = self._format_toolbar_text(obs.get("inventory", []), chosen_item)
        description_text = str(
            obs.get("description")
            or obs.get("image_description")
            or surroundings_text
            or ""
        ).strip()
        return {
            "description": description_text,
            "image_description": description_text,
            "surroundings": surroundings_text,
            "inventory": obs.get("inventory", []),
            "chosen_item": chosen_item,
            "selected_position": self._infer_selected_position(obs.get("inventory", []), chosen_item),
            "selected_item_name": self._extract_item_name(chosen_item),
            "toolbar_information": toolbar_information,
            "location": obs.get("location", ""),
            "position": position,
            "current_position": position,
            "facing_direction": facing_direction,
            "facing_position": self._infer_facing_position(position, facing_direction),
            "time": obs.get("time", ""),
            "day": obs.get("day", ""),
            "season": obs.get("season", ""),
            "energy": obs.get("energy", None),
            "health": obs.get("health", None),
            "money": obs.get("money", None),
            "current_menu": obs.get("current_menu", None),
            "crops": self._format_list_field(obs.get("crops", [])),
            "buildings": self._format_buildings(obs.get("buildings", []), position),
            "furniture": self._format_list_field(obs.get("furniture", [])),
            "npcs": self._format_list_field(obs.get("npcs", [])),
            "exits": self._format_exits(obs.get("exits", []), position),
        }

    def _sync_stardew_memory_from_obs(self, obs: Dict[str, Any], step_num: int, latest_image: str = "") -> Dict[str, Any]:
        recent_or_default = getattr(self, "_get_recent_or_default", None)
        if not callable(recent_or_default):
            recent_or_default = lambda key, default=None: default

        image_path = latest_image
        if not image_path:
            image_path = self._pick_latest_available_image_path(obs=obs, step_num=step_num)

        recent_augmented = image_path or recent_or_default(constants.AUGMENTED_IMAGES_MEM_BUCKET, "")

        selected_position = self._infer_selected_position(obs.get("inventory", []), obs.get("chosen_item", None))
        selected_item_name = self._extract_item_name(obs.get("chosen_item", None))
        surroundings_text = self._format_surroundings_text(obs.get("surroundings", ""), obs.get("position", ""))
        observed_gathered_info = self._obs_to_gathered_info(obs)
        subtask_values = self._resolve_current_subtask_values()
        current_state = self._cortex_state if isinstance(self._cortex_state, dict) else {}
        current_history_summary = str(
            current_state.get("history_summary", "")
            or current_state.get("summarization", "")
            or recent_or_default("history_summary", recent_or_default("summarization", ""))
            or ""
        ).strip()
        current_summarization = str(
            current_state.get("summarization", "")
            or current_state.get("history_summary", "")
            or recent_or_default("summarization", "")
            or ""
        ).strip()
        current_action_feedback = str(
            current_state.get("action_feedback", "")
            or current_state.get("latest_execution_summary", "")
            or recent_or_default("action_feedback", "")
            or ""
        ).strip()
        latest_executed_action = str(
            current_state.get("last_action", "")
            or current_state.get("pre_action", "")
            or recent_or_default("pre_action", recent_or_default("action", ""))
            or ""
        ).strip()
        latest_exec_info = current_state.get("last_exec_info", {})
        if not isinstance(latest_exec_info, dict) or not latest_exec_info:
            fallback_exec_info = recent_or_default("exec_info", {})
            latest_exec_info = fallback_exec_info if isinstance(fallback_exec_info, dict) else {}
        previous_toolbar_information = current_state.get("previous_toolbar_information", None)
        if previous_toolbar_information in (None, "", []):
            previous_toolbar_information = recent_or_default(
                "previous_toolbar_information",
                recent_or_default("toolbar_information", []),
            )
        subtask_hints = self._sanitize_subtask_memory(
            subtask_values.get("subtask_description", ""),
            subtask_values.get("subtask_reasoning", ""),
            obs.get("chosen_item", None),
        )
        prompt_fact_fields = extract_stardew_prompt_fact_fields(
            state={
                **current_state,
                "task": self.task_description,
                "main_task": self.task_description,
                "surroundings": surroundings_text,
                "position": obs.get("position", ""),
                "current_position": obs.get("position", ""),
                "facing_direction": obs.get("facing_direction", ""),
                "facing_position": self._infer_facing_position(obs.get("position", ""), obs.get("facing_direction", "")),
                "selected_position": selected_position,
                "selected_item_name": selected_item_name,
                "toolbar_information": self._format_toolbar_text(obs.get("inventory", []), obs.get("chosen_item", None)),
                "current_menu": obs.get("current_menu", None),
            },
            gathered_info=observed_gathered_info,
        )
        observed_gathered_info.update(
            {
                "front_tile_summary": prompt_fact_fields.get("front_tile_summary", "(none)"),
                "blocked_recovery_hint": prompt_fact_fields.get("blocked_recovery_hint", ""),
                "current_blocker_signature": prompt_fact_fields.get("current_blocker_signature", "(none)"),
                "nearest_grounded_target_summary": prompt_fact_fields.get("nearest_grounded_target_summary", "(none)"),
                "failure_root_cause": prompt_fact_fields.get("failure_root_cause", ""),
                "failure_signature": prompt_fact_fields.get("failure_signature", ""),
                "required_change_type": prompt_fact_fields.get("required_change_type", ""),
                "deadlock_signature": prompt_fact_fields.get("deadlock_signature", ""),
                "deadlock_reflection_cycles": str(prompt_fact_fields.get("deadlock_reflection_cycles", "0")),
            }
        )

        sync_payload = {
            "task_description": self.task_description,
            "start_frame_id": max(step_num - 1, 0),
            "end_frame_id": step_num,
            "toolbar_information": self._format_toolbar_text(obs.get("inventory", []), obs.get("chosen_item", None)),
            "selected_position": selected_position,
            "selected_item_name": selected_item_name,
            "inventory": obs.get("inventory", []),
            "chosen_item": obs.get("chosen_item", None),
            "surroundings": surroundings_text,
            "location": obs.get("location", ""),
            "position": obs.get("position", ""),
            "facing_direction": obs.get("facing_direction", ""),
            "facing_position": self._infer_facing_position(obs.get("position", ""), obs.get("facing_direction", "")),
            "time": obs.get("time", ""),
            "day": obs.get("day", ""),
            "date_time": f"{obs.get('day', '')} {obs.get('time', '')}".strip(),
            "season": obs.get("season", ""),
            "energy": obs.get("energy", None),
            "health": obs.get("health", None),
            "money": obs.get("money", None),
            "current_menu": obs.get("current_menu", None),
            "crops": self._format_list_field(obs.get("crops", [])),
            "buildings": self._format_buildings(obs.get("buildings", []), obs.get("position")),
            "furniture": self._format_list_field(obs.get("furniture", [])),
            "npcs": self._format_list_field(obs.get("npcs", [])),
            "exits": self._format_exits(obs.get("exits", []), obs.get("position")),
            "basic_knowledge": obs.get("basic_knowledge", []),
            "action_feedback": current_action_feedback,
            "gathered_info": observed_gathered_info,
            "history_summary": current_history_summary,
            "summarization": current_summarization,
            "subtask_description": subtask_values.get("subtask_description", self.initial_subtask_description),
            "subtask_reasoning": subtask_values.get("subtask_reasoning", self.initial_subtask_reasoning),
            "sanitized_subtask_hint": subtask_hints.get("sanitized_subtask_hint", ""),
            "redundant_tool_selection": bool(subtask_hints.get("redundant_tool_selection", False)),
            "selected_item_already_correct": bool(subtask_hints.get("selected_item_already_correct", False)),
            "front_tile_summary": prompt_fact_fields.get("front_tile_summary", "(none)"),
            "blocked_recovery_hint": prompt_fact_fields.get("blocked_recovery_hint", ""),
            "current_blocker_signature": prompt_fact_fields.get("current_blocker_signature", "(none)"),
            "nearest_grounded_target_summary": prompt_fact_fields.get("nearest_grounded_target_summary", "(none)"),
            "failure_root_cause": prompt_fact_fields.get("failure_root_cause", ""),
            "failure_signature": prompt_fact_fields.get("failure_signature", ""),
            "required_change_type": prompt_fact_fields.get("required_change_type", ""),
            "deadlock_signature": prompt_fact_fields.get("deadlock_signature", ""),
            "deadlock_reflection_cycles": str(prompt_fact_fields.get("deadlock_reflection_cycles", "0")),
            "previous_toolbar_information": previous_toolbar_information,
            "exec_info": latest_exec_info,
            "pre_action": latest_executed_action,
            "action": latest_executed_action,
            "pre_decision_making_reasoning": recent_or_default("pre_decision_making_reasoning", recent_or_default("action_planning_reasoning", "")),
            "decision_making_reasoning": recent_or_default("decision_making_reasoning", recent_or_default("action_planning_reasoning", "")),
            "self_reflection_reasoning": recent_or_default("self_reflection_reasoning", ""),
            **self.task_acquisition_context,
        }

        if image_path:
            sync_payload["image_paths"] = [image_path]
            sync_payload["image_introduction"] = [
                {
                    "introduction": "This screenshot is the current step of the game.",
                    "path": image_path,
                    "assistant": "",
                }
            ]
            sync_payload[constants.IMAGES_MEM_BUCKET] = image_path
            sync_payload[constants.AUGMENTED_IMAGES_MEM_BUCKET] = recent_augmented or image_path
            sync_payload["screenshot_path"] = image_path
            sync_payload["screenshot_augmented_path"] = recent_augmented or image_path

        self._update_memory_histories(sync_payload)
        return sync_payload

    def _derive_screenshot_candidates_from_step(
        self,
        reference_paths: List[str],
        step_num: int,
    ) -> List[str]:
        if step_num < 0:
            return []

        candidates: List[str] = []
        seen: set[str] = set()

        for raw_path in reference_paths:
            resolved = self._resolve_screenshot_path(raw_path)
            if not resolved:
                continue

            normalized = os.path.normpath(str(resolved))
            directory, filename = os.path.split(normalized)
            match = re.match(r"^(screenshot_(\d+)_)(\d+)(\.[^.]+)$", filename, re.IGNORECASE)
            if not match:
                continue

            prefix = match.group(1)
            ext = match.group(4)
            for candidate_step in (step_num, step_num - 1, step_num + 1, 0):
                if candidate_step < 0:
                    continue
                candidate = os.path.join(directory, f"{prefix}{candidate_step}{ext}")
                if candidate not in seen:
                    seen.add(candidate)
                    candidates.append(candidate)

        return candidates

    def _pick_latest_available_image_path(
        self,
        obs: Optional[Dict[str, Any]] = None,
        preferred_path: str = "",
        step_num: int = -1,
    ) -> str:
        obs = obs or {}

        resolved_obs_paths = self._resolve_image_paths(obs.get("image_paths", []))
        for candidate in reversed(resolved_obs_paths):
            if candidate and os.path.exists(candidate):
                return candidate

        reference_paths = list(resolved_obs_paths)
        if preferred_path:
            reference_paths.append(preferred_path)
        if step_num >= 0:
            for candidate in self._derive_screenshot_candidates_from_step(reference_paths, step_num):
                if candidate and os.path.exists(candidate):
                    return candidate

        resolved_preferred = self._resolve_screenshot_path(preferred_path) if preferred_path else ""
        if resolved_preferred and os.path.exists(resolved_preferred):
            return resolved_preferred

        # 全部不存在时，尝试扫描截图目录找到最新的文件
        all_candidates = list(resolved_obs_paths)
        if resolved_preferred:
            all_candidates.append(resolved_preferred)
        for candidate in reversed(all_candidates):
            if not candidate:
                continue
            scan_dir = os.path.dirname(candidate)
            if scan_dir and os.path.isdir(scan_dir):
                import glob
                pattern = os.path.join(scan_dir, "screenshot_*.jpeg")
                try:
                    found = sorted(
                        [f for f in glob.glob(pattern) if os.path.exists(f)],
                        key=lambda f: os.path.getmtime(f),
                        reverse=True,
                    )
                except (OSError, FileNotFoundError):
                    found = []
                if found:
                    return found[0]

        return ""  # 宁可返回空，也不返回不存在的路径

    def _refresh_stardew_memory_from_cortex_state(self) -> None:
        if not isinstance(self._cortex_state, dict):
            return

        frame_ids = self._cortex_state.get("frame_ids", (0, 0))
        if not isinstance(frame_ids, tuple) or len(frame_ids) != 2:
            frame_ids = (0, 0)

        try:
            step_num = int(frame_ids[1])
        except (TypeError, ValueError):
            step_num = int(self._cortex_state.get("step_count", 0) or 0)

        latest_image = self._pick_latest_available_image_path(
            obs=self._latest_obs,
            preferred_path=str(self._cortex_state.get("screenshot_path", "") or ""),
            step_num=step_num,
        )
        if not latest_image:
            return

        current_state_image = str(self._cortex_state.get("screenshot_path", "") or "")
        if current_state_image != latest_image:
            logger.write(
                f"[Cortex] Refreshing stale screenshot path: {current_state_image or '<empty>'} -> {latest_image}"
            )
            self._cortex_state["screenshot_path"] = latest_image

        self._sync_stardew_memory_from_obs(
            obs=self._latest_obs if isinstance(self._latest_obs, dict) else {},
            step_num=step_num,
            latest_image=latest_image,
        )

    def _update_memory_histories(self, data: Dict[str, Any]) -> None:
        self.memory.update_info_history(data)
        legacy_memory = getattr(self, "legacy_memory", None)
        if legacy_memory is not None and legacy_memory is not self.memory:
            legacy_memory.update_info_history(data)

    def _run_cortex_information_gathering(self) -> Dict[str, Any]:
        try:
            self._refresh_stardew_memory_from_cortex_state()
            self.information_gathering_preprocess()
            response = self.information_gathering()
            processed_response = self.information_gathering_postprocess(response)
            logger.write(f"[Cortex][InfoGathering processed response] {_truncate_for_log(processed_response, max_len=500)}")
            return processed_response if isinstance(processed_response, dict) else {}
        except Exception as e:
            import traceback
            logger.error(f"[Cortex][InfoGathering] fallback to text observation due to error: {e}\n{traceback.format_exc()}")
            fallback = self._obs_to_gathered_info(self._latest_obs)
            fallback["info_gathering_error"] = str(e)
            return fallback

    def _run_cortex_task_inference(self) -> Dict[str, Any]:
        try:
            if hasattr(self, "_prepare_big_brain_template_for_call"):
                self._prepare_big_brain_template_for_call("task_inference")
            elif hasattr(self, "_ensure_big_brain_template_integrity"):
                self._ensure_big_brain_template_integrity("task_inference")
            self._refresh_stardew_memory_from_cortex_state()
            self.task_inference_preprocess()
            response = self.task_inference()
            processed_response = _CradleTaskInferencePostprocess()(response)
            self._update_memory_histories(processed_response)
            logger.write(f"[Cortex][TaskInference processed response] {_truncate_for_log(processed_response, max_len=500)}")
            return processed_response if isinstance(processed_response, dict) else {}
        except Exception as e:
            logger.error(f"[Cortex][TaskInference] fallback to initial subtask due to error: {e}")
            return {
                "summarization": self._get_recent_or_default("summarization", ""),
                "history_summary": self._get_recent_or_default("history_summary", self._get_recent_or_default("summarization", "")),
                "subtask_description": self.initial_subtask_description,
                "subtask_reasoning": self.initial_subtask_reasoning,
                "task_inference_error": str(e),
            }

    def _resolve_screenshot_path(self, raw_path: str) -> str:
        if not raw_path:
            return ""

        raw = os.path.normpath(str(raw_path))
        project_root = os.path.dirname(os.path.dirname(agent_root))
        candidates: List[str] = []

        def _append_candidate(path: str) -> None:
            if path and path not in candidates:
                candidates.append(path)

        def _append_common_candidates(path: str) -> None:
            _append_candidate(os.path.abspath(path))
            _append_candidate(os.path.abspath(os.path.join(project_root, path)))
            _append_candidate(os.path.abspath(os.path.join(project_root, "env", path)))
            _append_candidate(os.path.abspath(os.path.join(project_root, "agent", path)))

        _append_common_candidates(raw)

        trimmed = raw
        up_prefix = ".." + os.sep
        if trimmed.startswith(up_prefix):
            while trimmed.startswith(up_prefix):
                trimmed = trimmed[len(up_prefix):]
            if trimmed:
                _append_common_candidates(trimmed)

        for p in candidates:
            if os.path.exists(p):
                return p
        return raw_path

    def _coerce_image_paths(self, image_paths: Any) -> List[str]:
        if image_paths is None:
            return []
        if isinstance(image_paths, str):
            text = image_paths.strip()
            return [text] if text else []

        if isinstance(image_paths, Iterable):
            values = list(image_paths)
        else:
            values = [image_paths]

        normalized: List[str] = []
        for value in values:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                normalized.append(text)
        return normalized

    def _resolve_image_paths(self, image_paths: Any) -> List[str]:
        resolved: List[str] = []
        for raw_path in self._coerce_image_paths(image_paths):
            path = self._resolve_screenshot_path(raw_path)
            if path:
                resolved.append(path)
        return resolved

    def _normalize_cortex_action(self, action_expr: str) -> str:
        action = (action_expr or "").strip()
        if not action:
            return ""

        action = action.strip("`").strip()
        skill_name = action.split("(", 1)[0].strip().lower()

        move_named = re.match(
            r"^move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)$",
            action,
            re.IGNORECASE,
        )
        if move_named:
            max_relative_move = 20
            x = int(move_named.group(1))
            y = int(move_named.group(2))
            clamped_x = max(-max_relative_move, min(max_relative_move, x))
            clamped_y = max(-max_relative_move, min(max_relative_move, y))
            if clamped_x != x:
                logger.warn(f"[Cortex] Clamped move x={x} -> {clamped_x}")
            if clamped_y != y:
                logger.warn(f"[Cortex] Clamped move y={y} -> {clamped_y}")
            return f"move(x={clamped_x}, y={clamped_y})"

        move_positional = re.match(
            r"^move\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$",
            action,
            re.IGNORECASE,
        )
        if move_positional:
            logger.warn(f"[Cortex] Normalized positional move syntax: '{action}'")
            return self._normalize_cortex_action(
                f"move(x={int(move_positional.group(1))}, y={int(move_positional.group(2))})"
            )

        directional_named = re.match(
            r"^(use|interact)\(\s*direction\s*=\s*[\"']?(up|down|left|right)[\"']?\s*\)$",
            action,
            re.IGNORECASE,
        )
        if directional_named:
            return f'{directional_named.group(1).lower()}(direction="{directional_named.group(2).lower()}")'

        directional_positional = re.match(
            r"^(use|interact)\(\s*(up|down|left|right)\s*\)$",
            action,
            re.IGNORECASE,
        )
        if directional_positional:
            logger.warn(f"[Cortex] Normalized legacy directional syntax: '{action}'")
            return f'{directional_positional.group(1).lower()}(direction="{directional_positional.group(2).lower()}")'

        slot_named = re.match(
            r"^(choose_item|attach_item)\(\s*slot_index\s*=\s*(-?\d+)\s*\)$",
            action,
            re.IGNORECASE,
        )
        if slot_named:
            return f"{slot_named.group(1).lower()}(slot_index={int(slot_named.group(2))})"

        slot_positional = re.match(
            r"^(choose_item|attach_item)\(\s*(-?\d+)\s*\)$",
            action,
            re.IGNORECASE,
        )
        if slot_positional:
            logger.warn(f"[Cortex] Normalized positional slot syntax: '{action}'")
            return f"{slot_positional.group(1).lower()}(slot_index={int(slot_positional.group(2))})"

        bare_call = re.match(
            r"^(unattach_item|nop)\(\s*\)$",
            action,
            re.IGNORECASE,
        )
        if bare_call:
            return f"{bare_call.group(1).lower()}()"

        passthrough = re.match(
            r"^(craft|choose_option|menu)\([^)\n\r`]*\)$",
            action,
            re.IGNORECASE,
        )
        if passthrough:
            return passthrough.group(0).strip()

        if action.startswith("use_tool("):
            return self._normalize_cortex_action(action.replace("use_tool(", "use(", 1))
        if action.startswith("do_action("):
            return self._normalize_cortex_action(action.replace("do_action(", "interact(", 1))
        if action.startswith("select_tool("):
            key_match = re.search(r"key\s*=\s*['\"]?([0-9\-\+])['\"]?", action)
            if key_match:
                key = key_match.group(1)
                mapping = {
                    "1": 0, "2": 1, "3": 2, "4": 3, "5": 4,
                    "6": 5, "7": 6, "8": 7, "9": 8, "0": 9,
                    "-": 10, "+": 11,
                }
                slot = mapping.get(key, 0)
                return f"choose_item(slot_index={slot})"

        # wait(...) → nop()
        if skill_name == "wait":
            return "nop()"

        # move_right/left/up/down(...) → warn + nop()
        if skill_name in ("move_right", "move_left", "move_up", "move_down"):
            logger.warn(f"[Cortex] Unexpected directional action '{action}', converting to nop()")
            return "nop()"


        # Composite skills are disabled — filter them out
        if skill_name in (
            "go_through_door",
            "get_out_of_house",
            "enter_door_and_sleep",
            "go_home",
            "go_to_store",
            "buy_item",
            "use_tool_multiple_times",
            "do_action_multiple_times",
        ):
            logger.warn(f"[Cortex] Composite skill '{action}' is disabled, filtering out")
            return ""

        # Unknown action → warn + filter out
        logger.warn(f"[Cortex] Unknown action '{action}', filtering out")
        return ""

    @staticmethod
    def _select_cortex_suggestion_for_logging(
        result_state: Dict[str, Any],
        normalized_actions: List[str],
        normalized_suggestion_actions: List[str],
    ) -> Optional[str]:
        return select_cortex_suggestion_for_logging(
            result_state=result_state,
            normalized_actions=normalized_actions,
            normalized_suggestion_actions=normalized_suggestion_actions,
        )

    def _resolve_cortex_result_actions(self, result_state: Dict[str, Any]) -> Dict[str, Any]:
        planned_actions = result_state.get("planned_actions", [])
        if not isinstance(planned_actions, list):
            planned_actions = [str(planned_actions)]

        normalized_planned_actions = [self._normalize_cortex_action(a) for a in planned_actions]
        normalized_planned_actions = [a for a in normalized_planned_actions if a]

        suggestions = result_state.get("suggestions", [])
        normalized_suggestion_actions: List[str] = []
        if isinstance(suggestions, list):
            for suggestion in suggestions:
                if isinstance(suggestion, dict):
                    normalized = self._normalize_cortex_action(str(suggestion.get("action", "")))
                else:
                    normalized = self._normalize_cortex_action(str(suggestion))
                if normalized:
                    normalized_suggestion_actions.append(normalized)

        action_resolution = resolve_cortex_executable_actions(
            result_state=result_state,
            normalized_actions=normalized_planned_actions,
            normalized_suggestion_actions=normalized_suggestion_actions,
        )
        return {
            "actions": list(action_resolution.get("actions", [])),
            "suggestion_actions": normalized_suggestion_actions,
            "execution_source": str(action_resolution.get("execution_source", "") or ""),
            "blocked_reason": str(action_resolution.get("blocked_reason", "") or ""),
            "used_suggestion_fallback": bool(action_resolution.get("used_suggestion_fallback", False)),
        }

    @staticmethod
    def _big_brain_only_repair_enabled(state: Dict[str, Any]) -> bool:
        return bool(state.get("big_brain_only", False))

    def _build_big_brain_only_repair_feedback(
        self,
        *,
        rejected_action: str,
        validation: Dict[str, Any],
        blocked_reason: str,
    ) -> str:
        invalid_reason = str(validation.get("invalid_reason", "") or blocked_reason or "pre_execution_validation").strip()
        failure_root_cause = str(validation.get("failure_root_cause", "") or "").strip()
        required_change_type = str(validation.get("required_change_type", "") or "").strip()
        failure_signature = str(validation.get("failure_signature", "") or "").strip()
        blocker_signature = str(
            validation.get("blocker_signature", "")
            or validation.get("current_blocker_signature", "")
            or ""
        ).strip()

        details = [
            "BIGBRAIN-ONLY REPAIR REQUIRED.",
            f"The previous single action `{rejected_action}` was rejected before execution: {invalid_reason}.",
            "Do not output the same action again.",
            "Use the same current observation and choose exactly one different immediately executable action.",
            "If the rejected action was use()/interact(), change the target tile or direction; if the facing line is blocked, move to a grounded adjacent target first.",
        ]
        if failure_root_cause:
            details.append(f"Failure root cause: {failure_root_cause}.")
        if required_change_type:
            details.append(f"Required change type: {required_change_type}.")
        if blocker_signature:
            details.append(f"Blocker signature: {blocker_signature}.")
        if failure_signature:
            details.append(f"Failure signature: {failure_signature}.")
        return " ".join(details)

    def _run_big_brain_only_repair_replan(
        self,
        *,
        state: Dict[str, Any],
        rejected_action: str,
        validation: Dict[str, Any],
        blocked_reason: str,
    ) -> tuple[Dict[str, Any], Dict[str, Any], float]:
        repair_feedback = self._build_big_brain_only_repair_feedback(
            rejected_action=rejected_action,
            validation=validation,
            blocked_reason=blocked_reason,
        )
        try:
            repair_attempt_count = int(state.get("planning_attempt_count", 0) or 0) + 1
        except (TypeError, ValueError):
            repair_attempt_count = 1
        repair_state = {
            **state,
            "action_feedback": repair_feedback,
            "big_brain_only_repair": True,
            "big_brain_only_rejected_action": rejected_action,
            "planning_attempt_count": repair_attempt_count,
            "force_big_brain_replan": True,
            "brain_mode": "big",
            "has_execution_feedback": False,
            "execution_pending": False,
            "pending_action": "",
            "pending_step_index": None,
            "pending_suggested_action": "",
        }
        logger.warn(f"[Cortex] BigBrain-only repair replan: {repair_feedback}")

        repair_started_at = time.time()
        repair_result_state = cast(
            Dict[str, Any],
            self.dual_brain_controller.step(
                cast(Dict[str, Any], repair_state),
                self._cortex_workflow_config,
            ),
        )
        repair_planning_sec = max(0.0, time.time() - repair_started_at)
        repaired_state = {**repair_state, **repair_result_state}
        repaired_state.update(
            record_cortex_planning_latency(
                state=repaired_state,
                planning_sec=repair_planning_sec,
            )
        )
        self._cortex_state = repaired_state
        return repaired_state, repair_result_state, repair_planning_sec

    def _run_cortex_planning(self, obs: Dict[str, Any], step_num: int) -> Optional[List[str]]:
        if self.dual_brain_controller is None or create_initial_state is None:
            return None

        recent_or_default = getattr(self, "_get_recent_or_default", None)
        if not callable(recent_or_default):
            recent_or_default = lambda key, default=None: default

        self._latest_obs = deepcopy(obs)

        latest_image = self._pick_latest_available_image_path(
            obs=obs,
            preferred_path=str((self._cortex_state or {}).get("screenshot_path", "") or ""),
            step_num=step_num,
        )

        sync_payload = self._sync_stardew_memory_from_obs(
            obs=obs,
            step_num=step_num,
            latest_image=latest_image,
        )

        if should_initialize_cortex_state(
            current_state=self._cortex_state,
            task_description=self.task_description,
        ):
            self._cortex_state = cast(Dict[str, Any], create_initial_state(
                frame_ids=(max(step_num - 1, 0), step_num),
                screenshot_path=latest_image,
                work_dir=config.work_dir,
                env_name=config.env_short_name,
                env_config=config.env_config,
            ))
            self._cortex_state["task"] = self.task_description
            self._cortex_state["main_task"] = self.task_description
            self._cortex_state["skill_library"] = self._skill_library_json
            self._cortex_state["previous_actions"] = []
            self._cortex_state["previous_results"] = []
            self._cortex_state["consecutive_failures"] = 0
            self._cortex_state["retry_count"] = 0
            self._cortex_state["is_first_step"] = True
            self._cortex_state["step_count"] = 0
            self._cortex_state["subtask_description"] = self.initial_subtask_description
            self._cortex_state["subtask_reasoning"] = self.initial_subtask_reasoning
            self._cortex_state["info_gathering_mode"] = "stardew_original"
            self._cortex_state["last_action"] = ""
            self._cortex_state["last_exec_info"] = {}
            self._cortex_state["latest_task_eval"] = {}
            self._cortex_state["task_progress"] = None
            self._cortex_state["previous_task_progress"] = None
            self._cortex_state["task_progress_quantity"] = None
            self._cortex_state["previous_task_progress_quantity"] = None
            self._cortex_state["task_progress_delta"] = None
            self._cortex_state["zero_progress_streak"] = 0
            self._cortex_state["repeated_action_streak"] = 0
            self._cortex_state["oscillation_streak"] = 0
            self._cortex_state["position_issue_detected"] = False
            self._cortex_state["latest_execution_summary"] = ""
            self._cortex_state["action_feedback"] = ""
            self._cortex_state["recent_execution_feedback"] = []
            self._cortex_state["last_state_changed"] = True
            self._cortex_state["last_inventory_changed"] = False
            self._cortex_state["last_menu_changed"] = False
            self._cortex_state["last_errors_info"] = ""
            self._cortex_state["failure_root_cause"] = ""
            self._cortex_state["failure_signature"] = ""
            self._cortex_state["required_change_type"] = ""
            self._cortex_state["has_execution_feedback"] = False
            self._cortex_state["execution_pending"] = False
            self._cortex_state["pending_action"] = ""
            self._cortex_state["pending_step_index"] = None
            self._cortex_state["pending_suggested_action"] = ""
            self._cortex_state["pending_local_recovery_action"] = ""
            self._cortex_state["pending_local_recovery_reason"] = ""
            self._cortex_state["force_big_brain_replan"] = False
            self._cortex_state["allow_suggestion_execution_fallback"] = False
            self._cortex_state["sanitized_subtask_hint"] = ""
            self._cortex_state["redundant_tool_selection"] = False
            self._cortex_state["selected_item_already_correct"] = False
            self._cortex_state["local_recovery_variation_toggle"] = 0
            self._cortex_state["planning_attempt_count"] = 0
            self._cortex_state["blocked_replan_count"] = 0
            self._cortex_state["no_execution_return_count"] = 0
            self._cortex_state["executed_step_count"] = 0
            self._cortex_state["recent_planning_sec_window"] = []
            self._cortex_state["last_planning_sec"] = None
            self._cortex_state["planning_sec_median"] = None
            self._cortex_state["recent_no_execution_planning_sec_window"] = []
            self._cortex_state["last_no_execution_planning_sec"] = None
            self._cortex_state["no_execution_planning_sec_median"] = None
            self._cortex_state["no_execution_planning_sample_count"] = 0
            self._cortex_state["watchdog_dynamic_timeout_sec"] = None
            self._cortex_state["same_step_elapsed_sec"] = 0.0
            self._cortex_state["same_step_no_execution_streak"] = 0
            self._cortex_state["same_step_blocked_since_ts"] = None
            self._cortex_state["last_no_execution_signature"] = ""
            self._cortex_state["last_no_execution_step"] = None
            self._cortex_state["last_blocker_signature"] = ""
            self._cortex_state["blocker_replan_streak"] = 0
            self._cortex_state["blocker_replan_only"] = False
            self._cortex_state["watchdog_triggered"] = False
            self._cortex_state["watchdog_reason"] = ""
            self._cortex_state["deadlock_signature"] = ""
            self._cortex_state["deadlock_reflection_cycles"] = 0
            self._cortex_state["uncertain_execution"] = False
            self._runtime_stop_signal = None

        state = cast(Dict[str, Any], self._cortex_state)
        self._cortex_state = state

        state["frame_ids"] = (max(step_num - 1, 0), step_num)
        state["screenshot_path"] = latest_image
        state["task"] = self.task_description
        state["main_task"] = self.task_description
        if isinstance(sync_payload, dict):
            state.update(sync_payload)
        state.setdefault("execution_pending", False)
        state.setdefault("pending_action", "")
        state.setdefault("pending_step_index", None)
        state.setdefault("pending_suggested_action", "")
        state.setdefault("pending_local_recovery_action", "")
        state.setdefault("pending_local_recovery_reason", "")
        state.setdefault("force_big_brain_replan", False)
        state.setdefault("allow_suggestion_execution_fallback", False)
        state.setdefault("sanitized_subtask_hint", "")
        state.setdefault("redundant_tool_selection", False)
        state.setdefault("selected_item_already_correct", False)
        state.setdefault("local_recovery_variation_toggle", 0)
        state.setdefault("action_feedback", "")
        state.setdefault("failure_root_cause", "")
        state.setdefault("failure_signature", "")
        state.setdefault("required_change_type", "")
        state.setdefault("planning_attempt_count", 0)
        state.setdefault("blocked_replan_count", 0)
        state.setdefault("no_execution_return_count", 0)
        state.setdefault("executed_step_count", 0)
        state.setdefault("recent_planning_sec_window", [])
        state.setdefault("last_planning_sec", None)
        state.setdefault("planning_sec_median", None)
        state.setdefault("recent_no_execution_planning_sec_window", [])
        state.setdefault("last_no_execution_planning_sec", None)
        state.setdefault("no_execution_planning_sec_median", None)
        state.setdefault("no_execution_planning_sample_count", 0)
        state.setdefault("watchdog_dynamic_timeout_sec", None)
        state.setdefault("same_step_elapsed_sec", 0.0)
        state.setdefault("same_step_no_execution_streak", 0)
        state.setdefault("same_step_blocked_since_ts", None)
        state.setdefault("last_no_execution_signature", "")
        state.setdefault("last_no_execution_step", None)
        state.setdefault("last_blocker_signature", "")
        state.setdefault("blocker_replan_streak", 0)
        state.setdefault("blocker_replan_only", False)
        state.setdefault("watchdog_triggered", False)
        state.setdefault("watchdog_reason", "")
        state.setdefault("deadlock_signature", "")
        state.setdefault("deadlock_reflection_cycles", 0)
        state.setdefault("uncertain_execution", False)
        state["gathered_info"] = self._obs_to_gathered_info(obs)
        state["is_first_step"] = should_treat_cortex_attempt_as_first_step(
            state=state,
            step_num=step_num,
        )
        state["step_count"] = step_num
        state["info_gathering_mode"] = "stardew_original"
        if state.get("last_no_execution_step", None) != step_num:
            state.update(
                reset_cortex_no_execution_watchdog(state=state)
            )
        current_subtask_values = self._resolve_current_subtask_values()
        latest_executed_action = str(
            state.get("last_action", "")
            or state.get("pre_action", "")
            or recent_or_default("pre_action", recent_or_default("action", ""))
            or ""
        ).strip()
        latest_exec_info = state.get("last_exec_info", {})
        if not isinstance(latest_exec_info, dict) or not latest_exec_info:
            fallback_exec_info = recent_or_default("exec_info", {})
            latest_exec_info = fallback_exec_info if isinstance(fallback_exec_info, dict) else {}
        previous_toolbar_information = state.get("previous_toolbar_information", None)
        if previous_toolbar_information in (None, "", []):
            previous_toolbar_information = recent_or_default(
                "previous_toolbar_information",
                recent_or_default("toolbar_information", []),
            )

        if self.cortex_memory is not None and self.cortex_memory is not self.memory and c_constants is not None:
            self.cortex_memory.update_info_history({
                "screenshot_path": latest_image,
                c_constants.IMAGES_MEM_BUCKET: latest_image,
                "task_description": self.task_description,
                "toolbar_information": self._format_toolbar_text(obs.get("inventory", []), obs.get("chosen_item", None)),
                "selected_position": self._infer_selected_position(obs.get("inventory", []), obs.get("chosen_item", None)),
                "surroundings": self._format_surroundings_text(obs.get("surroundings", ""), obs.get("position", "")),
                "inventory": obs.get("inventory", []),
                "chosen_item": obs.get("chosen_item", None),
                "start_frame_id": max(step_num - 1, 0),
                "end_frame_id": step_num,
                "skill_library": self._skill_library_json,
                "datetime": obs.get("time", ""),
                "pre_action": latest_executed_action,
                "action": latest_executed_action,
                "pre_decision_making_reasoning": recent_or_default("action_planning_reasoning", ""),
                "exec_info": latest_exec_info,
                "history_summary": recent_or_default("history_summary", recent_or_default("summarization", "")),
                "previous_toolbar_information": previous_toolbar_information,
                "pre_energy": recent_or_default("energy", None),
                "pre_money": recent_or_default("money", None),
                "pre_health": recent_or_default("health", None),
                "subtask_description": current_subtask_values.get("subtask_description", self.initial_subtask_description),
                "subtask_reasoning": current_subtask_values.get("subtask_reasoning", self.initial_subtask_reasoning),
                "sanitized_subtask_hint": state.get("sanitized_subtask_hint", ""),
                "redundant_tool_selection": state.get("redundant_tool_selection", False),
                "selected_item_already_correct": state.get("selected_item_already_correct", False),
                "image_introduction": [],
                "gathered_info": self._obs_to_gathered_info(obs),
                "info_gathering_mode": "stardew_original",
                **self.task_acquisition_context,
            })

        state["planning_attempt_count"] = int(
            state.get("planning_attempt_count", 0) or 0
        ) + 1
        if hasattr(self, "_prepare_big_brain_template_for_call"):
            self._prepare_big_brain_template_for_call("action_planning")
        elif hasattr(self, "_ensure_big_brain_template_integrity"):
            self._ensure_big_brain_template_integrity("action_planning")
        planning_started_at = time.time()
        current_state = cast(Dict[str, Any], state)
        result_state = cast(Dict[str, Any], self.dual_brain_controller.step(current_state, self._cortex_workflow_config))
        planning_duration_sec = max(0.0, time.time() - planning_started_at)
        state = {**state, **result_state}
        self._cortex_state = state
        state.update(
            record_cortex_planning_latency(
                state=state,
                planning_sec=planning_duration_sec,
            )
        )
        current_subtask_values = self._resolve_current_subtask_values()

        action_resolution = self._resolve_cortex_result_actions(result_state)
        normalized_actions = list(action_resolution.get("actions", []))
        normalized_suggestion_actions = list(action_resolution.get("suggestion_actions", []))
        execution_source = str(action_resolution.get("execution_source", "") or "")
        blocked_reason = str(action_resolution.get("blocked_reason", "") or "")

        if normalized_actions and normalized_suggestion_actions and execution_source in {"planned_actions", "pending_action"}:
            suggested_action_for_log = self._select_cortex_suggestion_for_logging(
                result_state=result_state,
                normalized_actions=normalized_actions,
                normalized_suggestion_actions=normalized_suggestion_actions,
            )
            if suggested_action_for_log and normalized_actions[0] != suggested_action_for_log:
                logger.write(
                    "[Cortex] LittleBrain overrode BigBrain suggestion: "
                    f"{normalized_actions[0]} (was: {suggested_action_for_log})"
                )
        elif normalized_suggestion_actions and action_resolution.get("used_suggestion_fallback", False):
            logger.warn(
                "[Cortex] Executing a validated current-step suggestion fallback"
            )
        elif normalized_suggestion_actions and blocked_reason:
            logger.warn(
                f"[Cortex] Suggestion execution blocked: {blocked_reason}"
            )
        elif blocked_reason:
            logger.warn(f"[Cortex] No executable action produced: {blocked_reason}")

        repair_attempted_this_turn = False
        while normalized_actions:
            validation = validate_runtime_pre_execution_action(
                state=self._cortex_state,
                action_text=normalized_actions[0],
            )
            if bool(validation.get("is_valid", True)):
                blocked_reason = ""
                break

            rejected_action = normalized_actions[0]
            fallback_action = self._normalize_cortex_action(
                str(validation.get("fallback_action", "") or "")
            )
            if fallback_action and fallback_action != rejected_action:
                fallback_validation = validate_runtime_pre_execution_action(
                    state=self._cortex_state,
                    action_text=fallback_action,
                )
                if bool(fallback_validation.get("is_valid", True)):
                    logger.warn(
                        "[Cortex] Pre-execution validation rewrote action: "
                        f"{rejected_action} -> {fallback_action}"
                    )
                    normalized_actions = [fallback_action]
                    blocked_reason = ""
                    break
                validation = fallback_validation

            blocked_reason = str(
                validation.get("invalid_reason", "") or "runtime_pre_execution_validation"
            )

            if (
                not repair_attempted_this_turn
                and self._big_brain_only_repair_enabled(self._cortex_state)
            ):
                repair_attempted_this_turn = True
                state, repair_result_state, repair_planning_sec = self._run_big_brain_only_repair_replan(
                    state=self._cortex_state,
                    rejected_action=rejected_action,
                    validation=validation,
                    blocked_reason=blocked_reason,
                )
                planning_duration_sec += repair_planning_sec
                current_subtask_values = self._resolve_current_subtask_values()
                repair_resolution = self._resolve_cortex_result_actions(repair_result_state)
                normalized_actions = list(repair_resolution.get("actions", []))
                normalized_suggestion_actions = list(repair_resolution.get("suggestion_actions", []))
                execution_source = str(repair_resolution.get("execution_source", "") or "")
                blocked_reason = str(repair_resolution.get("blocked_reason", "") or "")
                if normalized_actions:
                    logger.write(
                        "[Cortex] BigBrain-only repair produced candidate action: "
                        f"{normalized_actions[0]}"
                    )
                    continue
                logger.warn(
                    "[Cortex] BigBrain-only repair produced no executable action"
                )
                break

            failure_root_cause = self._normalize_cultivation_failure_root_cause(
                validation.get("failure_root_cause", "")
            )
            required_change_type = self._normalize_cultivation_required_change_type(
                validation.get("required_change_type", "")
            )
            failure_signature = self._build_cultivation_failure_signature(
                task_kind=self._infer_cultivation_task_kind(
                    self._cortex_state.get("main_task", "") or self.task_description
                ),
                action_text=rejected_action,
                failure_root_cause=failure_root_cause,
                required_change_type=required_change_type,
                blocker_signature=str(
                    self._cortex_state.get("current_blocker_signature", "") or ""
                ).strip(),
            )
            self._cortex_state.update({
                "failure_root_cause": failure_root_cause,
                "required_change_type": required_change_type,
                "failure_signature": failure_signature,
                "action_feedback": (
                    f"Pre-execution validation rejected action {rejected_action}: "
                    f"{blocked_reason}. Failure root cause: {failure_root_cause}. "
                    f"Required change type: {required_change_type}. "
                    f"Failure signature: {failure_signature}."
                ).strip(),
                "force_big_brain_replan": True,
                "brain_mode": "big",
                "execution_pending": False,
                "pending_action": "",
                "pending_step_index": None,
                "pending_suggested_action": "",
            })
            logger.warn(
                "[Cortex] Cultivation pre-execution validation blocked action: "
                f"{rejected_action} | reason={blocked_reason}"
            )
            normalized_actions = []
            break

        if not normalized_actions:
            no_execution_updates = record_cortex_no_execution(
                state=self._cortex_state,
                step_num=step_num,
                blocked_reason=blocked_reason,
                screenshot_path=latest_image or self._cortex_state.get("screenshot_path", ""),
                subtask_description=current_subtask_values.get("subtask_description", ""),
                planning_sec=planning_duration_sec,
            )
            self._cortex_state.update(no_execution_updates)
            if self._cortex_state.get("blocker_replan_only", False):
                logger.warn(
                    "[Cortex] Repeated same-step grounded blocker detected; "
                    "using blocker-aware replan-only mode on the next planning cycle"
                )
            if self._cortex_state.get("watchdog_triggered", False):
                watchdog_reason = str(self._cortex_state.get("watchdog_reason", "") or "cortex_no_execution_watchdog")
                warning = (
                    "cortex runtime watchdog stopped the task after repeated planning-only livelock "
                    f"({watchdog_reason}; planning_sec={planning_duration_sec:.1f}s; "
                    f"no_execution_median={self._cortex_state.get('no_execution_planning_sec_median')}; "
                    f"dynamic_timeout={self._cortex_state.get('watchdog_dynamic_timeout_sec')})"
                )
                self._runtime_stop_signal = {
                    "runtime_exit_reason": "cortex_no_execution_watchdog",
                    "warning": warning,
                    **self.get_runtime_task_metrics(),
                }
                logger.warn(f"[Cortex] {warning}")
            return []

        self._cortex_state.update(
            reset_cortex_no_execution_watchdog(state=self._cortex_state)
        )
        self._runtime_stop_signal = None
        return normalized_actions[:config.number_of_execute_skills]

    def _build_lightweight_execution_summary(
        self,
        exec_info: Dict[str, Any],
        action: Any = None,
        obs: Optional[Dict[str, Any]] = None,
        task_eval: Optional[Dict[str, Any]] = None,
    ) -> str:
        if not isinstance(exec_info, dict) or exec_info.get("errors", False) or not isinstance(obs, dict):
            return ""

        chosen_item = obs.get("chosen_item", None)
        selected_item_name = self._extract_item_name(chosen_item)
        selected_position = self._infer_selected_position(obs.get("inventory", []), chosen_item)
        last_skill = str(
            exec_info.get("last_skill")
            or (action[0] if isinstance(action, list) and action else action)
            or ""
        ).strip()

        task_text = str(
            self.task_description
            or (self._cortex_state or {}).get("main_task", "")
            or (self._cortex_state or {}).get("task", "")
        ).strip()
        location = str(obs.get("location") or "").strip()
        day_text = str(obs.get("day") or "").strip()
        time_text = str(obs.get("time") or "").strip()

        progress_quantity = None
        completed = None
        if isinstance(task_eval, dict):
            progress_quantity = task_eval.get("quantity")
            if progress_quantity is None:
                progress_quantity = task_eval.get("current_quantity")
            if progress_quantity is None:
                progress_quantity = task_eval.get("final_quantity")
            if "completed" in task_eval:
                completed = bool(task_eval.get("completed", False))

        parts: List[str] = []
        if day_text and time_text:
            parts.append(f"On {day_text} at {time_text},")
        elif time_text:
            parts.append(f"At {time_text},")

        if task_text:
            parts.append(f"the current task is {task_text}.")
        if location:
            parts.append(f"I am at {location}.")
        if selected_item_name:
            if selected_position is not None:
                parts.append(f"The {selected_item_name} is currently selected in slot {selected_position}.")
            else:
                parts.append(f"The {selected_item_name} is currently selected.")
        if last_skill:
            if execution_has_no_confirmation(exec_info):
                if selected_item_name:
                    parts.append(
                        f"The last executed action {last_skill} returned no explicit confirmation, "
                        f"but the current observation shows {selected_item_name} selected."
                    )
                else:
                    parts.append(
                        f"The last executed action {last_skill} returned no explicit confirmation."
                    )
            else:
                parts.append(f"The last executed action {last_skill} succeeded.")
        if progress_quantity is not None:
            parts.append(f"Recorded task progress is {progress_quantity}.")
        if completed is True:
            parts.append("The task is completed.")
        elif completed is False and progress_quantity is not None:
            parts.append("The task is not completed yet.")

        return " ".join(part for part in parts if part).strip()

    def _sync_external_little_brain_feedback(
        self,
        *,
        action_text: str,
        exec_info: Dict[str, Any],
        success: Optional[bool] = None,
        state_changed: Optional[bool] = None,
        uncertain_execution: Optional[bool] = None,
        heightened_failure_signal: Optional[bool] = None,
        progress_delta: Any = None,
        progress_quantity: Any = None,
    ) -> None:
        if self._cortex_state is None or self.dual_brain_controller is None:
            return

        little_brain = getattr(self.dual_brain_controller, "little_brain", None)
        if little_brain is None or getattr(little_brain, "execute_internally", True):
            return

        if not hasattr(little_brain, "record_external_execution_feedback"):
            return

        errors_info_text = ""
        if isinstance(exec_info, dict):
            errors_info_text = str(exec_info.get("errors_info", "") or "").strip()

        pending_action = self._normalize_action_text(
            self._cortex_state.get("pending_action", "")
        )
        feedback_action = action_text or pending_action
        if not feedback_action:
            return

        pending_step_index = self._cortex_state.get("pending_step_index", None)
        try:
            executed_step_index = int(cast(Any, pending_step_index))
        except (TypeError, ValueError):
            try:
                current_step = int(self._cortex_state.get("current_step", 0) or 0)
            except (TypeError, ValueError):
                current_step = 0
            executed_step_index = max(current_step - 1, 0)

        suggested_action = str(
            self._cortex_state.get("pending_suggested_action", "") or ""
        ).strip()
        suggestions = self._cortex_state.get("suggestions", [])
        if not suggested_action and isinstance(suggestions, list) and executed_step_index < len(suggestions):
            suggestion = suggestions[executed_step_index]
            if isinstance(suggestion, dict):
                suggested_action = str(suggestion.get("action", "") or "").strip()
            elif suggestion:
                suggested_action = str(suggestion).strip()

        feedback_payload = {
            "action": feedback_action,
            "success": bool(success) if success is not None else not bool(
                isinstance(exec_info, dict) and exec_info.get("errors", False)
            ),
            "errors_info": errors_info_text,
            "step": executed_step_index,
            "suggested_action": suggested_action,
        }
        if state_changed is not None:
            feedback_payload["state_changed"] = bool(state_changed)
        if uncertain_execution is not None:
            feedback_payload["uncertain_execution"] = bool(uncertain_execution)
        if heightened_failure_signal is not None:
            feedback_payload["heightened_failure_signal"] = bool(heightened_failure_signal)
        if progress_delta is not None:
            feedback_payload["progress_delta"] = progress_delta
        if progress_quantity is not None:
            feedback_payload["progress_quantity"] = progress_quantity

        little_brain.record_external_execution_feedback(**feedback_payload)
        self._cortex_state["execution_log"] = list(little_brain.execution_log)

    def update_execution_feedback(
        self,
        exec_info: Dict[str, Any],
        action: Any = None,
        obs: Optional[Dict[str, Any]] = None,
        task_eval: Optional[Dict[str, Any]] = None,
    ):
        if self._cortex_state is None:
            return

        action_text = self._normalize_action_text(action)
        previous_gathered_info = (
            self._cortex_state.get("gathered_info", {})
            if isinstance(self._cortex_state.get("gathered_info", {}), dict)
            else {}
        )
        previous_position = self._normalize_position(previous_gathered_info.get("position"))
        previous_selected_item_name = self._extract_item_name(
            previous_gathered_info.get("chosen_item") or previous_gathered_info.get("selected_item_name")
        )
        previous_selected_position = previous_gathered_info.get(
            "selected_position",
            self._cortex_state.get("selected_position", None),
        )
        previous_menu_snapshot = stable_snapshot_text(previous_gathered_info.get("current_menu"))
        previous_inventory_snapshot = stable_snapshot_text(previous_gathered_info.get("inventory"))
        previous_toolbar_snapshot = stable_snapshot_text(
            previous_gathered_info.get("toolbar_information")
            or self._cortex_state.get("toolbar_information", "")
        )
        previous_progress_quantity = self._cortex_state.get("task_progress_quantity", None)

        prev_actions = list(self._cortex_state.get("previous_actions", []))
        if action_text:
            prev_actions.append(action_text)
        self._cortex_state["previous_actions"] = prev_actions[-config.max_recent_steps:]

        failed = bool(isinstance(exec_info, dict) and exec_info.get("errors", False))
        errors_info_text = ""
        if isinstance(exec_info, dict):
            errors_info_text = str(exec_info.get("errors_info", "") or "").strip()
        little_brain = (
            getattr(self.dual_brain_controller, "little_brain", None)
            if self.dual_brain_controller is not None
            else None
        )
        external_pending = bool(
            little_brain is not None
            and not getattr(little_brain, "execute_internally", True)
            and (
                self._cortex_state.get("execution_pending", False)
                or self._cortex_state.get("pending_action")
            )
        )
        pending_step_index = self._cortex_state.get("pending_step_index", None)
        try:
            executed_step_index = int(cast(Any, pending_step_index))
        except (TypeError, ValueError):
            try:
                current_step_index = int(self._cortex_state.get("current_step", 0) or 0)
            except (TypeError, ValueError):
                current_step_index = 0
            executed_step_index = max(current_step_index - 1, 0)
        current_failures = int(self._cortex_state.get("consecutive_failures", 0))

        latest_gathered_info = None
        latest_toolbar_information = None
        latest_selected_position = None
        latest_selected_item_name = ""
        latest_summary = ""
        latest_execution_summary = ""
        current_progress_quantity = self._extract_progress_quantity(task_eval)
        progress_delta = None
        if (
            isinstance(previous_progress_quantity, (int, float))
            and current_progress_quantity is not None
        ):
            progress_delta = float(current_progress_quantity) - float(previous_progress_quantity)
        completed = bool(task_eval.get("completed", False)) if isinstance(task_eval, dict) and "completed" in task_eval else None
        productive_action = self._is_productive_action(action_text)
        position_changed = False
        selected_item_changed = False
        menu_changed = False
        inventory_changed = False
        toolbar_changed = False
        observation_confirmation_changed = False
        current_subtask_values = self._resolve_current_subtask_values()

        if obs is not None:
            latest_gathered_info = self._obs_to_gathered_info(obs)
            latest_toolbar_information = self._format_toolbar_text(obs.get("inventory", []), obs.get("chosen_item", None))
            latest_selected_position = self._infer_selected_position(obs.get("inventory", []), obs.get("chosen_item", None))
            latest_selected_item_name = self._extract_item_name(obs.get("chosen_item", None))
            latest_prompt_fact_fields = extract_stardew_prompt_fact_fields(
                state={
                    **self._cortex_state,
                    "task": self.task_description,
                    "main_task": self.task_description,
                    "surroundings": latest_gathered_info.get("surroundings", ""),
                    "position": obs.get("position", ""),
                    "current_position": obs.get("position", ""),
                    "facing_direction": obs.get("facing_direction", ""),
                    "facing_position": self._infer_facing_position(
                        obs.get("position", ""),
                        obs.get("facing_direction", ""),
                    ),
                    "selected_position": latest_selected_position,
                    "selected_item_name": latest_selected_item_name,
                    "toolbar_information": latest_toolbar_information,
                    "current_menu": obs.get("current_menu", None),
                    "subtask_description": current_subtask_values.get(
                        "subtask_description",
                        self.initial_subtask_description,
                    ),
                    "subtask_reasoning": current_subtask_values.get(
                        "subtask_reasoning",
                        self.initial_subtask_reasoning,
                    ),
                },
                gathered_info=latest_gathered_info,
            )
            latest_gathered_info.update(
                {
                    "front_tile_summary": latest_prompt_fact_fields.get(
                        "front_tile_summary",
                        "(none)",
                    ),
                    "blocked_recovery_hint": latest_prompt_fact_fields.get(
                        "blocked_recovery_hint",
                        "",
                    ),
                    "current_blocker_signature": latest_prompt_fact_fields.get(
                        "current_blocker_signature",
                        "(none)",
                    ),
                    "nearest_grounded_target_summary": latest_prompt_fact_fields.get(
                        "nearest_grounded_target_summary",
                        "(none)",
                    ),
                }
            )
            self._cortex_state["gathered_info"] = latest_gathered_info
            self._cortex_state["toolbar_information"] = latest_toolbar_information
            self._cortex_state["selected_position"] = latest_selected_position
            self._cortex_state["selected_item_name"] = latest_selected_item_name
            self._cortex_state["chosen_item"] = obs.get("chosen_item", None)
            self._cortex_state["location"] = obs.get("location", "")
            self._cortex_state["position"] = obs.get("position", "")
            self._cortex_state["current_menu"] = obs.get("current_menu", None)
            current_position = self._normalize_position(latest_gathered_info.get("position"))
            current_menu_snapshot = stable_snapshot_text(latest_gathered_info.get("current_menu"))
            current_inventory_snapshot = stable_snapshot_text(latest_gathered_info.get("inventory"))
            current_toolbar_snapshot = stable_snapshot_text(latest_toolbar_information)
            position_changed = previous_position is not None and current_position is not None and previous_position != current_position
            selected_item_changed = bool(
                (
                    latest_selected_item_name != previous_selected_item_name
                    and (
                        bool(latest_selected_item_name)
                        or bool(previous_selected_item_name)
                    )
                )
                or (
                    latest_selected_position is not None
                    and latest_selected_position != previous_selected_position
                )
            )
            menu_changed = previous_menu_snapshot != current_menu_snapshot
            inventory_changed = previous_inventory_snapshot != current_inventory_snapshot
            toolbar_changed = previous_toolbar_snapshot != current_toolbar_snapshot
            observation_confirmation_changed = execution_observation_confirms_change(
                previous_gathered_info,
                latest_gathered_info,
            )
            if toolbar_changed:
                selected_item_changed = True

        subtask_hints = {
            "sanitized_subtask_hint": "",
            "redundant_tool_selection": False,
            "selected_item_already_correct": False,
        }
        if obs is not None:
            subtask_hints = self._sanitize_subtask_memory(
                current_subtask_values.get("subtask_description", ""),
                current_subtask_values.get("subtask_reasoning", ""),
                obs.get("chosen_item", None),
            )
            self._cortex_state["sanitized_subtask_hint"] = subtask_hints.get("sanitized_subtask_hint", "")
            self._cortex_state["redundant_tool_selection"] = bool(subtask_hints.get("redundant_tool_selection", False))
            self._cortex_state["selected_item_already_correct"] = bool(subtask_hints.get("selected_item_already_correct", False))

        progress_made = bool(progress_delta is not None and progress_delta > 0)
        state_changed = bool(
            completed is True
            or progress_made
            or position_changed
            or selected_item_changed
            or inventory_changed
            or menu_changed
            or observation_confirmation_changed
        )
        confirmation_sensitive_action = action_text.startswith(
            (
                "choose_item(",
                "attach_item(",
                "unattach_item(",
                "use(",
                "interact(",
                "choose_option(",
                "menu(",
            )
        )
        missing_confirmation = execution_has_no_confirmation(exec_info)
        prev_results = list(self._cortex_state.get("previous_results", []))
        previous_result = prev_results[-1] if prev_results else {}
        previous_action_text = str(previous_result.get("action", "") or "").strip()
        same_action_as_previous = bool(action_text and previous_action_text == action_text)

        previous_progress_quantity_in_result = previous_result.get("progress_quantity")
        if isinstance(current_progress_quantity, (int, float)) and isinstance(
            previous_progress_quantity_in_result, (int, float)
        ):
            quantity_unchanged = float(previous_progress_quantity_in_result) == float(
                current_progress_quantity
            )
        else:
            quantity_unchanged = (
                previous_result.get("progress_delta") in (None, "", 0, 0.0)
                and progress_delta in (None, "", 0, 0.0)
            )

        uncertain_execution = bool(
            confirmation_sensitive_action
            and missing_confirmation
            and not state_changed
            and not progress_made
        )
        repeated_confirmation_gap = bool(
            confirmation_sensitive_action
            and missing_confirmation
            and not state_changed
            and same_action_as_previous
            and quantity_unchanged
            and bool(previous_result.get("uncertain_execution", False))
        )
        repeated_interaction_no_confirmation = bool(
            action_text.startswith(("use(", "interact(", "choose_option("))
            and missing_confirmation
            and not state_changed
            and not inventory_changed
            and same_action_as_previous
            and quantity_unchanged
            and not bool(previous_result.get("state_changed", False))
            and not bool(previous_result.get("inventory_changed", False))
        )
        repeated_menu_no_change = bool(
            action_text.startswith("menu(")
            and not state_changed
            and not menu_changed
            and same_action_as_previous
            and not bool(previous_result.get("state_changed", False))
            and not bool(previous_result.get("menu_changed", False))
        )
        heightened_failure_signal = (
            repeated_confirmation_gap
            or repeated_interaction_no_confirmation
            or repeated_menu_no_change
        )
        pending_local_recovery_action = ""
        pending_local_recovery_reason = ""
        if repeated_interaction_no_confirmation and action_text:
            try:
                local_recovery_variation_seed = int(
                    self._cortex_state.get("local_recovery_variation_toggle", 0) or 0
                ) + 1
            except (TypeError, ValueError):
                local_recovery_variation_seed = 1
            pending_local_recovery_action = build_runtime_local_recovery_action(
                result_state=self._cortex_state,
                suggestion_action=str(self._cortex_state.get("pending_suggested_action", "") or ""),
                failed_action=action_text,
                decision_reason="runtime_repeated_interaction_local_recovery",
                variation_seed=local_recovery_variation_seed,
            )
            if pending_local_recovery_action:
                pending_local_recovery_reason = "repeated_interaction_no_confirmation"
                self._cortex_state["local_recovery_variation_toggle"] = local_recovery_variation_seed
        recoverable_confirmation_gap = bool(
            uncertain_execution and not heightened_failure_signal
        )
        zero_progress_streak = int(self._cortex_state.get("zero_progress_streak", 0))
        if productive_action and not failed and not state_changed and not recoverable_confirmation_gap:
            zero_progress_streak += 1
        elif state_changed:
            zero_progress_streak = 0

        repeated_action_streak = self._count_same_action_tail(
            self._cortex_state.get("previous_actions", []),
            action_text,
        )
        oscillation_streak = self._detect_oscillation(
            self._cortex_state.get("previous_actions", [])
        )
        position_issue_detected = self._detect_position_issue(
            current_subtask_values.get("subtask_description", self.initial_subtask_description),
            current_subtask_values.get("subtask_reasoning", self.initial_subtask_reasoning),
            self._get_recent_or_default("history_summary", ""),
        )

        meaningful_failure = bool(
            failed
            or heightened_failure_signal
            or (productive_action and not state_changed and not recoverable_confirmation_gap)
        )
        failure_increment = 2 if heightened_failure_signal else (1 if meaningful_failure else 0)
        self._cortex_state["consecutive_failures"] = (
            current_failures + failure_increment if failure_increment else 0
        )
        self._cortex_state["success"] = not meaningful_failure
        self._cortex_state["execution_result"] = {
            "success": not meaningful_failure,
            "error": None if not meaningful_failure else "execution_failed",
            "pending": False,
            "exec_info": exec_info if isinstance(exec_info, dict) else {},
            "frame_ids": self._cortex_state.get("frame_ids", (0, 0)),
        }
        self._cortex_state["pre_action"] = action_text
        self._cortex_state["action"] = action_text
        self._cortex_state["last_action"] = action_text
        self._cortex_state["last_exec_info"] = exec_info if isinstance(exec_info, dict) else {}
        self._cortex_state["latest_task_eval"] = task_eval if isinstance(task_eval, dict) else {}
        self._cortex_state["previous_task_progress"] = previous_progress_quantity
        self._cortex_state["task_progress"] = current_progress_quantity
        self._cortex_state["previous_task_progress_quantity"] = previous_progress_quantity
        self._cortex_state["task_progress_quantity"] = current_progress_quantity
        self._cortex_state["task_progress_delta"] = progress_delta
        self._cortex_state["zero_progress_streak"] = zero_progress_streak
        self._cortex_state["repeated_action_streak"] = repeated_action_streak
        self._cortex_state["oscillation_streak"] = oscillation_streak
        self._cortex_state["position_issue_detected"] = position_issue_detected
        self._cortex_state["last_state_changed"] = state_changed
        self._cortex_state["last_inventory_changed"] = inventory_changed
        self._cortex_state["last_menu_changed"] = menu_changed
        self._cortex_state["last_toolbar_changed"] = toolbar_changed
        self._cortex_state["previous_toolbar_information"] = previous_toolbar_snapshot
        self._cortex_state["last_observation_confirmation_changed"] = observation_confirmation_changed
        self._cortex_state["last_errors_info"] = errors_info_text
        self._cortex_state["has_execution_feedback"] = True
        self._cortex_state["uncertain_execution"] = uncertain_execution
        self._cortex_state["heightened_failure_signal"] = heightened_failure_signal
        self._cortex_state["pending_local_recovery_action"] = pending_local_recovery_action
        self._cortex_state["pending_local_recovery_reason"] = pending_local_recovery_reason
        self._sync_external_little_brain_feedback(
            action_text=action_text,
            exec_info=exec_info if isinstance(exec_info, dict) else {},
            success=not meaningful_failure,
            state_changed=state_changed,
            uncertain_execution=uncertain_execution,
            heightened_failure_signal=heightened_failure_signal,
            progress_delta=progress_delta,
            progress_quantity=current_progress_quantity,
        )
        if external_pending:
            self._cortex_state["execution_pending"] = False
            self._cortex_state["pending_action"] = ""
            self._cortex_state["pending_step_index"] = None
            self._cortex_state["pending_suggested_action"] = ""
            self._cortex_state["planned_actions"] = []

            suggestions = self._cortex_state.get("suggestions", [])
            try:
                current_step_index = int(self._cortex_state.get("current_step", 0) or 0)
            except (TypeError, ValueError):
                current_step_index = 0
            has_remaining_suggestions = (
                isinstance(suggestions, list)
                and current_step_index < len(suggestions)
            )

            if pending_local_recovery_action:
                self._cortex_state["current_step"] = max(executed_step_index, 0)
                self._cortex_state["completed_steps"] = list(range(max(executed_step_index, 0)))
                self._cortex_state["force_big_brain_replan"] = False
                self._cortex_state["brain_mode"] = "little"
                self._cortex_state["escalation_reason"] = ""
            elif meaningful_failure or position_issue_detected or oscillation_streak >= 2:
                self._cortex_state["current_step"] = max(executed_step_index, 0)
                self._cortex_state["completed_steps"] = list(range(max(executed_step_index, 0)))
                self._cortex_state["force_big_brain_replan"] = True
                self._cortex_state["brain_mode"] = "big"
                self._cortex_state["escalation_reason"] = "external_execution_feedback"
            else:
                self._cortex_state["completed_steps"] = list(range(max(executed_step_index + 1, 0)))
                self._cortex_state["force_big_brain_replan"] = False
                if has_remaining_suggestions:
                    self._cortex_state["brain_mode"] = "little"
                    self._cortex_state["escalation_reason"] = ""
                else:
                    self._cortex_state["brain_mode"] = "big"
                    self._cortex_state["escalation_reason"] = "cycle_complete"
        if action_text:
            self._cortex_state["executed_step_count"] = int(
                self._cortex_state.get("executed_step_count", 0) or 0
            ) + 1
        self._cortex_state.update(
            reset_cortex_no_execution_watchdog(state=self._cortex_state)
        )
        self._runtime_stop_signal = None
        if uncertain_execution:
            logger.warn(
                f"[Cortex] Confirmation-sensitive action lacked confirmation without state change: {action_text}"
            )
        if heightened_failure_signal:
            logger.warn(
                "[Cortex] Escalating repeated no-confirmation pattern: "
                f"{action_text} | repeated_confirmation_gap={repeated_confirmation_gap}, "
                f"repeated_interaction_no_confirmation={repeated_interaction_no_confirmation}, "
                f"repeated_menu_no_change={repeated_menu_no_change}"
            )

        prev_results.append({
            "action": action_text,
            "success": not meaningful_failure,
            "execution_success_raw": infer_execution_success_raw(exec_info),
            "state_changed": state_changed,
            "inventory_changed": inventory_changed,
            "toolbar_changed": toolbar_changed,
            "completed": completed,
            "progress_delta": progress_delta,
            "progress_quantity": current_progress_quantity,
            "position_issue_detected": position_issue_detected,
            "uncertain_execution": uncertain_execution,
            "heightened_failure_signal": heightened_failure_signal,
            "observation_confirmation_changed": observation_confirmation_changed,
            "errors": failed,
            "errors_info": errors_info_text,
            "executed_skills": (
                list(exec_info.get("executed_skills", []))
                if isinstance(exec_info, dict) and isinstance(exec_info.get("executed_skills", []), list)
                else []
            ),
            "last_skill": (
                str(exec_info.get("last_skill", "") or "")
                if isinstance(exec_info, dict)
                else ""
            ),
            "exec_info": exec_info if isinstance(exec_info, dict) else {},
        })
        self._cortex_state["previous_results"] = prev_results[-config.max_recent_steps:]

        existing_history_summary = str(
            self._cortex_state.get("history_summary", "")
            or self._cortex_state.get("summarization", "")
            or ""
        ).strip()

        latest_summary = ""
        if not failed:
            latest_summary = self._build_lightweight_execution_summary(
                exec_info=exec_info,
                action=action,
                obs=obs,
                task_eval=task_eval,
            )
            if latest_summary and not existing_history_summary:
                mutable_state = cast(Dict[str, Any], self._cortex_state)
                mutable_state["summarization"] = latest_summary
                mutable_state["history_summary"] = latest_summary

        latest_execution_summary = self._build_task_progress_summary(
            action_text=action_text,
            previous_progress=(
                float(previous_progress_quantity)
                if isinstance(previous_progress_quantity, (int, float))
                else None
            ),
            current_progress=current_progress_quantity,
            progress_delta=progress_delta,
            zero_progress_streak=zero_progress_streak,
            repeated_action_streak=repeated_action_streak,
            productive_action=productive_action,
            state_changed=state_changed,
            position_issue_detected=position_issue_detected,
            completed=completed,
            errors_info=errors_info_text,
            oscillation_streak=oscillation_streak,
            missing_confirmation=execution_has_no_confirmation(exec_info),
        )
        cultivation_failure = self._classify_cultivation_failure(
            task_text=str(
                self.task_description
                or self._cortex_state.get("main_task", "")
                or self._cortex_state.get("task", "")
                or ""
            ),
            action_text=action_text,
            state_changed=state_changed,
            menu_changed=menu_changed,
            meaningful_failure=meaningful_failure,
            position_issue_detected=position_issue_detected,
            errors_info_text=errors_info_text,
        )
        failure_root_cause = cultivation_failure.get("failure_root_cause", "")
        required_change_type = cultivation_failure.get("required_change_type", "")
        failure_signature = cultivation_failure.get("failure_signature", "")
        action_feedback = self._build_action_feedback_text(
            latest_execution_summary=latest_execution_summary,
            failure_root_cause=failure_root_cause,
            required_change_type=required_change_type,
            failure_signature=failure_signature,
        )
        self._cortex_state["latest_execution_summary"] = latest_execution_summary
        self._cortex_state["action_feedback"] = action_feedback
        self._cortex_state["failure_root_cause"] = failure_root_cause
        self._cortex_state["required_change_type"] = required_change_type
        self._cortex_state["failure_signature"] = failure_signature
        recent_feedback = list(self._cortex_state.get("recent_execution_feedback", []))
        recent_feedback.append({
            "action": action_text,
            "state_changed": state_changed,
            "inventory_changed": inventory_changed,
            "toolbar_changed": toolbar_changed,
            "menu_changed": menu_changed,
            "progress_quantity": current_progress_quantity,
            "progress_delta": progress_delta,
            "zero_progress_streak": zero_progress_streak,
            "repeated_action_streak": repeated_action_streak,
            "position_issue_detected": position_issue_detected,
            "heightened_failure_signal": heightened_failure_signal,
            "observation_confirmation_changed": observation_confirmation_changed,
            "summary": latest_execution_summary,
            "action_feedback": action_feedback,
            "failure_root_cause": failure_root_cause,
            "required_change_type": required_change_type,
            "failure_signature": failure_signature,
            "errors_info": errors_info_text,
        })
        self._cortex_state["recent_execution_feedback"] = recent_feedback[-config.max_recent_steps:]

        memory_payload = {
            "execution_result": self._cortex_state.get("execution_result", {}),
            "exec_info": exec_info,
            "pre_action": action_text,
            "action": action_text,
            "previous_actions": self._cortex_state.get("previous_actions", []),
            "previous_results": self._cortex_state.get("previous_results", []),
            "last_action": action_text,
            "last_exec_info": self._cortex_state.get("last_exec_info", {}),
            "latest_task_eval": self._cortex_state.get("latest_task_eval", {}),
            "previous_task_progress": previous_progress_quantity,
            "task_progress": current_progress_quantity,
            "previous_task_progress_quantity": previous_progress_quantity,
            "task_progress_quantity": current_progress_quantity,
            "task_progress_delta": progress_delta,
            "zero_progress_streak": zero_progress_streak,
            "repeated_action_streak": repeated_action_streak,
            "position_issue_detected": position_issue_detected,
            "oscillation_streak": oscillation_streak,
            "last_state_changed": state_changed,
            "last_inventory_changed": inventory_changed,
            "last_menu_changed": menu_changed,
            "previous_toolbar_information": previous_toolbar_snapshot,
            "latest_execution_summary": latest_execution_summary,
            "action_feedback": action_feedback,
            "recent_execution_feedback": self._cortex_state.get("recent_execution_feedback", []),
            "has_execution_feedback": True,
            "last_errors_info": errors_info_text,
            "failure_root_cause": failure_root_cause,
            "required_change_type": required_change_type,
            "failure_signature": failure_signature,
            "deadlock_signature": self._cortex_state.get("deadlock_signature", ""),
            "deadlock_reflection_cycles": self._cortex_state.get("deadlock_reflection_cycles", 0),
            "uncertain_execution": uncertain_execution,
            "heightened_failure_signal": heightened_failure_signal,
            "execution_pending": self._cortex_state.get("execution_pending", False),
            "force_big_brain_replan": self._cortex_state.get("force_big_brain_replan", False),
            "sanitized_subtask_hint": self._cortex_state.get("sanitized_subtask_hint", ""),
            "redundant_tool_selection": self._cortex_state.get("redundant_tool_selection", False),
            "selected_item_already_correct": self._cortex_state.get("selected_item_already_correct", False),
        }

        if latest_gathered_info is not None:
            memory_payload.update({
                "gathered_info": latest_gathered_info,
                "toolbar_information": latest_toolbar_information,
                "selected_position": latest_selected_position,
                "selected_item_name": latest_selected_item_name,
                "chosen_item": obs.get("chosen_item", None) if isinstance(obs, dict) else None,
            })
        if latest_summary and not existing_history_summary:
            memory_payload.update({
                "summarization": latest_summary,
                "history_summary": latest_summary,
            })

        self.memory.update_info_history(memory_payload)

        if self.cortex_memory is not None and self.cortex_memory is not self.memory:
            self.cortex_memory.update_info_history(memory_payload)

    def _get_recent_or_default(self, key: str, default=None):
        try:
            values = self.memory.get_recent_history(key, k=1)
            if values and len(values) > 0:
                return values[0]
        except Exception:
            pass
        return default

    def pipeline_shutdown(self):
        if getattr(self, '_shutdown_done', False):
            return
        self._shutdown_done = True

        self.gm.cleanup_io()
        # self.video_recorder.finish_capture()

        candidate_work_dirs = []
        for candidate in (
            config.work_dir,
            getattr(logger, 'work_dir', None),
            os.path.dirname(getattr(logger, 'log_dir', '')) if getattr(logger, 'log_dir', None) else None,
        ):
            if candidate and candidate not in candidate_work_dirs:
                candidate_work_dirs.append(candidate)

        log = ""
        for work_dir in candidate_work_dirs:
            log_path = os.path.join(work_dir, 'logs', 'stardojo.log')
            if os.path.exists(log_path):
                log = process_log_messages(work_dir)
                break

        if not log:
            logger.warn('>>> Markdown generation skipped: stardojo.log not found in current run directory.')
            logger.write('>>> Bye.')
            return

        output_dir = config.work_dir if config.work_dir else candidate_work_dirs[0]
        os.makedirs(os.path.join(output_dir, 'logs'), exist_ok=True)
        task_label = str(self.task_description or "").strip()
        safe_task_label = re.sub(r'[<>:"/\\|?*]+', "_", task_label).strip(" ._")
        if not safe_task_label:
            safe_task_label = "task"
        with open(os.path.join(output_dir, 'logs', f'{safe_task_label}_log.md'), 'a', encoding='utf-8') as f:
            log = replace_unsupported_chars(log)
            f.write(log)

        logger.write('>>> Markdown generated.')
        logger.write('>>> Bye.')

    def skill_curation(self):
        all_generated_actions = self.memory.get_recent_history("all_generated_actions", k=1)[0]

        self.gm.register_generated_skills(all_generated_actions)

        self._refresh_skill_library()

        self.memory.update_info_history({"skill_library": self.skill_library})

    def run_self_reflection(self):

        logger.write("Stardew Self Reflection Preprocess")

        # 1. Prepare the parameters to call llm api
        self.self_reflection_preprocess()

        # 2. Call llm api for self reflection
        response = self.self_reflection()

        # 3. Postprocess the response
        self.self_reflection_postprocess(response)

    def run_task_inference(self):

        logger.write("Stardew Task Inferrence")

        # 1. Prepare the parameters to call llm api
        self.task_inference_preprocess()

        # 2. Call llm api for task inference
        response = self.task_inference()

        # 3. Postprocess the response
        self.task_inference_postprocess(response)
        
        
        
    def run_action_planning(self, obs, image_obs = True):
        '''
        add text_observation's memory segmentation
        '''
    

        # 1. Prepare the parameters to call llm api
        logger.write("Stardew Action Planning Preprocess")

        prompts = [
            # "Now, I will give you five screenshots for decision making."
            # "This screenshot is five steps before the current step of the game",
            # "This screenshot is three steps before the current step of the game",
            # "This screenshot is two steps before the current step of the game",
            #"This screenshot is the previous step of the game. The blue band represents the left side and the yellow band represents the right side.",
            #"This screenshot is the current step of the game. The blue band represents the left side and the yellow band represents the right side."
            "This screenshot is the previous step of the game.",
            "This screenshot is the current step of the game."
        ]

        processed_params: Dict[str, Any] = {
            "toolbar_information": self._format_toolbar_text(obs.get('inventory', []), obs.get('chosen_item', None)),
        }

        if image_obs:
            image_path = None
            images = obs.get(constants.IMAGES_INPUT_TAG_NAME, [])
            if isinstance(images, list) and len(images) > 0 and isinstance(images[0], dict):
                image_path = images[0].get(constants.IMAGE_PATH_TAG_NAME)

            if image_path:
                self.memory.update_info_history({"image": image_path})

            image_memory = self.memory.get_recent_history("image", k=config.action_planning_image_num)

            image_introduction = []
            max_items = min(len(image_memory), len(prompts))
            for i in range(max_items, 0, -1):
                image_introduction.append(
                    {
                        "introduction": prompts[-i],
                        "path": image_memory[-i] if image_obs else "",  # "" if you don't use image
                        "assistant": ""
                    })
            processed_params["image_introduction"] = image_introduction
        
        processed_params.update(obs)

        self.memory.update_info_history(processed_params)



        # 2. Call llm api for action planning
        params = deepcopy(self.memory.working_area)
        data = self.planner.action_planning(input=params)
        response = data['res_dict']
        del params

        # 3. Postprocess the response
        logger.write("Stardew Action Planning Postprocess")

        processed_response = deepcopy(response)

        skill_steps = []
        if 'actions' in response:
            skill_steps = response['actions']

        if skill_steps:
            skill_steps = [i for i in skill_steps if i != '']
        else:
            skill_steps = ['']

        skill_steps = skill_steps[:config.number_of_execute_skills]

        if config.number_of_execute_skills > 1:
            actions = "[" + ",".join(skill_steps) + "]"
        else:
            actions = str(skill_steps[0])

        action_planning_reasoning = response.get('reasoning', '')
        logger.write(f"Actions {actions}")
        logger.write(f"Action Planning Reasoning\n{action_planning_reasoning}")

        processed_response.update({
            "action": actions,
            "action_planning_reasoning": action_planning_reasoning,
            "skill_steps": skill_steps,
        })
        self.memory.update_info_history(processed_response)

        # 4. Execute the actions
        params = deepcopy(self.memory.working_area)

        skill_steps = params.get("skill_steps", [])
        recovery_inventory = params.get("inventory", [])
        if not isinstance(recovery_inventory, list):
            gathered_info = params.get("gathered_info", {})
            if isinstance(gathered_info, dict):
                recovery_inventory = gathered_info.get("inventory", [])
            else:
                recovery_inventory = []
        if not isinstance(recovery_inventory, list):
            recovery_inventory = []

        recovery_task_description = (
            params.get("task_description")
            or params.get("task")
            or self.task_description
            or ""
        )
        if hasattr(self.gm, "set_recovery_context"):
            try:
                self.gm.set_recovery_context(
                    inventory=recovery_inventory,
                    task_description=recovery_task_description,
                )
            except Exception as e:
                logger.warn(f"Failed to set recovery context before execute_actions: {e}")

        exec_info = self.gm.execute_actions(skill_steps)

        res_params = {
            "exec_info": exec_info,
        }

        self.memory.update_info_history(res_params)

        del params

    def _run_legacy_planning_fallback(self) -> List[str]:
        params = deepcopy(self.memory.working_area)
        data = self.planner.action_planning(input=params)
        response = data['res_dict']
        del params

        logger.write("Stardew Action Planning Postprocess [legacy fallback]")

        processed_response = deepcopy(response)

        skill_steps = []
        if 'actions' in response:
            skill_steps = response['actions']

        if skill_steps:
            skill_steps = [i for i in skill_steps if i != '']
        else:
            skill_steps = ['']

        skill_steps = skill_steps[:config.number_of_execute_skills]

        if config.number_of_execute_skills > 1:
            actions = "[" + ",".join(skill_steps) + "]"
        else:
            actions = str(skill_steps[0])

        action_planning_reasoning = response.get('reasoning', '')
        logger.write(f"Actions {actions}")
        logger.write(f"Action Planning Reasoning\n{action_planning_reasoning}")

        processed_response.update({
            "action": actions,
            "action_planning_reasoning": action_planning_reasoning,
            "skill_steps": skill_steps,
        })
        self.memory.update_info_history(processed_response)

        return skill_steps

    def run_planning(self, obs, step_num, image_obs=True):
        '''
        add text_observation's memory segmentation
        '''

        # 1. Prepare the parameters to call llm api
        logger.write("Stardew Action Planning Preprocess")

        prompts = [
            # "Now, I will give you five screenshots for decision making."
            # "This screenshot is five steps before the current step of the game",
            # "This screenshot is three steps before the current step of the game",
            # "This screenshot is two steps before the current step of the game",
            # "This screenshot is the previous step of the game. The blue band represents the left side and the yellow band represents the right side.",
            # "This screenshot is the current step of the game. The blue band represents the left side and the yellow band represents the right side."
            "This screenshot is the previous step of the game.",
            "This screenshot is the current step of the game."
        ]

        processed_params: Dict[str, Any] = {
            "toolbar_information": self._format_toolbar_text(obs.get('inventory', []), obs.get('chosen_item', None))
        }

        if image_obs:
            image_paths = self._resolve_image_paths(obs.get("image_paths", []))
            if image_paths:
                self.memory.update_info_history({"image": image_paths})

            image_memory = self.memory.get_recent_history("image", k=config.action_planning_image_num)

            image_introduction = []
            max_items = min(len(image_memory), len(prompts))
            for i in range(max_items, 0, -1):
                image_introduction.append(
                    {
                        "introduction": prompts[-i],
                        "path": image_memory[-i] if image_obs else "",  
                        "assistant": ""
                    })
            processed_params["image_introduction"] = image_introduction
        processed_params.update(obs)

        self.memory.update_info_history(processed_params)

        cortex_skill_steps = self._run_cortex_planning(obs=obs, step_num=step_num)
        if cortex_skill_steps is None:
            if not self._cortex_fallback_warned:
                fallback_reason = self._cortex_init_failure_reason or "dual_brain_controller/create_initial_state unavailable"
                logger.warn(
                    f"[Cortex] Planning unavailable ({fallback_reason}). Falling back to legacy action planning."
                )
                self._cortex_fallback_warned = True
            return self._run_legacy_planning_fallback()

        if cortex_skill_steps:
            actions = "[" + ",".join(cortex_skill_steps) + "]" if len(cortex_skill_steps) > 1 else str(cortex_skill_steps[0])
        else:
            actions = ""

        action_planning_reasoning = ""
        if isinstance(self._cortex_state, dict):
            action_planning_reasoning = str(self._cortex_state.get("decision_trace", "") or self._cortex_state.get("context_summary", ""))

        pre_energy = self._get_recent_or_default("energy", None)
        pre_money = self._get_recent_or_default("money", None)
        pre_health = self._get_recent_or_default("health", None)

        processed_response = {
            "action": actions,
            "pre_action": cortex_skill_steps[0] if cortex_skill_steps else "",
            "pre_energy": pre_energy,
            "pre_money": pre_money,
            "pre_health": pre_health,
            "action_planning_reasoning": action_planning_reasoning,
            "decision_making_reasoning": action_planning_reasoning,
            "pre_decision_making_reasoning": action_planning_reasoning,
            "skill_steps": cortex_skill_steps,
        }
        self.memory.update_info_history(processed_response)

        logger.write(f"Actions {actions}")
        if action_planning_reasoning:
            logger.write(f"Action Planning Reasoning\n{action_planning_reasoning}")

        return cortex_skill_steps

def exit_cleanup(runner):
    logger.write("Exiting pipeline.")
    runner.pipeline_shutdown()
