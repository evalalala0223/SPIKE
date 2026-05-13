from typing import Dict, Any
from copy import deepcopy

from stardojo.log import Logger
from stardojo.config import Config
from stardojo.memory import LocalMemory
from stardojo.provider import BaseProvider
from stardojo import constants
from stardojo.utils.task_bootstrap import build_task_acquisition_context
from stardojo.utils.stardew_prompt_state import extract_stardew_prompt_fact_fields

config = Config()
logger = Logger()
memory = LocalMemory()

# Cortex/dual-brain pipeline stores gathered_info in cradle's LocalMemory
# singleton (different class from stardojo's LocalMemory).  Import it here
# so postprocess can fall back to it when the stardojo memory is empty.
_cradle_memory = None
try:
    from cradle.memory import LocalMemory as _CradleLocalMemory
    _cradle_memory = _CradleLocalMemory()
except Exception:
    pass

class ActionPlanningPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 use_screenshot_augmented = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm
        self.use_screenshot_augmented = use_screenshot_augmented

    def __call__(self):

        prompts = [
            "This screenshot is the previous step of the game.",
            "This screenshot is the current step of the game."
        ]

        screenshot_paths = memory.get_recent_history("screenshot_path", k=config.action_planning_image_num)
        screenshot_augmnented_paths = memory.get_recent_history("screenshot_augmented_path", k=config.action_planning_image_num)

        if not self.use_screenshot_augmented:
            image_introduction = []
            for i in range(len(screenshot_paths), 0, -1):
                image_introduction.append(
                    {
                        "introduction": prompts[-i],
                        "path": screenshot_paths[-i],
                        "assistant": ""
                    })
        else:
            image_introduction = []
            for i in range(len(screenshot_augmnented_paths), 0, -1):
                image_introduction.append(
                    {
                        "introduction": prompts[-i],
                        "path": screenshot_augmnented_paths[-i],
                        "assistant": ""
                    })

        processed_params = {
            "image_introduction": image_introduction
        }

        memory.working_area.update(processed_params)

        return processed_params


class RDR2ActionPlanningPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm

    def __call__(self):

        logger.write("RDR2 Action Planning Preprocess")

        prompts = [
            "Now, I will give you five screenshots for decision making.",
            "This screenshot is five steps before the current step of the game",
            "This screenshot is three steps before the current step of the game",
            "This screenshot is two steps before the current step of the game",
            "This screenshot is the previous step of the game",
            "This screenshot is the current step of the game"
        ]

        response_keys = memory.get_recent_history("response_keys", k=1)[0]
        response = memory.get_recent_history("response", k=1)[0]
        pre_action = memory.get_recent_history("pre_action", k=1)[0]
        pre_self_reflection_reasoning = memory.get_recent_history("pre_self_reflection_reasoning", k=1)[0]
        pre_screen_classification = memory.get_recent_history("pre_screen_classification", k=1)[0]
        screen_classification = memory.get_recent_history("screen_classification", k=1)[0]
        skill_library = memory.get_recent_history("skill_library", k=1)[0]
        task_description = memory.get_recent_history("task_description", k=1)[0]

        previous_action = ""
        previous_reasoning = ""
        if pre_action:
            previous_action = memory.get_recent_history("action", k=1)[0]
            previous_reasoning = memory.get_recent_history("decision_making_reasoning", k=1)[0]

        previous_self_reflection_reasoning = ""
        if pre_self_reflection_reasoning:
            previous_self_reflection_reasoning = memory.get_recent_history("self_reflection_reasoning", k=1)[0]

        info_summary = memory.get_recent_history("summarization", k=1)[0]

        # @TODO Temporary solution with fake augmented entries if no bounding box exists. Ideally it should read images, then check for possible augmentation.
        image_memory = memory.get_recent_history("screenshot_path", k=config.action_planning_image_num)
        augmented_image_memory = memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET,
                                                           k=config.action_planning_image_num)

        image_introduction = []
        for i in range(len(image_memory), 0, -1):
            if len(augmented_image_memory) >= i and augmented_image_memory[-i] != constants.NO_IMAGE:
                if i == len(image_memory):
                    image_introduction.append(
                        {
                            "introduction": prompts[-i],
                            "path": augmented_image_memory[-i],
                            "assistant": "",
                            "resolution": "high",
                        })
                else:
                    image_introduction.append(
                        {
                            "introduction": prompts[-i],
                            "path": augmented_image_memory[-i],
                            "assistant": "",
                        })
            else:
                image_introduction.append(
                    {
                        "introduction": prompts[-i],
                        "path": image_memory[-i],
                        "assistant": ""
                    })

        # Minimap info tracking
        minimap_information = ""
        if constants.MINIMAP_INFORMATION in response_keys:
            minimap_information = response[constants.MINIMAP_INFORMATION]
            logger.write(f"{constants.MINIMAP_INFORMATION}: {minimap_information}")

            minimap_info_str = ""
            for key, value in minimap_information.items():
                if value:
                    for index, item in enumerate(value):
                        minimap_info_str = minimap_info_str + key + ' ' + str(index) + ': angle ' + str(
                            int(item['theta'])) + ' degree' + '\n'
            minimap_info_str = minimap_info_str.rstrip('\n')

            logger.write(f'minimap_info_str: {minimap_info_str}')
            minimap_information = minimap_info_str

        processed_params = {
            "pre_screen_classification": pre_screen_classification,
            "screen_classification": screen_classification,
            "previous_action": previous_action,
            "previous_reasoning": previous_reasoning,
            "previous_self_reflection_reasoning": previous_self_reflection_reasoning,
            "skill_library": skill_library,
            "task_description": task_description,
            "minimap_information": minimap_information,
            "info_summary": info_summary,
            "image_introduction": image_introduction
        }

        memory.working_area.update(processed_params)

        return processed_params

class StardewActionPlanningPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 toolbar_information: str,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm
        self.toolbar_information = toolbar_information

    def __call__(self):

        logger.write("Stardew Action Planning Preprocess")

        def _safe_recent(key: str, default=None):
            try:
                values = memory.get_recent_history(key, k=1)
                if values:
                    return values[0]
            except Exception:
                pass
            return default

        prompts = [
            "Now, I will give you five screenshots for decision making."
            "This screenshot is five steps before the current step of the game",
            "This screenshot is three steps before the current step of the game",
            "This screenshot is two steps before the current step of the game",
            "This screenshot is the previous step of the game. The blue band represents the left side and the yellow band represents the right side.",
            "This screenshot is the current step of the game. The blue band represents the left side and the yellow band represents the right side."
        ]

        pre_action = _safe_recent("pre_action", "")
        pre_self_reflection_reasoning = _safe_recent("pre_self_reflection_reasoning", "")
        toolbar_information = _safe_recent("toolbar_information", "")
        selected_position = _safe_recent("selected_position", None)
        chosen_item = _safe_recent("chosen_item", None)
        summarization = _safe_recent("summarization", "")
        skill_library = _safe_recent("skill_library", "")
        task_description = _safe_recent("task_description", "")
        subtask_description = _safe_recent("subtask_description", "")
        history_summary = _safe_recent("summarization", "")
        working_area = dict(getattr(memory, "working_area", {}) or {})

        def _prefer_working_area(current_value, key: str):
            if key not in working_area:
                return current_value
            candidate = working_area.get(key)
            if candidate in (None, ""):
                return current_value
            return candidate

        if "gathered_info" not in working_area:
            working_area["gathered_info"] = _safe_recent("gathered_info", {})
        # Cortex/dual-brain pipeline writes ALL state (gathered_info, self-
        # reflection output, execution feedback, etc.) into cradle's
        # LocalMemory singleton, NOT stardojo's.  Always merge any missing
        # fields from cradle memory so the prompt receives all available data.
        # Try dynamic cradle memory access in case module-level init failed
        _cm = _cradle_memory
        if _cm is None:
            try:
                from cradle.memory import LocalMemory as _DynCradleMem
                _cm = _DynCradleMem()
            except Exception:
                pass
        if _cm is not None:
            stardew_task_scope = str(getattr(memory, "_current_task_scope", "") or "").strip()
            cradle_task_scope = str(getattr(_cm, "_current_task_scope", "") or "").strip()
            cradle_wa = getattr(_cm, "working_area", {}) or {}
            should_merge_cradle_working_area = (
                bool(cradle_wa)
                and (
                    not stardew_task_scope
                    or (
                        cradle_task_scope
                        and cradle_task_scope == stardew_task_scope
                    )
                )
            )
            for _fk in (
                # Environment / observation fields
                "gathered_info",
                "location", "current_position", "position",
                "facing_direction", "facing_position",
                "current_menu", "time", "day", "season",
                "energy", "health", "money",
                "surroundings", "buildings", "exits", "npcs",
                "crops", "furniture", "toolbar_information",
                "chosen_item", "selected_position",
                "inventory",
                # Execution feedback fields
                "front_tile_summary", "blocked_recovery_hint",
                "current_blocker_signature",
                "nearest_grounded_target_summary",
                "task_progress_summary", "failure_signals",
                "recent_execution_feedback", "latest_execution_summary",
                "action_feedback",
                # Self-reflection output (CRITICAL: without this, the
                # action planning prompt shows empty self-reflection even
                # when the reflection node ran and produced output)
                "self_reflection_reasoning",
                "pre_self_reflection_reasoning",
                # History / context fields
                "summarization", "history_summary",
                "task_description", "subtask_description",
                "subtask_reasoning",
                "sanitized_subtask_hint",
                "skill_library",
                # Previous action fields
                "pre_action", "action",
                "pre_decision_making_reasoning",
                "decision_making_reasoning",
                "action_planning_reasoning",
                # Task acquisition context
                "target_item", "source_type", "source_detail",
                "basic_knowledge",
                # Memory reference
                "memory_reference",
            ):
                if not should_merge_cradle_working_area:
                    continue
                if _fk not in working_area or working_area[_fk] in (None, "", []):
                    _cv = cradle_wa.get(_fk)
                    if _cv not in (None, "", []):
                        working_area[_fk] = _cv
        prompt_fact_fields = extract_stardew_prompt_fact_fields(
            state=working_area,
            gathered_info=working_area.get("gathered_info"),
        )
        action_feedback = str(
            working_area.get("action_feedback", "")
            or _safe_recent("action_feedback", "")
            or working_area.get("latest_execution_summary", "")
            or _safe_recent("latest_execution_summary", "")
            or ""
        ).strip()
        latest_execution_summary = str(
            working_area.get("latest_execution_summary", "")
            or _safe_recent("latest_execution_summary", "")
            or ""
        ).strip()
        _raw_recent_feedback = (
            working_area.get("recent_execution_feedback")
            or _safe_recent("recent_execution_feedback", [])
            or []
        )
        # Convert list to readable string for prompt substitution
        if isinstance(_raw_recent_feedback, list):
            recent_execution_feedback = "\n".join(
                f"- {str(entry).strip()}" for entry in _raw_recent_feedback if str(entry).strip()
            )
        else:
            recent_execution_feedback = str(_raw_recent_feedback or "")
        memory_reference = str(
            working_area.get("memory_reference", "")
            or _safe_recent("memory_reference", "")
            or ""
        ).strip()
        task_progress_quantity = (
            working_area.get("task_progress")
            or working_area.get("task_progress_quantity")
            or _safe_recent("task_progress", None)
            or _safe_recent("task_progress_quantity", None)
        )
        task_progress_summary = str(
            working_area.get("task_progress_summary", "")
            or _safe_recent("task_progress_summary", "")
            or latest_execution_summary
            or (
                f"Recorded task progress is {task_progress_quantity}."
                if task_progress_quantity not in (None, "", [])
                else ""
            )
        ).strip()
        failure_signals = str(
            working_area.get("failure_signals", "")
            or _safe_recent("failure_signals", "")
            or ""
        ).strip()
        if not failure_signals:
            signal_parts = []
            zero_progress_streak = _prefer_working_area(
                _safe_recent("zero_progress_streak", 0),
                "zero_progress_streak",
            )
            repeated_action_streak = _prefer_working_area(
                _safe_recent("repeated_action_streak", 0),
                "repeated_action_streak",
            )
            oscillation_streak = _prefer_working_area(
                _safe_recent("oscillation_streak", 0),
                "oscillation_streak",
            )
            if int(zero_progress_streak):
                signal_parts.append(f"zero_progress_streak={int(zero_progress_streak)}")
            if int(repeated_action_streak):
                signal_parts.append(f"repeated_action_streak={int(repeated_action_streak)}")
            position_issue_detected = bool(
                _prefer_working_area(
                    _safe_recent("position_issue_detected", False),
                    "position_issue_detected",
                )
            )
            if position_issue_detected:
                signal_parts.append("position_issue_detected=true")
            if int(oscillation_streak):
                signal_parts.append(f"oscillation_streak={int(oscillation_streak)}")
            last_errors_info = str(
                _prefer_working_area(_safe_recent("last_errors_info", ""), "last_errors_info") or ""
            ).strip()
            if last_errors_info:
                signal_parts.append(f"last_error: {last_errors_info}")
            failure_signals = ", ".join(signal_parts)
        date_time = str(
            working_area.get("date_time", "")
            or _safe_recent("date_time", "")
            or ""
        ).strip()
        if not date_time:
            day_text = str(working_area.get("day", "") or prompt_fact_fields.get("day", "") or "").strip()
            time_text = str(working_area.get("time", "") or prompt_fact_fields.get("time", "") or "").strip()
            date_time = f"{day_text} {time_text}".strip()

        # Decision making preparation
        toolbar_information = toolbar_information or self.toolbar_information
        if not toolbar_information:
            toolbar_information = prompt_fact_fields.get("toolbar_information", "")
        if selected_position is None and isinstance(chosen_item, dict):
            fallback_index = chosen_item.get("index", chosen_item.get("slot_index"))
            if isinstance(fallback_index, int):
                selected_position = fallback_index
        if selected_position in (None, ""):
            selected_position = prompt_fact_fields.get("selected_position")
        selected_position = selected_position if selected_position is not None else 1
        prompt_fact_fields["toolbar_information"] = toolbar_information
        prompt_fact_fields["selected_position"] = selected_position

        # Re-read key variables from working_area (which now includes cradle
        # memory fallback values) to fix memory-mismatch where _safe_recent
        # from stardojo memory returned empty.
        pre_action = _prefer_working_area(pre_action, "pre_action") or working_area.get("action", "")
        toolbar_information = _prefer_working_area(toolbar_information, "toolbar_information")
        summarization = _prefer_working_area(summarization, "summarization") or working_area.get("history_summary", "")
        skill_library = _prefer_working_area(skill_library, "skill_library")
        task_description = _prefer_working_area(task_description, "task_description")
        subtask_description = _prefer_working_area(subtask_description, "subtask_description")
        history_summary = _prefer_working_area(history_summary, "history_summary") or summarization

        previous_action = ""
        previous_reasoning = ""
        if pre_action:
            previous_action = (
                _safe_recent("action", "")
                or working_area.get("action", "")
                or working_area.get("pre_action", "")
            )
            previous_reasoning = (
                _safe_recent("decision_making_reasoning", "")
                or working_area.get("decision_making_reasoning", "")
                or working_area.get("action_planning_reasoning", "")
            )

        action_planning_reasoning = (
            _safe_recent("pre_decision_making_reasoning", "")
            or working_area.get("pre_decision_making_reasoning", "")
        )
        if not action_planning_reasoning:
            action_planning_reasoning = (
                previous_reasoning
                or _safe_recent("action_planning_reasoning", "")
                or working_area.get("action_planning_reasoning", "")
            )

        previous_self_reflection_reasoning = (
            _safe_recent("self_reflection_reasoning", "")
            or working_area.get("self_reflection_reasoning", "")
            or ""
        )

        # @TODO Temporary solution with fake augmented entries if no bounding box exists. Ideally it should read images, then check for possible augmentation.
        image_memory = memory.get_recent_history("augmented_image", k=config.action_planning_image_num)

        image_introduction = []
        for i in range(len(image_memory), 0, -1):
            image_introduction.append(
                {
                    "introduction": prompts[-i],
                    "path": image_memory[-i],
                    "assistant": ""
                })
        acquisition_context = build_task_acquisition_context(task_description)

        processed_params = {
            "pre_self_reflection_reasoning": pre_self_reflection_reasoning,
            "summarization": summarization,
            "skill_library": skill_library,
            "task_description": task_description,
            "subtask_description": subtask_description,
            "history_summary": history_summary,
            "date_time": date_time,
            "action_feedback": action_feedback,
            "action": previous_action,
            "previous_action": previous_action,
            "action_planning_reasoning": action_planning_reasoning,
            "previous_reasoning": previous_reasoning,
            "self_reflection_reasoning": previous_self_reflection_reasoning,
            "previous_self_reflection_reasoning": previous_self_reflection_reasoning,
            "memory_reference": memory_reference,
            "latest_execution_summary": latest_execution_summary,
            "recent_execution_feedback": recent_execution_feedback,
            "task_progress_summary": task_progress_summary,
            "failure_signals": failure_signals,
            "image_introduction": image_introduction,
            **prompt_fact_fields,
            **acquisition_context,
        }

        memory.working_area.update(processed_params)

        return processed_params

class ActionPlanningPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, response: Dict):

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

        decision_making_reasoning = response['reasoning']

        processed_response.update({
            "actions": actions,
            "decision_making_reasoning": decision_making_reasoning,
            "skill_steps": skill_steps,
        })
        memory.update_info_history(processed_response)

        return processed_response

class RDR2ActionPlanningPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, response: Dict):

        logger.write("RDR2 Action Planning Postprocess")

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

        decision_making_reasoning = response['reasoning']
        pre_decision_making_reasoning = decision_making_reasoning

        processed_response.update({
            "action": actions,
            "pre_decision_making_reasoning": pre_decision_making_reasoning,
            "decision_making_reasoning": decision_making_reasoning,
            "skill_steps": skill_steps,
        })
        memory.update_info_history(processed_response)

        return processed_response


class StardewActionPlanningPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, response: Dict):

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
        pre_action = "[" + ",".join(skill_steps) + "]"

        if config.number_of_execute_skills > 1:
            actions = "[" + ",".join(skill_steps) + "]"
        else:
            actions = str(skill_steps[0])

        decision_making_reasoning = response['reasoning']
        pre_decision_making_reasoning = decision_making_reasoning

        processed_response.update({
            "pre_action": pre_action,
            "action": actions,
            "action_planning_reasoning": decision_making_reasoning,
            "pre_decision_making_reasoning": pre_decision_making_reasoning,
            "decision_making_reasoning": decision_making_reasoning,
            "skill_steps": skill_steps,
        })
        memory.update_info_history(processed_response)

        return processed_response
