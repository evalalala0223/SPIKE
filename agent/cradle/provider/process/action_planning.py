import re
from typing import Dict, Any, List, Optional, Tuple
from copy import deepcopy

from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider import BaseProvider
from cradle.runner.vllm_client import VLLMClient
from cradle import constants
from stardojo.utils.task_bootstrap import build_task_acquisition_context
from stardojo.utils.stardew_prompt_state import extract_stardew_prompt_fact_fields

config = Config()
logger = Logger()
memory = LocalMemory()

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

        if hasattr(memory, "update_working_area"):
            memory.update_working_area(processed_params)
        else:
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

        response_keys = memory.get_latest("response_keys", {})
        response = memory.get_latest("response", {})
        pre_action = memory.get_latest("pre_action", "")
        pre_self_reflection_reasoning = memory.get_latest("pre_self_reflection_reasoning", "")
        pre_screen_classification = memory.get_latest("pre_screen_classification", "")
        screen_classification = memory.get_latest("screen_classification", "")
        skill_library = memory.get_latest("skill_library", "")
        task_description = memory.get_latest("task_description", "")

        previous_action = ""
        previous_reasoning = ""
        if pre_action:
            previous_action = memory.get_latest("action", "")
            previous_reasoning = memory.get_latest("decision_making_reasoning", "")

        previous_self_reflection_reasoning = ""
        if pre_self_reflection_reasoning:
            previous_self_reflection_reasoning = memory.get_latest("self_reflection_reasoning", "")

        info_summary = memory.get_latest("summarization", "")

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

        # SA-KG Experience Retrieval (LLM-Assisted Judgment Mode)
        similar_experiences = []
        sa_kg_suggestion = ""
        try:
            # Build current state description for SA-KG retrieval
            current_state_desc = f"Screen: {screen_classification}, Task: {task_description}, Summary: {info_summary}"
            
            similar_experiences = memory.retrieve_similar_experiences(
                current_state=current_state_desc,
                top_k=3
            )
            
            if similar_experiences:
                # Format experiences as reference for LLM
                sa_kg_suggestion = "\\n--- Historical Experiences (Reference Only) ---\\n"
                for i, exp in enumerate(similar_experiences, 1):
                    state_description = str(
                        exp.get("state_description")
                        or getattr(exp.get("state"), "description", "")
                        or ""
                    ).strip()
                    action_text = exp.get("action", "")
                    if isinstance(action_text, dict):
                        action_text = action_text.get("action", "")
                    elif not isinstance(action_text, str):
                        action_text = getattr(action_text, "action", "")
                    action_text = str(action_text or "").strip()
                    state_preview = state_description[:100]
                    if len(state_description) > 100:
                        state_preview += "..."

                    sa_kg_suggestion += (
                        f"{i}. Similar State (similarity: {float(exp.get('similarity', 0) or 0):.2f}): "
                        f"{state_preview or '[missing state description]'}\\n"
                    )
                    sa_kg_suggestion += f"   Previous Action: {action_text or '[missing action]'}\\n"
                    sa_kg_suggestion += (
                        f"   Success Rate: {float(exp.get('success_rate', 0) or 0):.1%}\\n"
                    )
                sa_kg_suggestion += "--- End of Historical Experiences ---\\n"
                sa_kg_suggestion += "Note: These are suggestions only. You should evaluate them based on current context and make your own decision.\\n"
                
                logger.write(f"SA-KG retrieved {len(similar_experiences)} similar experiences as reference")
        except Exception as e:
            logger.warn(f"SA-KG retrieval failed (feature may be disabled): {e}")

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
            "image_introduction": image_introduction,
            "sa_kg_suggestion": sa_kg_suggestion  # Add SA-KG suggestions as reference
        }

        if hasattr(memory, "update_working_area"):
            memory.update_working_area(processed_params)
        else:
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
                if hasattr(memory, "get_latest"):
                    return memory.get_latest(key, default)
                values = memory.get_recent_history(key, k=1)
                if values:
                    latest = values[0]
                    if latest not in ("", None):
                        return latest
            except Exception:
                pass
            return default

        prompts = [
            "Now, I will give you five screenshots for decision making.",
            "This screenshot is five steps before the current step of the game",
            "This screenshot is three steps before the current step of the game",
            "This screenshot is two steps before the current step of the game",
            "This screenshot is the previous step of the game. The blue band represents the left side and the yellow band represents the right side.",
            "This screenshot is the current step of the game. The blue band represents the left side and the yellow band represents the right side."
        ]

        pre_action = _safe_recent("pre_action", "")
        pre_self_reflection_reasoning = _safe_recent("pre_self_reflection_reasoning", "")
        toolbar_information = _safe_recent("toolbar_information", "")
        selected_position = _safe_recent("selected_position", 1)
        summarization = _safe_recent("summarization", "")
        skill_library = _safe_recent("skill_library", "")
        task_description = _safe_recent("task_description", "")
        subtask_description = _safe_recent("subtask_description", "")
        history_summary = _safe_recent("summarization", "")
        date_time = _safe_recent("date_time", "")
        reflection_success = _safe_recent("success", None)
        reflection_status = _safe_recent("status", "")
        latest_execution_summary = _safe_recent("latest_execution_summary", "")
        recent_execution_feedback = _safe_recent("recent_execution_feedback", [])
        memory_reference = _safe_recent("memory_reference", "")
        task_progress_quantity = _safe_recent("task_progress_quantity", None)
        previous_task_progress_quantity = _safe_recent("previous_task_progress_quantity", None)
        task_progress_delta = _safe_recent("task_progress_delta", None)
        zero_progress_streak = int(_safe_recent("zero_progress_streak", 0) or 0)
        repeated_action_streak = int(_safe_recent("repeated_action_streak", 0) or 0)
        position_issue_detected = bool(_safe_recent("position_issue_detected", False))
        if hasattr(memory, "get_working_area_snapshot"):
            working_area = memory.get_working_area_snapshot()
        else:
            working_area = dict(getattr(memory, "working_area", {}) or {})
        def _prefer_working_area(current_value, key: str):
            if key not in working_area:
                return current_value
            candidate = working_area.get(key)
            if candidate in (None, ""):
                return current_value
            return candidate
        toolbar_information = _prefer_working_area(toolbar_information, "toolbar_information")
        selected_position = _prefer_working_area(selected_position, "selected_position")
        summarization = _prefer_working_area(summarization, "summarization")
        skill_library = _prefer_working_area(skill_library, "skill_library")
        task_description = _prefer_working_area(task_description, "task_description")
        subtask_description = _prefer_working_area(subtask_description, "subtask_description")
        history_summary = _prefer_working_area(history_summary, "history_summary")
        date_time = _prefer_working_area(date_time, "date_time")
        latest_execution_summary = _prefer_working_area(latest_execution_summary, "latest_execution_summary")
        recent_execution_feedback = _prefer_working_area(recent_execution_feedback, "recent_execution_feedback")
        memory_reference = _prefer_working_area(memory_reference, "memory_reference")
        task_progress_quantity = _prefer_working_area(task_progress_quantity, "task_progress_quantity")
        previous_task_progress_quantity = _prefer_working_area(previous_task_progress_quantity, "previous_task_progress_quantity")
        task_progress_delta = _prefer_working_area(task_progress_delta, "task_progress_delta")
        zero_progress_streak = int(_prefer_working_area(zero_progress_streak, "zero_progress_streak") or 0)
        repeated_action_streak = int(_prefer_working_area(repeated_action_streak, "repeated_action_streak") or 0)
        position_issue_detected = bool(_prefer_working_area(position_issue_detected, "position_issue_detected"))
        if "gathered_info" not in working_area:
            working_area["gathered_info"] = _safe_recent("gathered_info", {})
        if task_description not in (None, ""):
            working_area["task_description"] = task_description
            working_area.setdefault("task", task_description)
            working_area.setdefault("main_task", task_description)
        if subtask_description not in (None, ""):
            working_area["subtask_description"] = subtask_description
        prompt_fact_fields = extract_stardew_prompt_fact_fields(
            state=working_area,
            gathered_info=working_area.get("gathered_info"),
        )

        def _has_value(value: Any) -> bool:
            if value is None:
                return False
            if isinstance(value, str):
                return bool(value.strip())
            if isinstance(value, (list, tuple, dict, set)):
                return len(value) > 0
            return True

        def _backfill_prompt_fact(key: str, *aliases: str, default: Any = "") -> Any:
            current_value = prompt_fact_fields.get(key)
            if _has_value(current_value):
                return current_value

            for candidate_key in (key, *aliases):
                candidate = working_area.get(candidate_key)
                if _has_value(candidate):
                    return candidate

            for candidate_key in (key, *aliases):
                candidate = _safe_recent(candidate_key, None)
                if _has_value(candidate):
                    return candidate

            return default

        prompt_fact_fields.update({
            "basic_knowledge": _backfill_prompt_fact("basic_knowledge", default=[]),
            "location": _backfill_prompt_fact("location"),
            "time": _backfill_prompt_fact("time"),
            "season": _backfill_prompt_fact("season"),
            "health": _backfill_prompt_fact("health"),
            "energy": _backfill_prompt_fact("energy"),
            "money": _backfill_prompt_fact("money"),
            "position": _backfill_prompt_fact("position", "current_position"),
            "current_position": _backfill_prompt_fact("current_position", "position"),
            "facing_direction": _backfill_prompt_fact("facing_direction"),
            "facing_position": _backfill_prompt_fact("facing_position"),
            "current_menu": _backfill_prompt_fact("current_menu", "CurrentMenuData", default="No Menu"),
            "inventory": _backfill_prompt_fact("inventory", default=[]),
            "chosen_item": _backfill_prompt_fact("chosen_item"),
            "crops": _backfill_prompt_fact("crops", default="(none)"),
            "buildings": _backfill_prompt_fact("buildings", default="(none)"),
            "furniture": _backfill_prompt_fact("furniture", default="(none)"),
            "npcs": _backfill_prompt_fact("npcs", default="(none)"),
            "exits": _backfill_prompt_fact("exits", default="(none)"),
        })

        # Decision making preparation
        toolbar_information = toolbar_information if toolbar_information is not None else self.toolbar_information
        if not toolbar_information:
            toolbar_information = prompt_fact_fields.get("toolbar_information", "")
        prompt_fact_fields["toolbar_information"] = toolbar_information

        if selected_position in (None, ""):
            selected_position = prompt_fact_fields.get("selected_position")
        selected_position = selected_position if selected_position is not None else 1
        prompt_fact_fields["selected_position"] = selected_position

        previous_action = ""
        previous_reasoning = ""
        if pre_action:
            previous_action = _safe_recent("action", "")
            previous_reasoning = _safe_recent("decision_making_reasoning", "")

        action_planning_reasoning = _safe_recent("pre_decision_making_reasoning", "")
        if not action_planning_reasoning:
            action_planning_reasoning = previous_reasoning or _safe_recent(
                "action_planning_reasoning", ""
            )

        previous_self_reflection_reasoning = ""
        if pre_self_reflection_reasoning:
            raw_reasoning = _safe_recent("self_reflection_reasoning", "")
            # Prepend explicit success/failure flag so the LLM gets a clear signal
            if reflection_status == 'failure' or reflection_success is False:
                previous_self_reflection_reasoning = f"FAILED - The last action failed. {raw_reasoning}"
            elif reflection_status == 'success' or reflection_success is True:
                previous_self_reflection_reasoning = f"SUCCESS - The last action succeeded. {raw_reasoning}"
            else:
                previous_self_reflection_reasoning = f"IN PROGRESS - {raw_reasoning}"

        # 优先使用增强图像；若为空则回退到原始截图
        image_memory = memory.get_recent_history("augmented_image", k=config.action_planning_image_num)
        image_memory = [p for p in image_memory if p]
        if not image_memory:
            image_memory = memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=config.action_planning_image_num)
            image_memory = [p for p in image_memory if p]

        image_introduction = []
        for i in range(len(image_memory), 0, -1):
            image_introduction.append(
                {
                    "introduction": prompts[-i],
                    "path": image_memory[-i],
                    "assistant": ""
                })

        feedback_lines = []
        if isinstance(recent_execution_feedback, list):
            for entry in recent_execution_feedback[-4:]:
                if not isinstance(entry, dict):
                    continue
                summary = str(entry.get("summary", "") or "").strip()
                entry_errors = str(entry.get("errors_info", "") or "").strip()
                if entry_errors:
                    summary += f" [Error: {entry_errors}]"
                if summary:
                    feedback_lines.append(f"- {summary}")
        recent_execution_feedback_text = "\n".join(feedback_lines)

        task_progress_summary = str(latest_execution_summary or "").strip()
        if not task_progress_summary and task_progress_quantity is not None:
            if previous_task_progress_quantity is not None and task_progress_delta is not None:
                task_progress_summary = (
                    f"Task progress changed from {previous_task_progress_quantity} "
                    f"to {task_progress_quantity} (delta={task_progress_delta})."
                )
            else:
                task_progress_summary = f"Recorded task progress is {task_progress_quantity}."

        failure_signals = []
        if zero_progress_streak > 0:
            failure_signals.append(f"zero_progress_streak={zero_progress_streak}")
        if repeated_action_streak > 0:
            failure_signals.append(f"repeated_action_streak={repeated_action_streak}")
        if position_issue_detected:
            failure_signals.append("position_issue_detected=true")
        last_errors_info = str(_safe_recent("last_errors_info", "") or "").strip()
        if last_errors_info:
            failure_signals.append(f"last_error: {last_errors_info}")
        oscillation_streak = int(_safe_recent("oscillation_streak", 0) or 0)
        if oscillation_streak >= 2:
            failure_signals.append(f"oscillation_streak={oscillation_streak}")
        acquisition_context = build_task_acquisition_context(task_description)
        action_feedback = str(
            _prefer_working_area(_safe_recent("action_feedback", ""), "action_feedback")
            or latest_execution_summary
            or ""
        ).strip()

        processed_params = {
            "action_feedback": action_feedback,
            "pre_self_reflection_reasoning": pre_self_reflection_reasoning,
            "summarization": summarization,
            "skill_library": skill_library,
            "task_description": task_description,
            "subtask_description": subtask_description,
            "history_summary": history_summary,
            "memory_reference": memory_reference,
            "date_time": date_time if date_time else "",
            "action": previous_action,
            "previous_action": previous_action,
            "action_planning_reasoning": action_planning_reasoning,
            "previous_reasoning": previous_reasoning,
            "previous_self_reflection_reasoning": previous_self_reflection_reasoning,
            "image_introduction": image_introduction,
            "latest_execution_summary": latest_execution_summary,
            "recent_execution_feedback": recent_execution_feedback_text,
            "task_progress_summary": task_progress_summary,
            "failure_signals": ", ".join(failure_signals),
            **prompt_fact_fields,
            **acquisition_context,
        }

        if hasattr(memory, "update_working_area"):
            memory.update_working_area(processed_params)
        else:
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

        decision_making_reasoning = response.get('reasoning', 'No reasoning provided')

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

        decision_making_reasoning = response.get('reasoning', 'No reasoning provided')
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

    @staticmethod
    def _canonicalize_skill_steps(skill_steps: List[Any]) -> List[str]:
        canonical_steps: List[str] = []
        for raw_step in skill_steps or []:
            candidate = VLLMClient._extract_canonical_action_candidate(raw_step)
            if not candidate:
                if str(raw_step or "").strip():
                    logger.warn(
                        f"[ActionPlanning] Dropping malformed action candidate: {raw_step}"
                    )
                continue
            canonical_steps.append(candidate)
        return canonical_steps

    @staticmethod
    def _strip_leading_noop_move(skill_steps: List[str]) -> List[str]:
        if len(skill_steps) > 1 and skill_steps[0] == "move(x=0, y=0)":
            logger.warn(
                "[ActionPlanning] Dropping leading no-op move from BigBrain plan because later grounded actions exist."
            )
            return skill_steps[1:]
        return skill_steps

    @classmethod
    def _extract_safe_reasoning_action(cls, reasoning: Any) -> str:
        reasoning_text = str(reasoning or "")
        if not reasoning_text:
            return ""

        code_match = re.search(r"```(?:python)?\s*(.+?)```", reasoning_text, re.DOTALL | re.IGNORECASE)
        if code_match:
            for line in code_match.group(1).splitlines():
                candidate = VLLMClient._extract_canonical_action_candidate(line)
                if candidate:
                    return candidate

        actions_section = re.search(r"Actions:\s*(.+)$", reasoning_text, re.DOTALL | re.IGNORECASE)
        if actions_section:
            candidates = VLLMClient._extract_action_candidates_from_text(actions_section.group(1))
            if candidates:
                return candidates[0]

        for raw_line in reasoning_text.splitlines():
            line = str(raw_line or "").strip()
            explicit_match = re.match(
                r"^(?:action|next action|chosen action|recommended action)\s*[:\-]\s*(.+)$",
                line,
                re.IGNORECASE,
            )
            if not explicit_match:
                continue
            candidate = VLLMClient._extract_canonical_action_candidate(explicit_match.group(1))
            if candidate:
                return candidate

        return ""

    def _get_execute_skill_limit(self, skill_steps, response: Dict) -> int:
        # Dual-brain mode: big brain plans full cycle, no truncation
        if memory.get_working_area_value("dual_brain_enabled", False):
            return max(len(skill_steps), 4)

        limit = config.number_of_execute_skills
        try:
            recent = memory.get_recent_history(constants.NUMBER_OF_EXECUTE_SKILLS, k=1)
            if recent and recent[0] is not None:
                limit = int(recent[0])
        except Exception:
            pass

        task_text = " ".join([
            str(memory.get_latest("task_description", "")),
            str(memory.get_latest("subtask_description", "")),
            str(response.get("reasoning", "")),
        ]).lower()

        is_clearing_flow = any(k in task_text for k in ["clear obstacle", "clearing", "chop", "tree", "axe", "weed", "rock"])

        if is_clearing_flow and isinstance(skill_steps, list) and len(skill_steps) >= 2 and limit < 2:
            logger.warn("Stardew action planning: force execute skill limit to 2 for clearing flow")
            limit = 2

        return max(1, int(limit))

    STARDEW_DEFAULT_TOOL_KEYS = {
        "axe": "1", "hoe": "2", "watering can": "3",
        "pickaxe": "4", "scythe": "5",
    }
    STARDEW_KEY_TO_SLOT_INDEX = {
        "1": 0, "2": 1, "3": 2, "4": 3, "5": 4,
        "6": 5, "7": 6, "8": 7, "9": 8, "0": 9,
        "-": 10, "+": 11,
    }

    def _resolve_tool_key(self, tool_name: str) -> str:
        """Resolve a tool name to its toolbar key slot.

        Tries memory toolbar_information first, falls back to hardcoded defaults.
        """
        tool_name_lower = (tool_name or "").strip().lower()
        if not tool_name_lower:
            return ""

        # Try parsing toolbar_information from memory
        try:
            toolbar_info = memory.get_recent_history("toolbar_information", k=1)
            toolbar_text = toolbar_info[0] if toolbar_info else ""
            if toolbar_text:
                slot_index = self._find_tool_slot_in_toolbar_text(toolbar_text, tool_name_lower)
                if slot_index is not None:
                    for key, value in self.STARDEW_KEY_TO_SLOT_INDEX.items():
                        if value == slot_index:
                            logger.debug(f"[Tool-Mapping] Resolved '{tool_name}' -> key='{key}' from toolbar")
                            return key
        except Exception as e:
            logger.debug(f"[Tool-Mapping] toolbar parse failed: {e}")

        # Fallback to hardcoded defaults
        key = self.STARDEW_DEFAULT_TOOL_KEYS.get(tool_name_lower, "")
        if key:
            logger.debug(f"[Tool-Mapping] Resolved '{tool_name}' -> key='{key}' from defaults")
        else:
            logger.warn(f"[Tool-Mapping] Cannot resolve tool '{tool_name}' to any key")
        return key

    def _find_tool_slot_in_toolbar_text(self, toolbar_information: Any, tool_name: str) -> Optional[int]:
        tool_lower = str(tool_name or "").strip().lower()
        if not tool_lower:
            return None

        patterns = (
            (re.compile(r"slot_index\s+(\d+)\s*:\s*([^()]+)", re.IGNORECASE), False),
            (re.compile(r"^\s*(\d+)\.\s*([^:]+):", re.IGNORECASE), True),
        )
        for raw_line in str(toolbar_information or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            for pattern, one_based in patterns:
                match = pattern.search(line)
                if not match:
                    continue
                raw_slot_index = int(match.group(1))
                slot_index = max(raw_slot_index - 1, 0) if one_based else raw_slot_index
                item_name = match.group(2).strip().lower()
                if item_name == tool_lower:
                    return slot_index
        return None

    def _supports_skill(self, skill_name: str) -> bool:
        skill_library = memory.get_recent_history("skill_library", k=1)
        skill_text = str(skill_library[0]) if skill_library else ""
        return f"{skill_name}(" in skill_text

    def _extract_selected_item_name(self) -> str:
        for key in ("selected_item_name", "chosen_item"):
            recent = memory.get_recent_history(key, k=1)
            value = recent[0] if recent else ""
            if isinstance(value, dict):
                for candidate_key in ("currentitem", "current_item", "item_name", "name", "item"):
                    candidate = value.get(candidate_key)
                    if candidate:
                        return str(candidate).strip()
            elif value:
                return str(value).strip()

        toolbar_info = memory.get_recent_history("toolbar_information", k=1)
        toolbar_text = str(toolbar_info[0]) if toolbar_info else ""
        match = re.search(
            r"Currently selected item:(?:\s*slot_index\s+\d+\s*:)?\s*([^\n]+)",
            toolbar_text,
            re.IGNORECASE,
        )
        if match:
            return match.group(1).strip()
        return ""

    def _selected_tool_matches(self, required_tool: str) -> bool:
        selected_item = self._extract_selected_item_name()
        if not selected_item:
            return False
        return self._infer_required_tool(selected_item) == required_tool

    @staticmethod
    def _selected_item_requires_interact(item_name: Any) -> bool:
        text = str(item_name or "").strip().lower()
        if not text:
            return False

        placeable_tokens = (
            "seed",
            "seeds",
            "fertilizer",
            "speed-gro",
            "speed gro",
            "retaining soil",
            "basic retaining soil",
            "quality retaining soil",
            "deluxe retaining soil",
            "soil",
        )
        return any(token in text for token in placeable_tokens)

    @staticmethod
    def _is_tool_use_action(step: Any) -> bool:
        if not isinstance(step, str):
            return False
        stripped = step.strip()
        return stripped.startswith("use(") or stripped.startswith("use_tool")

    @staticmethod
    def _is_tool_selection_action(step: Any) -> bool:
        if not isinstance(step, str):
            return False
        stripped = step.strip()
        return stripped.startswith("choose_item(") or stripped.startswith("select_tool(")

    def _rewrite_placeable_item_use_actions(self, skill_steps: List[str]) -> List[str]:
        if not isinstance(skill_steps, list) or not skill_steps:
            return skill_steps

        selected_item = self._extract_selected_item_name()
        if not self._selected_item_requires_interact(selected_item):
            return skill_steps

        fixed_steps = list(skill_steps)
        tool_selected_before_step = False
        corrected = False

        for idx, step in enumerate(fixed_steps):
            if not isinstance(step, str):
                continue

            stripped = step.strip()
            if self._is_tool_selection_action(stripped):
                tool_selected_before_step = True
                continue

            if tool_selected_before_step or not stripped.startswith("use("):
                continue

            direction_match = re.search(r'direction\s*=\s*["\']([^"\']+)["\']', stripped)
            if not direction_match:
                continue

            direction = direction_match.group(1).strip().lower()
            if direction not in {"up", "right", "down", "left"}:
                continue

            fixed_steps[idx] = f'interact(direction="{direction}")'
            corrected = True

        if corrected:
            logger.warn(
                f"[Placeable-Item] rewrote use() to interact() while selected item is '{selected_item}'"
            )

        return fixed_steps

    def _selection_matches_tool(self, step: str, required_tool: str, required_key: str) -> bool:
        stripped = step.strip()
        if stripped.startswith("choose_item("):
            slot_match = re.search(r"slot_index\s*=\s*(-?\d+)", stripped)
            expected_slot = self.STARDEW_KEY_TO_SLOT_INDEX.get(required_key)
            return bool(slot_match and expected_slot is not None and int(slot_match.group(1)) == expected_slot)

        if stripped.startswith("select_tool("):
            key_match = re.search(r"key\s*=\s*['\"]?([0-9\-\+])['\"]?", stripped)
            if key_match:
                return key_match.group(1) == required_key
            tool_match = re.search(r"tool\s*=\s*['\"]([^'\"]+)['\"]", stripped)
            if tool_match:
                return self._infer_required_tool(tool_match.group(1)) == required_tool

        return False

    def _prefer_new_tool_api(self, skill_steps: List[str]) -> bool:
        if any(isinstance(step, str) and step.strip().startswith("choose_item(") for step in skill_steps):
            return True
        if any(isinstance(step, str) and step.strip().startswith("select_tool(") for step in skill_steps):
            return False
        if any(isinstance(step, str) and step.strip().startswith("use_tool") for step in skill_steps):
            return False

        supports_choose_item = self._supports_skill("choose_item")
        supports_select_tool = self._supports_skill("select_tool")
        if supports_choose_item and not supports_select_tool:
            return True
        if supports_select_tool and not supports_choose_item:
            return False
        return supports_choose_item

    def _build_tool_selection_action(self, required_key: str, skill_steps: List[str]) -> str:
        prefer_new_api = self._prefer_new_tool_api(skill_steps)
        slot_index = self.STARDEW_KEY_TO_SLOT_INDEX.get(required_key)
        if prefer_new_api and slot_index is not None:
            return f"choose_item(slot_index={slot_index})"
        return f"select_tool(key='{required_key}')"

    @staticmethod
    def _parse_surroundings_map(surroundings: Any) -> Dict[Tuple[int, int], str]:
        cells: Dict[Tuple[int, int], str] = {}
        for raw_line in str(surroundings or "").splitlines():
            line = raw_line.strip()
            match = re.match(r"^\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]", line)
            if not match:
                continue
            cell = (int(match.group(1)), int(match.group(2)))
            cells[cell] = line.rsplit(":", 1)[-1].strip()
        return cells

    @staticmethod
    def _is_probably_blocking_tile(tile_text: Any) -> bool:
        text = str(tile_text or "").strip().lower()
        if not text or text in {"empty", "none", "null"}:
            return False

        blocking_tokens = (
            "stone", "rock", "boulder", "ore", "twig", "branch", "wood", "log", "stump",
            "weed", "grass", "fiber", "fibre", "hay", "tree", "bush", "fence", "wall",
            "house", "farmhouse", "counter", "water", "pond", "river", "cliff", "building",
            "bed", "table", "chair", "chest", "debris",
        )
        return any(token in text for token in blocking_tokens)

    @staticmethod
    def _parse_move_offsets(action: Any) -> Optional[Tuple[int, int]]:
        if not isinstance(action, str):
            return None

        stripped = action.strip()
        if stripped.startswith("move("):
            x_match = re.search(r"x\s*=\s*(-?\d+)", stripped)
            y_match = re.search(r"y\s*=\s*(-?\d+)", stripped)
            if x_match and y_match:
                return int(x_match.group(1)), int(y_match.group(1))

        directional_moves = {
            "move_right": (1, 0),
            "move_left": (-1, 0),
            "move_up": (0, -1),
            "move_down": (0, 1),
        }
        for prefix, offsets in directional_moves.items():
            if stripped.startswith(f"{prefix}(") or stripped == prefix:
                return offsets
        return None

    def _count_recent_failure_streak(self, statuses: List[Any]) -> int:
        streak = 0
        for status in reversed(statuses):
            normalized = str(status or "").strip().lower()
            if normalized in ("failure", "in_progress"):
                streak += 1
            else:
                break
        return streak

    def _build_reposition_move(self, skill_steps: List[str], previous_action_str: str, failure_streak: int) -> str:
        surroundings_history = memory.get_recent_history("surroundings", k=1)
        surroundings_map = self._parse_surroundings_map(surroundings_history[0] if surroundings_history else "")

        candidates: List[Tuple[int, int]] = []
        seen = set()

        def add_candidate(dx: int, dy: int) -> None:
            if dx == 0 and dy == 0:
                return
            candidate = (int(dx), int(dy))
            if candidate not in seen:
                seen.add(candidate)
                candidates.append(candidate)

        planned_move = next(
            (
                parsed for parsed in
                (self._parse_move_offsets(step) for step in skill_steps[1:])
                if parsed is not None
            ),
            None,
        )

        # Anti-loop recovery is primarily for local alignment. Keep it to 1 tile by default,
        # and only allow a 2-tile nudge when the planned route already shows a larger move.
        magnitude = 1
        if planned_move is not None and failure_streak >= 2:
            planned_span = max(abs(planned_move[0]), abs(planned_move[1]))
            if planned_span >= 2:
                magnitude = 2

        def add_axis_candidates(move_offsets: Tuple[int, int], preferred_magnitude: int) -> None:
            move_x, move_y = move_offsets
            if move_x:
                add_candidate((1 if move_x > 0 else -1) * min(abs(move_x), preferred_magnitude), 0)
            if move_y:
                add_candidate(0, (1 if move_y > 0 else -1) * min(abs(move_y), preferred_magnitude))

        if planned_move is not None:
            move_x, move_y = planned_move
            if move_x and move_y:
                first_axis = (move_x, 0)
                second_axis = (0, move_y)
                if self._is_probably_blocking_tile(surroundings_map.get((1 if move_x > 0 else -1, 0))):
                    first_axis, second_axis = second_axis, first_axis
                elif abs(move_y) > abs(move_x):
                    first_axis, second_axis = second_axis, first_axis
                add_axis_candidates(first_axis, magnitude)
                add_axis_candidates(second_axis, magnitude)
            else:
                add_axis_candidates(planned_move, magnitude)

        previous_move = self._parse_move_offsets(previous_action_str)
        if previous_move is not None:
            prev_x, prev_y = previous_move
            if prev_x:
                add_candidate(0, magnitude)
                add_candidate(0, -magnitude)
                add_candidate(-1 if prev_x > 0 else 1, 0)
            if prev_y:
                add_candidate(magnitude, 0)
                add_candidate(-magnitude, 0)
                add_candidate(0, -1 if prev_y > 0 else 1)

        add_candidate(magnitude, 0)
        add_candidate(-magnitude, 0)
        add_candidate(0, magnitude)
        add_candidate(0, -magnitude)

        for dx, dy in candidates:
            first_cell = (
                0 if dx == 0 else (1 if dx > 0 else -1),
                0 if dy == 0 else (1 if dy > 0 else -1),
            )
            if not self._is_probably_blocking_tile(surroundings_map.get(first_cell)):
                return f"move(x={dx}, y={dy})"

        fallback_dx, fallback_dy = candidates[0] if candidates else (0, 1)
        return f"move(x={fallback_dx}, y={fallback_dy})"

    def _infer_required_tool(self, text: str) -> str:
        text = (text or "").lower()
        if not text:
            return ""

        # Priority 1: Explicit tool name mentions (exact match first)
        # "pickaxe" must be checked before "axe" to avoid substring collision
        if "pickaxe" in text:
            return "pickaxe"
        if "scythe" in text:
            return "scythe"
        if "hoe" in text:
            return "hoe"
        if "watering can" in text:
            return "watering can"
        # Check "axe" only if "pickaxe" was NOT matched above
        if "axe" in text:
            return "axe"

        # Priority 2: Obstacle keyword inference (only if no tool name found)
        if any(k in text for k in ["rock", "stone", "boulder", "ore"]):
            return "pickaxe"
        if any(k in text for k in ["tree", "stump", "branch", "wood", "log"]):
            return "axe"
        if any(k in text for k in ["weed", "grass", "fiber", "bush"]):
            return "scythe"
        return ""

    def _enforce_tool_obstacle_mapping(self, skill_steps, response: Dict):
        if not isinstance(skill_steps, list) or not skill_steps:
            return skill_steps

        # Only use subtask + self_reflection + LLM reasoning for tool inference.
        # Excluding task_description avoids the main task's "rocks, woods, trees,
        # grasses and weeds" always matching "rock" first in Priority 2 keywords,
        # which would permanently lock tool inference to pickaxe.
        task_text = " ".join([
            str(memory.get_latest("subtask_description", "")),
            str(memory.get_latest("self_reflection_reasoning", "")),
            str(response.get("reasoning", "")),
        ])

        required_tool = self._infer_required_tool(task_text)
        if not required_tool:
            return skill_steps

        has_tool_use = any(self._is_tool_use_action(step) for step in skill_steps)
        if not has_tool_use:
            return skill_steps

        fixed_steps = list(skill_steps)
        corrected = False

        resolved_key = self._resolve_tool_key(required_tool)
        if not resolved_key:
            toolbar_info = memory.get_recent_history("toolbar_information", k=1)
            slot_index = self._find_tool_slot_in_toolbar_text(
                toolbar_info[0] if toolbar_info else "",
                required_tool,
            )
            if slot_index is not None:
                for key, value in self.STARDEW_KEY_TO_SLOT_INDEX.items():
                    if value == slot_index:
                        resolved_key = key
                        break
        if not resolved_key:
            return skill_steps

        if self._selected_tool_matches(required_tool) and not any(
            self._is_tool_selection_action(step) for step in fixed_steps
        ):
            return skill_steps

        selection_action = self._build_tool_selection_action(resolved_key, fixed_steps)

        for idx, step in enumerate(fixed_steps):
            if not isinstance(step, str) or not self._is_tool_selection_action(step):
                continue
            if not self._selection_matches_tool(step, required_tool, resolved_key):
                fixed_steps[idx] = selection_action
                corrected = True
            break

        if not any(isinstance(s, str) and self._is_tool_selection_action(s) for s in fixed_steps):
            first_use_tool_index = next(
                (i for i, s in enumerate(fixed_steps) if self._is_tool_use_action(s)),
                -1,
            )
            if first_use_tool_index >= 0:
                fixed_steps.insert(first_use_tool_index, selection_action)
                corrected = True

        if corrected:
            # If we inserted a select_tool, trim trailing nop() to keep
            # total action count unchanged (avoids 5-action overflow).
            original_len = len(skill_steps)
            while len(fixed_steps) > original_len and fixed_steps:
                if isinstance(fixed_steps[-1], str) and fixed_steps[-1].strip() == "nop()":
                    fixed_steps.pop()
                else:
                    break
            logger.warn(f"[Tool-Mapping] enforced {selection_action} for '{required_tool}' before tool use")

        return fixed_steps

    def __call__(self, response: Dict):

        logger.write("Stardew Action Planning Postprocess")

        # DEBUG: Log the raw response to diagnose empty actions issue
        logger.write(f"DEBUG: Raw response keys: {response.keys()}")
        logger.write(f"DEBUG: Raw actions from response: {response.get('actions', 'NOT_FOUND')}")

        processed_response = deepcopy(response)

        skill_steps = []
        if 'actions' in response:
            skill_steps = response['actions']
            # logger.write(f"DEBUG: skill_steps after extraction: {skill_steps}")

        if skill_steps:
            skill_steps = [i for i in skill_steps if i != '' and i.strip() != '']
            skill_steps = self._canonicalize_skill_steps(skill_steps)
            skill_steps = self._strip_leading_noop_move(skill_steps)
            # logger.write(f"DEBUG: skill_steps after filtering empty: {skill_steps}")

        # Fallback: only extract actions from explicit action sections, never from free-form reasoning.
        if not skill_steps:
            fallback_action = self._extract_safe_reasoning_action(response.get('reasoning', ''))
            if fallback_action:
                skill_steps.append(fallback_action)
                logger.write(f"DEBUG: Fallback extracted action from explicit action text: {fallback_action}")

        if not skill_steps:
            if memory.get_working_area_value("dual_brain_enabled", False):
                skill_steps = []
                logger.warn("WARNING - No valid actions to execute in dual-brain mode; returning empty plan.")
            else:
                skill_steps = ['nop()']
                logger.warn("WARNING - No actions to execute! Fallback to nop(). Check LLM response format.")

        skill_steps = self._rewrite_placeable_item_use_actions(skill_steps)
        skill_steps = self._enforce_tool_obstacle_mapping(skill_steps, response)

        execute_skill_limit = self._get_execute_skill_limit(skill_steps, response)
        skill_steps = skill_steps[:execute_skill_limit]

        # Anti-loop guard: prevent repeating use()/use_tool() when self-reflection reported failure
        _prev_hist = memory.get_recent_history("action", k=1)
        previous_action_str = _prev_hist[0] if _prev_hist else ""
        _status_hist = memory.get_recent_history("status", k=1)
        reflection_status = _status_hist[0] if _status_hist else ""

        first_is_use = len(skill_steps) >= 1 and self._is_tool_use_action(skill_steps[0])
        prev_was_use = self._is_tool_use_action(previous_action_str)

        if (reflection_status in ('failure', 'in_progress')
                and prev_was_use
                and first_is_use):
            recent_statuses = memory.get_recent_history("status", k=3)
            consecutive_failures = self._count_recent_failure_streak(recent_statuses)
            reposition_move = self._build_reposition_move(skill_steps, previous_action_str, consecutive_failures)

            if consecutive_failures >= 2:
                logger.warn(
                    f"[Anti-Loop] {consecutive_failures} consecutive failures detected. "
                    f"Repositioning with {reposition_move} before retry."
                )
                skill_steps = [reposition_move]
                execute_skill_limit = 1
            else:
                logger.warn(
                    f"[Anti-Loop] Blocked repeat tool use after failure. Previous: {previous_action_str}. "
                    f"Prepending {reposition_move}."
                )
                skill_steps = [reposition_move, skill_steps[0]]
                execute_skill_limit = 2

        pre_action = "[" + ",".join(skill_steps) + "]"

        if execute_skill_limit > 1:
            actions = "[" + ",".join(skill_steps) + "]"
        else:
            actions = str(skill_steps[0])

        decision_making_reasoning = response.get('reasoning', 'No reasoning provided')
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


class SkylinesActionPlanningPreprocessProvider(BaseProvider):
    """Skylines-specific preprocessor that provides all required parameters with defaults."""

    def __init__(self, *args,
                 gm: Any,
                 use_screenshot_augmented=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm
        self.use_screenshot_augmented = use_screenshot_augmented

    def __call__(self):

        logger.write("Skylines Action Planning Preprocess")

        prompts = [
            "This screenshot is the previous step of the game.",
            "This screenshot is the current step of the game."
        ]

        # Get screenshot paths
        screenshot_paths = memory.get_recent_history("screenshot_path", k=config.action_planning_image_num)
        screenshot_augmented_paths = memory.get_recent_history("screenshot_augmented_path", k=config.action_planning_image_num)

        if not self.use_screenshot_augmented:
            image_introduction = []
            for i in range(len(screenshot_paths), 0, -1):
                image_introduction.append({
                    "introduction": prompts[-i],
                    "path": screenshot_paths[-i],
                    "assistant": ""
                })
        else:
            image_introduction = []
            paths = screenshot_augmented_paths if screenshot_augmented_paths else screenshot_paths
            for i in range(len(paths), 0, -1):
                image_introduction.append({
                    "introduction": prompts[-i],
                    "path": paths[-i],
                    "assistant": ""
                })

        # Get parameters from memory with safe defaults
        def safe_get(key, default=""):
            return memory.get_latest(key, default)

        # Required parameters for Skylines action_planning.prompt
        subtask_description = safe_get("subtask_description", "")
        if not subtask_description:
            subtask_description = safe_get("task_description", "Build and manage the city")

        coordinates = safe_get("coordinates", "No buildings constructed yet.")
        last_success_try_place_action = safe_get("last_success_try_place_action", "No previous placement action.")
        budget = safe_get("budget", "Unknown")
        population = safe_get("population", "0")
        actions = safe_get("actions", "No previous action.")
        self_reflection_reasoning = safe_get("self_reflection_reasoning", "No previous reflection.")
        error_message = safe_get("error_message", "")
        construction_information = safe_get("construction_information", "")
        history_summary = safe_get("history_summary", safe_get("summarization", "This is the beginning of the task."))
        skill_library = safe_get("skill_library", "")

        # Get success status from self-reflection - CRITICAL for decision making
        last_action_success = safe_get("success", None)
        if last_action_success is None:
            last_action_success_str = "Unknown (no self-reflection yet)"
        elif last_action_success:
            last_action_success_str = "SUCCESS - The last action was executed successfully."
        else:
            last_action_success_str = "FAILED - The last action failed. You may need to try a different approach."

        # Combine success status with reasoning for clearer context
        enhanced_self_reflection = f"Last action success: {last_action_success_str}\nReasoning: {self_reflection_reasoning}"

        processed_params = {
            "image_introduction": image_introduction,
            "subtask_description": subtask_description,
            "coordinates": coordinates,
            "last_success_try_place_action": last_success_try_place_action,
            "budget": str(budget),
            "population": str(population),
            "actions": actions,
            "self_reflection_reasoning": enhanced_self_reflection,
            "last_action_success": last_action_success_str,
            "error_message": error_message,
            "construction_information": construction_information,
            "history_summary": history_summary,
            "skill_library": skill_library,
        }

        if hasattr(memory, "update_working_area"):
            memory.update_working_area(processed_params)
        else:
            memory.working_area.update(processed_params)

        return processed_params
