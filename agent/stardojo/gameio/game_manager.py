from __future__ import annotations

import time
from typing import Tuple, Dict, Any

from stardojo import constants
from stardojo.config import Config
from stardojo.environment.ui_control import UIControl
from stardojo.log import Logger
from stardojo.gameio import IOEnvironment
from stardojo.gameio.lifecycle.ui_control import check_active_window
from stardojo.utils.file_utils import assemble_project_path

config = Config()
logger = Logger()
io_env = IOEnvironment()


class GameManager:

    def __init__(
        self,
        env_name,
        embedding_provider = None,
        llm_provider = None,
        skill_registry = None,
        ui_control: UIControl = None,
    ):

        self.env_name = env_name
        self.embedding_provider = embedding_provider
        self.llm_provider = llm_provider
        self.skill_registry = skill_registry
        self.ui_control = ui_control
        self.default_executer = None
        io_env.llm_provider = self.llm_provider # @TODO needs a better DI

        # Cross-plan consecutive-same-action tracking. Each time execute_actions()
        # is called, we compare the first action to the previous call's last
        # action and count how many times the same action has been generated
        # in a row. After a threshold, we REFUSE to execute it and inject a
        # strong failure signal so the LLM must regenerate a different action.
        self._last_problem_action_text = None
        self._consecutive_same_action_count = 0
        self.SAME_ACTION_REFUSE_THRESHOLD = 3

        # Axis-direction circuit breaker: tracks consecutive move() calls
        # that target the same axis direction (e.g. all positive-x) even
        # when the exact magnitudes differ. This catches patterns like
        # move(x=3,y=0) → move(x=2,y=0) → move(x=1,y=0) all failing.
        self._consecutive_failed_move_axis_count = 0
        self._last_failed_move_axis_direction = None  # e.g. "+x", "-y"

        # Recovery context: set by the agent before execute_actions() so that
        # the circuit-breaker can make smarter recovery choices (e.g. picking
        # the right inventory item instead of just closing the menu).
        self._recovery_inventory = None
        self._recovery_task_description = None
        self._recovery_variation_toggle = 0

    def set_recovery_context(self, inventory=None, task_description=None):
        """Provide inventory + task context for smarter circuit-breaker recovery."""
        self._recovery_inventory = inventory
        self._recovery_task_description = task_description

    def reset_task_state(self):
        """Clear all cross-plan tracking state at task boundaries.

        Must be called when switching to a new task to prevent stale
        recovery context, problem-action streaks, and axis-direction
        streaks from leaking across tasks.
        """
        self._recovery_inventory = None
        self._recovery_task_description = None
        self._recovery_variation_toggle = 0
        self._clear_problem_action_streak()
        self._clear_failed_move_axis_streak()

    @staticmethod
    def _extract_move_axis_direction(action: str) -> "str | None":
        """Extract the dominant axis direction from a move() action.

        Returns '+x', '-x', '+y', '-y', or None for non-move/diagonal moves.
        Only returns a direction for single-axis moves (one of x,y is zero).
        """
        import re as _re
        m = _re.match(r'move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)', str(action or ""))
        if not m:
            return None
        x, y = int(m.group(1)), int(m.group(2))
        if x != 0 and y == 0:
            return "+x" if x > 0 else "-x"
        if y != 0 and x == 0:
            return "+y" if y > 0 else "-y"
        # Diagonal moves: use the dominant axis
        if x != 0 and y != 0:
            if abs(x) > abs(y):
                return "+x" if x > 0 else "-x"
            return "+y" if y > 0 else "-y"
        return None

    @staticmethod
    def _normalize_action(action: str) -> str:
        """Normalize whitespace/quotes for reliable action comparison."""
        import re as _re
        s = str(action or "").strip()
        s = s.replace("'", '"')
        s = _re.sub(r'\s+', ' ', s)
        s = _re.sub(r'\s*=\s*', '=', s)
        s = _re.sub(r'\s*,\s*', ', ', s)
        s = _re.sub(r'\(\s+', '(', s)
        s = _re.sub(r'\s+\)', ')', s)
        return s

    @staticmethod
    def _is_busy_timeout_response(skill_response: Any) -> bool:
        return str(skill_response or "").strip().lower().startswith("busy_timeout:")

    @staticmethod
    def _format_busy_timeout_error(skill: str, skill_response: Any) -> str:
        raw = str(skill_response or "").strip()
        timed_out_method = raw.split(":", 1)[1].strip() if ":" in raw else ""
        if timed_out_method:
            return (
                f"{skill} FAILED - mod busy timeout while waiting for "
                f"{timed_out_method} confirmation"
            )
        return f"{skill} FAILED - mod busy timeout"

    @staticmethod
    def _normalize_craft_response(skill_response: Any) -> str:
        return str(skill_response or "").strip()

    @classmethod
    def _is_successful_craft_response(cls, skill_response: Any) -> bool:
        return cls._normalize_craft_response(skill_response).lower().startswith("craft_ok:")

    @classmethod
    def _is_explicit_craft_failure_response(cls, skill_response: Any) -> bool:
        return cls._normalize_craft_response(skill_response).lower().startswith("craft_fail:")

    @classmethod
    def _format_craft_failure_error(
        cls,
        *,
        skill_params: Dict[str, Any] | None,
        skill_response: Any,
    ) -> str:
        craft_item = skill_params.get("item", "?") if isinstance(skill_params, dict) else "?"
        normalized_response = cls._normalize_craft_response(skill_response)
        lowered_response = normalized_response.lower()
        prefix = f'craft(item="{craft_item}")'

        if lowered_response.startswith("craft_fail:"):
            failure_code = lowered_response.split(":", 1)[1].strip()
            message_map = {
                "unknown_recipe": "unknown recipe",
                "missing_materials": "missing materials",
                "inventory_full": "inventory is full",
                "create_item_failed": "item creation failed",
                "menu_open": "menu is open; direct craft requires the world view",
            }
            message = message_map.get(
                failure_code,
                failure_code.replace("_", " ") or "craft failed",
            )
            return f"{prefix} FAILED - {message}"

        if normalized_response and normalized_response != "Message received":
            return f"{prefix} returned unexpected response: {normalized_response}"

        return (
            f'{prefix} returned no confirmation. '
            f"Possible: missing materials or unknown recipe. "
            f'craft() works directly - do NOT open crafting menu first.'
        )

    def _find_target_item_slot(self) -> "int | None":
        """Parse the task description and find a matching inventory slot.

        Supports patterns like:
          fertilize_N_dirt_with_X  → look for item X
          sow_N_dirt_with_X       → look for item X
        Returns slot_index (int) or None.

        Inventory may be either:
          - List[dict] with "Name"/"name" keys (raw from C#), or
          - List[str] formatted as "slot_index N: ItemName (quantity: Q)"
            (after _process_index in the env layer).
        """
        import re as _re
        task = str(self._recovery_task_description or "").strip()
        inventory = self._recovery_inventory
        logger.write(
            f"[SmartRecovery] _find_target_item_slot: task={task!r}, "
            f"inventory_type={type(inventory).__name__}, "
            f"inventory_len={len(inventory) if isinstance(inventory, list) else 'N/A'}, "
            f"inventory_sample={str(inventory[:3])[:200] if isinstance(inventory, list) and inventory else 'empty'}"
        )
        if not task or not inventory or not isinstance(inventory, list):
            return None

        # Extract the item part after "with_"
        m = _re.search(r'(?:fertilize|sow)_\d+_\w+_with_(.+)', task, _re.IGNORECASE)
        if not m:
            return None

        raw_item = m.group(1).strip()
        # Build candidate names: "speed_gro" → "Speed-Gro", "Speed Gro", "Speed_Gro", "speed gro"
        candidates = set()
        candidates.add(raw_item)
        candidates.add(raw_item.lower())
        candidates.add(raw_item.replace("_", " "))
        candidates.add(raw_item.replace("_", "-"))
        # Title-case variants
        candidates.add(raw_item.replace("_", " ").title())
        candidates.add(raw_item.replace("_", "-").title())
        # Capitalize-first variant (e.g. "Speed-Gro")
        hyphen_variant = "-".join(w.capitalize() for w in raw_item.split("_"))
        candidates.add(hyphen_variant)

        lowered_candidates = {c.lower() for c in candidates}

        for idx, item in enumerate(inventory):
            if isinstance(item, dict):
                item_name = str(item.get("Name") or item.get("name") or "")
                if item_name and item_name.lower() in lowered_candidates:
                    # Prefer explicit slot_index from the dict if available;
                    # fall back to the Python list position.
                    slot = item.get("slot_index", item.get("SlotIndex", idx))
                    try:
                        return int(slot)
                    except (TypeError, ValueError):
                        return idx
            elif isinstance(item, str):
                # Format: "slot_index N: ItemName (quantity: Q)"
                # or      "slot_index N: No item"
                sm = _re.match(r'slot_index\s+(\d+):\s*(.+?)(?:\s*\(quantity:.*\))?$', item.strip())
                if sm:
                    slot_idx = int(sm.group(1))
                    item_name = sm.group(2).strip()
                    if item_name.lower() == "no item":
                        continue
                    if item_name.lower() in lowered_candidates:
                        return slot_idx
        return None

    def _pick_recovery_action(self, stuck_action: str) -> str:
        """Pick a recovery action that physically changes game state.

        Strategy per stuck action type:
        - menu(option="open", ...)  → menu(option="close", ...)
        - use(direction="X")        → move(1 tile in that direction)
        - interact(direction="X")   → move(1 tile in that direction)
        - choose_item(slot_index=N) → move(x=1, y=0) to break state
        - move(x=A, y=B) stuck      → move perpendicular 1 tile
        """
        import re as _re
        s = str(stuck_action or "").strip()

        # menu(option="open", menu_name=X) → menu(option="close", menu_name=X)
        m = _re.match(r'menu\(\s*option\s*=\s*"open"\s*,\s*menu_name\s*=\s*"([^"]+)"\s*\)', s)
        if m:
            return f'menu(option="close", menu_name="{m.group(1)}")'

        # use(direction="X") / interact(direction="X") loops usually mean the
        # player is close but misaligned. Sidestep perpendicular to the facing
        # line so the next replan sees a different local geometry instead of
        # repeating the same blocked direction.
        m = _re.match(r'(use|interact)\(\s*direction\s*=\s*"(up|down|left|right)"\s*\)', s)
        if m:
            direction = m.group(2)
            if direction in {"up", "down"}:
                return self._alternate_recovery_action("move(x=1, y=0)", "move(x=-1, y=0)")
            return self._alternate_recovery_action("move(x=0, y=1)", "move(x=0, y=-1)")

        # choose_item(slot_index=N) → small move to break state
        if s.startswith("choose_item"):
            return "move(x=1, y=0)"

        # move(x=A, y=B) stuck → perpendicular 1-tile sidestep with alternating
        # polarity so repeated recoveries can try both sides of the obstacle.
        m = _re.match(r'move\(\s*x\s*=\s*(-?\d+)\s*,\s*y\s*=\s*(-?\d+)\s*\)', s)
        if m:
            x_val = int(m.group(1))
            y_val = int(m.group(2))
            return self._perpendicular_recovery_move(x_val, y_val)

        # Fallback: small move
        return "move(x=1, y=0)"

    def _record_problem_action(self, action_text: str) -> None:
        normalized = self._normalize_action(action_text)
        if not normalized:
            return
        if normalized == self._last_problem_action_text:
            self._consecutive_same_action_count += 1
        else:
            self._last_problem_action_text = normalized
            self._consecutive_same_action_count = 1

    def _clear_problem_action_streak(self) -> None:
        self._last_problem_action_text = None
        self._consecutive_same_action_count = 0

    def _record_failed_move_axis(self, action_text: str) -> None:
        axis = self._extract_move_axis_direction(action_text)
        if axis is None:
            self._clear_failed_move_axis_streak()
            return
        if axis == self._last_failed_move_axis_direction:
            self._consecutive_failed_move_axis_count += 1
        else:
            self._last_failed_move_axis_direction = axis
            self._consecutive_failed_move_axis_count = 1

    def _clear_failed_move_axis_streak(self) -> None:
        self._consecutive_failed_move_axis_count = 0
        self._last_failed_move_axis_direction = None

    def _alternate_recovery_action(self, primary_action: str, secondary_action: str) -> str:
        toggle = int(getattr(self, "_recovery_variation_toggle", 0) or 0) + 1
        self._recovery_variation_toggle = toggle
        return primary_action if toggle % 2 == 1 else secondary_action

    def _perpendicular_recovery_move(self, x_val: int, y_val: int) -> str:
        if abs(x_val) >= abs(y_val) and x_val != 0:
            return self._alternate_recovery_action("move(x=0, y=1)", "move(x=0, y=-1)")
        if y_val != 0:
            return self._alternate_recovery_action("move(x=1, y=0)", "move(x=-1, y=0)")
        return self._alternate_recovery_action("move(x=1, y=0)", "move(x=-1, y=0)")


    def pause_game(self,
                   *args,
                   env_name=config.env_name,
                   ide_name=config.ide_name,
                   screen_type=constants.GENERAL_GAME_INTERFACE,
                   **kwargs):

        if screen_type==constants.PAUSE_INTERFACE:
            return False
        else:
            self.ui_control.pause_game(
                env_name=env_name,
                ide_name=ide_name,
                **kwargs
            )
            return True


    def unpause_game(self,
                     *args,
                     env_name=config.env_name,
                     ide_name=config.ide_name,
                     **kwargs):

        self.ui_control.unpause_game(
            env_name=env_name,
            ide_name=ide_name,
            **kwargs
        )
        return True


    def switch_to_game(self,
                       *args,
                       env_name=config.env_name,
                       ide_name=config.ide_name,
                       **kwargs):

        self.ui_control.switch_to_game(
            env_name=env_name,
            ide_name=ide_name,
            **kwargs
        )


    def check_active_window(self):
        return check_active_window()


    def exit_back_to_pause(self,
                           *args,
                           env_name=config.env_name,
                           ide_name=config.ide_name,
                           **kwargs):

        self.ui_control.exit_back_to_pause(
            env_name=env_name,
            ide_name=ide_name,
            **kwargs
        )


    def get_skill_information(self,
                              skill_list,
                              skill_library_with_code = False
                              ):

        filtered_skill_library = []

        for skill_name in skill_list:
            skill_item = self.skill_registry.get_from_skill_library(skill_name, skill_library_with_code = skill_library_with_code)
            filtered_skill_library.append(skill_item)

        return filtered_skill_library


    def add_new_skill(self,
                      skill_code,
                      overwrite = True,
                      trusted_source: bool = False):
        return self.skill_registry.register_skill_from_code(
            skill_code=skill_code,
            overwrite=overwrite,
            trusted_source=trusted_source,
        )


    def register_generated_skills(self, all_generated_actions) -> int:
        generated_codes = []
        for extracted_skills in all_generated_actions or []:
            if not isinstance(extracted_skills, dict):
                continue
            values = extracted_skills.get("values")
            if not isinstance(values, list):
                continue
            for extracted_skill in values:
                if not isinstance(extracted_skill, dict):
                    continue
                skill_code = extracted_skill.get("code")
                if isinstance(skill_code, str) and skill_code.strip():
                    generated_codes.append(skill_code)

        if not generated_codes:
            return 0

        allow_generated_registration = bool(
            config.skill_configs.get(
                constants.SKILL_CONFIG_ALLOW_GENERATED_REGISTRATION,
                False,
            )
        )
        if not allow_generated_registration:
            logger.warn(
                f"Blocked registration of {len(generated_codes)} model-generated skill(s); "
                f"set skill_configs.{constants.SKILL_CONFIG_ALLOW_GENERATED_REGISTRATION}=true "
                "only for trusted experiments."
            )
            return 0

        attempted = 0
        for skill_code in generated_codes:
            attempted += 1
            ok, info = self.add_new_skill(
                skill_code=skill_code,
                trusted_source=True,
            )
            if not ok:
                logger.warn(f"Generated skill registration failed: {info}")
        return attempted


    def delete_skill(self, skill_name):
        self.skill_registry.delete_skill(skill_name)


    def retrieve_skills(self, query_task, skill_num, screen_type):
        return self.skill_registry.retrieve_skills(query_task, skill_num, screen_type)


    def register_available_skills(self, candidates):
        self.skill_registry.register_available_skills(candidates)


    def get_skill_library_in_code(self, skill) -> Tuple[str, str]:
        return self.skill_registry.get_skill_code(skill)

    def convert_expression_to_skill(self, expression):
        return self.skill_registry.convert_expression_to_skill(expression)


    def execute_actions(self, actions, executer=None) -> Dict[str, Any]:
        if executer is None:
            executer = self.default_executer

        exec_info = {
            constants.EXECUTED_SKILLS: [],
            constants.LAST_SKILL: '',
            constants.ERRORS : False,
            constants.ERRORS_INFO: ""
        }

        io_env.update_timeouts()

        if isinstance(actions, str):
            actions = [actions]
        elif actions is None:
            actions = []

        cleaned_actions = []
        if isinstance(actions, list):
            for action in actions:
                action_text = str(action).strip()
                if action_text:
                    cleaned_actions.append(action_text)
        actions = cleaned_actions

        if actions is None or len(actions) == 0 or actions == '' or actions[0] == '':
            logger.warn(f"No actions to execute! Executing nop.")
            self.skill_registry.execute_nop_skill()

            exec_info[constants.ERRORS] = False
            exec_info[constants.ERRORS_INFO] = "empty_action_nop: LLM produced no executable actions, step wasted"
            return exec_info

        # Cross-plan same-action circuit breaker.
        # Normalize whitespace/quotes the same way the react agent does so
        # use(direction="down") and use(direction='down') count as identical.
        first_action_norm = self._normalize_action(actions[0])
        if (
            first_action_norm
            and first_action_norm == self._last_problem_action_text
            and self._consecutive_same_action_count >= self.SAME_ACTION_REFUSE_THRESHOLD
        ):
            refused_action = actions[0]
            recovery_action = self._pick_recovery_action(refused_action)
            warning = (
                f"CIRCUIT-BREAKER: action `{refused_action}` previously produced explicit failure "
                f"{self._consecutive_same_action_count} times in a row. This action is REFUSED for this step. "
                f"The next plan MUST choose a DIFFERENT action."
            )
            recovery_failure_note = ""
            logger.warn(warning)
            exec_info[constants.ERRORS] = True
            exec_info["refusal_type"] = "same_action_circuit_breaker"
            exec_info["refused_action"] = refused_action
            exec_info["refusal_count"] = self._consecutive_same_action_count

            if recovery_action and self._normalize_action(recovery_action) != first_action_norm:
                try:
                    rec_name, rec_params = self.skill_registry.convert_expression_to_skill(recovery_action)
                    recovery_response = self.skill_registry.execute_skill(
                        skill_name=rec_name,
                        skill_params=rec_params,
                        executer=executer,
                    )
                    if recovery_response is False:
                        raise RuntimeError(
                            f"recovery action `{recovery_action}` returned explicit failure"
                        )
                    if self._is_busy_timeout_response(recovery_response):
                        raise RuntimeError(
                            self._format_busy_timeout_error(recovery_action, recovery_response)
                        )
                    exec_info[constants.EXECUTED_SKILLS].append(
                        recovery_action + " # circuit-breaker-recovery"
                    )
                    exec_info[constants.LAST_SKILL] = recovery_action
                    self.post_action_wait(rec_name)
                except Exception as exc:
                    logger.warn(f"Circuit-breaker recovery failed: {exc}")
                    recovery_failure_note = f" Recovery action failed: {exc}"
                    self.skill_registry.execute_nop_skill()
                    exec_info[constants.EXECUTED_SKILLS].append("nop() # circuit-breaker")
                    exec_info[constants.LAST_SKILL] = "nop()"
            else:
                self.skill_registry.execute_nop_skill()
                exec_info[constants.EXECUTED_SKILLS].append("nop() # circuit-breaker")
                exec_info[constants.LAST_SKILL] = "nop()"

            exec_info[constants.ERRORS_INFO] = warning + recovery_failure_note
            return exec_info

        # Axis-direction circuit breaker for move() actions:
        # Catches move(x=3,y=0) → move(x=2,y=0) → move(x=1,y=0) pattern
        # where exact text differs but the direction is the same and all fail.
        move_axis = self._extract_move_axis_direction(first_action_norm)

        # Only trigger the axis-breaker after repeated CONFIRMED blocked moves
        # on the same axis. Normal multi-step navigation often repeats an axis
        # direction and must not be interrupted.
        AXIS_MOVE_THRESHOLD = 3
        if (
            move_axis is not None
            and move_axis == self._last_failed_move_axis_direction
            and self._consecutive_failed_move_axis_count >= AXIS_MOVE_THRESHOLD
        ):
            sidestep = self._pick_recovery_action(actions[0])
            warning = (
                f"AXIS-CIRCUIT-BREAKER: {self._consecutive_failed_move_axis_count} consecutive "
                f"blocked move() calls toward {move_axis}. "
                f"The path is blocked in this direction. Injecting recovery "
                f"move `{sidestep}`. Next plan MUST try a different direction."
            )
            recovery_failure_note = ""
            logger.warn(warning)
            try:
                rec_name, rec_params = self.skill_registry.convert_expression_to_skill(sidestep)
                recovery_response = self.skill_registry.execute_skill(
                    skill_name=rec_name, skill_params=rec_params, executer=executer
                )
                if recovery_response is False:
                    raise RuntimeError(
                        f"axis-breaker recovery `{sidestep}` returned explicit failure"
                    )
                if self._is_busy_timeout_response(recovery_response):
                    raise RuntimeError(
                        self._format_busy_timeout_error(sidestep, recovery_response)
                    )
                exec_info[constants.EXECUTED_SKILLS].append(sidestep + " # axis-breaker")
                exec_info[constants.LAST_SKILL] = sidestep
            except Exception as exc:
                logger.warn(f"Axis-breaker sidestep failed: {exc}")
                recovery_failure_note = f" Recovery move failed: {exc}"
                self.skill_registry.execute_nop_skill()
                exec_info[constants.EXECUTED_SKILLS].append("nop() # axis-breaker-failed")
                exec_info[constants.LAST_SKILL] = "nop()"
            exec_info[constants.ERRORS] = False
            exec_info[constants.ERRORS_INFO] = warning + recovery_failure_note
            exec_info["refusal_type"] = "axis_circuit_breaker"
            exec_info["refused_action"] = actions[0]
            exec_info["refusal_count"] = self._consecutive_failed_move_axis_count
            self._clear_failed_move_axis_streak()
            self._clear_problem_action_streak()
            try:
                self.post_action_wait("move")
            except Exception:
                pass
            return exec_info

        skill_name = '-'
        skill_params = '-'
        skill_response = None
        terminal_skills = set(getattr(config, "composite_terminal_skill_names", []) or [])
        first_action_recorded = False

        try:
            for skill in actions:

                if constants.INVALID_BBOX in skill:
                    exec_info[constants.ERRORS] = True
                    label_id = skill.split(": ")[1]
                    exec_info[constants.ERRORS_INFO] = f"Label ID {label_id} not found in SOM map."
                    if not first_action_recorded:
                        self._record_problem_action(skill)
                        self._clear_failed_move_axis_streak()
                    return exec_info

                skill_name, skill_params = self.skill_registry.convert_expression_to_skill(skill)

                if skill_name == "nop":
                    logger.write("Executing skill: nop with params: {}")
                    self.skill_registry.execute_nop_skill()
                    exec_info[constants.EXECUTED_SKILLS].append(skill)
                    exec_info[constants.LAST_SKILL] = skill
                    self.post_action_wait(skill_name)
                    logger.write(f"Finished executing skill: {skill} and wait.")
                    if not first_action_recorded:
                        self._clear_problem_action_streak()
                        self._clear_failed_move_axis_streak()
                        first_action_recorded = True
                    continue

                logger.write(f"Executing skill: {skill_name} with params: {skill_params}")

                # Enable OCR for composite skills, start the ocr check
                if skill_name in config.ocr_check_composite_skill_names:
                    if not config.ocr_fully_ban:
                        config.ocr_different_previous_text = False
                        config.enable_ocr = True
                    else:
                        config.ocr_different_previous_text = False
                        config.enable_ocr = False

                skill_response = self.skill_registry.execute_skill(skill_name=skill_name, skill_params=skill_params, executer = executer)

                if skill_name == "move" and skill_response is False:
                    # Build descriptive failure message with direction info
                    x = skill_params.get("x", 0)
                    y = skill_params.get("y", 0)
                    direction_parts = []
                    if y < 0:
                        direction_parts.append("up")
                    elif y > 0:
                        direction_parts.append("down")
                    if x < 0:
                        direction_parts.append("left")
                    elif x > 0:
                        direction_parts.append("right")
                    direction_str = "-".join(direction_parts) if direction_parts else "unknown"

                    logger.warn(f"move({x},{y}) toward {direction_str} did not change player position; "
                                "path may be blocked by an obstacle. Agent will re-plan.")
                    exec_info[constants.ERRORS] = False  # non-fatal
                    new_error = (
                        f"move(x={x}, y={y}) toward {direction_str} FAILED - "
                        f"player position did not change, path is likely blocked by an obstacle"
                    )
                    existing = str(exec_info.get(constants.ERRORS_INFO, "") or "").strip()
                    exec_info[constants.ERRORS_INFO] = (existing + " | " + new_error) if existing else new_error

                    # Stop executing remaining planned actions — the agent is
                    # at the wrong position, so subsequent steps are based on
                    # an invalid premise and would waste the action budget.
                    exec_info[constants.EXECUTED_SKILLS].append(skill)
                    exec_info[constants.LAST_SKILL] = skill
                    if not first_action_recorded:
                        self._record_problem_action(skill)
                        self._record_failed_move_axis(skill)
                        first_action_recorded = True
                    logger.write(
                        f"Blocked move terminates plan execution; "
                        f"skipping {len(actions) - actions.index(skill) - 1} remaining action(s) "
                        f"for fresh re-observation."
                    )
                    break

                elif self._is_busy_timeout_response(skill_response):
                    timeout_error = self._format_busy_timeout_error(skill, skill_response)
                    existing = str(exec_info.get(constants.ERRORS_INFO, "") or "").strip()
                    exec_info[constants.ERRORS] = True
                    exec_info[constants.ERRORS_INFO] = (
                        existing + " | " + timeout_error
                    ) if existing else timeout_error
                    exec_info[constants.EXECUTED_SKILLS].append(skill)
                    exec_info[constants.LAST_SKILL] = skill
                    if not first_action_recorded:
                        self._record_problem_action(skill)
                        self._clear_failed_move_axis_streak()
                        first_action_recorded = True
                    logger.write(
                        "Timed-out mod action terminates plan execution; "
                        f"skipping {len(actions) - actions.index(skill) - 1} remaining action(s) "
                        "for fresh re-observation."
                    )
                    break
                elif skill_name == "craft":
                    normalized_craft_response = self._normalize_craft_response(skill_response)
                    if not self._is_successful_craft_response(normalized_craft_response):
                        existing = str(exec_info.get(constants.ERRORS_INFO, "") or "").strip()
                        craft_error = self._format_craft_failure_error(
                            skill_params=skill_params if isinstance(skill_params, dict) else None,
                            skill_response=normalized_craft_response,
                        )
                        exec_info[constants.ERRORS] = bool(
                            exec_info.get(constants.ERRORS, False)
                            or self._is_explicit_craft_failure_response(normalized_craft_response)
                        )
                        exec_info[constants.ERRORS_INFO] = (
                            existing + " | " + craft_error
                        ) if existing else craft_error
                elif False and skill_name == "craft" and (
                    skill_response is None
                    or not str(skill_response or "").strip()
                ):
                    # craft() can meaningfully fail: missing materials,
                    # unknown recipe, wrong location. Provide feedback.
                    craft_item = skill_params.get("item", "?") if isinstance(skill_params, dict) else "?"
                    exec_info[constants.ERRORS_INFO] = (
                        exec_info.get(constants.ERRORS_INFO, "") or ""
                    ) + (f'craft(item="{craft_item}") returned no confirmation. '
                         f"Possible: missing materials or unknown recipe. "
                         f"craft() works directly — do NOT open crafting menu first. ")
                # NOTE: use(), interact(), choose_item(), attach_item() always
                # return None from the C# server (no confirmation protocol).
                # Previously we appended "no confirmation" warnings for every
                # call, but these are pure noise — they fire on EVERY use/
                # interact regardless of success. Removing them to keep the
                # prompt clean. The real safety net for stuck loops is the
                # problem-action streak tracker + axis circuit breaker above.
                elif skill_name == "menu" and skill_response is False:
                    menu_option = "?"
                    menu_name = "current_menu"
                    if isinstance(skill_params, dict):
                        menu_option = str(skill_params.get("option", menu_option) or menu_option)
                        menu_name = str(skill_params.get("menu_name", menu_name) or menu_name)
                    menu_error = (
                        f'menu(option="{menu_option}", menu_name="{menu_name}") '
                        "did not execute because the game window focus could not be confirmed"
                    )
                    existing = str(exec_info.get(constants.ERRORS_INFO, "") or "").strip()
                    exec_info[constants.ERRORS_INFO] = (
                        existing + " | " + menu_error
                    ) if existing else menu_error
                    exec_info[constants.EXECUTED_SKILLS].append(skill)
                    exec_info[constants.LAST_SKILL] = skill
                    if not first_action_recorded:
                        self._record_problem_action(skill)
                        self._clear_failed_move_axis_streak()
                        first_action_recorded = True
                    logger.write(
                        "Unconfirmed menu input terminates plan execution; "
                        f"skipping {len(actions) - actions.index(skill) - 1} remaining action(s) "
                        "for fresh re-observation."
                    )
                    break
                elif skill_name in terminal_skills and skill_response is False:
                    composite_error = f"{skill_name}() FAILED - composite skill reported failure"
                    existing = str(exec_info.get(constants.ERRORS_INFO, "") or "").strip()
                    exec_info[constants.ERRORS_INFO] = (
                        existing + " | " + composite_error
                    ) if existing else composite_error
                    if not first_action_recorded:
                        self._record_problem_action(skill)
                        self._clear_failed_move_axis_streak()
                        first_action_recorded = True

                if config.is_game is False:
                    skill = skill + " # " + f"""{str(skill_response)}""" if skill_response else skill

                exec_info[constants.EXECUTED_SKILLS].append(skill)
                exec_info[constants.LAST_SKILL] = skill

                self.post_action_wait(skill_name)
                logger.write(f"Finished executing skill: {skill} and wait.")
                if not first_action_recorded:
                    self._clear_problem_action_streak()
                    self._clear_failed_move_axis_streak()
                    first_action_recorded = True

                if skill_name in terminal_skills:
                    exec_info["composite_terminal_skill"] = skill_name
                    logger.write(
                        f"Terminal composite skill {skill_name} executed; "
                        "stopping remaining planned actions for fresh re-observation."
                    )
                    break

        except Exception as e:
            msg = f'Error executing skill {skill_name} with params {skill_params} (from actions: {actions}):\n{e}'
            logger.error(msg)
            exec_info[constants.ERRORS] = True
            exec_info[constants.ERRORS_INFO] = msg
            if not first_action_recorded and actions:
                self._record_problem_action(actions[0])
                if str(skill_name or "").strip() == "move":
                    self._record_failed_move_axis(actions[0])
                else:
                    self._clear_failed_move_axis_streak()

        # @TODO re-add hold timeout check call

        return exec_info


    # Currently all actions have wait in them, if needed
    def post_action_wait(self, skill_name: str = ""):
        wait_seconds = 3.0
        wait_overrides = getattr(config, "composite_skill_wait_seconds", {}) or {}
        if skill_name in wait_overrides:
            wait_seconds = float(wait_overrides[skill_name])
        time.sleep(wait_seconds)


    def get_out_screen(self):
        out_screen_file = "./res/software/samples/out_of_target_screen.jpg"
        full_path = assemble_project_path(out_screen_file)
        return full_path


    def capture_screen(self):
        tid = time.time()
        return self.ui_control.take_screenshot(tid)


    def get_mouse_position(self, absolute = False) -> Tuple[int, int]:
        return io_env.get_mouse_position(absolute)


    def list_session_screenshots(self, session_dir: str = config.work_dir):
        return io_env.list_session_screenshots(session_dir)


    def store_skills(self, path = None):
        self.skill_registry.store_skills(path)


    def load_skills(self, path = None):
        self.skill_registry.load_skill_library(path)


    def get_all_skills(self):
        return self.skill_registry.get_all_skills()


    def cleanup_io(self):
        io_env.release_held_keys()
        io_env.release_held_buttons()
