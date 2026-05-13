import os
import time
from typing import Dict, Any
from copy import deepcopy

from cradle.config.config import Config
from cradle.provider import BaseProvider
from cradle.log import Logger
from cradle.memory import LocalMemory
from cradle.provider import VideoRecordProvider
from cradle import constants

config = Config()
logger = Logger()


class SkillExecuteProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 video_recorder: Any = None,
                 use_unpause_game: bool = False,
                 **kwargs):

        super(SkillExecuteProvider, self).__init__()

        self.gm = gm
        self.memory = LocalMemory()
        self.video_recorder = video_recorder if video_recorder is not None else VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))


    @BaseProvider.write
    def __call__(self,
                 *args,
                 **kwargs) -> Dict[str, Any]:

        params = self.memory.get_working_area_snapshot()

        skill_steps = params.get("skill_steps", [])
        pre_screen_classification = params.get("pre_screen_classification", "")
        screen_classification = params.get("screen_classification", "")
        pre_action = params.get("pre_action", "")

        # 修复：支持 select_tool(tool="hoe") 形式，映射为 select_tool(key="2")
        try:
            toolbar_text = self.memory.get_latest("toolbar_information", "")
            if toolbar_text and isinstance(skill_steps, list):
                import re
                tool_map = {}
                for line in toolbar_text.splitlines():
                    match = re.match(r"\s*(\d+)\.\s*([A-Za-z\s]+):", line)
                    if match:
                        slot = match.group(1).strip()
                        name = match.group(2).strip().lower()
                        tool_map[name] = slot

                converted_steps = []
                STARDEW_DEFAULT_TOOL_KEYS = {
                    "axe": "1", "hoe": "2", "watering can": "3",
                    "pickaxe": "4", "scythe": "5",
                }
                for step in skill_steps:
                    if isinstance(step, str) and step.startswith("select_tool") and "tool=" in step:
                        tool_match = re.search(r"tool\s*=\s*['\"]([^'\"]+)['\"]", step)
                        if tool_match:
                            tool_name = tool_match.group(1).strip().lower()
                            slot = tool_map.get(tool_name)
                            if not slot:
                                slot = STARDEW_DEFAULT_TOOL_KEYS.get(tool_name)
                                if slot:
                                    logger.debug(f"Mapped select_tool tool='{tool_name}' -> key='{slot}' (fallback default)")
                            else:
                                logger.debug(f"Mapped select_tool tool='{tool_name}' -> key='{slot}'")
                            if slot:
                                step = f"select_tool(key='{slot}')"
                            else:
                                logger.warn(f"[Tool-Mapping] Unknown tool '{tool_name}', skipping invalid select_tool call")
                                continue  # Skip this step entirely to avoid crash
                    converted_steps.append(step)
                skill_steps = converted_steps
        except Exception as map_error:
            logger.warn(f"[DEBUG] select_tool mapping failed: {map_error}")

        self.gm.unpause_game()

        # @TODO: Rename GENERAL_GAME_INTERFACE
        if (pre_screen_classification.lower() == constants.GENERAL_GAME_INTERFACE and
                (screen_classification.lower() == constants.MAP_INTERFACE or
                 screen_classification.lower() == constants.SATCHEL_INTERFACE) and pre_action):
            exec_info = self.gm.execute_actions([pre_action])

        start_frame_id = self.video_recorder.get_current_frame_id()
        exec_info = self.gm.execute_actions(skill_steps)

        # Wait for game to render the result of the action
        time.sleep(0.5)

        # Record end frame ID for video clip extraction
        # Note: We don't capture screenshot here anymore because the next loop iteration
        # will capture the latest screenshot immediately. This avoids redundant screenshots.
        end_frame_id = self.video_recorder.get_current_frame_id()

        try:
            pause_flag = self.gm.pause_game(screen_classification.lower())
            logger.write(f'Pause flag: {pause_flag}')
            if not pause_flag:
                self.gm.pause_game(screen_type=None)
        except Exception as e:
            logger.write(f"Error while pausing the game: {e}")

        # exec_info also has the list of successfully executed skills. skill_steps is the full list, which may differ if there were execution errors.
        pre_action = exec_info["last_skill"]
        pre_screen_classification = screen_classification

        logger.write(f"Execute skill steps by frame id ({start_frame_id}, {end_frame_id}).")

        res_params = {
            "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id,
            # screenshot_path will be updated by the next loop iteration's capture_screen()
            "pre_action": pre_action,
            "pre_screen_classification": pre_screen_classification,
            "exec_info": exec_info,
        }

        # logger.write(f"[DEBUG] skill_execute: Frame IDs recorded ({start_frame_id}, {end_frame_id})")
        # logger.write(f"[DEBUG] skill_execute: Screenshot will be captured at next loop iteration")

        self.memory.update_info_history(res_params)

        del params

        return res_params
