import os
from typing import Dict, Any, List
from copy import deepcopy

from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider import BaseProvider
from cradle.provider import VideoRecordProvider
from cradle.utils.check import is_valid_value
from cradle import constants

config = Config()
logger = Logger()


def _coerce_frame_id(frame_id: Any, default: int = -1) -> int:
    if frame_id is None:
        return default
    if isinstance(frame_id, int):
        return frame_id
    if isinstance(frame_id, float):
        return int(frame_id)
    try:
        return int(str(frame_id).strip())
    except (TypeError, ValueError):
        logger.warn(
            f"[SelfReflection] Invalid frame id {frame_id!r}, using {default}"
        )
        return default


class SelfReflectionPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 use_screenshot_augmented = False,
                 use_video = False,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.gm = gm
        self.memory = LocalMemory()
        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))

        self.use_screenshot_augmented = use_screenshot_augmented
        self.use_video = use_video

    def __call__(self):

        if not self.use_video:
            prompts = [
                "This screenshot is the previous observation before executing the last action.",
                "This screenshot is the current observation after executing the last action."
            ]

            screenshot_paths = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=config.action_planning_image_num)
            screenshot_augmnented_paths = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, k=config.action_planning_image_num)

            if not self.use_screenshot_augmented:
                image_introduction = []
                for i in range(len(screenshot_paths), 0, -1):
                    image_introduction.append(
                        {
                            "introduction": prompts[-i],
                            "path": screenshot_paths[-i],
                            "assistant": "",
                            "resolution": "low"
                        })
            else:
                image_introduction = []
                for i in range(len(screenshot_augmnented_paths), 0, -1):
                    image_introduction.append(
                        {
                            "introduction": prompts[-i],
                            "path": screenshot_augmnented_paths[-i],
                            "assistant": "",
                            "resolution": "low"
                        })

            processed_params = {
                "image_introduction": image_introduction
            }

        else:

            start_frame_id = _coerce_frame_id(
                self.memory.get_latest("start_frame_id", 0)
            )
            end_frame_id = _coerce_frame_id(
                self.memory.get_latest("end_frame_id", start_frame_id),
                default=start_frame_id,
            )

            action_frames = []
            video_frames = self.video_recorder.get_frames(start_frame_id, end_frame_id)

            if len(video_frames) <= config.max_images_in_self_reflection * config.duplicate_frames + 1:
                action_frames = [frame[1] for frame in video_frames[1::config.duplicate_frames]]
            else:
                for i in range(config.max_images_in_self_reflection):
                    step = len(video_frames) // config.max_images_in_self_reflection * i + 1
                    action_frames.append(video_frames[step][1])

            image_introduction = [
                {
                    "introduction": "Here are the sequential frames of the character executing the last action.",
                    "path": action_frames,
                    "assistant": "",
                    "resolution": "low"
                }
            ]

            actions = [self.memory.get_latest("actions", "")]
            action_code = ""
            action_str = ""

            if is_valid_value(actions):
                pre_action = actions[0]
                pre_action_name, _ = self.gm.skill_registry.convert_expression_to_skill(pre_action)
                action_str = pre_action_name
                action_code, action_code_info = self.gm.get_skill_library_in_code(pre_action_name)
                action_code = action_code if action_code is not None else action_code_info

            processed_params = {
                "image_introduction": image_introduction,
                "actions": action_str,
                "action_code": action_code
            }

        if hasattr(self.memory, "update_working_area"):
            self.memory.update_working_area(processed_params)
        else:
            self.memory.working_area.update(processed_params)

        return processed_params


class RDR2SelfReflectionPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.gm = gm
        self.memory = LocalMemory()
        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))


    def __call__(self):

        logger.write(f'RDR2 Self Reflection Preprocess')

        start_frame_id = _coerce_frame_id(
            self.memory.get_latest("start_frame_id", 0)
        )
        end_frame_id = _coerce_frame_id(
            self.memory.get_latest("end_frame_id", start_frame_id),
            default=start_frame_id,
        )
        task_description = self.memory.get_latest("task_description", "")
        pre_action = self.memory.get_latest("pre_action", "")
        pre_decision_making_reasoning = self.memory.get_latest("pre_decision_making_reasoning", "")
        exec_info = self.memory.get_latest("exec_info", {})
        skill_library = self.memory.get_latest("skill_library", "")

        processed_params = {
            "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id,
            "task_description": task_description,
            "skill_library": skill_library,
            "exec_info": exec_info,
            "pre_action": pre_action,
            "pre_decision_making_reasoning": pre_decision_making_reasoning
        }

        if start_frame_id > -1:
            action_frames = []
            video_frames = self.video_recorder.get_frames(start_frame_id, end_frame_id)

            if video_frames:
                if len(video_frames) <= config.max_images_in_self_reflection * config.duplicate_frames + 1:
                    action_frames = [frame[1] for frame in video_frames[1::config.duplicate_frames]]
                else:
                    for i in range(config.max_images_in_self_reflection):
                        step = len(video_frames) // config.max_images_in_self_reflection * i + 1
                        action_frames.append(video_frames[step][1])

                image_introduction = [
                    {
                        "introduction": "Here are the sequential frames of the character executing the last action.",
                        "path": action_frames,
                        "assistant": "",
                        "resolution": "low"
                    }]
            else:
                # 回退：使用最近截图
                screenshot_paths = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=2)
                screenshot_paths = [p for p in screenshot_paths if p]
                fallback_prompt = "Here is a recent screenshot of the game."
                image_introduction = []
                for i in range(len(screenshot_paths), 0, -1):
                    image_introduction.append({
                        "introduction": fallback_prompt,
                        "path": screenshot_paths[-i],
                        "assistant": "",
                        "resolution": "low"
                    })

            if pre_action:
                pre_action_name, pre_action_params = self.gm.convert_expression_to_skill(pre_action)

                # only input the pre_action name
                previous_action = pre_action_name
                action_code, action_code_info = self.gm.get_skill_library_in_code(pre_action_name)
                action_code = action_code if action_code is not None else action_code_info
            else:
                previous_action = ""
                action_code = ""

            if exec_info["errors"]:
                executing_action_error = exec_info["errors_info"]
            else:
                executing_action_error = ""

            processed_params.update({
                "image_introduction": image_introduction,
                "task_description": task_description,
                "skill_library": skill_library,
                "previous_reasoning": pre_decision_making_reasoning,
                "previous_action": previous_action,
                "action_code": action_code,
                "executing_action_error": executing_action_error
            })

        if hasattr(self.memory, "update_working_area"):
            self.memory.update_working_area(processed_params)
        else:
            self.memory.working_area.update(processed_params)

        return processed_params


class StardewSelfReflectionPreprocessProvider(BaseProvider):

    def __init__(self, *args,
                 gm: Any,
                 augment_methods,
                 **kwargs):

        super().__init__(*args, **kwargs)

        self.gm = gm
        self.memory = LocalMemory()
        self.video_recorder = VideoRecordProvider(os.path.join(config.work_dir, 'video.mp4'))

        self.augment_methods = augment_methods


    def augment_image(self, image):
        """
        Apply augmentation methods and normalize outputs to avoid dict-based paths
        in image_introduction.
        """
        for augment_method in self.augment_methods:
            # Prefer AugmentProvider.run(image=...) to avoid __call__ which ignores input
            try:
                if hasattr(augment_method, "run"):
                    image = augment_method.run(image=image)
                else:
                    # Fallback: try keyword argument first, then positional
                    try:
                        image = augment_method(image=image)
                    except TypeError:
                        image = augment_method(image)
            except Exception as e:
                logger.warn(f"Stardew Self Reflection Preprocess: augment failed: {e}")

            # Normalize dict outputs to a usable image/path
            if isinstance(image, dict):
                if image.get("augmented_image"):
                    image = image.get("augmented_image")
                elif image.get("screenshot_augmented_path"):
                    image = image.get("screenshot_augmented_path")
                elif image.get("path"):
                    image = image.get("path")
                elif image.get("image") is not None:
                    image = image.get("image")
        return image


    def __call__(self):

        logger.write(f'Stardew Self Reflection Preprocess')

        prompts = [
            "Here are the sequential frames of the character executing the last action."
        ]

        start_frame_id = _coerce_frame_id(
            self.memory.get_latest("start_frame_id", 0)
        )
        end_frame_id = _coerce_frame_id(
            self.memory.get_latest("end_frame_id", start_frame_id),
            default=start_frame_id,
        )
        task_description = self.memory.get_latest("task_description", "")
        pre_action = self.memory.get_latest("pre_action", "")
        pre_decision_making_reasoning = self.memory.get_latest("pre_decision_making_reasoning", "")
        exec_info = self.memory.get_latest("exec_info", {})
        skill_library = self.memory.get_latest("skill_library", "")
        datetime = self.memory.get_latest("datetime", "")
        toolbar_information = self.memory.get_latest("toolbar_information", "")
        previous_toolbar_information = self.memory.get_latest("previous_toolbar_information", "")
        history_summary = self.memory.get_latest("history_summary", "")
        subtask_description = self.memory.get_latest("subtask_description", "")
        subtask_reasoning = self.memory.get_latest("subtask_reasoning", "")

        processed_params = {
            "start_frame_id": start_frame_id,
            "end_frame_id": end_frame_id,
            "task_description": task_description,
            "skill_library": skill_library,
            "exec_info": exec_info,
            "pre_decision_making_reasoning": pre_decision_making_reasoning,
            "datetime": datetime,
            "toolbar_information": toolbar_information,
            "previous_toolbar_information": previous_toolbar_information,
            "history_summary": history_summary,
            "subtask_description": subtask_description,
            "subtask_reasoning": subtask_reasoning
        }

        if start_frame_id > -1:
            action_frames_pre = []
            action_frames_post = []
            video_frames = self.video_recorder.get_frames(start_frame_id, end_frame_id)

            if video_frames:
                # Take two frames before action (start) and two after action (end)
                pre_count = min(2, len(video_frames))
                post_count = min(2, len(video_frames))

                for i in range(pre_count):
                    action_frames_pre.append(self.augment_image(video_frames[i][1]))

                for i in range(post_count):
                    action_frames_post.append(self.augment_image(video_frames[-(i + 1)][1]))
                action_frames_post = list(reversed(action_frames_post))

                # Avoid duplicates when video has few frames
                if len(video_frames) <= 2:
                    action_frames_post = [f for f in action_frames_post if f not in action_frames_pre]

            image_introduction = [
                {
                    "introduction": "Here are the frames BEFORE executing the last action.",
                    "path": action_frames_pre,
                    "assistant": "",
                    "resolution": "low"
                },
                {
                    "introduction": "Here are the frames AFTER executing the last action.",
                    "path": action_frames_post,
                    "assistant": "",
                    "resolution": "low"
                }
            ]

            if pre_action:
                pre_action_name = []
                pre_action_code = []

                if isinstance(pre_action, str):
                    if "[" not in pre_action:
                        pre_action = "[" + pre_action + "]"
                elif isinstance(pre_action, list):
                    pre_action = "[" + ",".join(pre_action) + "]"

                for item in self.gm.convert_expression_to_skill(pre_action):
                    name, params = item
                    action_code, action_info = self.gm.get_skill_library_in_code(name)

                    pre_action_name.append(name)
                    pre_action_code.append(action_code if action_code is not None else action_info)
                previous_action = ",".join(pre_action_name)
                action_code = "\n".join(list(set(pre_action_code)))
            else:
                previous_action = ""
                action_code = ""

            if exec_info["errors"]:
                executing_action_error = exec_info["errors_info"]
            else:
                executing_action_error = ""

            processed_params.update({
                "image_introduction": image_introduction,
                "previous_action": previous_action,
                "action_code": action_code,
                "executing_action_error": executing_action_error,
                "previous_reasoning": pre_decision_making_reasoning,
            })

        if hasattr(self.memory, "update_working_area"):
            self.memory.update_working_area(processed_params)
        else:
            self.memory.working_area.update(processed_params)

        return processed_params


class SelfReflectionPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.memory = LocalMemory()

    def _infer_success_from_reasoning(self, reasoning: str) -> bool:
        """
        Infer success status from reasoning text with improved keyword detection.
        Returns True if the action appears to have succeeded, False otherwise.
        """
        reasoning_lower = reasoning.lower()

        # Failure indicators - check these first (higher priority)
        failure_phrases = [
            'not successful', 'unsuccessful', 'failed', 'failure',
            'did not', 'didn\'t', 'cannot', 'can\'t', 'could not', 'couldn\'t',
            'unable to', 'no change', 'no effect', 'not placed', 'not built',
            'not constructed', 'error', 'invalid', 'blocked', 'obstructed',
            'not completed', 'incomplete', 'not finished', 'retry', 'try again',
            'wrong', 'incorrect', 'nothing happened', 'no progress',
            'still need', 'still needs', 'not yet', 'pending'
        ]

        # Success indicators
        success_phrases = [
            'successful', 'successfully', 'success', 'completed', 'complete',
            'placed', 'built', 'constructed', 'created', 'done', 'finished',
            'achieved', 'accomplished', 'worked', 'effective', 'visible',
            'appeared', 'now shows', 'can see', 'road is', 'has been placed',
            'has been built', 'was placed', 'was built', 'is now', 'correctly',
            'properly', 'as expected', 'confirmed', 'verified'
        ]

        # Check for explicit failure first
        for phrase in failure_phrases:
            if phrase in reasoning_lower:
                logger.write(f"Detected failure phrase: '{phrase}'")
                return False

        # Check for success indicators
        for phrase in success_phrases:
            if phrase in reasoning_lower:
                logger.write(f"Detected success phrase: '{phrase}'")
                return True

        # Default to False if no clear indicators
        logger.write("No clear success/failure indicators found, defaulting to False")
        return False

    def __call__(self, response: Dict):

        processed_response = deepcopy(response)

        processed_response = {
            key: processed_response[key] for key in processed_response
        }

        reasoning = processed_response.get('reasoning') or processed_response.get('self_reflection') or processed_response.get('reflection') or ''

        # If success is missing, infer from reasoning instead of defaulting to False
        if 'success' not in processed_response or processed_response.get('success') is None:
            if reasoning:
                processed_response['success'] = self._infer_success_from_reasoning(reasoning)
                logger.write(f"Inferred success from reasoning: {processed_response['success']}")
            else:
                processed_response['success'] = False
                logger.write("Missing success and reasoning; defaulting success to False")
        else:
            # Normalize success value if it exists (strict truthy only)
            success_val = processed_response['success']
            if isinstance(success_val, str):
                processed_response['success'] = success_val.strip().lower() in ['true', 'yes', '1', 'success']
            else:
                processed_response['success'] = bool(success_val)
            logger.write(f"Using explicit success={processed_response['success']}")

        processed_response.update({
            "self_reflection_reasoning": reasoning
        })

        self.memory.update_info_history(processed_response)
        
        # Add experience to SA-KG after self-reflection validation
        try:
            # Get recent action and state information
            recent_action = self.memory.get_recent_history("action", k=1)
            screen_classification = self.memory.get_recent_history("screen_classification", k=1)
            task_description = self.memory.get_recent_history("task_description", k=1)
            info_summary = self.memory.get_recent_history("summarization", k=1)
            
            if recent_action and recent_action[0]:
                # Build state description
                state_desc = f"Screen: {screen_classification[0] if screen_classification else 'unknown'}, "
                state_desc += f"Task: {task_description[0] if task_description else 'none'}, "
                state_desc += f"Context: {info_summary[0] if info_summary else 'none'}"
                
                # Determine reward based on success
                success = processed_response.get('success', False)
                reward = 1.0 if success else -0.5
                
                # Add to SA-KG
                self.memory.add_experience_to_sakg(
                    state_description=state_desc,
                    action=str(recent_action[0]),
                    reward=reward,
                    success=success
                )
                logger.debug(f"SA-KG experience recorded: action={recent_action[0]}, success={success}")
        except Exception as e:
            logger.warn(f"Failed to add experience to SA-KG: {e}")

        return processed_response


class RDR2SelfReflectionPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.memory = LocalMemory()


    def __call__(self, response: Dict):

        logger.write(f'RDR2 Self Reflection Postprocess')

        processed_response = deepcopy(response)

        if 'reasoning' in response:
            self_reflection_reasoning = response['reasoning']
        elif 'self_reflection' in response:
            self_reflection_reasoning = response['self_reflection']
        elif 'reflection' in response:
            self_reflection_reasoning = response['reflection']
        else:
            self_reflection_reasoning = ""

        progress = response.get('progress', 0)
        status_summary = response.get('status_summary', "")

        processed_response.update({
            "self_reflection_reasoning": self_reflection_reasoning,
            "pre_self_reflection_reasoning": self_reflection_reasoning,
            "self_reflection_progress": progress,
            "self_reflection_status_summary": status_summary
        })

        self.memory.update_info_history(processed_response)

        return processed_response


class StardewSelfReflectionPostprocessProvider(BaseProvider):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.memory = LocalMemory()

    def _infer_success_from_reasoning(self, reasoning: str):
        reasoning_lower = (reasoning or "").lower()

        failure_phrases = [
            'not successful', 'unsuccessful', 'failed', 'failure',
            'did not', 'didn\'t', 'cannot', 'can\'t', 'could not', 'couldn\'t',
            'unable to', 'no change', 'no effect', 'not completed', 'incomplete',
            'blocked', 'obstructed', 'wrong', 'incorrect', 'nothing happened',
            'no progress', 'still need', 'still needs', 'not yet', 'pending',
            'stuck', 'in progress', 'in_progress'
        ]

        success_phrases = [
            'successful', 'successfully', 'success', 'completed', 'complete',
            'done', 'finished', 'achieved', 'worked', 'effective',
            'cleared', 'removed', 'chopped', 'cut down', 'broken'
        ]

        for phrase in failure_phrases:
            if phrase in reasoning_lower:
                return False

        for phrase in success_phrases:
            if phrase in reasoning_lower:
                return True

        return None


    def __call__(self, response: Dict):

        logger.write(f'Stardew Self Reflection Postprocess')

        processed_response = deepcopy(response)

        if 'reasoning' in response:
            self_reflection_reasoning = response['reasoning']
        else:
            self_reflection_reasoning = ""

        status = response.get('status')
        if isinstance(status, str):
            status = status.strip().lower()
        else:
            status = None

        if 'success' in response and response.get('success') is not None:
            success_val = response.get('success')
            if isinstance(success_val, str):
                success_val = success_val.strip().lower() in ['true', 'yes', '1', 'success']
            else:
                success_val = bool(success_val)
        elif status in ['success', 'failure', 'in_progress']:
            success_val = True if status == 'success' else False if status == 'failure' else None
        else:
            success_val = self._infer_success_from_reasoning(self_reflection_reasoning)

        # For Stardew execution loops, ambiguous in_progress should not be left as None.
        # None causes memory store skip and weak correction signal, which can lead to repeated invalid actions.
        if success_val is None:
            inferred = self._infer_success_from_reasoning(self_reflection_reasoning)
            success_val = inferred if inferred is not None else False
            logger.write(f"Stardew self-reflection normalized ambiguous success to: {success_val}")

        if status not in ['success', 'failure', 'in_progress']:
            if success_val is True:
                status = 'success'
            elif success_val is False:
                status = 'failure'
            else:
                status = 'in_progress'

        processed_response.update({
            "self_reflection_reasoning": self_reflection_reasoning,
            "pre_self_reflection_reasoning": self_reflection_reasoning,
            "self_reflection_progress": response.get('progress', 0),
            "self_reflection_status_summary": response.get('status_summary', ''),
            "success": success_val,
            "status": status
        })

        self.memory.update_info_history(processed_response)

        return processed_response


class SkylinesSelfReflectionPreprocessProvider(BaseProvider):
    """Skylines-specific preprocessor that provides all required parameters for self-reflection."""

    def __init__(self, *args,
                 gm: Any,
                 use_screenshot_augmented=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.gm = gm
        self.memory = LocalMemory()
        self.use_screenshot_augmented = use_screenshot_augmented

    def __call__(self):

        logger.write("Skylines Self Reflection Preprocess")

        prompts = [
            "This screenshot is the previous observation before executing the last action.",
            "This screenshot is the current observation after executing the last action."
        ]

        # Get screenshot paths
        screenshot_paths = self.memory.get_recent_history(constants.IMAGES_MEM_BUCKET, k=config.action_planning_image_num)
        screenshot_augmented_paths = self.memory.get_recent_history(constants.AUGMENTED_IMAGES_MEM_BUCKET, k=config.action_planning_image_num)

        if not self.use_screenshot_augmented:
            image_introduction = []
            for i in range(len(screenshot_paths), 0, -1):
                image_introduction.append({
                    "introduction": prompts[-i],
                    "path": screenshot_paths[-i],
                    "assistant": "",
                    "resolution": "low"
                })
        else:
            image_introduction = []
            paths = screenshot_augmented_paths if screenshot_augmented_paths else screenshot_paths
            for i in range(len(paths), 0, -1):
                image_introduction.append({
                    "introduction": prompts[-i],
                    "path": paths[-i],
                    "assistant": "",
                    "resolution": "low"
                })

        # Get parameters from memory with safe defaults
        def safe_get(key, default=""):
            return self.memory.get_latest(key, default)

        # Required parameters for Skylines self_reflection.prompt
        task_description = safe_get("task_description", "Build and manage the city")
        subtask_description = safe_get("subtask_description", "")
        if not subtask_description:
            subtask_description = task_description

        coordinates = safe_get("coordinates", "No buildings constructed yet.")
        actions = safe_get("actions", "No previous action.")
        error_message = safe_get("error_message", "")
        construction_information = safe_get("construction_information", "")
        history_summary = safe_get("history_summary", safe_get("summarization", "This is the beginning of the task."))

        processed_params = {
            "image_introduction": image_introduction,
            "task_description": task_description,
            "subtask_description": subtask_description,
            "coordinates": coordinates,
            "actions": actions,
            "error_message": error_message,
            "construction_information": construction_information,
            "history_summary": history_summary,
        }

        if hasattr(self.memory, "update_working_area"):
            self.memory.update_working_area(processed_params)
        else:
            self.memory.working_area.update(processed_params)

        return processed_params
