import json
import os
import threading
from typing import Dict, Any, List, Optional, cast
import time
import asyncio
import re
from cradle.config import Config
from cradle.log import Logger
from cradle.planner.base import BasePlanner
from cradle.memory import LocalMemory
from cradle.utils.check import check_planner_params
from cradle.utils.file_utils import assemble_project_path, read_resource_file

# Thread-local persistent event loop to avoid "Event loop is closed" errors.
# httpx AsyncClient (used by LangChain ChatOpenAI) outlives any per-call loop.
# By reusing a single loop per thread, the client can clean up naturally.
_thread_loop_storage = threading.local()


def _get_thread_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop for the current thread."""
    loop = getattr(_thread_loop_storage, 'loop', None)
    if loop is None or loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _thread_loop_storage.loop = loop
    return loop
from cradle.utils.json_utils import load_json, parse_semi_formatted_text, JsonFrameStructure
from cradle.utils.template_matching import match_templates_images, selection_box_identifier
from cradle import constants

config = Config()
logger = Logger()

PROMPT_EXT = ".prompt"
JSON_EXT = ".json"


async def gather_information_get_completion_parallel(llm_provider, text_input_map, current_frame_path, time_stamp,
                                                     text_input, get_text_template, i,video_prefix,gathered_information_JSON):

    logger.write(f"Start gathering text information from the {i + 1}th frame")

    text_input = text_input_map if text_input is None else text_input
    image_introduction = text_input["image_introduction"]

    # Set the last frame path as the current frame path
    image_introduction[-1] = {
        "introduction": image_introduction[-1]["introduction"],
        "path": f"{current_frame_path}",
        "assistant": image_introduction[-1]["assistant"]
    }
    text_input["image_introduction"] = image_introduction
    message_prompts = llm_provider.assemble_prompt(template_str=get_text_template, params=text_input)

    logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

    success_flag = False
    max_retries = 5
    retry_count = 0
    while not success_flag and retry_count < max_retries:
        try:
            # 🚀 P0 Fix: 检查是否支持streaming
            use_streaming = False
            if hasattr(llm_provider, 'create_completion_streaming'):
                try:
                    import yaml
                    from cradle.utils.file_utils import assemble_project_path
                    config_path = assemble_project_path('./conf/enhanced_config.yaml')
                    if os.path.exists(config_path):
                        with open(config_path, 'r', encoding='utf-8') as f:
                            cfg = yaml.safe_load(f)
                            use_streaming = cfg.get('performance', {}).get('streaming', {}).get('enabled', False)
                except Exception:
                    pass

            if use_streaming:
                logger.write(f"[Streaming] ✅ Using streaming for frame {i + 1}")
                # 直接await异步streaming方法 - 返回tuple (response, info)
                response, info = await llm_provider.create_completion_streaming(message_prompts)
            else:
                response, info = await llm_provider.create_completion_async(message_prompts)

            logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')

            # Convert the response to dict
            processed_response = parse_semi_formatted_text(response)
            success_flag = True
        except Exception as e:
            retry_count += 1
            logger.error(f"Response is not in the correct format: {e}, retry {retry_count}/{max_retries}")
            if retry_count >= max_retries:
                logger.error("Max retries reached, returning empty result")
                processed_response = {}
                break
            success_flag = False

            # wait 2 seconds for the next request and retry
            await asyncio.sleep(2)

    # Convert the response to dict
    if processed_response is None or len(response) == 0:
        logger.warn('Empty response in gather text information call')
        logger.debug(f"response={response}, processed_response={processed_response}")

    objects = processed_response
    objects_index = str(video_prefix) + '_' + str(time_stamp)
    gathered_information_JSON.add_instance(objects_index, objects)
    logger.write(f"Finish gathering text information from the {i + 1}th frame")

    return True


def gather_information_get_completion_sequence(llm_provider, text_input_map, current_frame_path, time_stamp,
                                               text_input, get_text_template, i, video_prefix, gathered_information_JSON):

    logger.write(f"Start gathering text information from the {i + 1}th frame")
    text_input = text_input_map if text_input is None else text_input

    image_introduction = text_input["image_introduction"]

    # Set the last frame path as the current frame path
    image_introduction[-1] = {
        "introduction": image_introduction[-1]["introduction"],
        "path": f"{current_frame_path}",
        "assistant": image_introduction[-1]["assistant"]
    }
    text_input["image_introduction"] = image_introduction

    message_prompts = llm_provider.assemble_prompt(template_str=get_text_template, params=text_input)

    logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

    # 🚀 P0 Note: 同步上下文无法使用async streaming，回退到标准方法
    # Streaming需要await，但这是同步函数，所以暂时禁用
    response, info = llm_provider.create_completion(message_prompts)

    logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')
    success_flag = False
    max_retries = 3
    retry_count = 0
    while not success_flag and retry_count < max_retries:
        try:
            # Convert the response to dict
            processed_response = parse_semi_formatted_text(response)
            success_flag = True
        except Exception as e:
            retry_count += 1
            logger.error(f"Response is not in the correct format: {e}, retry {retry_count}/{max_retries}")
            if retry_count >= max_retries:
                logger.error("Max retries reached for parsing response, returning empty result")
                processed_response = {}
                break
            success_flag = False

            time.sleep(2)

    # Convert the response to dict
    if processed_response is None or len(response) == 0:
        logger.warn('Empty response in gather text information call')
        logger.debug(f"response={response}, processed_response={processed_response}")

    objects = processed_response
    objects_index = str(video_prefix) + '_' + time_stamp
    gathered_information_JSON.add_instance(objects_index, objects)

    logger.write(f"Finish gathering text information from the {i + 1}th frame")

    return True


async def get_completion_in_parallel(llm_provider, text_input_map, extracted_frame_paths, text_input,get_text_template,video_prefix,gathered_information_JSON):
    # Use semaphore to limit concurrent API requests and avoid 429 rate limit errors
    semaphore = asyncio.Semaphore(3)

    async def rate_limited_task(coro):
        async with semaphore:
            result = await coro
            await asyncio.sleep(1)  # Small delay between requests
            return result

    tasks = []

    for i, (current_frame_path, time_stamp) in enumerate(extracted_frame_paths):

        task = gather_information_get_completion_parallel(llm_provider, text_input_map, current_frame_path, time_stamp,
                                                   text_input, get_text_template, i,video_prefix,gathered_information_JSON)

        tasks.append(rate_limited_task(task))

    return await asyncio.gather(*tasks)


async def get_completion_in_parallel_tool(
        llm_provider,
        text_input_map,
        extracted_frame_paths,
        inventory_names,
        text_input,
        get_text_template,
        video_prefix,
        gathered_information_JSON,
):
    # Use semaphore to limit concurrent API requests and avoid 429 rate limit errors
    semaphore = asyncio.Semaphore(3)

    async def rate_limited_task(coro):
        async with semaphore:
            result = await coro
            await asyncio.sleep(1)
            return result

    tasks = []

    for i, (current_frame_path) in enumerate(extracted_frame_paths):
        inventory_name = inventory_names[i]

        text_input["image_introduction"][0]["inventory_name"] = inventory_name
        task = gather_information_get_completion_parallel(
            llm_provider,
            text_input_map,
            current_frame_path,
            i,
            text_input,
            get_text_template,
            i,
            video_prefix,
            gathered_information_JSON,
        )

        tasks.append(rate_limited_task(task))

    return await asyncio.gather(*tasks)

def get_completion_in_sequence(llm_provider, text_input_map, extracted_frame_paths, text_input, get_text_template,
                               video_prefix, gathered_information_JSON):
    for i, (current_frame_path, time_stamp) in enumerate(extracted_frame_paths):
        gather_information_get_completion_sequence(llm_provider, text_input_map, current_frame_path, time_stamp,
                                                   text_input, get_text_template, i,video_prefix,gathered_information_JSON)

    return True


class InformationGathering():

    STARDEW_ORIGINAL_ICON_LIST = [
        os.path.join("./res/stardew/icons/inventory", f) for f in os.listdir("./res/stardew/icons/inventory")
    ]

    def __init__(
            self,
            input_map: Optional[Dict] = None,
            template: Optional[str] = None,
            icon_replacer: Any = None,
            object_detector: Any = None,
            llm_provider: Any = None,
            text_input_map: Optional[Dict] = None,
            get_text_template: Optional[str] = None,
            toolbar_input_map: Optional[Dict] = None,
            get_toolbar_template: Optional[str] = None,
            frame_extractor: Any = None,
    ):

        self.input_map = input_map
        self.template = template
        self.icon_replacer = icon_replacer
        self.object_detector = object_detector
        self.llm_provider = llm_provider
        self.text_input_map = text_input_map
        self.get_text_template = get_text_template
        self.toolbar_input_map = toolbar_input_map
        self.get_toolbar_template = get_toolbar_template
        self.frame_extractor = frame_extractor


    def _pre(self, *args, input: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        return input or {}


    def __call__(self, *args, input: Optional[Dict[str, Any]] = None, class_=None, **kwargs) -> Dict[str, Any]:
        # 🔍 诊断：在函数开始时检查memory状态
        memory = LocalMemory()
        logger.debug(f"[MEMORY CHECK] working_area keys: {list(memory.working_area.keys())}")
        logger.debug(f"[MEMORY CHECK] target_frame_count in memory: {memory.working_area.get('target_frame_count')}")
        
        frame_extractor_gathered_information = None
        icon_replacer_gathered_information = None
        object_detector_gathered_information = None
        llm_description_gathered_information = None

        input = self.input_map if input is None else input
        input = self._pre(input=input)

        gather_information_configurations = input["gather_information_configurations"]
        cur_inventories_shot_paths = input["cur_inventories_shot_paths"]

        image_files = []
        if "image_introduction" in input.keys():
            for image_info in input["image_introduction"]:
                image_files.append(image_info["path"])

        # flag = True
        processed_response = {}

        # Gather information by frame extractor
        if gather_information_configurations[constants.FRAME_EXTRACTOR] is True:

            logger.write(f"Using frame extractor to gather information")

            if self.frame_extractor is not None:

                text_input = input["text_input"]
                video_path = input["video_clip_path"]

                if "test_text_image" in input.keys() and input["test_text_image"]:  # offline test
                    extracted_frame_paths = input["test_text_image"]

                else:  # online run
                    # extract the text information of the whole video
                    # run the frame_extractor to get the key frames
                    extracted_frame_paths = self.frame_extractor.extract(video_path=video_path)

                # Gather information by Icon replacer
                if gather_information_configurations["icon_replacer"] is True:
                    logger.write(f"Using icon replacer to gather information")
                    if self.icon_replacer is not None:
                        try:
                            extracted_frame_paths = self._replace_icon(extracted_frame_paths)
                        except Exception as e:
                            logger.error(f"Error in gather information by Icon replacer: {e}")
                            flag = False
                    else:
                        logger.warn('Icon replacer is not set, skipping gather information by Icon replacer')

                # ✂️ 动态帧数优化：按target_frame_count截断帧列表
                logger.write("=" * 60)
                logger.write("🎯 [FRAME TRUNCATION] Starting...")
                try:
                    memory = LocalMemory()
                    target_frame_count = memory.working_area.get("target_frame_count")
                    original_count = len(extracted_frame_paths)
                    
                    logger.write(f"🎯 [FRAME TRUNCATION] Original frame count: {original_count}")
                    logger.write(f"🎯 [FRAME TRUNCATION] Target frame count from memory: {target_frame_count}")
                    logger.write(f"🎯 [FRAME TRUNCATION] Type check: {type(target_frame_count)} (should be int)")
                    
                    if isinstance(target_frame_count, int) and target_frame_count > 0:
                        extracted_frame_paths = extracted_frame_paths[:target_frame_count]
                        logger.write(f"✅ [FRAME TRUNCATION] SUCCESS! Truncated {original_count} → {len(extracted_frame_paths)} frames")
                        logger.write(f"✅ [FRAME TRUNCATION] Will process frames 1-{len(extracted_frame_paths)}")
                    else:
                        logger.write(f"⚠️  [FRAME TRUNCATION] SKIPPED! Invalid target_frame_count: {target_frame_count}")
                        logger.write(f"⚠️  [FRAME TRUNCATION] Will process all {original_count} frames")
                except Exception as e:
                    logger.error(f"❌ [FRAME TRUNCATION] FAILED with exception: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                logger.write("=" * 60)

                # For each keyframe, use llm to get the text information
                video_prefix = os.path.basename(video_path).split('.')[0].split('_')[-1]  # Different video should have differen prefix for avoiding the same time stamp
                frame_extractor_gathered_information = JsonFrameStructure()

                if config.parallel_request_gather_information:
                    # Create completions in parallel
                    logger.write(f"Start gathering text information from the whole video in parallel")

                    loop = _get_thread_loop()

                    try:
                        loop.run_until_complete(
                            get_completion_in_parallel(self.llm_provider, self.text_input_map, extracted_frame_paths,
                                                       text_input,self.get_text_template,video_prefix,frame_extractor_gathered_information))

                    except KeyboardInterrupt:

                        tasks = [t for t in asyncio.all_tasks(loop) if t is not asyncio.current_task()]
                        for task in tasks:
                            task.cancel()

                        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))

                else:
                    logger.write(f"Start gathering text information from the whole video in sequence")
                    get_completion_in_sequence(self.llm_provider, self.text_input_map, extracted_frame_paths,
                                               text_input,self.get_text_template,video_prefix,frame_extractor_gathered_information)

                frame_extractor_gathered_information.sort_index_by_timestamp()
                logger.write(f"Finish gathering text information from the whole video")

            else:
                logger.warn('Frame extractor is not set, skipping gather information by frame extractor')
                frame_extractor_gathered_information = None

            # Get dialogue information from the gathered_information_JSON at the subfounder find the dialogue frames
            if frame_extractor_gathered_information is not None:
                dialogues = [item["values"] for item in frame_extractor_gathered_information.search_type_across_all_indices("dialogue")]
            else:
                if self.frame_extractor is not None:
                    msg = "No gathered_information_JSON received, so no dialogue information is provided."
                else:
                    msg = "No gathered_information_JSON available, no Frame Extractor in use."

                logger.warn(msg)
                dialogues = []

            # Update the <$task_description$> in the gather_information template with the latest task_description
            all_task_guidance = []
            if frame_extractor_gathered_information is not None:
                all_task_guidance = frame_extractor_gathered_information.search_type_across_all_indices(constants.TASK_GUIDANCE)

            # Remove the content of "task is none"
            all_task_guidance = [task_guidance for task_guidance in all_task_guidance if constants.NONE_TASK_OUTPUT not in task_guidance["values"].lower()]

            if len(all_task_guidance) != 0:
                # New task guidance is found, use the latest one
                last_task_guidance = max(all_task_guidance, key=lambda x: x['index'])['values']
                input[constants.TASK_DESCRIPTION] = last_task_guidance  # this is for the input of the gather_information

            # @TODO: summary the dialogue and use it

        # Gather information of the toolbar
        # 1.identify new item in the toolbar
        # TODO: identify new item in the toolbar (still not complete)
        # new_icon_template_list = self.gather_information_of_new_icon(cur_new_icon_image_shot_path,cur_new_icon_name_image_shot_path)
        new_icon_template_list = []

        # run gather toolbar info and llm_description in parallel
        # 🔧 Phase 2.1 Fix: 避免asyncio.run()嵌套，使用线程池执行async代码
        try:
            loop = asyncio.get_running_loop()
            # 已有running loop，在新线程中创建新loop执行
            import concurrent.futures

            def run_in_thread():
                tloop = _get_thread_loop()
                return tloop.run_until_complete(
                    self.execute_parallel(cur_inventories_shot_paths, gather_information_configurations, input)
                )

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                results = future.result()

        except RuntimeError:
            # 没有running loop，直接用当前线程的持久loop
            loop = _get_thread_loop()
            results = loop.run_until_complete(self.execute_parallel(cur_inventories_shot_paths, gather_information_configurations, input))
        
        toolbar_dict_list, selected_position, processed_response, flag = results
        llm_description_gathered_information=processed_response

        # Assemble the gathered_information_JSON

        if flag:
            objects = []

            if icon_replacer_gathered_information is not None and "objects" in icon_replacer_gathered_information:
                objects.extend(icon_replacer_gathered_information["objects"])
            if object_detector_gathered_information is not None and "objects" in object_detector_gathered_information:
                objects.extend(object_detector_gathered_information["objects"])
            if llm_description_gathered_information is not None and "objects" in llm_description_gathered_information:
                objects.extend(llm_description_gathered_information["objects"])

            objects = list(set(objects))

            processed_response["objects"] = objects
            processed_response['toolbar_dict_list'] = toolbar_dict_list
            processed_response['selected_position'] = selected_position

            # Merge the gathered_information_JSON to the processed_response
            processed_response["gathered_information_JSON"] = frame_extractor_gathered_information

            if gather_information_configurations[constants.FRAME_EXTRACTOR] is True:
                if len(all_task_guidance) == 0:
                    processed_response[constants.LAST_TASK_GUIDANCE] = ""
                else:
                    processed_response[constants.LAST_TASK_GUIDANCE] = last_task_guidance

        # Gather information by object detector, which is grounding dino.
        if gather_information_configurations[constants.OBJECT_DETECTOR] is True:
            logger.write(f"Using object detector to gather information")
            if self.object_detector is not None:
                try:
                    target_object_name = processed_response[constants.TARGET_OBJECT_NAME].lower() \
                        if constants.NONE_TARGET_OBJECT_OUTPUT not in processed_response[constants.TARGET_OBJECT_NAME].lower() else ""

                    image_source, boxes, logits, phrases = self.object_detector.detect(image_path=image_files[0],
                                                                                       text_prompt= target_object_name,
                                                                                       box_threshold=0.4, device='cuda')
                    processed_response["boxes"] = boxes
                    processed_response["logits"] = logits
                    processed_response["phrases"] = phrases
                except Exception as e:
                    logger.error(f"Error in gather information by object detector: {e}")
                    flag = False

                try:
                    minimap_detection_objects = self.object_detector.process_minimap_targets(image_files[0])

                    processed_response.update({constants.MINIMAP_INFORMATION:minimap_detection_objects})

                except Exception as e:
                    logger.error(f"Error in gather information by object detector for minimap: {e}")
                    flag = False

        success = self._check_success(data=processed_response)

        data = dict(
            flag=flag,
            success=success,
            input=input,
            res_dict=processed_response,
        )

        data = self._post(data=data)

        return data

    def _post(self, *args, data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        return data or {}


    def _check_success(self, *args, data, **kwargs):

        success = False

        prop_name = "description"

        if prop_name in data.keys():
            desc = data[prop_name]
            success = desc is not None and len(desc) > 0
        return success


    def _replace_icon(self, extracted_frame_paths):
        extracted_frames = [frame[0] for frame in extracted_frame_paths]
        extracted_timesteps = [frame[1] for frame in extracted_frame_paths]
        extracted_frames = self.icon_replacer(image_paths=extracted_frames)
        extracted_frame_paths = list(zip(extracted_frames, extracted_timesteps))
        return extracted_frame_paths


    def gather_information_of_new_icon(self, cur_new_icon_image_shot_path, cur_new_icon_name_image_shot_path):
        # if there is a new icon in the screenshot, save it to the workdir for later template matching

        # request the llm to decide if there is a new icon and get the name of the new icon

        # if there is a new icon, rename it with the name in LLM response

        # if LLM response is empty, delete the new icon images

        # return the list of icon paths

        pass

    async def template_matching_for_current_toolbar(self, sr_file_list, base_template_file_list, work_template_file_list):
        matching_dict = cast(Dict[str, str], match_templates_images(sr_file_list, base_template_file_list, work_template_file_list))
        selected_position=None
        for sr_file in sr_file_list:
            is_selected=selection_box_identifier(sr_file, config.selection_box_region)
            if is_selected:
                selected_position=sr_file_list.index(sr_file)+1
                break
        for key, value in list(matching_dict.items()):
            matching_dict[key] = os.path.splitext(os.path.basename(value))[0]
        return matching_dict,selected_position

    async def gather_toolbar_list(self, match_dict, get_number_flag=True):
        any_key = next(iter(match_dict.keys()))
        video_prefix = any_key.split("/")[-2]
        frame_paths = []
        for path in match_dict:
            frame_paths.append(path)
        names = []
        for path in match_dict:
            names.append(match_dict[path])
        
        # 🔥 CRITICAL FIX: 应用帧数截断（P2优化）
        memory = LocalMemory()
        target_frame_count = memory.working_area.get("target_frame_count")
        if isinstance(target_frame_count, int) and target_frame_count > 0:
            original_count = len(frame_paths)
            frame_paths = frame_paths[:target_frame_count]
            names = names[:target_frame_count]
            logger.write(f"✅ [FRAME TRUNCATION - TOOLBAR] Truncated {original_count} → {len(frame_paths)} frames")
        else:
            logger.write(f"⚠️  [FRAME TRUNCATION - TOOLBAR] SKIPPED (target={target_frame_count})")

        frame_extractor_gathered_information = JsonFrameStructure()
        text_input = self.toolbar_input_map

        if get_number_flag:
            # Create completions in parallel
            logger.write(
                f"Start gathering text information from the whole video in parallel"
            )

            await get_completion_in_parallel_tool(
                self.llm_provider,
                self.toolbar_input_map,
                frame_paths,
                names,
                text_input,
                self.get_toolbar_template,
                video_prefix,
                frame_extractor_gathered_information,
            )

            inventory_index_list = []
            item_number_list = []
            for key_1 in frame_extractor_gathered_information.data_structure:
                for key_2 in frame_extractor_gathered_information.data_structure[key_1]:
                    pattern = r"_([0-9]+)$"
                    match = re.search(pattern, key_2)
                    if match is None:
                        continue
                    inventory_index = match.group(1)
                    contents = frame_extractor_gathered_information.data_structure[key_1][key_2]
                    item_number = self.extract_number(contents)
                    inventory_index_list.append(inventory_index)
                    item_number_list.append(item_number)
        else:
            # creat a item_number_list will all 1
            item_number_list = [1] * len(names)
            inventory_index_list = [str(i) for i in range(len(names))]

        toolbar_dict_list = []
        for i in range(len(inventory_index_list)):

            name = names[i]
            number = item_number_list[inventory_index_list.index(str(i))]
            position = i + 1

            toolbar_dict_list.append({
                "name": name,
                "number": number,
                "position": position
            })

        return toolbar_dict_list

    def extract_number(self, data):
        for item in data:
            if None in item:
                value = item[None]  # 'Number: 1'
                parts = value.split(': ')  # ['Number', '1']
                if len(parts) == 2:
                    try:
                        return int(parts[1])
                    except ValueError:
                        print("Cannot convert to integer.")
                        return None

    async def gather_toolbar_parallel(self, cur_inventories_shot_paths, gather_information_configurations):
        new_icon_template_list = []
        match_dict, selected_position = await self.template_matching_for_current_toolbar(
            cur_inventories_shot_paths, self.STARDEW_ORIGINAL_ICON_LIST, new_icon_template_list
        )
        toolbar_dict_list = await self.gather_toolbar_list(
            match_dict, get_number_flag=gather_information_configurations[constants.GET_ITEM_NUMBER]
        )
        return toolbar_dict_list,selected_position

    async def gather_llm_description(self, input):
        flag=True
        llm_description_gathered_information = None  # 修复Bug：初始化变量避免UnboundLocalError
        gather_information_configurations = input["gather_information_configurations"]
        if gather_information_configurations[constants.LLM_DESCRIPTION] is True:
            logger.write(f"Using llm description to gather information")
            try:
                # 🔥 CRITICAL FIX: 应用帧数截断到image_introduction（P2优化）
                memory = LocalMemory()
                target_frame_count = memory.working_area.get("target_frame_count")
                if isinstance(target_frame_count, int) and target_frame_count > 0 and "image_introduction" in input:
                    original_count = len(input["image_introduction"])
                    input["image_introduction"] = input["image_introduction"][:target_frame_count]
                    logger.write(f"✅ [FRAME TRUNCATION - LLM] Truncated {original_count} → {len(input['image_introduction'])} images")
                else:
                    logger.write(f"⚠️  [FRAME TRUNCATION - LLM] SKIPPED (target={target_frame_count})")
                
                # Call the LLM provider for gather information json
                message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=input)

                logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

                gather_information_success_flag = False
                gather_info_retry_count = 0
                gather_info_max_retries = 5
                while gather_information_success_flag is False and gather_info_retry_count < gather_info_max_retries:
                    try:
                        # 🔥 CRITICAL FIX: 使用async streaming避免阻塞并行执行
                        use_streaming = False
                        llm_description_max_tokens = None
                        if hasattr(self.llm_provider, 'create_completion_streaming'):
                            try:
                                import yaml
                                from cradle.utils.file_utils import assemble_project_path
                                config_path = assemble_project_path('./conf/enhanced_config.yaml')
                                if os.path.exists(config_path):
                                    with open(config_path, 'r', encoding='utf-8') as f:
                                        cfg = yaml.safe_load(f)
                                        use_streaming = cfg.get('performance', {}).get('streaming', {}).get('enabled', False)
                                        llm_description_max_tokens = cfg.get('performance', {}).get('streaming', {}).get('llm_description_max_tokens')
                            except Exception:
                                pass
                        
                        if use_streaming:
                            logger.write(f"[Streaming] ✅ Using streaming for LLM description")
                            response, info = await self.llm_provider.create_completion_streaming(
                                message_prompts,
                                early_termination=False,
                                max_tokens=llm_description_max_tokens
                            )
                        else:
                            response, info = await self.llm_provider.create_completion_async(message_prompts)
                        
                        logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')

                        # Convert the response to dict
                        processed_response = parse_semi_formatted_text(response)
                        gather_information_success_flag = True

                    except Exception as e:
                        gather_info_retry_count += 1
                        logger.error(f"Response of image description is not in the correct format: {e}, retry {gather_info_retry_count}/{gather_info_max_retries}")
                        if gather_info_retry_count >= gather_info_max_retries:
                            logger.error("Max retries reached for gather_llm_description")
                            processed_response = {}
                            break
                        gather_information_success_flag = False

                        # Wait 2 seconds for the next request and retry
                        await asyncio.sleep(2)

                llm_description_gathered_information = processed_response

            except Exception as e:
                logger.error(f"Error in gather image description information: {e}")
                flag = False
            return llm_description_gathered_information,flag
        else:
            return None,False


    async def execute_parallel(self, cur_inventories_shot_paths, gather_information_configurations,
                               input):
        # try:
            use_toolbar = bool(gather_information_configurations.get("use_toolbar", True))
            has_inventory_frames = bool(cur_inventories_shot_paths)

            if use_toolbar and has_inventory_frames:
                task_a = self.gather_toolbar_parallel(cur_inventories_shot_paths, gather_information_configurations)
                task_b = self.gather_llm_description(input)
                tool_bar_results, llm_results = await asyncio.gather(task_a, task_b)

                toolbar_dict_list, selected_position = tool_bar_results
                processed_response, flag = llm_results
                llm_description_gathered_information = processed_response
                return toolbar_dict_list, selected_position, llm_description_gathered_information, flag

            logger.write("[Performance] Toolbar branch skipped in execute_parallel")
            processed_response, flag = await self.gather_llm_description(input)
            return [], None, processed_response, flag

        # except asyncio.CancelledError:
        #     # Handle task cancellation here
        #     print("Tasks were cancelled")
        #     return None, None, None, False
        # except Exception as e:
        #     print(f"An error occurred: {e}")
        #     return None, None, None, False



class ActionPlanning():
    def __init__(self,
                 input_map: Optional[Dict] = None,
                 template: Optional[Dict] = None,
                 llm_provider: Any = None,
                 ):

        self.input_map = input_map
        self.template = template
        self.llm_provider = llm_provider
        # 可选：流式检测到动作时的回调（不停止生成）
        self.on_action_callback = None


    def _pre(self, *args, input: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        return input or {}

    async def _async_llm_call(self, message_prompts):
        """异步LLM调用（支持streaming）"""
        use_streaming = False
        early_action_buffer_chars = 256
        action_planning_max_tokens = None
        try:
            from cradle.config.enhanced_config import EnhancedConfig
            enhanced_cfg = EnhancedConfig()
            streaming_cfg = enhanced_cfg._raw_config.get('performance', {}).get('streaming', {})
            use_streaming = streaming_cfg.get('enabled', False)
            early_action_buffer_chars = streaming_cfg.get('early_action_buffer_chars', 256)
            action_planning_max_tokens = streaming_cfg.get('action_planning_max_tokens', None)
        except Exception as e:
            logger.debug(f"[Streaming] ActionPlanning config load failed: {e}")

        if use_streaming and hasattr(self.llm_provider, 'create_completion_streaming'):
            logger.write(f"[Streaming] ✅ Using streaming for action_planning")
            return await self.llm_provider.create_completion_streaming(
                message_prompts,
                early_termination=False,
                on_action=self.on_action_callback,
                early_action_buffer_chars=early_action_buffer_chars,
                max_tokens=action_planning_max_tokens,
            )

        if hasattr(self.llm_provider, 'create_completion_async'):
            return await self.llm_provider.create_completion_async(
                message_prompts,
                max_tokens=action_planning_max_tokens,
            )

        return self.llm_provider.create_completion(
            message_prompts,
            max_tokens=action_planning_max_tokens,
        )

    def __call__(self, *args, input: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """同步入口：创建新event loop运行async版本"""
        try:
            loop = asyncio.get_running_loop()
            # 已有running loop，在新线程中执行
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_async, input, *args, **kwargs)
                return future.result()
        except RuntimeError:
            # 没有running loop，直接用当前线程的持久loop
            return self._run_async(input, *args, **kwargs)

    def _run_async(self, input, *args, **kwargs):
        """在线程局部持久event loop中运行async代码"""
        loop = _get_thread_loop()
        return loop.run_until_complete(self._async_call(input, *args, **kwargs))

    async def _async_call(self, input: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Dict[str, Any]:
        """异步实现（主逻辑）"""

        input = self.input_map if input is None else input
        input = self._pre(input=input)

        flag = True
        processed_response = {}

        try:
            message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=input)

            logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

            # 🔥 使用async streaming
            response, info = await self._async_llm_call(message_prompts)

            logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')

            if response is None or len(response) == 0:
                logger.warn('No response in decision making call')
                logger.debug(input)

            # Convert the response to dict
            processed_response = parse_semi_formatted_text(response)

        except Exception as e:
            logger.error(f"Error in decision_making: {e}")
            logger.error_ex(e)
            flag = False

        data = dict(
            flag=flag,
            input=input,
            res_dict=processed_response,
        )

        data = self._post(data=data)
        return data


    def _post(self, *args, data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        return data or {}


class SuccessDetection():
    def __init__(self,
                 input_map: Optional[Dict] = None,
                 template: Optional[Dict] = None,
                 llm_provider: Any = None,
                 ):
        self.input_map = input_map
        self.template = template
        self.llm_provider = llm_provider


    def _pre(self, *args, input: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        return input or {}


    def __call__(self, *args, input: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:

        input = self.input_map if input is None else input
        input = self._pre(input=input)

        flag = True
        processed_response = {}

        try:

            # Call the LLM provider for success detection
            message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=input)

            logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

            response, info = self.llm_provider.create_completion(message_prompts)

            logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')

            # Convert the response to dict
            processed_response = parse_semi_formatted_text(response)

        except Exception as e:
            logger.error(f"Error in success_detection: {e}")
            flag = False

        data = dict(
            flag=flag,
            input=input,
            res_dict=processed_response,
        )

        data = self._post(data=data)
        return data


    def _post(self, *args, data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        return data or {}


class SelfReflection():

    def __init__(self,
                 input_map: Optional[Dict] = None,
                 template: Optional[Dict] = None,
                 llm_provider: Any = None,
                 ):
        self.input_map = input_map
        self.template = template
        self.llm_provider = llm_provider


    def _pre(self, *args, input: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        return input or {}


    def __call__(self, *args, input: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:

        input = self.input_map if input is None else input
        input = self._pre(input=input)

        flag = True
        processed_response = {}

        try:

            # Call the LLM provider for self reflection
            message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=input)

            logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

            response, info = self.llm_provider.create_completion(message_prompts)

            logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')

            # Convert the response to dict
            processed_response = parse_semi_formatted_text(response)

        except Exception as e:
            logger.error(f"Error in self reflection: {e}")
            flag = False

        data = dict(
            flag=flag,
            input=input,
            res_dict=processed_response,
        )

        data = self._post(data=data)
        return data


    def _post(self, *args, data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        return data or {}


class TaskInference():

    def __init__(self,
                 input_map: Optional[Dict] = None,
                 template: Optional[Dict] = None,
                 llm_provider: Any = None,
                 ):

        self.input_map = input_map
        self.template = template
        self.llm_provider = llm_provider


    def _pre(self, *args, input: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        return input or {}

    async def _async_llm_call(self, message_prompts):
        """异步LLM调用（支持streaming）"""
        use_streaming = False
        task_inference_max_tokens = None
        try:
            from cradle.config.enhanced_config import EnhancedConfig
            enhanced_cfg = EnhancedConfig()
            streaming_cfg = enhanced_cfg._raw_config.get('performance', {}).get('streaming', {})
            use_streaming = streaming_cfg.get('enabled', False)
            task_inference_max_tokens = streaming_cfg.get('task_inference_max_tokens', None)
        except Exception as e:
            logger.debug(f"[Streaming] TaskInference config load failed: {e}")

        if use_streaming and hasattr(self.llm_provider, 'create_completion_streaming'):
            logger.write(f"[Streaming] ✅ Using streaming for task_inference")
            return await self.llm_provider.create_completion_streaming(
                message_prompts,
                early_termination=False,
                max_tokens=task_inference_max_tokens,
            )

        if hasattr(self.llm_provider, 'create_completion_async'):
            return await self.llm_provider.create_completion_async(
                message_prompts,
                max_tokens=task_inference_max_tokens,
            )

        return self.llm_provider.create_completion(
            message_prompts,
            max_tokens=task_inference_max_tokens,
        )

    def __call__(self, *args, input: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """同步入口：创建新event loop运行async版本"""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop - use isolated event loop
            return self._run_async(input, *args, **kwargs)
        else:
            # Event loop already running - run in thread pool
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_async, input, *args, **kwargs)
                return future.result()
    
    def _run_async(self, input, *args, **kwargs):
        loop = _get_thread_loop()
        return loop.run_until_complete(self._async_call(input, *args, **kwargs))
    
    async def _async_call(self, input: Optional[Dict[str, Any]] = None, *args, **kwargs) -> Dict[str, Any]:
        """异步实现（主逻辑）"""

        input = self.input_map if input is None else input
        input = self._pre(input=input)

        flag = True
        processed_response = {}

        try:
            message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=input)
            logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

            # 使用async streaming
            response, info = await self._async_llm_call(message_prompts)

            logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')
            processed_response = parse_semi_formatted_text(response)

        except Exception as e:
            logger.error(f"Error in information_summary: {e}")
            flag = False

        data = dict(
            flag=flag,
            input=input,
            res_dict=processed_response,
            # res_json=res_json,
        )

        data = self._post(data=data)
        return data


    def _post(self, *args, data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        return data or {}

class StardewPlanner(BasePlanner):

    def __init__(self,
                 llm_provider: Any = None,
                 planner_params: Optional[Dict] = None,
                 use_task_inference: bool = False,
                 use_self_reflection: bool = False,
                 gather_information_max_steps: int = 1,  # 5,
                 icon_replacer: Any = None,
                 object_detector: Any = None,
                 frame_extractor: Any = None,
                 ):
        """
        inputs: input key-value pairs
        templates: template for composing the prompt
        """

        super().__init__()

        self.llm_provider = llm_provider

        self.use_task_inference = use_task_inference
        self.use_self_reflection = use_self_reflection
        self.gather_information_max_steps = gather_information_max_steps

        self.icon_replacer = icon_replacer
        self.object_detector = object_detector
        self.frame_extractor = frame_extractor
        self.set_internal_params(planner_params=planner_params,
                                 use_task_inference=use_task_inference)


    # Allow re-configuring planner
    def set_internal_params(self,
                            planner_params: Optional[Dict] = None,
                            use_task_inference: bool = False):

        if planner_params is None:
            raise ValueError("planner_params cannot be None")

        self.planner_params = planner_params
        if not check_planner_params(self.planner_params):
            raise ValueError(f"Error in planner_params: {self.planner_params}")

        self.inputs = self._init_inputs()
        self.templates = self._init_templates()

        self.information_gathering_ = InformationGathering(input_map=self.inputs["information_gathering"],
                                                     template=self.templates["information_gathering"],
                                                     text_input_map=self.inputs["information_text_gathering"],
                                                     get_text_template=self.templates["information_text_gathering"],
                                                     toolbar_input_map=self.inputs["information_toolbar_gathering"],
                                                     get_toolbar_template=self.templates["information_toolbar_gathering"],
                                                     frame_extractor=self.frame_extractor,
                                                     icon_replacer=self.icon_replacer,
                                                     object_detector=self.object_detector,
                                                     llm_provider=self.llm_provider)

        self.action_planning_ = ActionPlanning(input_map=self.inputs["action_planning"],
                                               template=self.templates["action_planning"],
                                               llm_provider=self.llm_provider)

        self.success_detection_ = SuccessDetection(input_map=self.inputs["success_detection"],
                                                   template=self.templates["success_detection"],
                                                   llm_provider=self.llm_provider)

        if self.use_self_reflection:
            self.self_reflection_ = SelfReflection(input_map=self.inputs["self_reflection"],
                                                   template=self.templates["self_reflection"],
                                                   llm_provider=self.llm_provider)
        else:
            self.self_reflection_ = None

        if self.use_task_inference:
            self.task_inference_ = TaskInference(input_map=self.inputs["task_inference"],
                                                           template=self.templates["task_inference"],
                                                           llm_provider=self.llm_provider)
        else:
            self.task_inference_ = None


    def _init_inputs(self):

        input_examples = dict()
        prompt_paths = self.planner_params["prompt_paths"]
        input_example_paths = prompt_paths["inputs"]

        for key, value in input_example_paths.items():
            path = assemble_project_path(value)
            if path.endswith(PROMPT_EXT):
                input_examples[key] = read_resource_file(path)
            else:
                input_examples[key] = load_json(path)

        return input_examples


    def _init_templates(self):

        templates = dict()
        prompt_paths = self.planner_params["prompt_paths"]
        template_paths = prompt_paths["templates"]

        for key, value in template_paths.items():
            path = assemble_project_path(value)
            if path.endswith(PROMPT_EXT):
                templates[key] = read_resource_file(path)
            else:
                templates[key] = load_json(path)

        return templates


    def information_gathering(self, *args, input: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:

        if input is None:
            input = self.inputs["gather_information"]

        assert input is not None

        for i in range(self.gather_information_max_steps):
            data = self.information_gathering_(input=input, class_=None)

            success = data["success"]

            if success:
                break

        return data


    def action_planning(self, *args, input: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:

        if input is None:
            input = self.inputs["action_planning"]

        data = self.action_planning_(input=input)

        return data


    def success_detection(self, *args, input: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:

        if input is None:
            input = self.inputs["success_detection"]

        data = self.success_detection_(input=input)

        return data


    def self_reflection(self, *args, input: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:

        if input is None:
            input = self.inputs["self_reflection"]

        if self.self_reflection_ is None:
            return {
                "flag": False,
                "input": input,
                "res_dict": {},
            }

        data = self.self_reflection_(input=input)

        return data


    def task_inference(self, *args, input: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:

        if input is None:
            input = self.inputs["task_inference"]

        if self.task_inference_ is None:
            return {
                "flag": False,
                "input": input,
                "res_dict": {},
            }

        data = self.task_inference_(input=input)

        return data