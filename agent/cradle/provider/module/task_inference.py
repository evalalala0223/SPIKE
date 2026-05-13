from typing import Any
import json
from copy import deepcopy

from cradle.provider import BaseModuleProvider, BaseProvider
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.utils.json_utils import parse_semi_formatted_text

config = Config()
logger = Logger()


class TaskInferenceProvider(BaseModuleProvider):

    def __init__(self,
                 *args,
                 template_path: str,
                 llm_provider: Any = None,
                 gm: Any = None,
                 **kwargs):

        super(TaskInferenceProvider, self).__init__(template_path = template_path, **kwargs)

        self.template_path = template_path
        self.llm_provider = llm_provider
        self.gm = gm
        self.memory = LocalMemory(memory_path=config.work_dir, max_recent_steps=config.max_recent_steps)


    @BaseModuleProvider.debug
    @BaseModuleProvider.error
    @BaseModuleProvider.write
    def __call__(self,
                 *args,
                 use_screenshot_augmented = False,
                 used_video = False,
                 **kwargs):

        params = deepcopy(self.memory.working_area)

        self._check_input_keys(params)

        message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=params)
        logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

        response = {}
        try:
            # Phase 2.1: 集成流式输出（如果Provider支持）
            use_streaming = False
            if hasattr(self.llm_provider, 'create_completion_streaming'):
                try:
                    import yaml
                    import os
                    from cradle.utils.file_utils import assemble_project_path
                    config_path = assemble_project_path('./conf/enhanced_config.yaml')
                    if os.path.exists(config_path):
                        with open(config_path, 'r', encoding='utf-8') as f:
                            cfg = yaml.safe_load(f)
                            use_streaming = cfg.get('performance', {}).get('streaming', {}).get('enabled', False)
                except Exception as e:
                    logger.debug(f"[Streaming] Config check failed: {e}")

            if use_streaming:
                logger.debug(f"[Streaming] ENABLED for TaskInference")
                try:
                    from cradle.provider.output_schemas import TaskInferenceSchema
                    structured_output = self.llm_provider.create_completion_streaming(
                        message_prompts,
                        output_schema=TaskInferenceSchema
                    )
                    response = structured_output.model_dump()
                    info = {"input_tokens": 0, "output_tokens": 0}
                    logger.debug(f"[Streaming] Streaming completed successfully")
                    logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')
                except Exception as e:
                    logger.warn(f"[Streaming] Failed: {e}, falling back to standard method")
                    response, info = self.llm_provider.create_completion(message_prompts)
                    logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')
                    response = parse_semi_formatted_text(response)
            else:
                response, info = self.llm_provider.create_completion(message_prompts)
                logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')
                response = parse_semi_formatted_text(response)

        except Exception as e:
            logger.error(f"Task inference LLM call failed: {e}")

        self._check_output_keys(response)

        del params

        return response


class RDR2TaskInferenceProvider(BaseProvider):

    def __init__(self,
                 *args,
                 planner,
                 gm,
                 **kwargs):

        super(RDR2TaskInferenceProvider, self).__init__()

        self.planner = planner
        self.gm = gm
        self.memory = LocalMemory(memory_path=config.work_dir, max_recent_steps=config.max_recent_steps)


    def __call__(self, *args, **kwargs):

        params = deepcopy(self.memory.working_area)

        data = self.planner.task_inference(input=params)

        response = data['res_dict']

        del params

        return response


class StardewTaskInferenceProvider(BaseProvider):

    def __init__(self,
                 *args,
                 planner,
                 gm,
                 **kwargs):

        super(StardewTaskInferenceProvider, self).__init__()

        self.planner = planner
        self.gm = gm
        self.memory = LocalMemory(memory_path=config.work_dir, max_recent_steps=config.max_recent_steps)


    def __call__(self, *args, **kwargs):

        params = deepcopy(self.memory.working_area)

        data = self.planner.task_inference(input=params)

        response = data['res_dict']

        del params

        return response
