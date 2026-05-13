import json
from typing import Any, Dict, List
from copy import deepcopy

from cradle.provider import BaseModuleProvider, BaseProvider
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.provider import VideoRecordProvider
from cradle import constants
from cradle.utils.json_utils import parse_semi_formatted_text
from cradle.utils.image_utils import segment_toolbar, segment_new_icon, segement_inventory

config = Config()
logger = Logger()
memory = LocalMemory()


class InformationGatheringProvider(BaseModuleProvider):

    def __init__(self,
                 *args,
                 template_path: str,
                 llm_provider: Any = None,
                 gm: Any = None,
                 **kwargs):

        super(InformationGatheringProvider, self).__init__(template_path = template_path, **kwargs)

        self.template_path = template_path
        self.llm_provider = llm_provider
        self.gm = gm


    @BaseModuleProvider.debug
    @BaseModuleProvider.error
    @BaseModuleProvider.write
    def __call__(self,
                 *args,
                 **kwargs):

        params = deepcopy(memory.working_area)

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
                logger.debug(f"[Streaming] ENABLED for InformationGathering")
                try:
                    from cradle.provider.output_schemas import InfoGatheringSchema
                    structured_output = self.llm_provider.create_completion_streaming(
                        message_prompts,
                        output_schema=InfoGatheringSchema
                    )
                    response = structured_output.model_dump()
                    info = {"input_tokens": 0, "output_tokens": 0}
                    logger.debug(f"[Streaming] Streaming completed successfully")
                    logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')
                except Exception as e:
                    logger.warn(f"[Streaming] Failed: {e}, falling back to standard method")
                    response, info = self.llm_provider.create_completion(message_prompts)
                    logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')
                    # Convert the response to dict
                    response = parse_semi_formatted_text(response)
            else:
                response, info = self.llm_provider.create_completion(message_prompts)
                logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')
                # Convert the response to dict
                response = parse_semi_formatted_text(response)

        except Exception as e:
            logger.error(f"Information gathering LLM call failed: {e}")

        self._check_output_keys(response)

        del params

        return response

class RDR2InformationGatheringProvider(BaseProvider):
    def __init__(self,
                 *args,
                 planner = None,
                 gm = None,
                 **kwargs):
        super(RDR2InformationGatheringProvider, self).__init__()
        self.planner = planner
        self.gm = gm

    def __call__(self, *args, **kwargs):

        params = deepcopy(memory.working_area)

        data = self.planner.information_gathering(input=params)

        response = data['res_dict']

        del params

        return response

class StardewInformationGatheringProvider(BaseProvider):
    def __init__(self,
                 *args,
                 planner = None,
                 gm = None,
                 **kwargs):
        super(StardewInformationGatheringProvider, self).__init__()
        self.planner = planner
        self.gm = gm

    def __call__(self, *args, **kwargs):

        params = deepcopy(memory.working_area)

        data = self.planner.information_gathering(input=params)

        response = data['res_dict']

        del params

        return response