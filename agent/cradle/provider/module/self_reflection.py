from typing import List, Dict, Any
import json
import os
from copy import deepcopy
import yaml

from cradle.provider import BaseModuleProvider, BaseProvider
from cradle.utils.json_utils import parse_semi_formatted_text
from cradle.log import Logger
from cradle.config import Config
from cradle.memory import LocalMemory
from cradle.utils.file_utils import assemble_project_path

config = Config()
logger = Logger()


class SelfReflectionProvider(BaseModuleProvider):

    def __init__(self,
                 *args,
                 template_path: str,
                 llm_provider: Any = None,
                 gm: Any = None,
                 **kwargs):

        super(SelfReflectionProvider, self).__init__(template_path = template_path, **kwargs)

        self.template_path = template_path
        self.llm_provider = llm_provider
        self.gm = gm
        self.memory = LocalMemory()


    @BaseModuleProvider.debug
    @BaseModuleProvider.error
    @BaseModuleProvider.write
    def __call__(self,
                 *args,
                 **kwargs):

        params = deepcopy(self.memory.working_area)

        self._check_input_keys(params)

        message_prompts = self.llm_provider.assemble_prompt(template_str=self.template, params=params)
        logger.debug(f'{logger.UPSTREAM_MASK}{json.dumps(message_prompts, ensure_ascii=False)}\n')

        response = {}
        try:
            # Phase 2.1: SelfReflection-specific config (streaming/timeout/max_tokens)
            use_streaming = False
            timeout_seconds = 45
            max_tokens = 512
            if hasattr(self.llm_provider, 'create_completion_streaming'):
                try:
                    config_path = assemble_project_path('./conf/enhanced_config.yaml')
                    if os.path.exists(config_path):
                        with open(config_path, 'r', encoding='utf-8') as f:
                            cfg = yaml.safe_load(f)
                            stream_cfg = cfg.get('performance', {}).get('streaming', {})
                            use_streaming = bool(stream_cfg.get('self_reflection_use_streaming', False))
                            timeout_seconds = int(stream_cfg.get('self_reflection_timeout_seconds', 45))
                            max_tokens = int(stream_cfg.get('self_reflection_max_tokens', 512))
                except Exception as e:
                    logger.debug(f"[Streaming] Config check failed: {e}")

            if use_streaming:
                logger.debug(
                    f"[Streaming] ENABLED for SelfReflection "
                    f"(timeout={timeout_seconds}s, max_tokens={max_tokens})"
                )
                try:
                    from cradle.provider.output_schemas import SelfReflectionSchema
                    structured_output = self.llm_provider.create_completion_streaming(
                        message_prompts,
                        output_schema=SelfReflectionSchema,
                        max_tokens=max_tokens,
                        timeout_seconds=timeout_seconds,
                    )
                    response = structured_output.model_dump()
                    info = {"input_tokens": 0, "output_tokens": 0}
                    logger.debug(f"[Streaming] Streaming completed successfully")
                    logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')
                except Exception as e:
                    logger.warn(f"[Streaming] Failed: {e}, falling back to standard method")
                    response, info = self.llm_provider.create_completion(
                        message_prompts,
                        max_tokens=max_tokens,
                        timeout_seconds=timeout_seconds,
                    )
                    logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')
                    response = parse_semi_formatted_text(response)
            else:
                logger.debug(
                    f"[SelfReflection] Standard completion "
                    f"(timeout={timeout_seconds}s, max_tokens={max_tokens})"
                )
                response, info = self.llm_provider.create_completion(
                    message_prompts,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                )
                logger.debug(f'{logger.DOWNSTREAM_MASK}{response}\n')
                response = parse_semi_formatted_text(response)

        except Exception as e:
            logger.error(f"Self reflection LLM call failed: {e}")

        self._check_output_keys(response)

        del params

        return response


class RDR2SelfReflectionProvider(BaseProvider):
    def __init__(self,
                 *args,
                 planner,
                 gm,
                 **kwargs):
        super(RDR2SelfReflectionProvider, self).__init__()
        self.planner = planner
        self.gm = gm
        self.memory = LocalMemory()


    def __call__(self, *args, **kwargs):

        params = deepcopy(self.memory.working_area)

        data = self.planner.self_reflection(input=params)

        response = data['res_dict']

        del params

        return response


class StardewSelfReflectionProvider(BaseProvider):

    def __init__(self,
                 *args,
                 planner,
                 gm,
                 **kwargs):

        super(StardewSelfReflectionProvider, self).__init__()

        self.planner = planner
        self.gm = gm
        self.memory = LocalMemory()


    def __call__(self, *args, **kwargs):

        params = deepcopy(self.memory.working_area)

        data = self.planner.self_reflection(input=params)

        response = data['res_dict']

        del params

        return response
