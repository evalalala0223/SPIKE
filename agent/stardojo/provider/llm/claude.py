from __future__ import annotations

import os
import base64
import httpx
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)
import json
import re
import io
import asyncio

import backoff
import tiktoken
import numpy as np
import cv2
from PIL import Image
import anthropic
from anthropic import RateLimitError, APIError, APITimeoutError

from stardojo import constants
from stardojo.provider.base import LLMProvider, EmbeddingProvider
from stardojo.config import Config
from stardojo.log import Logger
from stardojo.utils.json_utils import load_json
from stardojo.utils.encoding_utils import encode_data_to_base64_path
from stardojo.utils.file_utils import assemble_project_path
try:
    from cradle.utils.llm_call_budget import increment_llm_call_counter
except ModuleNotFoundError:
    import sys as _sys
    _agent_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    _agent_root = os.path.normpath(_agent_root)
    if _agent_root not in _sys.path:
        _sys.path.insert(0, _agent_root)
    from cradle.utils.llm_call_budget import increment_llm_call_counter

config = Config()
logger = Logger()

MAX_TOKENS = {
    "claude-3-opus-20240229": 2 * 10 ** 6,
    "claude-3-sonnet-20240229": 2 * 10 ** 6,
    "claude-3-haiku-20240307": 2 * 10 ** 6,
}

PROVIDER_SETTING_KEY_VAR = "key_var"
PROVIDER_SETTING_COMP_MODEL = "comp_model"


class ClaudeProvider(LLMProvider):
    """A class that wraps a given model"""

    client: anthropic.Anthropic = None
    llm_model: str = ""
    embedding_model: str = ""

    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    chunk_size: int = 1000
    embedding_ctx_length: int = 2 * 10 ** 6
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    tiktoken_model_name: Optional[str] = None

    """Whether to skip empty strings when embedding or raise an error."""
    skip_empty: bool = False


    def __init__(self) -> None:
        """Initialize a class instance

        Args:
            cfg: Config object

        Returns:
            None
        """
        self.retries = 5


    def init_provider(self, provider_cfg ) -> None:
        self.provider_cfg = self._parse_config(provider_cfg)


    def _parse_config(self, provider_cfg) -> dict:
        """Parse the config object"""

        conf_dict = dict()

        if isinstance(provider_cfg, dict):
            conf_dict = provider_cfg
        else:
            path = assemble_project_path(provider_cfg)
            conf_dict = load_json(path)

        key_var_name = conf_dict[PROVIDER_SETTING_KEY_VAR]

        key = os.getenv(key_var_name)
        self.client = anthropic.Anthropic(api_key=key)

        self.llm_model = conf_dict[PROVIDER_SETTING_COMP_MODEL]

        return conf_dict


    def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = config.temperature,
        seed: int = config.seed,
        max_tokens: int = config.max_tokens,
    ) -> Tuple[str, Dict[str, int]]:
        """Create a chat completion using the OpenAI API

        Supports both GPT-4 and GPT-4V).

        Example Usage:
        image_path = "path_to_your_image.jpg"
        base64_image = encode_image(image_path)
        response, info = self.create_completion(
            model="gpt-4-vision-preview",
            messages=[
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": "What’s in this image?"
                  },
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                  }
                ]
              }
            ],
        )
        """

        if model is None:
            model = self.llm_model

        if config.debug_mode:
            logger.debug(f"Creating chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}")
        else:
            logger.write(f"Requesting {model} completion...")

        @backoff.on_exception(
            backoff.constant,
            (
                APIError,
                RateLimitError,
                APITimeoutError),
            max_tries=self.retries,
            interval=10,
        )

        def _generate_response_with_retry(
            messages: List[Dict[str, str]],
            model: str,
            temperature: float,
            seed: int = None,
            max_tokens: int = 512,
        ) -> Tuple[str, Dict[str, int]]:

            system_content = ""
            messages_for_request = []
            for message in messages:
                if message.get("role") == "system":
                    content = message.get("content", [])
                    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
                        system_content = content[0].get("text", "") or ""
                else:
                    messages_for_request.append(message)

            logger.write(f"Requesting completion..., System content: {system_content}")

            """Send a request to the Claude API."""
            increment_llm_call_counter("big_brain:claude")
            response = self.client.messages.create(
                model=model,
                messages=messages_for_request,
                system=system_content,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            if response is None:
                logger.error("Failed to get a response from Claude. Try again.")
                logger.double_check()

            message = response.content[0].text

            info = {
                "input_tokens" : response.usage.input_tokens,
                "output_tokens" : response.usage.output_tokens,
            }

            logger.write(f'Response received from {model}.')

            return message, info

        return _generate_response_with_retry(
            messages,
            model,
            temperature,
            seed,
            max_tokens,
        )


    async def create_completion_async(
            self,
            messages: List[Dict[str, str]],
            model: str | None = None,
            temperature: float = config.temperature,
            seed: int = config.seed,
            max_tokens: int = config.max_tokens,
    ) -> Tuple[str, Dict[str, int]]:

        if model is None:
            model = self.llm_model

        if config.debug_mode:
            logger.debug(
                f"Creating chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}")
        else:
            logger.write(f"Requesting {model} completion...")

        @backoff.on_exception(
            backoff.constant,
            (
                    APIError,
                    RateLimitError,
                    APITimeoutError),
            max_tries=self.retries,
            interval=10,
        )

        async def _generate_response_with_retry_async(
                messages: List[Dict[str, str]],
                model: str,
                temperature: float,
                seed: int = None,
                max_tokens: int = 512,
        ) -> Tuple[str, Dict[str, int]]:

            system_content = ""
            messages_for_request = []
            for message in messages:
                if message.get("role") == "system":
                    content = message.get("content", [])
                    if isinstance(content, list) and len(content) > 0 and isinstance(content[0], dict):
                        system_content = content[0].get("text", "") or ""
                else:
                    messages_for_request.append(message)

            """Send a request to the Claude API."""
            increment_llm_call_counter("big_brain:claude_async")
            response = await asyncio.to_thread(
                self.client.messages.create,
                model=model,
                messages=messages_for_request,
                system=system_content,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            if response is None:
                logger.error("Failed to get a response from Claude. Try again.")
                logger.double_check()

            message = response.content[0].text

            info = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

            logger.write(f'Response received from {model}.')

            return message, info

        return await _generate_response_with_retry_async(
            messages,
            model,
            temperature,
            seed,
            max_tokens,
        )


    def num_tokens_from_messages(self, messages, model):
        """Not implemented for Claude models."""
        raise NotImplementedError(
            f"num_tokens_from_messages() is not implemented for Claude model {model}."
        )


    def assemble_prompt_tripartite(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:

        """
        A tripartite prompt is a message with the following structure:
        <system message>

        <user message part 1 before image introduction>
        <image introduction>
        <user message part 2 after image introduction>
        """
        pattern = re.compile(r"(.+?)(?=\n\n|$)", re.DOTALL)

        paragraphs = re.findall(pattern, template_str)

        filtered_paragraphs = [p for p in paragraphs if p.strip() != '']

        system_content = filtered_paragraphs[0]  # the system content defaults to the first paragraph of the template
        system_message = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": f"{system_content}"
                }
            ]
        }

        # segmenting "paragraphs"
        image_introduction_paragraph_index = None
        image_introduction_paragraph = None
        for i, paragraph in enumerate(filtered_paragraphs):
            if constants.IMAGES_INPUT_TAG in paragraph:
                image_introduction_paragraph_index = i
                image_introduction_paragraph = paragraph
                break

        user_messages_part1_paragraphs = filtered_paragraphs[1:image_introduction_paragraph_index]
        user_messages_part2_paragraphs = filtered_paragraphs[image_introduction_paragraph_index + 1:]
        
        combined_user_message = {
            "role": "user",
            "content": [
            ]
        }

        # assemble user messages part 1
        user_messages_part1_contents = []
        for paragraph in user_messages_part1_paragraphs:
            search_placeholder_pattern = re.compile(r"<\$[^\$]+\$>")

            placeholder = re.search(search_placeholder_pattern, paragraph)
            if not placeholder:
                user_messages_part1_contents.append(paragraph)
            else:
                placeholder = placeholder.group()
                placeholder_name = placeholder.replace("<$", "").replace("$>", "")

                paragraph_input = params.get(placeholder_name, None)
                if paragraph_input is None or paragraph_input == "" or paragraph_input == []:
                    continue
                else:
                    if isinstance(paragraph_input, str):
                        paragraph_content = paragraph.replace(placeholder, paragraph_input)
                        user_messages_part1_contents.append(paragraph_content)
                    elif isinstance(paragraph_input, list) or isinstance(paragraph_input, dict):
                        paragraph_content = paragraph.replace(placeholder, json.dumps(paragraph_input))
                        user_messages_part1_contents.append(paragraph_content)
                    else:
                        raise ValueError(f"Unexpected input type: {type(paragraph_input)}")

        user_messages_part1_content = "\n\n".join(user_messages_part1_contents)
        # user_messages_part1 = {
        #     "role": "user",
        #     "content": [
        #         {
        #             "type": "text",
        #             "text": f"{user_messages_part1_content}"
        #         }
        #     ]
        # }
        combined_user_message["content"].append({
            "type": "text",
            "text": f"{user_messages_part1_content}"
        })

        # assemble image introduction messages
        image_introduction_messages = []
        paragraph_input = params.get(constants.IMAGES_INPUT_TAG_NAME, None)

        if paragraph_input is None or paragraph_input == "" or paragraph_input == []:
            image_introduction_messages = []
        else:
            # paragraph_content_pre = image_introduction_paragraph.replace(constants.IMAGES_INPUT_TAG, "")
            # message = {
            #     "role": "user",
            #     "content": [
            #         {
            #             "type": "text",
            #             "text": f"{paragraph_content_pre}"
            #         }
            #     ]
            # }

            # image_introduction_messages.append(message)

            # path = params["image_path"] if "image_path" in params else ""
            # if path is not None and path != "":
            #     media_type = f"image/jpeg"
            #     with open(path, 'rb') as file:
            #         binary_content = file.read()
            #         base64_encoded = base64.standard_b64encode(binary_content)
            #         base64_string = base64_encoded.decode('utf-8')
            #     encoded_images = [base64_string]

            #     for encoded_image in encoded_images:
            #         msg_content = {
            #             "type": "image",
            #             "source":
            #                 {
            #                     "type": "base64",
            #                     "media_type": media_type,
            #                     "data": f"{encoded_image}",
            #                 }
            #         }

            #         message["content"].append(msg_content)
            # paragraph_content_pre = image_introduction_paragraph.replace(constants.IMAGES_INPUT_TAG, "")
            paths = params["image_paths"] if "image_paths" in params else []
            encoded_images = []
            for i, path in enumerate(paths):
                if path is not None and path != "":
                    result = encode_data_to_base64_path(path)
                    if isinstance(result, list):
                        if len(result) > 0:
                            encoded_images.append(result[0])
                    elif result:
                        encoded_images.append(result)
 
            for i, encoded_image in enumerate(reversed(encoded_images)):
                msg_text = "This is a screenshot of the current step of the game." if i == 0 else f"This is the game screenshot from {i} steps ago"
                combined_user_message["content"].append({
                    "type": "text",
                    "text": f"{msg_text}"
                })
                msg_content = {
                    "type": "image",
                    "source":
                        {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": f"{encoded_image}"
                        }
                }
                combined_user_message["content"].append(msg_content)

        # assemble user messages part 2
        user_messages_part2_contents = []
        for paragraph in user_messages_part2_paragraphs:
            search_placeholder_pattern = re.compile(r"<\$[^\$]+\$>")

            placeholder = re.search(search_placeholder_pattern, paragraph)
            if not placeholder:
                user_messages_part2_contents.append(paragraph)
            else:
                placeholder = placeholder.group()
                placeholder_name = placeholder.replace("<$", "").replace("$>", "")

                paragraph_input = params.get(placeholder_name, None)
                if paragraph_input is None or paragraph_input == "" or paragraph_input == []:
                    continue
                else:
                    if isinstance(paragraph_input, str):
                        paragraph_content = paragraph.replace(placeholder, paragraph_input)
                        user_messages_part2_contents.append(paragraph_content)
                    elif isinstance(paragraph_input, list):
                        paragraph_content = paragraph.replace(placeholder, json.dumps(paragraph_input))
                        user_messages_part2_contents.append(paragraph_content)
                    else:
                        raise ValueError(f"Unexpected input type: {type(paragraph_input)}")

        user_messages_part2_content = "\n\n".join(user_messages_part2_contents)
        user_messages_part2 = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{user_messages_part2_content}"
                }
            ]
        }
        combined_user_message["content"].append({
            "type": "text",
            "text": f"{user_messages_part2_content}"
        })

        # messages = [system_message] + [user_messages_part1] + image_introduction_messages + [user_messages_part2]

        # # merge messages with the same role
        # messages = self._merge_messages(messages)

        # return messages
        return [system_message] + [combined_user_message]


    def _merge_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Merge messages with the same role into a single message"""
        merged_messages = []
        for message in messages:
            if not merged_messages:
                merged_messages.append(message)
            elif message["role"] == merged_messages[-1]["role"]:
                merged_messages[-1]["content"] += message["content"]
            else:
                merged_messages.append(message)

        for index, message in enumerate(merged_messages):

            """Merge contents"""
            contents = message["content"]
            merged_contents = []

            for content in contents:
                if content["type"] == "text" and not content["text"].strip():
                    continue
                if not merged_contents:
                    merged_contents.append(content)
                if content["type"] == "text" and merged_contents[-1]["type"] == "text":
                    merged_contents[-1]["text"] += content["text"]
                else:
                    merged_contents.append(content)

            merged_messages[index]["content"] = merged_contents

        return merged_messages


    def assemble_prompt_paragraph(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError("This method is not implemented yet.")


    def assemble_prompt(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if config.DEFAULT_MESSAGE_CONSTRUCTION_MODE == constants.MESSAGE_CONSTRUCTION_MODE_TRIPART:
            return self.assemble_prompt_tripartite(template_str=template_str, params=params)
        elif config.DEFAULT_MESSAGE_CONSTRUCTION_MODE == constants.MESSAGE_CONSTRUCTION_MODE_PARAGRAPH:
            return self.assemble_prompt_paragraph(template_str=template_str, params=params)
