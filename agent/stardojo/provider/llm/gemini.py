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
from google import genai
from google.genai.errors import APIError, ClientError, ServerError, UnknownFunctionCallArgumentError, UnsupportedFunctionError, FunctionInvocationError
from google.genai import types

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
    "gemini-2.0-flash": 8192
}

PROVIDER_SETTING_KEY_VAR = "key_var"
PROVIDER_SETTING_COMP_MODEL = "comp_model"


def _legacy_messages_to_genai(messages: List[Dict[str, Any]]) -> Tuple[Optional[str], List[types.Content]]:
    """Convert legacy dict-style messages to google-genai>=1.x Content/Part objects.

    Legacy shape (produced by assemble_prompt_tripartite):
        [
          {"role": "system", "parts": [{"text": "..."}]},
          {"role": "user",   "parts": [
              {"text": "..."},
              {"inline_data": {"mime_type": "image/jpeg", "data": "<base64>"}},
              ...
          ]},
          ...
        ]

    Returns:
        (system_instruction, contents)
        system_instruction: extracted "system" text or None
        contents: list[types.Content] suitable for client.models.generate_content(contents=...)
    """
    system_instruction: Optional[str] = None
    contents: List[types.Content] = []

    for message in messages or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "user").lower()
        raw_parts = message.get("parts") or []
        if not isinstance(raw_parts, list):
            raw_parts = [raw_parts]

        if role == "system":
            chunks: List[str] = []
            for p in raw_parts:
                if isinstance(p, dict):
                    text = p.get("text")
                    if isinstance(text, str) and text:
                        chunks.append(text)
                elif isinstance(p, str) and p:
                    chunks.append(p)
            if chunks:
                merged = "\n\n".join(chunks)
                system_instruction = (
                    merged if system_instruction is None else f"{system_instruction}\n\n{merged}"
                )
            continue

        # Gemini accepts roles "user" and "model" only.
        gen_role = "model" if role in ("model", "assistant") else "user"

        new_parts: List[types.Part] = []
        for p in raw_parts:
            if isinstance(p, types.Part):
                new_parts.append(p)
                continue
            if isinstance(p, str):
                if p:
                    new_parts.append(types.Part.from_text(text=p))
                continue
            if not isinstance(p, dict):
                continue

            if "text" in p and p["text"] is not None:
                text_val = str(p["text"])
                if text_val:
                    new_parts.append(types.Part.from_text(text=text_val))
                continue

            inline = p.get("inline_data") or p.get("inlineData")
            if isinstance(inline, dict):
                mime = str(inline.get("mime_type") or inline.get("mimeType") or "image/jpeg")
                data = inline.get("data")
                if isinstance(data, str):
                    # Strip data URI prefix if present, e.g. "data:image/jpeg;base64,/9j/..."
                    payload = data
                    if payload.startswith("data:"):
                        comma = payload.find(",")
                        if comma >= 0:
                            header = payload[5:comma]  # "image/jpeg;base64"
                            if ";" in header:
                                hdr_mime = header.split(";", 1)[0].strip()
                                if hdr_mime:
                                    mime = hdr_mime
                            payload = payload[comma + 1:]
                    try:
                        raw_bytes = base64.b64decode(payload, validate=False)
                    except Exception:
                        raw_bytes = payload.encode("utf-8", errors="ignore")
                elif isinstance(data, (bytes, bytearray)):
                    raw_bytes = bytes(data)
                else:
                    continue
                if not raw_bytes:
                    continue
                new_parts.append(types.Part.from_bytes(data=raw_bytes, mime_type=mime))
                continue

            file_data = p.get("file_data") or p.get("fileData")
            if isinstance(file_data, dict):
                file_uri = file_data.get("file_uri") or file_data.get("fileUri")
                mime = str(file_data.get("mime_type") or file_data.get("mimeType") or "")
                if file_uri:
                    try:
                        new_parts.append(types.Part.from_uri(file_uri=file_uri, mime_type=mime or None))
                    except Exception:
                        pass

        if new_parts:
            contents.append(types.Content(role=gen_role, parts=new_parts))

    return system_instruction, contents


def _build_generate_config(
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    seed: Optional[int],
    system_instruction: Optional[str],
) -> "types.GenerateContentConfig":
    """Build a GenerateContentConfig that works for both classic and "thinking" Gemini models.

    For Gemini "thinking" models (e.g. 3.x pro / flash-thinking), part of the
    `max_output_tokens` budget is consumed by hidden thoughts. If the budget is
    too tight, the response can finish with `MAX_TOKENS` and `parts=None` (no
    visible answer emitted). We adapt by:

      * For 3.x models that *require* thinking (e.g. gemini-3.1-pro-preview):
        do NOT pass thinking_config (passing budget=0 yields HTTP 400). Instead
        ensure `max_output_tokens` is large enough to leave room for both
        thoughts and the visible answer (configurable via env vars below).
      * For thinking-capable but optional models: try to disable thinking with
        `thinking_budget=0` to maximize tokens spent on the answer.

    Env overrides:
      GEMINI_THINKING_BUDGET   int  thinking budget for optional-thinking models (default 0)
      GEMINI_MIN_MAX_TOKENS    int  floor for max_output_tokens on thinking models (default 4096)
    """
    model_lc = (model or "").lower()
    is_thinking_3x = (
        "gemini-3" in model_lc or "3.1" in model_lc or "3.5" in model_lc
    )
    is_optional_thinking = ("thinking" in model_lc) and not is_thinking_3x

    # All modern Gemini chat models benefit from a generous output budget when
    # the agent prompt asks for reasoning + a code block. The default upstream
    # value (1024) often truncates the answer mid-block, leaving the parser
    # with no actions to execute. Apply a floor that's safe for Stardew prompts.
    effective_max_tokens = int(max_tokens or 0)
    try:
        floor = int(os.getenv("GEMINI_MIN_MAX_TOKENS", "0"))
    except ValueError:
        floor = 0
    if floor <= 0:
        # Auto-pick a floor based on whether the model needs to spend tokens on thinking.
        floor = 8192 if is_thinking_3x else 4096
    if effective_max_tokens < floor:
        effective_max_tokens = floor

    kwargs: Dict[str, Any] = dict(
        max_output_tokens=effective_max_tokens,
        temperature=temperature,
        seed=seed,
    )
    if system_instruction:
        kwargs["system_instruction"] = system_instruction

    if is_optional_thinking:
        try:
            budget = int(os.getenv("GEMINI_THINKING_BUDGET", "0"))
        except ValueError:
            budget = 0
        try:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=budget,
                include_thoughts=False,
            )
        except Exception:
            pass

    return types.GenerateContentConfig(**kwargs)


def _extract_response_text(response: Any, model: str) -> str:
    """Robustly extract text from a google-genai response.

    Handles cases where:
      - candidates is None / empty
      - content / parts is None (e.g. finish_reason=MAX_TOKENS with thinking)
      - response.text shortcut is preferred when available
    """
    if response is None:
        return ""

    # Prefer the SDK's own joined-text accessor when present.
    text_shortcut = getattr(response, "text", None)
    if isinstance(text_shortcut, str) and text_shortcut:
        return text_shortcut

    candidates = getattr(response, "candidates", None) or []
    pieces: List[str] = []
    finish_reasons: List[str] = []
    for cand in candidates:
        finish = getattr(cand, "finish_reason", None)
        if finish is not None:
            finish_reasons.append(str(finish))
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) if content is not None else None
        if not parts:
            continue
        for part in parts:
            t = getattr(part, "text", None)
            if isinstance(t, str) and t:
                pieces.append(t)

    if pieces:
        return "".join(pieces)

    usage = getattr(response, "usage_metadata", None)
    thoughts_tokens = getattr(usage, "thoughts_token_count", None)
    cand_tokens = getattr(usage, "candidates_token_count", None)
    logger.error(
        f"[Gemini] empty visible response from {model}: "
        f"finish_reasons={finish_reasons or 'unknown'}, "
        f"thoughts_tokens={thoughts_tokens}, candidates_tokens={cand_tokens}. "
        "If finish_reason=MAX_TOKENS, increase max_tokens or set "
        "GEMINI_THINKING_BUDGET=0 to disable thinking."
    )
    return ""



class GeminiProvider(LLMProvider):
    """A class that wraps a given model"""

    client: genai.Client = None
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
        self.client = genai.Client(api_key=key)

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
                "parts": [
                  {
                    "text": "What's in this image?"
                  },
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
                APIError, ClientError, ServerError, UnknownFunctionCallArgumentError, UnsupportedFunctionError, FunctionInvocationError
            ),
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

            system_instruction, contents_for_request = _legacy_messages_to_genai(messages)

            logger.write(
                f"Requesting completion..., System content: {system_instruction or ''}"
            )

            """Send a request to the Gemini API."""
            increment_llm_call_counter("big_brain:gemini")
            response = self.client.models.generate_content(
                model=model,
                contents=contents_for_request,
                config=_build_generate_config(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    system_instruction=system_instruction,
                ),
            )
            if response is None:
                logger.error("Failed to get a response from Gemini. Try again.")
                logger.double_check()

            message = _extract_response_text(response, model)

            usage = getattr(response, "usage_metadata", None)
            info = {
                "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
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
                    APIError, ClientError, ServerError, UnknownFunctionCallArgumentError, UnsupportedFunctionError, FunctionInvocationError
            ),
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

            system_instruction, contents_for_request = _legacy_messages_to_genai(messages)

            """Send a request to the Gemini API."""
            increment_llm_call_counter("big_brain:gemini_async")
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=model,
                contents=contents_for_request,
                config=_build_generate_config(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    seed=seed,
                    system_instruction=system_instruction,
                ),
            )

            if response is None:
                logger.error("Failed to get a response from Gemini. Try again.")
                logger.double_check()

            message = _extract_response_text(response, model)

            usage = getattr(response, "usage_metadata", None)
            info = {
                "input_tokens": getattr(usage, "prompt_token_count", 0) or 0,
                "output_tokens": getattr(usage, "candidates_token_count", 0) or 0,
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
        """Not implemented for Gemini models."""
        raise NotImplementedError(
            f"num_tokens_from_messages() is not implemented for Gemini model {model}."
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
            "parts": [
                {
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
            "parts": [
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
        #     "parts": [
        #         {
        #             "text": f"{user_messages_part1_content}"
        #         }
        #     ]
        # }
        combined_user_message["parts"].append({
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
            #     "parts": [
            #         {
            #             "text": f"{paragraph_content_pre}"
            #         }
            #     ]
            # }

            # image_introduction_messages.append(message)

            # path = params["image_path"] if "image_path" in params else ""
            # if path is not None and path != "":
            #     with open(path, 'rb') as file:
            #         binary_content = file.read()  # 读取文件内容
            #         base64_encoded = base64.standard_b64encode(binary_content)  # 转换为 Base64
            #         base64_string = base64_encoded.decode('utf-8')
            #     encoded_images = [base64_string]

            #     for encoded_image in encoded_images:
            #         msg_content = {
            #             "inline_data": {
            #                 "mime_type": "image/jpeg",
            #                 "data": encoded_image
            #             }
            #         }

            #         message["parts"].append(msg_content)
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
                combined_user_message["parts"].append({
                    "text": f"{msg_text}"
                })
                msg_content = {
                    "inline_data":
                        {
                            "mime_type": "image/jpeg",
                            "data": encoded_image
                        }
                }
                combined_user_message["parts"].append(msg_content)
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
            "parts": [
                {
                    "text": f"{user_messages_part2_content}"
                }
            ]
        }
        
        combined_user_message["parts"].append({
             "text": f"{user_messages_part2_content}"
         })

        # if user_messages_part1 is None:
        #     return [system_message] + image_introduction_messages + [user_messages_part2]
        # else:
        #     return [system_message] + [user_messages_part1] + image_introduction_messages + [user_messages_part2]
        
        return [system_message] + [combined_user_message]

    def assemble_prompt_paragraph(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError("This method is not implemented yet.")


    def assemble_prompt(self, template_str: str = None, params: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        if config.DEFAULT_MESSAGE_CONSTRUCTION_MODE == constants.MESSAGE_CONSTRUCTION_MODE_TRIPART:
            return self.assemble_prompt_tripartite(template_str=template_str, params=params)
        elif config.DEFAULT_MESSAGE_CONSTRUCTION_MODE == constants.MESSAGE_CONSTRUCTION_MODE_PARAGRAPH:
            return self.assemble_prompt_paragraph(template_str=template_str, params=params)
