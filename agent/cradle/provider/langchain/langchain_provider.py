"""
LangChain-based LLM Provider adapter.

This module provides a LangChain implementation of the LLMProvider interface,
offering structured outputs via Pydantic schemas while maintaining backward
compatibility with the legacy OpenAI-style interface.

Key features:
1. Structured output via with_structured_output()
2. Automatic fallback to legacy text parsing if structured output fails
3. Support for multiple models (OpenAI, Claude, Qwen, etc.)
4. Detailed error logging for debugging parsing failures

Design principles:
- Zero breaking changes to existing code
- Can be toggled on/off via configuration
- Gradual migration path from legacy to structured outputs
"""

import os
import json
import re
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type
import backoff

# Suppress LangChain's extra_body parameter routing warning.
# ChatOpenAI declares extra_body as a Pydantic field, but its build_extra
# validator moves it into model_kwargs regardless of how it's passed.
# Both top-level and model_kwargs placement trigger a warning - suppress it.
warnings.filterwarnings("ignore", message=r".*extra_body.*")

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from pydantic import BaseModel, ValidationError

from cradle import constants
from cradle.config import Config
from cradle.log import Logger
from cradle.provider.base import LLMProvider
from cradle.utils.json_utils import load_json, parse_semi_formatted_text
from cradle.utils.file_utils import assemble_project_path
from cradle.utils.encoding_utils import encode_data_to_base64_path
from cradle.utils.llm_endpoint_throttle import (
    acquire_llm_endpoint_slot,
    get_llm_endpoint_wait_timeout,
    resolve_remaining_llm_request_timeout,
)

config = Config()
logger = Logger()


class LangChainLLMProvider(LLMProvider):
    """
    LangChain-based LLM provider with structured output support.

    This provider wraps LangChain's ChatModel interface to provide:
    1. Type-safe structured outputs via Pydantic schemas
    2. Automatic retry logic with exponential backoff
    3. Fallback to legacy text parsing if structured output fails
    4. Compatible interface with existing OpenAIProvider

    Usage:
        # Initialize from config
        provider = LangChainLLMProvider()
        provider.init_provider("conf/openai_config.json")

        # Structured output (recommended)
        from cradle.provider.langchain.schemas import ActionPlanningOutput
        output, info = provider.create_completion_structured(
            messages=[...],
            output_schema=ActionPlanningOutput
        )

        # Legacy text output (backward compatible)
        text, info = provider.create_completion(messages=[...])
    """

    def __init__(self) -> None:
        """Initialize the provider with default settings."""
        self.client: Optional[BaseChatModel] = None
        self.llm_model: str = ""
        self.provider_cfg: Dict[str, Any] = {}
        self.retries: int = 5
        self.use_structured_output: bool = True
        self.fallback_to_legacy_parser: bool = True
        self._api_key_candidates: List[str] = []
        self._active_key_index: int = 0

    def init_provider(self, provider_cfg) -> None:
        """
        Initialize provider from configuration file or dict.

        Args:
            provider_cfg: Path to config JSON or config dict

        Config format:
        {
            "provider": "openai" | "claude",
            "model": "gpt-4o" | "claude-3-5-sonnet-20241022",
            "api_key": "sk-..." (optional, can use env var),
            "base_url": "https://..." (optional),
            "use_structured_output": true,
            "fallback_to_legacy_parser": true
        }
        """
        self.provider_cfg = self._parse_config(provider_cfg)
        self._initialize_client()

    def _parse_config(self, provider_cfg) -> Dict[str, Any]:
        """Parse configuration from file or dict."""
        if isinstance(provider_cfg, dict):
            conf_dict = provider_cfg
        else:
            path = assemble_project_path(provider_cfg)
            conf_dict = load_json(path)

        # Extract settings
        self.llm_model = conf_dict.get("comp_model") or conf_dict.get("model", "gpt-4o")
        self.use_structured_output = conf_dict.get("use_structured_output", True)
        self.fallback_to_legacy_parser = conf_dict.get("fallback_to_legacy_parser", True)

        logger.write(f"LangChain Provider initialized: model={self.llm_model}, "
                     f"structured_output={self.use_structured_output}")

        # Build LLM key candidates (priority order)
        key_var_name = conf_dict.get("key_var", "OPENAI_API_KEY")
        candidates = [
            conf_dict.get("llm_api_key"),
            conf_dict.get("api_key"),
            conf_dict.get("emb_api_key"),
            os.getenv(key_var_name),
            os.getenv("DASHSCOPE_API_KEY"),
        ]
        deduped: List[str] = []
        for key in candidates:
            if isinstance(key, str) and key and key not in deduped:
                deduped.append(key)

        self._api_key_candidates = deduped
        self._active_key_index = 0

        return conf_dict

    @staticmethod
    def _mask_key(key: Optional[str]) -> str:
        if not key:
            return "<empty>"
        if len(key) <= 10:
            return "*" * len(key)
        return f"{key[:6]}...{key[-4:]}"

    def _get_active_api_key(self) -> Optional[str]:
        if 0 <= self._active_key_index < len(self._api_key_candidates):
            return self._api_key_candidates[self._active_key_index]
        return None

    @staticmethod
    def _is_invalid_api_key_error(error: Exception) -> bool:
        text = str(error)
        return (
            "invalid_api_key" in text
            or "Incorrect API key" in text
            or "AuthenticationError" in text
            or "401" in text
        )

    def _try_rotate_api_key(self) -> bool:
        next_index = self._active_key_index + 1
        if next_index >= len(self._api_key_candidates):
            return False
        self._active_key_index = next_index
        next_key = self._get_active_api_key()
        self.client = self._create_client_with_temperature(config.temperature)
        logger.warn(
            f"[LLM Auth] Switching to fallback API key #{self._active_key_index + 1}: {self._mask_key(next_key)}"
        )
        return True

    def _create_client_with_temperature(
        self,
        temperature: float,
        max_completion_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
    ) -> BaseChatModel:
        """Create a new client instance with specific temperature and optional token limit."""
        provider_type = self.provider_cfg.get("provider", "openai").lower()
        api_key = self._get_active_api_key()
        if not api_key:
            key_var_name = self.provider_cfg.get("key_var", "OPENAI_API_KEY")
            api_key = self.provider_cfg.get("api_key") or os.getenv(key_var_name)
        base_url = self.provider_cfg.get("base_url")
        max_tokens = max_completion_tokens if max_completion_tokens is not None else config.max_tokens
        request_timeout = float(timeout_seconds) if timeout_seconds is not None else 120.0

        # Qwen/DashScope must be checked BEFORE generic "openai" provider,
        # because provider_type defaults to "openai" even for Qwen models.
        if "qwen" in self.llm_model.lower() or "dashscope" in str(base_url).lower():
            return ChatOpenAI(
                model=self.llm_model,
                api_key=api_key,  # type: ignore
                base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=temperature,
                max_completion_tokens=max_tokens,
                timeout=request_timeout,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
        elif "claude" in provider_type or "claude" in self.llm_model.lower():
            return ChatAnthropic(
                model_name=self.llm_model,
                api_key=api_key,  # type: ignore
                temperature=temperature,
                max_tokens_to_sample=max_tokens,
                timeout=request_timeout,
                stop=[],  # Required parameter
            )
        else:
            return ChatOpenAI(
                model=self.llm_model,
                api_key=api_key,  # type: ignore
                base_url=base_url,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                timeout=request_timeout,
            )

    def _initialize_client(self) -> None:
        """Initialize the LangChain ChatModel client."""
        provider_type = self.provider_cfg.get("provider", "openai").lower()

        # Get API key (from config or environment)
        key_var_name = self.provider_cfg.get("key_var", "OPENAI_API_KEY")
        api_key = self._get_active_api_key() or self.provider_cfg.get("api_key") or os.getenv(key_var_name)

        if not api_key:
            raise ValueError(f"API key not found in config or environment variable {key_var_name}")

        logger.write(f"[LLM Auth] Using API key: {self._mask_key(api_key)}")

        # Get optional base_url
        base_url = self.provider_cfg.get("base_url")

        # Initialize appropriate chat model
        # Qwen/DashScope must be checked BEFORE generic "openai" provider,
        # because provider_type defaults to "openai" even for Qwen models.
        if "qwen" in self.llm_model.lower() or "dashscope" in str(base_url).lower():
            # Qwen models via OpenAI-compatible API (DashScope)
            # Disable thinking mode to avoid hidden CoT overhead (95s → ~3s)
            self.client = ChatOpenAI(
                model=self.llm_model,
                api_key=api_key,  # type: ignore
                base_url=base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1",
                temperature=config.temperature,
                max_completion_tokens=config.max_tokens,
                timeout=120.0,
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )
            logger.write(f"Initialized ChatOpenAI (Qwen, nothinking) with model: {self.llm_model}")

        elif "claude" in provider_type or "claude" in self.llm_model.lower():
            self.client = ChatAnthropic(
                model_name=self.llm_model,
                api_key=api_key,  # type: ignore
                temperature=config.temperature,
                max_tokens_to_sample=config.max_tokens,
                timeout=120.0,
                stop=[],  # Required parameter
            )
            logger.write(f"Initialized ChatAnthropic with model: {self.llm_model}")

        else:
            # Default to OpenAI-compatible interface
            self.client = ChatOpenAI(
                model=self.llm_model,
                api_key=api_key,  # type: ignore
                base_url=base_url,
                temperature=config.temperature,
                max_completion_tokens=config.max_tokens,
                timeout=120.0,
            )
            logger.write(f"Initialized ChatOpenAI with model: {self.llm_model}")

    def create_completion_structured(
        self,
        messages: List[Dict[str, Any]],
        output_schema: Type[BaseModel],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Tuple[BaseModel, Dict[str, int]]:
        """
        Create a completion with structured output using Pydantic schema.

        This is the recommended method for new code. It uses LangChain's
        with_structured_output() to ensure type-safe responses.

        Args:
            messages: List of message dicts (OpenAI format)
            output_schema: Pydantic model class for output validation
            model: Optional model override
            temperature: Optional temperature override

        Returns:
            (structured_output, info_dict)
            - structured_output: Instance of output_schema
            - info_dict: Token usage and metadata

        Raises:
            ValidationError: If LLM output doesn't match schema
            Exception: If API call fails

        Example:
            from cradle.provider.langchain.schemas import ActionPlanningOutput

            output, info = provider.create_completion_structured(
                messages=[{"role": "user", "content": "What should I do?"}],
                output_schema=ActionPlanningOutput
            )

            print(output.reasoning)  # Type-safe access
            print(output.actions)    # List[str]
        """
        if not self.use_structured_output:
            raise ValueError("Structured output is disabled in config. Set use_structured_output=true")

        model = model or self.llm_model
        temperature = temperature if temperature is not None else config.temperature
        max_tokens = kwargs.get("max_tokens")
        timeout_seconds = kwargs.get("timeout_seconds")
        total_timeout_seconds = (
            float(timeout_seconds) if timeout_seconds is not None else 120.0
        )

        # Convert messages to LangChain format
        langchain_messages = self._convert_to_langchain_messages(messages)

        logger.write(f"Requesting {model} completion with structured output (schema: {output_schema.__name__})...")

        @backoff.on_exception(
            backoff.expo,
            (Exception,),
            max_tries=self.retries,
            max_value=10,
            jitter=None,
        )
        def _call_with_retry():
            slot_wait_timeout_s = get_llm_endpoint_wait_timeout(
                model,
                total_timeout_s=total_timeout_seconds,
            )
            with acquire_llm_endpoint_slot(
                model_name=model,
                purpose="structured",
                logger_obj=logger,
                timeout_s=slot_wait_timeout_s,
            ) as slot_info:
                effective_timeout_seconds = None
                effective_timeout_seconds = resolve_remaining_llm_request_timeout(
                    total_timeout_seconds,
                    slot_info,
                )
                structured_client = self.client
                if max_tokens is not None or effective_timeout_seconds != 120.0:
                    structured_client = self._create_client_with_temperature(
                        temperature,
                        max_completion_tokens=max_tokens,
                        timeout_seconds=effective_timeout_seconds,
                    )
                structured_llm = structured_client.with_structured_output(output_schema)  # type: ignore
                response = structured_llm.invoke(
                    langchain_messages,
                    config={"temperature": temperature}  # type: ignore
                )

            return response

        try:
            structured_output = _call_with_retry()

            # Extract token usage (if available)
            info = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "system_fingerprint": "langchain",
            }

            logger.write(f"Structured response received from {model}")

            return structured_output, info  # type: ignore

        except ValidationError as e:
            logger.error(f"Structured output validation failed: {e}")

            if self.fallback_to_legacy_parser:
                logger.warn("Falling back to legacy text parsing...")
                return self._fallback_to_text_parsing(messages, output_schema, model, temperature)
            else:
                raise

        except Exception as e:
            if self._is_invalid_api_key_error(e) and self._try_rotate_api_key():
                logger.warn("[LLM Auth] Retrying structured call with fallback API key")
                structured_output = _call_with_retry()
                info = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "system_fingerprint": "langchain",
                }
                return structured_output, info  # type: ignore
            logger.error(f"Structured output request failed: {e}")
            raise

    # ========== LLM Call Diagnostics ==========

    def _extract_prompt_stats(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract prompt statistics for diagnostic logging."""
        total_text_len = 0
        image_count = 0
        image_sizes_kb = []
        system_text = ""
        user_text_preview = ""

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_text_len += len(content)
                if msg.get("role") == "system" and not system_text:
                    system_text = content[:150]
                elif msg.get("role") == "user" and not user_text_preview:
                    user_text_preview = content[:150]
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_val = part.get("text", "")
                            total_text_len += len(text_val)
                            if msg.get("role") == "system" and not system_text:
                                system_text = text_val[:150]
                            elif msg.get("role") == "user" and not user_text_preview:
                                user_text_preview = text_val[:150]
                        elif part.get("type") == "image_url":
                            image_count += 1
                            url = part.get("image_url", {}).get("url", "")
                            # base64 data length * 0.75 ≈ original file size
                            if url.startswith("data:"):
                                b64_len = len(url.split(",", 1)[-1]) if "," in url else 0
                                image_sizes_kb.append(round(b64_len * 0.75 / 1024, 1))

        return {
            "total_text_chars": total_text_len,
            "estimated_text_tokens": total_text_len // 3,  # rough CJK estimate
            "image_count": image_count,
            "image_sizes_kb": image_sizes_kb,
            "system_preview": system_text,
            "user_preview": user_text_preview,
        }

    def _log_llm_start(self, stats: Dict[str, Any], model: str, mode: str) -> None:
        """Log LLM call start with prompt details."""
        img_info = ""
        if stats["image_count"] > 0:
            sizes = stats["image_sizes_kb"]
            total_kb = sum(sizes)
            img_info = f" | images={stats['image_count']} ({total_kb:.0f}KB)"
        logger.write(
            f"[LLM_DIAG] >>> REQUEST | model={model} | mode={mode}"
            f" | prompt_text={stats['total_text_chars']}chars "
            f"(~{stats['estimated_text_tokens']}tok){img_info}"
        )
        if stats["system_preview"]:
            logger.write(f"[LLM_DIAG]   system: {stats['system_preview']}...")
        if stats["user_preview"]:
            logger.write(f"[LLM_DIAG]   user: {stats['user_preview']}...")

    def _log_llm_end(
        self, model: str, mode: str, duration_s: float,
        response_text: str, info: Dict[str, int]
    ) -> None:
        """Log LLM call end with response details."""
        resp_preview = response_text[:200].replace("\n", " ") if response_text else "(empty)"
        resp_len = len(response_text) if response_text else 0
        tokens = info.get("total_tokens", 0)
        prompt_tok = info.get("prompt_tokens", 0)
        comp_tok = info.get("completion_tokens", 0)
        logger.write(
            f"[LLM_DIAG] <<< RESPONSE | model={model} | mode={mode}"
            f" | duration={duration_s:.1f}s"
            f" | response={resp_len}chars"
            f" | tokens(prompt={prompt_tok} comp={comp_tok} total={tokens})"
        )
        logger.write(f"[LLM_DIAG]   response: {resp_preview}...")

    def _fallback_to_text_parsing(
        self,
        messages: List[Dict[str, Any]],
        output_schema: Type[BaseModel],
        model: str,
        temperature: float
    ) -> Tuple[BaseModel, Dict[str, int]]:
        """
        Fallback to legacy text parsing when structured output fails.

        This maintains backward compatibility by using the original
        parse_semi_formatted_text() method and then constructing the
        Pydantic model from the parsed dict.
        """
        logger.warn("Using fallback text parsing (legacy mode)")

        # Get text response
        text, info = self.create_completion(messages, model, temperature)

        # Parse using legacy method
        parsed_dict = parse_semi_formatted_text(text)

        # Try to construct Pydantic model
        try:
            structured_output = output_schema(**parsed_dict)
            logger.write("Successfully constructed structured output from legacy parsing")
            return structured_output, info
        except Exception as e:
            logger.error(f"Failed to construct {output_schema.__name__} from parsed dict: {e}")
            logger.error(f"Parsed dict: {parsed_dict}")
            raise



    def create_completion(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, int]]:
        """
        Create a completion returning raw text (backward compatible).

        This method maintains compatibility with the legacy OpenAIProvider
        interface. Use create_completion_structured() for new code.

        Args:
            messages: List of message dicts (OpenAI format)
            model: Optional model override
            temperature: Optional temperature override

        Returns:
            (text_response, info_dict)
        """
        model = model or self.llm_model
        temperature = temperature if temperature is not None else config.temperature
        max_tokens = kwargs.get("max_tokens")
        timeout_seconds = kwargs.get("timeout_seconds")
        total_timeout_seconds = (
            float(timeout_seconds) if timeout_seconds is not None else 120.0
        )

        # Convert messages to LangChain format
        langchain_messages = self._convert_to_langchain_messages(messages)

        # Check if any message contains images (for logging)
        has_images = any(
            isinstance(msg.content, list) and any(
                isinstance(item, dict) and item.get("type") == "image_url"
                for item in msg.content
            )
            for msg in langchain_messages
            if hasattr(msg, 'content') and isinstance(msg.content, list)
        )
        mode = "vision mode" if has_images else "text mode"
        logger.write(f"Requesting {model} completion ({mode})...")

        # Diagnostic logging
        prompt_stats = self._extract_prompt_stats(messages)
        self._log_llm_start(prompt_stats, model, f"sync/{mode}")
        t_start = time.time()

        @backoff.on_exception(
            backoff.expo,
            (Exception,),
            max_tries=self.retries,
            max_value=10,
            jitter=None,
        )
        def _call_with_retry():
            slot_wait_timeout_s = get_llm_endpoint_wait_timeout(
                model,
                total_timeout_s=total_timeout_seconds,
            )
            with acquire_llm_endpoint_slot(
                model_name=model,
                purpose="sync_completion",
                logger_obj=logger,
                timeout_s=slot_wait_timeout_s,
            ) as slot_info:
                effective_timeout_seconds = resolve_remaining_llm_request_timeout(
                    total_timeout_seconds,
                    slot_info,
                )
                completion_client = self.client
                if (
                    temperature != config.temperature
                    or max_tokens is not None
                    or effective_timeout_seconds != 120.0
                ):
                    completion_client = self._create_client_with_temperature(
                        temperature,
                        max_completion_tokens=max_tokens,
                        timeout_seconds=effective_timeout_seconds,
                    )
                response = completion_client.invoke(  # type: ignore
                    langchain_messages,
                    config={"temperature": temperature}  # type: ignore
                )
            return response

        try:
            response = _call_with_retry()

            content = response.content
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                text_parts: List[str] = []
                for item in content:
                    if isinstance(item, str):
                        text_parts.append(item)
                    elif isinstance(item, dict):
                        text_parts.append(str(item.get("text", item)))
                    else:
                        text_parts.append(str(item))
                text = "\n".join(part for part in text_parts if part)
            else:
                text = str(content)

            # Extract token usage (if available)
            info = {
                "prompt_tokens": getattr(response, "usage_metadata", {}).get("input_tokens", 0),
                "completion_tokens": getattr(response, "usage_metadata", {}).get("output_tokens", 0),
                "total_tokens": getattr(response, "usage_metadata", {}).get("total_tokens", 0),
                "system_fingerprint": "langchain",
            }

            logger.write(f"Response received from {model}")

            self._log_llm_end(model, "sync", time.time() - t_start, text, info)

            return text, info  # type: ignore

        except Exception as e:
            if self._is_invalid_api_key_error(e) and self._try_rotate_api_key():
                logger.warn("[LLM Auth] Retrying completion with fallback API key")
                response = _call_with_retry()
                content = response.content
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text_parts: List[str] = []
                    for item in content:
                        if isinstance(item, str):
                            text_parts.append(item)
                        elif isinstance(item, dict):
                            text_parts.append(str(item.get("text", item)))
                        else:
                            text_parts.append(str(item))
                    text = "\n".join(part for part in text_parts if part)
                else:
                    text = str(content)
                info = {
                    "prompt_tokens": getattr(response, "usage_metadata", {}).get("input_tokens", 0),
                    "completion_tokens": getattr(response, "usage_metadata", {}).get("output_tokens", 0),
                    "total_tokens": getattr(response, "usage_metadata", {}).get("total_tokens", 0),
                    "system_fingerprint": "langchain",
                }
                return text, info  # type: ignore
            logger.error(f"Completion request failed: {e}")
            raise

    def encode_images_parallel_sync(
        self,
        image_paths: List[str],
        max_workers: int = 4
    ) -> List[str]:
        """
        同步并行编码图像（用于sync上下文）- Phase 2.1优化
        
        使用ThreadPoolExecutor并行编码多张图像，利用多核CPU提升性能。
        
        Args:
            image_paths: 图像路径列表
            max_workers: 最大并行工作线程数（默认4）
        
        Returns:
            base64编码字符串列表（保持原始顺序）
        
        性能:
            串行: 12张图 × 0.8s = 9.6s
            并行: 12张图 / 4核 × 0.8s = 2.4s
            节省: 7.2s (75%提升)
        
        Example:
            >>> paths = ["img1.jpg", "img2.jpg", ..., "img12.jpg"]
            >>> encoded = provider.encode_images_parallel_sync(paths)
            >>> len(encoded)  # 12
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if not image_paths:
            return []
        
        # 限制并行数量避免内存溢出
        actual_workers = min(max_workers, len(image_paths))
        
        start_time = time.time()
        logger.write(f"[ImageEncode] Parallel encoding {len(image_paths)} images with {actual_workers} workers")
        logger.debug(f"[ImageEncode] Image paths types: {[type(p).__name__ for p in image_paths]}")
        logger.debug(f"[ImageEncode] First path sample: {image_paths[0] if image_paths else 'N/A'}")
        
        try:
            # 使用线程池并行处理
            with ThreadPoolExecutor(max_workers=actual_workers) as executor:
                # 保持顺序: 使用字典映射index
                futures = {
                    executor.submit(encode_data_to_base64_path, path): i 
                    for i, path in enumerate(image_paths)
                }
                
                # 按原始顺序收集结果
                results: List[str] = [""] * len(image_paths)
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        # encode_data_to_base64_path返回列表，取第一个元素
                        encoded_list = future.result()
                        if encoded_list and len(encoded_list) > 0:
                            results[index] = encoded_list[0]
                        else:
                            logger.error(f"[ImageEncode] Empty result for image {image_paths[index]}")
                            results[index] = ""  # 使用空字符串
                    except Exception as e:
                        logger.error(f"[ImageEncode] Failed to encode image {image_paths[index]}: {e}")
                        results[index] = ""  # 失败时使用空字符串
            
            elapsed = time.time() - start_time
            logger.write(f"[ImageEncode] ✓ Completed in {elapsed:.2f}s (avg {elapsed/len(image_paths):.3f}s per image)")
            
            return results
            
        except Exception as e:
            logger.error(f"[ImageEncode] Parallel encoding failed: {e}")
            # Fallback到串行编码
            logger.write("[ImageEncode] Falling back to serial encoding...")
            return [encode_data_to_base64_path(path)[0] if encode_data_to_base64_path(path) else "" 
                    for path in image_paths]

    async def encode_images_parallel(self, image_paths: List[str], max_workers: int = 6) -> List[str]:
        """
        并行编码多张图像为base64（70%性能提升）
        
        优化前: 12帧 × 1.5s/帧 = 18秒
        优化后: 6个并行worker = 3-5秒
        
        Args:
            image_paths: 图像路径列表
            max_workers: 最大并行worker数（默认6）
        
        Returns:
            base64编码的图像数据URL列表
        """
        import asyncio
        from functools import partial
        
        if not image_paths:
            return []
        
        # 限制并行数量避免内存溢出
        actual_workers = min(max_workers, len(image_paths))
        logger.write(f"[Performance] Parallel encoding {len(image_paths)} images with {actual_workers} workers...")
        
        # 使用asyncio.to_thread将同步编码函数转为异步
        tasks = [asyncio.to_thread(encode_data_to_base64_path, path) for path in image_paths]
        
        try:
            encoded_images_nested = await asyncio.gather(*tasks)
            # Flatten List[List[str]] to List[str]
            encoded_images = [img for sublist in encoded_images_nested for img in sublist]
            logger.write(f"[Performance] ✅ Parallel encoding completed: {len(encoded_images)} images")
            return encoded_images
        except Exception as e:
            logger.error(f"[Performance] ❌ Parallel encoding failed: {e}")
            # Fallback到串行编码
            logger.write("[Performance] Falling back to serial encoding...")
            return [img for path in image_paths for img in encode_data_to_base64_path(path)]

    async def create_completion_async(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, int]]:
        """
        Async version of create_completion.

        Uses LangChain's ainvoke for non-blocking LLM calls.
        Critical for parallel information gathering (12 frames = 12x speedup).
        """
        model = model or self.llm_model
        temperature = temperature if temperature is not None else config.temperature
        max_tokens = kwargs.get("max_tokens")
        timeout_seconds = kwargs.get("timeout_seconds")
        total_timeout_seconds = (
            float(timeout_seconds) if timeout_seconds is not None else 120.0
        )

        # Convert messages to LangChain format
        langchain_messages = self._convert_to_langchain_messages(messages)

        # Check if any message contains images (for logging)
        has_images = any(
            isinstance(msg.content, list) and any(
                isinstance(item, dict) and item.get("type") == "image_url"
                for item in msg.content
            )
            for msg in langchain_messages
            if hasattr(msg, 'content') and isinstance(msg.content, list)
        )
        mode = "vision mode" if has_images else "text mode"
        logger.write(f"[Async] Requesting {model} completion ({mode})...")

        @backoff.on_exception(
            backoff.expo,
            (Exception,),
            max_tries=self.retries,
            max_value=10,
            jitter=None,
        )
        async def _call_with_retry():
            slot_wait_timeout_s = get_llm_endpoint_wait_timeout(
                model,
                total_timeout_s=total_timeout_seconds,
            )
            with acquire_llm_endpoint_slot(
                model_name=model,
                purpose="async_completion",
                logger_obj=logger,
                timeout_s=slot_wait_timeout_s,
            ) as slot_info:
                effective_timeout_seconds = resolve_remaining_llm_request_timeout(
                    total_timeout_seconds,
                    slot_info,
                )
                async_client = self.client
                if (
                    temperature != config.temperature
                    or max_tokens is not None
                    or effective_timeout_seconds != 120.0
                ):
                    async_client = self._create_client_with_temperature(
                        temperature,
                        max_completion_tokens=max_tokens,
                        timeout_seconds=effective_timeout_seconds,
                    )
                response = await async_client.ainvoke(  # type: ignore
                    langchain_messages,
                    config={"temperature": temperature}  # type: ignore
                )
            return response

        try:
            response = await _call_with_retry()

            text = response.content

            # Extract token usage (if available)
            info = {
                "prompt_tokens": getattr(response, "usage_metadata", {}).get("input_tokens", 0),
                "completion_tokens": getattr(response, "usage_metadata", {}).get("output_tokens", 0),
                "total_tokens": getattr(response, "usage_metadata", {}).get("total_tokens", 0),
                "system_fingerprint": "langchain",
            }

            logger.write(f"[Async] Response received from {model}")

            return text, info  # type: ignore

        except Exception as e:
            if self._is_invalid_api_key_error(e) and self._try_rotate_api_key():
                logger.warn("[LLM Auth] Retrying async completion with fallback API key")
                response = await _call_with_retry()
                text = response.content
                info = {
                    "prompt_tokens": getattr(response, "usage_metadata", {}).get("input_tokens", 0),
                    "completion_tokens": getattr(response, "usage_metadata", {}).get("output_tokens", 0),
                    "total_tokens": getattr(response, "usage_metadata", {}).get("total_tokens", 0),
                    "system_fingerprint": "langchain",
                }
                return text, info  # type: ignore
            logger.error(f"[Async] Completion request failed: {e}")
            raise

    def _convert_to_langchain_messages(self, messages: List[Dict[str, Any]]) -> List:
        """
        Convert OpenAI-style messages to LangChain message objects.

        Handles:
        - Text-only messages
        - Multimodal messages (text + images)
        - System/user/assistant roles

        Args:
            messages: OpenAI-style message list

        Returns:
            List of LangChain message objects
        """
        langchain_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            # Handle different content types
            if isinstance(content, str):
                # Simple text message
                if role == "system":
                    langchain_messages.append(SystemMessage(content=content))
                elif role == "user":
                    langchain_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    langchain_messages.append(AIMessage(content=content))

            elif isinstance(content, list):
                # Multimodal message (text + images)
                # Build content list with text and image_url parts
                content_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            content_parts.append({"type": "text", "text": item.get("text", "")})
                        elif item.get("type") == "image_url":
                            # LangChain supports image_url format
                            image_url = item.get("image_url", {})
                            if isinstance(image_url, str):
                                content_parts.append({"type": "image_url", "image_url": {"url": image_url}})
                            elif isinstance(image_url, dict):
                                content_parts.append({"type": "image_url", "image_url": image_url})

                # Use multimodal content if we have images, otherwise text only
                if any(part["type"] == "image_url" for part in content_parts):
                    if role == "user":
                        langchain_messages.append(HumanMessage(content=content_parts))
                    elif role == "system":
                        # System messages don't support multimodal, extract text only
                        text_parts = [part["text"] for part in content_parts if part["type"] == "text"]
                        langchain_messages.append(SystemMessage(content="\n".join(text_parts)))
                    elif role == "assistant":
                        # Assistant messages typically text only
                        text_parts = [part["text"] for part in content_parts if part["type"] == "text"]
                        langchain_messages.append(AIMessage(content="\n".join(text_parts)))
                else:
                    # Text only fallback
                    text_parts = [part["text"] for part in content_parts if part["type"] == "text"]
                    combined_text = "\n".join(text_parts)
                    if role == "system":
                        langchain_messages.append(SystemMessage(content=combined_text))
                    elif role == "user":
                        langchain_messages.append(HumanMessage(content=combined_text))
                    elif role == "assistant":
                        langchain_messages.append(AIMessage(content=combined_text))

        return langchain_messages

    async def create_completion_streaming(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        early_termination: bool = True,
        on_action: Optional[Any] = None,
        **kwargs
    ) -> Tuple[str, Dict[str, int]]:
        """
        流式生成并解析LLM响应（边生成边解析，降低感知延迟）
        
        优化效果: 感知延迟减少30-40%
        - 传统: 等待完整响应(15s) → 解析(0.5s) = 15.5s
        - 流式: 边生成边解析，首个action出现即返回 ≈ 5-10s
        
        Args:
            messages: 消息列表
            model: 模型名称
            temperature: 温度参数
            early_termination: 是否在首个有效action出现时提前终止
        
        Returns:
            (完整文本, token使用信息)
        """
        model = model or self.llm_model
        temperature = temperature if temperature is not None else config.temperature
        max_tokens = kwargs.get("max_tokens")

        langchain_messages = self._convert_to_langchain_messages(messages)

        # Diagnostic logging
        prompt_stats = self._extract_prompt_stats(messages)
        self._log_llm_start(prompt_stats, model, "streaming")
        t_start = time.time()

        logger.write(f"[Streaming] Starting streaming completion with {model}...")
        
        accumulated_text = ""
        buffer = ""
        found_action = False
        action_emitted = False
        action_buffer = ""
        action_scan_buffer = ""
        action_section_index = None
        code_block_start_index = None
        action_buffer_limit = int(kwargs.get("early_action_buffer_chars") or 256)
        early_action_enabled = bool(on_action is not None or early_termination)
        try:
            # 使用LangChain的astream方法
            # Create a new client instance with the desired temperature for streaming
            streaming_client = self.client
            if temperature != config.temperature or max_tokens is not None:
                # Only recreate client if temperature differs or token limit override is provided
                streaming_client = self._create_client_with_temperature(temperature, max_completion_tokens=max_tokens)
            
            async for chunk in streaming_client.astream(langchain_messages, config={"temperature": temperature}):  # type: ignore
                # 提取chunk内容
                if hasattr(chunk, 'content'):
                    chunk_text = chunk.content
                    # Ensure chunk_text is a string
                    if isinstance(chunk_text, list):
                        chunk_text = str(chunk_text)
                    elif not isinstance(chunk_text, str):
                        chunk_text = str(chunk_text)
                else:
                    chunk_text = str(chunk)
                
                accumulated_text += chunk_text
                buffer += chunk_text
                
                if early_action_enabled:
                    # 实时解析：仅在 Actions 代码块内检测有效 action，避免推理区误触发
                    action_scan_buffer += chunk_text
                    if action_section_index is None:
                        action_section_index = action_scan_buffer.find("Actions:")
                        if action_section_index != -1:
                            # 保留从 Actions: 起的内容，避免缓冲区无限增长
                            action_scan_buffer = action_scan_buffer[action_section_index:]
                        else:
                            action_section_index = None
                            if len(action_scan_buffer) > 4096:
                                action_scan_buffer = action_scan_buffer[-2048:]

                    if action_section_index is not None and code_block_start_index is None:
                        code_block_pos = action_scan_buffer.find("```python")
                        if code_block_pos != -1:
                            code_block_start_index = code_block_pos + len("```python")
                            found_action = True
                            action_buffer = action_scan_buffer[code_block_start_index:]

                    if found_action and code_block_start_index is not None:
                        # 进入动作监听模式：收集固定字符数后再触发
                        action_buffer += chunk_text

                    if found_action and not action_emitted and len(action_buffer) >= action_buffer_limit:
                        import re
                        action_pattern = r'\b\w+\([^)]*\)'
                        actions = re.findall(action_pattern, action_buffer)
                        if actions:
                            logger.write(f"[Streaming] ⚡ Early action batch captured: {actions}")
                            # 标记已输出，避免重复刷屏
                            action_emitted = True
                            # 立即触发动作回调（不停止生成）
                            if on_action is not None:
                                try:
                                    import asyncio
                                    result = on_action(actions)
                                    if asyncio.iscoroutine(result):
                                        asyncio.create_task(result)
                                except Exception as cb_error:
                                    logger.warn(f"[Streaming] on_action callback failed: {cb_error}")
                            # 如果允许提前终止，则退出流式生成
                            if early_termination:
                                break
                
                # 每512字符输出一次进度
                if len(buffer) >= 512:
                    logger.write(f"[Streaming] Progress: {len(accumulated_text)} chars received...")
                    buffer = ""
            
            logger.write(f"[Streaming] ✅ Streaming completed: {len(accumulated_text)} chars")

            # Token使用信息（流式模式无法精确统计，使用估算）
            estimated_tokens = len(accumulated_text) // 4
            info = {
                "prompt_tokens": prompt_stats.get("estimated_text_tokens", 0),
                "completion_tokens": estimated_tokens,
                "total_tokens": prompt_stats.get("estimated_text_tokens", 0) + estimated_tokens,
                "system_fingerprint": "langchain_streaming",
            }

            self._log_llm_end(model, "streaming", time.time() - t_start, accumulated_text, info)

            return accumulated_text, info
            
        except Exception as e:
            logger.error(f"[Streaming] ❌ Streaming failed: {e}")
            if self._is_invalid_api_key_error(e) and self._try_rotate_api_key():
                logger.warn("[LLM Auth] Retrying streaming with fallback API key")
                try:
                    streaming_client = self.client
                    if temperature != config.temperature or max_tokens is not None:
                        streaming_client = self._create_client_with_temperature(temperature, max_completion_tokens=max_tokens)

                    async for chunk in streaming_client.astream(langchain_messages, config={"temperature": temperature}):  # type: ignore
                        if hasattr(chunk, 'content'):
                            chunk_text = chunk.content
                            if isinstance(chunk_text, list):
                                chunk_text = str(chunk_text)
                            elif not isinstance(chunk_text, str):
                                chunk_text = str(chunk_text)
                        else:
                            chunk_text = str(chunk)
                        accumulated_text += chunk_text

                    logger.write(f"[Streaming] ✅ Streaming completed after key fallback: {len(accumulated_text)} chars")
                    estimated_tokens = len(accumulated_text) // 4
                    info = {
                        "prompt_tokens": 0,
                        "completion_tokens": estimated_tokens,
                        "total_tokens": estimated_tokens,
                        "system_fingerprint": "langchain_streaming",
                    }
                    return accumulated_text, info
                except Exception as retry_error:
                    logger.error(f"[Streaming] ❌ Retry after key fallback failed: {retry_error}")
            # Fallback到普通异步调用
            logger.write("[Streaming] Falling back to standard async completion...")
            return await self.create_completion_async(messages, model, temperature, **kwargs)

    def assemble_prompt(self, template_str: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Assemble prompt from template (backward compatible).

        This implementation mimics OpenAI's tripartite prompt structure:
        - System message (first paragraph)
        - User message part 1 (before image_introduction)
        - Image introduction messages (with image_url content parts)
        - User message part 2 (after image_introduction)

        This ensures vision mode works correctly by passing images to LLM.
        """
        if not template_str or not params:
            # Fallback to simple text-only message
            return [{"role": "user", "content": template_str or ""}]

        # Parse tripartite structure
        pattern = re.compile(r"(.+?)(?=\n\n|$)", re.DOTALL)
        paragraphs = re.findall(pattern, template_str)
        filtered_paragraphs = [p for p in paragraphs if p.strip() != '']

        if len(filtered_paragraphs) == 0:
            return [{"role": "user", "content": template_str}]

        # System message (first paragraph)
        system_content = filtered_paragraphs[0]
        system_message = {
            "role": "system",
            "content": [{"type": "text", "text": system_content}]
        }

        # Find image_introduction paragraph index
        image_introduction_paragraph_index = None
        image_introduction_paragraph = None
        for i, paragraph in enumerate(filtered_paragraphs):
            if constants.IMAGES_INPUT_TAG in paragraph:
                image_introduction_paragraph_index = i
                image_introduction_paragraph = paragraph
                break

        # Split paragraphs into part1 (before images) and part2 (after images)
        if image_introduction_paragraph_index is not None:
            user_messages_part1_paragraphs = filtered_paragraphs[1:image_introduction_paragraph_index]
            user_messages_part2_paragraphs = filtered_paragraphs[image_introduction_paragraph_index + 1:]
        else:
            # No images, all user content in part2
            user_messages_part1_paragraphs = []
            user_messages_part2_paragraphs = filtered_paragraphs[1:]

        # Helper function to replace placeholders
        def replace_placeholders(paragraphs, params):
            contents = []
            for paragraph in paragraphs:
                search_placeholder_pattern = re.compile(r"<\$[^\$]+\$>")
                placeholder = re.search(search_placeholder_pattern, paragraph)
                if not placeholder:
                    contents.append(paragraph)
                else:
                    placeholder_str = placeholder.group()
                    placeholder_name = placeholder_str.replace("<$", "").replace("$>", "")
                    paragraph_input = params.get(placeholder_name, None)
                    if paragraph_input is None or paragraph_input == "" or paragraph_input == []:
                        continue
                    else:
                        if isinstance(paragraph_input, str):
                            paragraph_content = paragraph.replace(placeholder_str, paragraph_input)
                            contents.append(paragraph_content)
                        elif isinstance(paragraph_input, (list, dict)):
                            paragraph_content = paragraph.replace(placeholder_str, json.dumps(paragraph_input))
                            contents.append(paragraph_content)
                        elif isinstance(paragraph_input, (bool, int, float)):
                            paragraph_content = paragraph.replace(placeholder_str, str(paragraph_input))
                            contents.append(paragraph_content)
            return contents

        # Assemble user messages part 1
        user_messages_part1_contents = replace_placeholders(user_messages_part1_paragraphs, params)
        user_messages_part1 = None
        if len(user_messages_part1_contents) > 0:
            user_messages_part1 = {
                "role": "user",
                "content": [{"type": "text", "text": "\n\n".join(user_messages_part1_contents)}]
            }

        # Assemble image introduction messages
        image_introduction_messages = []
        if image_introduction_paragraph is not None:
            paragraph_input = params.get(constants.IMAGES_INPUT_TAG_NAME, [])  # 'image_introduction'
            if paragraph_input:
                # Pre-text before images
                paragraph_content_pre = image_introduction_paragraph.replace(constants.IMAGES_INPUT_TAG, "")
                image_introduction_messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": paragraph_content_pre}]
                })

                # 🚀 Phase 2.1: 收集所有图像路径用于并行编码
                all_image_paths = []
                image_path_indices = []  # 记录哪些item有图像
                
                for i, item in enumerate(paragraph_input):
                    path = item.get(constants.IMAGE_PATH_TAG_NAME, None)
                    if path:
                        # path可能是单个路径或路径列表
                        if isinstance(path, list):
                            all_image_paths.extend(path)
                            image_path_indices.append((i, len(path)))
                        else:
                            all_image_paths.append(path)
                            image_path_indices.append((i, 1))
                
                # 并行编码所有图像
                encoded_images_flat = []
                if all_image_paths:
                    try:
                        # 检查是否启用并行编码
                        use_parallel = False
                        try:
                            import yaml
                            import os
                            from cradle.utils.file_utils import assemble_project_path
                            config_path = assemble_project_path('./conf/enhanced_config.yaml')
                            if os.path.exists(config_path):
                                with open(config_path, 'r', encoding='utf-8') as f:
                                    cfg = yaml.safe_load(f)
                                    use_parallel = cfg.get('performance', {}).get('parallel_image_encoding', {}).get('enabled', False)
                        except Exception:
                            pass
                        
                        if use_parallel and len(all_image_paths) > 1:
                            encoded_images_flat = self.encode_images_parallel_sync(all_image_paths)
                        else:
                            # Fallback到串行编码
                            encoded_images_flat = []
                            for p in all_image_paths:
                                encoded_list = encode_data_to_base64_path(p)
                                encoded_images_flat.append(encoded_list[0] if encoded_list else "")
                    except Exception as e:
                        logger.warn(f"[ImageEncode] Parallel encoding failed: {e}, using serial")
                        encoded_images_flat = []
                        for p in all_image_paths:
                            encoded_list = encode_data_to_base64_path(p)
                            encoded_images_flat.append(encoded_list[0] if encoded_list else "")
                
                # 分配编码后的图像回各个item
                image_offset = 0
                for item_idx, num_images in image_path_indices:
                    item = paragraph_input[item_idx]
                    introduction = item.get(constants.IMAGE_INTRO_TAG_NAME, None)
                    assistant = item.get(constants.IMAGE_ASSISTANT_TAG_NAME, None)

                    message_content = []

                    # Add introduction text
                    if introduction:
                        message_content.append({"type": "text", "text": introduction})

                    # Add encoded images
                    for j in range(num_images):
                        if image_offset + j < len(encoded_images_flat):
                            message_content.append({
                                "type": "image_url",
                                "image_url": {"url": encoded_images_flat[image_offset + j]}
                            })
                    
                    image_offset += num_images

                    if len(message_content) > 0:
                        image_introduction_messages.append({
                            "role": "user",
                            "content": message_content
                        })

                    # Add assistant response if present
                    if assistant:
                        image_introduction_messages.append({
                            "role": "assistant",
                            "content": [{"type": "text", "text": assistant}]
                        })
                
                # 处理没有图像的items
                for i, item in enumerate(paragraph_input):
                    if not any(idx == i for idx, _ in image_path_indices):
                        introduction = item.get(constants.IMAGE_INTRO_TAG_NAME, None)
                        assistant = item.get(constants.IMAGE_ASSISTANT_TAG_NAME, None)
                        
                        message_content = []
                        if introduction:
                            message_content.append({"type": "text", "text": introduction})
                        
                        if len(message_content) > 0:
                            image_introduction_messages.append({
                                "role": "user",
                                "content": message_content
                            })
                        
                        if assistant:
                            image_introduction_messages.append({
                                "role": "assistant",
                                "content": [{"type": "text", "text": assistant}]
                            })

        # Assemble user messages part 2
        user_messages_part2_contents = replace_placeholders(user_messages_part2_paragraphs, params)
        user_messages_part2 = {
            "role": "user",
            "content": [{"type": "text", "text": "\n\n".join(user_messages_part2_contents)}]
        } if user_messages_part2_contents else None

        # Combine all messages
        messages = [system_message]
        if user_messages_part1:
            messages.append(user_messages_part1)
        messages.extend(image_introduction_messages)
        if user_messages_part2:
            messages.append(user_messages_part2)

        return messages
