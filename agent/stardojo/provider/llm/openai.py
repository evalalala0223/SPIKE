from __future__ import annotations

import os
import hashlib
from collections.abc import Iterable
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
import asyncio
import threading
import time
from functools import lru_cache
from types import SimpleNamespace

import backoff
import httpx
import tiktoken
import numpy as np
from openai import OpenAI, AzureOpenAI, APIError, RateLimitError, APITimeoutError

from stardojo import constants
from stardojo.provider.base import LLMProvider, EmbeddingProvider
from stardojo.config import Config
from stardojo.log import Logger
from stardojo.utils.json_utils import load_json
from stardojo.utils.encoding_utils import encode_data_to_base64_path
from stardojo.utils.file_utils import assemble_project_path
from stardojo.utils.llm_timing_utils import add_llm_retry_overhead
try:
    from cradle.utils.llm_endpoint_throttle import (
        LLMEndpointThrottleTimeout,
        acquire_llm_endpoint_slot,
        get_llm_endpoint_wait_timeout,
        resolve_remaining_llm_request_timeout,
    )
    from cradle.utils.llm_call_budget import increment_llm_call_counter
except ModuleNotFoundError:
    # cradle 包位于 agent/ 目录下，但 sys.path 可能尚未包含该目录
    import sys as _sys
    _agent_root = os.path.join(os.path.dirname(__file__), "..", "..", "..")
    _agent_root = os.path.normpath(_agent_root)
    if _agent_root not in _sys.path:
        _sys.path.insert(0, _agent_root)
    from cradle.utils.llm_endpoint_throttle import (
        LLMEndpointThrottleTimeout,
        acquire_llm_endpoint_slot,
        get_llm_endpoint_wait_timeout,
        resolve_remaining_llm_request_timeout,
    )
    from cradle.utils.llm_call_budget import increment_llm_call_counter

config = Config()
logger = Logger()

MAX_TOKENS = {
    "gpt-3.5-turbo-0301": 4097,
    "gpt-3.5-turbo-0613": 4097,
    "gpt-3.5-turbo-16k-0613": 16385,
}

PROVIDER_SETTING_KEY_VAR = "key_var"
PROVIDER_SETTING_EMB_MODEL = "emb_model"
PROVIDER_SETTING_COMP_MODEL = "comp_model"
PROVIDER_SETTING_IS_AZURE = "is_azure"
PROVIDER_SETTING_API_KEY = "api_key"
PROVIDER_SETTING_EMB_API_KEY = "emb_api_key"
PROVIDER_SETTING_BASE_URL = "base_url"
PROVIDER_SETTING_EMB_BASE_URL = "emb_base_url"
PROVIDER_SETTING_EMB_FALLBACK_DIM = "emb_fallback_dim"
PROVIDER_SETTING_EMB_KEY_VAR = "emb_key_var"
PROVIDER_SETTING_BASE_VAR = "base_var"       # Azure-speficic setting
PROVIDER_SETTING_API_VERSION = "api_version" # Azure-speficic setting
PROVIDER_SETTING_DEPLOYMENT_MAP = "models"   # Azure-speficic setting

_STAGE_TIMEOUT_KEYS = {
    "llm_description": "llm_description_timeout_seconds",
    "task_inference": "task_inference_timeout_seconds",
    "action_planning": "action_planning_timeout_seconds",
    "self_reflection": "self_reflection_timeout_seconds",
}

_STAGE_RETRY_KEYS = {
    "task_inference": "task_inference_request_max_retries",
}


class OpenAIProvider(LLMProvider, EmbeddingProvider):
    """A class that wraps a given model"""

    _proxy_env_logged: bool = False
    _client_runtime_logged: bool = False

    client: Any = None
    llm_model: str = ""
    embedding_model: str = ""
    fallback_embedding_dim: int = 1024
    _local_st_model: Any = None
    _local_st_dim: int = 0

    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    chunk_size: int = 1000
    embedding_ctx_length: int = 8191
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    tiktoken_model_name: Optional[str] = "cl100k_base"

    """Whether to skip empty strings when embedding or raise an error."""
    skip_empty: bool = False


    def __init__(self, is_opensource = False) -> None:
        """Initialize a class instance
        is_opensource means whether it is opensource vlm (but use openai api).
        Args:
            cfg: Config object

        Returns:
            None
        """
        self.retries = int(os.getenv("OPENAI_CHAT_MAX_RETRIES", "10"))
        self.retry_interval_seconds = float(os.getenv("OPENAI_CHAT_RETRY_INTERVAL_SECONDS", "3"))
        self.is_opensource = is_opensource
        self.secondary_client = None
        self._rr_lock = threading.Lock()
        self._rr_counter = 0
        self.embedding_api_key = None
        self.embedding_base_url = None
        self.fallback_embedding_dim = 1024

    @staticmethod
    @lru_cache(maxsize=1)
    def _load_llm_runtime_config() -> Dict[str, Any]:
        config_path = os.getenv("STARDOJO_ENHANCED_CONFIG", "").strip()
        config_path = assemble_project_path(config_path) if config_path else assemble_project_path("./conf/enhanced_config.yaml")
        try:
            if os.path.exists(config_path):
                import yaml

                with open(config_path, "r", encoding="utf-8") as f:
                    raw = yaml.safe_load(f) or {}
                performance = raw.get("performance", {}) or {}
                streaming = performance.get("streaming", {}) or {}
                stage_timeout_seconds: Dict[str, float] = {}
                stage_max_retries: Dict[str, int] = {}
                for stage_name, timeout_key in _STAGE_TIMEOUT_KEYS.items():
                    timeout_value = streaming.get(timeout_key)
                    if timeout_value is None:
                        continue
                    try:
                        stage_timeout_seconds[stage_name] = float(timeout_value)
                    except (TypeError, ValueError):
                        logger.warn(
                            f"Invalid timeout value for {timeout_key}: {timeout_value}"
                        )
                for stage_name, retry_key in _STAGE_RETRY_KEYS.items():
                    retry_value = streaming.get(retry_key)
                    if retry_value is None:
                        continue
                    try:
                        stage_max_retries[stage_name] = int(retry_value)
                    except (TypeError, ValueError):
                        logger.warn(
                            f"Invalid retry value for {retry_key}: {retry_value}"
                        )
                return {
                    "default_timeout_seconds": float(
                        streaming.get("llm_default_timeout_seconds", 90)
                    ),
                    "max_retries": int(
                        streaming.get("llm_request_max_retries", 10)
                    ),
                    "retry_interval_seconds": float(
                        streaming.get("llm_request_retry_interval_seconds", 3)
                    ),
                    "stage_timeout_seconds": stage_timeout_seconds,
                    "stage_max_retries": stage_max_retries,
                }
        except Exception as e:
            logger.warn(f"Failed to load LLM runtime config: {e}")

        return {
            "default_timeout_seconds": 90.0,
            "max_retries": 10,
            "retry_interval_seconds": 3.0,
            "stage_timeout_seconds": {},
            "stage_max_retries": {},
        }

    def _resolve_request_controls(
        self,
        request_timeout_s: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_interval_s: Optional[float] = None,
        stage: Optional[str] = None,
    ) -> Tuple[float, int, float]:
        runtime_cfg = self._load_llm_runtime_config()
        stage_timeout_s = None
        stage_retry_count = None
        if request_timeout_s is None and stage:
            stage_timeout_s = (
                (runtime_cfg.get("stage_timeout_seconds", {}) or {}).get(stage)
            )
        if max_retries is None and stage:
            stage_retry_count = (
                (runtime_cfg.get("stage_max_retries", {}) or {}).get(stage)
            )
        timeout_s = float(
            request_timeout_s
            if request_timeout_s is not None
            else (
                stage_timeout_s
                if stage_timeout_s is not None
                else runtime_cfg.get("default_timeout_seconds", 90.0)
            )
        )
        retries = int(
            max_retries
            if max_retries is not None
            else (
                stage_retry_count
                if stage_retry_count is not None
                else runtime_cfg.get("max_retries", self.retries)
            )
        )
        interval_s = float(
            retry_interval_s
            if retry_interval_s is not None
            else runtime_cfg.get("retry_interval_seconds", self.retry_interval_seconds)
        )
        return max(1.0, timeout_s), max(1, retries), max(0.0, interval_s)

    @staticmethod
    def _is_retryable_api_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            response = getattr(exc, "response", None)
            status_code = getattr(response, "status_code", None)
        try:
            if int(status_code) == 503:
                return True
        except (TypeError, ValueError):
            pass

        lowered = str(exc or "").lower()
        return "503" in lowered and "service" in lowered and "unavailable" in lowered

    def _build_request_purpose(self, stage: Optional[str]) -> str:
        normalized_stage = re.sub(
            r"[^a-z0-9_:-]+",
            "_",
            str(stage or "chat_completion").strip().lower(),
        )
        return f"stardojo_{normalized_stage or 'chat_completion'}"

    @staticmethod
    def _build_qwen_disable_thinking_extra_body(model: Any) -> Optional[Dict[str, Any]]:
        model_text = str(model or "").strip().lower()
        if "qwen" not in model_text:
            return None
        return {"chat_template_kwargs": {"enable_thinking": False}}

    def _get_next_client(self):
        """Round-robin client selection for load distribution across API keys.

        Uses 5:1 ratio favoring primary key (higher quota).
        Secondary key (1/5 quota) gets every 6th request.
        """
        if self.secondary_client is None:
            return self.client
        with self._rr_lock:
            self._rr_counter += 1
            if self._rr_counter % 6 == 0:
                return self.secondary_client
            return self.client

    @staticmethod
    def _flatten_prompt_image_paths(raw_value: Any) -> List[str]:
        paths: List[str] = []

        def _append(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, os.PathLike):
                value = os.fspath(value)
            if isinstance(value, str):
                normalized = value.strip()
                if normalized:
                    paths.append(normalized)
                return
            if isinstance(value, Iterable) and not isinstance(value, (dict, bytes)):
                for item in value:
                    _append(item)

        _append(raw_value)
        return paths

    def _extract_prompt_image_entries(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        image_entries: List[Dict[str, Any]] = []
        paragraph_input = params.get(constants.IMAGES_INPUT_TAG_NAME, []) or []

        if isinstance(paragraph_input, dict):
            paragraph_items = [paragraph_input]
        elif isinstance(paragraph_input, Iterable) and not isinstance(paragraph_input, (str, bytes)):
            paragraph_items = list(paragraph_input)
        else:
            paragraph_items = [paragraph_input]

        for item in paragraph_items:
            if not isinstance(item, dict):
                for path in self._flatten_prompt_image_paths(item):
                    image_entries.append({"path": path})
                continue

            entry_template = {
                "introduction": item.get(constants.IMAGE_INTRO_TAG_NAME, ""),
                "assistant": item.get(constants.IMAGE_ASSISTANT_TAG_NAME, ""),
                "resolution": item.get(constants.IMAGE_RESOLUTION_TAG_NAME, ""),
                "resize": item.get(constants.IMAGE_RESIZE_TAG_NAME, ""),
            }
            for path in self._flatten_prompt_image_paths(item.get(constants.IMAGE_PATH_TAG_NAME)):
                image_entries.append({**entry_template, "path": path})

        if image_entries:
            return image_entries

        for path in self._flatten_prompt_image_paths(params.get("image_paths", [])):
            image_entries.append({"path": path})

        return image_entries


    def init_provider(self, provider_cfg, embedding_only: bool = False) -> None:
        self.provider_cfg = self._parse_config(
            provider_cfg,
            embedding_only=embedding_only,
        )


    def _sanitize_env_value(self, value: Optional[str]) -> str:
        if not value:
            return "<empty>"
        sanitized = re.sub(r"://[^@]+@", "://***@", value)
        if len(sanitized) > 120:
            sanitized = sanitized[:117] + "..."
        return sanitized


    def _log_proxy_env_once(self) -> None:
        """Intentionally disabled – proxy env vars are not relevant at runtime."""
        OpenAIProvider._proxy_env_logged = True


    def _log_client_runtime_once(self) -> None:
        if OpenAIProvider._client_runtime_logged:
            return
        client_retries = getattr(self.client, "max_retries", "unknown")
        emb_retries = getattr(self.embedding_client, "max_retries", "unknown") if self.embedding_client is not None else "unknown"
        logger.write(
            "OpenAI runtime config: "
            f"sdk_max_retries(chat)={client_retries}, "
            f"sdk_max_retries(embedding)={emb_retries}, "
            "trust_env=False"
        )
        OpenAIProvider._client_runtime_logged = True


    def _build_http_client(self, timeout: Optional[float] = None) -> httpx.Client:
        client_kwargs: Dict[str, Any] = {"trust_env": False}
        if timeout is not None:
            client_kwargs["timeout"] = timeout
        return httpx.Client(**client_kwargs)


    def _resolve_api_key(self, conf_dict: Dict[str, Any], api_key_field: str, key_var_field: str, fallback_env_var: Optional[str] = None) -> Optional[str]:
        direct_key = conf_dict.get(api_key_field)
        if isinstance(direct_key, str) and direct_key.strip():
            return direct_key.strip()

        key_var_name = conf_dict.get(key_var_field)
        if isinstance(key_var_name, str) and key_var_name.strip():
            key_var_name = key_var_name.strip()
            env_key = os.getenv(key_var_name)
            if env_key:
                return env_key
            if key_var_name.startswith("sk-"):
                return key_var_name

        if fallback_env_var:
            fallback_key = os.getenv(fallback_env_var)
            if fallback_key:
                return fallback_key

        return None


    def _parse_config(self, provider_cfg, embedding_only: bool = False) -> dict:
        """Parse the config object"""

        conf_dict = dict()

        if isinstance(provider_cfg, dict):
            conf_dict = provider_cfg
        else:
            path = assemble_project_path(provider_cfg)
            conf_dict = load_json(path)

        key_var_name = conf_dict[PROVIDER_SETTING_KEY_VAR]
        self.client = None
        self.secondary_client = None
        self.embedding_client = None
        self.embedding_model = conf_dict[PROVIDER_SETTING_EMB_MODEL]
        self.llm_model = conf_dict[PROVIDER_SETTING_COMP_MODEL]
        self.fallback_embedding_dim = int(
            conf_dict.get(PROVIDER_SETTING_EMB_FALLBACK_DIM, 1024)
        )
        self._log_proxy_env_once()

        if embedding_only and self._uses_local_sentence_transformers():
            self.embedding_api_key = None
            self.embedding_base_url = None
            logger.write(
                f"[Embedding] Using local sentence-transformers model "
                f"for embedding-only provider: {self.embedding_model}"
            )
            return conf_dict

        if conf_dict[PROVIDER_SETTING_IS_AZURE]:

            key = os.getenv(key_var_name)
            endpoint_var_name = conf_dict[PROVIDER_SETTING_BASE_VAR]
            endpoint = os.getenv(endpoint_var_name)

            self.client = AzureOpenAI(
                api_key = key,
                api_version = conf_dict[PROVIDER_SETTING_API_VERSION],
                azure_endpoint = endpoint,
                max_retries = 0,
                http_client = self._build_http_client(timeout=120.0),
            )
            self.embedding_client = self.client
        else:
            key = self._resolve_api_key(
                conf_dict,
                PROVIDER_SETTING_API_KEY,
                PROVIDER_SETTING_KEY_VAR,
                fallback_env_var="OPEN_SRC_KEY" if self.is_opensource else None,
            )
            base_url = conf_dict.get(PROVIDER_SETTING_BASE_URL)
            client_kwargs = {
                "api_key": key,
                "timeout": 120.0,
                "max_retries": 0,
                "http_client": self._build_http_client(timeout=120.0),
            }
            default_headers = conf_dict.get("default_headers")
            if isinstance(default_headers, dict) and default_headers:
                client_kwargs["default_headers"] = {
                    str(header_key): str(header_value)
                    for header_key, header_value in default_headers.items()
                    if str(header_key).strip() and str(header_value).strip()
                }
            if base_url:
                client_kwargs["base_url"] = base_url
            self.client = OpenAI(**client_kwargs)

            # Secondary API key for load distribution (round-robin)
            secondary_key = conf_dict.get("secondary_api_key")
            if secondary_key and isinstance(secondary_key, str) and secondary_key.strip():
                sec_kwargs = {
                    "api_key": secondary_key.strip(),
                    "timeout": 120.0,
                    "max_retries": 0,
                    "http_client": self._build_http_client(timeout=120.0),
                }
                if isinstance(default_headers, dict) and default_headers:
                    sec_kwargs["default_headers"] = {
                        str(header_key): str(header_value)
                        for header_key, header_value in default_headers.items()
                        if str(header_key).strip() and str(header_value).strip()
                    }
                if base_url:
                    sec_kwargs["base_url"] = base_url
                self.secondary_client = OpenAI(**sec_kwargs)
                logger.write("[OpenAI] Secondary API key configured for load distribution (5:1 round-robin, secondary ~1/5 quota)")

            emb_key = self._resolve_api_key(
                conf_dict,
                PROVIDER_SETTING_EMB_API_KEY,
                PROVIDER_SETTING_EMB_KEY_VAR,
            ) or key
            emb_base_url = conf_dict.get(PROVIDER_SETTING_EMB_BASE_URL) or base_url
            self.embedding_api_key = emb_key
            self.embedding_base_url = emb_base_url

            emb_client_kwargs = {
                "api_key": emb_key,
                "timeout": 120.0,
                "max_retries": 0,
                "http_client": self._build_http_client(timeout=120.0),
            }
            emb_default_headers = conf_dict.get("emb_default_headers", default_headers)
            if isinstance(emb_default_headers, dict) and emb_default_headers:
                emb_client_kwargs["default_headers"] = {
                    str(header_key): str(header_value)
                    for header_key, header_value in emb_default_headers.items()
                    if str(header_key).strip() and str(header_value).strip()
                }
            if emb_base_url:
                emb_client_kwargs["base_url"] = emb_base_url

            if emb_key != key or emb_base_url != base_url:
                self.embedding_client = OpenAI(**emb_client_kwargs)
            else:
                self.embedding_client = self.client

        self._log_client_runtime_once()

        return conf_dict

    @property
    def _emb_invocation_params(self) -> Dict:

        openai_args = {
            "model": self.embedding_model,
        }

        if self.provider_cfg[PROVIDER_SETTING_IS_AZURE]:
            engine = self._get_azure_deployment_id_for_model(self.embedding_model)
            openai_args = {
                "model": self.embedding_model,
            }

        return openai_args

    def _uses_dashscope_multimodal_embeddings(self) -> bool:
        return str(self.embedding_model).lower() == "multimodal-embedding-v1"

    def _uses_local_sentence_transformers(self) -> bool:
        model = str(self.embedding_model or "").lower()
        return model.startswith("bge-") or model.startswith("sentence-transformers/") or "/bge-" in model or "baai/" in model

    def _get_local_st_model(self):
        if self._local_st_model is not None:
            return self._local_st_model
        try:
            import os
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
            from sentence_transformers import SentenceTransformer
            model_name = self.embedding_model
            logger.write(f"[Embedding] Loading local sentence-transformers model: {model_name}")
            self._local_st_model = SentenceTransformer(model_name, local_files_only=True, device="cpu")
            self._local_st_dim = self._local_st_model.get_sentence_embedding_dimension()
            logger.write(f"[Embedding] Local model loaded (dim={self._local_st_dim})")
            return self._local_st_model
        except Exception as e:
            logger.error(f"[Embedding] Failed to load local model '{self.embedding_model}': {e}")
            return None

    def _get_dashscope_native_base_url(self) -> str:
        base_url = str(self.embedding_base_url or "").strip()
        if not base_url:
            return "https://dashscope.aliyuncs.com/api/v1"
        lowered = base_url.lower()
        if "compatible-mode" in lowered or "coding.dashscope" in lowered:
            return "https://dashscope.aliyuncs.com/api/v1"
        return base_url

    @staticmethod
    def _build_embedding_response(embeddings: List[List[float]]) -> Any:
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=embedding) for embedding in embeddings]
        )

    @staticmethod
    def _local_hash_embedding(text: str, dim: int) -> List[float]:
        text = text or ""
        vec = np.zeros(dim, dtype=np.float32)
        tokens = text.split()

        if not tokens:
            vec[0] = 1.0
            return vec.tolist()

        for token in tokens:
            digest = hashlib.sha1(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dim
            sign = 1.0 if (digest[4] % 2 == 0) else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm <= 0:
            vec[0] = 1.0
            norm = 1.0
        return (vec / norm).tolist()

    @classmethod
    def _local_hash_embeddings(cls, input_payload: Any, dim: int) -> List[List[float]]:
        if isinstance(input_payload, str):
            texts = [input_payload]
        elif isinstance(input_payload, list):
            texts = [str(item) for item in input_payload]
        else:
            texts = [str(input_payload)]
        return [cls._local_hash_embedding(text, dim) for text in texts]

    @staticmethod
    def _should_use_local_embedding_fallback(status_code: Any, message: str) -> bool:
        lowered = str(message or "").lower()
        if any(
            token in lowered
            for token in (
                "access denied",
                "arrearage",
                "overdue-payment",
                "account is in good standing",
                "invalid api-key",
                "authentication",
                "unauthorized",
            )
        ):
            return True
        try:
            code = int(status_code)
        except Exception:
            code = None
        return code in {401, 402, 403}

    @staticmethod
    def _split_text_for_multimodal_embedding(text: str, max_words: int = 60, max_chars: int = 300) -> List[Tuple[str, int]]:
        text = str(text or "").strip()
        if not text:
            return [(" ", 1)]

        words = text.split()
        if len(words) > 1:
            return [
                (" ".join(words[i : i + max_words]), len(words[i : i + max_words]))
                for i in range(0, len(words), max_words)
            ]

        if len(text) <= max_chars:
            return [(text, max(len(text), 1))]

        return [
            (text[i : i + max_chars], min(max_chars, len(text) - i))
            for i in range(0, len(text), max_chars)
        ]

    def embed_with_retry(self, **kwargs: Any) -> Any:
        """Use backoff to retry the embedding call."""

        @backoff.on_exception(
            backoff.expo,
            Exception,
            max_tries=self.retries,
            max_value=10,
            jitter=None,
        )
        def _dashscope_multimodal_embed_with_retry(**kwargs: Any) -> Any:
            import dashscope

            input_payload = kwargs.get("input", "")
            if isinstance(input_payload, list):
                texts = [str(item or " ").strip() or " " for item in input_payload]
            else:
                texts = [str(input_payload or " ").strip() or " "]

            dashscope.api_key = self.embedding_api_key
            dashscope.base_http_api_url = self._get_dashscope_native_base_url()

            response = dashscope.MultiModalEmbedding.call(
                model=self.embedding_model,
                input=[{"text": text, "factor": 1.0} for text in texts],
                api_key=self.embedding_api_key,
                auto_truncation=True,
            )
            status_code = getattr(response, "status_code", None)
            if status_code is None or int(status_code) != 200:
                message = getattr(response, "message", "")
                if self._should_use_local_embedding_fallback(status_code, message):
                    logger.warn(
                        f"[Embedding] DashScope multimodal embedding unavailable "
                        f"(status={status_code}, message={message}). "
                        f"Using local hash embedding fallback (dim={self.fallback_embedding_dim})."
                    )
                    return self._build_embedding_response(
                        self._local_hash_embeddings(texts, self.fallback_embedding_dim)
                    )
                raise RuntimeError(
                    f"DashScope multimodal embedding failed: "
                    f"status={status_code}, message={message}"
                )

            output = getattr(response, "output", {}) or {}
            data = output.get("embeddings", []) if isinstance(output, dict) else []
            if len(data) != len(texts):
                raise RuntimeError(
                    f"DashScope multimodal embedding returned {len(data)} vectors for {len(texts)} texts"
                )

            embeddings = [item.get("embedding", []) for item in data]
            if any(not embedding for embedding in embeddings):
                raise RuntimeError("DashScope multimodal embedding returned an empty embedding")
            return self._build_embedding_response(embeddings)

        if self._uses_local_sentence_transformers():
            model = self._get_local_st_model()
            if model is not None:
                input_payload = kwargs.get("input", "")
                if isinstance(input_payload, str):
                    texts = [input_payload.strip() or " "]
                elif isinstance(input_payload, list):
                    texts = [str(item).strip() or " " for item in input_payload]
                else:
                    texts = [str(input_payload).strip() or " "]
                embeddings = model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()
                return self._build_embedding_response(embeddings)
            logger.warn("[Embedding] Local model unavailable, falling back to hash embedding")
            input_payload = kwargs.get("input", "")
            if isinstance(input_payload, str):
                texts = [input_payload]
            elif isinstance(input_payload, list):
                texts = [str(item) for item in input_payload]
            else:
                texts = [str(input_payload)]
            return self._build_embedding_response(
                self._local_hash_embeddings(texts, self.fallback_embedding_dim)
            )

        if self._uses_dashscope_multimodal_embeddings():
            return _dashscope_multimodal_embed_with_retry(**kwargs)

        @backoff.on_exception(
            backoff.expo,
            (
                APIError,
                RateLimitError,
                APITimeoutError,
            ),
            max_tries=self.retries,
            max_value=10,
            jitter=None,
        )
        def _embed_with_retry(**kwargs: Any) -> Any:
            embed_client = self.embedding_client.with_options(max_retries=0)
            response = embed_client.embeddings.create(**kwargs)
            if any(len(d.embedding) == 1 for d in response.data):
                raise RuntimeError("OpenAI API returned an empty embedding")
            return response

        return _embed_with_retry(**kwargs)


    def _get_len_safe_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
        if self._uses_local_sentence_transformers():
            model = self._get_local_st_model()
            if model is not None:
                results = model.encode(
                    [t.strip() or " " for t in texts],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()
                return results
            return [self._local_hash_embedding(t, self.fallback_embedding_dim) for t in texts]
        if self._uses_dashscope_multimodal_embeddings():
            chunks: List[str] = []
            indices: List[int] = []
            chunk_weights: List[int] = []

            for i, text in enumerate(texts):
                for chunk_text, weight in self._split_text_for_multimodal_embedding(text):
                    chunks.append(chunk_text)
                    indices.append(i)
                    chunk_weights.append(weight)

            batched_embeddings: List[List[float]] = []
            for i in range(0, len(chunks), self.chunk_size):
                response = self.embed_with_retry(
                    input=chunks[i : i + self.chunk_size],
                    **self._emb_invocation_params,
                )
                batched_embeddings.extend(r.embedding for r in response.data)

            results: List[List[List[float]]] = [[] for _ in range(len(texts))]
            num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
            for i, text_index in enumerate(indices):
                results[text_index].append(batched_embeddings[i])
                num_tokens_in_batch[text_index].append(chunk_weights[i])

            for i in range(len(texts)):
                _result = results[i]
                if len(_result) == 0:
                    average = self.embed_with_retry(
                        input=" ",
                        **self._emb_invocation_params,
                    ).data[0].embedding
                else:
                    average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
                norm = np.linalg.norm(average)
                embeddings[i] = (average / norm if norm > 0 else average).tolist()

            return embeddings

        try:
            import tiktoken
        except ImportError:
            raise ImportError(
                "Could not import tiktoken python package. "
                "This is needed in order to for OpenAIEmbeddings. "
                "Please install it with `pip install tiktoken`."
            )

        tokens = []
        indices = []
        model_name = self.tiktoken_model_name or self.embedding_model
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            logger.debug("Model not found in tiktoken. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = tiktoken.get_encoding(model)
        for i, text in enumerate(texts):
            token = encoding.encode(
                text,
                allowed_special=self.allowed_special,
                disallowed_special=self.disallowed_special,
            )
            for j in range(0, len(token), self.embedding_ctx_length):
                tokens.append(token[j : j + self.embedding_ctx_length])
                indices.append(i)

        batched_embeddings: List[List[float]] = []
        _chunk_size = self.chunk_size
        _iter = range(0, len(tokens), _chunk_size)

        for i in _iter:
            response = self.embed_with_retry(
                input=[encoding.decode(t) for t in tokens[i : i + self.chunk_size]],
                **self._emb_invocation_params,
            )
            batched_embeddings.extend(r.embedding for r in response.data)

        results: List[List[List[float]]] = [[] for _ in range(len(texts))]
        num_tokens_in_batch: List[List[int]] = [[] for _ in range(len(texts))]
        for i in range(len(indices)):
            if self.skip_empty and len(batched_embeddings[i]) == 1:
                continue
            results[indices[i]].append(batched_embeddings[i])
            num_tokens_in_batch[indices[i]].append(len(tokens[i]))

        for i in range(len(texts)):
            _result = results[i]
            if len(_result) == 0:
                average = self.embed_with_retry(
                    input="",
                    **self._emb_invocation_params,
                ).data[0].embedding
            else:
                average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
            embeddings[i] = (average / np.linalg.norm(average)).tolist()

        return embeddings

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        """Call out to 's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        return self._get_len_safe_embeddings(texts)


    def embed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]


    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        if self._uses_local_sentence_transformers():
            if self._local_st_dim > 0:
                return self._local_st_dim
            model = self._get_local_st_model()
            if model is not None:
                return self._local_st_dim
            return self.fallback_embedding_dim
        if self.embedding_model == "text-embedding-ada-002":
            embedding_dim = 1536
        elif PROVIDER_SETTING_EMB_FALLBACK_DIM in self.provider_cfg:
            embedding_dim = int(self.provider_cfg[PROVIDER_SETTING_EMB_FALLBACK_DIM])
        else:
            raise ValueError(f"Unknown embedding model: {self.embedding_model}")
        return embedding_dim


    def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = config.temperature,
        seed: int = config.seed,
        max_tokens: int = config.max_tokens,
        request_timeout_s: Optional[float] = None,
        max_retries: Optional[int] = None,
        retry_interval_s: Optional[float] = None,
        stage: Optional[str] = None,
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

        request_timeout_s, effective_retries, retry_interval_s = (
            self._resolve_request_controls(
                request_timeout_s=request_timeout_s,
                max_retries=max_retries,
                retry_interval_s=retry_interval_s,
                stage=stage,
            )
        )
        request_purpose = self._build_request_purpose(stage)

        if config.debug_mode:
            logger.debug(
                f"Creating chat completion with model {model}, "
                f"temperature {temperature}, max_tokens {max_tokens}, "
                f"timeout {request_timeout_s}s, retries {effective_retries}, "
                f"stage={stage or 'default'}"
            )
        else:
            logger.write(
                f"Requesting {model} completion "
                f"(stage={stage or 'default'} timeout={request_timeout_s:.0f}s "
                f"retries={effective_retries})..."
            )

        def _generate_response_with_retry(
            messages: List[Dict[str, str]],
            model: str,
            temperature: float,
            seed: int = None,
            max_tokens: int = 512,
        ) -> Tuple[str, Dict[str, int]]:

            """Send a request to the OpenAI API."""
            overall_started_at = time.time()
            last_attempt_duration_s = 0.0
            attempt = 0

            while True:
                attempt += 1
                attempt_started_at = time.time()
                try:
                    slot_wait_timeout_s = get_llm_endpoint_wait_timeout(
                        model,
                        total_timeout_s=request_timeout_s,
                    )
                    with acquire_llm_endpoint_slot(
                        model_name=model,
                        purpose=request_purpose,
                        logger_obj=logger,
                        timeout_s=slot_wait_timeout_s,
                    ) as slot_info:
                        effective_request_timeout_s = resolve_remaining_llm_request_timeout(
                            request_timeout_s,
                            slot_info,
                        )
                        increment_llm_call_counter(f"big_brain:{stage or 'default'}")
                        request_client = self._get_next_client().with_options(
                            max_retries=0,
                            timeout=effective_request_timeout_s,
                        )
                        if self.provider_cfg[PROVIDER_SETTING_IS_AZURE]:
                            response = request_client.chat.completions.create(
                                model=model,
                                messages=messages,
                                temperature=temperature,
                                seed=seed,
                                max_tokens=max_tokens,
                            )
                        elif model.startswith("o3"):
                            response = request_client.chat.completions.create(
                                model=model,
                                messages=messages,
                                temperature=temperature,
                                seed=seed,
                                max_completion_tokens=max_tokens,
                            )
                        else:
                            request_kwargs = {
                                "model": model,
                                "messages": messages,
                                "temperature": temperature,
                                "seed": seed,
                                "max_tokens": max_tokens,
                            }
                            extra_body = self._build_qwen_disable_thinking_extra_body(model)
                            if extra_body is not None:
                                request_kwargs["extra_body"] = extra_body
                            response = request_client.chat.completions.create(**request_kwargs)

                    last_attempt_duration_s = max(0.0, time.time() - attempt_started_at)
                    if response is None:
                        logger.error("Failed to get a response from OpenAI. Try again.")
                        logger.double_check()

                    message = response.choices[0].message.content
                    retry_overhead_s = max(
                        0.0,
                        (time.time() - overall_started_at) - last_attempt_duration_s,
                    )
                    if retry_overhead_s > 0:
                        add_llm_retry_overhead(retry_overhead_s)

                    info = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                        "system_fingerprint": response.system_fingerprint,
                        "effective_duration_s": last_attempt_duration_s,
                        "retry_overhead_s": retry_overhead_s,
                        "retry_attempts": max(0, attempt - 1),
                    }

                    logger.write(f'Response received from {model}.')
                    return message, info
                except (RateLimitError, APITimeoutError) as exc:
                    last_attempt_duration_s = max(0.0, time.time() - attempt_started_at)
                    if attempt >= effective_retries:
                        retry_overhead_s = max(
                            0.0,
                            (time.time() - overall_started_at) - last_attempt_duration_s,
                        )
                        if retry_overhead_s > 0:
                            add_llm_retry_overhead(retry_overhead_s)
                        raise
                    logger.warn(
                        f"[OpenAI] Retryable {type(exc).__name__} on attempt {attempt}/{effective_retries} "
                        f"for stage={stage or 'default'}; retrying in {retry_interval_s:.1f}s"
                    )
                    time.sleep(retry_interval_s)
                except APIError as exc:
                    last_attempt_duration_s = max(0.0, time.time() - attempt_started_at)
                    if attempt >= effective_retries or not self._is_retryable_api_error(exc):
                        retry_overhead_s = max(
                            0.0,
                            (time.time() - overall_started_at) - last_attempt_duration_s,
                        )
                        if retry_overhead_s > 0:
                            add_llm_retry_overhead(retry_overhead_s)
                        raise
                    logger.warn(
                        f"[OpenAI] Retryable 503 APIError on attempt {attempt}/{effective_retries} "
                        f"for stage={stage or 'default'}; retrying in {retry_interval_s:.1f}s"
                    )
                    time.sleep(retry_interval_s)

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
            request_timeout_s: Optional[float] = None,
            max_retries: Optional[int] = None,
            retry_interval_s: Optional[float] = None,
            stage: Optional[str] = None,
    ) -> Tuple[str, Dict[str, int]]:

        if model is None:
            model = self.llm_model

        request_timeout_s, effective_retries, retry_interval_s = (
            self._resolve_request_controls(
                request_timeout_s=request_timeout_s,
                max_retries=max_retries,
                retry_interval_s=retry_interval_s,
                stage=stage,
            )
        )
        request_purpose = self._build_request_purpose(stage)

        if config.debug_mode:
            logger.debug(
                f"Creating chat completion with model {model}, temperature {temperature}, max_tokens {max_tokens}, "
                f"timeout {request_timeout_s}s, retries {effective_retries}, "
                f"stage={stage or 'default'}")
        else:
            logger.write(
                f"Requesting {model} completion "
                f"(stage={stage or 'default'} timeout={request_timeout_s:.0f}s "
                f"retries={effective_retries})..."
            )

        async def _generate_response_with_retry_async(
                messages: List[Dict[str, str]],
                model: str,
                temperature: float,
                seed: int = None,
                max_tokens: int = 512,
        ) -> Tuple[str, Dict[str, int]]:

            """Send a request to the OpenAI API."""
            overall_started_at = time.time()
            last_attempt_duration_s = 0.0
            attempt = 0

            while True:
                attempt += 1
                attempt_started_at = time.time()
                try:
                    slot_wait_timeout_s = get_llm_endpoint_wait_timeout(
                        model,
                        total_timeout_s=request_timeout_s,
                    )
                    with acquire_llm_endpoint_slot(
                        model_name=model,
                        purpose=request_purpose,
                        logger_obj=logger,
                        timeout_s=slot_wait_timeout_s,
                    ) as slot_info:
                        effective_request_timeout_s = resolve_remaining_llm_request_timeout(
                            request_timeout_s,
                            slot_info,
                        )
                        increment_llm_call_counter(f"big_brain:{stage or 'default'}_async")
                        request_client = self._get_next_client().with_options(
                            max_retries=0,
                            timeout=effective_request_timeout_s,
                        )
                        if self.provider_cfg[PROVIDER_SETTING_IS_AZURE]:
                            response = await asyncio.to_thread(
                                request_client.chat.completions.create,
                                model=model,
                                messages=messages,
                                temperature=temperature,
                                seed=seed,
                                max_tokens=max_tokens,
                            )
                        elif model.startswith("o3"):
                            response = await asyncio.to_thread(
                                request_client.chat.completions.create,
                                model=model,
                                messages=messages,
                                temperature=temperature,
                                seed=seed,
                                max_completion_tokens=max_tokens,
                            )
                        else:
                            request_kwargs = {
                                "model": model,
                                "messages": messages,
                                "temperature": temperature,
                                "seed": seed,
                                "max_tokens": max_tokens,
                            }
                            extra_body = self._build_qwen_disable_thinking_extra_body(model)
                            if extra_body is not None:
                                request_kwargs["extra_body"] = extra_body
                            response = await asyncio.to_thread(
                                request_client.chat.completions.create,
                                **request_kwargs,
                            )

                    last_attempt_duration_s = max(0.0, time.time() - attempt_started_at)
                    if response is None:
                        logger.error("Failed to get a response from OpenAI. Try again.")
                        logger.double_check()

                    message = response.choices[0].message.content
                    retry_overhead_s = max(
                        0.0,
                        (time.time() - overall_started_at) - last_attempt_duration_s,
                    )
                    if retry_overhead_s > 0:
                        add_llm_retry_overhead(retry_overhead_s)

                    info = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                        "system_fingerprint": response.system_fingerprint,
                        "effective_duration_s": last_attempt_duration_s,
                        "retry_overhead_s": retry_overhead_s,
                        "retry_attempts": max(0, attempt - 1),
                    }

                    logger.write(f'Response received from {model}.')
                    return message, info
                except (RateLimitError, APITimeoutError) as exc:
                    last_attempt_duration_s = max(0.0, time.time() - attempt_started_at)
                    if attempt >= effective_retries:
                        retry_overhead_s = max(
                            0.0,
                            (time.time() - overall_started_at) - last_attempt_duration_s,
                        )
                        if retry_overhead_s > 0:
                            add_llm_retry_overhead(retry_overhead_s)
                        raise
                    logger.warn(
                        f"[OpenAI] Retryable {type(exc).__name__} on attempt {attempt}/{effective_retries} "
                        f"for stage={stage or 'default'}; retrying in {retry_interval_s:.1f}s"
                    )
                    await asyncio.sleep(retry_interval_s)
                except APIError as exc:
                    last_attempt_duration_s = max(0.0, time.time() - attempt_started_at)
                    if attempt >= effective_retries or not self._is_retryable_api_error(exc):
                        retry_overhead_s = max(
                            0.0,
                            (time.time() - overall_started_at) - last_attempt_duration_s,
                        )
                        if retry_overhead_s > 0:
                            add_llm_retry_overhead(retry_overhead_s)
                        raise
                    logger.warn(
                        f"[OpenAI] Retryable 503 APIError on attempt {attempt}/{effective_retries} "
                        f"for stage={stage or 'default'}; retrying in {retry_interval_s:.1f}s"
                    )
                    await asyncio.sleep(retry_interval_s)

        return await _generate_response_with_retry_async(
            messages,
            model,
            temperature,
            seed,
            max_tokens,
        )


    def num_tokens_from_messages(self, messages, model):
        """Return the number of tokens used by a list of messages.
        Borrowed from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            logger.debug("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")
        if model in {
            "gpt-4-1106-vision-preview",
        }:
            raise ValueError("We don't support counting tokens of GPT-4V yet.")

        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
            "gpt-4-1106-preview",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name

        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        return num_tokens


    def _get_azure_deployment_id_for_model(self, model_label) -> list:
        return self.provider_cfg[PROVIDER_SETTING_DEPLOYMENT_MAP][model_label]


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

        if len(user_messages_part1_contents) > 0:

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

        else:
            user_messages_part1 = None

        # assemble image introduction messages
        image_introduction_messages = []

        image_entries = self._extract_prompt_image_entries(params)

        if not image_entries:
            image_introduction_messages = []
        else:
            paths = [entry.get("path", "") for entry in image_entries if entry.get("path")]
            logger.write("--------------------------------")
            logger.write(f"imagepaths: {paths}")
            logger.write("--------------------------------")
            for i, image_entry in enumerate(reversed(image_entries)):
                path = image_entry.get("path")
                if not path:
                    continue
                result = encode_data_to_base64_path(path)
                if isinstance(result, list):
                    encoded_image = result[0] if result else None
                else:
                    encoded_image = result
                if not encoded_image:
                    continue

                msg_text = image_entry.get(constants.IMAGE_INTRO_TAG_NAME) or (
                    "This is a screenshot of the current step of the game."
                    if i == 0
                    else f"This is the game screenshot from {i} steps ago"
                )
                combined_user_message["content"].append({
                    "type": "text",
                    "text": f"{msg_text}"
                })
                msg_content = {
                    "type": "image_url",
                    "image_url":
                        {
                            "url": f"{encoded_image}"
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
                    elif isinstance(paragraph_input, bool) or isinstance(paragraph_input, int) or isinstance(paragraph_input, float):
                        paragraph_content = paragraph.replace(placeholder, str(paragraph_input))
                        user_messages_part2_contents.append(paragraph_content)
                    elif isinstance(paragraph_input, list) or isinstance(paragraph_input, dict):
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

        # if user_messages_part1 is None:
        #     return [system_message] + image_introduction_messages + [combined_user_message]
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
