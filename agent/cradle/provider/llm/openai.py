from __future__ import annotations

import os
import hashlib
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
from types import SimpleNamespace

import backoff
import tiktoken
import numpy as np
from openai import OpenAI, AzureOpenAI, APIError, RateLimitError, APITimeoutError, NotFoundError, AuthenticationError

from cradle import constants
from cradle.provider.base import LLMProvider, EmbeddingProvider
from cradle.config import Config
from cradle.log import Logger
from cradle.utils.llm_call_budget import increment_llm_call_counter
from cradle.utils.json_utils import load_json
from cradle.utils.encoding_utils import encode_data_to_base64_path
from cradle.utils.file_utils import assemble_project_path

config = Config()
logger = Logger()

MAX_TOKENS = {
    "gpt-3.5-turbo-0301": 4097,
    "gpt-3.5-turbo-0613": 4097,
    "gpt-3.5-turbo-16k-0613": 16385,
}

# Mapping for models that tiktoken doesn't recognize natively
# These models should use cl100k_base encoding (GPT-4 compatible)
TIKTOKEN_MODEL_MAPPING = {
    "qwen": "cl100k_base",
    "qwen3": "cl100k_base",
    "qwen-vl": "cl100k_base",
    "qwen3-vl": "cl100k_base",
    "deepseek": "cl100k_base",
    "glm": "cl100k_base",
    "yi": "cl100k_base",
    "text-embedding": "cl100k_base",  # Alibaba Cloud embedding models
}

PROVIDER_SETTING_KEY_VAR = "key_var"
PROVIDER_SETTING_EMB_MODEL = "emb_model"
PROVIDER_SETTING_COMP_MODEL = "comp_model"
PROVIDER_SETTING_IS_AZURE = "is_azure"
PROVIDER_SETTING_BASE_VAR = "base_var"       # Azure-speficic setting
PROVIDER_SETTING_API_VERSION = "api_version" # Azure-speficic setting
PROVIDER_SETTING_DEPLOYMENT_MAP = "models"   # Azure-speficic setting


def _get_tiktoken_encoding(model_name: str):
    """Get tiktoken encoding for a model, with fallback for non-OpenAI models."""
    try:
        return tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Check if any prefix in TIKTOKEN_MODEL_MAPPING matches the model name
        model_lower = model_name.lower()
        for prefix, encoding_name in TIKTOKEN_MODEL_MAPPING.items():
            if model_lower.startswith(prefix):
                logger.debug(f"Using {encoding_name} encoding for model {model_name}")
                return tiktoken.get_encoding(encoding_name)
        # Final fallback
        logger.warn(f"Warning: model {model_name} not found. Using cl100k_base encoding.")
        return tiktoken.get_encoding("cl100k_base")


class OpenAIProvider(LLMProvider, EmbeddingProvider):
    """A class that wraps a given model"""

    client: Any = None
    embedding_client: Any = None
    llm_model: str = ""
    embedding_model: str = ""
    fallback_embedding_dim: int = 1024
    _local_st_model: Any = None
    _local_st_dim: int = 0
    embedding_use_dedicated_key: bool = False

    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    chunk_size: int = 1000
    embedding_ctx_length: int = 8191
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
        self.embedding_api_key = None
        self.embedding_base_url = None


    def init_provider(self, provider_cfg, embedding_only: bool = False) -> None:
        self.provider_cfg = self._parse_config(
            provider_cfg,
            embedding_only=embedding_only,
        )


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
        self.embedding_client = None
        self.embedding_use_dedicated_key = False
        self.embedding_model = conf_dict[PROVIDER_SETTING_EMB_MODEL]
        self.llm_model = conf_dict[PROVIDER_SETTING_COMP_MODEL]
        self.fallback_embedding_dim = int(conf_dict.get("emb_fallback_dim", 1024))

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

            if endpoint is None or endpoint == "":
                raise ValueError(f"Azure endpoint env var '{endpoint_var_name}' is missing or empty")

            self.client = AzureOpenAI(
                api_key = key,
                api_version = conf_dict[PROVIDER_SETTING_API_VERSION],
                azure_endpoint = endpoint
            )
            self.embedding_client = self.client
        else:
            # 优先使用硬编码的 api_key，否则从环境变量读取
            key = conf_dict.get("api_key") or os.getenv(key_var_name)
            base_url = conf_dict.get("base_url")

            if base_url:
                self.client = OpenAI(api_key=key, base_url=base_url)
            else:
                self.client = OpenAI(api_key=key)

            # embedding 专用 key（可选）：未配置时回退到通用 key，保持兼容
            emb_key = conf_dict.get("emb_api_key") or key
            emb_base_url = conf_dict.get("emb_base_url") or base_url
            self.embedding_api_key = emb_key
            self.embedding_base_url = emb_base_url
            self.embedding_use_dedicated_key = bool(conf_dict.get("emb_api_key")) and emb_key != key
            if emb_base_url:
                self.embedding_client = OpenAI(api_key=emb_key, base_url=emb_base_url)
            else:
                self.embedding_client = OpenAI(api_key=emb_key)

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
            self._local_st_model = SentenceTransformer(model_name, local_files_only=True)
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

        class _EmbeddingData:
            def __init__(self, embedding: List[float]):
                self.embedding = embedding

        class _EmbeddingResponse:
            def __init__(self, embeddings: List[List[float]]):
                self.data = [_EmbeddingData(e) for e in embeddings]

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

        def _local_hash_embeddings(input_payload: Any, dim: int) -> List[List[float]]:
            if isinstance(input_payload, str):
                texts = [input_payload]
            elif isinstance(input_payload, list):
                texts = [str(item) for item in input_payload]
            else:
                texts = [str(input_payload)]
            return [_local_hash_embedding(text, dim) for text in texts]

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
            if isinstance(input_payload, str):
                texts = [input_payload.strip() or " "]
            elif isinstance(input_payload, list):
                texts = [str(item).strip() or " " for item in input_payload]
            else:
                texts = [str(input_payload).strip() or " "]

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
                    return _EmbeddingResponse(
                        _local_hash_embeddings(texts, self.fallback_embedding_dim)
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
            return _EmbeddingResponse(
                _local_hash_embeddings(texts, self.fallback_embedding_dim)
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
            try:
                response = self.embedding_client.embeddings.create(**kwargs)
            except AuthenticationError as e:
                # embedding 专用 key 无效时回退到主 key（若不同）
                if self.embedding_use_dedicated_key and self.client is not None:
                    logger.warn(f"[Embedding] emb_api_key auth failed, fallback to api_key. detail: {e}")
                    self.embedding_client = self.client
                    self.embedding_use_dedicated_key = False
                    response = self.embedding_client.embeddings.create(**kwargs)
                else:
                    # 两个 key 都不可用时，进入本地向量兜底，保持流程可继续
                    logger.warn("[Embedding] API auth failed, using local hash embedding fallback")
                    fallback_embeddings = _local_hash_embeddings(
                        kwargs.get("input", ""),
                        self.fallback_embedding_dim,
                    )
                    return _EmbeddingResponse(fallback_embeddings)
            except NotFoundError as e:
                # 兼容模式下部分模型（如 multimodal-embedding-v1）可能不支持 embeddings API。
                # 不改变 emb_model 配置，直接本地向量兜底，避免长时间 backoff 阻塞主流程。
                err = str(e)
                if "model_not_supported" in err or "Unsupported model" in err:
                    logger.warn(
                        f"[Embedding] Model '{self.embedding_model}' unsupported in OpenAI compatibility mode. "
                        f"Using local hash embedding fallback (dim={self.fallback_embedding_dim})."
                    )
                    fallback_embeddings = _local_hash_embeddings(
                        kwargs.get("input", ""),
                        self.fallback_embedding_dim,
                    )
                    return _EmbeddingResponse(fallback_embeddings)
                raise
            if any(len(d.embedding) == 1 for d in response.data):
                raise RuntimeError("OpenAI API returned an empty embedding")
            return response

        return _embed_with_retry(**kwargs)


    def _get_len_safe_embeddings(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        embeddings: List[List[float]] = [[] for _ in range(len(texts))]
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
        encoding = _get_tiktoken_encoding(model_name)
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

        _iter = range(0, len(tokens), _chunk_size)

        for i in _iter:
            # Decode tokens back to text strings for API compatibility (DashScope requires text)
            batch_tokens = tokens[i : i + self.chunk_size]
            batch_texts = [encoding.decode(t) for t in batch_tokens]
            
            response = self.embed_with_retry(
                input=batch_texts,
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
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        if self._uses_local_sentence_transformers():
            model = self._get_local_st_model()
            if model is not None:
                return model.encode(
                    [t.strip() or " " for t in texts],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist()
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
        if self.embedding_model == "text-embedding-ada-002":
            embedding_dim = 1536
        else:
            raise ValueError(f"Unknown embedding model: {self.embedding_model}")
        return embedding_dim


    def create_completion(
        self,
        messages: List[Dict[str, str]],
        model: str | None = None,
        temperature: float = config.temperature,
        seed: Optional[int] = config.seed,
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
            interval=5,
            on_backoff=lambda details: logger.warn(
                f"[LLM Retry] Attempt {details.get('tries', '?')}/{self.retries}, "
                f"waiting 5s... ({type(details.get('exception')).__name__}: {details.get('exception')})"
            ),
        )
        def _generate_response_with_retry(
            messages: List[Dict[str, str]],
            model: str,
            temperature: float,
            seed: Optional[int] = None,
            max_tokens: int = 512,
        ) -> Tuple[str, Dict[str, int]]:

            """Send a request to the OpenAI API."""
            increment_llm_call_counter("big_brain:default")
            if self.provider_cfg[PROVIDER_SETTING_IS_AZURE]:
                response = self.client.chat.completions.create(model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,)
            else:
                response = self.client.chat.completions.create(model=model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_tokens=max_tokens,)

            if response is None:
                logger.error("Failed to get a response from OpenAI. Try again.")
                raise RuntimeError("OpenAI response is None")

            message = response.choices[0].message.content

            info = {
                "prompt_tokens" : response.usage.prompt_tokens,
                "completion_tokens" : response.usage.completion_tokens,
                "total_tokens" : response.usage.total_tokens,
                "system_fingerprint" : response.system_fingerprint,
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
            seed: Optional[int] = config.seed,
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
            interval=5,
            on_backoff=lambda details: logger.warn(
                f"[LLM Retry] Async attempt {details.get('tries', '?')}/{self.retries}, "
                f"waiting 5s... ({type(details.get('exception')).__name__}: {details.get('exception')})"
            ),
        )
        async def _generate_response_with_retry_async(
                messages: List[Dict[str, str]],
                model: str,
                temperature: float,
            seed: Optional[int] = None,
                max_tokens: int = 512,
        ) -> Tuple[str, Dict[str, int]]:

            """Send a request to the OpenAI API."""
            increment_llm_call_counter("big_brain:default_async")
            if self.provider_cfg[PROVIDER_SETTING_IS_AZURE]:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                )
            else:
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    seed=seed,
                    max_tokens=max_tokens,
                )

            if response is None:
                logger.error("Failed to get a response from OpenAI. Try again.")
                raise RuntimeError("OpenAI response is None")

            message = response.choices[0].message.content

            info = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
                "system_fingerprint": response.system_fingerprint,
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
        """Return the number of tokens used by a list of messages.
        Borrowed from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
        """
        encoding = _get_tiktoken_encoding(model)
        if model in {
            "gpt-4-1106-vision-preview",
        }:
            raise ValueError("We don't support counting tokens of GPT-4V yet.")

        # Default token counts for non-OpenAI models or newer models
        tokens_per_message = 3
        tokens_per_name = 1

        if model == "gpt-3.5-turbo-0301":
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted

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


    def assemble_prompt_tripartite(self, template_str: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:

        """
        A tripartite prompt is a message with the following structure:
        <system message>

        <user message part 1 before image introduction>
        <image introduction>
        <user message part 2 after image introduction>
        """
        if template_str is None:
            raise ValueError("template_str cannot be None")
        if params is None:
            params = {}

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

        if image_introduction_paragraph_index is None:
            user_messages_part1_paragraphs = filtered_paragraphs[1:]
            user_messages_part2_paragraphs = []
        else:
            user_messages_part1_paragraphs = filtered_paragraphs[1:image_introduction_paragraph_index]
            user_messages_part2_paragraphs = filtered_paragraphs[image_introduction_paragraph_index + 1:]

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
                    elif isinstance(paragraph_input, list):
                        paragraph_content = paragraph.replace(placeholder, json.dumps(paragraph_input))
                        user_messages_part1_contents.append(paragraph_content)
                    else:
                        raise ValueError(f"Unexpected input type: {type(paragraph_input)}")

        if len(user_messages_part1_contents) > 0:

            user_messages_part1_content = "\n\n".join(user_messages_part1_contents)

            user_messages_part1 = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{user_messages_part1_content}"
                    }
                ]
            }

        else:
            user_messages_part1 = None

        # assemble image introduction messages
        image_introduction_messages = []

        paragraph_input = params.get(constants.IMAGES_INPUT_TAG_NAME, []) # 'image_introduction'

        if paragraph_input is None or paragraph_input == "" or paragraph_input == []:
            image_introduction_messages = []
        else:
            if image_introduction_paragraph is None:
                raise ValueError(f"Template is missing {constants.IMAGES_INPUT_TAG} paragraph")
            paragraph_content_pre = image_introduction_paragraph.replace(constants.IMAGES_INPUT_TAG, "")
            message = {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{paragraph_content_pre}"
                    }
                ]
            }

            image_introduction_messages.append(message)

            for item in paragraph_input:
                introduction = item.get(constants.IMAGE_INTRO_TAG_NAME, None)
                path = item.get(constants.IMAGE_PATH_TAG_NAME, None)
                assistant = item.get(constants.IMAGE_ASSISTANT_TAG_NAME, None)
                resolution = item.get(constants.IMAGE_RESOLUTION_TAG_NAME, None)
                resize = item.get(constants.IMAGE_RESIZE_TAG_NAME, None)

                message = {
                    "role": "user",
                    "content": [],
                }

                if introduction is not None and introduction != "":
                    message["content"].append(
                        {
                            "type": "text",
                            "text": f"{introduction}"
                        })

                if path is not None and path != "":
                    encoded_images = encode_data_to_base64_path(path)

                    for encoded_image in encoded_images:
                        msg_content = {
                                "type": "image_url",
                                "image_url":
                                    {
                                        "url": f"{encoded_image}"
                                    }
                            }

                        if resolution is not None and resolution != "":
                            msg_content["image_url"]["detail"] = resolution

                        if resize is not None and resize != "":
                            msg_content["image_url"]["resize"] = resize

                        message["content"].append(msg_content)

                if len(message["content"]) > 0:
                    image_introduction_messages.append(message)

                if assistant is not None and assistant != "":
                    message = {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": f"{assistant}"
                            }
                        ]
                    }
                    image_introduction_messages.append(message)

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

        if user_messages_part1 is None:
            return [system_message] + image_introduction_messages + [user_messages_part2]
        else:
            return [system_message] + [user_messages_part1] + image_introduction_messages + [user_messages_part2]


    def assemble_prompt_paragraph(self, template_str: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError("This method is not implemented yet.")


    def assemble_prompt(self, template_str: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if config.DEFAULT_MESSAGE_CONSTRUCTION_MODE == constants.MESSAGE_CONSTRUCTION_MODE_TRIPART:
            return self.assemble_prompt_tripartite(template_str=template_str, params=params)
        elif config.DEFAULT_MESSAGE_CONSTRUCTION_MODE == constants.MESSAGE_CONSTRUCTION_MODE_PARAGRAPH:
            return self.assemble_prompt_paragraph(template_str=template_str, params=params)
        return self.assemble_prompt_tripartite(template_str=template_str, params=params)
