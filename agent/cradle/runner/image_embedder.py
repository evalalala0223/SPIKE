"""
Image embedder for environment change detection.

Defaults to a local, deterministic image embedding so dual-brain scene
change detection stays fully offline unless a remote provider is
explicitly configured.
"""
import base64
import json
import math
import os
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

from cradle.log import Logger

logger = Logger()


class ImageEmbedder:
    """Image embedding for environment change detection."""

    def __init__(
        self,
        provider: str = "local",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: str = "local-image-embedding-v1",
        dimensions: int = 1024,
    ):
        self.provider = str(provider or "local").strip()
        self.model = str(model or "local-image-embedding-v1").strip()
        self.dimensions = int(dimensions)
        self.api_key = api_key
        self.base_url = base_url
        logger.write(
            "[ImageEmbedder] Initialized "
            f"(provider={self.provider}, model={self.model}, dim={self.dimensions})"
        )

    @classmethod
    def from_config(
        cls,
        image_embedding_cfg: Optional[dict] = None,
        openai_config_path: str = "conf/openai_config.json",
    ) -> "ImageEmbedder":
        """Create from dual-brain image embedding config."""
        image_embedding_cfg = dict(image_embedding_cfg or {})
        provider = str(image_embedding_cfg.get("provider", "local") or "local").strip()
        default_model = (
            "local-image-embedding-v1"
            if provider.lower() == "local"
            else "multimodal-embedding-v1"
        )
        model = str(image_embedding_cfg.get("model", default_model) or default_model).strip()
        dimensions = int(
            image_embedding_cfg.get(
                "dimensions",
                image_embedding_cfg.get("dim", 1024),
            )
        )
        api_key = image_embedding_cfg.get("api_key")
        base_url = image_embedding_cfg.get("base_url")

        if provider.lower() != "local" and (not api_key or not base_url):
            from cradle.utils.file_utils import assemble_project_path

            resolved_path = assemble_project_path(openai_config_path)
            with open(resolved_path, "r", encoding="utf-8") as f:
                conf = json.load(f)
            api_key = api_key or conf.get("emb_api_key") or conf.get("api_key") or os.getenv(conf.get("key_var", ""))
            base_url = base_url or conf.get("emb_base_url") or conf.get("base_url")
            if "dimensions" not in image_embedding_cfg and "dim" not in image_embedding_cfg:
                dimensions = int(conf.get("image_emb_fallback_dim", dimensions))

        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            dimensions=dimensions,
        )

    @classmethod
    def from_openai_config(cls, config_path: str = "conf/openai_config.json") -> "ImageEmbedder":
        """Create from openai config.

        Defaults to local image embedding unless the config explicitly
        sets a remote image embedding provider/model.
        """
        from cradle.utils.file_utils import assemble_project_path

        resolved_path = assemble_project_path(config_path)
        with open(resolved_path, "r", encoding="utf-8") as f:
            conf = json.load(f)
        provider = str(conf.get("image_emb_provider", "local") or "local").strip()
        default_model = (
            "local-image-embedding-v1"
            if provider.lower() == "local"
            else "multimodal-embedding-v1"
        )
        api_key = conf.get("emb_api_key") or conf.get("api_key") or os.getenv(conf.get("key_var", ""))
        base_url = conf.get("emb_base_url") or conf.get("base_url")
        dim = int(conf.get("image_emb_fallback_dim", 1024))
        return cls(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=str(conf.get("image_emb_model", default_model) or default_model),
            dimensions=dim,
        )

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Compute a 1-D float32 image embedding."""
        if self._uses_local_image_embedding():
            return self._get_local_image_embedding(image_path)

        b64 = self._encode_image(image_path)
        image_payload = f"data:{self._guess_mime(image_path)};base64,{b64}"

        try:
            import dashscope

            if self.api_key:
                dashscope.api_key = self.api_key

            response = dashscope.MultiModalEmbedding.call(
                model=self.model,
                input=[{"image": image_payload}],
            )

            if getattr(response, "status_code", None) != 200:
                logger.warn(
                    f"[ImageEmbedder] API returned non-200 status: "
                    f"status_code={getattr(response, 'status_code', None)}, "
                    f"message={getattr(response, 'message', '')}"
                )
                return self._fallback_to_local_embedding(image_path)

            output = getattr(response, "output", {}) or {}
            embeddings = output.get("embeddings", []) if isinstance(output, dict) else []
            if not embeddings or not isinstance(embeddings[0], dict) or "embedding" not in embeddings[0]:
                logger.warn("[ImageEmbedder] Missing embedding field in API response")
                return self._fallback_to_local_embedding(image_path)

            embedding = np.array(embeddings[0]["embedding"], dtype=np.float32)
            if embedding.shape[0] != self.dimensions:
                logger.warn(
                    f"[ImageEmbedder] embedding dim mismatch: got={embedding.shape[0]}, expected={self.dimensions}. "
                    f"Using actual dim."
                )
                self.dimensions = int(embedding.shape[0])
        except Exception as e:
            logger.warn(f"[ImageEmbedder] API call failed: {e}")
            return self._fallback_to_local_embedding(image_path)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def _uses_local_image_embedding(self) -> bool:
        provider = self.provider.lower()
        model = self.model.lower()
        return provider == "local" or model.startswith("local-image-")

    def _fallback_to_local_embedding(self, image_path: str) -> np.ndarray:
        logger.warn(
            "[ImageEmbedder] Falling back to local image embedding "
            f"(provider={self.provider}, model={self.model})"
        )
        try:
            return self._get_local_image_embedding(image_path)
        except Exception as e:
            logger.warn(f"[ImageEmbedder] Local fallback failed: {e}")
            return np.zeros(self.dimensions, dtype=np.float32)

    def _get_local_image_embedding(self, image_path: str) -> np.ndarray:
        width, height = self._resolve_local_embedding_shape(self.dimensions)
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            image = image.convert("L")
            image = image.resize((width, height), Image.BILINEAR)
            embedding = np.asarray(image, dtype=np.float32).reshape(-1)

        if embedding.shape[0] != self.dimensions:
            if embedding.shape[0] > self.dimensions:
                embedding = embedding[: self.dimensions]
            else:
                embedding = np.pad(
                    embedding,
                    (0, self.dimensions - embedding.shape[0]),
                    mode="constant",
                )

        embedding = embedding / 255.0
        embedding = embedding - float(np.mean(embedding))
        norm = np.linalg.norm(embedding)
        if norm <= 0:
            embedding = np.zeros(self.dimensions, dtype=np.float32)
            embedding[0] = 1.0
            return embedding
        return embedding / norm

    @staticmethod
    def _resolve_local_embedding_shape(dimensions: int) -> tuple[int, int]:
        dim = max(1, int(dimensions))
        root = int(math.sqrt(dim))
        for height in range(root, 0, -1):
            if dim % height == 0:
                return dim // height, height
        return dim, 1

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(emb1, emb2))

    @staticmethod
    def _encode_image(image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _guess_mime(image_path: str) -> str:
        suffix = os.path.splitext(image_path)[1].lower()
        if suffix in {".png"}:
            return "image/png"
        if suffix in {".webp"}:
            return "image/webp"
        if suffix in {".bmp"}:
            return "image/bmp"
        return "image/jpeg"
