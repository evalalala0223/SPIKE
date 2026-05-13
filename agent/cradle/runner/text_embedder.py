"""
Text embedder wrapping the existing EmbeddingProvider (Phase 3.1).

Reuses the project's configured embedding provider (dashscope text-embedding-v4)
with an optional SHA-256 keyed cache for frequently queried texts.
"""
import hashlib
import os
import pickle
from typing import Dict, List, Optional

import numpy as np

from cradle.log import Logger

logger = Logger()


class TextEmbedder:
    """Text embedding via the existing EmbeddingProvider infrastructure.

    This is a thin wrapper that adds:
    - numpy array output (instead of List[float])
    - L2 normalization
    - Persistent disk cache for SA-KG / Mem0 records
    """

    def __init__(
        self,
        embedding_provider,
        cache_dir: Optional[str] = None,
    ):
        """
        Args:
            embedding_provider: Any object implementing ``embed_query(text) -> List[float]``.
                                Typically an ``OpenAIProvider`` instance.
            cache_dir: Directory for the persistent cache file.
                       ``None`` disables disk persistence (in-memory only).
        """
        self._provider = embedding_provider
        self._cache: Dict[str, np.ndarray] = {}
        self._cache_file = os.path.join(cache_dir, "text_embedding_cache.pkl") if cache_dir else None
        self._load_cache()

    def get_embedding(self, text: str, persist: bool = False) -> np.ndarray:
        """Get L2-normalized text embedding.

        Args:
            text: Input text.
            persist: If True, save this embedding to the disk cache
                     (use for SA-KG records that should survive restarts).

        Returns:
            1-D numpy array (float32), L2-normalized.
        """
        cache_key = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            raw: List[float] = self._provider.embed_query(text)
            embedding = np.array(raw, dtype=np.float32)
        except Exception as e:
            logger.warn(f"[TextEmbedder] embed_query failed: {e}")
            dim = self._guess_dim()
            return np.zeros(dim, dtype=np.float32)

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        if persist:
            self._cache[cache_key] = embedding
            self._save_cache()

        return embedding

    @staticmethod
    def compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(emb1, emb2))

    def _guess_dim(self) -> int:
        """Best-effort dimension guess from provider or cache."""
        if self._cache:
            return next(iter(self._cache.values())).shape[0]
        try:
            return self._provider.get_embedding_dim()
        except Exception:
            return 1024

    def _load_cache(self):
        if self._cache_file and os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, "rb") as f:
                    self._cache = pickle.load(f)
                logger.write(f"[TextEmbedder] Loaded {len(self._cache)} cached embeddings")
            except Exception as e:
                logger.warn(f"[TextEmbedder] Failed to load cache: {e}")
                self._cache = {}

    def _save_cache(self):
        if not self._cache_file:
            return
        os.makedirs(os.path.dirname(self._cache_file), exist_ok=True)
        tmp = f"{self._cache_file}.tmp"
        try:
            with open(tmp, "wb") as f:
                pickle.dump(self._cache, f)
            os.replace(tmp, self._cache_file)
        except Exception as e:
            logger.warn(f"[TextEmbedder] Failed to save cache: {e}")
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except Exception:
                    pass
