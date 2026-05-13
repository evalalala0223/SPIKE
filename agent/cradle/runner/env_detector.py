"""
Environment change detector using image embeddings (Phase 3.1).

Compares consecutive screenshot embeddings to detect significant
scene changes (e.g. entering a new area, opening a menu).
"""
from typing import Optional, Tuple

import numpy as np

from cradle.log import Logger
from cradle.runner.image_embedder import ImageEmbedder

logger = Logger()


class EnvironmentChangeDetector:
    """Detect significant environment changes between consecutive frames.

    Uses ImageEmbedder to compute cosine similarity between the current
    screenshot and the previous one.  A ``change_score`` above
    ``threshold`` signals a scene change that should trigger big-brain
    re-planning.
    """

    def __init__(self, image_embedder: Optional[ImageEmbedder] = None, threshold: float = 0.35):
        self.image_embedder = image_embedder
        self.threshold = threshold
        self.last_embedding: Optional[np.ndarray] = None
        self._embedding_disabled: bool = False

    def detect_change(self, screenshot_path: str) -> Tuple[bool, float]:
        """Compare current screenshot against the previous one.

        Args:
            screenshot_path: Path to the current screenshot image.

        Returns:
            (changed, change_score) where ``changed`` is True when the
            score exceeds the threshold.  ``change_score`` is in [0, 1]
            (0 = identical, 1 = completely different).
        """
        if self.image_embedder is None or self._embedding_disabled:
            return False, 0.0

        current_embedding = self.image_embedder.get_image_embedding(screenshot_path)

        # Zero vector means the API call failed; check for persistent failure (e.g. 403 quota exceeded)
        if np.allclose(current_embedding, 0):
            logger.warn("[EnvDetector] Got zero embedding, disabling embedding for session")
            self._embedding_disabled = True
            return False, 0.0

        if self.last_embedding is None:
            self.last_embedding = current_embedding
            return False, 0.0

        similarity = ImageEmbedder.compute_similarity(current_embedding, self.last_embedding)
        change_score = max(0.0, 1.0 - similarity)

        self.last_embedding = current_embedding

        if change_score > self.threshold:
            logger.write(f"[EnvDetector] Environment change detected: {change_score:.3f} > {self.threshold}")

        return change_score > self.threshold, change_score

    def reset(self):
        """Clear the stored embedding (e.g. on big-brain cycle start)."""
        self.last_embedding = None
