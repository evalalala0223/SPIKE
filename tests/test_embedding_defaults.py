from __future__ import annotations

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / "agent"

if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

from cradle.config.enhanced_config import EnhancedConfig, SAKGConfig
from cradle.runner.dual_brain import DualBrainController
from cradle.runner.image_embedder import ImageEmbedder


class TestEmbeddingDefaults(unittest.TestCase):
    def test_sakg_defaults_are_local(self) -> None:
        defaults = SAKGConfig()

        self.assertEqual(defaults.embedding_provider, "local")
        self.assertEqual(defaults.embedding_model, "BAAI/bge-base-en-v1.5")
        self.assertEqual(defaults.embedding_dim, 768)

    def test_partial_sakg_config_keeps_local_fallbacks(self) -> None:
        cfg = EnhancedConfig()
        old_raw = copy.deepcopy(getattr(cfg, "_raw_config", {}))
        old_sakg = copy.deepcopy(cfg.sa_kg)

        try:
            cfg._raw_config = {"sa_kg": {"enabled": True}}
            cfg.sa_kg = SAKGConfig()
            cfg._parse_config()

            self.assertEqual(cfg.sa_kg.embedding_provider, "local")
            self.assertEqual(cfg.sa_kg.embedding_model, "BAAI/bge-base-en-v1.5")
            self.assertEqual(cfg.sa_kg.embedding_dim, 768)
        finally:
            cfg._raw_config = old_raw
            cfg.sa_kg = old_sakg

    def test_image_embedder_defaults_to_local_mode(self) -> None:
        embedder = ImageEmbedder()

        self.assertEqual(embedder.provider, "local")
        self.assertEqual(embedder.model, "local-image-embedding-v1")
        self.assertEqual(embedder.dimensions, 1024)

    def test_local_image_embedding_returns_expected_shape(self) -> None:
        embedder = ImageEmbedder(dimensions=64)
        image_array = np.arange(64, dtype=np.uint8).reshape(8, 8)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            image_path = Path(temp_file.name)

        try:
            Image.fromarray(image_array).save(image_path)
            embedding = embedder.get_image_embedding(str(image_path))
        finally:
            image_path.unlink(missing_ok=True)

        self.assertEqual(embedding.shape, (64,))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0, places=5)

    def test_dual_brain_image_embedder_builder_uses_local_config(self) -> None:
        embedder = DualBrainController._build_image_embedder(
            {
                "provider": "local",
                "model": "local-image-embedding-v1",
                "dimensions": 64,
            }
        )

        self.assertIsInstance(embedder, ImageEmbedder)
        self.assertEqual(embedder.provider, "local")
        self.assertEqual(embedder.model, "local-image-embedding-v1")
        self.assertEqual(embedder.dimensions, 64)


if __name__ == "__main__":
    unittest.main()
