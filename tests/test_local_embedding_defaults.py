from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
LOCAL_EMBED_MODEL = "BAAI/bge-base-en-v1.5"


class TestLocalEmbeddingDefaults(unittest.TestCase):
    def test_default_embedding_configs_use_local_model(self) -> None:
        config_paths = (
            "conf/openai_config.json",
            "conf/opensrc_config.json",
            "conf/openai_t_config.json",
            "conf/azure_text_config.json",
            "conf/azure_vis_config.json",
        )

        for relative_path in config_paths:
            with self.subTest(config=relative_path):
                config = json.loads(
                    (AGENT_ROOT / relative_path).read_text(encoding="utf-8")
                )
                self.assertEqual(config["emb_model"], LOCAL_EMBED_MODEL)
                self.assertEqual(int(config["emb_fallback_dim"]), 768)

    def test_stardojo_provider_source_keeps_embedding_only_local_short_circuit(self) -> None:
        source = (AGENT_ROOT / "stardojo/provider/llm/openai.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "def init_provider(self, provider_cfg, embedding_only: bool = False)",
            source,
        )
        self.assertIn(
            "if embedding_only and self._uses_local_sentence_transformers():",
            source,
        )

    def test_stardojo_provider_source_gates_qwen_thinking_override(self) -> None:
        source = (AGENT_ROOT / "stardojo/provider/llm/openai.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "def _build_qwen_disable_thinking_extra_body(model: Any) -> Optional[Dict[str, Any]]:",
            source,
        )
        self.assertIn(
            'if "qwen" not in model_text:',
            source,
        )
        self.assertIn(
            'if extra_body is not None:',
            source,
        )

    def test_stardojo_factory_source_marks_separate_embed_provider_as_embedding_only(self) -> None:
        source = (AGENT_ROOT / "stardojo/provider/llm/llm_factory.py").read_text(
            encoding="utf-8"
        )

        self.assertEqual(source.count("embedding_only=True"), 3)

    def test_cradle_provider_source_keeps_embedding_only_local_short_circuit(self) -> None:
        source = (AGENT_ROOT / "cradle/provider/llm/openai.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "def init_provider(self, provider_cfg, embedding_only: bool = False)",
            source,
        )
        self.assertIn(
            "if embedding_only and self._uses_local_sentence_transformers():",
            source,
        )

    def test_cradle_factory_source_marks_separate_embed_provider_as_embedding_only(self) -> None:
        source = (AGENT_ROOT / "cradle/provider/llm/llm_factory.py").read_text(
            encoding="utf-8"
        )

        self.assertEqual(source.count("embedding_only=True"), 2)
