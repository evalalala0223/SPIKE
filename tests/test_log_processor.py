from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = ROOT / "agent"
for path in (AGENT_ROOT, ROOT):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

from agent.log_processor import _replacer


class TestLogProcessor(unittest.TestCase):
    def test_replacer_handles_single_hash_mapping_for_multiple_images(self) -> None:
        encoded_images = ["AAAA", "BBBB"]
        image_paths = {"only_hash": "screenshots\\step_00001.jpeg"}

        replaced = _replacer(
            text="AAAA then BBBB",
            encoded_images=encoded_images,
            image_paths=image_paths,
            work_dir=".",
        )

        self.assertIn("screenshots\\step_00001.jpeg", replaced)
        self.assertNotIn("AAAA", replaced)
        self.assertNotIn("BBBB", replaced)


if __name__ == "__main__":
    unittest.main()
