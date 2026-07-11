from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import yaml

from kura.data_contracts import project_dataset_facts


PNG_2X1 = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (2).to_bytes(4, "big") + (1).to_bytes(4, "big")
PNG_1X1 = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8 + (1).to_bytes(4, "big") + (1).to_bytes(4, "big")


class TrainingDataContractTests(unittest.TestCase):
    def test_colocated_sidecar_layout_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "001.png").write_bytes(PNG_1X1)
            (root / "001.txt").write_text("caption", encoding="utf-8")
            (root / "dataset.yaml").write_text(yaml.safe_dump({"stats": {"count": 1}}), encoding="utf-8")

            result = project_dataset_facts(root)

        self.assertEqual(result["facts"]["sample_count"], 1)
        self.assertEqual(result["samples"][0]["target"], "001.png")
        self.assertEqual(result["samples"][0]["caption_path"], "001.txt")
        self.assertEqual(result["issues"], [])

    def test_declared_separate_image_caption_layout_normalizes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "image").mkdir()
            (root / "caption").mkdir()
            (root / "image" / "001.png").write_bytes(PNG_1X1)
            (root / "caption" / "001.txt").write_text("caption", encoding="utf-8")
            metadata = {"stats": {"count": 1}, "layout": {"image_dir": "image", "caption_dir": "caption"}}
            (root / "dataset.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")

            result = project_dataset_facts(root)

        self.assertEqual(result["samples"][0]["target"], "image/001.png")
        self.assertEqual(result["samples"][0]["caption_path"], "caption/001.txt")
        self.assertEqual(result["facts"]["captions_missing"], 0)

    def test_items_jsonl_paths_override_directory_matching(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "target").mkdir()
            (root / "source").mkdir()
            (root / "target" / "different.png").write_bytes(PNG_1X1)
            (root / "source" / "source.png").write_bytes(PNG_1X1)
            item = {"id": "explicit", "path": "target/different.png", "source_path": "source/source.png", "caption": "edit"}
            (root / "items.jsonl").write_text(json.dumps(item) + "\n", encoding="utf-8")
            (root / "dataset.yaml").write_text(yaml.safe_dump({"stats": {"count": 1}}), encoding="utf-8")

            result = project_dataset_facts(root)

        sample = result["samples"][0]
        self.assertEqual(sample["target"], "target/different.png")
        self.assertEqual(sample["conditions"]["source"]["path"], "source/source.png")
        self.assertEqual(sample["caption"], "edit")

    def test_pair_count_and_aspect_mismatch_are_facts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("target", "source", "caption"):
                (root / name).mkdir()
            (root / "target" / "001.png").write_bytes(PNG_2X1)
            (root / "target" / "002.png").write_bytes(PNG_1X1)
            (root / "source" / "001.png").write_bytes(PNG_1X1)
            (root / "caption" / "001.txt").write_text("one", encoding="utf-8")
            (root / "caption" / "002.txt").write_text("two", encoding="utf-8")
            metadata = {"stats": {"count": 2}, "layout": {"target_dir": "target", "source_dir": "source", "caption_dir": "caption"}}
            (root / "dataset.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")

            result = project_dataset_facts(root)

        self.assertEqual(result["facts"]["condition_counts"], {"source": 1})
        self.assertEqual(result["facts"]["aspect_ratio_mismatches"], {"source": 1})
        self.assertEqual(result["facts"]["sample_count"], 2)

    def test_escaping_explicit_path_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "items.jsonl").write_text(json.dumps({"id": "bad", "path": "../outside.png"}) + "\n", encoding="utf-8")

            result = project_dataset_facts(root)

        codes = {item["code"] for item in result["issues"]}
        self.assertIn("path_outside_dataset", codes)
        self.assertIn("missing_target", codes)


if __name__ == "__main__":
    unittest.main()
