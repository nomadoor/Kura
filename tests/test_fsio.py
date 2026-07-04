from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from kura.fsio import atomic_write_json, atomic_write_text, atomic_write_yaml


class FsioTests(unittest.TestCase):
    def test_atomic_write_text_replaces_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            path.write_text("old\n", encoding="utf-8")

            atomic_write_text(path, "new\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "new\n")

    def test_atomic_write_json_uses_expected_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"

            atomic_write_json(path, {"state": "compiled", "outputs": ["モデル"]})

            text = path.read_text(encoding="utf-8")
            self.assertEqual(json.loads(text)["outputs"], ["モデル"])
            self.assertTrue(text.endswith("\n"))

    def test_atomic_write_yaml_uses_workspace_dump_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.yaml"

            atomic_write_yaml(path, {"name": "テスト", "items": [1, 2]})

            text = path.read_text(encoding="utf-8")
            self.assertEqual(yaml.safe_load(text)["items"], [1, 2])
            self.assertIn("name: テスト", text)

    def test_atomic_write_leaves_no_temporary_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "index.jsonl"

            atomic_write_text(path, "{}\n")

            self.assertEqual([item.name for item in root.iterdir()], ["index.jsonl"])

    def test_atomic_write_cleans_temporary_file_on_replace_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "status.json"
            path.write_text("old\n", encoding="utf-8")

            with patch("kura.fsio.os.replace", side_effect=OSError("boom")):
                with self.assertRaises(OSError):
                    atomic_write_text(path, "new\n")

            self.assertEqual(path.read_text(encoding="utf-8"), "old\n")
            self.assertEqual([item.name for item in root.iterdir()], ["status.json"])


if __name__ == "__main__":
    unittest.main()
