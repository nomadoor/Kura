from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from kura.cli import cmd_dataset_inspect
from kura.dataset_inspect import inspect_dataset


def png_bytes(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00" + b"\x00" * 16


class DatasetInspectTests(unittest.TestCase):
    def test_inspect_reports_dataset_facts_without_verdicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets" / "example"
            dataset.mkdir(parents=True)
            (dataset / "dataset.yaml").write_text("trigger_word: myaku\n", encoding="utf-8")
            (dataset / "a.png").write_bytes(png_bytes(400, 600))
            (dataset / "b.png").write_bytes(png_bytes(768, 768))
            (dataset / "c.png").write_bytes(png_bytes(1200, 1024))
            (dataset / "source").mkdir()
            (dataset / "target").mkdir()
            (dataset / "source" / "p1.png").write_bytes(png_bytes(512, 512))
            (dataset / "target" / "p1.png").write_bytes(png_bytes(512, 512))
            (dataset / "target" / "p2.png").write_bytes(png_bytes(512, 512))
            records = [
                {"id": "a", "path": "a.png", "caption": "myaku red suit"},
                {"id": "b", "path": "b.png", "caption": ""},
                {"id": "c", "path": "c.png", "caption": "myaku red suit"},
                {"id": "p", "target": "target/p1.png", "source": "source/p1.png", "caption": "side view"},
                {"id": "missing", "target": "target/p2.png", "caption": "side view"},
            ]
            (dataset / "items.jsonl").write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")

            report = inspect_dataset("example", workspace=root)

        self.assertEqual(report["images"]["items_jsonl_count"], 5)
        self.assertEqual(report["images"]["directory_count"], 6)
        self.assertEqual(report["images"]["resolution"]["min"], [400, 512])
        self.assertEqual(report["images"]["resolution"]["max"], [1200, 1024])
        self.assertEqual(report["images"]["resolution"]["below_512_count"], 1)
        self.assertEqual(report["captions"]["total"], 5)
        self.assertEqual(report["captions"]["empty"], 1)
        self.assertEqual(report["captions"]["duplicate_exact_count"], 4)
        self.assertEqual(report["captions"]["first_tokens_top3"][0], {"token": "myaku", "count": 2, "coverage": "2/5"})
        self.assertEqual(report["captions"]["trigger_word"]["occurrences"], 2)
        self.assertEqual(report["captions"]["trigger_word"]["first_matches"], 2)
        self.assertEqual(report["paired_control"]["source_count"], 1)
        self.assertEqual(report["paired_control"]["target_count"], 5)
        self.assertEqual(report["paired_control"]["missing_source_count"], 4)
        self.assertEqual(report["paired_control"]["directory_missing_source_count"], 1)

    def test_dataset_inspect_json_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets" / "example"
            dataset.mkdir(parents=True)
            (dataset / "dataset.yaml").write_text("{}\n", encoding="utf-8")
            (dataset / "items.jsonl").write_text(json.dumps({"id": "a", "path": "a.png", "caption": "plain"}) + "\n", encoding="utf-8")
            (dataset / "a.png").write_bytes(png_bytes(512, 512))
            output = io.StringIO()
            with mock.patch("kura.cli._workspace", return_value=root), contextlib.redirect_stdout(output):
                code = cmd_dataset_inspect(argparse.Namespace(dataset="example", json=True))

        self.assertEqual(code, 0)
        payload = json.loads(output.getvalue())
        self.assertEqual(payload["dataset"]["input"], "example")
        self.assertEqual(payload["captions"]["trigger_word"], {"declared": False, "value": None})

    def test_simple_dataset_id_prefers_workspace_datasets_over_cwd_collision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets" / "docs"
            dataset.mkdir(parents=True)
            (dataset / "dataset.yaml").write_text("{}\n", encoding="utf-8")
            (dataset / "items.jsonl").write_text(json.dumps({"id": "a", "path": "a.png", "caption": "dataset"}) + "\n", encoding="utf-8")
            (dataset / "a.png").write_bytes(png_bytes(512, 512))
            cwd = root / "docs"
            cwd.mkdir()
            previous = Path.cwd()
            try:
                os.chdir(root)
                report = inspect_dataset("docs", workspace=root)
            finally:
                os.chdir(previous)

        self.assertEqual(report["dataset"]["path"], str(dataset))
        self.assertEqual(report["images"]["directory_count"], 1)

    def test_dataset_inspect_missing_directory_exits_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with mock.patch("kura.cli._workspace", return_value=Path(tmp)), contextlib.redirect_stderr(stderr):
                code = cmd_dataset_inspect(argparse.Namespace(dataset="missing", json=True))

        self.assertEqual(code, 1)
        self.assertIn("cannot inspect dataset", stderr.getvalue())
