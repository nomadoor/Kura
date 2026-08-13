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
from kura.dataset_inspect import format_dataset_inspect, inspect_dataset


def png_bytes(width: int, height: int) -> bytes:
    return b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">II", width, height) + b"\x08\x02\x00\x00\x00" + b"\x00" * 16


class DatasetInspectTests(unittest.TestCase):
    def test_image_only_declared_layout_is_not_paired_control(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets" / "lora"
            images = dataset / "images"
            images.mkdir(parents=True)
            (images / "one.png").write_bytes(png_bytes(1, 1))
            (dataset / "dataset.yaml").write_text(
                "layout:\n  root: images\n  image_dir: images\n",
                encoding="utf-8",
            )
            (dataset / "items.jsonl").write_text(
                json.dumps({"id": "one", "path": "images/one.png", "caption": "plain", "role": "target"}) + "\n",
                encoding="utf-8",
            )

            report = inspect_dataset("lora", workspace=root)

        self.assertEqual(
            report["paired_control"],
            {
                "applicable": False,
                "source_count": None,
                "target_count": None,
                "missing_source_count": None,
                "missing_target_count": None,
                "directory_source_count": 0,
                "directory_target_count": 1,
                "directory_missing_source_count": None,
                "directory_missing_target_count": None,
            },
        )
        self.assertIn("paired_control: (not applicable)", format_dataset_inspect(report))

    def test_declared_layout_drives_pair_counts_and_observations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets" / "declared"
            targets = dataset / "renders"
            controls = dataset / "guides"
            captions = dataset / "texts"
            targets.mkdir(parents=True)
            controls.mkdir()
            captions.mkdir()
            (targets / "one.png").write_bytes(png_bytes(2, 1))
            (targets / "two.png").write_bytes(png_bytes(1, 1))
            (controls / "one.png").write_bytes(png_bytes(1, 1))
            (captions / "one.txt").write_text("one", encoding="utf-8")
            (captions / "two.txt").write_text("two", encoding="utf-8")
            (dataset / "dataset.yaml").write_text(
                "stats:\n  count: 2\nlayout:\n  target_dir: renders\n  control_dir: guides\n  caption_dir: texts\n",
                encoding="utf-8",
            )

            report = inspect_dataset("declared", workspace=root)

        paired = report["paired_control"]
        self.assertEqual(paired["directory_source_count"], 1)
        self.assertEqual(paired["directory_target_count"], 2)
        self.assertEqual(paired["directory_missing_source_count"], 1)
        self.assertEqual(paired["directory_missing_target_count"], 0)
        self.assertEqual(report["observations"]["sample_count"], 2)
        self.assertEqual(report["observations"]["captions_missing"], 0)
        self.assertEqual(report["observations"]["condition_counts"], {"control": 1})
        self.assertEqual(report["observations"]["aspect_ratio_mismatches"], {"control": 1})
        self.assertEqual(report["structural_findings"], [])
        text = format_dataset_inspect(report)
        self.assertIn("observations.aspect_ratio_mismatches.control: 1", text)
        self.assertIn("structural_findings.count: 0", text)

    def test_text_report_groups_structural_findings_by_code(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets" / "findings"
            images = dataset / "images"
            images.mkdir(parents=True)
            (images / "one.png").write_bytes(png_bytes(1, 1))
            (dataset / "dataset.yaml").write_text("stats:\n  count: 2\n", encoding="utf-8")
            (dataset / "items.jsonl").write_text(
                "{not json}\n" + json.dumps({"id": "one", "path": "images/one.png", "caption": "plain"}) + "\n",
                encoding="utf-8",
            )

            report = inspect_dataset("findings", workspace=root)

        text = format_dataset_inspect(report)
        self.assertIn("structural_findings.count: 2", text)
        self.assertIn("structural_findings.declared_count_mismatch: 1", text)
        self.assertIn("structural_findings.invalid_items_jsonl: 1", text)
        self.assertLess(
            text.index("structural_findings.declared_count_mismatch"),
            text.index("structural_findings.invalid_items_jsonl"),
        )

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
        self.assertIn("observations.aspect_ratio_mismatches: (none)", format_dataset_inspect(report))

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
        self.assertEqual(
            payload["paired_control"],
            {
                "applicable": False,
                "source_count": None,
                "target_count": None,
                "missing_source_count": None,
                "missing_target_count": None,
                "directory_source_count": 0,
                "directory_target_count": 0,
                "directory_missing_source_count": None,
                "directory_missing_target_count": None,
            },
        )

    def test_dataset_inspect_marks_declared_paired_dataset_applicable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets" / "edit"
            dataset.mkdir(parents=True)
            (dataset / "dataset.yaml").write_text("task: image-edit\n", encoding="utf-8")
            (dataset / "items.jsonl").write_text(json.dumps({"id": "a", "path": "a.png", "caption": "plain"}) + "\n", encoding="utf-8")
            (dataset / "a.png").write_bytes(png_bytes(512, 512))

            report = inspect_dataset("edit", workspace=root)

        self.assertTrue(report["paired_control"]["applicable"])
        self.assertEqual(report["paired_control"]["missing_source_count"], 1)

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
