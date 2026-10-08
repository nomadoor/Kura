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
    def test_a_folder_of_images_and_captions_is_counted_from_the_caption_text(self) -> None:
        # Before items.jsonl exists, the captions are the .txt files beside the images.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets" / "folder"
            dataset.mkdir(parents=True)
            (dataset / "dataset.yaml").write_text("id: folder\ntrigger_word: myaku\n", encoding="utf-8")
            for stem, text in (("a", "myaku red suit"), ("b", "myaku blue suit"), ("c", "  \n")):
                (dataset / f"{stem}.png").write_bytes(png_bytes(512, 512))
                (dataset / f"{stem}.txt").write_text(text, encoding="utf-8")

            report = inspect_dataset("folder", workspace=root)

        self.assertEqual(report["observations"]["captions_present"], 2)
        self.assertEqual(report["observations"]["captions_missing"], 1)
        self.assertNotIn("empty", report["captions"])
        self.assertEqual(report["captions"]["first_tokens_top3"][0]["token"], "myaku")
        self.assertEqual(report["captions"]["trigger_word"]["caption_count"], 2)

    def test_a_missing_caption_file_is_one_finding_and_an_empty_inline_caption_defers_to_the_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets" / "rows"
            dataset.mkdir(parents=True)
            (dataset / "dataset.yaml").write_text("id: rows\n", encoding="utf-8")
            for stem in ("a", "b"):
                (dataset / f"{stem}.png").write_bytes(png_bytes(512, 512))
            (dataset / "b.caption").write_text("red suit", encoding="utf-8")
            records = [
                {"id": "a", "path": "a.png", "caption_path": "a.caption"},
                {"id": "b", "path": "b.png", "caption": "", "caption_path": "b.caption"},
            ]
            (dataset / "items.jsonl").write_text("\n".join(json.dumps(item) for item in records) + "\n", encoding="utf-8")

            report = inspect_dataset("rows", workspace=root)

        self.assertEqual(report["observations"]["captions_missing"], 1)
        findings = [item for item in report["structural_findings"] if item.get("sample") == "a"]
        self.assertEqual(len(findings), 1, findings)

    def test_inspect_reports_v2_typed_inputs_and_caption_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets" / "typed"
            dataset.mkdir(parents=True)
            (dataset / "target.png").write_bytes(png_bytes(2, 1))
            (dataset / "control.png").write_bytes(png_bytes(1, 1))
            (dataset / "caption.txt").write_text("hello", encoding="utf-8")
            (dataset / "dataset.yaml").write_text(
                "id: typed\nitems_schema_version: 2\n", encoding="utf-8",
            )
            (dataset / "items.jsonl").write_text(json.dumps({
                "id": "pair",
                "files": [
                    {"type": "file", "role": "target", "path": "target.png"},
                    {"type": "file", "role": "control", "path": "control.png"},
                ],
                "caption": {"file": {"type": "file", "path": "caption.txt"}},
            }) + "\n", encoding="utf-8")

            report = inspect_dataset("typed", workspace=root)

        self.assertEqual(report["images"]["items_jsonl_count"], 1)
        self.assertEqual(report["captions"]["total"], 1)
        self.assertEqual(report["observations"]["captions_missing"], 0)
        self.assertEqual(report["paired_control"]["source_count"], 1)
        self.assertEqual(report["paired_control"]["target_count"], 1)
        self.assertEqual(report["paired_control"]["missing_source_count"], 0)
        self.assertEqual(report["observations"]["condition_counts"], {"control": 1})

    def test_inspect_reports_v2_typed_video_targets_and_caption_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            dataset = workspace / "datasets" / "clips"
            dataset.mkdir(parents=True)
            (dataset / "dataset.yaml").write_text("id: clips\nitems_schema_version: 2\n", encoding="utf-8")
            for name in ("a.mp4", "b.mov", "unlisted.mp4"):
                (dataset / name).write_bytes(b"video")
            (dataset / "a.txt").write_text("a caption\n", encoding="utf-8")
            rows = [
                {"id": "a", "files": [{"type": "file", "role": "target", "path": "a.mp4"}],
                 "caption": {"file": {"type": "file", "path": "a.txt"}}},
                {"id": "b", "files": [{"type": "file", "role": "target", "path": "b.mov"}],
                 "caption": {"text": "b caption"}},
            ]
            (dataset / "items.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
            )

            report = inspect_dataset("clips", workspace=workspace)
            text = format_dataset_inspect(report)

            self.assertEqual(report["videos"]["items_jsonl_count"], 2)
            self.assertEqual(report["videos"]["count"], 3)
            self.assertEqual(report["images"]["items_jsonl_count"], 0)
            self.assertEqual(report["observations"]["captions_missing"], 0)
            self.assertIn("videos.items_jsonl_count: 2", text)
            self.assertIn("videos.directory_count: 3", text)

    def test_inspect_preserves_unicode_line_separator_in_v2_caption(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "datasets" / "typed"
            dataset.mkdir(parents=True)
            caption = "first\u2028second"
            (dataset / "target.png").write_bytes(png_bytes(1, 1))
            (dataset / "dataset.yaml").write_text(
                "id: typed\nitems_schema_version: 2\n", encoding="utf-8",
            )
            (dataset / "items.jsonl").write_text(json.dumps({
                "id": "target",
                "files": [{"type": "file", "role": "target", "path": "target.png"}],
                "caption": {"text": caption},
            }, ensure_ascii=False) + "\n", encoding="utf-8")

            report = inspect_dataset("typed", workspace=root)

        self.assertEqual(report["items_jsonl"], {"records": 1, "parse_errors": 0})
        self.assertEqual(report["captions"]["total"], 1)
        self.assertEqual(report["observations"]["captions_missing"], 0)
        self.assertNotIn("invalid_items_jsonl", {
            item.get("code") for item in report["structural_findings"]
        })

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
        self.assertEqual(report["observations"]["captions_missing"], 1)
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
