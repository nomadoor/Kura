"""Manifest selection, projection completeness, and run-view contracts."""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from kura.backends.ai_toolkit import compile_ai_toolkit, project_ai_toolkit_dataset
from kura.backends.musubi_command import _musubi_max_resolution, _musubi_video_preflight_env
from kura.backends.musubi_datasets import (
    MUSUBI_PROJECTION_PROFILES,
    _musubi_h3_effective_task,
    _write_musubi_dataset_config,
    project_musubi_dataset,
)
from kura.backends.musubi_native_selectors import (
    MUSUBI_NATIVE_TASKS,
    musubi_native_task,
    musubi_native_task_profile,
)
from kura.cli import cmd_run_compile
from kura.dataset_handoff import freeze_dataset_handoff, inspect_dataset_handoff, inspect_dataset_sources, materialize_dataset_view, remove_dataset_views
from kura.dataset_handoff import local_training_mounts
from kura.executors.docker import docker_command, launch_docker
from kura.paths import inspect_workspace_symlinks
from kura.run_commands.plan import _dataset_layout_preflight_report, _dataset_runtime_checks, format_run_plan


class DatasetHandoffTests(unittest.TestCase):
    def test_musubi_task_table_is_the_single_source_for_defaults_and_dataset_properties(self) -> None:
        self.assertEqual(musubi_native_task("wan", None), "t2v-1.3B")
        self.assertEqual(musubi_native_task("kandinsky_5", None), "k5-pro-t2v-5s-sd")
        self.assertEqual(
            musubi_native_task_profile("wan", "t2v-14B-FC").dataset_kind,
            "video-control",
        )
        self.assertEqual(
            musubi_native_task_profile("wan", "i2v-14B").one_frame_kind,
            "single",
        )
        self.assertEqual(
            musubi_native_task_profile("wan", "flf2v-14B").one_frame_kind,
            "intermediate",
        )
        self.assertTrue(MUSUBI_NATIVE_TASKS["kandinsky5"]["k5-pro-t2v-5s-sd"].default)
        for profile in MUSUBI_PROJECTION_PROFILES.values():
            self.assertNotIn("effective_task", profile["mode"])
            for override in profile.get("mode_by_architecture", {}).values():
                self.assertNotIn("effective_task", override)

    def make_run(self, root: Path) -> tuple[dict, Path]:
        dataset = root / "datasets" / "tiny"
        dataset.mkdir(parents=True)
        (dataset / "dataset.yaml").write_text("id: tiny\nitems_schema_version: 2\n", encoding="utf-8")
        (dataset / "a.png").write_bytes(b"image")
        (dataset / "a.txt").write_text("caption\n", encoding="utf-8")
        (dataset / "items.jsonl").write_text(json.dumps({
            "id": "a",
            "files": [{"type": "file", "role": "target", "path": "a.png"}],
            "caption": {"file": {"type": "file", "path": "a.txt"}},
        }) + "\n", encoding="utf-8")
        run = {
            "id": "example",
            "backend": {"name": "ai-toolkit", "config": {}},
            "datasets": [{"id": "tiny"}],
        }
        resolved = root / "runs" / "example" / "resolved"
        resolved.mkdir(parents=True)
        return run, resolved

    def make_two_captionless_samples(self, root: Path) -> tuple[dict, Path]:
        run, resolved = self.make_run(root)
        dataset = root / "datasets" / "tiny"
        (dataset / "a.txt").unlink()
        (dataset / "b.jpg").write_bytes(b"second image")
        rows = [
            {
                "id": "a",
                "files": [{"type": "file", "role": "target", "path": "a.png"}],
                "caption": None,
            },
            {
                "id": "b",
                "files": [{"type": "file", "role": "target", "path": "b.jpg"}],
                "caption": None,
            },
        ]
        (dataset / "items.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
        )
        return run, resolved

    @staticmethod
    def write_tiny_manifest(root: Path, rows: list[dict], payloads: dict[str, bytes]) -> None:
        dataset = root / "datasets" / "tiny"
        for path in dataset.iterdir():
            if path.name not in {"dataset.yaml", "items.jsonl"} and path.is_file():
                path.unlink()
        for relative, content in payloads.items():
            path = dataset / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        (dataset / "items.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
        )

    @staticmethod
    def image_projection(selection: dict) -> dict:
        dataset = selection["datasets"][0]
        sample = dataset["samples"][0]
        target = sample["files"][0]
        caption = sample["caption"]
        view_root = f"runs/{selection['run_id']}/cache/dataset-view/ai-toolkit/tiny"
        return {
            "schema_version": 1,
            "backend": "ai-toolkit",
            "datasets": [{
                "id": "tiny",
                "consumed": [target["input_id"], caption["input_id"]],
                "unrepresentable": [],
                "semantic": {"caption_ext": ".txt"},
                "native_runtime": {"folder_path": "/workspace/" + view_root},
                "native": {"caption_ext": ".txt", "folder_path": "/workspace/" + view_root},
                "native_string_fields": ["/caption_ext"],
                "views": [{
                    "id": "primary",
                    "root": view_root,
                    "links": [{
                        "path": view_root + "/000000.png",
                        "target": "/workspace/datasets/tiny/a.png",
                        "input_id": target["input_id"],
                    }],
                    "files": [{
                        "path": view_root + "/000000.txt",
                        "text": caption["text"],
                        "input_id": caption["input_id"],
                    }],
                    "native_files": [],
                    "write_roots": [{"path": view_root, "native_pointer": "/folder_path"}],
                    "consumers": [{
                        "id": "dataset",
                        "kind": "recursive-directory",
                        "native_pointer": "/folder_path",
                        "path": view_root,
                        "input_ids": [target["input_id"], caption["input_id"]],
                    }],
                    "repeat": 1,
                    "bindings": [{
                        "rule": "same-relative-stem",
                        "key": "000000",
                        "members": [
                            {"input_id": target["input_id"], "root": view_root},
                            {"input_id": caption["input_id"], "root": view_root},
                        ],
                    }],
                }],
            }],
        }

    @classmethod
    def jsonl_projection(cls, selection: dict) -> dict:
        projection = cls.image_projection(selection)
        dataset = projection["datasets"][0]
        view = dataset["views"][0]
        target = view["links"][0]
        caption = view["files"][0]
        row = {
            "image_path": "/workspace/" + target["path"],
            "caption_path": "/workspace/" + caption["path"],
            "kind": "image",
        }
        view["native_files"] = [{
            "path": view["root"] + "/native/items.jsonl",
            "text": json.dumps(row) + "\n",
            "format": "jsonl",
            "literal_string_fields": ["/kind"],
            "rows": [{
                "row_id": "row-a",
                "sample_id": "a",
                "repeat": None,
                "references": [
                    {
                        "kind": "path",
                        "pointer": "/image_path",
                        "input_id": target["input_id"],
                        "path": target["path"],
                    },
                    {
                        "kind": "path",
                        "pointer": "/caption_path",
                        "input_id": caption["input_id"],
                        "path": caption["path"],
                    },
                ],
                "literal_strings": [{"pointer": "/kind", "value": "image"}],
            }],
        }]
        native_file = view["native_files"][0]["path"]
        dataset["native_runtime"] = {"image_jsonl_file": "/workspace/" + native_file}
        dataset["native"] = {**dataset["semantic"], **dataset["native_runtime"]}
        view["write_roots"][0]["native_pointer"] = "/image_jsonl_file"
        view["consumers"] = [{
            "id": "items",
            "kind": "jsonl",
            "native_pointer": "/image_jsonl_file",
            "native_file": native_file,
        }]
        view.pop("bindings")
        return projection

    @classmethod
    def inline_caption_jsonl_projection(cls, selection: dict) -> dict:
        projection = cls.jsonl_projection(selection)
        dataset = projection["datasets"][0]
        sample = selection["datasets"][0]["samples"][0]
        target = sample["files"][0]
        caption = sample["caption"]
        view = dataset["views"][0]
        view["files"] = []
        native = view["native_files"][0]
        row = json.loads(native["text"])
        row.pop("caption_path")
        row["caption"] = caption["text"]
        native["text"] = json.dumps(row) + "\n"
        native["rows"][0]["references"][1] = {
            "kind": "caption-text",
            "pointer": "/caption",
            "input_id": caption["input_id"],
        }
        self_reference = native["rows"][0]["references"][0]
        assert self_reference["input_id"] == target["input_id"]
        return projection

    @classmethod
    def c3_image_projection(cls, selection: dict) -> dict:
        return cls.image_projection(selection)

    @classmethod
    def c3_jsonl_projection(cls, selection: dict) -> dict:
        return cls.jsonl_projection(selection)

    def test_complete_projection_freezes_lock_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=self.image_projection,
            )

            self.assertEqual(lock["verification"], "content-hash-at-compile")
            self.assertEqual(
                {item["source"] for item in lock["files"]},
                {"datasets/tiny/a.png", "datasets/tiny/a.txt"},
            )
            self.assertEqual(lock["views"][0]["links"][0]["target"], "/workspace/datasets/tiny/a.png")
            report = json.loads((resolved / "dataset-projection.lock.json").read_text(encoding="utf-8"))
            self.assertEqual(report["datasets"][0]["native"]["folder_path"],
                             "/workspace/runs/example/cache/dataset-view/ai-toolkit/tiny")
            self.assertTrue((resolved / "dataset-input.lock.json").is_file())

    def test_generated_jsonl_rows_are_verified_and_materialized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=self.jsonl_projection,
            )
            view = materialize_dataset_view(workspace, lock)

            native = lock["views"][0]["native_files"][0]
            self.assertEqual(native["rows"][0]["row_id"], "row-a")
            self.assertEqual(native["rows"][0]["sample_id"], "a")
            self.assertEqual((view / "native" / "items.jsonl").read_text(encoding="utf-8"), native["text"])
            self.assertEqual(inspect_dataset_handoff(workspace, lock), [])

    def test_generated_jsonl_row_must_reference_every_sample_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def omits_caption(selection: dict) -> dict:
                projection = self.jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                row = json.loads(native["text"])
                row.pop("caption_path")
                native["text"] = json.dumps(row) + "\n"
                native["rows"][0]["references"].pop()
                return projection

            with self.assertRaisesRegex(ValueError, "does not include every sample input exactly once"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=omits_caption,
                )

    def test_generated_jsonl_repeat_requires_an_explicit_complete_declaration(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def undeclared_repeat(selection: dict) -> dict:
                projection = self.jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                native["text"] += native["text"]
                second = json.loads(json.dumps(native["rows"][0]))
                second["row_id"] = "row-a-second"
                native["rows"].append(second)
                return projection

            with self.assertRaisesRegex(ValueError, "repeat.*must be declared"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=undeclared_repeat,
                )

    def test_generated_jsonl_explicit_repeat_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def explicit_repeat(selection: dict) -> dict:
                projection = self.jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                native["text"] += native["text"]
                second = json.loads(json.dumps(native["rows"][0]))
                second["row_id"] = "row-a-second"
                native["rows"][0]["repeat"] = {"index": 0, "count": 2}
                second["repeat"] = {"index": 1, "count": 2}
                native["rows"].append(second)
                return projection

            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=explicit_repeat,
            )
            self.assertEqual(
                [row["repeat"]["index"] for row in lock["views"][0]["native_files"][0]["rows"]],
                [0, 1],
            )

    def test_generated_jsonl_can_consume_exact_manifest_caption_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=self.inline_caption_jsonl_projection,
            )

            native = lock["views"][0]["native_files"][0]
            self.assertEqual(native["rows"][0]["references"][1]["kind"], "caption-text")
            self.assertEqual(lock["semantic"]["datasets"][0]["samples"][0]["caption"], "caption\n")

    def test_generated_jsonl_preserves_unicode_line_separator_inside_caption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            caption = "first\u2028second"
            (workspace / "datasets" / "tiny" / "a.txt").write_text(caption, encoding="utf-8")

            def unicode_line_separator(selection: dict) -> dict:
                projection = self.inline_caption_jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                row = json.loads(native["text"])
                native["text"] = json.dumps(row, ensure_ascii=False) + "\n"
                return projection

            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=unicode_line_separator,
            )

            native = lock["views"][0]["native_files"][0]
            self.assertIn(caption, native["text"])
            self.assertEqual(
                lock["semantic"]["datasets"][0]["samples"][0]["caption"],
                caption,
            )

    def test_generated_jsonl_rejects_changed_manifest_caption_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def changed_caption(selection: dict) -> dict:
                projection = self.inline_caption_jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                row = json.loads(native["text"])
                row["caption"] = "different"
                native["text"] = json.dumps(row) + "\n"
                return projection

            with self.assertRaisesRegex(ValueError, "does not preserve its caption input"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=changed_caption,
                )

    def test_generated_jsonl_strip_caption_reference_allows_only_the_declared_transform(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            (workspace / "datasets" / "tiny" / "a.txt").write_text("  caption\n", encoding="utf-8")

            def stripped_caption(selection: dict) -> dict:
                projection = self.inline_caption_jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                row = json.loads(native["text"])
                row["caption"] = row["caption"].strip()
                native["text"] = json.dumps(row) + "\n"
                native["rows"][0]["references"][1]["kind"] = "caption-text-strip"
                return projection

            freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=stripped_caption,
            )

            def changed_after_strip(selection: dict) -> dict:
                projection = stripped_caption(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                row = json.loads(native["text"])
                row["caption"] = "different"
                native["text"] = json.dumps(row) + "\n"
                return projection

            with self.assertRaisesRegex(ValueError, "does not preserve its caption input"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=changed_after_strip,
                )

    def test_generated_jsonl_identity_is_stable_across_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first_run, first_resolved = self.make_run(workspace)
            first = freeze_dataset_handoff(
                first_run, workspace, first_resolved, backend="ai-toolkit", project=self.jsonl_projection,
            )
            second_run = {**first_run, "id": "other"}
            second_resolved = workspace / "runs" / "other" / "resolved"
            second = freeze_dataset_handoff(
                second_run, workspace, second_resolved, backend="ai-toolkit", project=self.jsonl_projection,
            )

            self.assertEqual(first["input_sha256"], second["input_sha256"])

    def test_generated_jsonl_non_path_values_are_part_of_input_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first_run, first_resolved = self.make_run(workspace)

            def with_fps(selection: dict, fps: int) -> dict:
                projection = self.jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                row = json.loads(native["text"])
                row["fps"] = fps
                native["text"] = json.dumps(row) + "\n"
                return projection

            first = freeze_dataset_handoff(
                first_run,
                workspace,
                first_resolved,
                backend="ai-toolkit",
                project=lambda selection: with_fps(selection, 24),
            )
            second_run = {**first_run, "id": "other"}
            second_resolved = workspace / "runs" / "other" / "resolved"
            second = freeze_dataset_handoff(
                second_run,
                workspace,
                second_resolved,
                backend="ai-toolkit",
                project=lambda selection: with_fps(selection, 25),
            )

            self.assertNotEqual(first["input_sha256"], second["input_sha256"])

    def test_generated_jsonl_rejects_an_unreported_string_field(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def unreported(selection: dict) -> dict:
                projection = self.jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                row = json.loads(native["text"])
                row["other_path"] = row["image_path"]
                native["text"] = json.dumps(row) + "\n"
                return projection

            with self.assertRaisesRegex(ValueError, "unclassified string field.*other_path"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=unreported,
                )

    def test_generated_jsonl_rejects_a_reference_to_the_wrong_view_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def wrong_entry(selection: dict) -> dict:
                projection = self.jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                native["rows"][0]["references"][0]["input_id"] = (
                    native["rows"][0]["references"][1]["input_id"]
                )
                return projection

            with self.assertRaisesRegex(ValueError, "does not match its projected view entry"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=wrong_entry,
                )

    def test_generated_jsonl_rejects_unreported_row_multiplicity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def extra_row(selection: dict) -> dict:
                projection = self.jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                native["text"] += native["text"]
                return projection

            with self.assertRaisesRegex(ValueError, "row count"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=extra_row,
                )

    def test_generated_jsonl_rejects_rows_in_a_different_order_than_reported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def reordered(selection: dict) -> dict:
                projection = self.jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                first_value = json.loads(native["text"])
                first_value["variant"] = "first"
                second_value = {**first_value, "variant": "second"}
                first_report = native["rows"][0]
                first_report["literal_strings"].append({"pointer": "/variant", "value": "first"})
                second_report = json.loads(json.dumps(first_report))
                second_report["row_id"] = "row-a-second"
                second_report["literal_strings"][-1]["value"] = "second"
                native["rows"] = [first_report, second_report]
                native["text"] = json.dumps(second_value) + "\n" + json.dumps(first_value) + "\n"
                return projection

            with self.assertRaisesRegex(ValueError, "differs from the generated row"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=reordered,
                )

    def test_generated_jsonl_rejects_unknown_or_duplicate_row_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def unknown_sample(selection: dict) -> dict:
                projection = self.jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                native["rows"][0]["sample_id"] = "missing"
                return projection

            with self.assertRaisesRegex(ValueError, "unknown sample"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=unknown_sample,
                )

            def duplicate_row(selection: dict) -> dict:
                projection = self.jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                native["text"] += native["text"]
                native["rows"].append(json.loads(json.dumps(native["rows"][0])))
                return projection

            with self.assertRaisesRegex(ValueError, "duplicate row identity"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=duplicate_row,
                )

    def test_generated_jsonl_rejects_a_workspace_path_classified_as_literal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def disguised_path(selection: dict) -> dict:
                projection = self.jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                row = json.loads(native["text"])
                row["kind"] = row["image_path"]
                native["text"] = json.dumps(row) + "\n"
                native["rows"][0]["literal_strings"][0]["value"] = row["kind"]
                return projection

            with self.assertRaisesRegex(ValueError, "classifies a path-like value as a literal"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=disguised_path,
                )

    def test_generated_jsonl_rejects_crlf_row_separators(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def crlf(selection: dict) -> dict:
                projection = self.jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                native["text"] = native["text"].replace("\n", "\r\n")
                return projection

            with self.assertRaisesRegex(ValueError, "LF row separators"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=crlf,
                )

    def test_generated_jsonl_literal_must_be_declared_by_the_adapter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def undeclared_literal(selection: dict) -> dict:
                projection = self.jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                row = json.loads(native["text"])
                row["note"] = "ordinary"
                native["text"] = json.dumps(row) + "\n"
                native["rows"][0]["literal_strings"].append({"pointer": "/note", "value": "ordinary"})
                return projection

            with self.assertRaisesRegex(ValueError, "literal field.*not declared"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=undeclared_literal,
                )

    def test_generated_jsonl_rejects_path_like_literal_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def path_literal(selection: dict) -> dict:
                projection = self.jsonl_projection(selection)
                native = projection["datasets"][0]["views"][0]["native_files"][0]
                row = json.loads(native["text"])
                row["note"] = "image.png"
                native["text"] = json.dumps(row) + "\n"
                native["literal_string_fields"].append("/note")
                native["rows"][0]["literal_strings"].append({
                    "pointer": "/note", "value": "image.png",
                })
                return projection

            with self.assertRaisesRegex(ValueError, "path-like value as a literal"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=path_literal,
                )

    def test_materialized_generated_jsonl_is_checked_for_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=self.jsonl_projection,
            )
            view = materialize_dataset_view(workspace, lock)
            (view / "native" / "items.jsonl").write_text("{}\n", encoding="utf-8")

            self.assertIn("changed generated view file", " ".join(inspect_dataset_handoff(workspace, lock)))

    def test_projection_missing_one_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def drops_caption(selection: dict) -> dict:
                projection = self.image_projection(selection)
                projection["datasets"][0]["consumed"].pop()
                return projection

            with self.assertRaisesRegex(ValueError, "consumed inputs differ from verified native inputs"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=drops_caption,
                )
            self.assertFalse((resolved / "dataset-input.lock.json").exists())

    def test_projection_cannot_claim_an_input_missing_from_the_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def missing_link(selection: dict) -> dict:
                projection = self.image_projection(selection)
                projection["datasets"][0]["views"][0]["links"] = []
                return projection

            with self.assertRaisesRegex(ValueError, "binding.*absent from the view"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=missing_link,
                )

    def test_projection_cannot_shift_caption_to_a_different_stem(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def shifted(selection: dict) -> dict:
                projection = self.image_projection(selection)
                projection["datasets"][0]["views"][0]["files"][0]["path"] = (
                    "runs/example/cache/dataset-view/ai-toolkit/tiny/000001.txt"
                )
                return projection

            with self.assertRaisesRegex(ValueError, "violates its association key"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=shifted,
                )

    def test_projection_same_stem_in_different_directories_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def shifted_directory(selection: dict) -> dict:
                projection = self.image_projection(selection)
                projection["datasets"][0]["views"][0]["files"][0]["path"] = (
                    "runs/example/cache/dataset-view/ai-toolkit/tiny/other/000000.txt"
                )
                return projection

            with self.assertRaisesRegex(ValueError, "violates its association key"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=shifted_directory,
                )

    def test_projection_associates_same_relative_stem_across_role_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def cross_folder(selection: dict) -> dict:
                projection = self.image_projection(selection)
                dataset = projection["datasets"][0]
                view = dataset["views"][0]
                view_root = view["root"]
                target = view["links"][0]
                caption = view["files"][0]
                target["path"] = view_root + "/targets/000000.png"
                caption["path"] = view_root + "/captions/000000.txt"
                view["bindings"] = [{
                    "rule": "same-relative-stem",
                    "key": "000000",
                    "members": [
                        {"input_id": target["input_id"], "root": view_root + "/targets"},
                        {"input_id": caption["input_id"], "root": view_root + "/captions"},
                    ],
                }]
                return projection

            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=cross_folder,
            )

            self.assertEqual(
                {Path(item["path"]).parent.name for item in lock["views"][0]["links"] + lock["views"][0]["files"]},
                {"targets", "captions"},
            )

    def test_projection_does_not_pair_global_duplicate_basenames(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def unrelated_nested_paths(selection: dict) -> dict:
                projection = self.image_projection(selection)
                dataset = projection["datasets"][0]
                view = dataset["views"][0]
                view_root = view["root"]
                target = view["links"][0]
                caption = view["files"][0]
                target["path"] = view_root + "/targets/a/000000.png"
                caption["path"] = view_root + "/captions/b/000000.txt"
                view["bindings"] = [{
                    "rule": "same-relative-stem",
                    "key": "a/000000",
                    "members": [
                        {"input_id": target["input_id"], "root": view_root + "/targets"},
                        {"input_id": caption["input_id"], "root": view_root + "/captions"},
                    ],
                }]
                return projection

            with self.assertRaisesRegex(ValueError, "violates its association key"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=unrelated_nested_paths,
                )

    def test_projection_association_can_cover_three_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            dataset_root = workspace / "datasets" / "tiny"
            (dataset_root / "control").mkdir()
            (dataset_root / "control" / "a.png").write_bytes(b"control")
            row = json.loads((dataset_root / "items.jsonl").read_text(encoding="utf-8"))
            row["files"].append({"type": "file", "role": "control", "path": "control/a.png"})
            (dataset_root / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            def three_inputs(selection: dict) -> dict:
                projection = self.image_projection(selection)
                selected = selection["datasets"][0]["samples"][0]
                control = next(item for item in selected["files"] if item["role"] == "control")
                projected = projection["datasets"][0]
                view = projected["views"][0]
                view_root = view["root"]
                target = view["links"][0]
                caption = view["files"][0]
                target["path"] = view_root + "/targets/000000.png"
                caption["path"] = view_root + "/captions/000000.txt"
                view["links"].append({
                    "path": view_root + "/controls/000000.png",
                    "target": "/workspace/datasets/tiny/control/a.png",
                    "input_id": control["input_id"],
                })
                view["consumers"][0]["input_ids"].append(control["input_id"])
                projected["consumed"].append(control["input_id"])
                view["bindings"] = [{
                    "rule": "same-relative-stem",
                    "key": "000000",
                    "members": [
                        {"input_id": target["input_id"], "root": view_root + "/targets"},
                        {"input_id": control["input_id"], "root": view_root + "/controls"},
                        {"input_id": caption["input_id"], "root": view_root + "/captions"},
                    ],
                }]
                return projection

            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=three_inputs,
            )

            self.assertEqual(len(lock["views"][0]["links"]), 2)

    def test_projection_rejects_legacy_same_stem_binding_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def legacy_binding(selection: dict) -> dict:
                projection = self.image_projection(selection)
                dataset = projection["datasets"][0]
                view = dataset["views"][0]
                member_ids = [member["input_id"] for member in view["bindings"][0]["members"]]
                view["bindings"] = [{"rule": "same-stem", "inputs": member_ids}]
                return projection

            with self.assertRaisesRegex(ValueError, "unsupported rule"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=legacy_binding,
                )

    def test_projection_rejects_duplicate_association_key_across_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_two_captionless_samples(workspace)

            def duplicate_key(selection: dict) -> dict:
                dataset = selection["datasets"][0]
                first = dataset["samples"][0]["files"][0]
                second = dataset["samples"][1]["files"][0]
                view_root = "runs/example/cache/dataset-view/test/tiny"
                semantic = {"format": "test"}
                runtime = {"folder_path": "/workspace/" + view_root}
                return {
                    "schema_version": 1,
                    "backend": "ai-toolkit",
                    "datasets": [{
                        "id": "tiny",
                        "consumed": [first["input_id"], second["input_id"]],
                        "unrepresentable": [],
                        "semantic": semantic,
                        "native_runtime": runtime,
                        "native": {**semantic, **runtime},
                        "native_string_fields": ["/format"],
                        "views": [{
                            "id": "primary",
                            "root": view_root,
                            "links": [
                                {
                                    "path": view_root + "/000000.png",
                                    "target": "/workspace/datasets/tiny/a.png",
                                    "input_id": first["input_id"],
                                },
                                {
                                    "path": view_root + "/000000.jpg",
                                    "target": "/workspace/datasets/tiny/b.jpg",
                                    "input_id": second["input_id"],
                                },
                            ],
                            "files": [],
                            "native_files": [],
                            "write_roots": [{"path": view_root, "native_pointer": "/folder_path"}],
                            "consumers": [{
                                "id": "dataset",
                                "kind": "recursive-directory",
                                "native_pointer": "/folder_path",
                                "path": view_root,
                                "input_ids": [first["input_id"], second["input_id"]],
                            }],
                            "repeat": 1,
                            "bindings": [
                                {
                                    "rule": "same-relative-stem",
                                    "key": "000000",
                                    "members": [{"input_id": first["input_id"], "root": view_root}],
                                },
                                {
                                    "rule": "same-relative-stem",
                                    "key": "000000",
                                    "members": [{"input_id": second["input_id"], "root": view_root}],
                                },
                            ],
                        }],
                    }],
                }

            with self.assertRaisesRegex(ValueError, "duplicate association key"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=duplicate_key,
                )

    def test_projection_rejects_one_sample_split_across_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def split_sample(selection: dict) -> dict:
                projection = self.image_projection(selection)
                dataset = projection["datasets"][0]
                view = dataset["views"][0]
                view_root = view["root"]
                target = view["links"][0]
                caption = view["files"][0]
                caption["path"] = view_root + "/000001.txt"
                view["bindings"] = [
                    {
                        "rule": "same-relative-stem",
                        "key": "000000",
                        "members": [{"input_id": target["input_id"], "root": view_root}],
                    },
                    {
                        "rule": "same-relative-stem",
                        "key": "000001",
                        "members": [{"input_id": caption["input_id"], "root": view_root}],
                    },
                ]
                return projection

            with self.assertRaisesRegex(ValueError, "sample .* has more than one binding"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=split_sample,
                )

    def test_projection_rejects_one_input_kind_in_multiple_role_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_two_captionless_samples(workspace)

            def inconsistent_roots(selection: dict) -> dict:
                dataset = selection["datasets"][0]
                first = dataset["samples"][0]["files"][0]
                second = dataset["samples"][1]["files"][0]
                view_root = "runs/example/cache/dataset-view/test/tiny"
                first_root = view_root + "/targets-a"
                second_root = view_root + "/targets-b"
                semantic = {"format": "test"}
                runtime = {"folder_path": "/workspace/" + view_root}
                return {
                    "schema_version": 1,
                    "backend": "ai-toolkit",
                    "datasets": [{
                        "id": "tiny",
                        "consumed": [first["input_id"], second["input_id"]],
                        "unrepresentable": [],
                        "semantic": semantic,
                        "native_runtime": runtime,
                        "native": {**semantic, **runtime},
                        "native_string_fields": ["/format"],
                        "views": [{
                            "id": "primary",
                            "root": view_root,
                            "links": [
                                {
                                    "path": first_root + "/000000.png",
                                    "target": "/workspace/datasets/tiny/a.png",
                                    "input_id": first["input_id"],
                                },
                                {
                                    "path": second_root + "/000001.jpg",
                                    "target": "/workspace/datasets/tiny/b.jpg",
                                    "input_id": second["input_id"],
                                },
                            ],
                            "files": [],
                            "native_files": [],
                            "write_roots": [{"path": view_root, "native_pointer": "/folder_path"}],
                            "consumers": [{
                                "id": "dataset",
                                "kind": "recursive-directory",
                                "native_pointer": "/folder_path",
                                "path": view_root,
                                "input_ids": [first["input_id"], second["input_id"]],
                            }],
                            "repeat": 1,
                            "bindings": [
                                {
                                    "rule": "same-relative-stem",
                                    "key": "000000",
                                    "members": [{"input_id": first["input_id"], "root": first_root}],
                                },
                                {
                                    "rule": "same-relative-stem",
                                    "key": "000001",
                                    "members": [{"input_id": second["input_id"], "root": second_root}],
                                },
                            ],
                        }],
                    }],
                }

            with self.assertRaisesRegex(ValueError, "input kind .* uses multiple role roots"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=inconsistent_roots,
                )

    def test_projection_can_explicitly_bind_one_captionless_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            dataset = workspace / "datasets" / "tiny"
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["caption"] = None
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            (dataset / "a.txt").unlink()

            def singleton(selection: dict) -> dict:
                target = selection["datasets"][0]["samples"][0]["files"][0]
                view_root = "runs/example/cache/dataset-view/ai-toolkit/tiny"
                semantic = {"caption_ext": ".txt"}
                runtime = {"folder_path": "/workspace/" + view_root}
                return {
                    "schema_version": 1,
                    "backend": "ai-toolkit",
                    "datasets": [{
                        "id": "tiny",
                        "consumed": [target["input_id"]],
                        "unrepresentable": [],
                        "semantic": semantic,
                        "native_runtime": runtime,
                        "native": {**semantic, **runtime},
                        "native_string_fields": ["/caption_ext"],
                        "views": [{
                            "id": "primary",
                            "root": view_root,
                            "links": [{
                                "path": view_root + "/000000.png",
                                "target": "/workspace/datasets/tiny/a.png",
                                "input_id": target["input_id"],
                            }],
                            "files": [],
                            "native_files": [],
                            "write_roots": [{"path": view_root, "native_pointer": "/folder_path"}],
                            "consumers": [{
                                "id": "dataset",
                                "kind": "recursive-directory",
                                "native_pointer": "/folder_path",
                                "path": view_root,
                                "input_ids": [target["input_id"]],
                            }],
                            "repeat": 1,
                            "bindings": [{
                                "rule": "same-relative-stem",
                                "key": "000000",
                                "members": [{"input_id": target["input_id"], "root": view_root}],
                            }],
                        }],
                    }],
                }

            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=singleton,
            )
            self.assertEqual(len(lock["files"]), 1)

    def test_projection_accepts_multiple_uniquely_identified_native_views(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_two_captionless_samples(workspace)

            def multiple_views(selection: dict) -> dict:
                samples = selection["datasets"][0]["samples"]
                view_base = "runs/example/cache/dataset-view/test/tiny"
                views = []
                native_paths = []
                consumed = []
                for index, sample in enumerate(samples):
                    input_item = sample["files"][0]
                    root = f"{view_base}/sample-{index}"
                    native_paths.append({"folder_path": "/workspace/" + root})
                    consumed.append(input_item["input_id"])
                    views.append({
                        "id": f"sample-{index}",
                        "root": root,
                        "links": [{
                            "path": f"{root}/000000{Path(input_item['path']).suffix}",
                            "target": f"/workspace/datasets/tiny/{input_item['path']}",
                            "input_id": input_item["input_id"],
                        }],
                        "files": [],
                        "native_files": [],
                        "write_roots": [{
                            "path": root,
                            "native_pointer": f"/datasets/{index}/folder_path",
                        }],
                        "consumers": [{
                            "id": "dataset",
                            "kind": "recursive-directory",
                            "native_pointer": f"/datasets/{index}/folder_path",
                            "path": root,
                            "input_ids": [input_item["input_id"]],
                        }],
                        "repeat": 1,
                        "bindings": [{
                            "rule": "same-relative-stem",
                            "key": "000000",
                            "members": [{"input_id": input_item["input_id"], "root": root}],
                        }],
                    })
                semantic = {"format": "test"}
                runtime = {"datasets": native_paths}
                return {
                    "schema_version": 1,
                    "backend": "ai-toolkit",
                    "datasets": [{
                        "id": "tiny",
                        "consumed": consumed,
                        "unrepresentable": [],
                        "semantic": semantic,
                        "native_runtime": runtime,
                        "native": {**semantic, **runtime},
                        "native_string_fields": ["/format"],
                        "views": views,
                    }],
                }

            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=multiple_views,
            )
            materialize_dataset_view(workspace, lock)

            self.assertEqual([view["id"] for view in lock["views"]], ["sample-0", "sample-1"])
            self.assertEqual(inspect_dataset_handoff(workspace, lock), [])
            for view in lock["views"]:
                self.assertTrue((workspace / view["links"][0]["path"]).is_symlink())

    def test_projection_rejects_duplicate_native_view_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def duplicate_view(selection: dict) -> dict:
                projection = self.c3_image_projection(selection)
                duplicate = json.loads(json.dumps(projection["datasets"][0]["views"][0]))
                projection["datasets"][0]["views"].append(duplicate)
                projection["datasets"][0]["consumed"] *= 2
                return projection

            with self.assertRaisesRegex(ValueError, "duplicate.*view"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=duplicate_view,
                )

    def test_projection_rejects_native_consumer_path_that_differs_from_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def mismatched_consumer(selection: dict) -> dict:
                projection = self.c3_image_projection(selection)
                projection["datasets"][0]["native_runtime"]["folder_path"] = "/workspace/elsewhere"
                projection["datasets"][0]["native"]["folder_path"] = "/workspace/elsewhere"
                return projection

            with self.assertRaisesRegex(ValueError, "native (?:consumer|write root).*(?:view|declared path)"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=mismatched_consumer,
                )

    def test_recursive_consumer_rejects_nested_role_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def nested_roles(selection: dict) -> dict:
                projection = self.c3_image_projection(selection)
                view = projection["datasets"][0]["views"][0]
                caption = view["files"][0]
                caption["path"] = view["root"] + "/captions/000000.txt"
                view["bindings"][0]["members"][1]["root"] = view["root"] + "/captions"
                return projection

            with self.assertRaisesRegex(ValueError, "nested role roots"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=nested_roles,
                )

    def test_jsonl_consumer_uses_row_references_instead_of_directory_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=self.c3_jsonl_projection,
            )

            self.assertEqual(lock["views"][0]["consumers"][0]["kind"], "jsonl")
            self.assertNotIn("bindings", lock["views"][0])

    def test_jsonl_consumer_rejects_a_second_directory_binding_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def duplicate_contract(selection: dict) -> dict:
                projection = self.c3_jsonl_projection(selection)
                projection["datasets"][0]["views"][0]["bindings"] = []
                return projection

            with self.assertRaisesRegex(ValueError, "JSONL consumer.*bindings"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=duplicate_contract,
                )

    def test_view_repeat_is_part_of_input_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            first = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=self.c3_image_projection,
            )
            other = {**run, "id": "other"}
            other_resolved = workspace / "runs" / "other" / "resolved"

            def repeated(selection: dict) -> dict:
                projection = self.c3_image_projection(selection)
                dataset = projection["datasets"][0]
                dataset["views"][0]["repeat"] = 2
                dataset["views"][0]["repeat_pointer"] = "/num_repeats"
                dataset["native_runtime"]["num_repeats"] = 2
                dataset["native"]["num_repeats"] = 2
                return projection

            second = freeze_dataset_handoff(
                other, workspace, other_resolved, backend="ai-toolkit", project=repeated,
            )

            self.assertNotEqual(first["input_sha256"], second["input_sha256"])

    def test_repeated_view_requires_a_matching_native_repeat_setting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def unbound_repeat(selection: dict) -> dict:
                projection = self.c3_image_projection(selection)
                projection["datasets"][0]["views"][0]["repeat"] = 2
                return projection

            with self.assertRaisesRegex(ValueError, "repeat has no native pointer"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=unbound_repeat,
                )

    def test_projection_native_must_be_derived_from_semantic_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def divergent(selection: dict) -> dict:
                projection = self.image_projection(selection)
                projection["datasets"][0]["native"]["caption_ext"] = ".caption"
                return projection

            with self.assertRaisesRegex(ValueError, "derived exactly from semantic"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=divergent,
                )

    def test_projection_rejects_unclassified_native_path_string(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def undeclared_path(selection: dict) -> dict:
                projection = self.c3_image_projection(selection)
                dataset = projection["datasets"][0]
                dataset["native_runtime"]["control_directory"] = "/workspace/unreported-control"
                dataset["native"]["control_directory"] = "/workspace/unreported-control"
                return projection

            with self.assertRaisesRegex(ValueError, "native path-like string.*control_directory.*not classified"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=undeclared_path,
                )

    def test_projection_rejects_path_disguised_as_declared_native_string(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def disguised_path(selection: dict) -> dict:
                projection = self.c3_image_projection(selection)
                dataset = projection["datasets"][0]
                dataset["native_runtime"]["control_directory"] = "/workspace/datasets/Vivi/control"
                dataset["native"]["control_directory"] = "/workspace/datasets/Vivi/control"
                dataset["native_string_fields"].append("/control_directory")
                return projection

            with self.assertRaisesRegex(ValueError, "native string field.*control_directory.*path-like"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=disguised_path,
                )

    def test_projection_rejects_unclassified_native_non_path_string(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def undeclared_string(selection: dict) -> dict:
                projection = self.c3_image_projection(selection)
                dataset = projection["datasets"][0]
                dataset["semantic"]["mode"] = "ordinary"
                dataset["native"]["mode"] = "ordinary"
                return projection

            with self.assertRaisesRegex(ValueError, "native string.*mode.*not classified"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=undeclared_string,
                )

    def test_projection_write_root_must_match_its_native_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def mismatched_write_root(selection: dict) -> dict:
                projection = self.c3_image_projection(selection)
                dataset = projection["datasets"][0]
                view = dataset["views"][0]
                view["write_roots"][0]["path"] = view["root"] + "/cache"
                return projection

            with self.assertRaisesRegex(ValueError, "native write root.*declared path"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=mismatched_write_root,
                )

    def test_projection_reports_unrepresentable_input_with_sample_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def rejects_target(selection: dict) -> dict:
                projection = self.image_projection(selection)
                target = selection["datasets"][0]["samples"][0]["files"][0]
                projection["datasets"][0]["unrepresentable"] = [{
                    "input_id": target["input_id"], "reason": "video is unsupported in image mode",
                }]
                return projection

            with self.assertRaisesRegex(ValueError, "tiny.*a.*video is unsupported"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=rejects_target,
                )

    def test_core_rejects_a_projection_that_retargets_a_consumed_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def dishonest(selection: dict) -> dict:
                projection = self.image_projection(selection)
                projection["datasets"][0]["views"][0]["links"][0]["target"] = (
                    "/workspace/datasets/tiny/other.png"
                )
                return projection

            with self.assertRaisesRegex(ValueError, "invalid input or target"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=dishonest,
                )

    def test_core_rejects_a_projection_that_changes_caption_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def dishonest(selection: dict) -> dict:
                projection = self.image_projection(selection)
                projection["datasets"][0]["views"][0]["files"][0]["text"] = "different caption"
                return projection

            with self.assertRaisesRegex(ValueError, "does not preserve its caption input"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=dishonest,
                )

    def test_materialize_checks_stat_and_exact_link_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=self.image_projection,
            )

            view = materialize_dataset_view(workspace, lock)

            link = view / "000000.png"
            self.assertTrue(link.is_symlink())
            self.assertEqual(link.readlink().as_posix(), "/workspace/datasets/tiny/a.png")
            self.assertEqual((view / "000000.txt").read_text(encoding="utf-8"), "caption\n")
            self.assertEqual(inspect_dataset_handoff(workspace, lock), [])
            link.unlink()
            link.symlink_to("/workspace/datasets/tiny/other.png")
            self.assertIn("retargeted view link", " ".join(inspect_dataset_handoff(workspace, lock)))

    def test_generated_caption_preserves_crlf_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            (workspace / "datasets" / "tiny" / "a.txt").write_bytes(b"first\r\nsecond\r\n")
            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run, selection),
            )

            view = materialize_dataset_view(workspace, lock)

            caption = next(view.rglob("*.txt"))
            self.assertEqual(caption.read_bytes(), b"first\r\nsecond\r\n")
            self.assertEqual(inspect_dataset_handoff(workspace, lock), [])

    def test_changed_authoring_manifest_stops_launch_stat_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=self.image_projection,
            )
            items = workspace / "datasets" / "tiny" / "items.jsonl"
            items.write_text(items.read_text(encoding="utf-8") + "\n", encoding="utf-8")

            self.assertIn("items.jsonl", " ".join(inspect_dataset_sources(workspace, lock)))

    def test_unexpected_regular_media_in_view_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=self.image_projection,
            )
            view = materialize_dataset_view(workspace, lock)
            (view / "stale.png").write_bytes(b"stale")

            self.assertIn("unexpected regular media", " ".join(inspect_dataset_handoff(workspace, lock)))

    def test_changed_source_stops_before_view_materialization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=self.image_projection,
            )
            (workspace / "datasets" / "tiny" / "a.png").write_bytes(b"changed")

            with self.assertRaisesRegex(ValueError, "changed since compile"):
                materialize_dataset_view(workspace, lock)
            self.assertFalse((workspace / "runs" / "example" / "cache" / "dataset-view").exists())

    def test_external_dataset_root_is_frozen_and_retargeting_stops_launch(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external_directory:
            workspace = Path(directory)
            external = Path(external_directory)
            run, resolved = self.make_run(external)
            (workspace / "datasets").mkdir()
            (workspace / "datasets" / "tiny").symlink_to(external / "datasets" / "tiny")
            run["id"] = "example"
            resolved = workspace / "runs" / "example" / "resolved"
            resolved.mkdir(parents=True)

            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=self.image_projection,
            )

            self.assertEqual(lock["dataset_roots"][0]["physical"], str((external / "datasets" / "tiny").resolve()))
            self.assertEqual(inspect_dataset_handoff(workspace, lock), [
                "missing view link: runs/example/cache/dataset-view/ai-toolkit/tiny/000000.png",
                "missing generated view file: runs/example/cache/dataset-view/ai-toolkit/tiny/000000.txt",
            ])
            replacement = external / "replacement"
            replacement.mkdir()
            link = workspace / "datasets" / "tiny"
            link.unlink()
            link.symlink_to(replacement)
            with self.assertRaisesRegex(ValueError, "dataset root changed"):
                materialize_dataset_view(workspace, lock)

    def test_absolute_symlink_inside_dataset_is_not_container_portable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            dataset = workspace / "datasets" / "tiny"
            real = dataset / "real.png"
            (dataset / "a.png").replace(real)
            (dataset / "a.png").symlink_to(real.resolve())

            with self.assertRaisesRegex(ValueError, "absolute symlinks.*not container-portable"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=self.image_projection,
                )

    def test_relative_symlink_chain_cannot_end_at_an_absolute_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            dataset = workspace / "datasets" / "tiny"
            real = dataset / "real.png"
            (dataset / "a.png").replace(real)
            absolute = dataset / "absolute.png"
            absolute.symlink_to(real.resolve())
            (dataset / "a.png").symlink_to("absolute.png")

            with self.assertRaisesRegex(ValueError, "absolute symlinks.*not container-portable"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=self.image_projection,
                )

    def test_ai_toolkit_projects_one_image_caption_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run, selection),
            )

            report = json.loads((resolved / "dataset-projection.lock.json").read_text(encoding="utf-8"))
            projected = report["datasets"][0]
            self.assertEqual(projected["native"], {
                "folder_path": "/workspace/runs/example/cache/dataset-view/ai-toolkit/tiny",
                "caption_ext": ".txt",
                "cache_latents_to_disk": True,
            })
            self.assertEqual(projected["unrepresentable"], [])
            self.assertEqual(len(lock["views"][0]["links"]), 1)
            target_name = Path(lock["views"][0]["links"][0]["path"]).stem
            caption_name = Path(lock["views"][0]["files"][0]["path"]).stem
            self.assertEqual(target_name, caption_name)
            self.assertRegex(target_name, r"^000000-[0-9a-f]{12}$")

    def test_musubi_projects_one_image_caption_dataset_with_sibling_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux2"}}

            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            materialize_dataset_view(workspace, lock)
            _write_musubi_dataset_config(
                run,
                resolved / "musubi" / "dataset.toml",
                workspace=workspace,
                strict=True,
            )

            report = json.loads((resolved / "dataset-projection.lock.json").read_text(encoding="utf-8"))
            projected = report["datasets"][0]
            view = lock["views"][0]
            image_jsonl = Path(projected["native"]["image_jsonl_file"])
            cache_directory = Path(projected["native"]["cache_directory"])
            self.assertEqual(image_jsonl.parent.parent, cache_directory.parent)
            self.assertNotEqual(image_jsonl.parent, cache_directory)
            self.assertEqual(view["consumers"][0]["native_pointer"], "/image_jsonl_file")
            self.assertEqual(view["consumers"][0]["kind"], "jsonl")
            self.assertEqual(view["write_roots"][0]["native_pointer"], "/cache_directory")
            self.assertNotIn("caption_extension", projected["native"])
            self.assertEqual(projected["policy"]["profile"], "ordinary-image")
            self.assertEqual(projected["policy"]["codec"], "plain-image-jsonl")
            self.assertEqual(projected["policy"]["caption_transform"], "strip")
            self.assertEqual(projected["policy"]["audio_selection"], "unsupported")
            self.assertEqual(lock["semantic"]["projection"][0]["policy"], projected["policy"])
            generated = json.loads(view["native_files"][0]["text"])
            self.assertEqual(generated["caption"], "caption")
            parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
            self.assertEqual(parsed["datasets"], [projected["native"]])

    def test_musubi_profile_table_rejects_an_unlisted_ordinary_image_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "flux2",
                "one_frame": True,
            }}

            with self.assertRaisesRegex(
                ValueError,
                "no verified Musubi projection profile matches.*flux2.*shape='image'.*one_frame.*True",
            ):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_profile_mismatch_names_minority_shape_sample_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            dataset = workspace / "datasets" / "tiny"
            (dataset / "b.png").write_bytes(b"second image")
            (dataset / "b.txt").write_text("second caption\n", encoding="utf-8")
            (dataset / "minority.mp4").write_bytes(b"video")
            (dataset / "minority.txt").write_text("video caption\n", encoding="utf-8")
            rows = [
                {
                    "id": name,
                    "files": [{"type": "file", "role": "target", "path": path}],
                    "caption": {"file": {"type": "file", "path": caption}},
                }
                for name, path, caption in (
                    ("a", "a.png", "a.txt"),
                    ("b", "b.png", "b.txt"),
                    ("minority-video", "minority.mp4", "minority.txt"),
                )
            ]
            (dataset / "items.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
            )
            run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux2"}}

            with self.assertRaisesRegex(
                ValueError,
                "no verified Musubi projection profile matches.*minority sample IDs.*minority-video",
            ):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_h3_video_codecs_restore_t2va_fl2va_and_explicit_target_audio(self) -> None:
        for task in ("t2va", "fl2va"):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, resolved = self.make_run(workspace)
                self.write_tiny_manifest(
                    workspace,
                    [{
                        "id": "video-audio",
                        "files": [
                            {"type": "file", "role": "target", "path": "clip.mp4"},
                            {"type": "file", "role": "audio", "path": "clip.wav"},
                        ],
                        "caption": {"text": "  caption  "},
                    }],
                    {"clip.mp4": b"video", "clip.wav": b"audio"},
                )
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "minimax_h3",
                    "task": task,
                    "dataset_options": {"tiny": {"target_frames": [22]}},
                }}

                lock = freeze_dataset_handoff(
                    run, workspace, resolved, backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )
                materialize_dataset_view(workspace, lock)
                _write_musubi_dataset_config(
                    run, resolved / "musubi" / "dataset.toml", workspace=workspace, strict=True,
                )

                projection = lock["semantic"]["projection"][0]
                row = projection["views"][0]["native_jsonl"][0]["rows"][0]["value"]
                self.assertEqual(row["caption"]["$kura_kind"], "caption-text-strip")
                self.assertIn("inputs/audio/", row["audio_path"]["$kura_view_path"])
                self.assertEqual(
                    projection["policy"]["audio_selection"],
                    "explicit-role-else-preflight-rejects-resolved-sidecar-then-embedded-or-silence",
                )
                self.assertEqual(projection["policy"]["profile"], f"h3-video-{task}")
                parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
                frozen = json.loads((resolved / "dataset-projection.lock.json").read_text(encoding="utf-8"))["datasets"][0]
                self.assertEqual(parsed["datasets"], [frozen["native"]])

    def test_musubi_h3_ref2va_codec_preserves_ordered_references_and_audio_choices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            self.write_tiny_manifest(
                workspace,
                [
                    {
                        "id": "ordered",
                        "files": [
                            {"type": "file", "role": "target", "path": "target.mp4"},
                            {"type": "file", "role": "reference", "path": "face.png"},
                            {"type": "file", "role": "reference", "path": "motion.mp4"},
                            {"type": "file", "role": "reference-audio", "path": "motion.wav"},
                            {"type": "file", "role": "reference-muted", "path": "silent.mp4"},
                            {"type": "file", "role": "reference", "path": "voice.wav"},
                        ],
                        "caption": {"text": "subject"},
                    },
                    {
                        "id": "shorter-reference-list",
                        "files": [
                            {"type": "file", "role": "target", "path": "other.mp4"},
                            {"type": "file", "role": "reference", "path": "other.png"},
                        ],
                        "caption": {"text": "other"},
                    },
                ],
                {
                    "target.mp4": b"target", "face.png": b"image", "motion.mp4": b"motion",
                    "motion.wav": b"motion audio", "silent.mp4": b"silent", "voice.wav": b"voice",
                    "other.mp4": b"other target", "other.png": b"other reference",
                },
            )
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "minimax_h3",
                "task": "ref2va",
                "dataset_options": {"tiny": {"target_frames": [22]}},
            }}

            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            materialize_dataset_view(workspace, lock)
            _write_musubi_dataset_config(
                run, resolved / "musubi" / "dataset.toml", workspace=workspace, strict=True,
            )

            projection = lock["semantic"]["projection"][0]
            row = projection["views"][0]["native_jsonl"][0]["rows"][0]["value"]
            references = row["references"]
            self.assertEqual([item["type"] for item in references], ["image", "video", "video", "audio"])
            self.assertIn("audio_path", references[1])
            self.assertIsNone(references[2]["audio_path"])
            self.assertIn("inputs/reference/", references[3]["path"]["$kura_view_path"])
            parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
            frozen = json.loads((resolved / "dataset-projection.lock.json").read_text(encoding="utf-8"))["datasets"][0]
            self.assertEqual(parsed["datasets"], [frozen["native"]])

    def test_musubi_h3_one_frame_codecs_restore_plain_timed_and_ordered_reference_inputs(self) -> None:
        cases = (
            ("plain", "t2va", [{"type": "file", "role": "target", "path": "target.png"}], {}, "h3-one-frame-plain"),
            (
                "timed", "fl2va",
                [
                    {"type": "file", "role": "target", "path": "target.png"},
                    {"type": "file", "role": "control", "path": "control.png"},
                ],
                {"fp_1f_clean_indices": [0], "fp_1f_target_index": 24},
                "h3-one-frame-fl2va",
            ),
            (
                "reference", "ref2va",
                [
                    {"type": "file", "role": "target", "path": "target.png"},
                    {"type": "file", "role": "reference", "path": "reference.png"},
                ],
                {},
                "h3-one-frame-ref2va",
            ),
        )
        for name, task, files, options, expected_profile in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, resolved = self.make_run(workspace)
                payloads = {str(item["path"]): b"media" for item in files}
                self.write_tiny_manifest(
                    workspace,
                    [{"id": name, "files": files, "caption": {"text": "caption"}}],
                    payloads,
                )
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "minimax_h3", "task": task, "one_frame": True,
                    "dataset_options": {"tiny": options},
                }}

                lock = freeze_dataset_handoff(
                    run, workspace, resolved, backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )
                materialize_dataset_view(workspace, lock)
                _write_musubi_dataset_config(
                    run, resolved / "musubi" / "dataset.toml", workspace=workspace, strict=True,
                )

                projection = lock["semantic"]["projection"][0]
                self.assertEqual(projection["policy"]["profile"], expected_profile)
                parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
                frozen = json.loads((resolved / "dataset-projection.lock.json").read_text(encoding="utf-8"))["datasets"][0]
                self.assertEqual(parsed["datasets"], [frozen["native"]])

    def test_musubi_h3_teacher_matching_selects_the_profile_for_each_teacher_condition(self) -> None:
        cases = (
            ("first,last", False, "video", "h3-video-teacher-endpoints"),
            ("ref", False, "video", "h3-video-teacher-ref"),
            ("subject_ref", True, "image-reference", "h3-one-frame-teacher-subject-ref"),
        )
        for teacher_conditions, one_frame, shape, expected_profile in cases:
            with self.subTest(teacher_conditions=teacher_conditions), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, resolved = self.make_run(workspace)
                suffix = ".png" if one_frame else ".mp4"
                files = [{"type": "file", "role": "target", "path": "target" + suffix}]
                payloads = {"target" + suffix: b"target"}
                options = {} if one_frame else {"target_frames": [22]}
                if shape == "image-reference":
                    files.append({"type": "file", "role": "reference", "path": "reference.png"})
                    payloads["reference.png"] = b"reference"
                self.write_tiny_manifest(
                    workspace,
                    [{"id": "teacher", "files": files, "caption": {"text": "caption"}}],
                    payloads,
                )
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "minimax_h3", "task": "t2va", "one_frame": one_frame,
                    "h3_loss_method": "teacher_matching", "h3_teacher_conditions": teacher_conditions,
                    "dataset_options": {"tiny": options},
                }}

                lock = freeze_dataset_handoff(
                    run, workspace, resolved, backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

                materialize_dataset_view(workspace, lock)
                _write_musubi_dataset_config(
                    run, resolved / "musubi" / "dataset.toml", workspace=workspace, strict=True,
                )
                projection = lock["semantic"]["projection"][0]
                self.assertEqual(projection["policy"]["profile"], expected_profile)
                parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
                frozen = json.loads((resolved / "dataset-projection.lock.json").read_text(encoding="utf-8"))["datasets"][0]
                self.assertEqual(parsed["datasets"], [frozen["native"]])

    def test_musubi_h3_video_rejects_frames_outside_the_5_plus_17n_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            self.write_tiny_manifest(
                workspace,
                [{
                    "id": "video",
                    "files": [{"type": "file", "role": "target", "path": "clip.mp4"}],
                    "caption": {"text": "caption"},
                }],
                {"clip.mp4": b"video"},
            )
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "minimax_h3", "task": "t2va",
                "dataset_options": {"tiny": {"target_frames": [21]}},
            }}

            with self.assertRaisesRegex(ValueError, r"5\+17n grid.*21"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_h3_video_rejects_source_fps_that_the_pinned_loader_ignores(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            self.write_tiny_manifest(
                workspace,
                [{
                    "id": "video",
                    "files": [{"type": "file", "role": "target", "path": "clip.mp4"}],
                    "caption": {"text": "caption"},
                }],
                {"clip.mp4": b"video"},
            )
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "minimax_h3", "task": "t2va",
                "dataset_options": {"tiny": {"target_frames": [124], "source_fps": 60.0}},
            }}

            with self.assertRaisesRegex(ValueError, "h3-video-t2va does not accept.*source_fps"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_h3_ref2va_accepts_a_muted_video_as_the_only_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            self.write_tiny_manifest(
                workspace,
                [{
                    "id": "muted",
                    "files": [
                        {"type": "file", "role": "target", "path": "target.mp4"},
                        {"type": "file", "role": "reference-muted", "path": "reference.mp4"},
                    ],
                    "caption": {"text": "caption"},
                }],
                {"target.mp4": b"target", "reference.mp4": b"reference"},
            )
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "minimax_h3", "task": "ref2va",
                "dataset_options": {"tiny": {"target_frames": [22]}},
            }}

            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )

            row = lock["semantic"]["projection"][0]["views"][0]["native_jsonl"][0]["rows"][0]["value"]
            self.assertEqual(row["references"][0]["type"], "video")
            self.assertIsNone(row["references"][0]["audio_path"])

    def test_musubi_h3_reference_audio_must_follow_an_unmuted_video_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            self.write_tiny_manifest(
                workspace,
                [{
                    "id": "bad-order",
                    "files": [
                        {"type": "file", "role": "target", "path": "target.mp4"},
                        {"type": "file", "role": "reference-audio", "path": "reference.wav"},
                        {"type": "file", "role": "reference", "path": "reference.mp4"},
                    ],
                    "caption": {"text": "caption"},
                }],
                {"target.mp4": b"target", "reference.mp4": b"reference", "reference.wav": b"audio"},
            )
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "minimax_h3", "task": "ref2va",
                "dataset_options": {"tiny": {"target_frames": [22]}},
            }}

            with self.assertRaisesRegex(ValueError, "reference-audio must immediately follow"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_h3_subject_reference_teacher_rejects_video_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            self.write_tiny_manifest(
                workspace,
                [{
                    "id": "subject",
                    "files": [
                        {"type": "file", "role": "target", "path": "target.png"},
                        {"type": "file", "role": "reference", "path": "reference.mp4"},
                    ],
                    "caption": {"text": "caption"},
                }],
                {"target.png": b"target", "reference.mp4": b"reference"},
            )
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "minimax_h3", "task": "t2va", "one_frame": True,
                "h3_loss_method": "teacher_matching", "h3_teacher_conditions": "subject_ref",
            }}

            with self.assertRaisesRegex(ValueError, "subject-reference teacher accepts image references only"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_dataset_toml_accepts_only_a_verified_jsonl_consumer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux2"}}
            freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            projection_path = resolved / "dataset-projection.lock.json"
            projection = json.loads(projection_path.read_text(encoding="utf-8"))
            dataset = projection["datasets"][0]
            dataset["native"].pop("image_jsonl_file")
            dataset["native"]["image_directory"] = "/workspace/datasets/tiny"
            consumer = dataset["views"][0]["consumers"][0]
            consumer.update({"kind": "directory", "native_pointer": "/image_directory"})
            projection_path.write_text(json.dumps(projection), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "bypasses its verified source"):
                _write_musubi_dataset_config(
                    run,
                    resolved / "musubi" / "dataset.toml",
                    workspace=workspace,
                    strict=True,
                )

    def test_musubi_caption_strip_is_visible_and_manifest_text_remains_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first_run, first_resolved = self.make_run(workspace)
            first_run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux2"}}
            caption = workspace / "datasets" / "tiny" / "a.txt"
            caption.write_text("  caption\r\n", encoding="utf-8", newline="")
            first = freeze_dataset_handoff(
                first_run, workspace, first_resolved, backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(first_run, selection),
            )
            first_row = json.loads(first["views"][0]["native_files"][0]["text"])
            self.assertEqual(first_row["caption"], "caption")
            self.assertEqual(first["semantic"]["projection"][0]["policy"]["caption_transform"], "strip")

            caption.write_text("caption\n", encoding="utf-8")
            second_run = deepcopy(first_run)
            second_run["id"] = "caption-without-padding"
            second_resolved = workspace / "runs" / second_run["id"] / "resolved"
            second = freeze_dataset_handoff(
                second_run, workspace, second_resolved, backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(second_run, selection),
            )
            second_row = json.loads(second["views"][0]["native_files"][0]["text"])
            self.assertEqual(second_row["caption"], "caption")
            self.assertNotEqual(first["input_sha256"], second["input_sha256"])

    def test_musubi_projects_flux_kontext_control_to_a_separate_bound_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux_kontext"}}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "control").mkdir()
            (dataset / "control" / "a.webp").write_bytes(b"control")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"].append({"type": "file", "role": "control", "path": "control/a.webp"})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            materialize_dataset_view(workspace, lock)
            _write_musubi_dataset_config(
                run,
                resolved / "musubi" / "dataset.toml",
                workspace=workspace,
                strict=True,
            )

            report = json.loads((resolved / "dataset-projection.lock.json").read_text(encoding="utf-8"))
            projected = report["datasets"][0]
            native = projected["native"]
            view = lock["views"][0]
            self.assertEqual(native["image_jsonl_file"], "/workspace/" + view["native_files"][0]["path"])
            self.assertEqual({item["native_pointer"] for item in view["consumers"]}, {"/image_jsonl_file"})
            image_link = next(item for item in view["links"] if "/images/" in item["path"])
            control_link = next(item for item in view["links"] if "/controls/" in item["path"])
            generated = json.loads(view["native_files"][0]["text"])
            self.assertEqual(generated["image_path"], "/workspace/" + image_link["path"])
            self.assertEqual(generated["control_path"], "/workspace/" + control_link["path"])
            self.assertEqual(projected["policy"]["codec"], "image-control-jsonl")
            self.assertEqual(projected["policy"]["profile"], "flux-kontext-control")
            self.assertNotIn("bindings", view)
            parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
            self.assertEqual(parsed["datasets"], [native])

    def test_musubi_separate_control_stops_for_an_unverified_architecture(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "qwen_image"}}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "control.png").write_bytes(b"control")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"].append({"type": "file", "role": "control", "path": "control.png"})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "no verified Musubi projection profile matches.*qwen_image.*image-control",
            ):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_control_content_changes_the_projected_pair_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            first_run, first_resolved = self.make_run(workspace)
            first_run["backend"] = {
                "name": "musubi-tuner",
                "config": {"architecture": "flux_kontext"},
            }
            dataset = workspace / "datasets" / "tiny"
            (dataset / "control.png").write_bytes(b"first control")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"].append({"type": "file", "role": "control", "path": "control.png"})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            first_lock = freeze_dataset_handoff(
                first_run,
                workspace,
                first_resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(first_run, selection),
            )
            first_view = first_lock["views"][0]
            first_image = next(item for item in first_view["links"] if "/images/" in item["path"])
            first_control = next(item for item in first_view["links"] if "/controls/" in item["path"])

            (dataset / "control.png").write_bytes(b"second control")
            second_run = deepcopy(first_run)
            second_run["id"] = "changed-control"
            second_resolved = workspace / "runs" / "changed-control" / "resolved"
            second_resolved.mkdir(parents=True)
            second_lock = freeze_dataset_handoff(
                second_run,
                workspace,
                second_resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(second_run, selection),
            )
            second_view = second_lock["views"][0]
            second_image = next(item for item in second_view["links"] if "/images/" in item["path"])
            second_control = next(item for item in second_view["links"] if "/controls/" in item["path"])

            self.assertEqual(Path(first_image["path"]).stem, Path(first_control["path"]).stem)
            self.assertEqual(Path(second_image["path"]).stem, Path(second_control["path"]).stem)
            self.assertNotEqual(Path(first_image["path"]).stem, Path(second_image["path"]).stem)

    def test_musubi_separate_control_rejects_multiple_controls_until_verified(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux1_kontext"}}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "control-1.png").write_bytes(b"control 1")
            (dataset / "control-2.png").write_bytes(b"control 2")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"].extend([
                {"type": "file", "role": "control", "path": "control-1.png"},
                {"type": "file", "role": "control", "path": "control-2.png"},
            ])
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "no verified Musubi projection profile matches.*flux1_kontext.*image-control.*control.*2",
            ):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_separate_control_requires_one_control_for_every_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux_kontext"}}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "control-a.png").write_bytes(b"control a")
            (dataset / "b.png").write_bytes(b"image b")
            (dataset / "b.txt").write_text("caption b\n", encoding="utf-8")
            first = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            first["files"].append({"type": "file", "role": "control", "path": "control-a.png"})
            second = {
                "id": "b",
                "files": [{"type": "file", "role": "target", "path": "b.png"}],
                "caption": {"file": {"type": "file", "path": "b.txt"}},
            }
            (dataset / "items.jsonl").write_text(
                json.dumps(first) + "\n" + json.dumps(second) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "no verified Musubi projection profile matches.*flux_kontext.*mixed:image,image-control",
            ):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_kontext_requires_a_control_for_every_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux_kontext"}}

            with self.assertRaisesRegex(
                ValueError,
                "no verified Musubi projection profile matches.*flux_kontext.*shape='image'",
            ):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_control_resize_options_are_typed_semantic_and_native(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "flux_kontext",
                "dataset_options": {
                    "tiny": {"control_resolution": [768, 512], "no_resize_control": True},
                },
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "control.png").write_bytes(b"control")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"].append({"type": "file", "role": "control", "path": "control.png"})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            _write_musubi_dataset_config(
                run,
                resolved / "musubi" / "dataset.toml",
                workspace=workspace,
                strict=True,
            )

            report = json.loads((resolved / "dataset-projection.lock.json").read_text(encoding="utf-8"))
            projected = report["datasets"][0]
            self.assertEqual(projected["semantic"]["control_resolution"], [768, 512])
            self.assertIs(projected["semantic"]["no_resize_control"], True)
            self.assertEqual(projected["native"]["control_resolution"], [768, 512])
            self.assertIs(projected["native"]["no_resize_control"], True)
            self.assertEqual(lock["semantic"]["projection"][0]["semantic"]["control_resolution"], [768, 512])
            parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
            self.assertEqual(parsed["datasets"], [projected["native"]])

            changed_run = deepcopy(run)
            changed_run["id"] = "changed-control-resolution"
            changed_run["backend"]["config"]["dataset_options"]["tiny"]["control_resolution"] = [1024, 1024]
            changed_resolved = workspace / "runs" / changed_run["id"] / "resolved"
            changed_lock = freeze_dataset_handoff(
                changed_run,
                workspace,
                changed_resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(changed_run, selection),
            )
            self.assertNotEqual(lock["input_sha256"], changed_lock["input_sha256"])

    def test_musubi_rejects_multiple_control_resolution_blocks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "flux_kontext",
                "dataset_options": {
                    "tiny": {"control_resolution": [[768, 768], [1024, 1024]]},
                },
            }}

            with self.assertRaisesRegex(ValueError, "multiple control-resolution blocks.*not yet supported"):
                project_musubi_dataset(run, {"datasets": []})

    def test_musubi_rejects_invalid_control_resize_options(self) -> None:
        cases = (
            ({"control_resolution": [768]}, "control_resolution.*two positive integers"),
            ({"control_resolution": [768, 0]}, "control_resolution.*two positive integers"),
            ({"control_resolution": [768, True]}, "control_resolution.*two positive integers"),
            ({"no_resize_control": 1}, "no_resize_control must be boolean"),
        )
        for options, message in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, _resolved = self.make_run(workspace)
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "flux_kontext",
                    "dataset_options": {"tiny": options},
                }}
                with self.assertRaisesRegex(ValueError, message):
                    project_musubi_dataset(run, {"datasets": []})

    def test_musubi_resource_resolution_includes_typed_control_resolution(self) -> None:
        run = {
            "id": "control-resolution",
            "datasets": [{"id": "tiny"}],
            "backend": {"name": "musubi-tuner", "config": {
                "architecture": "flux_kontext",
                "resolution": [512, 512],
                "dataset_options": {"tiny": {"control_resolution": [768, 1024]}},
            }},
        }

        self.assertEqual(_musubi_max_resolution(run, run["backend"]["config"]), 1024)

    def test_musubi_projects_minimax_h3_one_frame_fl2va_control_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "minimax_h3",
                "task": "fl2va",
                "one_frame": True,
                "dataset_options": {
                    "tiny": {"fp_1f_clean_indices": [0], "fp_1f_target_index": 24},
                },
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "source.png").write_bytes(b"control")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"].append({"type": "file", "role": "control", "path": "source.png"})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            materialize_dataset_view(workspace, lock)
            _write_musubi_dataset_config(
                run,
                resolved / "musubi" / "dataset.toml",
                workspace=workspace,
                strict=True,
            )

            report = json.loads((resolved / "dataset-projection.lock.json").read_text(encoding="utf-8"))
            projected = report["datasets"][0]
            view = lock["views"][0]
            native_file = view["native_files"][0]
            row_value = json.loads(native_file["text"])
            self.assertEqual(projected["semantic"]["fp_1f_clean_indices"], [0])
            self.assertEqual(projected["semantic"]["fp_1f_target_index"], 24)
            self.assertEqual(projected["native"]["image_jsonl_file"], "/workspace/" + native_file["path"])
            self.assertEqual(row_value["caption"], "caption")
            self.assertEqual(projected["policy"]["profile"], "h3-one-frame-fl2va")
            self.assertEqual(projected["policy"]["codec"], "h3-one-frame-control-jsonl")
            self.assertEqual(projected["policy"]["caption_transform"], "strip")
            self.assertEqual(projected["policy"]["audio_selection"], "unsupported")
            self.assertIn("image_path", row_value)
            self.assertIn("control_path", row_value)
            self.assertEqual(view["consumers"][0]["kind"], "jsonl")
            self.assertNotIn("bindings", view)
            parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
            self.assertEqual(parsed["datasets"], [projected["native"]])

            changed_run = deepcopy(run)
            changed_run["id"] = "changed-h3-timing"
            changed_run["backend"]["config"]["dataset_options"]["tiny"]["fp_1f_target_index"] = 48
            changed_resolved = workspace / "runs" / changed_run["id"] / "resolved"
            changed_lock = freeze_dataset_handoff(
                changed_run,
                workspace,
                changed_resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(changed_run, selection),
            )
            self.assertNotEqual(lock["input_sha256"], changed_lock["input_sha256"])

    def test_musubi_minimax_h3_effective_task_is_shared_with_teacher_matching(self) -> None:
        cases = (
            ({"task": "fl2va"}, "fl2va"),
            ({"task": "t2va", "h3_loss_method": "teacher_matching", "h3_teacher_conditions": "first,last"}, "fl2va"),
            ({"task": "t2va", "h3_loss_method": "teacher_matching", "h3_teacher_conditions": "ref"}, "t2va"),
            ({"task": "t2va", "h3_loss_method": "teacher_matching", "h3_teacher_conditions": "subject_ref"}, "ref2va"),
            ({"task": "t2va", "h3_loss_method": "guidance", "h3_teacher_conditions": "first,last"}, "t2va"),
        )
        for override, expected in cases:
            with self.subTest(override=override):
                self.assertEqual(_musubi_h3_effective_task(override), expected)

    def test_musubi_minimax_h3_fl2va_control_requires_explicit_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "minimax_h3",
                "task": "fl2va",
                "one_frame": True,
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "source.png").write_bytes(b"control")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"].append({"type": "file", "role": "control", "path": "source.png"})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"profile h3-one-frame-fl2va requires dataset option\(s\).*fp_1f_clean_indices.*fp_1f_target_index",
            ):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_rejects_invalid_minimax_h3_one_frame_timing_options(self) -> None:
        cases = (
            ({"fp_1f_clean_indices": []}, "fp_1f_clean_indices.*non-empty list"),
            ({"fp_1f_clean_indices": [True]}, "fp_1f_clean_indices.*nonnegative integers"),
            ({"fp_1f_clean_indices": [-1]}, "fp_1f_clean_indices.*nonnegative integers"),
            ({"fp_1f_target_index": True}, "fp_1f_target_index.*nonnegative integer"),
            ({"fp_1f_target_index": -1}, "fp_1f_target_index.*nonnegative integer"),
        )
        for options, message in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, _resolved = self.make_run(workspace)
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "minimax_h3",
                    "task": "fl2va",
                    "one_frame": True,
                    "dataset_options": {"tiny": options},
                }}

                with self.assertRaisesRegex(ValueError, message):
                    project_musubi_dataset(run, {"datasets": []})

    def test_musubi_minimax_h3_rejects_kontext_control_resize_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "minimax_h3",
                "task": "fl2va",
                "one_frame": True,
                "dataset_options": {
                    "tiny": {
                        "fp_1f_clean_indices": [0],
                        "fp_1f_target_index": 24,
                        "control_resolution": [768, 768],
                    },
                },
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "source.png").write_bytes(b"control")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"].append({"type": "file", "role": "control", "path": "source.png"})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"profile h3-one-frame-fl2va does not accept dataset option\(s\): control_resolution",
            ):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_minimax_h3_fl2va_timing_requires_a_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "minimax_h3",
                "task": "fl2va",
                "one_frame": True,
                "dataset_options": {
                    "tiny": {"fp_1f_clean_indices": [0], "fp_1f_target_index": 24},
                },
            }}

            with self.assertRaisesRegex(
                ValueError,
                "no verified Musubi projection profile matches.*minimax_h3.*shape='image'.*task_conditioning.*first-last-frame",
            ):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_h3_timing_options_stop_outside_one_frame_fl2va(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "minimax_h3",
                "task": "t2va",
                "one_frame": True,
                "dataset_options": {"tiny": {"fp_1f_target_index": 24}},
            }}

            with self.assertRaisesRegex(
                ValueError,
                "profile h3-one-frame-plain does not accept dataset option.*fp_1f_target_index",
            ):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_initial_projection_stops_unsupported_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux2"}}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "a.png").unlink()
            (dataset / "a.mp4").write_bytes(b"video")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"][0]["path"] = "a.mp4"
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "no verified Musubi projection profile matches.*flux2.*shape='video'",
            ):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

            run["backend"]["config"]["dataset_config"] = {"general": {"resolution": [512, 512]}}
            with self.assertRaisesRegex(ValueError, "dataset_config.*replaced by manifest projection"):
                project_musubi_dataset(run, {"datasets": []})

    def test_musubi_initial_projection_rejects_an_absent_caption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux2"}}
            dataset = workspace / "datasets" / "tiny"
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["caption"] = None
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sample 'a'.*caption.*cannot be absent"):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_projects_wan_video_caption_with_explicit_frame_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "wan",
                "dataset_options": {
                    "tiny": {"target_frames": [1, 25], "frame_extraction": "head", "source_fps": 16.0},
                },
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "a.png").unlink()
            (dataset / "a.mp4").write_bytes(b"video")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"][0]["path"] = "a.mp4"
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            materialize_dataset_view(workspace, lock)
            _write_musubi_dataset_config(
                run,
                resolved / "musubi" / "dataset.toml",
                workspace=workspace,
                strict=True,
            )

            report = json.loads((resolved / "dataset-projection.lock.json").read_text(encoding="utf-8"))
            native = report["datasets"][0]["native"]
            self.assertEqual(native["target_frames"], [1, 25])
            self.assertEqual(native["frame_extraction"], "head")
            self.assertEqual(native["source_fps"], 16.0)
            self.assertEqual(lock["views"][0]["consumers"][0]["native_pointer"], "/video_jsonl_file")
            self.assertTrue(lock["views"][0]["links"][0]["path"].endswith(".mp4"))
            self.assertEqual(report["datasets"][0]["policy"]["codec"], "plain-video-jsonl")
            self.assertEqual(report["datasets"][0]["policy"]["profile"], "wan-video")
            self.assertEqual(report["datasets"][0]["policy"]["target_fps"], 16.0)
            self.assertEqual(_musubi_video_preflight_env(run, resolved / "musubi"), {
                "KURA_MUSUBI_ARCHITECTURE": "wan",
                "KURA_MUSUBI_TARGET_FPS": "16.0",
                "KURA_MUSUBI_FPS_RESAMPLE_MODE": "source-fps-when-declared",
                "KURA_MUSUBI_PROFILES": "wan-video",
            })
            self.assertEqual(report["datasets"][0]["policy"]["caption_transform"], "strip")
            self.assertEqual(report["datasets"][0]["policy"]["audio_selection"], "unsupported")
            parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
            self.assertEqual(parsed["datasets"], [native])

    def test_musubi_existing_codecs_cover_hidream_i2i_and_reject_task_shape_mismatches(self) -> None:
        for task, with_control, expected_profile in (
            ("i2i", True, "hidream-i2i"),
            ("t2i", False, "ordinary-image"),
        ):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, resolved = self.make_run(workspace)
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "hidream_o1",
                    "task": task,
                }}
                if with_control:
                    dataset = workspace / "datasets" / "tiny"
                    (dataset / "control.png").write_bytes(b"control")
                    row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
                    row["files"].append({"type": "file", "role": "control", "path": "control.png"})
                    (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )
                projected = json.loads(
                    (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
                )["datasets"][0]
                self.assertEqual(projected["policy"]["profile"], expected_profile)
                self.assertEqual(
                    projected["policy"]["codec"],
                    "image-control-jsonl" if with_control else "plain-image-jsonl",
                )

        for task, with_control in (("t2i", True), ("i2i", False)):
            with self.subTest(rejected_task=task), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, resolved = self.make_run(workspace)
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "hidream_o1",
                    "task": task,
                }}
                if with_control:
                    dataset = workspace / "datasets" / "tiny"
                    (dataset / "control.png").write_bytes(b"control")
                    row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
                    row["files"].append({"type": "file", "role": "control", "path": "control.png"})
                    (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, r"no verified Musubi projection profile matches"):
                    freeze_dataset_handoff(
                        run,
                        workspace,
                        resolved,
                        backend="musubi-tuner",
                        project=lambda selection: project_musubi_dataset(run, selection),
                    )

    def test_musubi_existing_codecs_cover_wan_official_dual_t2i_and_single_frame_modes(self) -> None:
        video_tasks = ("t2v-1.3B", "t2v-14B", "i2v-14B", "t2v-A14B", "i2v-A14B")
        for task in video_tasks:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, resolved = self.make_run(workspace)
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "wan",
                    "task": task,
                    "dataset_options": {"tiny": {"target_frames": [1, 25]}},
                }}
                dataset = workspace / "datasets" / "tiny"
                (dataset / "a.png").unlink()
                (dataset / "a.mp4").write_bytes(b"video")
                row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
                row["files"][0]["path"] = "a.mp4"
                (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )
                projected = json.loads(
                    (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
                )["datasets"][0]
                self.assertEqual(projected["policy"]["profile"], "wan-video")

        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "wan", "task": "t2i-14B",
            }}
            freeze_dataset_handoff(
                run, workspace, resolved, backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            projected = json.loads(
                (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
            )["datasets"][0]
            self.assertEqual(projected["policy"]["profile"], "wan-image")
            self.assertEqual(projected["policy"]["codec"], "plain-image-jsonl")

        for task, control_count in (("i2v-14B", 1), ("flf2v-14B", 2)):
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, resolved = self.make_run(workspace)
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "wan", "task": task, "one_frame": True,
                    "dataset_options": {"tiny": {
                        "fp_1f_clean_indices": [0] if control_count == 1 else [0, 2],
                        "fp_1f_target_index": 1,
                    }},
                }}
                dataset = workspace / "datasets" / "tiny"
                row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
                for index in range(control_count):
                    name = f"control-{index}.png"
                    (dataset / name).write_bytes(name.encode("utf-8"))
                    row["files"].append({"type": "file", "role": "control", "path": name})
                (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )
                projected = json.loads(
                    (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
                )["datasets"][0]
                self.assertEqual(
                    projected["policy"]["profile"],
                    "wan-single-frame" if control_count == 1 else "wan-single-frame-intermediate",
                )
                self.assertEqual(projected["native"]["fp_1f_target_index"], 1)

    def test_musubi_wan_fun_control_waits_for_a_video_control_codec(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "wan", "task": "t2v-14B-FC",
                "dataset_options": {"tiny": {"target_frames": [1, 25]}},
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "a.png").unlink()
            (dataset / "a.mp4").write_bytes(b"video")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"][0]["path"] = "a.mp4"
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, r"no verified Musubi projection profile matches"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_existing_video_codec_covers_pinned_kandinsky5_video_tasks(self) -> None:
        tasks = (
            "k5-lite-t2v-5s-sd", "k5-lite-t2v-10s-sd", "k5-lite-i2v-5s-sd",
            "k5-pro-t2v-5s-sd", "k5-pro-t2v-5s-hd", "k5-pro-t2v-10s-sd",
            "k5-pro-t2v-10s-hd", "k5-pro-i2v-5s-sd", "k5-pro-i2v-5s-hd",
            "k5-lite-t2v-5s-distil-sd", "k5-lite-t2v-10s-distil-sd",
            "k5-lite-t2v-5s-nocfg-sd", "k5-lite-t2v-10s-nocfg-sd",
            "k5-lite-t2v-5s-pretrain-sd", "k5-lite-t2v-10s-pretrain-sd",
        )
        for task in tasks:
            with self.subTest(task=task), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, resolved = self.make_run(workspace)
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "kandinsky5", "task": task,
                    "dataset_options": {"tiny": {"target_frames": [1, 25]}},
                }}
                dataset = workspace / "datasets" / "tiny"
                (dataset / "a.png").unlink()
                (dataset / "a.mp4").write_bytes(b"video")
                row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
                row["files"][0]["path"] = "a.mp4"
                (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )
                projected = json.loads(
                    (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
                )["datasets"][0]
                self.assertEqual(projected["policy"]["profile"], "kandinsky5-video")
                self.assertEqual(projected["policy"]["codec"], "plain-video-jsonl")
                self.assertEqual(projected["policy"]["target_fps"], 24.0)

    def test_musubi_projects_hunyuan_video_jsonl_with_verified_frame_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "hunyuan_video",
                "dataset_options": {
                    "tiny": {"target_frames": [1, 25], "frame_extraction": "head", "source_fps": 24.0},
                },
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "a.png").unlink()
            (dataset / "a.mp4").write_bytes(b"video")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"][0]["path"] = "a.mp4"
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            materialize_dataset_view(workspace, lock)
            _write_musubi_dataset_config(
                run,
                resolved / "musubi" / "dataset.toml",
                workspace=workspace,
                strict=True,
            )

            report = json.loads((resolved / "dataset-projection.lock.json").read_text(encoding="utf-8"))
            projected = report["datasets"][0]
            self.assertEqual(projected["policy"]["profile"], "hunyuan-video")
            self.assertEqual(projected["policy"]["codec"], "plain-video-jsonl")
            self.assertEqual(projected["policy"]["target_fps"], 24.0)
            self.assertEqual(_musubi_video_preflight_env(run, resolved / "musubi"), {
                "KURA_MUSUBI_ARCHITECTURE": "hunyuan_video",
                "KURA_MUSUBI_TARGET_FPS": "24.0",
                "KURA_MUSUBI_FPS_RESAMPLE_MODE": "source-fps-when-declared",
                "KURA_MUSUBI_PROFILES": "hunyuan-video",
            })
            self.assertEqual(projected["native"]["target_frames"], [1, 25])
            self.assertEqual(projected["native"]["frame_extraction"], "head")
            self.assertEqual(projected["native"]["source_fps"], 24.0)
            self.assertEqual(lock["views"][0]["consumers"][0]["native_pointer"], "/video_jsonl_file")
            parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
            self.assertEqual(parsed["datasets"], [projected["native"]])

    def test_musubi_hunyuan_video_rejects_frames_outside_the_pinned_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "hunyuanvideo",
                "dataset_options": {"tiny": {"target_frames": [24]}},
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "a.png").unlink()
            (dataset / "a.mp4").write_bytes(b"video")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"][0]["path"] = "a.mp4"
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"profile hunyuan-video.*1\+4n grid"):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_projects_hunyuan_video_1_5_video_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "hunyuan_video_1_5",
                "task": "t2v",
                "dataset_options": {
                    "tiny": {"target_frames": [1, 25], "frame_extraction": "head", "source_fps": 30.0},
                },
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "a.png").unlink()
            (dataset / "a.mp4").write_bytes(b"video")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"][0]["path"] = "a.mp4"
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            _write_musubi_dataset_config(
                run,
                resolved / "musubi" / "dataset.toml",
                workspace=workspace,
                strict=True,
            )

            projected = json.loads(
                (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
            )["datasets"][0]
            self.assertEqual(projected["policy"]["profile"], "hunyuan-video-1.5-video")
            self.assertEqual(projected["policy"]["codec"], "plain-video-jsonl")
            self.assertEqual(projected["policy"]["target_fps"], 24.0)
            self.assertEqual(projected["native"]["target_frames"], [1, 25])
            self.assertEqual(projected["native"]["source_fps"], 30.0)
            self.assertEqual(lock["views"][0]["consumers"][0]["native_pointer"], "/video_jsonl_file")
            self.assertEqual(_musubi_video_preflight_env(run, resolved / "musubi")["KURA_MUSUBI_PROFILES"], "hunyuan-video-1.5-video")
            parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
            self.assertEqual(parsed["datasets"], [projected["native"]])

    def test_musubi_projects_hunyuan_video_1_5_t2v_image_through_ordinary_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "hunyuan_video_1_5",
                "task": "t2v",
            }}

            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            _write_musubi_dataset_config(
                run,
                resolved / "musubi" / "dataset.toml",
                workspace=workspace,
                strict=True,
            )

            projected = json.loads(
                (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
            )["datasets"][0]
            self.assertEqual(projected["policy"]["profile"], "ordinary-image")
            self.assertEqual(projected["policy"]["codec"], "plain-image-jsonl")
            self.assertNotIn("target_fps", projected["policy"])
            self.assertEqual(lock["views"][0]["consumers"][0]["native_pointer"], "/image_jsonl_file")
            parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
            self.assertEqual(parsed["datasets"], [projected["native"]])

    def test_musubi_rejects_hunyuan_video_1_5_i2v_image_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "hunyuan_video_1_5",
                "task": "i2v",
            }}

            with self.assertRaisesRegex(
                ValueError,
                r"no verified Musubi projection profile matches.*hunyuan_video_1_5.*task_conditioning.*first-frame",
            ):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_projects_hunyuan_video_image_through_ordinary_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "hunyuan_video",
            }}

            freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )

            projected = json.loads(
                (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
            )["datasets"][0]
            self.assertEqual(projected["policy"]["profile"], "ordinary-image")
            self.assertEqual(projected["policy"]["codec"], "plain-image-jsonl")

    def test_musubi_hunyuan_video_1_5_rejects_frames_outside_the_pinned_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "hunyuan_video_1_5",
                "dataset_options": {"tiny": {"target_frames": [24]}},
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "a.png").unlink()
            (dataset / "a.mp4").write_bytes(b"video")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"][0]["path"] = "a.mp4"
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"profile hunyuan-video-1.5-video.*1\+4n grid"):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_projects_framepack_normal_and_f1_video_profiles(self) -> None:
        for f1, expected_profile in (
            (False, "framepack-video"),
            (True, "framepack-f1-video"),
        ):
            with self.subTest(f1=f1), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, resolved = self.make_run(workspace)
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "framepack",
                    "f1": f1,
                    "dataset_options": {"tiny": {"target_frames": [37]}},
                }}
                dataset = workspace / "datasets" / "tiny"
                (dataset / "a.png").unlink()
                (dataset / "a.mp4").write_bytes(b"video")
                row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
                row["files"][0]["path"] = "a.mp4"
                (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )
                _write_musubi_dataset_config(
                    run,
                    resolved / "musubi" / "dataset.toml",
                    workspace=workspace,
                    strict=True,
                )

                projected = json.loads(
                    (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
                )["datasets"][0]
                self.assertEqual(projected["policy"]["profile"], expected_profile)
                self.assertEqual(projected["policy"]["target_fps"], 30.0)
                self.assertEqual(projected["native"]["fp_latent_window_size"], 9)
                self.assertEqual(projected["native"]["target_frames"], [37])
                self.assertEqual(projected["native"]["frame_extraction"], "full")
                self.assertEqual(projected["native"]["max_frames"], 129)
                parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
                self.assertEqual(parsed["datasets"], [projected["native"]])

    def test_musubi_framepack_can_select_head_or_full_extraction_and_max_frames(self) -> None:
        for frame_extraction, max_frames in (("head", 73), ("full", 109)):
            with self.subTest(frame_extraction=frame_extraction), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, resolved = self.make_run(workspace)
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "framepack",
                    "dataset_options": {"tiny": {
                        "target_frames": [37],
                        "frame_extraction": frame_extraction,
                        "max_frames": max_frames,
                    }},
                }}
                dataset = workspace / "datasets" / "tiny"
                (dataset / "a.png").unlink()
                (dataset / "a.mp4").write_bytes(b"video")
                row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
                row["files"][0]["path"] = "a.mp4"
                (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )
                projected = json.loads(
                    (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
                )["datasets"][0]
                self.assertEqual(projected["native"]["frame_extraction"], frame_extraction)
                self.assertEqual(projected["native"]["max_frames"], max_frames)

    def test_musubi_framepack_video_requires_at_least_one_full_latent_window(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "framepack",
                "dataset_options": {"tiny": {"target_frames": [33]}},
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "a.png").unlink()
            (dataset / "a.mp4").write_bytes(b"video")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"][0]["path"] = "a.mp4"
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"framepack-video.*at least 37 frames"):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_framepack_one_frame_waits_for_its_control_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "framepack",
                "one_frame": True,
            }}

            with self.assertRaisesRegex(
                ValueError,
                r"no verified Musubi projection profile matches.*framepack.*one_frame.*True",
            ):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_projects_framepack_single_frame_target_and_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "framepack",
                "one_frame": True,
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "control.png").write_bytes(b"control")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"].append({"type": "file", "role": "control", "path": "control.png"})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            _write_musubi_dataset_config(
                run,
                resolved / "musubi" / "dataset.toml",
                workspace=workspace,
                strict=True,
            )

            projected = json.loads(
                (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
            )["datasets"][0]
            self.assertEqual(projected["policy"]["profile"], "framepack-single-frame")
            self.assertEqual(projected["policy"]["codec"], "image-control-jsonl")
            self.assertEqual(projected["native"]["fp_latent_window_size"], 9)
            self.assertEqual(projected["native"]["fp_1f_clean_indices"], [0])
            self.assertEqual(projected["native"]["fp_1f_target_index"], 9)
            self.assertFalse(projected["native"]["fp_1f_no_post"])
            generated = json.loads(
                projected["views"][0]["native_files"][0]["text"]
            )
            self.assertIn("image_path", generated)
            self.assertIn("control_path", generated)
            parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
            self.assertEqual(parsed["datasets"], [projected["native"]])

    def test_musubi_framepack_single_frame_freezes_explicit_dataset_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "framepack",
                "one_frame": True,
                "dataset_options": {"tiny": {
                    "fp_1f_clean_indices": [2],
                    "fp_1f_target_index": 13,
                    "fp_1f_no_post": True,
                }},
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "control.png").write_bytes(b"control")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"].append({"type": "file", "role": "control", "path": "control.png"})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            projected = json.loads(
                (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
            )["datasets"][0]
            self.assertEqual(projected["native"]["fp_1f_clean_indices"], [2])
            self.assertEqual(projected["native"]["fp_1f_target_index"], 13)
            self.assertTrue(projected["native"]["fp_1f_no_post"])

    def test_musubi_framepack_single_frame_projects_1f_mc_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "framepack",
                "one_frame": True,
                "dataset_options": {"tiny": {
                    "fp_1f_clean_indices": [0, 1],
                    "fp_1f_target_index": 9,
                }},
            }}
            dataset = workspace / "datasets" / "tiny"
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            for name in ("control-0.png", "control-1.png"):
                (dataset / name).write_bytes(name.encode("utf-8"))
                row["files"].append({"type": "file", "role": "control", "path": name})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )

            projected = json.loads(
                (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
            )["datasets"][0]
            self.assertEqual(projected["policy"]["profile"], "framepack-single-frame-multi-control")
            self.assertEqual(projected["native"]["fp_1f_clean_indices"], [0, 1])
            self.assertEqual(projected["native"]["fp_1f_target_index"], 9)
            generated = json.loads(projected["views"][0]["native_files"][0]["text"])
            self.assertEqual(
                sorted(key for key in generated if key.startswith("control_path")),
                ["control_path_0", "control_path_1"],
            )

    def test_musubi_framepack_multi_control_projects_numbered_jsonl_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "framepack",
                "one_frame": True,
                "dataset_options": {"tiny": {
                    "fp_1f_clean_indices": [0, 10],
                    "fp_1f_target_index": 1,
                    "fp_1f_no_post": True,
                }},
            }}
            dataset = workspace / "datasets" / "tiny"
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            for name in ("start.png", "reference.png"):
                (dataset / name).write_bytes(name.encode("utf-8"))
                row["files"].append({"type": "file", "role": "control", "path": name})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )

            projected = json.loads(
                (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
            )["datasets"][0]
            self.assertEqual(projected["policy"]["profile"], "framepack-single-frame-multi-control")
            self.assertEqual(projected["native"]["fp_1f_clean_indices"], [0, 10])
            self.assertEqual(projected["native"]["fp_1f_target_index"], 1)
            self.assertTrue(projected["native"]["fp_1f_no_post"])
            native_file = projected["views"][0]["native_files"][0]
            generated = json.loads(native_file["text"])
            self.assertNotIn("control_path", generated)
            self.assertIn("control_path_0", generated)
            self.assertIn("control_path_1", generated)
            references = native_file["rows"][0]["references"]
            self.assertEqual(
                [item["pointer"] for item in references if item["kind"] == "path"],
                ["/image_path", "/control_path_0", "/control_path_1"],
            )
            self.assertEqual(
                MUSUBI_PROJECTION_PROFILES["framepack-single-frame-multi-control"]
                ["role_limits"]["control"],
                (2, None),
            )
            self.assertEqual(
                MUSUBI_PROJECTION_PROFILES["framepack-single-frame-multi-control"]["shape"],
                "image-control",
            )
            self.assertTrue(all(
                "control_count" not in profile
                for profile in MUSUBI_PROJECTION_PROFILES.values()
            ))

    def test_musubi_qwen_edit_profiles_enforce_pinned_control_limits(self) -> None:
        cases = (
            ("edit", 1, "qwen-image-edit"),
            ("edit-2509", 3, "qwen-image-edit-multi-control"),
            ("edit-2511", 3, "qwen-image-edit-multi-control"),
            ("EDIT_2509", 3, "qwen-image-edit-multi-control"),
        )
        for model_version, control_count, expected_profile in cases:
            with self.subTest(model_version=model_version), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, resolved = self.make_run(workspace)
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "qwen_image",
                    "model_version": model_version,
                }}
                dataset = workspace / "datasets" / "tiny"
                row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
                for index in range(control_count):
                    name = f"control-{index}.png"
                    (dataset / name).write_bytes(name.encode("utf-8"))
                    row["files"].append({"type": "file", "role": "control", "path": name})
                (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )
                _write_musubi_dataset_config(
                    run,
                    resolved / "musubi" / "dataset.toml",
                    workspace=workspace,
                    strict=True,
                )
                projected = json.loads(
                    (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
                )["datasets"][0]
                self.assertEqual(projected["policy"]["profile"], expected_profile)
                generated = json.loads(projected["views"][0]["native_files"][0]["text"])
                expected_keys = (
                    ["control_path"]
                    if control_count == 1
                    else [f"control_path_{index}" for index in range(control_count)]
                )
                self.assertEqual(
                    sorted(key for key in generated if key.startswith("control_path")),
                    expected_keys,
                )
                parsed = tomllib.loads(
                    (resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8")
                )
                self.assertEqual(parsed["datasets"], [projected["native"]])

        self.assertEqual(MUSUBI_PROJECTION_PROFILES["qwen-image-edit"]["role_limits"]["control"], (1, 1))
        self.assertEqual(
            MUSUBI_PROJECTION_PROFILES["qwen-image-edit-multi-control"]["role_limits"]["control"],
            (1, 3),
        )

    def test_musubi_qwen_edit_stops_above_each_model_control_limit(self) -> None:
        for model_version, control_count in (("edit", 2), ("edit-2509", 4), ("edit-2511", 4)):
            with self.subTest(model_version=model_version), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, resolved = self.make_run(workspace)
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "qwen_image",
                    "model_version": model_version,
                }}
                dataset = workspace / "datasets" / "tiny"
                row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
                for index in range(control_count):
                    name = f"control-{index}.png"
                    (dataset / name).write_bytes(name.encode("utf-8"))
                    row["files"].append({"type": "file", "role": "control", "path": name})
                (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

                expected_error = (
                    r"no verified Musubi projection profile matches.*image-control.*control.*2"
                    if model_version == "edit"
                    else r"role cardinalities=.*control.*4"
                )
                with self.assertRaisesRegex(ValueError, expected_error):
                    freeze_dataset_handoff(
                        run,
                        workspace,
                        resolved,
                        backend="musubi-tuner",
                        project=lambda selection: project_musubi_dataset(run, selection),
                    )

    def test_musubi_qwen_original_accepts_image_only_but_edit_requires_controls(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "qwen_image",
                "model_version": "original",
            }}

            freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            projected = json.loads(
                (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
            )["datasets"][0]
            self.assertEqual(projected["policy"]["profile"], "ordinary-image")

        for model_version in ("edit", "edit-2509", "edit-2511"):
            with self.subTest(model_version=model_version), tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory)
                run, resolved = self.make_run(workspace)
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "qwen_image",
                    "model_version": model_version,
                }}
                with self.assertRaisesRegex(
                    ValueError,
                    rf"no verified Musubi projection profile matches.*model_version.*{model_version}",
                ):
                    freeze_dataset_handoff(
                        run,
                        workspace,
                        resolved,
                        backend="musubi-tuner",
                        project=lambda selection: project_musubi_dataset(run, selection),
                    )

    def test_musubi_qwen_layered_projects_ordered_multiple_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "qwen_image",
                "model_version": "layered",
            }}
            dataset = workspace / "datasets" / "tiny"
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            for name in ("layer-1.png", "layer-2.png"):
                (dataset / name).write_bytes(name.encode("utf-8"))
                row["files"].append({"type": "file", "role": "target", "path": name})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            freeze_dataset_handoff(
                run, workspace, resolved, backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            _write_musubi_dataset_config(
                run, resolved / "musubi" / "dataset.toml", workspace=workspace, strict=True,
            )

            projected = json.loads(
                (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
            )["datasets"][0]
            self.assertEqual(projected["policy"]["profile"], "qwen-image-layered")
            self.assertEqual(projected["policy"]["codec"], "layered-image-jsonl")
            self.assertEqual(MUSUBI_PROJECTION_PROFILES["qwen-image-layered"]["shape"], "image")
            self.assertTrue(projected["native"]["multiple_target"])
            generated = json.loads(projected["views"][0]["native_files"][0]["text"])
            self.assertEqual(
                [key for key in generated if key.startswith("image_path_")],
                ["image_path_0", "image_path_1", "image_path_2"],
            )
            references = projected["views"][0]["native_files"][0]["rows"][0]["references"]
            self.assertEqual(
                [item["pointer"] for item in references if item["kind"] == "path"],
                ["/image_path_0", "/image_path_1", "/image_path_2"],
            )
            parsed = tomllib.loads((resolved / "musubi" / "dataset.toml").read_text(encoding="utf-8"))
            self.assertEqual(parsed["datasets"], [projected["native"]])

    def test_musubi_qwen_layered_requires_at_least_base_and_one_layer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "qwen_image",
                "model_version": "layered",
            }}

            with self.assertRaisesRegex(
                ValueError, r"no verified Musubi projection profile matches.*layered",
            ):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_framepack_kisekaeichi_rejects_a_separate_mask_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "framepack",
                "one_frame": True,
                "dataset_options": {"tiny": {
                    "fp_1f_clean_indices": [0, 10],
                    "fp_1f_target_index": 1,
                    "fp_1f_no_post": True,
                }},
            }}
            dataset = workspace / "datasets" / "tiny"
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            for role, name in (
                ("control", "start.png"),
                ("control", "reference.png"),
                ("mask", "mask.png"),
            ):
                (dataset / name).write_bytes(name.encode("utf-8"))
                row["files"].append({"type": "file", "role": role, "path": name})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"shape='roles:mask'"):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_framepack_multi_control_requires_one_index_per_control(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "framepack",
                "one_frame": True,
                "dataset_options": {"tiny": {
                    "fp_1f_clean_indices": [0],
                    "fp_1f_target_index": 9,
                }},
            }}
            dataset = workspace / "datasets" / "tiny"
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            for index in range(2):
                name = f"control-{index}.png"
                (dataset / name).write_bytes(name.encode("utf-8"))
                row["files"].append({"type": "file", "role": "control", "path": name})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"fp_1f_clean_indices=1 for control=2"):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_flux2_profile_accepts_an_unbounded_number_of_references(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "flux2",
                "model_version": "dev",
                "dataset_options": {"tiny": {
                    "no_resize_control": True,
                    "control_resolution": [1024, 1024],
                }},
            }}
            dataset = workspace / "datasets" / "tiny"
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            for index in range(4):
                name = f"reference-{index}.png"
                (dataset / name).write_bytes(name.encode("utf-8"))
                row["files"].append({"type": "file", "role": "control", "path": name})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )

            projected = json.loads(
                (resolved / "dataset-projection.lock.json").read_text(encoding="utf-8")
            )["datasets"][0]
            self.assertEqual(projected["policy"]["profile"], "flux2-image-references")
            self.assertTrue(projected["native"]["no_resize_control"])
            self.assertEqual(projected["native"]["control_resolution"], [1024, 1024])
            self.assertEqual(
                MUSUBI_PROJECTION_PROFILES["flux2-image-references"]["role_limits"]["control"],
                (1, None),
            )

    def test_musubi_video_projection_uses_the_command_architecture_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "model_arch": "wan",
                "dataset_options": {"tiny": {"target_frames": [25]}},
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "a.png").unlink()
            (dataset / "a.mp4").write_bytes(b"video")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"][0]["path"] = "a.mp4"
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )

            self.assertEqual(lock["views"][0]["consumers"][0]["native_pointer"], "/video_jsonl_file")

    def test_musubi_wan_frame_settings_change_input_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "wan",
                "dataset_options": {"tiny": {"target_frames": [25]}},
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "a.png").unlink()
            (dataset / "a.mp4").write_bytes(b"video")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"][0]["path"] = "a.mp4"
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            first = freeze_dataset_handoff(
                run, workspace, resolved, backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(run, selection),
            )
            changed = deepcopy(run)
            changed["id"] = "wan-49-frames"
            changed["backend"]["config"]["dataset_options"]["tiny"]["target_frames"] = [49]
            second = freeze_dataset_handoff(
                changed, workspace, workspace / "runs" / changed["id"] / "resolved",
                backend="musubi-tuner",
                project=lambda selection: project_musubi_dataset(changed, selection),
            )
            self.assertNotEqual(first["input_sha256"], second["input_sha256"])

    def test_musubi_video_projection_requires_explicit_frame_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "wan"}}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "a.png").unlink()
            (dataset / "a.mp4").write_bytes(b"video")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"][0]["path"] = "a.mp4"
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                r"profile wan-video requires dataset option\(s\): target_frames",
            ):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

    def test_musubi_video_projection_validates_typed_frame_options(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, _resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "wan",
                "dataset_options": {"tiny": {"target_frames": [24]}},
            }}
            dataset = workspace / "datasets" / "tiny"
            (dataset / "a.png").unlink()
            (dataset / "a.mp4").write_bytes(b"video")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"][0]["path"] = "a.mp4"
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"1\+4n grid"):
                freeze_dataset_handoff(
                    run, workspace, _resolved, backend="musubi-tuner",
                    project=lambda selection: project_musubi_dataset(run, selection),
                )

            run["backend"]["config"]["dataset_options"]["tiny"] = {
                "target_frames": [25],
                "frame_extraction": "uniform",
            }
            with self.assertRaisesRegex(ValueError, "currently supports only 'head'"):
                project_musubi_dataset(run, {"datasets": []})

            run["backend"]["config"]["dataset_options"]["tiny"] = {
                "target_frames": [25],
                "source_fps": float("nan"),
            }
            with self.assertRaisesRegex(ValueError, "source_fps must be positive and finite"):
                project_musubi_dataset(run, {"datasets": []})

            run["backend"]["config"]["dataset_options"] = {
                "not-declared": {"target_frames": [25]},
            }
            with self.assertRaisesRegex(ValueError, "undeclared dataset"):
                project_musubi_dataset(run, {"datasets": []})

    def test_uncompiled_plan_defers_manifest_projection_to_compile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, _resolved = self.make_run(workspace)
            run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux2"}}

            records = _dataset_layout_preflight_report(run, workspace)

            self.assertEqual(records[0]["severity"], "info")
            self.assertIn("compilation", records[0]["fact"])
            self.assertNotIn("missing", records[0]["fact"])

    def test_musubi_video_plan_discloses_container_only_frame_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            resolved = run_dir / "resolved"
            resolved.mkdir(parents=True)
            (resolved / "dataset-projection.lock.json").write_text(json.dumps({
                "backend": "musubi-tuner",
                "datasets": [{
                    "id": "clips",
                    "native": {
                        "video_jsonl_file": "/workspace/runs/example/cache/dataset-view/musubi/clips/native/items.jsonl",
                        "target_frames": [1, 25, 49],
                    },
                }],
            }), encoding="utf-8")

            checks = _dataset_runtime_checks(run_dir)
            output = format_run_plan({
                "id": "example",
                "type": "train",
                "backend": {"name": "musubi-tuner", "config": {}},
                "model": {},
                "compute": {},
                "datasets": [],
                "dataset_input": {
                    "status": "current",
                    "verification": "content-hash-at-compile",
                    "selection": [],
                    "views": [],
                    "changes": [],
                    "runtime_checks": checks,
                    "projection_rules": [{
                        "dataset": "clips",
                        "profile": "wan-video",
                        "codec": "plain-video-jsonl",
                        "caption_transform": "strip",
                        "audio_selection": "unsupported",
                    }],
                },
                "write_roots": [],
                "recipe": {},
                "sampling": {},
                "resources": {},
                "runpod_capacity": None,
                "model_downloads": {},
                "disk_cache": {},
                "preflight": [],
                "experiment": {},
                "training_state": {},
                "resume": None,
            })

            self.assertEqual(checks[0]["required_frames"], 49)
            self.assertIn("immediately after container launch, before model acquisition", output)
            self.assertIn("host_verification unavailable", output)
            self.assertIn("codec        plain-video-jsonl", output)
            self.assertIn("caption_transform strip", output)
            self.assertIn("audio_selection unsupported", output)
            self.assertIn("profile      wan-video", output)

    def test_musubi_h3_video_plan_warns_outside_the_released_frame_range(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            resolved = run_dir / "resolved"
            resolved.mkdir(parents=True)
            (resolved / "dataset-projection.lock.json").write_text(json.dumps({
                "backend": "musubi-tuner",
                "datasets": [{
                    "id": "clips",
                    "policy": {"profile": "h3-video-t2va"},
                    "native": {
                        "video_jsonl_file": "/workspace/runs/example/cache/dataset-view/musubi/clips/native/items.jsonl",
                        "target_frames": [22, 362],
                    },
                }],
            }), encoding="utf-8")

            checks = _dataset_runtime_checks(run_dir)

            self.assertEqual(checks[0]["released_frame_range"], [124, 345])
            self.assertEqual(checks[0]["released_range_warning"], [22, 362])

    def test_input_identity_is_stable_across_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_a, _ = self.make_run(workspace)
            run_a["id"] = "run-a"
            run_b = {**run_a, "id": "run-b"}
            lock_a = freeze_dataset_handoff(
                run_a,
                workspace,
                workspace / "runs" / "run-a" / "resolved",
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run_a, selection),
            )
            lock_b = freeze_dataset_handoff(
                run_b,
                workspace,
                workspace / "runs" / "run-b" / "resolved",
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run_b, selection),
            )
            self.assertEqual(lock_a["input_sha256"], lock_b["input_sha256"])

    def test_recompile_after_content_change_uses_a_new_view_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            first = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run, selection),
            )
            first_path = workspace / first["views"][0]["links"][0]["path"]
            first_name = first_path.name
            materialize_dataset_view(workspace, first)
            self.assertTrue(first_path.is_symlink())
            (workspace / "datasets" / "tiny" / "a.png").write_bytes(b"different image")
            second = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run, selection),
            )
            second_path = workspace / second["views"][0]["links"][0]["path"]
            second_name = second_path.name
            self.assertNotEqual(first_name, second_name)
            materialize_dataset_view(workspace, second)
            self.assertFalse(first_path.exists())
            self.assertTrue(second_path.is_symlink())

    def test_recompile_after_caption_change_uses_a_new_view_name(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            first = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run, selection),
            )
            first_link = Path(first["views"][0]["links"][0]["path"])
            first_caption = Path(first["views"][0]["files"][0]["path"])

            (workspace / "datasets" / "tiny" / "a.txt").write_text(
                "different caption\n", encoding="utf-8",
            )
            second = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run, selection),
            )

            self.assertNotEqual(first_link.stem, Path(second["views"][0]["links"][0]["path"]).stem)
            self.assertNotEqual(first_caption.stem, Path(second["views"][0]["files"][0]["path"]).stem)

    def test_ai_toolkit_rejects_a_role_it_cannot_represent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            dataset = workspace / "datasets" / "tiny"
            (dataset / "control.png").write_bytes(b"control")
            row = json.loads((dataset / "items.jsonl").read_text(encoding="utf-8"))
            row["files"].append({"type": "file", "role": "control", "path": "control.png"})
            (dataset / "items.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "sample 'a'.*control.*image mode"):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="ai-toolkit",
                    project=lambda selection: project_ai_toolkit_dataset(run, selection),
                )

    def test_ai_toolkit_explicit_command_cannot_bypass_manifest_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run["backend"]["config"]["command"] = {
                "cwd": "/app/ai-toolkit", "argv": ["python", "custom.py"], "env": {},
            }

            with self.assertRaisesRegex(ValueError, "explicit command cannot yet prove"):
                freeze_dataset_handoff(
                    run,
                    workspace,
                    resolved,
                    backend="ai-toolkit",
                    project=lambda selection: project_ai_toolkit_dataset(run, selection),
                )

    def test_cli_explicit_command_records_unverified_native_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, _ = self.make_run(workspace)
            run.update({
                "schema_version": 2,
                "type": "train",
                "model": {"base": "example/model"},
                "compute": {"executor": "docker"},
                "backend": {"name": "ai-toolkit", "config": {"command": {
                    "cwd": "/app/ai-toolkit", "argv": ["python", "custom.py"], "env": {},
                }}},
                "recovery": {"training_state": {"enabled": False}},
            })
            run_dir = workspace / "runs" / "example"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            (workspace / "workspace.yaml").write_text(yaml.safe_dump({
                "schema_version": 1,
                "docker": {"images": {"ai-toolkit": {
                    "local": "example:image",
                    "remote": "example:image@sha256:" + "1" * 64,
                    "dockerfile": "docker/ai-toolkit/Dockerfile",
                    "context": ".",
                }}},
            }), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(workspace)
            try:
                with patch("kura.executors.docker._docker_image_id", return_value="sha256:" + "2" * 64):
                    self.assertEqual(cmd_run_compile(argparse.Namespace(run_id="example")), 0)
            finally:
                os.chdir(previous)

            lock = json.loads((run_dir / "resolved" / "dataset-input.lock.json").read_text(encoding="utf-8"))
            self.assertEqual(lock["verification"], "unverified-native-source")
            self.assertFalse((run_dir / "resolved" / "dataset-projection.lock.json").exists())

    def test_ai_toolkit_strict_compile_uses_only_the_frozen_projection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run.update({
                "model": {"base": "example/model"},
                "recipe": {"steps": 1, "seed": 1},
            })
            freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run, selection),
            )

            compile_ai_toolkit(run, resolved / "ai-toolkit", workspace=workspace, strict=True)

            native = yaml.safe_load((resolved / "ai-toolkit.yaml").read_text(encoding="utf-8"))
            self.assertEqual(
                native["config"]["process"][0]["datasets"],
                [{
                    "folder_path": "/workspace/runs/example/cache/dataset-view/ai-toolkit/tiny",
                    "caption_ext": ".txt",
                    "cache_latents_to_disk": True,
                }],
            )
            self.assertFalse((resolved / "ai-toolkit" / "dataset-stage.lock.json").exists())

    def test_ai_toolkit_compile_rejects_native_path_that_bypasses_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            run.update({"model": {"base": "example/model"}, "recipe": {"steps": 1, "seed": 1}})
            freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run, selection),
            )
            projection_path = resolved / "dataset-projection.lock.json"
            projection = json.loads(projection_path.read_text(encoding="utf-8"))
            projection["datasets"][0]["native"]["folder_path"] = "/workspace/datasets/tiny"
            projection_path.write_text(json.dumps(projection), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "bypasses its run-owned view"):
                compile_ai_toolkit(run, resolved / "ai-toolkit", workspace=workspace, strict=True)

    def test_cli_compile_freezes_projection_before_native_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, _ = self.make_run(workspace)
            run.update({
                "schema_version": 2,
                "type": "train",
                "model": {"base": "example/model"},
                "recipe": {"steps": 1, "seed": 1},
                "compute": {"executor": "docker"},
                "recovery": {"training_state": {"enabled": False}},
            })
            run_dir = workspace / "runs" / "example"
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            (workspace / "workspace.yaml").write_text(yaml.safe_dump({
                "schema_version": 1,
                "docker": {"images": {"ai-toolkit": {
                    "local": "example:image",
                    "remote": "example:image@sha256:" + "1" * 64,
                    "dockerfile": "docker/ai-toolkit/Dockerfile",
                    "context": ".",
                }}},
            }), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(workspace)
            try:
                with patch("kura.executors.docker._docker_image_id", return_value="sha256:" + "2" * 64):
                    self.assertEqual(cmd_run_compile(argparse.Namespace(run_id="example")), 0)
            finally:
                os.chdir(previous)
            projection = json.loads(
                (run_dir / "resolved" / "dataset-projection.lock.json").read_text(encoding="utf-8")
            )
            native = yaml.safe_load((run_dir / "resolved" / "ai-toolkit.yaml").read_text(encoding="utf-8"))
            self.assertEqual(
                native["config"]["process"][0]["datasets"][0],
                projection["datasets"][0]["native"],
            )

    def test_v2_local_mounts_have_no_broad_writable_workspace_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run, selection),
            )
            mounts = local_training_mounts(workspace, resolved.parent, lock, configured=[])

            self.assertEqual(mounts, [
                {"source": str((workspace / "datasets" / "tiny").resolve()),
                 "target": "/workspace/datasets/tiny", "mode": "ro"},
                {"source": str(resolved.parent.resolve()),
                 "target": "/workspace/runs/example", "mode": "rw"},
                {"source": str(resolved.resolve()),
                 "target": "/workspace/runs/example/resolved", "mode": "ro"},
                {"source": str((workspace / "cache").resolve()),
                 "target": "/workspace/cache", "mode": "rw"},
            ])
            argv, runtime_env, _ = docker_command(
                workspace,
                resolved.parent,
                {"cwd": "/opt/ai-toolkit", "argv": ["python", "run.py"], "env": {}},
                "example:image",
                mounts,
                False,
                "r1",
                mount_workspace=False,
            )
            volumes = [argv[index + 1] for index, value in enumerate(argv) if value == "--volume"]
            self.assertNotIn(f"{workspace.resolve()}:/workspace", volumes)
            self.assertIn(f"{(workspace / 'datasets' / 'tiny').resolve()}:/workspace/datasets/tiny:ro", volumes)
            self.assertIn(f"{resolved.resolve()}:/workspace/runs/example/resolved:ro", volumes)
            mappings = json.loads(runtime_env["KURA_WORKSPACE_PATH_MAPS"])
            self.assertNotIn({"container": "/workspace", "workspace": "/workspace"}, mappings)

    def test_run_view_links_are_not_doctor_fix_link_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run, selection),
            )
            materialize_dataset_view(workspace, lock)

            inspected = inspect_workspace_symlinks(workspace)

            self.assertEqual(inspected["unsafe"], [])

    def test_configured_mount_cannot_reexpose_dataset_as_writable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run, selection),
            )
            with self.assertRaisesRegex(ValueError, "protected dataset mount"):
                local_training_mounts(workspace, resolved.parent, lock, configured=[{
                    "source": "datasets/tiny", "target": "/workspace/datasets/tiny", "mode": "rw",
                }])

            with self.assertRaisesRegex(ValueError, "mount source overlaps"):
                local_training_mounts(workspace, resolved.parent, lock, configured=[{
                    "source": "datasets", "target": "/data", "mode": "rw",
                }])

    def test_managed_writable_cache_cannot_overlap_dataset_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            dataset = workspace / "datasets" / "tiny"
            cache = workspace / "cache"
            cache.mkdir()
            physical = cache / "tiny"
            dataset.rename(physical)
            dataset.symlink_to(physical)
            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=self.image_projection,
            )

            with self.assertRaisesRegex(ValueError, "managed writable workspace cache overlaps"):
                local_training_mounts(workspace, resolved.parent, lock, configured=[])

    def test_v2_docker_dry_run_uses_closed_mounts_without_creating_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run, selection),
            )
            with patch("kura.executors.docker._docker_image_id", return_value=None):
                command, realization_id = launch_docker(
                    workspace=workspace,
                    run_dir=resolved.parent,
                    spec={"cwd": "/opt/ai-toolkit", "argv": ["python", "run.py"], "env": {}},
                    image="example:image",
                    dockerfile="docker/ai-toolkit/Dockerfile",
                    mounts=[],
                    gpu=False,
                    dry_run=True,
                )

            self.assertIsNone(realization_id)
            self.assertFalse((workspace / lock["views"][0]["root"]).exists())
            volumes = [command[index + 1] for index, value in enumerate(command) if value == "--volume"]
            self.assertNotIn(f"{workspace.resolve()}:/workspace", volumes)
            self.assertIn(f"{resolved.resolve()}:/workspace/runs/example/resolved:ro", volumes)

    def test_v2_docker_stops_on_changed_source_before_docker_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run, selection),
            )
            (workspace / "datasets" / "tiny" / "a.png").write_bytes(b"changed")

            with patch("kura.executors.docker.docker_preflight") as preflight:
                with self.assertRaisesRegex(ValueError, "changed since compile"):
                    launch_docker(
                        workspace=workspace,
                        run_dir=resolved.parent,
                        spec={"cwd": "/opt/ai-toolkit", "argv": ["python", "run.py"], "env": {}},
                        image="example:image",
                        dockerfile="docker/ai-toolkit/Dockerfile",
                        mounts=[],
                        gpu=False,
                        dry_run=False,
                    )
            preflight.assert_not_called()

    def test_compiled_plan_checks_source_stat_without_materializing_the_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)
            lock = freeze_dataset_handoff(
                run,
                workspace,
                resolved,
                backend="ai-toolkit",
                project=lambda selection: project_ai_toolkit_dataset(run, selection),
            )

            records = _dataset_layout_preflight_report(run, workspace)

            self.assertEqual(records[0]["severity"], "info")
            self.assertIn("stat matches", records[0]["fact"])
            self.assertFalse((workspace / lock["views"][0]["root"]).exists())
            (workspace / "datasets" / "tiny" / "a.txt").write_text("changed", encoding="utf-8")
            changed = _dataset_layout_preflight_report(run, workspace)
            self.assertEqual(changed[0]["severity"], "error")
            self.assertIn("changed since compile", changed[0]["fact"])

    def test_view_cleanup_rejects_a_symlinked_parent_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_dir = workspace / "runs" / "example"
            external = workspace / "external-cache"
            target = external / "dataset-view"
            target.mkdir(parents=True)
            (target / "keep.txt").write_text("keep", encoding="utf-8")
            run_dir.mkdir(parents=True)
            (run_dir / "cache").symlink_to(external, target_is_directory=True)
            lock = {
                "views": [{"root": "runs/example/cache/dataset-view/ai-toolkit/tiny"}],
            }

            with self.assertRaisesRegex(ValueError, "symlink"):
                remove_dataset_views(workspace, run_dir, lock)

            self.assertTrue((target / "keep.txt").is_file())


if __name__ == "__main__":
    unittest.main()
