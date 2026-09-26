"""Manifest selection, projection completeness, and run-view contracts."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from kura.backends.ai_toolkit import compile_ai_toolkit, project_ai_toolkit_dataset
from kura.cli import cmd_run_compile
from kura.dataset_handoff import freeze_dataset_handoff, inspect_dataset_handoff, inspect_dataset_sources, materialize_dataset_view, remove_dataset_views
from kura.dataset_handoff import local_training_mounts
from kura.executors.docker import docker_command, launch_docker
from kura.paths import inspect_workspace_symlinks
from kura.run_commands.plan import _dataset_layout_preflight_report


class DatasetHandoffTests(unittest.TestCase):
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
