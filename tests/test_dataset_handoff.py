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

    @staticmethod
    def image_projection(selection: dict) -> dict:
        dataset = selection["datasets"][0]
        sample = dataset["samples"][0]
        target = sample["files"][0]
        caption = sample["caption"]
        view_root = "runs/example/cache/dataset-view/ai-toolkit/tiny"
        return {
            "schema_version": 1,
            "backend": "ai-toolkit",
            "datasets": [{
                "id": "tiny",
                "consumed": [target["input_id"], caption["input_id"]],
                "unrepresentable": [],
                "semantic": {"caption_ext": ".txt"},
                "bindings": [{
                    "rule": "same-stem",
                    "inputs": [target["input_id"], caption["input_id"]],
                }],
                "native_runtime": {"folder_path": "/workspace/" + view_root},
                "native": {"caption_ext": ".txt", "folder_path": "/workspace/" + view_root},
                "view": {
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
                },
            }],
        }

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

    def test_projection_missing_one_input_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def drops_caption(selection: dict) -> dict:
                projection = self.image_projection(selection)
                projection["datasets"][0]["consumed"].pop()
                return projection

            with self.assertRaisesRegex(ValueError, "consumed inputs differ from materialized view inputs"):
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
                projection["datasets"][0]["view"]["links"] = []
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
                projection["datasets"][0]["view"]["files"][0]["path"] = (
                    "runs/example/cache/dataset-view/ai-toolkit/tiny/000001.txt"
                )
                return projection

            with self.assertRaisesRegex(ValueError, "violates same-stem pairing"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=shifted,
                )

    def test_projection_same_stem_in_different_directories_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run, resolved = self.make_run(workspace)

            def shifted_directory(selection: dict) -> dict:
                projection = self.image_projection(selection)
                projection["datasets"][0]["view"]["files"][0]["path"] = (
                    "runs/example/cache/dataset-view/ai-toolkit/tiny/other/000000.txt"
                )
                return projection

            with self.assertRaisesRegex(ValueError, "violates same-stem pairing"):
                freeze_dataset_handoff(
                    run, workspace, resolved, backend="ai-toolkit", project=shifted_directory,
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
                        "bindings": [{
                            "rule": "same-stem",
                            "inputs": [target["input_id"]],
                            "allow_singleton": True,
                        }],
                        "view": {
                            "root": view_root,
                            "links": [{
                                "path": view_root + "/000000.png",
                                "target": "/workspace/datasets/tiny/a.png",
                                "input_id": target["input_id"],
                            }],
                            "files": [],
                        },
                    }],
                }

            lock = freeze_dataset_handoff(
                run, workspace, resolved, backend="ai-toolkit", project=singleton,
            )
            self.assertEqual(len(lock["files"]), 1)

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
                projection["datasets"][0]["view"]["links"][0]["target"] = (
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
                projection["datasets"][0]["view"]["files"][0]["text"] = "different caption"
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
