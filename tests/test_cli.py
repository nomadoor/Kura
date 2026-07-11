"""Small regression tests for workspace initialization."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import importlib.util
import os
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

import yaml

from kura.backends import MUSUBI_ADAPTER_SCRIPTS, _safetensors_validator_code, command_ai_toolkit, command_musubi_tuner, compile_ai_toolkit, compile_musubi_tuner
from kura.cli import _docker_cleanup_image, _load_env_local, _notification_channels, _notify, _parse_duration_seconds, _runpod_run_over_ssh, _runpod_secret_env_payload, _select_remote_outputs, _sync_runpod_remote_stdout, _workspace, cmd_cleanup, cmd_dataset_validate, cmd_doctor_comfyui, cmd_doctor_disk, cmd_doctor_docker, cmd_doctor_musubi, cmd_doctor_runpod, cmd_doctor_workspace, cmd_fix_links, cmd_fix_permissions, cmd_image_build, cmd_init, cmd_monitor, cmd_render_new, cmd_run_compile, cmd_run_discard, cmd_run_download, cmd_run_launch, cmd_run_new, cmd_run_plan, cmd_run_prune, cmd_run_reconcile, cmd_run_remote, cmd_run_status
from kura.container_scripts import script_source
from kura.executors import _redact_secret_text, docker_command, docker_preflight, launch_runpod, launch_runpod_session, reconcile_docker, reconcile_runpod, stage_runpod, stop_runpod
from kura.executors.common import _safe_env
from kura.init_templates import RUNPOD_OBJECT_JOB_TEMPLATE
from kura.monitor import collect_run_summaries, _read_activity_from_stdout
from kura.render import _cleanup_lora_stage, _ensure_lora_stage_visible, insert_lora_loader, _materialize_lora_stage, _safe_stage_name, compile_render, launch_render
from kura.run_commands import _as_positive_int, _checkpoint_safety_preflight, _configured_gib, _ensure_free_bytes, _estimate_musubi_download_bytes, _local_launch_disk_preflight, _render_runpod_lora, _runpod_launch_disk_preflight, _runpod_ssh_details, _scp_to_runpod, _start_runpod_comfyui, _start_runpod_session_lease_guard, execute_run, launch_run, plan_run, stop_run
from kura.run_commands.plan import _disk_warnings, _hf_file_size_probe, _model_download_preflight_report, _model_download_safety_preflight
from kura.run_commands.runpod_ssh import _runpod_remote_job_script
from kura.storage import StorageStatus, probe_storage
from kura.tui import KuraMonitorApp, _compact_path


class InitCommandTests(unittest.TestCase):
    def test_cli_version_and_help_text(self) -> None:
        command = [sys.executable, "-c", "from kura.cli import main; main()"]
        version = subprocess.run([*command, "--version"], text=True, capture_output=True, check=False)
        self.assertEqual(version.returncode, 0)
        self.assertIn("kura 0.1.0", version.stdout)

        help_result = subprocess.run([*command, "--help"], text=True, capture_output=True, check=False)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("Agent-first, file-first workspace", help_result.stdout)
        self.assertIn("Create the workspace folders and default config", help_result.stdout)

        run_help = subprocess.run([*command, "run", "--help"], text=True, capture_output=True, check=False)
        self.assertEqual(run_help.returncode, 0)
        self.assertIn("Run on RunPod, download outputs, then auto-stop", run_help.stdout)

        doctor_help = subprocess.run([*command, "doctor", "--help"], text=True, capture_output=True, check=False)
        self.assertEqual(doctor_help.returncode, 0)
        self.assertIn("Report local disk, cache, Docker storage", doctor_help.stdout)
        self.assertIn("Check RunPod API, Pods, and Network Volumes", doctor_help.stdout)
        self.assertIn("Smoke-test Musubi adapter scripts", doctor_help.stdout)

    def test_init_creates_required_files_and_is_idempotent(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                self.assertEqual(cmd_init(argparse.Namespace()), 0)
                self.assertEqual(cmd_init(argparse.Namespace()), 0)
                root = Path(directory)
                for relative in ("workspace.yaml", "AGENTS.md", "index.jsonl", "datasets", "runs", "workflows", "promptsets", "cache/huggingface", "cache/models", "docker/ai-toolkit/Dockerfile"):
                    self.assertTrue((root / relative).exists(), relative)
                self.assertTrue((root / "docker/musubi-tuner/Dockerfile").exists())
                self.assertTrue((root / "docker/ai-toolkit/kura_runpod_object_job.py").exists())
                for dockerfile in (root / "docker/ai-toolkit/Dockerfile", root / "docker/musubi-tuner/Dockerfile"):
                    for line in dockerfile.read_text(encoding="utf-8").splitlines():
                        if line.startswith("COPY "):
                            source = line.split()[1]
                            self.assertTrue((root / source).exists(), f"{dockerfile}: {source}")
                workspace = yaml.safe_load((root / "workspace.yaml").read_text(encoding="utf-8"))
                self.assertEqual(workspace["docker"]["mounts"][0]["source"], "./cache/huggingface")
                self.assertEqual(workspace["docker"]["mounts"][0]["target"], "/workspace/cache/huggingface")
                self.assertEqual(workspace["runpod"]["gpu_type_ids"], ["NVIDIA RTX A5000", "NVIDIA A40"])
                self.assertEqual(workspace["runpod"]["gpu_type_priority"], "custom")
                self.assertEqual(workspace["runpod"]["default_image"]["ai-toolkit"], "ostris/aitoolkit:0.10.22")
                self.assertEqual(workspace["comfyui"]["lora_dir"], "")
                self.assertEqual(workspace["comfyui"]["lora_stage_cleanup"], "remove_after_render")
                self.assertIn("AI_TOOLKIT_IMAGE=ostris/aitoolkit:0.10.22@sha256:", (root / "docker/ai-toolkit/Dockerfile").read_text(encoding="utf-8"))
                self.assertIn("MUSUBI_TUNER_REF=v0.3.4", (root / "docker/musubi-tuner/Dockerfile").read_text(encoding="utf-8"))
            finally:
                os.chdir(previous)

    def test_run_new_accepts_backend_and_executor(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                self.assertEqual(cmd_init(argparse.Namespace()), 0)
                stdout = io.StringIO()
                with patch("sys.stdout", stdout):
                    code = cmd_run_new(argparse.Namespace(experiment="exp", slug="krea2-run", backend="musubi-tuner", executor="runpod", gpu="NVIDIA RTX A5000"))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            run_id = stdout.getvalue().strip()
            run = yaml.safe_load((Path(directory) / "runs" / run_id / "run.yaml").read_text(encoding="utf-8"))
            self.assertEqual(run["backend"]["name"], "musubi-tuner")
            self.assertEqual(run["compute"]["executor"], "runpod")
            self.assertEqual(run["compute"]["gpu"], "NVIDIA RTX A5000")
            self.assertEqual(
                sorted(path.name for path in (Path(directory) / "runs" / run_id).iterdir()),
                ["notes.md", "plan.md", "run.yaml", "status.json"],
            )

    def test_render_new_creates_only_draft_files(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                self.assertEqual(cmd_init(argparse.Namespace()), 0)
                stdout = io.StringIO()
                with patch("sys.stdout", stdout):
                    code = cmd_render_new(argparse.Namespace(slug="render-draft"))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            run_id = stdout.getvalue().strip()
            self.assertEqual(
                sorted(path.name for path in (Path(directory) / "runs" / run_id).iterdir()),
                ["notes.md", "plan.md", "run.yaml", "status.json"],
            )

    def test_run_compile_rejects_musubi_dataset_without_images(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chdir(root)
            try:
                self.assertEqual(cmd_init(argparse.Namespace()), 0)
                dataset = root / "datasets" / "tiny"
                dataset.mkdir(parents=True)
                (dataset / "dataset.yaml").write_text("id: tiny\n", encoding="utf-8")
                (dataset / "items.jsonl").write_text("{}\n", encoding="utf-8")
                stdout = io.StringIO()
                with patch("sys.stdout", stdout):
                    self.assertEqual(cmd_run_new(argparse.Namespace(experiment="exp", slug="empty-musubi", backend="musubi-tuner", executor="docker", gpu=None)), 0)
                run_id = stdout.getvalue().strip()
                run_path = root / "runs" / run_id / "run.yaml"
                run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
                run["model"]["base"] = "black-forest-labs/FLUX.2-klein-base-4B"
                run["datasets"] = [{"id": "tiny"}]
                run["backend_overrides"] = {"musubi-tuner": {"architecture": "flux2", "model_paths": {"dit": "/models/dit.safetensors", "vae": "/models/vae.safetensors", "text_encoder": "/models/text.safetensors"}}}
                run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
                stderr = io.StringIO()
                with patch("sys.stderr", stderr):
                    code = cmd_run_compile(argparse.Namespace(run_id=run_id))
                manifest_exists = (root / "runs" / run_id / "resolved" / "manifest.lock.yaml").exists()
            finally:
                os.chdir(previous)
        self.assertEqual(code, 1)
        self.assertIn("dataset tiny has no images/ directory and no image files at its root", stderr.getvalue())
        self.assertFalse(manifest_exists)

    def test_run_compile_rejects_backend_override_mismatch(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chdir(root)
            try:
                self.assertEqual(cmd_init(argparse.Namespace()), 0)
                dataset = root / "datasets" / "tiny" / "images"
                dataset.mkdir(parents=True)
                (root / "datasets" / "tiny" / "dataset.yaml").write_text("id: tiny\n", encoding="utf-8")
                (root / "datasets" / "tiny" / "items.jsonl").write_text("{}\n", encoding="utf-8")
                (dataset / "001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
                stdout = io.StringIO()
                with patch("sys.stdout", stdout):
                    self.assertEqual(cmd_run_new(argparse.Namespace(experiment="exp", slug="wrong-backend", backend="ai-toolkit", executor="docker", gpu=None)), 0)
                run_id = stdout.getvalue().strip()
                run_path = root / "runs" / run_id / "run.yaml"
                run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
                run["model"]["base"] = "black-forest-labs/FLUX.2-klein-base-4B"
                run["datasets"] = [{"id": "tiny"}]
                run["backend_overrides"] = {"musubi-tuner": {"architecture": "flux2"}}
                run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
                stderr = io.StringIO()
                with patch("sys.stderr", stderr):
                    code = cmd_run_compile(argparse.Namespace(run_id=run_id))
            finally:
                os.chdir(previous)
        self.assertEqual(code, 1)
        self.assertIn("backend is ai-toolkit but backend_overrides.musubi-tuner is set", stderr.getvalue())

    def test_run_compile_cleans_up_after_invalid_musubi_model_downloads(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chdir(root)
            try:
                self.assertEqual(cmd_init(argparse.Namespace()), 0)
                dataset = root / "datasets" / "tiny"
                (dataset / "images").mkdir(parents=True)
                (dataset / "images" / "001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
                (dataset / "dataset.yaml").write_text("id: tiny\nstats:\n  count: 1\n", encoding="utf-8")
                (dataset / "items.jsonl").write_text('{"id":"1","path":"images/001.png","caption":"ok","hash":"sha256:abc"}\n', encoding="utf-8")
                stdout = io.StringIO()
                with patch("sys.stdout", stdout):
                    self.assertEqual(cmd_run_new(argparse.Namespace(experiment="exp", slug="bad-download", backend="musubi-tuner", executor="docker", gpu=None)), 0)
                run_id = stdout.getvalue().strip()
                run_path = root / "runs" / run_id / "run.yaml"
                run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
                run["model"]["base"] = "example/model"
                run["datasets"] = [{"id": "tiny"}]
                run["backend_overrides"] = {
                    "musubi-tuner": {
                        "architecture": "flux2",
                        "model_bundle": "none",
                        "model_downloads": {"dit": "not-a-download-mapping"},
                    }
                }
                run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
                stderr = io.StringIO()
                with patch("sys.stderr", stderr):
                    code = cmd_run_compile(argparse.Namespace(run_id=run_id))
                resolved_exists = (root / "runs" / run_id / "resolved").exists()
            finally:
                os.chdir(previous)
        self.assertEqual(code, 1)
        self.assertIn("model_downloads must map model keys to download mappings", stderr.getvalue())
        self.assertFalse(resolved_exists)

    def test_dataset_validate_checks_referenced_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "tiny"
            dataset.mkdir(parents=True)
            (dataset / "dataset.yaml").write_text("id: tiny\nstats:\n  count: 1\n", encoding="utf-8")
            (dataset / "items.jsonl").write_text('{"id":"1","path":"images/001.png","caption":"ok","hash":"sha256:abc"}\n', encoding="utf-8")
            stderr = io.StringIO()
            with patch("sys.stderr", stderr):
                self.assertEqual(cmd_dataset_validate(argparse.Namespace(dataset_dir=str(dataset))), 1)
            self.assertIn("referenced file does not exist", stderr.getvalue())
            (dataset / "images").mkdir()
            (dataset / "images" / "001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            self.assertEqual(cmd_dataset_validate(argparse.Namespace(dataset_dir=str(dataset))), 0)

    def test_dataset_validate_rejects_paths_outside_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "tiny"
            dataset.mkdir(parents=True)
            outside = root / "outside.png"
            outside.write_bytes(b"\x89PNG\r\n\x1a\n")
            (dataset / "dataset.yaml").write_text("id: tiny\nstats:\n  count: 1\n", encoding="utf-8")
            (dataset / "items.jsonl").write_text('{"id":"1","path":"../../outside.png","caption":"ok","hash":"sha256:abc"}\n', encoding="utf-8")

            stderr = io.StringIO()
            with patch("sys.stderr", stderr):
                self.assertEqual(cmd_dataset_validate(argparse.Namespace(dataset_dir=str(dataset))), 1)

            self.assertIn("path must stay inside the dataset directory", stderr.getvalue())

    def test_run_compile_preserves_requested_executor_in_env_lock(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chdir(root)
            try:
                self.assertEqual(cmd_init(argparse.Namespace()), 0)
                dataset = root / "datasets" / "tiny"
                (dataset / "images").mkdir(parents=True)
                (dataset / "images" / "001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
                (dataset / "dataset.yaml").write_text("id: tiny\nstats:\n  count: 1\n", encoding="utf-8")
                (dataset / "items.jsonl").write_text('{"id":"1","path":"images/001.png","caption":"ok","hash":"sha256:abc"}\n', encoding="utf-8")
                stdout = io.StringIO()
                with patch("sys.stdout", stdout):
                    self.assertEqual(cmd_run_new(argparse.Namespace(experiment="exp", slug="runpod", backend="ai-toolkit", executor="runpod", gpu="NVIDIA A40")), 0)
                run_id = stdout.getvalue().strip()
                run_path = root / "runs" / run_id / "run.yaml"
                run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
                run["model"]["base"] = "black-forest-labs/FLUX.2-klein-base-4B"
                run["datasets"] = [{"id": "tiny"}]
                run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
                self.assertEqual(cmd_run_compile(argparse.Namespace(run_id=run_id)), 0)
            finally:
                os.chdir(previous)

            env_lock = yaml.safe_load((root / "runs" / run_id / "resolved" / "env.lock").read_text(encoding="utf-8"))
            self.assertEqual(env_lock["declared_executor"], "runpod")
            requirements_lock = yaml.safe_load((root / "runs" / run_id / "resolved" / "model-requirements.lock.yaml").read_text(encoding="utf-8"))
            self.assertEqual(requirements_lock["schema_version"], 1)
            self.assertEqual(requirements_lock["requirements"][0]["acquisition"], "backend")
            self.assertEqual(requirements_lock["requirements"][0]["identity"]["repo_id"], "black-forest-labs/FLUX.2-klein-base-4B")

    def test_init_repairs_cache_directories_in_existing_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                Path("workspace.yaml").write_text("schema_version: 1\nname: existing\n", encoding="utf-8")
                self.assertEqual(cmd_init(argparse.Namespace()), 0)
                self.assertTrue((Path(directory) / "cache" / "huggingface").is_dir())
                self.assertTrue((Path(directory) / "cache" / "models").is_dir())
                self.assertEqual(yaml.safe_load((Path(directory) / "workspace.yaml").read_text(encoding="utf-8"))["name"], "existing")
            finally:
                os.chdir(previous)

    def test_object_job_template_rejects_download_keys_outside_workspace(self) -> None:
        start = RUNPOD_OBJECT_JOB_TEMPLATE.index("def download_prefix")
        end = RUNPOD_OBJECT_JOB_TEMPLATE.index("\n\ndef upload_tree")
        namespace: dict[str, Any] = {"Path": Path}
        exec(RUNPOD_OBJECT_JOB_TEMPLATE[start:end], namespace)

        class FakePaginator:
            def paginate(self, **_: object) -> list[dict[str, object]]:
                return [{"Contents": [{"Key": "prefix/../../escape.txt"}]}]

        class FakeClient:
            def get_paginator(self, _: str) -> FakePaginator:
                return FakePaginator()

            def download_file(self, *_: object) -> None:
                raise AssertionError("unsafe key should not be downloaded")

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "unsafe object key"):
                namespace["download_prefix"](FakeClient(), "bucket", "prefix", Path(directory))


class ImageCommandTests(unittest.TestCase):
    def test_image_build_resolves_paths_from_workspace_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "datasets" / "tiny"
            nested.mkdir(parents=True)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({
                    "docker": {
                        "images": {
                            "ai-toolkit": {
                                "local": "kura/ai-toolkit:test",
                                "remote": "registry.example/kura/ai-toolkit:test",
                                "dockerfile": "docker/ai-toolkit/Dockerfile",
                                "context": ".",
                            },
                        },
                    },
                }),
                encoding="utf-8",
            )
            calls: list[list[str]] = []

            def fake_docker_run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, "sha256:image\n" if capture else "", "")

            previous = Path.cwd()
            os.chdir(nested)
            try:
                with patch("kura.cli._docker_run", side_effect=fake_docker_run), patch("kura.cli._docker_storage_summary", return_value={"usage": []}):
                    self.assertEqual(cmd_image_build(argparse.Namespace(name="ai-toolkit", ref=None)), 0)
            finally:
                os.chdir(previous)

            self.assertGreaterEqual(len(calls), 1)
            build = calls[0]
            self.assertEqual(build[build.index("--file") + 1], str(root / "docker/ai-toolkit/Dockerfile"))
            self.assertIn(
                "AI_TOOLKIT_IMAGE=ostris/aitoolkit:0.10.22@sha256:5a810f50de920aaa3439487959ae392bf0d1458345baddee24a7bf33787c0438",
                build,
            )
            self.assertEqual(build[-1], str(root))

    def test_ai_toolkit_image_build_ref_overrides_upstream_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({
                    "docker": {
                        "images": {
                            "ai-toolkit": {
                                "local": "kura/ai-toolkit:test",
                                "remote": "registry.example/kura/ai-toolkit:test",
                                "dockerfile": "docker/ai-toolkit/Dockerfile",
                                "context": ".",
                            },
                        },
                    },
                }),
                encoding="utf-8",
            )
            commands: list[list[str]] = []

            def fake_docker_run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, "sha256:image\n" if capture else "", "")

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.cli._docker_run", side_effect=fake_docker_run), patch("kura.cli._docker_storage_summary", return_value={"usage": []}):
                    self.assertEqual(cmd_image_build(argparse.Namespace(name="ai-toolkit", ref="ostris/aitoolkit:custom")), 0)
            finally:
                os.chdir(previous)

            self.assertIn("AI_TOOLKIT_IMAGE=ostris/aitoolkit:custom", commands[0])

    def test_image_build_rejects_large_build_cache_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({
                    "docker": {
                        "images": {
                            "ai-toolkit": {
                                "local": "kura/ai-toolkit:test",
                                "remote": "registry.example/kura/ai-toolkit:test",
                                "dockerfile": "docker/ai-toolkit/Dockerfile",
                                "context": ".",
                            },
                        },
                    },
                }),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.cli._docker_storage_summary", return_value={"usage": [{"Type": "Build Cache", "size_bytes": 31 * 1024**3}]}),
                    patch("sys.stderr", new_callable=__import__("io").StringIO) as stderr,
                ):
                    code = cmd_image_build(argparse.Namespace(name="ai-toolkit", ref=None, allow_large_build_cache=False))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            self.assertIn("Docker build cache exceeds 30GiB", stderr.getvalue())


class DoctorDockerTests(unittest.TestCase):
    def test_cleanup_all_is_dry_run_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (root / "cache" / "huggingface").mkdir(parents=True)
            (root / "cache" / "models").mkdir(parents=True, exist_ok=True)
            (root / "runs" / "example").mkdir(parents=True)
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.cli._path_size_bytes", return_value=123),
                    patch("kura.cli._docker_storage_summary", return_value={"daemon_reachable": True, "usage": []}),
                    patch("kura.cli._root_owned_files", return_value={"supported": True, "count": 0, "samples": []}),
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                ):
                    code = cmd_cleanup(argparse.Namespace(target="all", keep_last=30, delete_final_artifacts=False, yes=False))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["workspace_root"], str(root))
            self.assertIn("cache/huggingface", {item.get("target") for item in payload["actions"]})
            self.assertIn("docker system", {item.get("target") for item in payload["actions"]})

    def test_cleanup_image_uses_available_remote_name_when_local_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump(
                    {
                        "docker": {
                            "images": {
                                "ai-toolkit": {
                                    "local": "kura/ai-toolkit:test",
                                    "remote": "registry.example/kura/ai-toolkit:test",
                                    "dockerfile": "Dockerfile",
                                    "context": ".",
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.cli._docker_image_exists", side_effect=lambda image: image == "registry.example/kura/ai-toolkit:test"):
                    self.assertEqual(_docker_cleanup_image(), "registry.example/kura/ai-toolkit:test")
            finally:
                os.chdir(previous)

    def test_cleanup_runs_keeps_outputs_without_explicit_final_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "old"
            (run / "cache").mkdir(parents=True)
            (run / "outputs").mkdir()
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (run / "run.yaml").write_text("id: old\ncreated: '2026-01-01T00:00:00+00:00'\n", encoding="utf-8")
            (run / "status.json").write_text(json.dumps({"state": "completed", "ended": "2026-01-01T00:00:00+00:00"}), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.cli._path_size_bytes", return_value=1),
                    patch("kura.cli._root_owned_files", return_value={"supported": True, "count": 0, "samples": []}),
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                ):
                    code = cmd_cleanup(argparse.Namespace(target="runs", keep_last=0, delete_final_artifacts=False, yes=True))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertFalse((run / "cache").exists())
            self.assertTrue((run / "outputs").exists())
            self.assertFalse(payload["dry_run"])

    def test_cleanup_runs_requires_explicit_final_delete_for_whole_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "old"
            (run / "outputs").mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (run / "run.yaml").write_text("id: old\ncreated: '2026-01-01T00:00:00+00:00'\n", encoding="utf-8")
            (run / "status.json").write_text(json.dumps({"state": "completed", "ended": "2026-01-01T00:00:00+00:00"}), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.cli._path_size_bytes", return_value=1),
                    patch("kura.cli._root_owned_files", return_value={"supported": True, "count": 0, "samples": []}),
                    patch("sys.stdout", new_callable=__import__("io").StringIO),
                ):
                    self.assertEqual(cmd_cleanup(argparse.Namespace(target="runs", keep_last=0, delete_final_artifacts=True, yes=True)), 0)
            finally:
                os.chdir(previous)
            self.assertFalse(run.exists())

    def test_fix_permissions_dry_run_reports_root_owned_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (root / "cache").mkdir()
            (root / "runs").mkdir()
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.cli._root_owned_files", return_value={"supported": True, "count": 1, "samples": ["cache/root"]}),
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                ):
                    code = cmd_fix_permissions(argparse.Namespace(target="all", yes=False))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(payload["dry_run"])
            self.assertEqual(payload["root_owned"]["count"], 1)

    def test_fix_links_rewrites_repairable_container_private_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({"docker": {"mounts": [{"source": "./cache/huggingface", "target": "/root/.cache/huggingface", "mode": "rw"}]}}),
                encoding="utf-8",
            )
            link = root / "cache" / "models" / "musubi" / "repo--model" / "dit" / "weights.safetensors"
            link.parent.mkdir(parents=True)
            link.symlink_to("/root/.cache/huggingface/hub/models--repo--model/snapshots/abc/weights.safetensors")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_fix_links(argparse.Namespace(yes=True))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertFalse(payload["dry_run"])
            self.assertEqual(len(payload["actions"]), 1)
            self.assertEqual(
                os.readlink(link),
                "../../../../huggingface/hub/models--repo--model/snapshots/abc/weights.safetensors",
            )

    def test_fix_links_reports_unmapped_absolute_symlink_without_deleting(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(yaml.safe_dump({"docker": {"mounts": []}}), encoding="utf-8")
            link = root / "cache" / "models" / "bad.safetensors"
            link.parent.mkdir(parents=True)
            link.symlink_to("/opt/models/bad.safetensors")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_fix_links(argparse.Namespace(yes=True))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["actions"][0]["repairable"], False)
            self.assertEqual(os.readlink(link), "/opt/models/bad.safetensors")

    def test_doctor_disk_reports_workspace_storage_and_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump(
                    {
                        "docker": {
                            "mounts": [
                                {"source": "./cache/huggingface", "target": "/root/.cache/huggingface", "mode": "rw"}
                            ]
                        }
                    }
                ),
                encoding="utf-8",
            )
            (root / "cache" / "huggingface").mkdir(parents=True)
            (root / "runs").mkdir()

            def fake_disk_usage(path: Path) -> dict[str, object]:
                return {"path": str(path), "probe": str(path), "total_bytes": 200 * 1024**3, "used_bytes": 150 * 1024**3, "free_bytes": 50 * 1024**3}

            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.doctor._path_size_bytes", return_value=20),
                    patch("kura.doctor._disk_usage_for", side_effect=fake_disk_usage),
                    patch("kura.doctor._docker_storage_summary", return_value={"daemon_reachable": True, "usage": [{"Type": "Build Cache", "size_bytes": 31 * 1024**3}], "kura_managed": {}}),
                    patch("kura.doctor._root_owned_files", return_value={"supported": True, "count": 2, "samples": ["cache/root-owned"], "truncated": False}),
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                ):
                    code = cmd_doctor_disk(argparse.Namespace())
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertEqual(payload["workspace_root"], str(root))
            self.assertEqual(payload["sizes"]["huggingface_cache"]["path"], str(root / "cache" / "huggingface"))
            self.assertEqual(payload["issues"][0]["severity"], "warning")
            self.assertIn("workspace filesystem has less than 100GiB free", payload["warnings"])
            self.assertIn("Docker build cache exceeds 30GiB", payload["warnings"])
            self.assertIn("cache/runs contain root-owned files; cleanup may require permission repair", payload["warnings"])

    def test_doctor_disk_reports_large_cache_runs_as_advisory_when_space_is_ok(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(yaml.safe_dump({"docker": {"mounts": []}}), encoding="utf-8")
            (root / "cache").mkdir()
            (root / "runs").mkdir()
            sizes = {
                str(root / "cache"): 31 * 1024**3,
                str(root / "runs"): 1 * 1024**3,
            }

            def fake_size(path: Path) -> int:
                return sizes.get(str(path), 0)

            def fake_disk_usage(path: Path) -> dict[str, object]:
                return {"path": str(path), "probe": str(path), "total_bytes": 500 * 1024**3, "used_bytes": 100 * 1024**3, "free_bytes": 400 * 1024**3}

            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.doctor._path_size_bytes", side_effect=fake_size),
                    patch("kura.doctor._disk_usage_for", side_effect=fake_disk_usage),
                    patch("kura.doctor._docker_storage_summary", return_value={"daemon_reachable": True, "usage": [], "kura_managed": {}}),
                    patch("kura.doctor._root_owned_files", return_value={"supported": True, "count": 0, "samples": [], "truncated": False}),
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                ):
                    code = cmd_doctor_disk(argparse.Namespace())
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["warnings"], [])
            self.assertIn("workspace cache+runs exceed 30GiB", payload["advisories"])
            issue = next(item for item in payload["issues"] if item["code"] == "workspace_cache_runs_large")
            self.assertEqual(issue["severity"], "advisory")
            self.assertEqual(issue["size_bytes"], 32 * 1024**3)

    def test_doctor_disk_warns_about_container_private_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({"docker": {"mounts": [{"source": "./cache/huggingface", "target": "/root/.cache/huggingface", "mode": "rw"}]}}),
                encoding="utf-8",
            )
            (root / "cache" / "huggingface").mkdir(parents=True)
            link = root / "cache" / "models" / "bad.safetensors"
            link.parent.mkdir(parents=True)
            link.symlink_to("/root/.cache/huggingface/hub/models--repo--model/snapshots/abc/weights.safetensors")

            def fake_disk_usage(path: Path) -> dict[str, object]:
                return {"path": str(path), "probe": str(path), "total_bytes": 500 * 1024**3, "used_bytes": 100 * 1024**3, "free_bytes": 400 * 1024**3}

            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.doctor._path_size_bytes", return_value=0),
                    patch("kura.doctor._disk_usage_for", side_effect=fake_disk_usage),
                    patch("kura.doctor._docker_storage_summary", return_value={"daemon_reachable": True, "usage": [], "kura_managed": {}}),
                    patch("kura.doctor._root_owned_files", return_value={"supported": True, "count": 0, "samples": [], "truncated": False}),
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                ):
                    code = cmd_doctor_disk(argparse.Namespace())
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertIn("workspace contains symlinks with container-private or workspace-external absolute targets", payload["warnings"])
            self.assertEqual(payload["symlinks"]["unsafe"][0]["path"], "cache/models/bad.safetensors")
            self.assertEqual(payload["symlinks"]["unsafe"][0]["workspace_target"], "cache/huggingface/hub/models--repo--model/snapshots/abc/weights.safetensors")

    def test_doctor_disk_warns_about_wsl_ext4_virtual_free_space(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(yaml.safe_dump({"docker": {"mounts": []}}), encoding="utf-8")
            (root / "cache").mkdir()
            (root / "runs").mkdir()

            def fake_disk_usage(path: Path) -> dict[str, object]:
                return {"path": str(path), "probe": str(path), "total_bytes": 1000 * 1024**3, "used_bytes": 100 * 1024**3, "free_bytes": 900 * 1024**3}

            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.doctor._path_size_bytes", return_value=0),
                    patch("kura.doctor._disk_usage_for", side_effect=fake_disk_usage),
                    patch("kura.doctor._docker_storage_summary", return_value={"daemon_reachable": True, "usage": [], "kura_managed": {}}),
                    patch("kura.doctor._root_owned_files", return_value={"supported": True, "count": 0, "samples": [], "truncated": False}),
                    patch("kura.storage.is_wsl", return_value=True),
                    patch("kura.storage._findmnt_for", return_value={"available": True, "fstype": "ext4", "target": "/", "source": "/dev/sdd"}),
                    patch("kura.storage._auto_wsl_host_drive", return_value=None),
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                ):
                    code = cmd_doctor_disk(argparse.Namespace())
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertEqual(payload["storage"]["workspace"]["confidence"], "unknown")
            self.assertTrue(any("WSL Linux ext4" in warning for warning in payload["warnings"]))

    def test_wsl_storage_probe_treats_unknown_backing_as_unknown(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("kura.storage.is_wsl", return_value=True),
                patch("kura.storage._findmnt_for", return_value={"available": False, "reason": "findmnt not found"}),
            ):
                status = probe_storage(root, role="workspace")
        self.assertEqual(status.backing_kind, "wsl2")
        self.assertEqual(status.confidence, "unknown")
        self.assertIn("could not identify the physical backing store", status.warning or "")

    def test_doctor_disk_uses_auto_detected_wsl_host_drive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(yaml.safe_dump({"docker": {"mounts": []}}), encoding="utf-8")
            (root / "cache").mkdir()
            (root / "runs").mkdir()

            def fake_disk_usage(path: Path) -> dict[str, object]:
                return {"path": str(path), "probe": str(path), "total_bytes": 1000 * 1024**3, "used_bytes": 100 * 1024**3, "free_bytes": 900 * 1024**3}

            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.doctor._path_size_bytes", return_value=0),
                    patch("kura.doctor._disk_usage_for", side_effect=fake_disk_usage),
                    patch("kura.doctor._docker_storage_summary", return_value={"daemon_reachable": True, "usage": [], "kura_managed": {}}),
                    patch("kura.doctor._root_owned_files", return_value={"supported": True, "count": 0, "samples": [], "truncated": False}),
                    patch("kura.storage.is_wsl", return_value=True),
                    patch("kura.storage._findmnt_for", return_value={"available": True, "fstype": "ext4", "target": "/", "source": "/dev/sdd"}),
                    patch("kura.storage._auto_wsl_host_drive", return_value="F:"),
                    patch("kura.storage._windows_drive_free_bytes", return_value=290 * 1024**3),
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                ):
                    code = cmd_doctor_disk(argparse.Namespace())
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["storage"]["workspace"]["backing_id"], "F:")
            self.assertEqual(payload["storage"]["workspace"]["confidence"], "estimated")
            self.assertEqual(payload["storage"]["workspace"]["effective_free_bytes"], 290 * 1024**3)

    def test_doctor_disk_warns_on_low_effective_backing_free_space(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(yaml.safe_dump({"docker": {"mounts": []}}), encoding="utf-8")
            (root / "cache").mkdir()
            (root / "runs").mkdir()

            def fake_disk_usage(path: Path) -> dict[str, object]:
                return {"path": str(path), "probe": str(path), "total_bytes": 1000 * 1024**3, "used_bytes": 100 * 1024**3, "free_bytes": 900 * 1024**3}

            def fake_probe(paths: dict[str, Path], config: dict[str, object] | None = None) -> dict[str, StorageStatus]:
                return {
                    name: StorageStatus(
                        path=str(path),
                        probe=str(path),
                        backing_id="F:",
                        backing_kind="wsl2_vhdx",
                        linux_free_bytes=900 * 1024**3,
                        linux_total_bytes=1000 * 1024**3,
                        host_free_bytes=90 * 1024**3,
                        effective_free_bytes=90 * 1024**3,
                        confidence="estimated",
                        mount={"available": True},
                    )
                    for name, path in paths.items()
                }

            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.doctor._path_size_bytes", return_value=0),
                    patch("kura.doctor._disk_usage_for", side_effect=fake_disk_usage),
                    patch("kura.doctor.probe_storages", side_effect=fake_probe),
                    patch("kura.doctor._docker_storage_summary", return_value={"daemon_reachable": True, "usage": [], "kura_managed": {}}),
                    patch("kura.doctor._root_owned_files", return_value={"supported": True, "count": 0, "samples": [], "truncated": False}),
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                ):
                    code = cmd_doctor_disk(argparse.Namespace())
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertEqual(payload["storage"]["workspace"]["effective_free_bytes"], 90 * 1024**3)
            self.assertIn("workspace backing store has less than 100GiB effective free (F:)", payload["warnings"])

    def test_doctor_docker_reports_kura_managed_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump(
                    {
                        "docker": {
                            "images": {
                                "ai-toolkit": {
                                    "local": "kura/ai-toolkit:test",
                                    "remote": "registry.example/kura/ai-toolkit:test",
                                    "dockerfile": "docker/ai-toolkit/Dockerfile",
                                    "context": ".",
                                }
                            },
                            "mounts": [{"source": "./cache/huggingface", "target": "/root/.cache/huggingface", "mode": "rw"}],
                        }
                    }
                ),
                encoding="utf-8",
            )

            def fake_docker_run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
                text = ""
                if command[:2] == ["docker", "info"] and "--format" not in command:
                    text = "ok"
                elif command[:3] == ["docker", "info", "--format"]:
                    text = "/var/lib/docker\n"
                elif command[:2] == ["docker", "version"]:
                    text = "Docker version\n"
                elif command[:3] == ["docker", "system", "df"]:
                    text = '{"Type":"Images","TotalCount":"1"}\n'
                elif command[:2] == ["docker", "ps"]:
                    text = '{"ID":"abc","Names":"kura-old","State":"exited","Status":"Exited (0)"}\n'
                elif command[:3] == ["docker", "volume", "ls"]:
                    text = '{"Name":"kura-cache","Driver":"local"}\n'
                elif command[:3] == ["docker", "image", "inspect"]:
                    text = "[]\n"
                elif command[:2] == ["docker", "run"] and "/opt/kura-runtime.json" in command:
                    text = "{}\n"
                return subprocess.CompletedProcess(command, 0, text, "")

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.shutil.which", return_value="/usr/bin/docker"), patch("kura.doctor._docker_run", side_effect=fake_docker_run), patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    self.assertEqual(cmd_doctor_docker(argparse.Namespace()), 0)
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            managed = payload["docker_storage"]["kura_managed"]
            self.assertEqual(managed["containers"][0]["Names"], "kura-old")
            self.assertEqual(managed["stopped_containers"][0]["ID"], "abc")
            self.assertEqual(managed["volumes"][0]["Name"], "kura-cache")

    def test_doctor_musubi_reports_adapter_script_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump(
                    {
                        "docker": {
                            "images": {
                                "musubi-tuner": {
                                    "local": "kura/musubi-tuner:test",
                                    "remote": "registry.example/kura/musubi-tuner:test",
                                    "dockerfile": "docker/musubi-tuner/Dockerfile",
                                    "context": ".",
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                if command[:3] == ["/usr/bin/docker", "image", "inspect"]:
                    return subprocess.CompletedProcess(command, 0, "[]", "")
                if command[:3] == ["/usr/bin/docker", "run", "--rm"] and "--entrypoint" in command:
                    results = [
                        {"adapter": adapter, "script": script, "exists": True, "help_returncode": 0}
                        for adapter, scripts in MUSUBI_ADAPTER_SCRIPTS.items()
                        for script in scripts
                    ]
                    return subprocess.CompletedProcess(command, 0, json.dumps({"results": results}) + "\n", "")
                raise AssertionError(command)

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.shutil.which", return_value="/usr/bin/docker"), patch("kura.doctor.subprocess.run", side_effect=fake_run), patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_musubi(argparse.Namespace(skip_help=False, no_gpu=False, timeout=30.0, script_timeout=5.0, image=None))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(payload["checks"]["adapter_scripts_exist"])
            self.assertTrue(payload["checks"]["adapter_help_smoke"])
            self.assertIn({"adapter": "flux2", "script": "flux_2_train_network.py", "exists": True, "help_returncode": 0}, payload["diagnostics"]["scripts"])

    def test_doctor_musubi_accepts_image_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump(
                    {
                        "docker": {
                            "images": {
                                "musubi-tuner": {
                                    "local": "configured/missing:test",
                                    "remote": "remote",
                                    "dockerfile": "Dockerfile",
                                    "context": ".",
                                }
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            seen: list[list[str]] = []

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                seen.append(command)
                if command[:3] == ["/usr/bin/docker", "image", "inspect"]:
                    return subprocess.CompletedProcess(command, 0, "[]", "")
                if command[:3] == ["/usr/bin/docker", "run", "--rm"]:
                    results = [
                        {"adapter": adapter, "script": script, "exists": True}
                        for adapter, scripts in MUSUBI_ADAPTER_SCRIPTS.items()
                        for script in scripts
                    ]
                    return subprocess.CompletedProcess(command, 0, json.dumps({"results": results}) + "\n", "")
                raise AssertionError(command)

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.shutil.which", return_value="/usr/bin/docker"), patch("kura.doctor.subprocess.run", side_effect=fake_run), patch("sys.stdout", new_callable=__import__("io").StringIO):
                    self.assertEqual(cmd_doctor_musubi(argparse.Namespace(skip_help=True, no_gpu=True, timeout=30.0, script_timeout=5.0, image="override/musubi:test")), 0)
            finally:
                os.chdir(previous)
            self.assertIn(["/usr/bin/docker", "image", "inspect", "override/musubi:test"], seen)
            self.assertTrue(any("override/musubi:test" in command for command in seen if command[:3] == ["/usr/bin/docker", "run", "--rm"]))


class MonitorCommandTests(unittest.TestCase):
    def test_monitor_passes_limit_to_textual_app(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.cli.run_textual_monitor", return_value=0) as monitor:
                    code = cmd_monitor(argparse.Namespace(interval=1.5, stale_after=12.0, limit=7, all=True))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            monitor.assert_called_once_with(root, interval=1.5, stale_after=12.0, limit=7, include_drafts=True)


class TuiPathDisplayTests(unittest.TestCase):
    def test_initial_watch_only_exempts_target_draft_from_filter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run_id, state in (("draft-watch", "draft"), ("draft-other", "draft"), ("compiled", "compiled")):
                run_dir = root / "runs" / run_id
                run_dir.mkdir(parents=True)
                (run_dir / "run.yaml").write_text(f"id: {run_id}\ntype: train\n", encoding="utf-8")
                (run_dir / "status.json").write_text(json.dumps({"state": state}), encoding="utf-8")

            app = KuraMonitorApp(root, initial_run_id="draft-watch")
            summaries = app.collect_summaries_cached()

            self.assertEqual({summary.id for summary in summaries}, {"draft-watch", "compiled"})
            self.assertEqual(app.hidden_draft_count, 1)

            app.include_drafts = True
            summaries = app.collect_summaries_cached()

            self.assertEqual({summary.id for summary in summaries}, {"draft-watch", "draft-other", "compiled"})
            self.assertEqual(app.hidden_draft_count, 0)

    def test_compact_path_keeps_tail_at_narrow_widths(self) -> None:
        path = Path("/home/nomax/working-linux/Development/Kura/runs/example/outputs")
        self.assertEqual(_compact_path(path, max_len=1), "…")
        self.assertEqual(len(_compact_path(path, max_len=12)), 12)
        self.assertTrue(_compact_path(path, max_len=12).endswith("outputs"))

    def test_download_activity_shows_item_progress_percent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "stdout.log"
            log.write_text(
                "\n".join(
                    [
                        "[kura] hf download start dit:raw.safetensors attempt 1/4",
                        "[kura] hf download progress dit:raw.safetensors files=10 bytes=1000",
                        "[kura] downloaded dit -> /cache/raw.safetensors",
                        "[kura] downloaded vae -> /cache/vae.safetensors",
                        "[kura] hf download progress text_encoder:qwen.safetensors files=20 bytes=2000",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            activity = _read_activity_from_stdout(log, download_keys=["dit", "vae", "text_encoder"])
        self.assertEqual(activity, "downloading text_encoder qwen.safetensors · 2/3 · 67% · 2.0KB")

    def test_monitor_summary_extracts_download_keys_from_command_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "resolved" / "musubi").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (root / "index.jsonl").write_text(json.dumps({"id": "example"}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text("id: example\ntype: train\n", encoding="utf-8")
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("id: example\ntype: train\nparams: {steps: 1}\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_step": 0}), encoding="utf-8")
            (run_dir / "resolved" / "musubi" / "command.json").write_text(
                json.dumps(
                    {
                        "argv": [
                            "bash",
                            "-lc",
                            "python -c 'pass' '[{\"key\":\"dit\",\"repo_id\":\"r/a\",\"filename\":\"a.safetensors\",\"link_path\":\"/workspace/cache/a\"},{\"key\":\"vae\",\"repo_id\":\"r/b\",\"filename\":\"b.safetensors\",\"link_path\":\"/workspace/cache/b\"}]'",
                        ]
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "logs" / "stdout.log").write_text(
                "[kura] downloaded dit -> /cache/a\n[kura] hf download progress vae:b.safetensors files=2 bytes=2048\n",
                encoding="utf-8",
            )
            summaries = collect_run_summaries(root)
        self.assertEqual(summaries[0].activity, "downloading vae b.safetensors · 1/2 · 50% · 2.0KB")

    def test_monitor_app_reuses_completed_summary_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "realizations").mkdir()
            (root / "index.jsonl").write_text(json.dumps({"id": "example"}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text("id: example\ntype: train\n", encoding="utf-8")
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("id: example\ntype: train\nparams: {steps: 1}\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "completed", "last_step": 1, "total_steps": 1}), encoding="utf-8")
            app = KuraMonitorApp(root)
            first = app.collect_summaries_cached()
            second = app.collect_summaries_cached()
            self.assertIs(first[0], second[0])

    def test_textual_monitor_smoke_handles_empty_active_and_tab_switch(self) -> None:
        async def run_case() -> None:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = root / "runs" / "example"
                (run_dir / "resolved").mkdir(parents=True)
                (run_dir / "logs").mkdir()
                (run_dir / "realizations").mkdir()
                (root / "index.jsonl").write_text(json.dumps({"id": "example"}) + "\n", encoding="utf-8")
                (run_dir / "run.yaml").write_text("id: example\ntype: train\n", encoding="utf-8")
                (run_dir / "resolved" / "manifest.lock.yaml").write_text("id: example\ntype: train\nparams: {steps: 1}\n", encoding="utf-8")
                (run_dir / "status.json").write_text(json.dumps({"state": "completed", "last_step": 1, "total_steps": 1}), encoding="utf-8")
                app = KuraMonitorApp(root, interval=999)
                async with app.run_test(size=(100, 30)) as pilot:
                    await pilot.pause(0.1)
                    await pilot.press("d")
                    await pilot.pause(0.1)
                    await pilot.press("r")
                    await pilot.pause(0.1)
                    await pilot.press("down")
                    await pilot.pause(0.1)

        asyncio.run(run_case())


class EnvLocalTests(unittest.TestCase):
    def test_env_local_loads_values_without_overriding_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_path = Path(directory) / ".env.local"
            env_path.write_text(
                "\n".join([
                    "# local secrets",
                    "KURA_NOTIFY=desktop,ntfy",
                    "export KURA_NTFY_TOPIC=kura-test-topic",
                    "KURA_NTFY_SERVER='https://ntfy.example.com'",
                    'KURA_NTFY_TOKEN="token-example"',
                    "EXISTING=value-from-file",
                ]),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"EXISTING": "value-from-env"}, clear=False):
                for key in ("KURA_NOTIFY", "KURA_NTFY_TOPIC", "KURA_NTFY_SERVER", "KURA_NTFY_TOKEN"):
                    os.environ.pop(key, None)
                _load_env_local(env_path)
                self.assertEqual(os.environ["KURA_NOTIFY"], "desktop,ntfy")
                self.assertEqual(os.environ["KURA_NTFY_TOPIC"], "kura-test-topic")
                self.assertEqual(os.environ["KURA_NTFY_SERVER"], "https://ntfy.example.com")
                self.assertEqual(os.environ["KURA_NTFY_TOKEN"], "token-example")
                self.assertEqual(os.environ["EXISTING"], "value-from-env")

    def test_env_local_loads_from_workspace_root_when_called_in_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "datasets" / "tiny"
            nested.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (root / ".env.local").write_text("KURA_NTFY_TOPIC=root-topic\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(nested)
            try:
                with patch.dict(os.environ, {}, clear=True):
                    _load_env_local()
                    self.assertEqual(os.environ["KURA_NTFY_TOPIC"], "root-topic")
                    self.assertEqual(_workspace(), root)
            finally:
                os.chdir(previous)


class WorkspaceDiscoveryTests(unittest.TestCase):
    def test_run_status_resolves_workspace_from_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            nested = root / "datasets" / "tiny"
            run_dir.mkdir(parents=True)
            nested.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(nested)
            try:
                self.assertEqual(cmd_run_status(argparse.Namespace(run_id="example")), 0)
            finally:
                os.chdir(previous)

    def test_doctor_workspace_reports_resolved_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "runs"
            nested.mkdir()
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(nested)
            try:
                self.assertEqual(cmd_doctor_workspace(argparse.Namespace()), 0)
            finally:
                os.chdir(previous)

    def test_doctor_workspace_warns_on_legacy_local_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "docker": {
                            "images": {
                                "ai-toolkit": {"local": "kura/ai-toolkit:dev"},
                                "musubi-tuner": {"local": "kura/musubi-tuner:dev"},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(cmd_doctor_workspace(argparse.Namespace()), 1)
                payload = json.loads(stdout.getvalue())
                self.assertTrue(payload["docker_images"]["ai-toolkit"]["legacy_default"])
                self.assertTrue(payload["docker_images"]["musubi-tuner"]["legacy_default"])
                self.assertIn("nomadoor/kura-ai-toolkit:dev", "\n".join(payload["warnings"]))
                self.assertIn("nomadoor/kura-musubi-tuner:dev", "\n".join(payload["warnings"]))
            finally:
                os.chdir(previous)

    def test_doctor_workspace_accepts_published_local_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump(
                    {
                        "schema_version": 1,
                        "docker": {
                            "images": {
                                "ai-toolkit": {"local": "nomadoor/kura-ai-toolkit:dev"},
                                "musubi-tuner": {"local": "nomadoor/kura-musubi-tuner:dev"},
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(cmd_doctor_workspace(argparse.Namespace()), 0)
                payload = json.loads(stdout.getvalue())
                self.assertEqual(payload["warnings"], [])
                self.assertFalse(payload["docker_images"]["ai-toolkit"]["legacy_default"])
                self.assertFalse(payload["docker_images"]["musubi-tuner"]["legacy_default"])
            finally:
                os.chdir(previous)


class RunPlanTests(unittest.TestCase):
    def test_disk_warnings_do_not_flag_a_single_checkpoint_as_frequent(self) -> None:
        run = {
            "params": {"steps": 1},
            "backend_overrides": {"musubi-tuner": {"save_every_n_steps": 1}},
            "compute": {"executor": "runpod"},
        }

        self.assertEqual(_disk_warnings(run, {"save_every_n_steps": 1}), [])

    def test_run_plan_prints_uncompiled_train_settings_from_run_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "plan-example"
            dataset_dir = root / "datasets" / "tiny"
            run_dir.mkdir(parents=True)
            dataset_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (dataset_dir / "items.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": "plan-example",
                        "type": "train",
                        "backend": {"name": "musubi-tuner"},
                        "model": {"base": "black-forest-labs/FLUX.2-klein-base-9B", "revision": "main"},
                        "compute": {"executor": "runpod", "gpu": "NVIDIA RTX A5000"},
                        "datasets": [{"id": "tiny", "role": "target", "digest": "sha256:abc"}],
                        "params": {
                            "rank": 16,
                            "alpha": 1024,
                            "lr": "0.00005",
                            "scheduler": "constant",
                            "steps": 1500,
                            "batch_size": 2,
                            "resolution": [768],
                            "seed": 42,
                        },
                        "sampling": {"cadence_steps": 100},
                        "backend_overrides": {
                            "musubi-tuner": {
                            "fp8_base": True,
                            "gradient_checkpointing": True,
                            "save_every_n_steps": 100,
                            "extra_args": ["--blocks_to_swap", "3"],
                        }
                    },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                    patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "missing_metadata", "size_bytes": None, "detail": "Content-Length header is absent"}),
                    patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "NVIDIA A40, 46068\n", "")),
                ):
                    self.assertEqual(cmd_run_plan(argparse.Namespace(run_id="plan-example", json=False)), 0)
            finally:
                os.chdir(previous)
            output = stdout.getvalue()
            self.assertIn("compiled     no", output)
            self.assertIn("musubi-tuner", output)
            self.assertIn("black-forest-labs/FLUX.2-klein-base-9B", output)
            self.assertIn("datasets/tiny", output)
            self.assertIn("items        2", output)
            self.assertIn("lr           0.00005", output)
            self.assertIn("extra_args   --blocks_to_swap, 3", output)
            self.assertIn("Model downloads", output)
            self.assertIn("unknown-size files", output)
            self.assertIn("Resources", output)
            self.assertIn("local_gpu    NVIDIA A40", output)
            self.assertIn("vram_mb      46068", output)
            self.assertIn("runpod_gpu_type_ids", output)
            self.assertIn("batch_size   2", output)
            self.assertIn("rank         16", output)
            self.assertIn("fp8_base     True", output)
            self.assertIn("blocks_to_swap 3", output)
            self.assertIn("Preflight", output)
            self.assertIn("[warning] disk", output)
            self.assertNotIn("Disk warnings", output)
            self.assertIn("checkpoint cadence may create about 15 checkpoints", output)

    def test_run_plan_prints_musubi_download_estimates_and_cache_hits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "download-plan"
            cached = root / "cache" / "models" / "musubi" / "example--model" / "vae" / "vae.safetensors"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"x" * 1024)
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": "download-plan",
                        "type": "train",
                        "backend": {"name": "musubi-tuner"},
                        "model": {"base": "custom"},
                        "backend_overrides": {
                            "musubi-tuner": {
                                "architecture": "flux_kontext",
                                "model_downloads": {
                                    "dit": {"repo": "example/model", "filename": "dit.safetensors"},
                                    "vae": {"repo": "example/model", "filename": "vae.safetensors"},
                                },
                            }
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                    patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "ok", "size_bytes": 3 * 1024**3}),
                    patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")),
                ):
                    self.assertEqual(cmd_run_plan(argparse.Namespace(run_id="download-plan", json=True)), 0)
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
        downloads = payload["model_downloads"]
        self.assertEqual(downloads["bytes"], 3 * 1024**3)
        self.assertEqual(downloads["cached_bytes"], 1024)
        self.assertEqual(len(downloads["items"]), 2)
        cached_items = [item for item in downloads["items"] if item["key"] == "vae"]
        self.assertEqual(cached_items[0]["download_bytes"], 0)
        self.assertTrue(cached_items[0]["cached"])
        resources = payload["resources"]
        self.assertEqual(resources["hardware"]["local_gpu"]["name"], "unknown")
        self.assertEqual(resources["model"]["architecture"], "flux_kontext")
        self.assertEqual(resources["memory_flags"]["common"]["batch_size"], "(not set)")
        artifact_filenames = {item["filename"] for item in resources["model"]["artifacts"]}
        self.assertEqual(artifact_filenames, {"dit.safetensors", "vae.safetensors"})
        requirements = resources["model"]["requirements"]
        self.assertEqual({item["acquisition"] for item in requirements}, {"kura"})
        self.assertEqual({item["measurement"]["scope"] for item in requirements}, {"controller"})
        checks = {(item["check"], item["severity"]) for item in payload["preflight"]}
        self.assertIn(("model-downloads", "info"), checks)
        self.assertIn(("dataset-images", "info"), checks)

    def test_run_plan_json_uses_compiled_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "compiled-example"
            (run_dir / "resolved").mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text("id: compiled-example\ntype: train\nbackend: {name: ai-toolkit}\nparams: {lr: 1e-4}\n", encoding="utf-8")
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("id: compiled-example\ntype: train\nbackend: {name: musubi-tuner}\nparams: {lr: 0.00005}\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    self.assertEqual(cmd_run_plan(argparse.Namespace(run_id="compiled-example", json=True)), 0)
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["compiled"])
            self.assertEqual(payload["source"], "runs/compiled-example/resolved/manifest.lock.yaml")
            self.assertEqual(payload["intent_source"], "runs/compiled-example/run.yaml")
            self.assertEqual(payload["resolved_manifest"], "runs/compiled-example/resolved/manifest.lock.yaml")
            self.assertEqual(payload["backend"]["name"], "musubi-tuner")
            self.assertEqual(payload["params"]["lr"], 0.00005)
            self.assertIn("preflight", payload)
            disk_records = [item for item in payload["preflight"] if item["check"] == "disk"]
            self.assertTrue(disk_records)
            self.assertIn("local Docker launch requires a disk preflight", disk_records[0]["fact"])
            self.assertNotIn("disk_warnings", payload)

    def test_run_plan_prints_preflight_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "preflight-example"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": "preflight-example",
                        "type": "train",
                        "backend": {"name": "ai-toolkit"},
                        "model": {"base": "example"},
                        "compute": {"executor": "docker"},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                    patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")),
                ):
                    self.assertEqual(cmd_run_plan(argparse.Namespace(run_id="preflight-example", json=False)), 0)
            finally:
                os.chdir(previous)
        output = stdout.getvalue()
        self.assertIn("Preflight", output)
        self.assertIn("[info] model-acquisition", output)
        self.assertIn("controller download size is not measured", output)
        self.assertNotIn("estimated model downloads write 0 B", output)
        self.assertIn("[warning] disk", output)
        self.assertNotIn("Disk warnings", output)

    def test_run_plan_preflight_bytes_preserve_small_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "small-download-plan"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": "small-download-plan",
                        "type": "train",
                        "backend": {"name": "musubi-tuner"},
                        "model": {"base": "custom"},
                        "safety": {"large_model_download_gb": 1},
                        "backend_overrides": {
                            "musubi-tuner": {
                                "architecture": "flux2",
                                "model_downloads": {"dit": {"repo": "example/model", "filename": "small.safetensors"}},
                            }
                        },
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                    patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "ok", "size_bytes": 50 * 1024**2}),
                    patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")),
                ):
                    self.assertEqual(cmd_run_plan(argparse.Namespace(run_id="small-download-plan", json=True)), 0)
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
        facts = [item["fact"] for item in payload["preflight"] if item["check"] == "model-downloads"]
        self.assertTrue(any("50.0 MiB" in fact for fact in facts))
        self.assertFalse(any("1 GiB" in fact for fact in facts))

    def test_run_plan_rejects_render_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "render-example"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text("id: render-example\ntype: render\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("sys.stderr", new_callable=__import__("io").StringIO) as stderr:
                    self.assertEqual(cmd_run_plan(argparse.Namespace(run_id="render-example", json=False)), 1)
            finally:
                os.chdir(previous)
            self.assertIn("for train runs", stderr.getvalue())

    def test_positive_integer_parsing_rejects_boolean_values(self) -> None:
        self.assertIsNone(_as_positive_int(True))
        self.assertIsNone(_as_positive_int(False))
        self.assertEqual(_as_positive_int("2"), 2)
        with self.assertRaisesRegex(ValueError, "integer GiB"):
            _configured_gib(True, default=50)
        with self.assertRaisesRegex(ValueError, "integer GiB"):
            _configured_gib(False, default=50)

    def test_musubi_download_estimate_handles_malformed_overrides(self) -> None:
        payload = _estimate_musubi_download_bytes(
            {
                "type": "train",
                "backend": {"name": "musubi-tuner"},
                "backend_overrides": True,
            }
        )
        self.assertEqual(payload["bytes"], 0)
        self.assertIn("invalid musubi model download spec", payload["unknown"])

    def test_runpod_plan_counts_remote_downloads_even_when_local_cache_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (root / "datasets" / "tiny").mkdir(parents=True)
            run_dir = root / "runs" / "remote"
            run_dir.mkdir(parents=True)
            cache_file = root / "cache" / "models" / "musubi" / "repo--model" / "dit" / "weights.safetensors"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_bytes(b"x" * 100)
            run = {
                "id": "remote",
                "type": "train",
                "backend": {"name": "musubi-tuner"},
                "model": {"base": "repo/model"},
                "datasets": [{"id": "tiny"}],
                "params": {"steps": 1},
                "compute": {"executor": "runpod"},
                "backend_overrides": {"musubi-tuner": {"architecture": "flux2", "model_bundle": "none", "model_downloads": {"dit": {"repo": "repo/model", "filename": "weights.safetensors"}}}},
            }
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "ok", "size_bytes": 200}):
                    payload = plan_run("remote")
            finally:
                os.chdir(previous)
        self.assertEqual(payload["model_downloads"]["bytes"], 200, payload["model_downloads"])
        self.assertEqual(payload["model_downloads"]["cached_bytes"], 0)
        self.assertFalse(payload["model_downloads"]["items"][0]["cached"])

    def test_local_plan_treats_unmapped_absolute_symlink_as_not_cached(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (root / "datasets" / "tiny").mkdir(parents=True)
            run_dir = root / "runs" / "local"
            run_dir.mkdir(parents=True)
            target = root / "cache" / "huggingface" / "hub" / "models--repo--model" / "snapshots" / "abc" / "weights.safetensors"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"x" * 123)
            cache_file = root / "cache" / "models" / "musubi" / "repo--model" / "dit" / "weights.safetensors"
            cache_file.parent.mkdir(parents=True)
            cache_file.symlink_to("/root/.cache/huggingface/hub/models--repo--model/snapshots/abc/weights.safetensors")
            run = {
                "id": "local",
                "type": "train",
                "backend": {"name": "musubi-tuner"},
                "model": {"base": "repo/model"},
                "datasets": [{"id": "tiny"}],
                "params": {"steps": 1},
                "compute": {"executor": "docker"},
                "backend_overrides": {"musubi-tuner": {"architecture": "flux2", "model_bundle": "none", "model_downloads": {"dit": {"repo": "repo/model", "filename": "weights.safetensors"}}}},
            }
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "ok", "size_bytes": 200}):
                    payload = plan_run("local")
            finally:
                os.chdir(previous)
        self.assertEqual(payload["model_downloads"]["bytes"], 200, payload["model_downloads"])
        self.assertEqual(payload["model_downloads"]["cached_bytes"], 0)
        self.assertFalse(payload["model_downloads"]["items"][0]["cached"])

    def test_runpod_disk_preflight_counts_downloads_and_checkpoints(self) -> None:
        run = {
            "type": "train",
            "backend": {"name": "musubi-tuner"},
            "params": {"steps": 20},
            "backend_overrides": {"musubi-tuner": {"save_every_n_steps": 10}},
            "safety": {"allow_many_checkpoints": True, "checkpoint_estimate_gb": 2},
        }
        download_estimate = {"bytes": 8 * 1024**3}
        with self.assertRaisesRegex(ValueError, "container_disk_gb=10"):
            _runpod_launch_disk_preflight(run, {"container_disk_gb": 10}, download_estimate)
        run["safety"]["allow_runpod_disk_risk"] = True
        result = _runpod_launch_disk_preflight(run, {"container_disk_gb": 10}, download_estimate)
        self.assertEqual(result["estimated_write_bytes"], 12 * 1024**3)

    def test_checkpoint_safety_preflight_rejects_many_unpruned_checkpoints(self) -> None:
        run = {
            "type": "train",
            "backend": {"name": "musubi-tuner"},
            "params": {"steps": 3000},
            "backend_overrides": {"musubi-tuner": {"save_every_n_steps": 100}},
        }
        with self.assertRaisesRegex(ValueError, "may create about 30 checkpoints"):
            _checkpoint_safety_preflight(run)
        run["backend_overrides"]["musubi-tuner"]["prune_checkpoints_before_step"] = 1000
        _checkpoint_safety_preflight(run)

    def test_checkpoint_safety_preflight_counts_backend_max_train_steps(self) -> None:
        run = {
            "type": "train",
            "backend": {"name": "musubi-tuner"},
            "params": {},
            "backend_overrides": {"musubi-tuner": {"max_train_steps": 3000, "save_every_n_steps": 100}},
        }
        with self.assertRaisesRegex(ValueError, "may create about 30 checkpoints"):
            _checkpoint_safety_preflight(run)

    def test_checkpoint_safety_preflight_accepts_musubi_keep_last_policy(self) -> None:
        run = {
            "type": "train",
            "backend": {"name": "musubi-tuner"},
            "params": {"steps": 3000},
            "backend_overrides": {
                "musubi-tuner": {
                    "save_every_n_steps": 100,
                    "extra_args": ["--save_last_n_steps", "300"],
                }
            },
        }
        _checkpoint_safety_preflight(run)
        run["backend_overrides"]["musubi-tuner"]["extra_args"] = ["--save_last_n_epochs=2"]
        _checkpoint_safety_preflight(run)

    def test_checkpoint_safety_preflight_can_be_explicitly_overridden(self) -> None:
        run = {
            "type": "train",
            "backend": {"name": "musubi-tuner"},
            "params": {"steps": 3000},
            "backend_overrides": {"musubi-tuner": {"save_every_n_steps": 100}},
            "safety": {"allow_many_checkpoints": True},
        }
        _checkpoint_safety_preflight(run)


class NotificationTests(unittest.TestCase):
    def test_notification_channels_auto_detect_ntfy_topic(self) -> None:
        with patch.dict(os.environ, {"KURA_NTFY_TOPIC": "kura-test-topic"}, clear=True), patch("kura.notifications.shutil.which", return_value=None):
            self.assertEqual(_notification_channels(None), ["ntfy"])

    def test_notification_channels_auto_detect_desktop(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch("kura.notifications.shutil.which", return_value="/usr/bin/notify-send"):
            self.assertEqual(_notification_channels(None), ["desktop"])

    def test_notification_channels_explicit_none_disables_auto_detection(self) -> None:
        with patch.dict(os.environ, {"KURA_NOTIFY": "none", "KURA_NTFY_TOPIC": "kura-test-topic"}, clear=True), patch("kura.notifications.shutil.which", return_value="/usr/bin/notify-send"):
            self.assertEqual(_notification_channels(None), [])

    def test_notification_channels_list_none_disables_auto_detection(self) -> None:
        with patch.dict(os.environ, {"KURA_NTFY_TOPIC": "kura-test-topic"}, clear=True), patch("kura.notifications.shutil.which", return_value="/usr/bin/notify-send"):
            self.assertEqual(_notification_channels(["desktop", "none", "ntfy"]), [])

    def test_ntfy_notification_posts_to_topic(self) -> None:
        class Response:
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return b""

        captured: dict[str, object] = {}

        def fake_urlopen(request: object, timeout: int) -> Response:
            captured["url"] = request.full_url  # type: ignore[attr-defined]
            captured["data"] = request.data  # type: ignore[attr-defined]
            captured["title"] = request.headers.get("Title")  # type: ignore[attr-defined]
            captured["priority"] = request.headers.get("Priority")  # type: ignore[attr-defined]
            captured["timeout"] = timeout
            return Response()

        with patch.dict(os.environ, {"KURA_NTFY_TOPIC": "kura-test-topic"}, clear=False), patch("kura.notifications.urllib.request.urlopen", fake_urlopen):
            _notify("ntfy", subject="finished", body="run done")

        self.assertEqual(captured["url"], "https://ntfy.sh/kura-test-topic")
        self.assertEqual(captured["data"], b"run done")
        self.assertEqual(captured["title"], "finished")
        self.assertEqual(captured["priority"], "4")
        self.assertEqual(captured["timeout"], 20)

    def test_ntfy_notification_rejects_non_http_server(self) -> None:
        with patch.dict(os.environ, {"KURA_NTFY_TOPIC": "kura-test-topic", "KURA_NTFY_SERVER": "file:///tmp/ntfy", "KURA_NTFY_TOKEN": "secret"}, clear=False), patch("kura.notifications.urllib.request.urlopen") as urlopen:
            _notify("ntfy", subject="finished", body="run done")
        urlopen.assert_not_called()

    def test_runpod_remote_notify_secrets_are_temp_env_only(self) -> None:
        env = {
            "KURA_NTFY_TOPIC": "kura-topic",
            "KURA_NTFY_SERVER": "https://ntfy.example.com",
            "KURA_NTFY_TOKEN": "ntfy-secret",
            "KURA_NTFY_PRIORITY": "4",
        }
        with patch.dict(os.environ, env, clear=False):
            payload = _runpod_secret_env_payload(remote_notify=True)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertIn("KURA_REMOTE_NOTIFY_NTFY=1", payload)
        self.assertIn("KURA_NTFY_TOPIC=kura-topic", payload)
        self.assertIn("KURA_NTFY_TOKEN=ntfy-secret", payload)


class RenderNotificationTests(unittest.TestCase):
    def test_render_launch_notifies_on_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            run_dir = root / "runs" / "render-1"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("type: render\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.launch.launch_render", return_value=0) as launch, patch("kura.run_commands.launch._notify") as notify:
                    code = cmd_run_launch(argparse.Namespace(run_id="render-1", executor="local", dry_run=False, notify="ntfy"))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            launch.assert_called_once()
            notify.assert_called_once()
            self.assertIn("completed", notify.call_args.kwargs["subject"])

    def test_render_dry_run_failure_does_not_notify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            run_dir = root / "runs" / "render-1"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("type: render\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.launch.launch_render", side_effect=ValueError("render broke")), patch("kura.run_commands.launch._notify") as notify:
                    code = launch_run("render-1", executor="docker", dry_run=True, notify_channels="ntfy")
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            notify.assert_not_called()

    def test_runpod_render_dry_run_failure_does_not_notify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "render-1"
            (run_dir / "resolved").mkdir(parents=True)
            (root / "workspace.yaml").write_text(
                "docker:\n"
                "  images:\n"
                "    comfyui:\n"
                "      local: local/comfy\n"
                "      remote: remote/comfy\n"
                "      dockerfile: docker/comfyui/Dockerfile\n"
                "      context: .\n"
                "runpod:\n"
                "  storage_mode: upload\n",
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                yaml.safe_dump({
                    "type": "render",
                    "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/wf.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "runpod"},
                    "workflow_patches": {},
                    "render": {"default_seed": 1},
                }),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.render_runpod._notify") as notify:
                    code = launch_run("render-1", executor="runpod", dry_run=True, notify_channels="ntfy")
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            notify.assert_not_called()

    def test_render_stages_local_lora_for_comfyui_and_cleans_it_up(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_dir = root / "comfyui" / "models" / "loras"
            workflow_dir = root / "workflows"
            promptset_dir = root / "promptsets"
            run_dir = root / "runs" / "render-1"
            output_run = root / "runs" / "train-1" / "outputs"
            for path in (workflow_dir, promptset_dir, run_dir / "resolved", output_run):
                path.mkdir(parents=True)
            (output_run.parent / "run.yaml").write_text("id: train-1\ntype: train\n", encoding="utf-8")
            (root / "workspace.yaml").write_text(
                f"comfyui:\n  lora_dir: {lora_dir}\n  lora_stage_subdir: Kura_tmp\n  lora_stage_cleanup: remove_after_render\n  local_note: should-not-freeze\n  custom: {{nested: private}}\n",
                encoding="utf-8",
            )
            checkpoint = output_run / "example.safetensors"
            checkpoint.write_bytes(b"fake-lora")
            (workflow_dir / "wf.json").write_text(
                json.dumps({
                    "3": {"inputs": {"seed": 0}},
                    "6": {"inputs": {"text": ""}},
                    "7": {"inputs": {"text": ""}},
                    "12": {"inputs": {"lora_name": "old.safetensors"}},
                }),
                encoding="utf-8",
            )
            (promptset_dir / "prompts.jsonl").write_text(json.dumps({"id": "p1", "prompt": "hello", "seeds": [123]}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "schema_version": 1,
                    "type": "render",
                    "inputs": {
                        "train_run": "train-1",
                        "checkpoint": {"path": "runs/train-1/outputs/example.safetensors", "hash": None},
                        "workflow": {"path": "workflows/wf.json", "digest": None},
                        "promptset": {"path": "promptsets/prompts.jsonl", "digest": None},
                    },
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "local"},
                    "workflow_patches": {"prompt": {"node": "6", "field": "inputs.text"}, "negative_prompt": {"node": "7", "field": "inputs.text"}, "seed": {"node": "3", "field": "inputs.seed"}, "lora": {"node": "12", "field": "inputs.lora_name"}},
                    "render": {"output_dir": "samples/images", "timeout_sec": 5, "default_seed": None},
                }),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            compile_render(root, run_dir)
            manifest = yaml.safe_load((run_dir / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["comfyui"]), {"lora_dir", "lora_stage_subdir", "lora_stage_cleanup"})
            captured: dict[str, Any] = {}

            class FakeClient:
                def __init__(self, endpoint: str, timeout: int) -> None:
                    captured["endpoint"] = endpoint
                    captured["timeout"] = timeout

                def queue(self, workflow: dict[str, Any]) -> str:
                    captured["workflow"] = workflow
                    staged_name = workflow["12"]["inputs"]["lora_name"]
                    staged_path = lora_dir / staged_name
                    captured["staged_path"] = staged_path
                    captured["staged_exists_during_queue"] = staged_path.is_symlink() or staged_path.is_file()
                    return "prompt-1"

                def wait(self, prompt_id: str) -> list[dict[str, Any]]:
                    return [{"filename": "image.png", "subfolder": "", "type": "output"}]

                def download(self, image: dict[str, Any]) -> bytes:
                    return b"png"

            with patch("kura.render.ComfyUIClient", FakeClient):
                code = launch_render(root, run_dir)
            self.assertEqual(code, 0)
            self.assertTrue(captured["staged_exists_during_queue"])
            lora_name = captured["workflow"]["12"]["inputs"]["lora_name"]
            self.assertTrue(lora_name.startswith("Kura_tmp/render-1-example-"))
            self.assertFalse(captured["staged_path"].exists())
            image_record = json.loads((run_dir / "samples" / "images.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(image_record["train_run"], "train-1")
            self.assertIn("comfyui_lora_name", image_record)
            realization_path = root / "runs" / "render-1" / json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["last_realization"]
            realization = json.loads(realization_path.read_text(encoding="utf-8"))
            self.assertEqual(realization["train_run"], "train-1")

    def test_render_compile_validates_train_run_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "render-1"
            workflow = root / "workflows" / "wf.json"
            prompts = root / "promptsets" / "prompts.jsonl"
            for path in (run_dir, workflow.parent, prompts.parent):
                path.mkdir(parents=True, exist_ok=True)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            workflow.write_text(json.dumps({"1": {"inputs": {"seed": 0}}}), encoding="utf-8")
            prompts.write_text(json.dumps({"id": "p1", "prompt": "hello", "seeds": [1]}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "type": "render",
                    "inputs": {
                        "train_run": "missing-train",
                        "checkpoint": {"path": ""},
                        "workflow": {"path": "workflows/wf.json"},
                        "promptset": {"path": "promptsets/prompts.jsonl"},
                    },
                    "workflow_patches": {"seed": {"node": "1", "field": "inputs.seed"}},
                }),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "does not exist: missing-train"):
                compile_render(root, run_dir)

    def test_render_inserts_sidecar_lora_loader_when_checkpoint_is_available(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_dir = root / "comfyui" / "models" / "loras"
            workflow_dir = root / "workflows"
            promptset_dir = root / "promptsets"
            run_dir = root / "runs" / "render-1"
            output_run = root / "runs" / "train-1" / "outputs"
            for path in (workflow_dir, promptset_dir, run_dir / "resolved", output_run):
                path.mkdir(parents=True)
            (root / "workspace.yaml").write_text(f"comfyui:\n  lora_dir: {lora_dir}\n  lora_stage_subdir: Kura_tmp\n", encoding="utf-8")
            checkpoint = output_run / "example.safetensors"
            checkpoint.write_bytes(b"fake-lora")
            (workflow_dir / "wf.json").write_text(
                json.dumps({
                    "3": {"class_type": "KSampler", "inputs": {"seed": 0, "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0]}},
                    "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "base.safetensors"}},
                    "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
                    "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["4", 1]}},
                }),
                encoding="utf-8",
            )
            (workflow_dir / "wf.kura.yaml").write_text(
                "lora_insert:\n"
                "  kind: model_clip\n"
                "  model_node: '4'\n"
                "  clip_node: '4'\n",
                encoding="utf-8",
            )
            (promptset_dir / "prompts.jsonl").write_text(json.dumps({"id": "p1", "prompt": "hello", "seeds": [123]}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "schema_version": 1,
                    "type": "render",
                    "inputs": {
                        "checkpoint": {"path": "runs/train-1/outputs/example.safetensors", "hash": None},
                        "workflow": {"path": "workflows/wf.json", "digest": None},
                        "promptset": {"path": "promptsets/prompts.jsonl", "digest": None},
                    },
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "local"},
                    "workflow_patches": {"prompt": {"node": "6", "field": "inputs.text"}, "negative_prompt": {"node": "7", "field": "inputs.text"}, "seed": {"node": "3", "field": "inputs.seed"}},
                    "render": {"output_dir": "samples/images", "timeout_sec": 5, "default_seed": None},
                }),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            compile_render(root, run_dir)
            manifest = yaml.safe_load((run_dir / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8"))
            self.assertEqual(manifest["lora_insert"]["class_type"], "LoraLoader")
            captured: dict[str, Any] = {}

            class FakeClient:
                def __init__(self, endpoint: str, timeout: int) -> None:
                    pass

                def queue(self, workflow: dict[str, Any]) -> str:
                    captured["workflow"] = workflow
                    return "prompt-1"

                def wait(self, prompt_id: str) -> list[dict[str, Any]]:
                    return [{"filename": "image.png", "subfolder": "", "type": "output"}]

                def download(self, image: dict[str, Any]) -> bytes:
                    return b"png"

            with patch("kura.render.ComfyUIClient", FakeClient):
                self.assertEqual(launch_render(root, run_dir), 0)
            queued = captured["workflow"]
            lora_node = queued["8"]
            self.assertEqual(lora_node["class_type"], "LoraLoader")
            self.assertTrue(lora_node["inputs"]["lora_name"].startswith("Kura_tmp/render-1-example-"))
            self.assertEqual(lora_node["inputs"]["model"], ["4", 0])
            self.assertEqual(lora_node["inputs"]["clip"], ["4", 1])
            self.assertEqual(queued["3"]["inputs"]["model"], ["8", 0])
            self.assertEqual(queued["6"]["inputs"]["clip"], ["8", 1])
            self.assertEqual(queued["7"]["inputs"]["clip"], ["8", 1])

    def test_insert_lora_loader_skips_empty_lora_name(self) -> None:
        workflow = {"1": {"inputs": {"model": ["2", 0]}}, "2": {"inputs": {}}}
        self.assertEqual(insert_lora_loader(workflow, {"class_type": "LoraLoaderModelOnly", "model_node": "2", "model_output": 0}, ""), workflow)

    def test_lora_insert_kind_error_lists_accepted_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow_dir = root / "workflows"
            promptset_dir = root / "promptsets"
            run_dir = root / "runs" / "render-1"
            for path in (workflow_dir, promptset_dir, run_dir):
                path.mkdir(parents=True)
            (root / "workspace.yaml").write_text("comfyui:\n  model_registry: {}\n", encoding="utf-8")
            (workflow_dir / "wf.json").write_text(json.dumps({"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "base.safetensors"}}}), encoding="utf-8")
            (workflow_dir / "wf.kura.yaml").write_text("lora_insert:\n  kind: typo\n  model_node: '1'\n", encoding="utf-8")
            (promptset_dir / "prompts.jsonl").write_text(json.dumps({"id": "p1", "prompt": "hello", "seeds": [1]}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "type": "render",
                    "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/wf.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "local"},
                    "workflow_patches": {},
                    "render": {"default_seed": None},
                }),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model_only, model_clip, full, LoraLoaderModelOnly, LoraLoader"):
                compile_render(root, run_dir)

    def test_runpod_lora_upload_plan_uses_sidecar_insert(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "render-1"
            output_dir = root / "runs" / "train-1" / "outputs"
            output_dir.mkdir(parents=True)
            source = output_dir / "example.safetensors"
            source.write_bytes(b"fake-lora")
            lora_source, lora_name = _render_runpod_lora(
                root,
                run_dir,
                {
                    "inputs": {"checkpoint": {"path": "runs/train-1/outputs/example.safetensors"}},
                    "workflow_patches": {},
                    "lora_insert": {"class_type": "LoraLoader", "model_node": "4", "model_output": 0},
                },
            )
            self.assertEqual(lora_source, source.resolve())
            self.assertIsNotNone(lora_name)
            self.assertTrue(lora_name.startswith("Kura_tmp/render-1-example-"))

    def test_comfyui_prepare_model_ready_logs_json_paths(self) -> None:
        spec = importlib.util.spec_from_file_location("kura_comfy_prepare", Path("docker/comfyui/kura_comfy_prepare.py"))
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloaded = root / "cache" / "toy.safetensors"
            downloaded.parent.mkdir()
            downloaded.write_bytes(b"toy")
            module._download_model = lambda spec, cache_dir: downloaded
            workflow = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "toy.safetensors"}}}
            registry = {"checkpoints": {"toy.safetensors": {"repo": "owner/toy", "filename": "toy.safetensors"}}}
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer), patch.dict(os.environ, {"KURA_WORKSPACE": str(root)}, clear=False):
                module.prepare(workflow, comfyui_root=root / "ComfyUI", cache_dir=root / "cache", registry=registry)
            events = [json.loads(line) for line in buffer.getvalue().splitlines()]
            self.assertEqual(events[0]["event"], "model_ready")
            self.assertIsInstance(events[0]["source"], str)
            self.assertTrue((root / "ComfyUI" / "models" / "checkpoints" / "toy.safetensors").is_symlink())

    def test_comfyui_prepare_preserves_existing_real_model_file(self) -> None:
        spec = importlib.util.spec_from_file_location("kura_comfy_prepare", Path("docker/comfyui/kura_comfy_prepare.py"))
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            downloaded = root / "cache" / "toy.safetensors"
            downloaded.parent.mkdir()
            downloaded.write_bytes(b"downloaded")
            target = root / "ComfyUI" / "models" / "checkpoints" / "toy.safetensors"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"existing")
            module._download_model = lambda spec, cache_dir: downloaded
            workflow = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "toy.safetensors"}}}
            registry = {"checkpoints": {"toy.safetensors": {"repo": "owner/toy", "filename": "toy.safetensors"}}}
            with patch.dict(os.environ, {"KURA_WORKSPACE": str(root)}, clear=False), self.assertRaisesRegex(ValueError, "refusing to replace existing ComfyUI model target"):
                module.prepare(workflow, comfyui_root=root / "ComfyUI", cache_dir=root / "cache", registry=registry)
            self.assertFalse(target.is_symlink())
            self.assertEqual(target.read_bytes(), b"existing")

    def test_comfyui_prepare_requires_cache_dir_before_download(self) -> None:
        spec = importlib.util.spec_from_file_location("kura_comfy_prepare", Path("docker/comfyui/kura_comfy_prepare.py"))
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        workflow = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "toy.safetensors"}}}
        registry = {"checkpoints": {"toy.safetensors": {"repo": "owner/toy", "filename": "toy.safetensors"}}}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires HF_HUB_CACHE or --cache-dir"):
                module.prepare(workflow, comfyui_root=Path(directory) / "ComfyUI", cache_dir=None, registry=registry)

    def test_comfyui_prepare_rejects_private_cache_dir_before_download(self) -> None:
        spec = importlib.util.spec_from_file_location("kura_comfy_prepare", Path("docker/comfyui/kura_comfy_prepare.py"))
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "toy.safetensors"}}}
            registry = {"checkpoints": {"toy.safetensors": {"repo": "owner/toy", "filename": "toy.safetensors"}}}
            module._download_model = Mock(side_effect=AssertionError("download should not start"))
            with patch.dict(os.environ, {"KURA_WORKSPACE": str(root / "workspace")}, clear=False):
                with self.assertRaisesRegex(ValueError, "cache_dir must be under"):
                    module.prepare(workflow, comfyui_root=root / "ComfyUI", cache_dir=root / "private-cache", registry=registry)
            module._download_model.assert_not_called()

    def test_comfyui_prepare_direct_download_rejects_unsafe_urls(self) -> None:
        spec = importlib.util.spec_from_file_location("kura_comfy_prepare", Path("docker/comfyui/kura_comfy_prepare.py"))
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(module.urllib.request, "urlopen") as urlopen:
                with self.assertRaisesRegex(ValueError, "https:// URL"):
                    module._download_model({"url": "http://example.com/model.safetensors", "filename": "model.safetensors"}, root)
                urlopen.assert_not_called()
            with (
                patch.object(module.socket, "getaddrinfo", return_value=[(module.socket.AF_INET, module.socket.SOCK_STREAM, 0, "", ("127.0.0.1", 443))]),
                patch.object(module.urllib.request, "urlopen") as urlopen,
            ):
                with self.assertRaisesRegex(ValueError, "non-public address"):
                    module._download_model({"url": "https://localhost/model.safetensors", "filename": "model.safetensors"}, root)
                urlopen.assert_not_called()

    def test_comfyui_prepare_direct_download_allows_public_https(self) -> None:
        spec = importlib.util.spec_from_file_location("kura_comfy_prepare", Path("docker/comfyui/kura_comfy_prepare.py"))
        self.assertIsNotNone(spec)
        self.assertIsNotNone(spec.loader)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        class Response(io.BytesIO):
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
                return None

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch.object(module.socket, "getaddrinfo", return_value=[(module.socket.AF_INET, module.socket.SOCK_STREAM, 0, "", ("93.184.216.34", 443))]),
                patch.object(module.urllib.request, "urlopen", return_value=Response(b"model-bytes")) as urlopen,
            ):
                target = module._download_model({"url": "https://example.com/model.safetensors", "filename": "model.safetensors"}, root)
            self.assertEqual(target.read_bytes(), b"model-bytes")
            urlopen.assert_called_once_with("https://example.com/model.safetensors", timeout=60)

    def test_render_failure_appends_to_existing_stdout_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow_dir = root / "workflows"
            promptset_dir = root / "promptsets"
            run_dir = root / "runs" / "render-1"
            output_run = root / "runs" / "train-1" / "outputs"
            for path in (workflow_dir, promptset_dir, run_dir / "resolved", output_run):
                path.mkdir(parents=True)
            (root / "workspace.yaml").write_text("comfyui:\n  lora_dir: ''\n", encoding="utf-8")
            checkpoint = output_run / "example.safetensors"
            checkpoint.write_bytes(b"fake-lora")
            (workflow_dir / "wf.json").write_text(
                json.dumps({
                    "3": {"inputs": {"seed": 0}},
                    "6": {"inputs": {"text": ""}},
                    "7": {"inputs": {"text": ""}},
                    "12": {"inputs": {"lora_name": "old.safetensors"}},
                }),
                encoding="utf-8",
            )
            (promptset_dir / "prompts.jsonl").write_text(json.dumps({"id": "p1", "prompt": "hello", "seeds": [123]}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "schema_version": 1,
                    "type": "render",
                    "inputs": {
                        "checkpoint": {"path": "runs/train-1/outputs/example.safetensors", "hash": None},
                        "workflow": {"path": "workflows/wf.json", "digest": None},
                        "promptset": {"path": "promptsets/prompts.jsonl", "digest": None},
                    },
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "local"},
                    "workflow_patches": {"prompt": {"node": "6", "field": "inputs.text"}, "negative_prompt": {"node": "7", "field": "inputs.text"}, "seed": {"node": "3", "field": "inputs.seed"}, "lora": {"node": "12", "field": "inputs.lora_name"}},
                    "render": {"output_dir": "samples/images", "timeout_sec": 5, "default_seed": None},
                }),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            compile_render(root, run_dir)

            class FailingClient:
                def __init__(self, endpoint: str, timeout: int) -> None:
                    pass

                def queue(self, workflow: dict[str, Any]) -> str:
                    return "prompt-1"

                def wait(self, prompt_id: str) -> list[dict[str, Any]]:
                    raise RuntimeError("render broke")

            with patch("kura.render.ComfyUIClient", FailingClient):
                code = launch_render(root, run_dir)
            self.assertEqual(code, 1)
            stdout = (run_dir / "logs" / "stdout.log").read_text(encoding="utf-8")
            self.assertIn("render endpoint: http://127.0.0.1:8188", stdout)
            self.assertIn("queued p1 seed=123 prompt_id=prompt-1", stdout)
            self.assertIn("RuntimeError: render broke", stdout)

    def test_render_fails_when_comfyui_returns_no_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow_dir = root / "workflows"
            promptset_dir = root / "promptsets"
            run_dir = root / "runs" / "render-1"
            for path in (workflow_dir, promptset_dir, run_dir / "resolved"):
                path.mkdir(parents=True)
            (workflow_dir / "wf.json").write_text(
                json.dumps({
                    "3": {"inputs": {"seed": 0}},
                    "6": {"inputs": {"text": ""}},
                    "7": {"inputs": {"text": ""}},
                }),
                encoding="utf-8",
            )
            (promptset_dir / "prompts.jsonl").write_text(json.dumps({"id": "p1", "prompt": "hello", "seeds": [123]}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "schema_version": 1,
                    "type": "render",
                    "inputs": {
                        "checkpoint": {"path": "", "hash": None},
                        "workflow": {"path": "workflows/wf.json", "digest": None},
                        "promptset": {"path": "promptsets/prompts.jsonl", "digest": None},
                    },
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "local"},
                    "workflow_patches": {"prompt": {"node": "6", "field": "inputs.text"}, "negative_prompt": {"node": "7", "field": "inputs.text"}, "seed": {"node": "3", "field": "inputs.seed"}},
                    "render": {"output_dir": "samples/images", "timeout_sec": 5, "default_seed": None},
                }),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            compile_render(root, run_dir)

            class EmptyClient:
                def __init__(self, endpoint: str, timeout: int) -> None:
                    pass

                def queue(self, workflow: dict[str, Any]) -> str:
                    return "prompt-1"

                def wait(self, prompt_id: str) -> list[dict[str, Any]]:
                    return []

            with patch("kura.render.ComfyUIClient", EmptyClient):
                code = launch_render(root, run_dir)

            self.assertEqual(code, 1)
            state = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(state["state"], "failed")
            stdout = (run_dir / "logs" / "stdout.log").read_text(encoding="utf-8")
            self.assertIn("RuntimeError: ComfyUI completed without returning any images", stdout)

    def test_runpod_render_compile_requires_model_registry_entries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workflows").mkdir()
            (root / "promptsets").mkdir()
            run_dir = root / "runs" / "render-1"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text("comfyui:\n  model_registry: {}\n", encoding="utf-8")
            (root / "workflows" / "wf.json").write_text(json.dumps({"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "missing.safetensors"}}}), encoding="utf-8")
            (root / "promptsets" / "prompts.jsonl").write_text(json.dumps({"id": "p1", "prompt": "hello", "seeds": [1]}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "type": "render",
                    "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/wf.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "runpod"},
                    "workflow_patches": {},
                    "render": {"default_seed": None},
                }),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown model loader"):
                compile_render(root, run_dir)

    def test_runpod_render_compile_freezes_workspace_registry_model_specs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workflows").mkdir()
            (root / "promptsets").mkdir()
            run_dir = root / "runs" / "render-1"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text(
                "comfyui:\n"
                "  model_registry:\n"
                "    checkpoints:\n"
                "      toy.safetensors:\n"
                "        repo: owner/toy\n"
                "        filename: weights/toy.safetensors\n"
                "        revision: abc123\n",
                encoding="utf-8",
            )
            (root / "workflows" / "wf.json").write_text(json.dumps({"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "toy.safetensors"}}}), encoding="utf-8")
            (root / "promptsets" / "prompts.jsonl").write_text(json.dumps({"id": "p1", "prompt": "hello", "seeds": [1]}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "type": "render",
                    "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/wf.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "runpod"},
                    "workflow_patches": {},
                    "render": {"default_seed": None},
                }),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            compile_render(root, run_dir)
            specs = json.loads((run_dir / "resolved" / "comfyui_models.json").read_text(encoding="utf-8"))
            registry = json.loads((run_dir / "resolved" / "comfyui_model_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(specs[0]["repo"], "owner/toy")
            self.assertEqual(specs[0]["filename"], "weights/toy.safetensors")
            self.assertEqual(specs[0]["target_dir"], "checkpoints")
            self.assertEqual(registry["checkpoints"]["toy.safetensors"]["revision"], "abc123")

    def test_runpod_render_compile_merges_sample_sidecar_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_dir = root / "workflows" / "samples" / "toy"
            (root / "promptsets").mkdir(parents=True)
            sample_dir.mkdir(parents=True)
            run_dir = root / "runs" / "render-1"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text("comfyui:\n  model_registry: {}\n", encoding="utf-8")
            (sample_dir / "toy-text2image-api.json").write_text(json.dumps({"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "toy.safetensors"}}}), encoding="utf-8")
            (sample_dir / "toy-text2image-api.kura.yaml").write_text(
                "models:\n"
                "  checkpoints:\n"
                "    toy.safetensors:\n"
                "      repo: curated/toy\n"
                "      filename: curated/toy.safetensors\n",
                encoding="utf-8",
            )
            (root / "promptsets" / "prompts.jsonl").write_text(json.dumps({"id": "p1", "prompt": "hello", "seeds": [1]}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "type": "render",
                    "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/samples/toy/toy-text2image-api.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "runpod"},
                    "workflow_patches": {},
                    "render": {"default_seed": None},
                }),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            compile_render(root, run_dir)

            specs = json.loads((run_dir / "resolved" / "comfyui_models.json").read_text(encoding="utf-8"))
            registry = json.loads((run_dir / "resolved" / "comfyui_model_registry.json").read_text(encoding="utf-8"))
            self.assertEqual(specs[0]["repo"], "curated/toy")
            self.assertEqual(specs[0]["filename"], "curated/toy.safetensors")
            self.assertEqual(registry["checkpoints"]["toy.safetensors"]["repo"], "curated/toy")

    def test_runpod_render_compile_accepts_sidecar_url_and_target_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_dir = root / "workflows" / "samples" / "toy"
            (root / "promptsets").mkdir(parents=True)
            sample_dir.mkdir(parents=True)
            run_dir = root / "runs" / "render-1"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text("comfyui:\n  model_registry: {}\n", encoding="utf-8")
            (sample_dir / "toy_text2image_api.json").write_text(
                json.dumps({
                    "1": {"class_type": "CLIPLoader", "inputs": {"clip_name": "encoder.safetensors"}},
                    "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "direct.safetensors"}},
                }),
                encoding="utf-8",
            )
            (sample_dir / "toy_text2image_api.kura.yaml").write_text(
                "models:\n"
                "  clip:\n"
                "    encoder.safetensors:\n"
                "      repo: owner/toy\n"
                "      filename: text_encoders/encoder.safetensors\n"
                "      target_dir: text_encoders\n"
                "  checkpoints:\n"
                "    direct.safetensors:\n"
                "      url: https://civitai.example/api/download/models/1?fileId=2\n"
                "      filename: direct.safetensors\n",
                encoding="utf-8",
            )
            (root / "promptsets" / "prompts.jsonl").write_text(json.dumps({"id": "p1", "prompt": "hello", "seeds": [1]}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "type": "render",
                    "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/samples/toy/toy_text2image_api.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "runpod"},
                    "workflow_patches": {},
                    "render": {"default_seed": None},
                }),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            compile_render(root, run_dir)

            specs = json.loads((run_dir / "resolved" / "comfyui_models.json").read_text(encoding="utf-8"))
            registry = json.loads((run_dir / "resolved" / "comfyui_model_registry.json").read_text(encoding="utf-8"))
            specs_by_name = {item["name"]: item for item in specs}
            self.assertEqual(specs_by_name["encoder.safetensors"]["repo"], "owner/toy")
            self.assertEqual(specs_by_name["encoder.safetensors"]["filename"], "text_encoders/encoder.safetensors")
            self.assertEqual(specs_by_name["encoder.safetensors"]["target_dir"], "text_encoders")
            self.assertEqual(specs_by_name["direct.safetensors"]["url"], "https://civitai.example/api/download/models/1?fileId=2")
            self.assertEqual(specs_by_name["direct.safetensors"]["target_dir"], "checkpoints")
            self.assertEqual(registry["clip"]["encoder.safetensors"]["repo"], "owner/toy")
            self.assertEqual(registry["checkpoints"]["direct.safetensors"]["url"], "https://civitai.example/api/download/models/1?fileId=2")

    def test_workspace_registry_overrides_sample_sidecar_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_dir = root / "workflows" / "samples" / "toy"
            (root / "promptsets").mkdir(parents=True)
            sample_dir.mkdir(parents=True)
            run_dir = root / "runs" / "render-1"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text(
                "comfyui:\n"
                "  model_registry:\n"
                "    checkpoints:\n"
                "      toy.safetensors:\n"
                "        repo: local/toy\n"
                "        filename: local/toy.safetensors\n",
                encoding="utf-8",
            )
            (sample_dir / "toy-text2image-api.json").write_text(json.dumps({"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "toy.safetensors"}}}), encoding="utf-8")
            (sample_dir / "toy-text2image-api.kura.yaml").write_text(
                "models:\n"
                "  checkpoints:\n"
                "    toy.safetensors:\n"
                "      repo: curated/toy\n"
                "      filename: curated/toy.safetensors\n",
                encoding="utf-8",
            )
            (root / "promptsets" / "prompts.jsonl").write_text(json.dumps({"id": "p1", "prompt": "hello", "seeds": [1]}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "type": "render",
                    "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/samples/toy/toy-text2image-api.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "runpod"},
                    "workflow_patches": {},
                    "render": {"default_seed": None},
                }),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            compile_render(root, run_dir)

            specs = json.loads((run_dir / "resolved" / "comfyui_models.json").read_text(encoding="utf-8"))
            self.assertEqual(specs[0]["repo"], "local/toy")
            self.assertEqual(specs[0]["filename"], "local/toy.safetensors")

    def test_runpod_render_launch_dry_run_prints_plan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = Path.cwd()
            try:
                os.chdir(root)
                (root / "runs" / "render-1" / "resolved").mkdir(parents=True)
                run_dir = root / "runs" / "render-1"
                (root / "workspace.yaml").write_text(
                    "docker:\n"
                    "  images:\n"
                    "    comfyui:\n"
                    "      local: local/comfy\n"
                    "      remote: remote/comfy\n"
                    "      dockerfile: docker/comfyui/Dockerfile\n"
                    "      context: .\n"
                    "runpod:\n"
                    "  storage_mode: upload\n"
                    "  gpu_type_ids: [NVIDIA RTX A5000]\n"
                    "  default_image:\n"
                    "    comfyui: remote/default-comfy\n"
                    "comfyui:\n"
                    "  runpod:\n"
                    "    ports: [22/tcp]\n",
                    encoding="utf-8",
                )
                (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
                (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                    yaml.safe_dump({
                        "type": "render",
                        "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/wf.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                        "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                        "executor": {"name": "runpod"},
                        "workflow_patches": {},
                        "render": {"default_seed": 1},
                        "comfyui_model_registry": {"checkpoints": {"toy.safetensors": {"repo": "owner/toy", "filename": "toy.safetensors"}}},
                        "comfyui_models": [{"name": "toy.safetensors", "repo": "owner/toy", "filename": "toy.safetensors", "target_dir": "checkpoints"}],
                    }),
                    encoding="utf-8",
                )
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    code = launch_run("render-1", executor="runpod", dry_run=True)
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            plan = json.loads(buffer.getvalue())
            self.assertEqual(plan["image"], "remote/default-comfy")
            self.assertEqual(plan["executor"], "runpod")
            self.assertEqual(plan["models"][0]["repo"], "owner/toy")

    def test_runpod_render_launch_requires_runpod_compiled_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = Path.cwd()
            try:
                os.chdir(root)
                (root / "runs" / "render-1" / "resolved").mkdir(parents=True)
                run_dir = root / "runs" / "render-1"
                (root / "workspace.yaml").write_text(
                    "docker:\n"
                    "  images:\n"
                    "    comfyui:\n"
                    "      local: local/comfy\n"
                    "      remote: remote/comfy\n"
                    "      dockerfile: docker/comfyui/Dockerfile\n"
                    "      context: .\n"
                    "runpod:\n"
                    "  storage_mode: upload\n"
                    "comfyui:\n"
                    "  runpod:\n"
                    "    ports: [22/tcp]\n",
                    encoding="utf-8",
                )
                (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
                (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                    yaml.safe_dump({
                        "type": "render",
                        "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/wf.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                        "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                        "executor": {"name": "local"},
                        "workflow_patches": {},
                        "render": {"default_seed": 1},
                    }),
                    encoding="utf-8",
                )
                buffer = io.StringIO()
                with contextlib.redirect_stderr(buffer):
                    code = launch_run("render-1", executor="runpod", dry_run=False)
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            self.assertIn("compiled for executor.name=runpod", buffer.getvalue())

    def test_render_cleanup_keeps_preexisting_stage_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            target = root / "Kura_tmp" / "source.safetensors"
            target.parent.mkdir()
            source.write_bytes(b"same-lora")
            target.write_bytes(b"same-lora")
            plan = {
                "source": str(source),
                "target": str(target),
                "lora_name": "Kura_tmp/source.safetensors",
                "mode": "copy",
                "cleanup": "remove_after_render",
                "created": False,
            }

            _materialize_lora_stage(plan)
            _cleanup_lora_stage(plan)

            self.assertFalse(plan["created"])
            self.assertTrue(target.exists())

    def test_render_stage_rejects_same_size_different_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.safetensors"
            target = root / "Kura_tmp" / "source.safetensors"
            target.parent.mkdir()
            source.write_bytes(b"abcd")
            target.write_bytes(b"wxyz")
            plan = {
                "source": str(source),
                "target": str(target),
                "lora_name": "Kura_tmp/source.safetensors",
                "mode": "copy",
                "cleanup": "remove_after_render",
                "created": False,
            }

            with self.assertRaisesRegex(ValueError, "different content"):
                _materialize_lora_stage(plan)

    def test_render_fails_when_configured_lora_dir_is_not_visible_to_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_dir = root / "wrong" / "loras"
            workflow_dir = root / "workflows"
            promptset_dir = root / "promptsets"
            run_dir = root / "runs" / "render-1"
            output_run = root / "runs" / "train-1" / "outputs"
            for path in (lora_dir, workflow_dir, promptset_dir, run_dir / "resolved", output_run):
                path.mkdir(parents=True)
            (root / "workspace.yaml").write_text(
                f"comfyui:\n  lora_dir: {lora_dir}\n  lora_stage_subdir: Kura_tmp\n  lora_stage_mode: copy\n  lora_stage_cleanup: remove_after_render\n",
                encoding="utf-8",
            )
            checkpoint = output_run / "example.safetensors"
            checkpoint.write_bytes(b"fake-lora")
            (workflow_dir / "wf.json").write_text(
                json.dumps({
                    "3": {"inputs": {"seed": 0}},
                    "6": {"inputs": {"text": ""}},
                    "7": {"inputs": {"text": ""}},
                    "12": {"inputs": {"model": ["4", 0]}},
                    "56": {"inputs": {"images": ["8", 0]}},
                }),
                encoding="utf-8",
            )
            (workflow_dir / "wf.kura.yaml").write_text("lora_insert:\n  kind: model_only\n  model_node: '12'\n", encoding="utf-8")
            (promptset_dir / "prompts.jsonl").write_text(json.dumps({"id": "p1", "prompt": "hello", "seeds": [123]}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "schema_version": 1,
                    "type": "render",
                    "inputs": {
                        "checkpoint": {"path": "runs/train-1/outputs/example.safetensors", "hash": None},
                        "workflow": {"path": "workflows/wf.json", "digest": None},
                        "promptset": {"path": "promptsets/prompts.jsonl", "digest": None},
                    },
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8189"},
                    "executor": {"name": "local"},
                    "workflow_patches": {"prompt": {"node": "6", "field": "inputs.text"}, "negative_prompt": {"node": "7", "field": "inputs.text"}, "seed": {"node": "3", "field": "inputs.seed"}},
                    "render": {"output_dir": "samples/images", "timeout_sec": 5, "default_seed": None},
                }),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            compile_render(root, run_dir)
            captured: dict[str, Any] = {}

            class FakeClient:
                def __init__(self, endpoint: str, timeout: int) -> None:
                    captured["endpoint"] = endpoint

                def lora_names(self) -> set[str]:
                    return set()

                def queue(self, workflow: dict[str, Any]) -> str:
                    captured["queued"] = True
                    return "prompt-1"

                def wait(self, prompt_id: str) -> list[dict[str, Any]]:
                    return [{"filename": "image.png", "subfolder": "", "type": "output"}]

                def download(self, image: dict[str, Any]) -> bytes:
                    return b"png"

            with patch("kura.render.ComfyUIClient", FakeClient):
                code = launch_render(root, run_dir)

            self.assertEqual(code, 1)
            self.assertNotIn("queued", captured)
            self.assertFalse(any((lora_dir / "Kura_tmp").glob("*.safetensors")))
            stdout = (run_dir / "logs" / "stdout.log").read_text(encoding="utf-8")
            self.assertIn("LoRA stage is not visible", stdout)
            self.assertIn("http://127.0.0.1:8189", stdout)

    def test_lora_visibility_check_distinguishes_object_info_failure(self) -> None:
        class FailingClient:
            def lora_names(self) -> set[str]:
                raise RuntimeError("object_info unavailable")

        plan = {"target": "/tmp/Kura_tmp/example.safetensors", "lora_name": "Kura_tmp/example.safetensors"}

        with self.assertRaisesRegex(ValueError, "object_info is unavailable"):
            _ensure_lora_stage_visible(FailingClient(), "http://127.0.0.1:8190", plan)

    def test_lora_visibility_check_redacts_endpoint_userinfo(self) -> None:
        class FailingClient:
            def lora_names(self) -> set[str]:
                raise RuntimeError("object_info unavailable")

        plan = {"target": "/tmp/Kura_tmp/example.safetensors", "lora_name": "Kura_tmp/example.safetensors"}

        with self.assertRaises(ValueError) as caught:
            _ensure_lora_stage_visible(FailingClient(), "http://user:secret@127.0.0.1:8190", plan)

        message = str(caught.exception)
        self.assertIn("http://***@127.0.0.1:8190", message)
        self.assertNotIn("secret", message)

    def test_lora_visibility_check_retries_once_for_stale_object_info(self) -> None:
        class EventuallyVisibleClient:
            def __init__(self) -> None:
                self.calls = 0

            def lora_names(self) -> set[str]:
                self.calls += 1
                if self.calls == 1:
                    return set()
                return {"Kura_tmp/example.safetensors"}

        client = EventuallyVisibleClient()
        plan = {"target": "/tmp/Kura_tmp/example.safetensors", "lora_name": "Kura_tmp/example.safetensors"}

        with patch("kura.render.time.sleep") as sleep:
            _ensure_lora_stage_visible(client, "http://127.0.0.1:8190", plan)

        self.assertEqual(client.calls, 2)
        sleep.assert_called_once()

    def test_lora_stage_name_preserves_safetensors_suffix_when_truncated(self) -> None:
        source = Path("/tmp") / (("very-long-checkpoint-name-" * 20) + ".safetensors")
        name = _safe_stage_name("20260629-" + ("long-run-id-" * 20), source)

        self.assertLessEqual(len(name), 220)
        self.assertTrue(name.endswith(".safetensors"))
        self.assertRegex(name, r"-[0-9a-f]{8}\.safetensors$")


class RunPodLiveSyncTests(unittest.TestCase):
    def test_sync_remote_stdout_appends_progress_and_materializes_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "remote-run"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "status.json").write_text(
                json.dumps({"state": "running", "last_step": 0}),
                encoding="utf-8",
            )
            stdout = (
                b"steps:  23%|##        | 7/30 [00:14<00:46,  2.00s/it, avr_loss=0.123]\n"
                b"\n__KURA_LOG_SIZE__:77\n"
            )
            result = subprocess.CompletedProcess([], 0, stdout, b"")

            with patch("kura.cli.subprocess.run", return_value=result):
                synced = _sync_runpod_remote_stdout(
                    run_dir,
                    {"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"},
                    workspace="/workspace",
                    run_id="remote-run",
                )

            self.assertTrue(synced)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["last_step"], 7)
            self.assertEqual(status["total_steps"], 30)
            self.assertEqual(status["remote_log_bytes"], 77)
            self.assertIn("avr_loss=0.123", (run_dir / "logs" / "stdout.log").read_text(encoding="utf-8"))


class RunPodPullSelectionTests(unittest.TestCase):
    def test_duration_parser_accepts_common_suffixes(self) -> None:
        self.assertEqual(_parse_duration_seconds("30m"), 1800)
        self.assertEqual(_parse_duration_seconds("2h"), 7200)
        self.assertEqual(_parse_duration_seconds("45"), 45)
        self.assertEqual(_parse_duration_seconds(None), 0)

    def test_select_remote_outputs_defaults_to_latest_step(self) -> None:
        items = [
            {"name": "model-step00000100.safetensors", "path": "/workspace/a", "size": 1},
            {"name": "model-step00001000.safetensors", "path": "/workspace/b", "size": 2},
            {"name": "model-step00000500.safetensors", "path": "/workspace/c", "size": 3},
        ]
        selected = _select_remote_outputs(items)
        self.assertEqual([item["name"] for item in selected], ["model-step00001000.safetensors"])

    def test_select_remote_outputs_can_filter_since_step(self) -> None:
        items = [
            {"name": "model-step00000100.safetensors", "path": "/workspace/a", "size": 1},
            {"name": "model-step00001000.safetensors", "path": "/workspace/b", "size": 2},
            {"name": "model-step00001500.safetensors", "path": "/workspace/c", "size": 3},
        ]
        selected = _select_remote_outputs(items, since_step=1000)
        self.assertEqual([item["name"] for item in selected], ["model-step00001000.safetensors", "model-step00001500.safetensors"])


class AiToolkitBackendTests(unittest.TestCase):
    def _run(self) -> dict[str, object]:
        return {
            "id": "ai-toolkit-example",
            "type": "train",
            "backend": {"name": "ai-toolkit", "adapter_version": 1},
            "model": {"base": "black-forest-labs/FLUX.2-klein-base-4B"},
            "datasets": [{"id": "tiny", "digest": "sha256:abc"}],
            "params": {"rank": 4, "alpha": 4, "lr": 1.0e-4, "steps": 1, "batch_size": 1, "resolution": 512, "seed": 42},
            "backend_overrides": {},
        }

    def test_default_compile_writes_runnable_yaml_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ai-toolkit"
            compile_ai_toolkit(self._run(), destination)
            config = yaml.safe_load((destination.with_suffix(".yaml")).read_text(encoding="utf-8"))
        command = command_ai_toolkit(self._run())

        self.assertEqual(config["config"]["name"], "ai-toolkit-example")
        process = config["config"]["process"][0]
        self.assertEqual(process["model"]["name_or_path"], "black-forest-labs/FLUX.2-klein-base-4B")
        self.assertEqual(process["datasets"][0]["folder_path"], "/workspace/datasets/tiny/images")
        self.assertEqual(process["network"]["linear"], 4)
        self.assertEqual(process["train"]["steps"], 1)
        self.assertFalse(process["train"]["gradient_checkpointing"])
        self.assertFalse(process["model"]["quantize"])
        self.assertFalse(process["model"]["quantize_te"])
        self.assertFalse(process["model"]["low_vram"])
        self.assertEqual(command, {"cwd": "/opt/ai-toolkit", "argv": ["python", "run.py", "/workspace/runs/ai-toolkit-example/resolved/ai-toolkit.yaml"], "env": {}})

    def test_compile_rejects_non_mapping_native_config_override(self) -> None:
        run = self._run()
        run["backend_overrides"] = {"ai-toolkit": {"config": ["not", "a", "mapping"]}}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "backend_overrides.ai-toolkit.config"):
                compile_ai_toolkit(run, Path(directory) / "ai-toolkit")

    def test_malformed_backend_overrides_do_not_crash_ai_toolkit(self) -> None:
        run = self._run()
        run["backend_overrides"] = True
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ai-toolkit"
            compile_ai_toolkit(run, destination)
            config = yaml.safe_load(destination.with_suffix(".yaml").read_text(encoding="utf-8"))
        command = command_ai_toolkit(run)
        self.assertEqual(config["config"]["name"], "ai-toolkit-example")
        self.assertEqual(command["cwd"], "/opt/ai-toolkit")


class MusubiBackendTests(unittest.TestCase):
    def _run(self) -> dict[str, object]:
        return {
            "id": "musubi-example",
            "type": "train",
            "backend": {"name": "musubi-tuner", "adapter_version": 1},
            "model": {"base": "black-forest-labs/FLUX.2-klein-base-4B"},
            "datasets": [{"id": "tiny", "digest": "sha256:abc"}],
            "params": {"rank": 4, "alpha": 4, "lr": 1.0e-4, "steps": 30, "batch_size": 1, "resolution": [512, 512], "seed": 42},
            "backend_overrides": {
                "musubi-tuner": {
                    "architecture": "flux2",
                    "model_version": "klein-base-4b",
                    "model_paths": {
                        "dit": "/models/flux2-klein-base-4b.safetensors",
                        "vae": "/models/flux2-vae.safetensors",
                        "text_encoder": "/models/qwen_3_4b.safetensors",
                    },
                }
            },
        }

    def test_compile_musubi_writes_dataset_toml_and_command_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "musubi"
            compile_musubi_tuner(self._run(), destination)
            dataset_toml = (destination / "dataset.toml").read_text(encoding="utf-8")
            command = json.loads((destination / "command.json").read_text(encoding="utf-8"))
            bundle = yaml.safe_load((destination / "model-bundle.lock.yaml").read_text(encoding="utf-8"))
        self.assertIn("image_directory = \"/workspace/datasets/tiny/images\"", dataset_toml)
        self.assertIn("cache_directory = \"/workspace/runs/musubi-example/cache/musubi/tiny\"", dataset_toml)
        self.assertEqual(command["cwd"], "/opt/musubi-tuner")
        self.assertEqual(command["argv"][:2], ["bash", "-lc"])
        self.assertIn('export PATH="/opt/conda/bin:/usr/local/bin:$PATH"', command["argv"][2])
        self.assertIn("src/musubi_tuner/flux_2_cache_latents.py", command["argv"][2])
        self.assertIn("src/musubi_tuner/flux_2_cache_text_encoder_outputs.py", command["argv"][2])
        self.assertIn("src/musubi_tuner/flux_2_train_network.py", command["argv"][2])
        self.assertIn("--max_train_steps 30", command["argv"][2])
        self.assertIn("--save_precision bf16", command["argv"][2])
        self.assertNotIn("--gradient_checkpointing", command["argv"][2])
        self.assertNotIn("--blocks_to_swap", command["argv"][2])
        self.assertNotIn("--fp8_base", command["argv"][2])
        self.assertNotIn("hf_hub_download", command["argv"][2])
        self.assertEqual(bundle["architecture"], "flux2")
        sources = {item["role"]: item.get("source") for item in bundle["models"]}
        self.assertEqual(sources["dit"], "model_paths")
        self.assertEqual(sources["vae"], "model_paths")
        self.assertEqual(sources["text_encoder"], "model_paths")
        expected = {item["role"]: item["expected_format"] for item in bundle["models"]}
        self.assertEqual(expected["dit"], "flux2_dit")
        self.assertEqual(expected["vae"], "flux2_ae_or_vae")
        self.assertEqual(expected["text_encoder"], "qwen3_4b_text_encoder")
        self.assertEqual(bundle["output"]["lora_format"], "comfyui")

    def test_compile_musubi_uses_dataset_root_when_images_subdir_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "tiny"
            dataset.mkdir(parents=True)
            (dataset / "001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            run = self._run()
            destination = root / "runs" / "musubi-example" / "resolved" / "musubi"

            compile_musubi_tuner(run, destination)

            dataset_toml = (destination / "dataset.toml").read_text(encoding="utf-8")
        self.assertIn('image_directory = "/workspace/datasets/tiny"', dataset_toml)

    def test_compile_musubi_prefers_images_subdir_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "tiny"
            (dataset / "images").mkdir(parents=True)
            (dataset / "001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (dataset / "images" / "001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            run = self._run()
            destination = root / "runs" / "musubi-example" / "resolved" / "musubi"

            compile_musubi_tuner(run, destination)

            dataset_toml = (destination / "dataset.toml").read_text(encoding="utf-8")
        self.assertIn('image_directory = "/workspace/datasets/tiny/images"', dataset_toml)

    def test_command_musubi_only_adds_memory_saving_flags_when_explicit(self) -> None:
        run = self._run()
        run["backend_overrides"]["musubi-tuner"]["gradient_checkpointing"] = True
        run["backend_overrides"]["musubi-tuner"]["extra_args"] = ["--fp8_base", "--fp8_scaled", "--blocks_to_swap", "4"]

        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("--gradient_checkpointing", script)
        self.assertIn("--fp8_base", script)
        self.assertIn("--fp8_scaled", script)
        self.assertIn("--blocks_to_swap 4", script)

    def test_command_musubi_save_precision_defaults_to_bf16_but_can_be_overridden(self) -> None:
        run = self._run()
        script = command_musubi_tuner(run)["argv"][2]
        self.assertIn("--save_precision bf16", script)

        run["backend_overrides"]["musubi-tuner"]["save_precision"] = "fp16"
        script = command_musubi_tuner(run)["argv"][2]
        self.assertIn("--save_precision fp16", script)

    def test_command_musubi_rejects_invalid_save_precision(self) -> None:
        run = self._run()
        run["backend_overrides"]["musubi-tuner"]["save_precision"] = "int8"

        with self.assertRaisesRegex(ValueError, "save_precision"):
            command_musubi_tuner(run)

    def test_command_musubi_rejects_h2d_block_swap_without_gradient_checkpointing(self) -> None:
        run = self._run()
        run["backend_overrides"]["musubi-tuner"]["extra_args"] = ["--blocks_to_swap", "4", "--block_swap_h2d_only"]

        with self.assertRaisesRegex(ValueError, "H2D-only block swap requires explicit gradient_checkpointing"):
            command_musubi_tuner(run)

    def test_command_musubi_rejects_a40_flux2_9b_large_micro_batch(self) -> None:
        run = self._run()
        run["model"] = {"base": "black-forest-labs/FLUX.2-klein-base-9B"}
        run["compute"] = {"executor": "docker", "gpu": "NVIDIA A40"}
        run["params"]["batch_size"] = 4
        run["backend_overrides"] = {"musubi-tuner": {"architecture": "flux2", "model_version": "klein-base-9b"}}

        with self.assertRaisesRegex(ValueError, "batch_size=4 has been observed to OOM"):
            command_musubi_tuner(run)

    def test_command_musubi_allows_a40_flux2_9b_accumulated_effective_batch(self) -> None:
        run = self._run()
        run["model"] = {"base": "black-forest-labs/FLUX.2-klein-base-9B"}
        run["compute"] = {"executor": "docker", "gpu": "NVIDIA A40"}
        run["params"]["batch_size"] = 1
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "flux2",
                "model_version": "klein-base-9b",
                "gradient_checkpointing": True,
                "extra_args": ["--gradient_accumulation_steps", "4"],
            }
        }

        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("--gradient_checkpointing", script)
        self.assertIn("--gradient_accumulation_steps 4", script)

    def test_command_musubi_rejects_a40_flux2_9b_1024_without_checkpointing(self) -> None:
        run = self._run()
        run["model"] = {"base": "black-forest-labs/FLUX.2-klein-base-9B"}
        run["compute"] = {"executor": "docker", "gpu": "NVIDIA A40"}
        run["params"].update({"rank": 32, "batch_size": 1, "resolution": [1024, 1024]})
        run["backend_overrides"] = {"musubi-tuner": {"architecture": "flux2", "model_version": "klein-base-9b"}}

        with self.assertRaisesRegex(ValueError, "observed to OOM even with batch_size=1"):
            command_musubi_tuner(run)

    def test_command_musubi_rejects_secret_explicit_env(self) -> None:
        run = self._run()
        run["backend_overrides"]["musubi-tuner"]["command"] = {"cwd": "/opt/musubi-tuner", "argv": ["python", "train.py"], "env": {"HF_TOKEN": "secret"}}

        with self.assertRaisesRegex(ValueError, "env must not contain secrets"):
            command_musubi_tuner(run)

    def test_command_musubi_allows_non_secret_generated_env(self) -> None:
        run = self._run()
        run["backend_overrides"]["musubi-tuner"]["env"] = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

        command = command_musubi_tuner(run)

        self.assertEqual(command["env"], {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})

    def test_command_musubi_rejects_secret_generated_env(self) -> None:
        run = self._run()
        run["backend_overrides"]["musubi-tuner"]["env"] = {"HF_TOKEN": "secret"}

        with self.assertRaisesRegex(ValueError, "env must not contain secrets"):
            command_musubi_tuner(run)

    def test_command_musubi_rejects_invalid_extra_args_shape(self) -> None:
        run = self._run()
        run["backend_overrides"]["musubi-tuner"]["extra_args"] = "--fp8_base"

        with self.assertRaisesRegex(ValueError, "extra_args must be a list of strings"):
            command_musubi_tuner(run)

    def test_compile_musubi_can_write_paired_control_jsonl_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "paired"
            (dataset / "paired" / "target").mkdir(parents=True)
            (dataset / "paired" / "cond").mkdir()
            (dataset / "paired" / "caption").mkdir()
            (dataset / "paired" / "target" / "item1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (dataset / "paired" / "cond" / "item1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (dataset / "paired" / "caption" / "item1.txt").write_text("control caption\n", encoding="utf-8")
            run = self._run()
            run["id"] = "paired-run"
            run["datasets"] = [{"id": "paired", "digest": "sha256:abc"}]
            run["backend_overrides"]["musubi-tuner"]["dataset_config"] = {
                "general": {"resolution": [1024, 1024], "batch_size": 4},
                "datasets": [
                    {
                        "paired_jsonl": {
                            "filename": "paired_1024.jsonl",
                            "target_dir": "paired/target",
                            "control_dir": "paired/cond",
                            "caption_dir": "paired/caption",
                        },
                        "resolution": [1024, 1024],
                        "control_resolution": [1024, 1024],
                    }
                ],
            }
            destination = root / "runs" / "paired-run" / "resolved" / "musubi"

            compile_musubi_tuner(run, destination)

            dataset_toml = (destination / "dataset.toml").read_text(encoding="utf-8")
            rows = (destination / "paired_1024.jsonl").read_text(encoding="utf-8").splitlines()
            payload = json.loads(rows[0])
        self.assertIn('image_jsonl_file = "/workspace/runs/paired-run/resolved/musubi/paired_1024.jsonl"', dataset_toml)
        self.assertNotIn("image_directory", dataset_toml)
        self.assertIn("control_resolution = [1024, 1024]", dataset_toml)
        self.assertEqual(payload["image_path"], "/workspace/datasets/paired/paired/target/item1.png")
        self.assertEqual(payload["control_path"], "/workspace/datasets/paired/paired/cond/item1.png")
        self.assertEqual(payload["caption"], "control caption")

    def test_compile_musubi_paired_jsonl_uses_explicit_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "paired"
            (dataset / "paired" / "target").mkdir(parents=True)
            (dataset / "paired" / "cond").mkdir()
            (dataset / "paired" / "caption").mkdir()
            (dataset / "paired" / "target" / "item1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (dataset / "paired" / "cond" / "item1.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            (dataset / "paired" / "caption" / "item1.txt").write_text("control caption\n", encoding="utf-8")
            run = self._run()
            run["id"] = "paired-run"
            run["datasets"] = [{"id": "paired", "digest": "sha256:abc"}]
            run["backend_overrides"]["musubi-tuner"]["dataset_config"] = {
                "datasets": [{"paired_jsonl": {"filename": "paired.jsonl", "target_dir": "paired/target", "control_dir": "paired/cond", "caption_dir": "paired/caption"}}],
            }
            destination = root / "scratch" / "musubi"

            compile_musubi_tuner(run, destination, workspace=root)

            self.assertTrue((destination / "paired.jsonl").is_file())

    def test_compile_musubi_rejects_duplicate_resolution_sections_for_same_paired_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "paired"
            (dataset / "paired" / "target").mkdir(parents=True)
            (dataset / "paired" / "cond").mkdir()
            (dataset / "paired" / "caption").mkdir()
            for name in ("item1", "item2"):
                (dataset / "paired" / "target" / f"{name}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
                (dataset / "paired" / "cond" / f"{name}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
                (dataset / "paired" / "caption" / f"{name}.txt").write_text("control caption\n", encoding="utf-8")
            run = self._run()
            run["id"] = "paired-buckets"
            run["datasets"] = [{"id": "paired", "role": "paired-768"}, {"id": "paired", "role": "paired-1024"}]
            run["backend_overrides"]["musubi-tuner"]["dataset_config"] = {
                "general": {"resolution": [1024, 1024], "batch_size": 4, "enable_bucket": True},
                "datasets": [
                    {
                        "paired_jsonl": {"filename": "paired_768.jsonl", "target_dir": "paired/target", "control_dir": "paired/cond", "caption_dir": "paired/caption"},
                        "resolution": [768, 768],
                        "control_resolution": [768, 768],
                        "batch_size": 4,
                    },
                    {
                        "paired_jsonl": {"filename": "paired_1024.jsonl", "target_dir": "paired/target", "control_dir": "paired/cond", "caption_dir": "paired/caption"},
                        "resolution": [1024, 1024],
                        "control_resolution": [1024, 1024],
                        "batch_size": 4,
                    },
                ],
            }
            destination = root / "runs" / "paired-buckets" / "resolved" / "musubi"

            with self.assertRaisesRegex(ValueError, "ambiguous Musubi duplicate dataset blocks"):
                compile_musubi_tuner(run, destination)

    def test_compile_musubi_allows_disjoint_resolution_sections_for_same_paired_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "paired"
            (dataset / "paired" / "target").mkdir(parents=True)
            (dataset / "paired" / "cond").mkdir()
            (dataset / "paired" / "caption").mkdir()
            for name in ("item1", "item2", "item3", "item4"):
                (dataset / "paired" / "target" / f"{name}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
                (dataset / "paired" / "cond" / f"{name}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
                (dataset / "paired" / "caption" / f"{name}.txt").write_text("control caption\n", encoding="utf-8")
            run = self._run()
            run["id"] = "paired-buckets"
            run["datasets"] = [{"id": "paired", "role": "paired-768"}, {"id": "paired", "role": "paired-1024"}]
            run["backend_overrides"]["musubi-tuner"]["dataset_config"] = {
                "general": {"resolution": [1024, 1024], "batch_size": 4, "enable_bucket": True},
                "datasets": [
                    {
                        "paired_jsonl": {
                            "filename": "paired_768.jsonl",
                            "target_dir": "paired/target",
                            "control_dir": "paired/cond",
                            "caption_dir": "paired/caption",
                            "select": {"modulo": 2, "remainder": 0},
                        },
                        "resolution": [768, 768],
                        "control_resolution": [768, 768],
                        "batch_size": 4,
                    },
                    {
                        "paired_jsonl": {
                            "filename": "paired_1024.jsonl",
                            "target_dir": "paired/target",
                            "control_dir": "paired/cond",
                            "caption_dir": "paired/caption",
                            "select": {"modulo": 2, "remainder": 1},
                        },
                        "resolution": [1024, 1024],
                        "control_resolution": [1024, 1024],
                        "batch_size": 4,
                    },
                ],
            }
            destination = root / "runs" / "paired-buckets" / "resolved" / "musubi"

            compile_musubi_tuner(run, destination)

            dataset_toml = (destination / "dataset.toml").read_text(encoding="utf-8")
            jsonl_files = sorted(path.name for path in destination.glob("*.jsonl"))
            rows_768 = (destination / "paired_768.jsonl").read_text(encoding="utf-8").splitlines()
            rows_1024 = (destination / "paired_1024.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(dataset_toml.count("[[datasets]]"), 2)
        self.assertIn("resolution = [768, 768]", dataset_toml)
        self.assertIn("resolution = [1024, 1024]", dataset_toml)
        self.assertIn("control_resolution = [768, 768]", dataset_toml)
        self.assertIn("control_resolution = [1024, 1024]", dataset_toml)
        self.assertEqual(jsonl_files, ["paired_1024.jsonl", "paired_768.jsonl"])
        self.assertEqual(len(rows_768), 2)
        self.assertEqual(len(rows_1024), 2)

    def test_command_musubi_requires_explicit_model_paths(self) -> None:
        run = self._run()
        run["backend_overrides"] = {"musubi-tuner": {"architecture": "wan"}}
        with self.assertRaisesRegex(ValueError, "model_paths, model_downloads, or a known model.base bundle"):
            command_musubi_tuner(run)

    def test_command_musubi_unknown_architecture_names_kura_adapter_layer(self) -> None:
        run = self._run()
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "sdxl",
                "model_paths": {"unet": "/models/sdxl.safetensors"},
            }
        }
        with self.assertRaisesRegex(ValueError, "unsupported Kura built-in Musubi adapter"):
            command_musubi_tuner(run)

    def test_command_musubi_generates_image_architecture_adapters(self) -> None:
        cases = [
            (
                "qwen_image",
                {"dit": "/models/qwen-dit.safetensors", "vae": "/models/qwen-vae.safetensors", "text_encoder": "/models/qwen-vl.safetensors"},
                ("qwen_image_cache_latents.py", "qwen_image_cache_text_encoder_outputs.py", "qwen_image_train_network.py", "networks.lora_qwen_image", "--model_version original"),
            ),
            (
                "zimage",
                {"dit": "/models/zimage-dit.safetensors", "vae": "/models/zimage-vae.safetensors", "text_encoder": "/models/zimage-qwen3.safetensors"},
                ("zimage_cache_latents.py", "zimage_cache_text_encoder_outputs.py", "zimage_train_network.py", "networks.lora_zimage", "--dit /models/zimage-dit.safetensors"),
            ),
            (
                "flux_kontext",
                {"dit": "/models/kontext-dit.safetensors", "vae": "/models/kontext-vae.safetensors", "text_encoder1": "/models/t5.safetensors", "text_encoder2": "/models/clip.safetensors"},
                ("flux_kontext_cache_latents.py", "flux_kontext_cache_text_encoder_outputs.py", "flux_kontext_train_network.py", "networks.lora_flux", "--text_encoder2 /models/clip.safetensors"),
            ),
            (
                "ideogram4",
                {"dit": "/models/ideogram4.safetensors", "vae": "/models/flux2-vae.safetensors", "text_encoder": "/models/qwen3vl.safetensors"},
                ("ideogram4_cache_latents.py", "ideogram4_cache_text_encoder_outputs.py", "ideogram4_train_network.py", "networks.lora_ideogram4", "--dit /models/ideogram4.safetensors"),
            ),
            (
                "hidream_o1",
                {"dit": "/models/hidream-o1.safetensors"},
                ("hidream_o1_cache_pixel.py", "hidream_o1_cache_text_encoder_outputs.py", "hidream_o1_train_network.py", "networks.lora_hidream_o1", "--model_type full", "--task t2i"),
            ),
        ]
        for architecture, model_paths, expected in cases:
            with self.subTest(architecture=architecture):
                run = self._run()
                run["backend_overrides"] = {"musubi-tuner": {"architecture": architecture, "model_paths": model_paths}}
                script = command_musubi_tuner(run)["argv"][2]
                for text in expected:
                    self.assertIn(text, script)
                self.assertIn("--max_train_steps 30", script)
                self.assertIn("--save_precision bf16", script)

    def test_command_musubi_generates_video_architecture_adapters(self) -> None:
        cases = [
            (
                "hunyuan_video",
                {"dit": "/models/hv-dit.safetensors", "vae": "/models/hv-vae.safetensors", "text_encoder1": "hunyuanvideo-community/HunyuanVideo", "text_encoder2": "openai/clip-vit-large-patch14"},
                ("cache_latents.py", "cache_text_encoder_outputs.py", "hv_train_network.py", "networks.lora", "--text_encoder1 hunyuanvideo-community/HunyuanVideo"),
            ),
            (
                "hunyuan_video_1_5",
                {"dit": "/models/hv15-dit.safetensors", "vae": "/models/hv15-vae.safetensors", "text_encoder": "Qwen/Qwen2.5-VL-7B-Instruct", "byt5": "google/byt5-small"},
                ("hv_1_5_cache_latents.py", "hv_1_5_cache_text_encoder_outputs.py", "hv_1_5_train_network.py", "networks.lora_hv_1_5", "--task t2v"),
            ),
            (
                "framepack",
                {"dit": "/models/framepack.safetensors", "vae": "/models/fpack-vae.safetensors", "text_encoder1": "hunyuanvideo-community/HunyuanVideo", "text_encoder2": "openai/clip-vit-large-patch14", "image_encoder": "/models/siglip.safetensors"},
                ("fpack_cache_latents.py", "fpack_cache_text_encoder_outputs.py", "fpack_train_network.py", "networks.lora_framepack", "--image_encoder /models/siglip.safetensors"),
            ),
            (
                "kandinsky5",
                {"dit": "/models/k5-dit.safetensors", "vae": "/models/k5-vae.safetensors", "text_encoder_qwen": "Qwen/Qwen2.5-VL-7B-Instruct", "text_encoder_clip": "openai/clip-vit-large-patch14"},
                ("kandinsky5_cache_text_encoder_outputs.py", "kandinsky5_cache_latents.py", "kandinsky5_train_network.py", "networks.lora_kandinsky", "--task k5-pro-t2v-5s-sd"),
            ),
        ]
        for architecture, model_paths, expected in cases:
            with self.subTest(architecture=architecture):
                run = self._run()
                run["backend_overrides"] = {"musubi-tuner": {"architecture": architecture, "model_paths": model_paths}}
                script = command_musubi_tuner(run)["argv"][2]
                for text in expected:
                    self.assertIn(text, script)
                self.assertIn("--max_train_steps 30", script)
                self.assertIn("--save_precision bf16", script)

    def test_command_musubi_framepack_uses_current_fp8_base_flag(self) -> None:
        run = self._run()
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "framepack",
                "model_paths": {
                    "dit": "/models/framepack.safetensors",
                    "vae": "/models/fpack-vae.safetensors",
                    "text_encoder1": "hunyuanvideo-community/HunyuanVideo",
                    "text_encoder2": "openai/clip-vit-large-patch14",
                    "image_encoder": "/models/siglip.safetensors",
                },
                "fp8_base": True,
            }
        }

        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("--fp8_base", script)
        self.assertNotIn("--fp8 ", script)

    def test_command_musubi_wan_22_supports_dual_noise_models(self) -> None:
        run = self._run()
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "wan",
                "task": "t2v-A14B",
                "model_paths": {
                    "dit": "/models/wan22-low.safetensors",
                    "dit_high_noise": "/models/wan22-high.safetensors",
                    "vae": "/models/wan-vae.safetensors",
                    "t5": "/models/umt5.pth",
                },
                "timestep_boundary": 0.875,
            }
        }

        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("--dit_high_noise /models/wan22-high.safetensors", script)
        self.assertIn("--timestep_boundary 0.875", script)

    def test_command_musubi_flux2_dev_uses_dev_contract(self) -> None:
        run = self._run()
        run["model"]["base"] = "black-forest-labs/FLUX.2-dev"
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "flux2",
                "model_version": "dev",
                "model_paths": {
                    "dit": "/models/flux2-dev.safetensors",
                    "vae": "/models/ae.safetensors",
                    "text_encoder": "/models/mistral-00001-of-00010.safetensors",
                },
                "fp8_base": True,
                "fp8_scaled": True,
                "vae_dtype": "bfloat16",
            }
        }

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "musubi"
            compile_musubi_tuner(run, destination)
            bundle = yaml.safe_load((destination / "model-bundle.lock.yaml").read_text(encoding="utf-8"))
            script = json.loads((destination / "command.json").read_text(encoding="utf-8"))["argv"][2]

        expected = {item["role"]: item["expected_format"] for item in bundle["models"]}
        self.assertEqual(expected["vae"], "flux2_ae_or_vae")
        self.assertEqual(expected["text_encoder"], "safetensors")
        self.assertIn("--model_version dev", script)
        self.assertIn("--fp8_base", script)
        self.assertIn("--fp8_scaled", script)
        self.assertIn("--vae_dtype bfloat16", script)

    def test_command_musubi_flux2_dev_rejects_qwen_only_fp8_text_encoder(self) -> None:
        run = self._run()
        run["backend_overrides"]["musubi-tuner"].update({"model_version": "dev", "fp8_text_encoder": True})

        with self.assertRaisesRegex(ValueError, "does not support fp8_text_encoder"):
            command_musubi_tuner(run)

    def test_command_musubi_wan_one_frame_updates_cache_and_train(self) -> None:
        run = self._run()
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "wan",
                "task": "i2v-14B",
                "one_frame": True,
                "model_paths": {
                    "dit": "/models/wan-i2v.safetensors",
                    "vae": "/models/wan-vae.safetensors",
                    "t5": "/models/umt5.pth",
                    "clip": "/models/clip.pth",
                },
            }
        }

        script = command_musubi_tuner(run)["argv"][2]

        self.assertGreaterEqual(script.count("--one_frame"), 2)
        self.assertIn("--clip /models/clip.pth", script)

    def test_command_musubi_wan_21_i2v_requires_clip(self) -> None:
        run = self._run()
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "wan",
                "task": "i2v-14B",
                "model_paths": {
                    "dit": "/models/wan-i2v.safetensors",
                    "vae": "/models/wan-vae.safetensors",
                    "t5": "/models/umt5.pth",
                },
            }
        }

        with self.assertRaisesRegex(ValueError, "requires model_paths.clip"):
            command_musubi_tuner(run)

    def test_command_musubi_framepack_one_frame_updates_cache_and_train(self) -> None:
        run = self._run()
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "framepack",
                "one_frame": True,
                "one_frame_no_2x": True,
                "one_frame_no_4x": True,
                "model_paths": {
                    "dit": "/models/framepack.safetensors",
                    "vae": "/models/fpack-vae.safetensors",
                    "text_encoder1": "hunyuanvideo-community/HunyuanVideo",
                    "text_encoder2": "openai/clip-vit-large-patch14",
                    "image_encoder": "/models/siglip.safetensors",
                },
            }
        }

        script = command_musubi_tuner(run)["argv"][2]

        self.assertGreaterEqual(script.count("--one_frame"), 4)
        self.assertIn("--one_frame_no_2x", script)
        self.assertIn("--one_frame_no_4x", script)

    def test_command_musubi_qwen_model_versions_reach_all_three_stages(self) -> None:
        for model_version in ("original", "edit", "edit-2509", "edit-2511", "layered"):
            with self.subTest(model_version=model_version):
                run = self._run()
                run["backend_overrides"] = {
                    "musubi-tuner": {
                        "architecture": "qwen_image",
                        "model_version": model_version,
                        "model_paths": {
                            "dit": "/models/qwen-dit.safetensors",
                            "vae": "/models/qwen-vae.safetensors",
                            "text_encoder": "/models/qwen-vl.safetensors",
                        },
                    }
                }

                script = command_musubi_tuner(run)["argv"][2]

                self.assertEqual(script.count(f"--model_version {model_version}"), 3)

    def test_command_musubi_hunyuan_15_i2v_updates_cache_and_train(self) -> None:
        run = self._run()
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "hunyuan_video_1_5",
                "task": "i2v",
                "model_paths": {
                    "dit": "/models/hv15-i2v.safetensors",
                    "vae": "/models/hv15-vae.safetensors",
                    "text_encoder": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "byt5": "google/byt5-small",
                    "image_encoder": "/models/bytedance-byt5-small.safetensors",
                },
            }
        }

        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("--task i2v", script)
        self.assertGreaterEqual(script.count("--image_encoder /models/bytedance-byt5-small.safetensors"), 2)
        self.assertIn("--i2v", script)

    def test_command_musubi_hidream_i2i_preserves_control_contract(self) -> None:
        run = self._run()
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "hidream_o1",
                "task": "i2i",
                "model_type": "dev",
                "model_paths": {"dit": "/models/hidream-dev.safetensors"},
                "dataset_config": {"datasets": [{"control_directory": "/workspace/datasets/tiny/control"}]},
                "extra_args": ["--network_args", "conv_dim=4", "conv_alpha=1"],
            }
        }

        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("--task i2i", script)
        self.assertIn("--model_type dev", script)
        self.assertIn("--network_args conv_dim=4 conv_alpha=1", script)

    def test_command_musubi_kandinsky_i2v_preserves_task(self) -> None:
        run = self._run()
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "kandinsky5",
                "task": "k5-pro-i2v-5s-sd",
                "model_paths": {
                    "dit": "/models/k5-i2v.safetensors",
                    "vae": "/models/k5-vae.safetensors",
                    "text_encoder_qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "text_encoder_clip": "openai/clip-vit-large-patch14",
                },
            }
        }

        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("--task k5-pro-i2v-5s-sd", script)

    def test_command_musubi_kandinsky_can_quantize_qwen_cache(self) -> None:
        run = self._run()
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "kandinsky5",
                "model_paths": {
                    "dit": "/models/k5-dit.safetensors",
                    "vae": "/models/k5-vae.safetensors",
                    "text_encoder_qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "text_encoder_clip": "openai/clip-vit-large-patch14",
                },
                "quantized_qwen": True,
            }
        }

        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("kandinsky5_cache_text_encoder_outputs.py", script)
        self.assertIn("--quantized_qwen", script)

    def test_command_musubi_can_download_models_from_huggingface(self) -> None:
        run = self._run()
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "flux2",
                "model_version": "klein-base-4b",
                "model_downloads": {
                    "dit": {"repo": "black-forest-labs/FLUX.2-klein-base-4B", "filename": "flux2-klein-base-4b.safetensors"},
                    "vae": {"repo": "black-forest-labs/FLUX.2-dev", "filename": "flux2-vae.safetensors"},
                    "text_encoder": {"repo": "black-forest-labs/FLUX.2-klein-4B", "filename": "text_encoder/qwen_3_4b.safetensors"},
                },
            }
        }
        command = command_musubi_tuner(run)
        script = command["argv"][2]
        self.assertIn("hf_hub_download", script)
        self.assertIn("HF_HUB_DISABLE_XET", script)
        self.assertIn('cache_dir = os.environ.get("HF_HUB_CACHE")', script)
        self.assertIn("HF_HUB_CACHE is required before downloading models", script)
        self.assertNotIn('or "/root/.cache/huggingface"', script)
        self.assertNotIn("local_dir", script)
        self.assertNotIn("/workspace/cache/hf-models/musubi", script)
        self.assertIn("KURA_HF_DOWNLOAD_NO_PROGRESS_SEC", script)
        self.assertIn("repo_cache_dirs(cache_dir, item)", script)
        self.assertNotIn("remove_incomplete_files(cache_dir)", script)
        self.assertIn("removed {removed} incomplete", script)
        self.assertLess(script.index("musubi dataset ok"), script.index("hf_hub_download"))
        self.assertIn("def stable_link_target", script)
        self.assertIn("require_cache_mappable(cache_dir, link_path)", script)
        self.assertIn("os.symlink(stable_link_target(path, link_path), link_path)", script)
        self.assertIn("black-forest-labs/FLUX.2-klein-base-4B", script)
        self.assertIn("/workspace/cache/models/musubi/black-forest-labs--FLUX.2-klein-base-4B/dit/flux2-klein-base-4b.safetensors", script)
        self.assertIn("--dit /workspace/cache/models/musubi/black-forest-labs--FLUX.2-klein-base-4B/dit/flux2-klein-base-4b.safetensors", script)
        self.assertIn("flux2_ae_or_vae", script)
        self.assertIn("qwen3_4b_text_encoder", script)
        self.assertIn("lora_unet_*", script)
        self.assertLess(script.index("hf_hub_download"), script.index("src/musubi_tuner/flux_2_cache_latents.py"))
        self.assertLess(script.index("expected_format"), script.index("src/musubi_tuner/flux_2_cache_latents.py"))
        self.assertLess(script.index("src/musubi_tuner/flux_2_train_network.py"), script.rindex("lora_unet_*"))

    def test_hf_download_links_workspace_cache_relatively(self) -> None:
        namespace: dict[str, Any] = {"__name__": "__test__"}
        exec(script_source("hf_download.py"), namespace)

        stable_link_target = namespace["stable_link_target"]

        target = "/workspace/cache/huggingface/hub/models--repo--model/snapshots/abc/weights.safetensors"
        link_path = "/workspace/cache/models/musubi/repo--model/dit/weights.safetensors"
        self.assertEqual(
            stable_link_target(target, link_path),
            "../../../../huggingface/hub/models--repo--model/snapshots/abc/weights.safetensors",
        )
        with self.assertRaisesRegex(SystemExit, "cannot map downloaded model path"):
            stable_link_target("/root/.cache/huggingface/weights.safetensors", link_path)

    def test_hf_download_rejects_unmapped_cache_before_download(self) -> None:
        namespace: dict[str, Any] = {"__name__": "__test__"}
        exec(script_source("hf_download.py"), namespace)

        require_cache_mappable = namespace["require_cache_mappable"]
        link_path = "/workspace/cache/models/musubi/repo--model/dit/weights.safetensors"
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "HF_HUB_CACHE must be inside /workspace"):
                require_cache_mappable("/root/.cache/huggingface", link_path)
        require_cache_mappable("/workspace/cache/huggingface/hub", link_path)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "HF_HUB_CACHE must be inside /workspace"):
                require_cache_mappable("/root/.cache/huggingface", "/tmp/model.safetensors")

    def test_hf_download_requires_hf_hub_cache_before_download(self) -> None:
        namespace: dict[str, Any] = {"__name__": "__test__"}
        exec(script_source("hf_download.py"), namespace)

        run_one = namespace["run_one"]
        item = {"key": "dit", "repo_id": "owner/model", "filename": "weights.safetensors", "link_path": "/workspace/cache/models/musubi/owner--model/dit/weights.safetensors"}
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "HF_HUB_CACHE is required"):
                run_one(item)

    def test_hf_download_uses_workspace_path_maps_for_symlink_targets(self) -> None:
        namespace: dict[str, Any] = {"__name__": "__test__"}
        exec(script_source("hf_download.py"), namespace)

        stable_link_target = namespace["stable_link_target"]
        require_cache_mappable = namespace["require_cache_mappable"]
        link_path = "/workspace/cache/models/musubi/repo--model/dit/weights.safetensors"
        with patch.dict(
            os.environ,
            {"KURA_WORKSPACE_PATH_MAPS": json.dumps([{"container": "/cache/hf", "workspace": "/workspace/shared/hf"}])},
        ):
            require_cache_mappable("/cache/hf", link_path)
            self.assertEqual(
                stable_link_target("/cache/hf/hub/models--repo--model/snapshots/abc/weights.safetensors", link_path),
                "../../../../../shared/hf/hub/models--repo--model/snapshots/abc/weights.safetensors",
            )

    def test_command_musubi_rejects_model_download_local_dir(self) -> None:
        run = self._run()
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "flux2",
                "model_version": "klein-base-4b",
                "model_downloads": {
                    "dit": {
                        "repo": "black-forest-labs/FLUX.2-klein-base-4B",
                        "filename": "flux2-klein-base-4b.safetensors",
                        "local_dir": "/workspace/cache/hf-models/musubi/legacy",
                    },
                },
            }
        }
        with self.assertRaisesRegex(ValueError, "model_downloads.local_dir is not supported"):
            command_musubi_tuner(run)

    def test_command_musubi_resolves_known_flux2_klein_bundle(self) -> None:
        run = self._run()
        run["backend_overrides"] = {"musubi-tuner": {"architecture": "flux2", "model_version": "klein-base-4b"}}
        command = command_musubi_tuner(run)
        script = command["argv"][2]
        self.assertIn("Comfy-Org/vae-text-encorder-for-flux-klein-4b", script)
        self.assertIn("split_files/diffusion_models/flux-2-klein-base-4b.safetensors", script)
        self.assertIn("split_files/vae/flux2-vae.safetensors", script)
        self.assertIn("split_files/text_encoders/qwen_3_4b.safetensors", script)
        self.assertIn("--vae /workspace/cache/models/musubi/Comfy-Org--vae-text-encorder-for-flux-klein-4b/vae/split_files/vae/flux2-vae.safetensors", script)

    def test_command_musubi_resolves_known_flux2_klein_base_9b_bundle(self) -> None:
        run = self._run()
        run["model"] = {"base": "black-forest-labs/FLUX.2-klein-base-9B"}
        run["backend_overrides"] = {"musubi-tuner": {"architecture": "flux2", "model_version": "klein-base-9b"}}
        command = command_musubi_tuner(run)
        script = command["argv"][2]

        self.assertIn("black-forest-labs/FLUX.2-klein-base-9B", script)
        self.assertIn("flux-2-klein-base-9b.safetensors", script)
        self.assertIn("vae/diffusion_pytorch_model.safetensors", script)
        self.assertIn("text_encoder/model-00001-of-00004.safetensors", script)
        self.assertIn("text_encoder/model-00004-of-00004.safetensors", script)
        self.assertIn("text_encoder/model.safetensors.index.json", script)
        self.assertIn("--model_version klein-base-9b", script)
        self.assertIn("--vae /workspace/cache/models/musubi/black-forest-labs--FLUX.2-klein-base-9B/vae/vae/diffusion_pytorch_model.safetensors", script)
        self.assertIn("--text_encoder /workspace/cache/models/musubi/black-forest-labs--FLUX.2-klein-base-9B/text_encoder/text_encoder/model-00001-of-00004.safetensors", script)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "musubi"
            compile_musubi_tuner(run, destination)
            bundle = yaml.safe_load((destination / "model-bundle.lock.yaml").read_text(encoding="utf-8"))
        expected = {item["role"]: item["expected_format"] for item in bundle["models"]}
        self.assertEqual(expected["vae"], "flux2_ae_or_vae")
        self.assertEqual(expected["text_encoder"], "qwen3_8b_text_encoder")

    def test_command_musubi_resolves_known_krea2_bundle(self) -> None:
        run = self._run()
        run["model"] = {"base": "krea/Krea-2-Raw"}
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "krea2",
                "gradient_checkpointing": True,
                "fp8_base": True,
            }
        }
        command = command_musubi_tuner(run)
        script = command["argv"][2]

        self.assertIn("krea/Krea-2-Raw", script)
        self.assertIn("raw.safetensors", script)
        self.assertIn("Comfy-Org/Qwen-Image_ComfyUI", script)
        self.assertIn("split_files/vae/qwen_image_vae.safetensors", script)
        self.assertIn("Comfy-Org/Qwen3-VL", script)
        self.assertIn("text_encoders/qwen3vl_4b_bf16.safetensors", script)
        self.assertIn("src/musubi_tuner/krea2_cache_latents.py", script)
        self.assertIn("src/musubi_tuner/krea2_cache_text_encoder_outputs.py", script)
        self.assertIn("src/musubi_tuner/krea2_train_network.py", script)
        self.assertIn("--network_module networks.lora_krea2", script)
        self.assertIn("--timestep_sampling krea2_shift", script)
        self.assertIn("--fp8_base --fp8_scaled", script)
        self.assertIn("--gradient_checkpointing", script)
        self.assertIn("--save_precision bf16", script)
        self.assertNotIn("--text_encoder /workspace/cache/models/musubi/Comfy-Org--Qwen3-VL/text_encoder", script.split("src/musubi_tuner/krea2_train_network.py", 1)[1])

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "musubi"
            compile_musubi_tuner(run, destination)
            bundle = yaml.safe_load((destination / "model-bundle.lock.yaml").read_text(encoding="utf-8"))
        self.assertEqual(bundle["architecture"], "krea2")
        expected = {item["role"]: item["expected_format"] for item in bundle["models"]}
        self.assertEqual(expected["dit"], "safetensors")
        self.assertEqual(expected["vae"], "safetensors")
        self.assertEqual(expected["text_encoder"], "safetensors")

    def test_command_musubi_krea2_can_include_turbo_for_samples(self) -> None:
        run = self._run()
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "krea2",
                "include_turbo_dit": True,
                "extra_args": ["--sample_prompts", "/workspace/prompts.txt", "--sample_every_n_steps", "100"],
            }
        }
        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("krea/Krea-2-Turbo", script)
        self.assertIn("turbo.safetensors", script)
        self.assertIn("--text_encoder /workspace/cache/models/musubi/Comfy-Org--Qwen3-VL/text_encoder/text_encoders/qwen3vl_4b_bf16.safetensors", script)
        self.assertIn("--turbo_dit /workspace/cache/models/musubi/krea--Krea-2-Turbo/turbo_dit/turbo.safetensors", script)

    def test_command_musubi_krea2_rejects_paired_control_dataset(self) -> None:
        run = self._run()
        run["backend_overrides"] = {
            "musubi-tuner": {
                "architecture": "krea2",
                "dataset_config": {
                    "datasets": [
                        {
                            "id": "tiny",
                            "paired_jsonl": [{"image": "a.png", "conditioning_image": "b.png", "caption": "caption"}],
                        }
                    ]
                },
            }
        }

        with self.assertRaisesRegex(ValueError, "plain image/caption datasets only"):
            command_musubi_tuner(run)

    def test_command_musubi_infers_flux2_model_version_from_model_base(self) -> None:
        run = self._run()
        run["model"] = {"base": "black-forest-labs/FLUX.2-klein-base-9B"}
        run["backend_overrides"] = {"musubi-tuner": {"architecture": "flux2"}}
        command = command_musubi_tuner(run)

        self.assertIn("--model_version klein-base-9b", command["argv"][2])
        self.assertNotIn("klein-base-4b", command["argv"][2])

    def test_command_musubi_refuses_unknown_flux2_model_version_default(self) -> None:
        run = self._run()
        run["model"] = {"base": "custom/flux2-checkpoint"}
        run["backend_overrides"]["musubi-tuner"].pop("model_version", None)

        with self.assertRaisesRegex(ValueError, "refusing to default to 4B"):
            command_musubi_tuner(run)

    def test_command_musubi_can_prune_early_step_checkpoints(self) -> None:
        run = self._run()
        run["backend_overrides"]["musubi-tuner"]["save_every_n_steps"] = 100
        run["backend_overrides"]["musubi-tuner"]["prune_checkpoints_before_step"] = 1000
        command = command_musubi_tuner(run)
        script = command["argv"][2]

        self.assertIn("--save_every_n_steps 100", script)
        self.assertIn("[kura] pruned", script)
        self.assertIn("threshold", script)
        self.assertIn("musubi-example 1000", script)
        self.assertLess(script.index("flux_2_train_network.py"), script.index("[kura] pruned"))
        self.assertLess(script.index("[kura] pruned"), script.rindex("lora_unet_*"))

    def test_safetensors_preflight_rejects_ambiguous_flux1_ae_filename(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ae.safetensors"
            _write_fake_safetensors(path, ["encoder.down.0.block.0.weight", "decoder.up.0.block.0.weight", "quant_conv.weight"])
            spec = {"models": [{"role": "vae", "path": str(path), "expected_format": "flux2_vae"}]}
            result = subprocess.run([sys.executable, "-c", _safetensors_validator_code(), json.dumps(spec)], text=True, capture_output=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ae.safetensors", result.stderr)

    def test_safetensors_preflight_accepts_flux2_native_vae_layout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "flux2-vae.safetensors"
            _write_fake_safetensors(
                path,
                [
                    "encoder.down.0.block.0.conv1.weight",
                    "decoder.up.0.block.0.conv1.weight",
                    "decoder.post_quant_conv.weight",
                ],
            )
            spec = {"models": [{"role": "vae", "path": str(path), "expected_format": "flux2_vae"}]}
            result = subprocess.run([sys.executable, "-c", _safetensors_validator_code(), json.dumps(spec)], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_safetensors_preflight_accepts_official_flux2_ae_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ae.safetensors"
            _write_fake_safetensors(
                path,
                [
                    "encoder.down.0.block.0.conv1.weight",
                    "decoder.up.0.block.0.conv1.weight",
                    "quant_conv.weight",
                ],
            )
            spec = {"models": [{"role": "vae", "path": str(path), "expected_format": "flux2_ae_or_vae"}]}
            result = subprocess.run([sys.executable, "-c", _safetensors_validator_code(), json.dumps(spec)], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_model_validator_accepts_hf_model_id_but_rejects_missing_paths(self) -> None:
        code = _safetensors_validator_code()
        accepted = {"models": [{"role": "text_encoder", "path": "Qwen/Qwen2.5-VL-7B-Instruct", "expected_format": "hf_model_id_or_path"}]}
        absolute = {"models": [{"role": "text_encoder", "path": "/models/missing.safetensors", "expected_format": "hf_model_id_or_path"}]}
        relative = {"models": [{"role": "text_encoder", "path": "models/missing", "expected_format": "hf_model_id_or_path"}]}

        accepted_result = subprocess.run([sys.executable, "-c", code, json.dumps(accepted)], text=True, capture_output=True, check=False)
        absolute_result = subprocess.run([sys.executable, "-c", code, json.dumps(absolute)], text=True, capture_output=True, check=False)
        relative_result = subprocess.run([sys.executable, "-c", code, json.dumps(relative)], text=True, capture_output=True, check=False)

        self.assertEqual(accepted_result.returncode, 0, accepted_result.stderr)
        self.assertNotEqual(absolute_result.returncode, 0)
        self.assertIn("path does not exist", absolute_result.stderr)
        self.assertNotEqual(relative_result.returncode, 0)
        self.assertIn("path does not exist", relative_result.stderr)

    def test_safetensors_postflight_accepts_musubi_flux2_lora(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.safetensors"
            _write_fake_safetensors(
                path,
                [
                    "lora_unet_double_blocks_0_img_attn_proj.lora_down.weight",
                    "lora_unet_double_blocks_0_img_attn_proj.lora_up.weight",
                    "lora_unet_double_blocks_0_img_attn_proj.alpha",
                ],
                {"ss_network_module": "networks.lora_flux_2", "modelspec.architecture": "Flux.2-klein-4b/lora"},
            )
            spec = {"architecture": "flux2", "lora": {"pattern": str(path), "compatibility": "comfyui"}}
            result = subprocess.run([sys.executable, "-c", _safetensors_validator_code(), json.dumps(spec)], text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)


def _write_fake_safetensors(path: Path, keys: list[str], metadata: dict[str, str] | None = None) -> None:
    header = {key: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]} for key in keys}
    if metadata:
        header["__metadata__"] = metadata
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"\0\0\0\0")


class DockerLifecycleTests(unittest.TestCase):
    def _storage_probe(self, free_gib: int, *, confidence: str = "exact", host_free_gib: int | None = None, backing_kind: str = "native"):
        def fake(paths: dict[str, Path], config: dict[str, object] | None = None) -> dict[str, StorageStatus]:
            result: dict[str, StorageStatus] = {}
            for name, path in paths.items():
                result[name] = StorageStatus(
                    path=str(path),
                    probe=str(path),
                    backing_id="test-backing",
                    backing_kind=backing_kind,
                    linux_free_bytes=free_gib * 1024**3,
                    linux_total_bytes=200 * 1024**3,
                    host_free_bytes=None if host_free_gib is None else host_free_gib * 1024**3,
                    effective_free_bytes=free_gib * 1024**3,
                    confidence=confidence,
                    mount={"available": True},
                    warning=None if confidence != "unknown" else "unknown backing",
                )
            return result

        return fake

    def _run_dir(self, root: Path) -> Path:
        run_dir = root / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "logs" / "events.jsonl").touch()
        (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/r1.json", "container_id": "container-1"}), encoding="utf-8")
        (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "container": {"id": "container-1", "name": "kura-example-r1"}}), encoding="utf-8")
        return run_dir

    def test_private_key_env_names_are_redacted(self) -> None:
        with patch.dict(os.environ, {"SSH_PRIVATE_KEY": "secret-key"}, clear=False):
            safe = _safe_env({"SSH_PRIVATE_KEY": "secret-key", "NORMAL": "hello secret-key"})

        self.assertEqual(safe["SSH_PRIVATE_KEY"], "***")
        self.assertEqual(safe["NORMAL"], "hello ***")

    def test_model_download_safety_rejects_unknown_sizes(self) -> None:
        with self.assertRaisesRegex(ValueError, "model download sizes are unknown"):
            _model_download_safety_preflight({"safety": {}}, {"bytes": 0, "unknown": ["repo:file.safetensors"]})

        _model_download_safety_preflight({"safety": {"allow_large_model_downloads": True}}, {"bytes": 0, "unknown": ["repo:file.safetensors"]})

    def test_hf_size_probe_preserves_http_and_connectivity_failures(self) -> None:
        item = {"repo_id": "repo/model", "filename": "weights.safetensors"}
        with patch("kura.run_commands.plan.urllib.request.urlopen", side_effect=__import__("urllib").error.HTTPError("https://huggingface.co", 401, "unauthorized", {}, None)):
            auth = _hf_file_size_probe(item)
        with patch("kura.run_commands.plan.urllib.request.urlopen", side_effect=__import__("urllib").error.URLError("DNS failed")):
            unreachable = _hf_file_size_probe(item)
        with patch("kura.run_commands.plan.urllib.request.urlopen", side_effect=__import__("urllib").error.HTTPError("https://huggingface.co", 404, "missing", {}, None)):
            missing = _hf_file_size_probe(item)

        self.assertEqual(auth["status"], "auth_error")
        self.assertEqual(auth["detail"], "HTTP 401")
        self.assertEqual(unreachable["status"], "unreachable")
        self.assertIn("DNS failed", unreachable["detail"])
        self.assertEqual(missing["status"], "not_found")

    def test_connectivity_failure_is_not_a_large_download_override(self) -> None:
        estimate = {
            "bytes": 0,
            "unknown": [],
            "probe_failures": [{"artifact": "repo:model.safetensors", "status": "unreachable", "detail": "DNS failed"}],
        }
        with self.assertRaisesRegex(ValueError, "metadata probe failed"):
            _model_download_safety_preflight({"safety": {"allow_large_model_downloads": True}}, estimate, executor="docker")

        _model_download_safety_preflight({"safety": {}}, estimate, executor="runpod")
        local_records = _model_download_preflight_report({}, estimate, executor="docker")
        remote_records = _model_download_preflight_report({}, estimate, executor="runpod")
        self.assertIn(("model-metadata-connectivity", "error"), {(item["check"], item["severity"]) for item in local_records})
        self.assertIn(("model-metadata-connectivity", "warning"), {(item["check"], item["severity"]) for item in remote_records})
        self.assertTrue(any("known portion" in item["fact"] for item in remote_records if item["check"] == "model-downloads"))

        auth_estimate = {
            "bytes": 0,
            "unknown": [],
            "probe_failures": [{"artifact": "repo:model.safetensors", "status": "auth_error", "detail": "HTTP 401"}],
        }
        with self.assertRaisesRegex(ValueError, "auth_error"):
            _model_download_safety_preflight({}, auth_estimate, executor="runpod")
        auth_records = _model_download_preflight_report({}, auth_estimate, executor="runpod")
        self.assertIn(("model-metadata-connectivity", "error"), {(item["check"], item["severity"]) for item in auth_records})

    def test_missing_hf_artifact_is_not_collapsed_into_unknown_size(self) -> None:
        run = {
            "backend": {"name": "musubi-tuner"},
            "backend_overrides": {
                "musubi-tuner": {
                    "architecture": "flux2",
                    "model_bundle": "none",
                    "model_downloads": {"dit": {"repo": "repo/model", "filename": "missing.safetensors"}},
                }
            },
        }
        with patch(
            "kura.run_commands.plan._hf_file_size_probe",
            return_value={"status": "not_found", "size_bytes": None, "detail": "HTTP 404"},
        ):
            estimate = _estimate_musubi_download_bytes(run)

        self.assertEqual(estimate["unknown"], [])
        self.assertEqual(estimate["probe_failures"][0]["status"], "not_found")
        records = _model_download_preflight_report(run, estimate, executor="runpod")
        self.assertIn(("model-metadata-connectivity", "error"), {(item["check"], item["severity"]) for item in records})

    def test_command_is_detached_labeled_and_writes_to_mounted_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            run_dir.mkdir(parents=True)
            command, runtime_env, _ = docker_command(root, run_dir, {"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, "example:image", [], True, "r1")
        self.assertIn("-d", command)
        self.assertIn("--init", command)
        self.assertIn("--stop-timeout", command)
        self.assertNotIn("--rm", command)
        self.assertIn("io.kura.realization_id=r1", command)
        self.assertIn("PYTHONUNBUFFERED=1", command)
        self.assertEqual(runtime_env["HF_HOME"], "/workspace/cache/huggingface")
        self.assertEqual(runtime_env["HF_HUB_CACHE"], "/workspace/cache/huggingface/hub")
        self.assertEqual(runtime_env["KURA_RUN_ID"], "example")
        self.assertIn("HF_HOME=/workspace/cache/huggingface", command)
        self.assertIn("HF_HUB_CACHE=/workspace/cache/huggingface/hub", command)
        self.assertIn(f"{os.getuid()}:{os.getgid()}", command)
        self.assertIn("HOME=/tmp/kura-home", command)
        self.assertIn("KURA_WORKSPACE_PATH_MAPS", runtime_env)
        command_text = "\n".join(command)
        self.assertIn('mkdir -p "$HOME" "$(dirname "$KURA_LOG_PATH")"', command_text)
        self.assertIn('"/workspace/runs/$KURA_RUN_ID/outputs"', command_text)
        self.assertIn('"/workspace/runs/$KURA_RUN_ID/checkpoints"', command_text)
        self.assertIn('exec "$@" >> "$KURA_LOG_PATH" 2>&1', command_text)

    def test_docker_mount_sources_are_resolved_from_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            run_dir.mkdir(parents=True)
            mounts = [{"source": "./cache/huggingface", "target": "/root/.cache/huggingface", "mode": "rw"}]
            command, runtime_env, _ = docker_command(root, run_dir, {"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, "example:image", mounts, True, "r1")
        self.assertIn(f"{root}/cache/huggingface:/workspace/cache/huggingface", command)
        self.assertEqual(
            json.loads(runtime_env["KURA_WORKSPACE_PATH_MAPS"]),
            [
                {"container": "/workspace/cache/huggingface", "workspace": "/workspace/cache/huggingface"},
                {"container": "/workspace", "workspace": "/workspace"},
            ],
        )

    def test_docker_preflight_creates_writable_mount_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mounts = [{"source": "./cache/huggingface", "target": "/root/.cache/huggingface", "mode": "rw"}]
            with patch("kura.executors.docker.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
                docker_preflight(root, mounts)
            self.assertTrue((root / "cache" / "huggingface").is_dir())

    def test_docker_preflight_rejects_low_disk_space(self) -> None:
        class Usage:
            total = 100 * 1024**3
            used = 80 * 1024**3
            free = 20 * 1024**3

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("kura.executors.docker.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")),
                patch("kura.executors.docker.shutil.disk_usage", return_value=Usage()),
            ):
                with self.assertRaisesRegex(ValueError, "requires at least 50 GiB"):
                    docker_preflight(root, [])

    def test_docker_preflight_honors_configured_disk_floor(self) -> None:
        class Usage:
            total = 100 * 1024**3
            used = 80 * 1024**3
            free = 20 * 1024**3

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("kura.executors.docker.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")),
                patch("kura.executors.docker.shutil.disk_usage", return_value=Usage()),
            ):
                payload = docker_preflight(root, [], min_free_gb=10)
        self.assertEqual(payload["disk"]["workspace"]["free_bytes"], 20 * 1024**3)

    def test_local_launch_disk_preflight_uses_configured_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(60)), patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
                with self.assertRaisesRegex(ValueError, "requires at least 100 GiB"):
                    _local_launch_disk_preflight(root, {"type": "train"}, {}, [])
                payload = _local_launch_disk_preflight(root, {"type": "train"}, {"min_free_gb": 50}, [])
        self.assertEqual(payload["required_gib"], 50)

    def test_local_launch_disk_preflight_counts_estimated_hf_downloads(self) -> None:
        run = {
            "type": "train",
            "backend": {"name": "musubi-tuner"},
            "model": {"base": "custom"},
            "backend_overrides": {
                "musubi-tuner": {
                    "architecture": "flux2",
                    "model_downloads": {
                        "dit": {"repo": "example/model", "filename": "dit.safetensors"},
                    },
                }
            },
        }
        mounts = [{"source": "./cache/huggingface", "target": "/root/.cache/huggingface", "mode": "rw"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(60)),
                patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")),
                patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "ok", "size_bytes": 20 * 1024**3}),
            ):
                with self.assertRaisesRegex(ValueError, "requires at least 70 GiB"):
                    _local_launch_disk_preflight(root, run, {"min_free_gb": 50}, mounts)
                payload = _local_launch_disk_preflight(root, run, {"min_free_gb": 40}, mounts)
        self.assertEqual(payload["estimates"]["musubi_downloads"]["bytes"], 20 * 1024**3)
        self.assertEqual(payload["paths"]["hf_cache"]["estimated_write_bytes"], 20 * 1024**3)

    def test_local_launch_disk_preflight_rejects_large_unapproved_model_downloads(self) -> None:
        run = {
            "type": "train",
            "backend": {"name": "musubi-tuner"},
            "model": {"base": "custom"},
            "backend_overrides": {
                "musubi-tuner": {
                    "architecture": "flux2",
                    "model_downloads": {
                        "dit": {"repo": "example/model", "filename": "dit.safetensors"},
                    },
                }
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(100)),
                patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")),
                patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "ok", "size_bytes": 30 * 1024**3}),
            ):
                with self.assertRaisesRegex(ValueError, "allow_large_model_downloads"):
                    _local_launch_disk_preflight(root, run, {"min_free_gb": 50}, [])
                run["safety"] = {"allow_large_model_downloads": True}
                payload = _local_launch_disk_preflight(root, run, {"min_free_gb": 50}, [])
        self.assertEqual(payload["estimates"]["musubi_downloads"]["bytes"], 30 * 1024**3)

    def test_local_launch_disk_preflight_counts_allowed_checkpoint_budget(self) -> None:
        run = {
            "type": "train",
            "backend": {"name": "musubi-tuner"},
            "params": {},
            "backend_overrides": {"musubi-tuner": {"max_train_steps": 3000, "save_every_n_steps": 100}},
            "safety": {"allow_many_checkpoints": True, "checkpoint_estimate_gb": 2},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(100)),
                patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")),
            ):
                with self.assertRaisesRegex(ValueError, "requires at least 110 GiB"):
                    _local_launch_disk_preflight(root, run, {"min_free_gb": 50}, [])
        self.assertEqual(_checkpoint_safety_preflight(run), None)

    def test_local_launch_disk_preflight_sums_estimates_on_shared_backing(self) -> None:
        run = {
            "type": "train",
            "backend": {"name": "musubi-tuner"},
            "params": {},
            "backend_overrides": {
                "musubi-tuner": {
                    "architecture": "flux2",
                    "max_train_steps": 2000,
                    "save_every_n_steps": 100,
                    "model_downloads": {
                        "dit": {"repo": "example/model", "filename": "dit.safetensors"},
                    },
                }
            },
            "safety": {"allow_many_checkpoints": True, "checkpoint_estimate_gb": 2, "allow_large_model_downloads": True},
        }
        mounts = [{"source": "./cache/huggingface", "target": "/root/.cache/huggingface", "mode": "rw"}]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(100)),
                patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")),
                patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "ok", "size_bytes": 40 * 1024**3}),
            ):
                with self.assertRaisesRegex(ValueError, "requires at least 130 GiB"):
                    _local_launch_disk_preflight(root, run, {"min_free_gb": 50}, mounts)
                payload = _local_launch_disk_preflight(root, run, {"min_free_gb": 10}, mounts)
        self.assertEqual(payload["paths"]["workspace"]["estimated_write_bytes"], 40 * 1024**3)
        self.assertEqual(payload["paths"]["hf_cache"]["estimated_write_bytes"], 40 * 1024**3)
        self.assertEqual(payload["paths"]["workspace"]["backing_estimated_write_bytes"], 80 * 1024**3)
        self.assertEqual(payload["paths"]["hf_cache"]["backing_estimated_write_bytes"], 80 * 1024**3)

    def test_local_launch_disk_preflight_honors_run_disk_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(140)), patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
                with self.assertRaisesRegex(ValueError, "requires at least 150 GiB"):
                    _local_launch_disk_preflight(root, {"safety": {"max_run_disk_gb": 150}}, {"min_free_gb": 50}, [])

    def test_local_launch_disk_preflight_rejects_unknown_wsl_backing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(900, confidence="unknown", backing_kind="wsl2_vhdx")), patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
                with self.assertRaisesRegex(ValueError, "unknown physical backing free space"):
                    _local_launch_disk_preflight(root, {"type": "train"}, {}, [])
                payload = _local_launch_disk_preflight(root, {"safety": {"allow_storage_risk": True}}, {}, [])
        self.assertEqual(payload["paths"]["workspace"]["confidence"], "unknown")

    def test_download_disk_guard_rejects_when_free_space_is_too_low(self) -> None:
        class Usage:
            total = 100 * 1024**3
            used = 95 * 1024**3
            free = 5 * 1024**3

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with patch("kura.run_commands.plan.shutil.disk_usage", return_value=Usage()):
                with self.assertRaisesRegex(ValueError, "needs about 10 GiB free"):
                    _ensure_free_bytes(target, 10 * 1024**3, context="test download")

    def test_docker_command_keeps_hf_token_value_out_of_argv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            run_dir.mkdir(parents=True)
            with patch.dict(os.environ, {"HF_TOKEN": "hf-secret"}, clear=False):
                command, runtime_env, _ = docker_command(root, run_dir, {"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, "example:image", [], True, "r1")
        self.assertEqual(runtime_env["HF_TOKEN"], "hf-secret")
        self.assertIn("HF_TOKEN", command)
        self.assertNotIn("HF_TOKEN=hf-secret", command)
        self.assertNotIn("hf-secret", " ".join(command))

    def test_reconcile_known_exit_code_sets_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            result = __import__("subprocess").CompletedProcess([], 0, '{"Running": false, "ExitCode": 0}')
            with patch("kura.executors.docker.subprocess.run", return_value=result):
                status = reconcile_docker(run_dir)
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["exit_code"], 0)
            self.assertTrue(list((run_dir / "realizations").glob("r1.observed-*.json")))

    def test_reconcile_materializes_ai_toolkit_stdout_progress(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "logs" / "stdout.log").write_text(
                "\rexample:  99%|█████████▉| 99/100 [04:07<00:02, 2.49s/it, lr: 1.0e-04 loss: 3.478e-01]\n",
                encoding="utf-8",
            )
            result = __import__("subprocess").CompletedProcess([], 0, '{"Running": false, "ExitCode": 0}')
            with patch("kura.executors.docker.subprocess.run", return_value=result):
                status = reconcile_docker(run_dir)
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["last_step"], 100)
            self.assertEqual(status["total_steps"], 100)

    def test_reconcile_materializes_musubi_stdout_progress_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "outputs").mkdir()
            (run_dir / "outputs" / "example.safetensors").write_text("artifact", encoding="utf-8")
            (run_dir / "logs" / "stdout.log").write_text(
                "\rsteps: 100%|██████████| 5/5 [00:10<00:00,  2.10s/it, avr_loss=0.383]\n",
                encoding="utf-8",
            )
            result = __import__("subprocess").CompletedProcess([], 0, '{"Running": false, "ExitCode": 0}')
            with patch("kura.executors.docker.subprocess.run", return_value=result):
                status = reconcile_docker(run_dir)
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["last_step"], 5)
            self.assertEqual(status["total_steps"], 5)
            self.assertEqual(status["outputs"], ["outputs/example.safetensors"])

    def test_reconcile_missing_container_is_interrupted_not_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            result = __import__("subprocess").CompletedProcess([], 1, "", "Error: No such container")
            with patch("kura.executors.docker.subprocess.run", return_value=result):
                status = reconcile_docker(run_dir)
            self.assertEqual(status["state"], "interrupted")
            self.assertIsNone(status["exit_code"])

    def test_launch_wait_blocks_for_local_docker_and_reconciles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "realizations").mkdir()
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": "example",
                        "type": "train",
                        "backend": {"name": "ai-toolkit"},
                        "backend_overrides": {"ai-toolkit": {"command": {"cwd": "/workspace", "argv": ["python", "-c", "print(1)"], "env": {}}}},
                    }
                ),
                encoding="utf-8",
            )
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({"docker": {"images": {"ai-toolkit": {"local": "local", "remote": "remote", "dockerfile": "Dockerfile", "context": "."}}, "mounts": []}}),
                encoding="utf-8",
            )

            def fake_launch(**_: Any) -> tuple[list[str], str]:
                (run_dir / "status.json").write_text(json.dumps({"state": "running", "container_id": "container-1"}), encoding="utf-8")
                return [], "r1"

            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch("kura.run_commands.launch.launch_docker", side_effect=fake_launch), patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(200)), patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "0B\n", "")), patch("kura.run_commands.launch.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "0\n", "")) as wait, patch("kura.run_commands.launch.reconcile_docker", return_value={"state": "completed", "exit_code": 0}) as reconcile, patch("sys.stdout", new_callable=__import__("io").StringIO):
                    self.assertEqual(launch_run("example", executor="docker", dry_run=False, wait=True), 0)
            finally:
                os.chdir(previous)
            wait.assert_any_call(["docker", "wait", "container-1"], text=True, capture_output=True, check=False)
            reconcile.assert_called_once_with(run_dir)

    def test_launch_docker_uses_image_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "realizations").mkdir()
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": "example",
                        "type": "train",
                        "backend": {"name": "ai-toolkit"},
                        "backend_overrides": {"ai-toolkit": {"command": {"cwd": "/workspace", "argv": ["python", "-c", "print(1)"], "env": {}}}},
                    }
                ),
                encoding="utf-8",
            )
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({"docker": {"min_free_gb": 10, "images": {"ai-toolkit": {"local": "configured-local", "remote": "remote", "dockerfile": "Dockerfile", "context": "."}}, "mounts": []}}),
                encoding="utf-8",
            )
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch("kura.run_commands.launch.launch_docker") as launch, patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(200)), patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, '{"Type":"Build Cache","Size":"0B"}\n', "")):
                    self.assertEqual(launch_run("example", executor="docker", dry_run=False, image="override-image:dev"), 0)
            finally:
                os.chdir(previous)
            self.assertEqual(launch.call_args.kwargs["image"], "override-image:dev")
            self.assertEqual(launch.call_args.kwargs["min_free_gb"], 10)

    def test_launch_rejects_non_default_workspace_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "realizations").mkdir()
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": "example",
                        "type": "train",
                        "backend": {"name": "ai-toolkit"},
                        "backend_overrides": {"ai-toolkit": {"command": {"cwd": "/workspace", "argv": ["python", "-c", "print(1)"], "env": {}}}},
                    }
                ),
                encoding="utf-8",
            )
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({"docker": {"workspace_target": "/ws", "images": {"ai-toolkit": {"local": "local", "remote": "remote", "dockerfile": "Dockerfile", "context": "."}}, "mounts": []}}),
                encoding="utf-8",
            )
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch("kura.run_commands.launch.launch_docker") as launch, patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(200)), patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, '{"Type":"Build Cache","Size":"0B"}\n', "")), patch("sys.stderr", new_callable=__import__("io").StringIO) as stderr:
                    self.assertEqual(launch_run("example", executor="docker", dry_run=True), 1)
            finally:
                os.chdir(previous)
            launch.assert_not_called()
            self.assertIn("docker.workspace_target must be /workspace", stderr.getvalue())


class RunDiscardTests(unittest.TestCase):
    def _make_run(self, root: Path, run_id: str, *, state: str) -> Path:
        run_dir = root / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "run.yaml").write_text(f"id: {run_id}\n", encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
        (run_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")
        return run_dir

    def test_run_discard_defaults_to_dry_run(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            run_dir = self._make_run(root, "draft", state="draft")
            os.chdir(root)
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    status = cmd_run_discard(argparse.Namespace(run_id="draft", yes=False))
            finally:
                os.chdir(previous)
            self.assertEqual(status, 0)
            self.assertTrue(run_dir.exists())
            result = json.loads(stdout.getvalue())
            self.assertTrue(result["dry_run"])
            self.assertEqual(result["target"], "runs/draft")
            self.assertEqual(result["file_count"], 3)

    def test_run_discard_deletes_unlaunched_compiled_run_with_yes(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            run_dir = self._make_run(root, "compiled", state="compiled")
            (run_dir / "realizations").mkdir()
            (run_dir / "outputs").mkdir()
            os.chdir(root)
            stdout = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout):
                    status = cmd_run_discard(argparse.Namespace(run_id="compiled", yes=True))
            finally:
                os.chdir(previous)
            self.assertEqual(status, 0)
            self.assertFalse(run_dir.exists())

    def test_run_discard_rejects_realizations(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            run_dir = self._make_run(root, "running", state="running")
            (run_dir / "realizations").mkdir()
            (run_dir / "realizations" / "r1.json").write_text("{}", encoding="utf-8")
            os.chdir(root)
            stderr = io.StringIO()
            try:
                with contextlib.redirect_stderr(stderr):
                    status = cmd_run_discard(argparse.Namespace(run_id="running", yes=True))
            finally:
                os.chdir(previous)
            self.assertEqual(status, 1)
            self.assertTrue(run_dir.exists())
            self.assertIn("run has execution history (state=running, 1 realizations", stderr.getvalue())
            self.assertIn("use kura run prune for old runs", stderr.getvalue())

    def test_run_discard_rejects_outputs(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            run_dir = self._make_run(root, "compiled", state="compiled")
            (run_dir / "outputs").mkdir()
            (run_dir / "outputs" / "artifact.safetensors").write_text("artifact", encoding="utf-8")
            os.chdir(root)
            stderr = io.StringIO()
            try:
                with contextlib.redirect_stderr(stderr):
                    status = cmd_run_discard(argparse.Namespace(run_id="compiled", yes=True))
            finally:
                os.chdir(previous)
            self.assertEqual(status, 1)
            self.assertTrue(run_dir.exists())
            self.assertIn("1 output entries", stderr.getvalue())

    def test_run_discard_rejects_unsafe_run_id(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            outside = root / "outside"
            outside.mkdir()
            (outside / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            (outside / "run.yaml").write_text("id: outside\n", encoding="utf-8")
            os.chdir(root)
            stderr = io.StringIO()
            try:
                with contextlib.redirect_stderr(stderr):
                    status = cmd_run_discard(argparse.Namespace(run_id="../outside", yes=True))
            finally:
                os.chdir(previous)
            self.assertEqual(status, 1)
            self.assertTrue(outside.exists())
            self.assertIn("run_id must be a safe run directory name", stderr.getvalue())


class RunPruneTests(unittest.TestCase):
    def _make_run(self, root: Path, run_id: str, *, state: str, created: str) -> Path:
        run_dir = root / "runs" / run_id
        (run_dir / "outputs").mkdir(parents=True)
        (run_dir / "downloads").mkdir()
        (run_dir / "outputs" / "artifact.bin").write_text("artifact", encoding="utf-8")
        (run_dir / "downloads" / "remote.bin").write_text("download", encoding="utf-8")
        (run_dir / "run.yaml").write_text(f"id: {run_id}\ncreated: {created}\n", encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": state, "ended": created}), encoding="utf-8")
        return run_dir

    def test_run_prune_defaults_to_dry_run(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (root / "runs").mkdir()
            old = self._make_run(root, "old", state="completed", created="2026-01-01T00:00:00+00:00")
            os.chdir(root)
            try:
                status = cmd_run_prune(argparse.Namespace(keep=0, states="completed", outputs_only=False, yes=False))
                self.assertTrue(old.exists())
            finally:
                os.chdir(previous)
        self.assertEqual(status, 0)

    def test_run_prune_outputs_only_with_yes_preserves_run(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (root / "runs").mkdir()
            old = self._make_run(root, "old", state="completed", created="2026-01-01T00:00:00+00:00")
            os.chdir(root)
            try:
                status = cmd_run_prune(argparse.Namespace(keep=0, states="completed", outputs_only=True, yes=True))
            finally:
                os.chdir(previous)
            self.assertEqual(status, 0)
            self.assertTrue(old.exists())
            self.assertTrue((old / "run.yaml").exists())
            self.assertFalse((old / "outputs").exists())
            self.assertFalse((old / "downloads").exists())

    def test_run_prune_requires_workspace_before_deleting(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs").mkdir()
            old = self._make_run(root, "old", state="completed", created="2026-01-01T00:00:00+00:00")
            os.chdir(root)
            try:
                with self.assertRaisesRegex(ValueError, "workspace.yaml was not found"):
                    cmd_run_prune(argparse.Namespace(keep=0, states="completed", outputs_only=False, yes=True))
            finally:
                os.chdir(previous)
            self.assertTrue(old.exists())
            self.assertTrue((old / "run.yaml").exists())

    def test_run_prune_can_preview_kura_managed_docker_containers(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (root / "runs").mkdir()
            os.chdir(root)
            try:
                with patch(
                    "kura.cli.subprocess.run",
                    return_value=subprocess.CompletedProcess(
                        [],
                        0,
                        '{"ID":"abc","Names":"kura-old","State":"exited","Status":"Exited (0)"}\n'
                        '{"ID":"def","Names":"kura-live","State":"running","Status":"Up 1 minute"}\n',
                        "",
                    ),
                ), patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    status = cmd_run_prune(argparse.Namespace(keep=0, states="completed", outputs_only=False, docker_containers=True, docker_volumes=False, yes=False))
            finally:
                os.chdir(previous)
        self.assertEqual(status, 0)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["docker_actions"]["containers"], [{"id": "abc", "name": "kura-old", "state": "exited", "status": "Exited (0)"}])

    def test_run_prune_deletes_kura_managed_docker_containers_only_with_yes(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (root / "runs").mkdir()
            calls: list[list[str]] = []

            def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
                calls.append(command)
                if command[:2] == ["docker", "ps"]:
                    return subprocess.CompletedProcess(command, 0, '{"ID":"abc","Names":"kura-old","State":"exited","Status":"Exited (0)"}\n', "")
                if command[:2] == ["docker", "rm"]:
                    return subprocess.CompletedProcess(command, 0, "abc\n", "")
                return subprocess.CompletedProcess(command, 1, "", "unexpected")

            os.chdir(root)
            try:
                with patch("kura.cli.subprocess.run", side_effect=fake_run), patch("sys.stdout", new_callable=__import__("io").StringIO):
                    status = cmd_run_prune(argparse.Namespace(keep=0, states="completed", outputs_only=False, docker_containers=True, docker_volumes=False, yes=True))
            finally:
                os.chdir(previous)
        self.assertEqual(status, 0)
        self.assertIn(["docker", "rm", "abc"], calls)

    def test_run_prune_reports_docker_container_delete_failure(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (root / "runs").mkdir()

            def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
                if command[:2] == ["docker", "ps"]:
                    return subprocess.CompletedProcess(command, 0, '{"ID":"abc","Names":"kura-old","State":"exited","Status":"Exited (0)"}\n', "")
                if command[:2] == ["docker", "rm"]:
                    return subprocess.CompletedProcess(command, 1, "", "permission denied")
                return subprocess.CompletedProcess(command, 1, "", "unexpected")

            os.chdir(root)
            try:
                with patch("kura.cli.subprocess.run", side_effect=fake_run), patch("sys.stderr", new_callable=__import__("io").StringIO) as stderr:
                    status = cmd_run_prune(argparse.Namespace(keep=0, states="completed", outputs_only=False, docker_containers=True, docker_volumes=False, yes=True))
            finally:
                os.chdir(previous)
        self.assertEqual(status, 1)
        self.assertIn("cannot prune Docker containers: permission denied", stderr.getvalue())

    def test_run_prune_reports_docker_volume_delete_failure(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (root / "runs").mkdir()

            def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
                if command[:3] == ["docker", "volume", "ls"]:
                    return subprocess.CompletedProcess(command, 0, '{"Name":"kura-cache","Driver":"local"}\n', "")
                if command[:3] == ["docker", "volume", "rm"]:
                    return subprocess.CompletedProcess(command, 1, "", "volume is in use")
                return subprocess.CompletedProcess(command, 1, "", "unexpected")

            os.chdir(root)
            try:
                with patch("kura.cli.subprocess.run", side_effect=fake_run), patch("sys.stderr", new_callable=__import__("io").StringIO) as stderr:
                    status = cmd_run_prune(argparse.Namespace(keep=0, states="completed", outputs_only=False, docker_containers=False, docker_volumes=True, yes=True))
            finally:
                os.chdir(previous)
        self.assertEqual(status, 1)
        self.assertIn("cannot prune Docker volumes: volume is in use", stderr.getvalue())

    def test_run_prune_falls_back_to_docker_for_root_owned_artifacts(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({"docker": {"images": {"ai-toolkit": {"local": "kura/ai-toolkit:test"}}}}),
                encoding="utf-8",
            )
            (root / "runs").mkdir()
            old = self._make_run(root, "old", state="completed", created="2026-01-01T00:00:00+00:00")
            docker_calls: list[list[str]] = []

            def fake_rmtree(path: Path) -> None:
                if path == old:
                    raise PermissionError("root-owned")
                shutil.rmtree(path)

            def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
                docker_calls.append(command)
                return subprocess.CompletedProcess(command, 0, "", "")

            os.chdir(root)
            try:
                with patch("kura.cli.shutil.rmtree", side_effect=fake_rmtree), patch("kura.cli.subprocess.run", side_effect=fake_run), patch("sys.stdout", new_callable=__import__("io").StringIO):
                    status = cmd_run_prune(argparse.Namespace(keep=0, states="completed", outputs_only=False, docker_containers=False, docker_volumes=False, yes=True))
            finally:
                os.chdir(previous)
        self.assertEqual(status, 0)
        docker_run = next(call for call in docker_calls if call[:2] == ["docker", "run"])
        self.assertEqual(docker_run[:7], ["docker", "run", "--rm", "--volume", f"{root.resolve()}:/workspace", "--entrypoint", "sh"])
        self.assertIn("/workspace/runs/old", docker_run)

    def test_run_prune_reports_artifact_delete_failure(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({"docker": {"images": {"ai-toolkit": {"local": "kura/ai-toolkit:test"}}}}),
                encoding="utf-8",
            )
            (root / "runs").mkdir()
            old = self._make_run(root, "old", state="completed", created="2026-01-01T00:00:00+00:00")

            def fake_rmtree(path: Path) -> None:
                if path == old:
                    raise PermissionError("root-owned")
                shutil.rmtree(path)

            def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
                if command[:3] == ["docker", "image", "inspect"]:
                    return subprocess.CompletedProcess(command, 0, "[]", "")
                return subprocess.CompletedProcess(command, 1, "", "docker cleanup failed")

            os.chdir(root)
            try:
                with patch("kura.cli.shutil.rmtree", side_effect=fake_rmtree), patch("kura.cli.subprocess.run", side_effect=fake_run), patch("sys.stderr", new_callable=__import__("io").StringIO) as stderr:
                    status = cmd_run_prune(argparse.Namespace(keep=0, states="completed", outputs_only=False, docker_containers=False, docker_volumes=False, yes=True))
            finally:
                os.chdir(previous)
        self.assertEqual(status, 1)
        self.assertIn("cannot prune run artifacts: docker cleanup failed", stderr.getvalue())


class RunPodLifecycleTests(unittest.TestCase):
    def test_execute_run_uses_compiled_docker_executor_and_waits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("compute: {executor: docker}\n", encoding="utf-8")
            with (
                patch("kura.run_commands.launch._run_path", return_value=run_dir),
                patch("kura.run_commands.launch.launch_run", return_value=0) as launch,
            ):
                self.assertEqual(execute_run("example"), 0)

        launch.assert_called_once_with("example", executor="docker", dry_run=False, image=None, notify_channels=None, wait=True)

    def test_execute_run_uses_compiled_runpod_executor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("compute: {executor: runpod}\n", encoding="utf-8")
            with (
                patch("kura.run_commands.launch._run_path", return_value=run_dir),
                patch("kura.run_commands.launch.run_remote", return_value=0) as remote,
            ):
                self.assertEqual(execute_run("example", max_lease="3h"), 0)

        self.assertEqual(remote.call_args.args, ("example",))
        self.assertEqual(remote.call_args.kwargs["hold_for"], "0")
        self.assertEqual(remote.call_args.kwargs["max_lease"], "3h")

    @staticmethod
    def _config() -> dict[str, object]:
        return {"storage_mode": "upload", "gpu_type_ids": ["NVIDIA A40"]}

    @staticmethod
    def _container_disk_config() -> dict[str, object]:
        return {"storage_mode": "container_disk", "gpu_type_ids": ["NVIDIA A40"]}

    @staticmethod
    def _object_config() -> dict[str, object]:
        return {
            "storage_mode": "object_staging",
            "gpu_type_ids": ["NVIDIA A40"],
            "object_store": {
                "endpoint_url": "https://example.r2.cloudflarestorage.com",
                "bucket": "kura",
                "region": "auto",
                "prefix": "tests",
                "access_key_env": "R2_ACCESS_KEY_ID",
                "secret_key_env": "R2_SECRET_ACCESS_KEY",
            },
        }

    def _run_dir(self, root: Path) -> Path:
        run_dir = root / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "logs" / "events.jsonl").touch()
        (run_dir / "status.json").write_text(json.dumps({"state": "compiled", "started": None, "ended": None, "exit_code": None}), encoding="utf-8")
        return run_dir

    def _stage_upload(self, root: Path, run_dir: Path) -> None:
        (run_dir / "run.yaml").write_text("id: example\n", encoding="utf-8")
        (run_dir / "resolved").mkdir(exist_ok=True)
        (run_dir / "resolved" / "manifest.lock.yaml").write_text("locked: true\n", encoding="utf-8")
        dataset = root / "datasets" / "tiny" / "images"
        dataset.mkdir(parents=True)
        (dataset / "one.txt").write_text("caption\n", encoding="utf-8")
        stage_runpod(workspace=root, run_dir=run_dir, dataset_id="tiny", config=self._config())

    def test_launch_runpod_records_pod_without_secret_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret", "HF_TOKEN": "hf-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request:
                    realization_id = launch_runpod(run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=self._config())
            self.assertIsNotNone(realization_id)
            payload = request.call_args.args[3]
            self.assertNotIn("networkVolumeId", payload)
            self.assertNotIn("volumeMountPath", payload)
            self.assertEqual(payload["volumeInGb"], 0)
            self.assertEqual(payload["dockerStartCmd"][:2], ["sh", "-lc"])
            self.assertNotIn("runpodctl receive", payload["dockerStartCmd"][2])
            self.assertIn("/usr/sbin/sshd", payload["dockerStartCmd"][2])
            self.assertIn("sleep infinity", payload["dockerStartCmd"][2])
            self.assertIn("KURA_UPLOAD_CODE", payload["env"])
            self.assertIn("KURA_DOWNLOAD_CODE", payload["env"])
            self.assertEqual(payload["env"]["HF_HOME"], "/workspace/cache/huggingface")
            self.assertEqual(payload["env"]["HF_HUB_CACHE"], "/workspace/cache/huggingface/hub")
            self.assertEqual(payload["env"]["KURA_WORKSPACE"], "/workspace")
            self.assertEqual(payload["env"]["KURA_RUN_ID"], "example")
            self.assertIn('"$KURA_WORKSPACE/runs/$KURA_RUN_ID/outputs"', payload["dockerStartCmd"][2])
            self.assertIn('"$KURA_WORKSPACE/runs/$KURA_RUN_ID/checkpoints"', payload["dockerStartCmd"][2])
            self.assertNotIn("HF_TOKEN", payload["env"])
            self.assertEqual(payload["cloudType"], "COMMUNITY")
            record = (run_dir / "realizations" / f"{realization_id}.json").read_text(encoding="utf-8")
            self.assertNotIn("api-secret", record)
            self.assertNotIn("hf-secret", record)
            self.assertIn('"cloudTypeCandidates": [', record)
            self.assertIn('"upload_code":', record)
            self.assertIn('"pod_id": "pod-1"', (run_dir / "status.json").read_text(encoding="utf-8"))

    def test_launch_runpod_can_use_template_and_ports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {
                "storage_mode": "upload",
                "gpu_type_ids": ["NVIDIA A40"],
                "template_id": "0fqzfjy6f3",
                "ports": ["8675/http", "22/tcp"],
            }
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request:
                    launch_runpod(run_dir=run_dir, spec={"cwd": "/app/ai-toolkit", "argv": ["python", "run.py"], "env": {}}, image="ostris/aitoolkit:latest", config=config)
            payload = request.call_args.args[3]
            self.assertEqual(payload["templateId"], "0fqzfjy6f3")
            self.assertEqual(payload["ports"], ["8675/http", "22/tcp"])
            self.assertEqual(payload["volumeInGb"], 0)
            self.assertNotIn("imageName", payload)
            self.assertNotIn("dockerStartCmd", payload)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            record = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
            self.assertEqual(record["container_cwd"], "/app/ai-toolkit")

    def test_launch_runpod_session_bootstrap_includes_max_lease_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request:
                    realization_id = launch_runpod_session(run_dir=run_dir, image="registry/comfy:tag", config=self._config(), purpose="comfyui-render")
            self.assertIsNotNone(realization_id)
            payload = request.call_args.args[3]
            self.assertEqual(payload["env"]["KURA_MAX_LEASE_SEC"], "43200")
            self.assertEqual(payload["env"]["HF_HOME"], "/workspace/cache/huggingface")
            self.assertEqual(payload["env"]["HF_HUB_CACHE"], "/workspace/cache/huggingface/hub")
            self.assertIn("runpodctl pod delete", payload["dockerStartCmd"][2])
            self.assertIn("RUNPOD_POD_ID", payload["dockerStartCmd"][2])

    def test_launch_runpod_can_pin_availability_filters(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {
                "storage_mode": "upload",
                "gpu_type_ids": ["NVIDIA A40"],
                "data_center_ids": ["US-GA-1"],
                "data_center_priority": "availability",
                "gpu_type_priority": "availability",
                "country_codes": ["US"],
            }
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request:
                    launch_runpod(run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=config)
            payload = request.call_args.args[3]
            self.assertEqual(payload["dataCenterIds"], ["US-GA-1"])
            self.assertEqual(payload["dataCenterPriority"], "availability")
            self.assertEqual(payload["gpuTypePriority"], "availability")
            self.assertEqual(payload["countryCodes"], ["US"])

    def test_launch_runpod_falls_back_across_cloud_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", side_effect=[ValueError("no community capacity"), {"id": "pod-1", "desiredStatus": "RUNNING"}]) as request:
                    launch_runpod(run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=self._config())
            first = request.call_args_list[0].args[3]
            second = request.call_args_list[1].args[3]
            self.assertEqual(first["cloudType"], "COMMUNITY")
            self.assertEqual(second["cloudType"], "SECURE")
            self.assertNotIn("dataCenterIds", first)
            self.assertNotIn("countryCodes", first)

    def test_launch_runpod_falls_back_across_gpu_types_before_secure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {"storage_mode": "upload", "gpu_type_ids": ["NVIDIA RTX A5000", "NVIDIA A40"], "cloud_types": ["COMMUNITY"], "gpu_type_priority": "custom"}
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", side_effect=[ValueError("no A5000 capacity"), {"id": "pod-1", "desiredStatus": "RUNNING"}]) as request:
                    launch_runpod(run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=config)
            first = request.call_args_list[0].args[3]
            second = request.call_args_list[1].args[3]
            self.assertEqual(first["gpuTypeIds"], ["NVIDIA RTX A5000"])
            self.assertEqual(second["gpuTypeIds"], ["NVIDIA A40"])

    def test_run_launch_uses_explicit_compute_gpu_for_runpod(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "runs" / "example" / "resolved").mkdir(parents=True)
            (root / "runs" / "example" / "logs").mkdir()
            (root / "runs" / "example" / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            (root / "runs" / "example" / "resolved" / "manifest.lock.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": "example",
                        "type": "train",
                        "backend": {"name": "ai-toolkit"},
                        "compute": {"executor": "runpod", "gpu": "NVIDIA A40"},
                        "backend_overrides": {"ai-toolkit": {"command": {"cwd": "/workspace", "argv": ["python", "-c", "print(1)"], "env": {}}}},
                    }
                ),
                encoding="utf-8",
            )
            (root / "workspace.yaml").write_text(
                yaml.safe_dump(
                    {
                        "runpod": {"storage_mode": "upload", "gpu_type_ids": ["NVIDIA RTX A5000", "NVIDIA A40"], "cloud_type": "COMMUNITY"},
                        "docker": {"images": {"ai-toolkit": {"local": "local", "remote": "remote", "dockerfile": "Dockerfile", "context": "."}}},
                    }
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch("kura.run_commands.launch.launch_runpod") as launch:
                    self.assertEqual(cmd_run_launch(argparse.Namespace(run_id="example", executor="runpod", dry_run=False, image=None)), 0)
            finally:
                os.chdir(previous)
            self.assertEqual(launch.call_args.kwargs["config"]["gpu_type_ids"], ["NVIDIA A40"])
            self.assertEqual(launch.call_args.kwargs["config"]["gpu_type_priority"], "custom")

    def test_launch_runpod_rejects_unsupported_udp_ports(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {
                "storage_mode": "upload",
                "gpu_type_ids": ["NVIDIA A40"],
                "template_id": "0fqzfjy6f3",
                "ports": ["8675/http", "22/tcp", "22/udp"],
            }
            with self.assertRaisesRegex(ValueError, "only supports /http and /tcp"):
                launch_runpod(run_dir=run_dir, spec={"cwd": "/app/ai-toolkit", "argv": ["python", "run.py"], "env": {}}, image="ostris/aitoolkit:latest", config=config, dry_run=True)

    def test_launch_runpod_object_staging_is_disabled_until_secret_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            env = {"RUNPOD_API_KEY": "api-secret", "R2_ACCESS_KEY_ID": "r2-access", "R2_SECRET_ACCESS_KEY": "r2-secret"}
            with patch.dict(os.environ, env, clear=False):
                with self.assertRaisesRegex(ValueError, "disabled until object-store credentials"):
                    launch_runpod(run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=self._object_config())

    def test_launch_runpod_records_failed_attempt_without_stale_pod(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "status.json").write_text(json.dumps({"state": "interrupted", "pod_id": "stale-pod", "last_observation": "realizations/old.observed.json"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret", "HF_TOKEN": "hf-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", side_effect=ValueError("RunPod API POST /pods failed (500): echoed api-secret hf-secret")):
                    with self.assertRaisesRegex(ValueError, r"\\*\\*\\*"):
                        launch_runpod(run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=self._container_disk_config())
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "launch_failed")
            self.assertNotIn("pod_id", status)
            self.assertNotIn("last_observation", status)
            self.assertIn("last_realization", status)
            record = (run_dir / status["last_realization"]).read_text(encoding="utf-8")
            self.assertIn('"state": "launch_failed"', record)
            self.assertIn('"gpuTypeIds": [', record)
            self.assertIn('"NVIDIA A40"', record)
            self.assertNotIn("api-secret", record)
            self.assertNotIn("hf-secret", record)

    def test_launch_runpod_rejects_secret_pod_env(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            with self.assertRaisesRegex(ValueError, "pod env must not contain secrets"):
                launch_runpod(run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {"HF_TOKEN": "should-not-enter-pod-env"}}, image="registry/image:tag", config=self._container_disk_config())

    def test_runpod_ssh_secret_injection_keeps_token_out_of_argv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "transfer").mkdir()
            (run_dir / "transfer" / "bundle.tar.gz").write_bytes(b"bundle")
            stage_path = run_dir / "realizations" / "stage.json"
            stage_path.write_text(json.dumps({"storage_mode": "upload", "archive": "transfer/bundle.tar.gz", "archive_name": "bundle.tar.gz"}), encoding="utf-8")
            realization_path = run_dir / "realizations" / "r1.json"
            realization_path.write_text(json.dumps({"executor": "runpod", "request": {"env": {"KURA_WORKSPACE": "/workspace"}}, "container_cwd": "/opt/tool", "backend_command": ["python", "train.py"]}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"pod_id": "pod-1", "last_stage": "realizations/stage.json", "last_realization": "realizations/r1.json"}), encoding="utf-8")
            calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

            def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                calls.append((args, kwargs))
                command_text = " ".join(map(str, args[0])) if isinstance(args[0], list) else str(args[0])
                if "nohup sh" in command_text:
                    return subprocess.CompletedProcess(args[0], 0, "1234\n", "")
                if "remote-exit-*.json" in command_text:
                    return subprocess.CompletedProcess(args[0], 0, json.dumps({"event": "remote_exit", "exit_code": 0}), "")
                if "__KURA_LOG_SIZE__" in command_text:
                    return subprocess.CompletedProcess(args[0], 0, b"\n__KURA_LOG_SIZE__:0\n", b"")
                return subprocess.CompletedProcess(args[0], 0, "", "")

            with patch.dict(os.environ, {"HF_TOKEN": "hf-secret"}, clear=False):
                with patch("kura.run_commands.runpod_ssh._runpod_ssh_details", return_value={"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}):
                    with patch("kura.cli.subprocess.run", side_effect=fake_run):
                        self.assertEqual(_runpod_run_over_ssh(run_dir, ssh_timeout_sec=1, job_timeout_sec=1), 0)

            argv_text = "\n".join(" ".join(map(str, call[0][0])) if isinstance(call[0][0], list) else str(call[0][0]) for call in calls)
            self.assertNotIn("hf-secret", argv_text)
            self.assertIn("sleep 43200", argv_text)
            self.assertIn("KURA_LEASE_LOG_PATH=/workspace/runs/example/logs/stdout.log", argv_text)
            self.assertIn("RUNPOD_POD_ID=pod-1", argv_text)
            self.assertIn("runpodctl pod delete", argv_text)
            input_text = "\n".join(str(call[1].get("input") or "") for call in calls)
            self.assertIn('export HF_HOME="$KURA_WORKSPACE/cache/huggingface"', input_text)
            self.assertIn('export HF_HUB_CACHE="$HF_HOME/hub"', input_text)
            self.assertIn('"$KURA_WORKSPACE/runs/$KURA_RUN_ID/outputs"', input_text)
            self.assertIn('"$KURA_WORKSPACE/runs/$KURA_RUN_ID/checkpoints"', input_text)
            self.assertIn('mkdir -p "$HF_HUB_CACHE" "$KURA_WORKSPACE/cache/models"', input_text)
            self.assertIn('HF_HOME must be under KURA_WORKSPACE before remote job start', input_text)
            self.assertIn('collect_runtime_diagnostics before_backend', input_text)
            self.assertIn('collect_runtime_diagnostics after_backend', input_text)
            self.assertIn('/sys/fs/cgroup/memory.events', input_text)
            self.assertIn('/sys/fs/cgroup/memory/memory.oom_control', input_text)
            self.assertIn('/sys/fs/cgroup/memory/memory.limit_in_bytes', input_text)
            self.assertIn('"cgroup_oom_kill_delta"', input_text)
            self.assertTrue(any(call[1].get("input") and "hf-secret" in str(call[1]["input"]) for call in calls))

    def test_runpod_remote_job_diagnostics_script_has_valid_shell_syntax(self) -> None:
        script = _runpod_remote_job_script(
            workspace="/workspace",
            run_id="example",
            remote_secret_path="/tmp/kura-secrets/example.env",
            archive_name="bundle.tar.gz",
            remote_archive="/workspace/bundle.tar.gz",
            cwd="/opt/tool",
            command="true",
        )
        result = subprocess.run(["sh", "-n"], input=script, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runpod_ssh_can_disable_pod_side_max_lease_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "transfer").mkdir()
            (run_dir / "transfer" / "bundle.tar.gz").write_bytes(b"bundle")
            (run_dir / "realizations" / "stage.json").write_text(json.dumps({"storage_mode": "upload", "archive": "transfer/bundle.tar.gz", "archive_name": "bundle.tar.gz"}), encoding="utf-8")
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"executor": "runpod", "request": {"env": {"KURA_WORKSPACE": "/workspace"}}, "container_cwd": "/opt/tool", "backend_command": ["python", "train.py"]}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"pod_id": "pod-1", "last_stage": "realizations/stage.json", "last_realization": "realizations/r1.json"}), encoding="utf-8")
            calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

            def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                calls.append((args, kwargs))
                command_text = " ".join(map(str, args[0])) if isinstance(args[0], list) else str(args[0])
                if "nohup sh" in command_text:
                    return subprocess.CompletedProcess(args[0], 0, "1234\n", "")
                if "remote-exit-*.json" in command_text:
                    return subprocess.CompletedProcess(args[0], 0, json.dumps({"event": "remote_exit", "exit_code": 0}), "")
                if "__KURA_LOG_SIZE__" in command_text:
                    return subprocess.CompletedProcess(args[0], 0, b"\n__KURA_LOG_SIZE__:0\n", b"")
                return subprocess.CompletedProcess(args[0], 0, "", "")

            with patch("kura.run_commands.runpod_ssh._runpod_ssh_details", return_value={"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}):
                with patch("kura.cli.subprocess.run", side_effect=fake_run):
                    self.assertEqual(_runpod_run_over_ssh(run_dir, ssh_timeout_sec=1, job_timeout_sec=1, max_lease_sec=0), 0)

            argv_text = "\n".join(" ".join(map(str, call[0][0])) if isinstance(call[0][0], list) else str(call[0][0]) for call in calls)
            self.assertNotIn("runpodctl pod delete", argv_text)

    def test_runpod_render_session_starts_lease_guard_over_ssh(self) -> None:
        details = {"pod_id": "pod-1", "ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}
        with patch("kura.run_commands.runpod_ssh.subprocess.run", return_value=subprocess.CompletedProcess(["ssh"], 0, "", "")) as run:
            _start_runpod_session_lease_guard(details, workspace="/workspace", run_id="render-1", max_lease_sec=60)
        command = run.call_args.args[0]
        command_text = "\n".join(map(str, command))
        self.assertIn("sleep 60", command_text)
        self.assertIn("runpodctl pod delete", command_text)
        self.assertIn("pod-1", command_text)
        self.assertIn("/workspace/runs/render-1/logs/stdout.log", command_text)
        self.assertEqual(run.call_args.kwargs["timeout"], 60)

    def test_runpod_render_session_lease_guard_timeout_surfaces_as_value_error(self) -> None:
        details = {"pod_id": "pod-1", "ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}
        with patch("kura.run_commands.runpod_ssh.subprocess.run", side_effect=subprocess.TimeoutExpired(["ssh"], 60)):
            with self.assertRaisesRegex(ValueError, "remote lease guard setup timed out"):
                _start_runpod_session_lease_guard(details, workspace="/workspace", run_id="render-1", max_lease_sec=60)

    def test_runpod_comfyui_lease_guard_starts_before_model_prepare(self) -> None:
        details = {"pod_id": "pod-1", "ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}
        with patch("kura.run_commands.runpod_ssh.subprocess.run", return_value=subprocess.CompletedProcess(["ssh"], 0, "", "")) as run:
            _start_runpod_comfyui(
                details,
                workspace="/workspace",
                run_id="render-1",
                workflow_remote="/workspace/runs/render-1/resolved/workflow_used.json",
                registry_remote="/workspace/runs/render-1/resolved/comfyui_model_registry.json",
                lora_remote_name=None,
                lora_remote_path=None,
                max_lease_sec=60,
            )
        script = run.call_args.args[0][-1]
        self.assertIn("trap cleanup EXIT", script)
        self.assertIn('rm -f "$secret_file"', script)
        self.assertLess(script.index("runpodctl pod delete"), script.index("kura_comfy_prepare.py"))
        self.assertLess(script.index("kura_comfy_prepare.py"), script.index("nohup python main.py"))

    def test_runpod_comfyui_start_failure_reports_ssh_error(self) -> None:
        details = {"pod_id": "pod-1", "ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}
        with patch("kura.run_commands.render_runpod.subprocess.run", return_value=subprocess.CompletedProcess(["ssh"], 2, "", "remote broke")):
            with self.assertRaisesRegex(ValueError, "remote ComfyUI start failed"):
                _start_runpod_comfyui(
                    details,
                    workspace="/workspace",
                    run_id="render-1",
                    workflow_remote="/workspace/runs/render-1/resolved/workflow_used.json",
                    registry_remote="/workspace/runs/render-1/resolved/comfyui_model_registry.json",
                    lora_remote_name=None,
                    lora_remote_path=None,
                    max_lease_sec=0,
                )

    def test_runpod_scp_is_non_interactive_and_bounded(self) -> None:
        details = {"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "workflow.json"
            source.write_text("{}", encoding="utf-8")
            with patch("kura.run_commands.runpod_ssh.subprocess.run", return_value=subprocess.CompletedProcess(["scp"], 0, "", "")) as run:
                _scp_to_runpod(details, source, "/workspace/workflow.json")
        command = run.call_args.args[0]
        self.assertIn("BatchMode=yes", command)
        self.assertIn("ConnectTimeout=20", command)
        self.assertEqual(run.call_args.kwargs["timeout"], 600)

    def test_runpod_scp_timeout_surfaces_as_value_error(self) -> None:
        details = {"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "workflow.json"
            source.write_text("{}", encoding="utf-8")
            with patch("kura.run_commands.runpod_ssh.subprocess.run", side_effect=subprocess.TimeoutExpired(["scp"], 600)):
                with self.assertRaisesRegex(ValueError, "scp upload timed out"):
                    _scp_to_runpod(details, source, "/workspace/workflow.json")

    def test_runpod_ssh_details_retries_after_pod_get_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"pod_id": "pod-1"}), encoding="utf-8")
            pod = {"ssh": {"ip": "127.0.0.1", "port": 22, "ssh_key": {"path": "/tmp/key"}}}
            with patch(
                "kura.run_commands.runpod_ssh.subprocess.run",
                side_effect=[
                    subprocess.TimeoutExpired(["runpodctl", "pod", "get", "pod-1"], 1),
                    subprocess.CompletedProcess(["runpodctl"], 0, json.dumps(pod), ""),
                ],
            ) as run:
                details = _runpod_ssh_details(run_dir, timeout_sec=5, interval_sec=0)

            self.assertEqual(details["pod_id"], "pod-1")
            self.assertEqual(run.call_args.kwargs["timeout"], 1)

    def test_reconcile_runpod_exited_is_unknown_without_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/r1.json"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "EXITED"}):
                    status = reconcile_runpod(run_dir, self._config())
            self.assertEqual(status["state"], "unknown")
            self.assertIsNone(status["exit_code"])

    def test_cli_reconcile_runpod_syncs_remote_log_without_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {api_key_env: RUNPOD_API_KEY, gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
            run_dir = self._run_dir(root)
            (run_dir / "run.yaml").write_text("id: example\n", encoding="utf-8")
            realization = {
                "id": "r1",
                "executor": "runpod",
                "pod": {"id": "pod-1"},
                "request": {"env": {"KURA_WORKSPACE": "/workspace"}},
            }
            (run_dir / "realizations" / "r1.json").write_text(json.dumps(realization), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/r1.json", "pod_id": "pod-1"}), encoding="utf-8")
            remote_stdout = (
                b"steps:  10%|#         | 3/30 [00:06<00:54,  2.00s/it, avr_loss=0.234]\n"
                b"\n__KURA_LOG_SIZE__:77\n"
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch.dict(os.environ, {"RUNPOD_API_KEY": ""}, clear=False):
                    with patch("kura.run_commands.runpod_ssh._runpod_ssh_details", return_value={"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}):
                        with patch("kura.cli.subprocess.run", return_value=subprocess.CompletedProcess([], 0, remote_stdout, b"")):
                            self.assertEqual(cmd_run_reconcile(argparse.Namespace(run_id="example")), 0)
            finally:
                os.chdir(previous)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["last_step"], 3)
            self.assertEqual(status["total_steps"], 30)

    def test_run_remote_does_not_stop_pod_when_download_is_unconfirmed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {api_key_env: RUNPOD_API_KEY, gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
            (root / "runs" / "example").mkdir(parents=True)
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.launch.stage_run", return_value=0), \
                     patch("kura.run_commands.launch.launch_run", return_value=0), \
                     patch("kura.run_commands.launch._runpod_run_over_ssh", return_value=0), \
                     patch("kura.run_commands.launch.download_with_retries", return_value=1), \
                     patch("kura.run_commands.launch.stop_run") as stop:
                    code = cmd_run_remote(argparse.Namespace(run_id="example", upload_timeout=1, job_timeout=1, download_attempts=1, download_interval=1))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            stop.assert_not_called()

    def test_run_remote_notifies_loudly_on_controller_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {api_key_env: RUNPOD_API_KEY, gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
            (root / "runs" / "example").mkdir(parents=True)
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.launch.stage_run", return_value=0), \
                     patch("kura.run_commands.launch.launch_run", return_value=0), \
                     patch("kura.run_commands.launch._runpod_run_over_ssh", side_effect=subprocess.TimeoutExpired(["runpod-remote-job", "example"], 1)), \
                     patch("kura.run_commands.launch._notify") as notify, \
                     patch("kura.run_commands.launch.stop_run") as stop:
                    code = cmd_run_remote(argparse.Namespace(run_id="example", upload_timeout=1, job_timeout=1, download_attempts=1, download_interval=1, notify="ntfy", hold_for="30m", notify_repeat_interval="10m"))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            stop.assert_not_called()
            notify.assert_called_once()
            self.assertIn("controller failed", notify.call_args.kwargs["subject"])
            self.assertIn("may still be running and billing", notify.call_args.kwargs["body"])
            self.assertIn("kura run stop example", notify.call_args.kwargs["body"])

    def test_stop_run_without_realization_reports_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                stderr = io.StringIO()
                with patch("sys.stderr", stderr):
                    code = stop_run("example")
            finally:
                os.chdir(previous)

            self.assertEqual(code, 1)
            self.assertIn("run has no realization to stop", stderr.getvalue())

    def test_run_remote_stops_pod_immediately_when_hold_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {api_key_env: RUNPOD_API_KEY, gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
            (root / "runs" / "example").mkdir(parents=True)
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.launch.stage_run", return_value=0), \
                     patch("kura.run_commands.launch.launch_run", return_value=0), \
                     patch("kura.run_commands.launch._runpod_run_over_ssh", return_value=0), \
                     patch("kura.run_commands.launch.download_with_retries", return_value=0), \
                     patch("kura.run_commands.launch.stop_run") as stop:
                    code = cmd_run_remote(argparse.Namespace(run_id="example", upload_timeout=1, job_timeout=1, download_attempts=1, download_interval=1, hold_for="0"))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            stop.assert_called_once()

    def test_run_remote_defaults_to_bounded_review_hold_after_confirmed_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {api_key_env: RUNPOD_API_KEY, gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
            (root / "runs" / "example").mkdir(parents=True)
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.launch.stage_run", return_value=0), \
                     patch("kura.run_commands.launch.launch_run", return_value=0), \
                     patch("kura.run_commands.launch._runpod_run_over_ssh", return_value=0) as remote_run, \
                     patch("kura.run_commands.launch.download_with_retries", return_value=0), \
                     patch("kura.run_commands.launch._sleep_with_completion_reminders") as hold, \
                     patch("kura.run_commands.launch.stop_run") as stop:
                    code = cmd_run_remote(argparse.Namespace(run_id="example", upload_timeout=1, job_timeout=1, download_attempts=1, download_interval=1))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            hold.assert_called_once()
            self.assertEqual(hold.call_args.kwargs["delay_sec"], 1800)
            self.assertEqual(remote_run.call_args.kwargs["max_lease_sec"], 43200)
            stop.assert_called_once()

    def test_run_remote_repeats_completion_notification_during_review_hold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {api_key_env: RUNPOD_API_KEY, gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
            (root / "runs" / "example").mkdir(parents=True)
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.launch.stage_run", return_value=0), \
                     patch("kura.run_commands.launch.launch_run", return_value=0), \
                     patch("kura.run_commands.launch._runpod_run_over_ssh", return_value=0), \
                     patch("kura.run_commands.launch.download_with_retries", return_value=0), \
                     patch("kura.notifications.time.sleep") as sleep, \
                     patch("kura.run_commands.launch._notify") as initial_notify, \
                     patch("kura.notifications.notify") as reminder_notify, \
                     patch("kura.run_commands.launch.stop_run") as stop:
                    code = cmd_run_remote(argparse.Namespace(run_id="example", upload_timeout=1, job_timeout=1, download_attempts=1, download_interval=1, hold_for="20m", notify="ntfy", notify_repeat_interval="10m"))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [600, 600])
            initial_notify.assert_called_once()
            reminder_notify.assert_called_once()
            self.assertIn("completed", initial_notify.call_args.kwargs["subject"])
            self.assertIn("reminder", reminder_notify.call_args.kwargs["subject"])
            stop.assert_called_once()

    def test_run_remote_stops_pod_when_review_hold_is_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {api_key_env: RUNPOD_API_KEY, gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
            (root / "runs" / "example").mkdir(parents=True)
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.launch.stage_run", return_value=0), \
                     patch("kura.run_commands.launch.launch_run", return_value=0), \
                     patch("kura.run_commands.launch._runpod_run_over_ssh", return_value=0), \
                     patch("kura.run_commands.launch.download_with_retries", return_value=0), \
                     patch("kura.run_commands.launch._sleep_with_completion_reminders", side_effect=KeyboardInterrupt), \
                     patch("kura.run_commands.launch.stop_run") as stop:
                    code = cmd_run_remote(argparse.Namespace(run_id="example", upload_timeout=1, job_timeout=1, download_attempts=1, download_interval=1, hold_for="20m", notify="ntfy", notify_repeat_interval="10m"))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            stop.assert_called_once()

    def test_run_download_rejects_snapshot_without_remote_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "downloads" / "example" / "realizations").mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1"}), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                code = cmd_run_download(argparse.Namespace(run_id="example", force=False))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)

    def test_run_download_materializes_outputs_at_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            output_dir = run_dir / "downloads" / "example" / "outputs"
            realization_dir = run_dir / "downloads" / "example" / "realizations"
            output_dir.mkdir(parents=True)
            realization_dir.mkdir(parents=True)
            (output_dir / "artifact.safetensors").write_text("artifact", encoding="utf-8")
            (realization_dir / "remote-exit-20260101.json").write_text(json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "exit_code": 0}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1"}), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                code = cmd_run_download(argparse.Namespace(run_id="example", force=False))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            self.assertEqual((run_dir / "outputs" / "artifact.safetensors").read_text(encoding="utf-8"), "artifact")
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["outputs"], ["outputs/artifact.safetensors"])
            self.assertEqual(status["downloaded_run"], "downloads/example")

    def test_run_download_rejects_fresh_archive_without_remote_exit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1"}), encoding="utf-8")
            real_run = subprocess.run

            def fake_run(command: list[str], *args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                if command[:3] == ["runpodctl", "pod", "get"]:
                    pod = {"ssh": {"ip": "127.0.0.1", "port": 22, "ssh_key": {"path": "/tmp/key"}}}
                    return subprocess.CompletedProcess(command, 0, json.dumps(pod), "")
                if command and command[0] == "ssh":
                    return subprocess.CompletedProcess(command, 0, "", "")
                if command and command[0] == "scp":
                    archive_path = Path(command[-1])
                    source = root / "remote-snapshot"
                    (source / "example" / "realizations").mkdir(parents=True)
                    (source / "example" / "run.yaml").write_text("id: example\n", encoding="utf-8")
                    with tarfile.open(archive_path, "w:gz") as archive:
                        archive.add(source / "example", arcname="example")
                    return subprocess.CompletedProcess(command, 0, "", "")
                if command and command[0] == "tar":
                    return real_run(command, *args, **kwargs)
                return subprocess.CompletedProcess(command, 0, "", "")

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.cli.shutil.which", return_value="/usr/bin/runpodctl"), \
                     patch("kura.cli.subprocess.run", side_effect=fake_run):
                    code = cmd_run_download(argparse.Namespace(run_id="example", force=True))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)

    def test_doctor_runpod_fails_when_network_volumes_remain(self) -> None:
        class FakeResponse:
            def __init__(self, payload: object) -> None:
                self.payload = payload

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request: object, timeout: int = 20) -> FakeResponse:
            url = getattr(request, "full_url", "")
            if str(url).endswith("/networkvolumes"):
                return FakeResponse([{"id": "volume-1"}])
            return FakeResponse([])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {api_key_env: RUNPOD_API_KEY}\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                completed = subprocess.CompletedProcess([], 0, "ok", "")
                with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                     patch("kura.doctor.shutil.which", return_value="/usr/bin/runpodctl"), \
                     patch("kura.doctor.subprocess.run", return_value=completed), \
                     patch("kura.doctor.urllib.request.urlopen", side_effect=fake_urlopen):
                    code = cmd_doctor_runpod(argparse.Namespace())
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)

    def test_doctor_runpod_fails_when_network_volume_check_is_unknown(self) -> None:
        class FakeResponse:
            def __init__(self, payload: object) -> None:
                self.payload = payload

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(self.payload).encode("utf-8")

        def fake_urlopen(request: object, timeout: int = 20) -> FakeResponse:
            url = getattr(request, "full_url", "")
            if str(url).endswith("/networkvolumes"):
                raise OSError("network volume endpoint unavailable")
            return FakeResponse([])

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {api_key_env: RUNPOD_API_KEY}\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                completed = subprocess.CompletedProcess([], 0, "ok", "")
                with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                     patch("kura.doctor.shutil.which", return_value="/usr/bin/runpodctl"), \
                     patch("kura.doctor.subprocess.run", return_value=completed), \
                     patch("kura.doctor.urllib.request.urlopen", side_effect=fake_urlopen), \
                     patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_runpod(argparse.Namespace())
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertIsNone(payload["checks"]["network_volumes_empty"])
            self.assertIn("network_volumes_error", payload["diagnostics"])

    def test_doctor_comfyui_handles_unreachable_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("comfyui: {endpoint: http://127.0.0.1:8188}\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.urllib.request.urlopen", side_effect=OSError("connection refused")):
                    code = cmd_doctor_comfyui(argparse.Namespace())
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)

    def test_doctor_comfyui_rejects_non_http_endpoint_scheme(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("comfyui: {endpoint: file:///tmp/comfyui.sock}\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.urllib.request.urlopen") as urlopen, patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_comfyui(argparse.Namespace())
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            urlopen.assert_not_called()
            self.assertIn("unsupported comfyui.endpoint scheme", payload["diagnostics"]["object_info_error"])

    def test_doctor_comfyui_redacts_endpoint_userinfo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("comfyui: {endpoint: 'http://user:pa55@example.invalid:8188?debug=abc123#frag123'}\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.urllib.request.urlopen", side_effect=OSError("connection refused")), patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_comfyui(argparse.Namespace())
            finally:
                os.chdir(previous)
            payload_text = stdout.getvalue()
            payload = json.loads(payload_text)
            self.assertEqual(code, 1)
            self.assertNotIn("pa55", payload_text)
            self.assertNotIn("abc123", payload_text)
            self.assertNotIn("frag123", payload_text)
            self.assertEqual(payload["diagnostics"]["endpoint"], "http://***@example.invalid:8188")

    def test_doctor_comfyui_reports_lora_loader_count_and_stage_dir(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({
                    "LoraLoader": {
                        "input": {
                            "required": {
                                "lora_name": [["one.safetensors", "two.safetensors"], {}],
                            },
                        },
                    },
                }).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_dir = root / "comfyui" / "models" / "loras"
            (lora_dir / "Kura_tmp").mkdir(parents=True)
            (root / "workspace.yaml").write_text(
                f"comfyui:\n  endpoint: http://127.0.0.1:8188\n  lora_dir: {lora_dir}\n  lora_stage_subdir: Kura_tmp\n",
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.urllib.request.urlopen", return_value=FakeResponse()):
                    code = cmd_doctor_comfyui(argparse.Namespace())
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)

    def test_doctor_comfyui_reports_kura_stage_files(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({"LoraLoader": {"input": {"required": {"lora_name": [[], {}]}}}}).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_dir = root / "comfyui" / "models" / "loras"
            stage_dir = lora_dir / "Kura_tmp"
            stage_dir.mkdir(parents=True)
            (stage_dir / "render-1-example.safetensors").write_bytes(b"leftover")
            (root / "workspace.yaml").write_text(
                f"comfyui:\n  endpoint: http://127.0.0.1:8188\n  lora_dir: {lora_dir}\n  lora_stage_subdir: Kura_tmp\n",
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.urllib.request.urlopen", return_value=FakeResponse()), patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_comfyui(argparse.Namespace(endpoint=None, probe_stage=False))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["diagnostics"]["kura_stage_file_count"], 1)
            self.assertEqual(payload["diagnostics"]["kura_stage_file_samples"], ["render-1-example.safetensors"])

    def test_doctor_comfyui_endpoint_override_is_measured(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({"LoraLoader": {"input": {"required": {"lora_name": [[], {}]}}}}).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("comfyui: {endpoint: http://127.0.0.1:8188}\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.urllib.request.urlopen", return_value=FakeResponse()) as urlopen, patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_comfyui(argparse.Namespace(endpoint="http://127.0.0.1:8190/"))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertEqual(payload["diagnostics"]["endpoint"], "http://127.0.0.1:8190")
            self.assertEqual(urlopen.call_args.args[0], "http://127.0.0.1:8190/object_info")

    def test_doctor_comfyui_probe_stage_checks_lora_visibility(self) -> None:
        class FakeResponse:
            def __init__(self, payload: dict[str, Any]) -> None:
                self.payload = payload

            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(self.payload).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_dir = root / "models" / "loras"
            stage_dir = lora_dir / "Kura_tmp"
            stage_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text(
                f"comfyui:\n  endpoint: http://127.0.0.1:8190\n  lora_dir: {lora_dir}\n  lora_stage_subdir: Kura_tmp\n",
                encoding="utf-8",
            )

            def fake_urlopen(url: str, timeout: int = 5) -> FakeResponse:
                staged = [f"Kura_tmp/{path.name}" for path in stage_dir.glob("kura-doctor-probe-*.safetensors")]
                return FakeResponse({"LoraLoader": {"input": {"required": {"lora_name": [staged, {}]}}}})

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.urllib.request.urlopen", side_effect=fake_urlopen), patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_comfyui(argparse.Namespace(endpoint=None, probe_stage=True))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertTrue(payload["checks"]["lora_stage_visible"])
            self.assertFalse(list(stage_dir.glob("kura-doctor-probe-*.safetensors")))

    def test_doctor_comfyui_probe_stage_reports_invisible_lora_dir(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({"LoraLoader": {"input": {"required": {"lora_name": [[], {}]}}}}).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_dir = root / "models" / "loras"
            (lora_dir / "Kura_tmp").mkdir(parents=True)
            (root / "workspace.yaml").write_text(
                f"comfyui:\n  endpoint: http://127.0.0.1:8190\n  lora_dir: {lora_dir}\n  lora_stage_subdir: Kura_tmp\n",
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.urllib.request.urlopen", return_value=FakeResponse()), patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_comfyui(argparse.Namespace(endpoint=None, probe_stage=True))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertFalse(payload["checks"]["lora_stage_visible"])
            self.assertIn("not visible", payload["diagnosis"])

    def test_doctor_comfyui_probe_stage_prefers_unreachable_endpoint_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lora_dir = root / "models" / "loras"
            (lora_dir / "Kura_tmp").mkdir(parents=True)
            (root / "workspace.yaml").write_text(
                f"comfyui:\n  endpoint: http://127.0.0.1:8190\n  lora_dir: {lora_dir}\n  lora_stage_subdir: Kura_tmp\n",
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.doctor.urllib.request.urlopen", side_effect=OSError("connection refused")),
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                ):
                    code = cmd_doctor_comfyui(argparse.Namespace(endpoint=None, probe_stage=True))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertFalse(payload["checks"]["endpoint_reachable"])
            self.assertFalse(payload["checks"]["lora_stage_visible"])
            self.assertIn("not ready", payload["diagnosis"])
            self.assertNotIn("not visible", payload["diagnosis"])

    def test_stage_runpod_object_staging_is_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "realizations").mkdir()
            (run_dir / "logs").mkdir()
            (run_dir / "logs" / "events.jsonl").touch()
            (run_dir / "run.yaml").write_text("id: example\n", encoding="utf-8")
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("locked: true\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            dataset = root / "datasets" / "tiny" / "images"
            dataset.mkdir(parents=True)
            (dataset / "one.txt").write_text("caption\n", encoding="utf-8")
            with patch.dict(os.environ, {"R2_ACCESS_KEY_ID": "r2-access", "R2_SECRET_ACCESS_KEY": "r2-secret"}, clear=False):
                with self.assertRaisesRegex(ValueError, "object_staging is experimental and disabled"):
                    stage_runpod(workspace=root, run_dir=run_dir, dataset_id="tiny", config=self._object_config())

    def test_stage_runpod_object_staging_fails_before_source_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "missing"
            with self.assertRaisesRegex(ValueError, "object_staging is experimental and disabled"):
                stage_runpod(workspace=root, run_dir=run_dir, dataset_id="missing-dataset", config=self._object_config())

    def test_stop_runpod_terminates_disposable_pod(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", return_value={}) as request:
                    status = stop_runpod(run_dir, self._config())
            self.assertEqual(status["state"], "interrupted")
            self.assertEqual(request.call_args.args[:3], ("DELETE", "/pods/pod-1", "api-secret"))

    def test_stop_runpod_preserves_completed_status_after_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "status.json").write_text(json.dumps({"state": "completed", "exit_code": 0, "pod_id": "pod-1"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", return_value={}):
                    status = stop_runpod(run_dir, self._config())
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["exit_code"], 0)
            self.assertIn("pod_stopped_at", status)
