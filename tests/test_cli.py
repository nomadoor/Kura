"""Small regression tests for workspace initialization."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
from copy import deepcopy
import hashlib
import io
import importlib.util
import os
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
import json
from pathlib import Path
from typing import Any
from types import SimpleNamespace
from unittest.mock import Mock, patch

import yaml

from kura import __version__
from kura.backends import BACKENDS, MUSUBI_ADAPTER_SCRIPTS, _safetensors_validator_code, command_ai_toolkit, command_musubi_tuner, compile_ai_toolkit, compile_musubi_tuner
from kura.backends.ai_toolkit import AI_TOOLKIT_VIDEO_SUFFIXES, project_ai_toolkit_dataset
from kura.backends.musubi_datasets import MUSUBI_AUDIO_SUFFIXES, MUSUBI_IMAGE_SUFFIXES, MUSUBI_VIDEO_SUFFIXES, project_musubi_dataset
from kura.backends.musubi_command import display_musubi_tuner
from kura.backends.musubi_models import requirements_musubi
from kura.cli import _docker_cleanup_image, _notification_channels, _notify, _parse_duration_seconds, _runpod_run_over_ssh, _runpod_secret_env_payload, _select_remote_outputs, _sync_runpod_remote_stdout, _workspace, cmd_cleanup, cmd_dataset_validate, cmd_doctor_comfyui, cmd_doctor_disk, cmd_doctor_docker, cmd_doctor_musubi, cmd_doctor_runpod, cmd_doctor_sd_scripts, cmd_doctor_workspace, cmd_fix_links, cmd_fix_permissions, cmd_image_build, cmd_init, cmd_monitor, cmd_render_launch, cmd_render_new, cmd_run_compile, cmd_run_discard, cmd_run_download, cmd_run_new, cmd_run_plan, cmd_run_prune, cmd_run_reconcile, cmd_run_status
from kura.run_commands.runpod_ssh import POD_SELF_DELETE_FUNCTION, _mark_runpod_outputs_collected, _mark_runpod_outputs_collecting, _runpod_lease_guard_shell, _unattended_completion_shell, _record_pulled_training_states, _ssh_base, _start_ssh_master, _extract_snapshot_delta_archive, _link_or_copy_snapshot_file, _local_reusable_snapshot_source, _mutate_run_status, _pull_remote_output_items, _record_pulled_outputs, _run_operation_lock, _same_remote_output_version, _try_sync_runpod_checkpoints, validate_safetensors_file, _validated_snapshot_manifest
from kura.container_scripts import script_source
from kura.executors import _redact_secret_text, docker_command, docker_preflight, launch_runpod, launch_runpod_session, observe_run, reconcile_docker, reconcile_runpod, runpod_gpu_availability, stage_runpod, stop_runpod
from kura.executors.common import _safe_env, format_launch_phases, launch_phases, record_launch_phase, unresolved_create_intents
from kura.executors.docker import _docker_timestamp
from kura.run_commands.experiment import format_run_completion
import kura.run_commands.runpod_ssh as runpod_ssh_module
from kura.media_types import frozen_suffixes
from kura.executors.runpod import _confirm_runpod_launch, RunPodAPIError, _is_runpod_capacity_error, _runpod_graphql_create_input, _runpod_request
from kura.fsio import FileLockBusy, file_lock
from kura.images import NEWEST_KNOWN_CUDA, PINNED_IMAGES
from kura.monitor import collect_run_summaries, _read_activity_from_stdout
from kura.render import _cleanup_stage, _ensure_lora_stage_visible, checkpoint_application, insert_lora_loader, _materialize_stage, _safe_stage_name, compile_render, launch_render
from kura.run_commands import _as_positive_int, _checkpoint_safety_preflight, _configured_gib, _estimate_backend_download_bytes, _local_launch_disk_preflight, _runpod_launch_disk_preflight, _runpod_ssh_details, _scp_to_runpod, _start_runpod_comfyui, _start_runpod_session_lease_guard, execute_run, launch_run, plan_run, stop_run
from kura.run_commands.plan import _disk_warnings, _hf_file_size_probe, _model_download_preflight_report, _model_download_safety_preflight, _runpod_capacity_payload, _image_preflight_report
from kura.run_commands.runpod_ssh import _record_remote_exit_observation, _run_operation_lock, _runpod_remote_job_script
from kura.doctor import readiness_gaps
from kura.paths import local_docker_mounts
from kura.storage import StorageStatus, ensure_free_bytes, probe_storage
from kura.tui import KuraMonitorApp, RunRow, _compact_path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from handoff_fixtures import freeze_fixture  # noqa: E402
from tests.platform_support import DATASET_IO, POSIX_PATHS, posix_only



# SSH connection reuse creates a socket directory under the user's home and
# probes it with ssh. Tests stay hermetic unless one opts back in with
# _REAL_SSH_CONTROL_DIR.
_REAL_SSH_CONTROL_DIR = runpod_ssh_module._ssh_control_dir
_SSH_REUSE_OFF = patch("kura.run_commands.runpod_ssh._ssh_control_dir", return_value=None)
# Collection marks are placed on the Pod over ssh; tests never reach a Pod.
_REAL_RUNPOD_MARK_COMMAND = runpod_ssh_module._run_runpod_mark_command
_RUNPOD_MARKS_OFF = patch("kura.run_commands.runpod_ssh._run_runpod_mark_command", return_value=True)


REPOSITORY = Path(__file__).resolve().parents[1]
RUNPOD_OBJECT_JOB_SOURCE = (Path(__file__).resolve().parents[1] / "docker" / "ai-toolkit" / "kura_runpod_object_job.py").read_text(encoding="utf-8")


# `kura init` probes Docker and ComfyUI for its readiness summary; unit tests
# must not depend on what this machine has running.
_READINESS_OFF = patch("kura.init_templates.readiness_gaps", return_value=[])


def setUpModule() -> None:
    _SSH_REUSE_OFF.start()
    _RUNPOD_MARKS_OFF.start()
    _READINESS_OFF.start()


def tearDownModule() -> None:
    _READINESS_OFF.stop()
    _RUNPOD_MARKS_OFF.stop()
    _SSH_REUSE_OFF.stop()


def _wsl_with_short_host_drive(*, linux_free_gib: int = 900, host_free_gib: int = 5) -> contextlib.ExitStack:
    """A WSL host whose Linux disk looks roomy while the Windows drive behind it is nearly full."""
    stack = contextlib.ExitStack()
    stack.enter_context(patch("kura.storage.is_wsl", return_value=True))
    stack.enter_context(patch("kura.storage._findmnt_for", return_value={"available": True, "fstype": "ext4", "target": "/", "source": "/dev/sdd"}))
    stack.enter_context(patch("kura.storage._auto_wsl_host_drive", return_value="C:"))
    stack.enter_context(patch("kura.storage._windows_drive_free_bytes", return_value=host_free_gib * 1024**3))
    stack.enter_context(patch("kura.storage.shutil.disk_usage", return_value=Mock(total=1000 * 1024**3, used=100 * 1024**3, free=linux_free_gib * 1024**3)))
    return stack


def _run_remote_in_process(args: argparse.Namespace) -> int:
    """A RunPod run through its in-process controller, as the runner's follower runs it."""
    from kura.run_commands.launch import run_remote

    options = {key: value for key, value in vars(args).items() if key not in {"run_id", "notify", "hold_for", "notify_repeat_interval"}}
    return run_remote(args.run_id, notify_channels=getattr(args, "notify", None), **options)

class InitCommandTests(unittest.TestCase):
    def test_cli_version_and_help_text(self) -> None:
        command = [sys.executable, "-c", "from kura.cli import main; main()"]
        version = subprocess.run([*command, "--version"], text=True, capture_output=True, check=False)
        self.assertEqual(version.returncode, 0)
        self.assertIn(f"kura {__version__}", version.stdout)

        help_result = subprocess.run([*command, "--help"], text=True, capture_output=True, check=False)
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("Agent-first, file-first workspace", help_result.stdout)
        self.assertIn("Create a workspace here", help_result.stdout)

        run_help = subprocess.run([*command, "run", "--help"], text=True, capture_output=True, check=False)
        self.assertEqual(run_help.returncode, 0)
        self.assertIn("Execute using the executor frozen in the compiled run", run_help.stdout)

        for subcommand in (("run", "execute"), ("render", "launch")):
            launch_help = subprocess.run([*command, *subcommand, "--help"], text=True, capture_output=True, check=False)
            self.assertEqual(launch_help.returncode, 0)
            self.assertIn("--yes", launch_help.stdout)
            self.assertIn("explicit user instruction", launch_help.stdout)

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
                for relative in ("workspace.yaml", "index.jsonl", "datasets", "runs", "workflows", "promptsets", "cache/huggingface", "cache/models", "knowledge/regrets.md"):
                    self.assertTrue((root / relative).exists(), relative)
                for relative in ("experiments", "backends", "executors", "docker"):
                    self.assertFalse((root / relative).exists(), relative)
                workspace = yaml.safe_load((root / "workspace.yaml").read_text(encoding="utf-8"))
                self.assertNotIn("mounts", workspace.get("docker", {}))
                self.assertEqual(workspace["runpod"]["gpu_type_ids"], ["NVIDIA RTX A5000", "NVIDIA A40"])
                self.assertEqual(workspace["runpod"]["gpu_type_priority"], "custom")
                self.assertNotIn("images", workspace)
                self.assertEqual(workspace["comfyui"]["lora_dir"], "")
                self.assertEqual(workspace["comfyui"]["lora_stage_cleanup"], "remove_after_render")
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
            self.assertEqual(run["schema_version"], 2)
            self.assertEqual(run["compute"]["executor"], "runpod")
            self.assertEqual(run["compute"]["gpu"], "NVIDIA RTX A5000")
            # No capacity policy is written: a RunPod run waits for a GPU by default.
            self.assertNotIn("capacity", run["compute"])
            status = json.loads((Path(directory) / "runs" / run_id / "status.json").read_text(encoding="utf-8"))
            self.assertIsNone(status["last_step"])
            self.assertEqual(
                sorted(path.name for path in (Path(directory) / "runs" / run_id).iterdir()),
                ["notes.md", "plan.md", "run.yaml", "status.json"],
            )

    def test_render_new_and_doctor_use_the_configured_comfyui_endpoint(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                Path("workspace.yaml").write_text("schema_version: 2\ncomfyui:\n  endpoint: http://10.0.0.5:8188/\n", encoding="utf-8")
                stdout = io.StringIO()
                with patch("sys.stdout", stdout):
                    self.assertEqual(cmd_render_new(argparse.Namespace(slug="remote-comfy")), 0)
                with patch("kura.doctor.shutil.which", return_value=None), \
                        patch("kura.doctor.urllib.request.urlopen", side_effect=OSError("offline")) as opened:
                    readiness_gaps(Path(directory))
            finally:
                os.chdir(previous)
            run = yaml.safe_load((Path(directory) / "runs" / stdout.getvalue().strip() / "run.yaml").read_text(encoding="utf-8"))
        self.assertEqual(run["generator"]["endpoint"], "http://10.0.0.5:8188")
        self.assertEqual(opened.call_args.args[0], "http://10.0.0.5:8188/system_stats")

    def test_readiness_never_prints_credentials_from_the_comfyui_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\ncomfyui:\n  endpoint: http://user:hunter2-secret@10.0.0.5:8188\n", encoding="utf-8")
            with patch("kura.doctor.shutil.which", return_value=None), patch("kura.doctor.urllib.request.urlopen", side_effect=OSError("offline")):
                gaps = readiness_gaps(root)
        comfy = [gap for gap in gaps if gap.startswith("ComfyUI")]
        self.assertEqual(len(comfy), 1)
        self.assertNotIn("hunter2-secret", comfy[0])
        self.assertIn("10.0.0.5:8188", comfy[0])

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
            run = yaml.safe_load((Path(directory) / "runs" / run_id / "run.yaml").read_text(encoding="utf-8"))
            self.assertIn("cases", run["inputs"])
            self.assertNotIn("promptset", run["inputs"])
            self.assertEqual(run["generator"]["endpoint"], "http://127.0.0.1:8188")
            status = json.loads((Path(directory) / "runs" / run_id / "status.json").read_text(encoding="utf-8"))
            self.assertIsNone(status["last_step"])
            self.assertEqual(
                sorted(path.name for path in (Path(directory) / "runs" / run_id).iterdir()),
                ["notes.md", "plan.md", "run.yaml", "status.json"],
            )

    @posix_only(DATASET_IO)
    def test_run_compile_rejects_musubi_dataset_without_images(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chdir(root)
            try:
                self.assertEqual(cmd_init(argparse.Namespace()), 0)
                dataset = root / "datasets" / "tiny"
                dataset.mkdir(parents=True)
                (dataset / "dataset.yaml").write_text(
                    "id: tiny\nitems_schema_version: 2\n", encoding="utf-8",
                )
                (dataset / "items.jsonl").write_text(json.dumps({
                    "id": "missing",
                    "files": [{"type": "file", "role": "target", "path": "missing.png"}],
                    "caption": None,
                }) + "\n", encoding="utf-8")
                stdout = io.StringIO()
                with patch("sys.stdout", stdout):
                    self.assertEqual(cmd_run_new(argparse.Namespace(experiment="exp", slug="empty-musubi", backend="musubi-tuner", executor="docker", gpu=None)), 0)
                run_id = stdout.getvalue().strip()
                run_path = root / "runs" / run_id / "run.yaml"
                run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
                run["model"]["base"] = "black-forest-labs/FLUX.2-klein-base-4B"
                run["datasets"] = [{"id": "tiny"}]
                run["recipe"] = {"steps": 1, "seed": 1}
                run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux2", "model_paths": {"dit": "/models/dit.safetensors", "vae": "/models/vae.safetensors", "text_encoder": "/models/text.safetensors"}}}
                run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
                stderr = io.StringIO()
                with patch("sys.stderr", stderr):
                    code = cmd_run_compile(argparse.Namespace(run_id=run_id))
                manifest_exists = (root / "runs" / run_id / "resolved" / "manifest.lock.yaml").exists()
            finally:
                os.chdir(previous)
        self.assertEqual(code, 1)
        self.assertIn("missing.png", stderr.getvalue())
        self.assertIn("does not exist", stderr.getvalue())
        self.assertFalse(manifest_exists)

    def test_run_compile_rejects_removed_backend_overrides(self) -> None:
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
        self.assertIn("backend_overrides is not supported", stderr.getvalue())

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
                run["recipe"] = {"steps": 1, "seed": 1}
                run["backend"] = {"name": "musubi-tuner", "config": {
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
        self.assertIn("model_downloads must map model roles to download mappings", stderr.getvalue())
        self.assertFalse(resolved_exists)

    @posix_only(DATASET_IO)
    def test_dataset_validate_checks_referenced_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "tiny"
            dataset.mkdir(parents=True)
            (dataset / "dataset.yaml").write_text("id: tiny\nitems_schema_version: 2\n", encoding="utf-8")
            (dataset / "items.jsonl").write_text('{"id":"1","files":[{"type":"file","role":"target","path":"images/001.png"}],"caption":{"text":"ok"}}\n', encoding="utf-8")
            stderr = io.StringIO()
            with patch("sys.stderr", stderr):
                self.assertEqual(cmd_dataset_validate(argparse.Namespace(dataset_dir=str(dataset))), 1)
            self.assertIn("referenced file does not exist", stderr.getvalue())
            (dataset / "images").mkdir()
            (dataset / "images" / "001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
            self.assertEqual(cmd_dataset_validate(argparse.Namespace(dataset_dir=str(dataset))), 0)

    @posix_only(DATASET_IO)
    def test_dataset_validate_rejects_paths_outside_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "datasets" / "tiny"
            dataset.mkdir(parents=True)
            outside = root / "outside.png"
            outside.write_bytes(b"\x89PNG\r\n\x1a\n")
            (dataset / "dataset.yaml").write_text("id: tiny\nitems_schema_version: 2\n", encoding="utf-8")
            (dataset / "items.jsonl").write_text('{"id":"1","files":[{"type":"file","role":"target","path":"../../outside.png"}],"caption":{"text":"ok"}}\n', encoding="utf-8")

            stderr = io.StringIO()
            with patch("sys.stderr", stderr):
                self.assertEqual(cmd_dataset_validate(argparse.Namespace(dataset_dir=str(dataset))), 1)

            self.assertIn("path must stay inside the dataset directory", stderr.getvalue())

    @posix_only(DATASET_IO)
    def test_provider_only_manifest_v2_runpod_compiles_and_plans_the_selected_transfer(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            os.chdir(root)
            try:
                self.assertEqual(cmd_init(argparse.Namespace()), 0)
                dataset = root / "datasets" / "tiny"
                (dataset / "images").mkdir(parents=True)
                (dataset / "images" / "001.png").write_bytes(b"\x89PNG\r\n\x1a\n")
                (dataset / "dataset.yaml").write_text(
                    "id: tiny\nitems_schema_version: 2\nstats:\n  count: 1\n", encoding="utf-8",
                )
                (dataset / "items.jsonl").write_text(
                    '{"id":"1","files":[{"type":"file","role":"target","path":"images/001.png"}],"caption":{"text":"ok"}}\n',
                    encoding="utf-8",
                )
                stdout = io.StringIO()
                with patch("sys.stdout", stdout):
                    self.assertEqual(cmd_run_new(argparse.Namespace(experiment="exp", slug="runpod", backend="ai-toolkit", executor="runpod", gpu="NVIDIA A40")), 0)
                run_id = stdout.getvalue().strip()
                run_path = root / "runs" / run_id / "run.yaml"
                run = yaml.safe_load(run_path.read_text(encoding="utf-8"))
                run["model"]["base"] = "stabilityai/stable-diffusion-xl-base-1.0"
                run["backend"]["config"]["model_arch"] = "sdxl"
                run["datasets"] = [{"id": "tiny"}]
                run["recipe"] = {"steps": 1, "seed": 1}
                run["compute"] = {"provider": "runpod", "gpu": "NVIDIA A40"}
                run_path.write_text(yaml.safe_dump(run), encoding="utf-8")
                (dataset / "unselected.bin").write_bytes(b"x" * 4096)
                stderr = io.StringIO()
                with patch("sys.stderr", stderr), patch("sys.stdout", io.StringIO()):
                    code = cmd_run_compile(argparse.Namespace(run_id=run_id))
                self.assertEqual(code, 0, stderr.getvalue())
                env_lock = yaml.safe_load((root / "runs" / run_id / "resolved" / "env.lock").read_text(encoding="utf-8"))
                self.assertIn(env_lock["kura_source"]["kind"], {"git", "editable", "unknown"})
                with patch("kura.run_commands.plan._hf_file_size_probe", return_value={"size": None, "error": "offline"}):
                    payload = plan_run(run_id)
                    locked = yaml.safe_load((root / "runs" / run_id / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8"))
                    disk = _runpod_launch_disk_preflight(locked, {"container_disk_gb": 50}, {"bytes": 0})
            finally:
                os.chdir(previous)

            transfer = payload["dataset_input"]["runpod_transfer"]
            self.assertEqual(transfer["payload_bytes"] > 0, True)
            self.assertEqual(transfer["remote_peak_bytes"], transfer["tar_bytes"] + transfer["payload_bytes"])
            self.assertEqual(disk["estimates"]["input_transfer"], transfer)
            self.assertGreaterEqual(disk["estimated_write_bytes"], transfer["remote_peak_bytes"])
            from kura.dataset_transfer import build_transfer_inventory

            inventory = build_transfer_inventory(root, root / "runs" / run_id, locked)
            sources = [item["destination"] for item in inventory["entries"] if item["namespace"] == "source"]
            self.assertEqual(sources, ["datasets/tiny/images/001.png"])

    def test_init_repairs_cache_directories_in_existing_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                Path("workspace.yaml").write_text("schema_version: 2\nname: existing\n", encoding="utf-8")
                self.assertEqual(cmd_init(argparse.Namespace()), 0)
                self.assertTrue((Path(directory) / "cache" / "huggingface").is_dir())
                self.assertTrue((Path(directory) / "cache" / "models").is_dir())
                self.assertEqual(yaml.safe_load((Path(directory) / "workspace.yaml").read_text(encoding="utf-8"))["name"], "existing")
            finally:
                os.chdir(previous)

    def test_object_job_template_rejects_download_keys_outside_workspace(self) -> None:
        start = RUNPOD_OBJECT_JOB_SOURCE.index("def download_prefix")
        end = RUNPOD_OBJECT_JOB_SOURCE.index("\n\ndef upload_tree")
        namespace: dict[str, Any] = {"Path": Path}
        exec(RUNPOD_OBJECT_JOB_SOURCE[start:end], namespace)

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
    def test_ai_toolkit_image_records_embedded_source_commit(self) -> None:
        dockerfile = (Path(__file__).parents[1] / "docker" / "ai-toolkit" / "Dockerfile").read_text(encoding="utf-8")

        self.assertIn('"ai_toolkit_commit": ai_toolkit_commit', dockerfile)
        self.assertIn('["git", "-C", "/app/ai-toolkit", "rev-parse", "HEAD"]', dockerfile)

    def test_ai_toolkit_image_applies_and_records_the_pinned_h3_gradient_patch(self) -> None:
        root = Path(__file__).parents[1]
        dockerfile = (root / "docker" / "ai-toolkit" / "Dockerfile").read_text(encoding="utf-8")
        patch_path = root / "docker" / "ai-toolkit" / "minimax_h3_finite_gradients.patch"
        patch_source = patch_path.read_text(encoding="utf-8")

        self.assertTrue(patch_path.is_file())
        self.assertIn("d1985f9bf380b6ce1c409b7875e2d367df486e19", dockerfile)
        self.assertIn("COPY docker/ai-toolkit/minimax_h3_finite_gradients.patch", dockerfile)
        self.assertIn("git apply --check", dockerfile)
        self.assertIn('"source_patches": source_patches', dockerfile)
        self.assertNotIn("bool(finite.all())", patch_source)
        self.assertGreaterEqual(patch_source.count("torch.where(finite, "), 2)
        self.assertIn("checkpoint-static-nonfinite-masking-v1", dockerfile)

    def test_musubi_image_build_uses_pinned_release_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({'docker': {}, 'images': {'musubi-tuner': 'kura/musubi-tuner:test'}}),
                encoding="utf-8",
            )
            commands: list[list[str]] = []

            def fake_docker_run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, "sha256:image\n" if capture else "", "")

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.cli._docker_run", side_effect=fake_docker_run), patch("kura.cli.development_checkout", return_value=REPOSITORY), patch("kura.cli._docker_storage_summary", return_value={"usage": []}):
                    self.assertEqual(cmd_image_build(argparse.Namespace(name="musubi-tuner", ref=None)), 0)
            finally:
                os.chdir(previous)

            self.assertIn("MUSUBI_TUNER_REF=v0.3.5", commands[0])

    def test_image_build_reads_the_checkout_from_a_nested_workspace_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            nested = root / "datasets" / "tiny"
            nested.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            calls: list[list[str]] = []

            def fake_docker_run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, "sha256:image\n" if capture else "", "")

            previous = Path.cwd()
            os.chdir(nested)
            try:
                with patch("kura.cli._docker_run", side_effect=fake_docker_run), patch("kura.cli.development_checkout", return_value=REPOSITORY), patch("kura.cli._docker_storage_summary", return_value={"usage": []}):
                    self.assertEqual(cmd_image_build(argparse.Namespace(name="ai-toolkit", ref=None)), 0)
            finally:
                os.chdir(previous)

            self.assertGreaterEqual(len(calls), 1)
            build = calls[0]
            self.assertEqual(build[build.index("--file") + 1], str(REPOSITORY / "docker/ai-toolkit/Dockerfile"))
            self.assertIn(
                "AI_TOOLKIT_IMAGE=ostris/aitoolkit:0.13.18@sha256:9bc99d51efc5b6c38a951b3bf8547bda0f9db58abeb75573548d449f82b34bcc",
                build,
            )
            self.assertEqual(build[-1], str(REPOSITORY))

    def test_ai_toolkit_image_build_ref_overrides_upstream_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({'docker': {}, 'images': {'ai-toolkit': 'kura/ai-toolkit:test'}}),
                encoding="utf-8",
            )
            commands: list[list[str]] = []

            def fake_docker_run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                return subprocess.CompletedProcess(command, 0, "sha256:image\n" if capture else "", "")

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.cli._docker_run", side_effect=fake_docker_run), patch("kura.cli.development_checkout", return_value=REPOSITORY), patch("kura.cli._docker_storage_summary", return_value={"usage": []}):
                    self.assertEqual(cmd_image_build(argparse.Namespace(name="ai-toolkit", ref="ostris/aitoolkit:custom")), 0)
            finally:
                os.chdir(previous)

            self.assertIn("AI_TOOLKIT_IMAGE=ostris/aitoolkit:custom", commands[0])

    def test_image_build_rejects_large_build_cache_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({'docker': {}, 'images': {'ai-toolkit': 'kura/ai-toolkit:test'}}),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.cli.development_checkout", return_value=REPOSITORY),
                    patch("kura.cli._docker_storage_summary", return_value={"usage": [{"Type": "Build Cache", "size_bytes": 31 * 1024**3}]}),
                    patch("sys.stderr", new_callable=__import__("io").StringIO) as stderr,
                ):
                    code = cmd_image_build(argparse.Namespace(name="ai-toolkit", ref=None, allow_large_build_cache=False))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            self.assertIn("Docker build cache exceeds 30GiB", stderr.getvalue())

    def test_image_build_reads_the_build_cache_limit_doctor_reads(self) -> None:
        # One setting decides both doctor's warning and image build's stop; outside a workspace the default applies.
        cache = {"usage": [{"Type": "Build Cache", "size_bytes": 25 * 1024**3}]}
        for config, expected_code in (({"docker": {"build_cache_limit_gb": 20}}, 1), ({"docker": {"build_cache_limit_gb": 40}}, 0), (None, 0)):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                if config is not None:
                    (root / "workspace.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
                previous = Path.cwd()
                os.chdir(root)
                try:
                    with (
                        patch("kura.cli.development_checkout", return_value=REPOSITORY),
                        patch("kura.cli._docker_storage_summary", return_value=cache),
                        patch("kura.cli._docker_run", return_value=subprocess.CompletedProcess([], 0, "", "")),
                        patch("sys.stderr", new_callable=__import__("io").StringIO) as stderr,
                    ):
                        code = cmd_image_build(argparse.Namespace(name="ai-toolkit", ref=None, allow_large_build_cache=False))
                finally:
                    os.chdir(previous)
                self.assertEqual(code, expected_code, stderr.getvalue())
                if expected_code:
                    self.assertIn("Docker build cache exceeds 20GiB", stderr.getvalue())


class DoctorDockerTests(unittest.TestCase):
    def test_cleanup_all_is_dry_run_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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

    def test_cleanup_image_uses_the_workspace_image_when_it_is_present(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump(
                    {'docker': {}, 'images': {'ai-toolkit': 'kura/ai-toolkit:test'}}
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.cli._docker_image_exists", side_effect=lambda image: image == "kura/ai-toolkit:test"):
                    self.assertEqual(_docker_cleanup_image(), "kura/ai-toolkit:test")
            finally:
                os.chdir(previous)

    def test_cleanup_runs_keeps_outputs_without_explicit_final_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "old"
            (run / "cache").mkdir(parents=True)
            (run / "outputs").mkdir()
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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

    def test_cleanup_runs_offers_only_disposable_dataset_view_remnants(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            dataset_file = root / "datasets" / "tiny" / "a.png"
            dataset_file.parent.mkdir(parents=True)
            dataset_file.write_bytes(b"image")
            missing = {"realization_id": "launch", "state": "unknown", "container_missing": True}
            unclear = {"realization_id": "launch", "state": "unknown", "container_missing": False}
            observed = {
                "state": "unknown", "started": "2026-01-04T00:00:00+00:00",
                "last_realization": "realizations/launch.json",
                "last_observation": "realizations/launch.observed.json",
            }
            observations = {"gone": missing, "deferred-then-gone": missing, "unclear": unclear}
            runs = {
                "gone": (observed, True),
                "deferred-then-gone": ({
                    **observed, "state": "recovery_required", "recovery_required": True,
                    "dataset_input_postflight": {"view_cleanup": "deferred"},
                }, True),
                "unclear": (observed, False),
                "failed-cleanup": ({
                    "state": "completed", "ended": "2026-01-03T00:00:00+00:00",
                    "dataset_input_postflight": {"view_cleanup": "failed"},
                }, True),
                "deferred": ({
                    "state": "completed", "ended": "2026-01-02T00:00:00+00:00",
                    "dataset_input_postflight": {"view_cleanup": "deferred"},
                }, False),
                "running": ({"state": "running", "started": "2026-01-01T00:00:00+00:00"}, False),
            }
            for run_id, (status, _) in runs.items():
                view = root / "runs" / run_id / "cache" / "dataset-view" / "ai-toolkit" / "tiny"
                view.mkdir(parents=True)
                (view / "a.png").symlink_to(dataset_file)
                (root / "runs" / run_id / "status.json").write_text(json.dumps(status), encoding="utf-8")
                if run_id in observations:
                    record = root / "runs" / run_id / "realizations" / "launch.observed.json"
                    record.parent.mkdir()
                    record.write_text(json.dumps(observations[run_id]), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.cli._path_size_bytes", return_value=1),
                    patch("kura.cli._root_owned_files", return_value={"supported": True, "count": 0, "samples": []}),
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                ):
                    code = cmd_cleanup(argparse.Namespace(target="runs", keep_last=30, delete_final_artifacts=False, yes=False))
                payload = json.loads(stdout.getvalue())
                run_actions = next(item for item in payload["actions"] if item.get("target") == "runs/*")["run_actions"]
                offered = {item["id"] for item in run_actions if item["classification"] == "safe-run-dataset-view-remnant"}
                self.assertEqual(code, 0)
                self.assertEqual(offered, {run_id for run_id, (_, expected) in runs.items() if expected})
                for run_id in runs:
                    self.assertTrue((root / "runs" / run_id / "cache" / "dataset-view").is_dir())

                with (
                    patch("kura.cli._path_size_bytes", return_value=1),
                    patch("kura.cli._root_owned_files", return_value={"supported": True, "count": 0, "samples": []}),
                    patch("sys.stdout", new_callable=__import__("io").StringIO),
                ):
                    self.assertEqual(cmd_cleanup(argparse.Namespace(target="runs", keep_last=30, delete_final_artifacts=False, yes=True)), 0)
            finally:
                os.chdir(previous)
            self.assertFalse((root / "runs" / "gone" / "cache" / "dataset-view").exists())
            self.assertTrue((root / "runs" / "deferred" / "cache" / "dataset-view").is_dir())
            self.assertTrue(dataset_file.is_file())

    def test_cleanup_runs_requires_explicit_final_delete_for_whole_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "runs" / "old"
            (run / "outputs").mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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

    @posix_only(POSIX_PATHS)
    def test_fix_permissions_dry_run_reports_root_owned_targets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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

    @posix_only(POSIX_PATHS)
    def test_fix_links_rewrites_repairable_container_private_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({"docker": {}}),
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

    @posix_only(POSIX_PATHS)
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

    @posix_only(POSIX_PATHS)
    def test_doctor_disk_reports_workspace_storage_and_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "workspace.yaml").write_text(
                yaml.safe_dump(
                    {"docker": {}}
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
            self.assertIn("Docker build cache exceeds 30GiB", payload["warnings"])
            self.assertIn("cache/runs contain root-owned files; cleanup may require permission repair", payload["warnings"])

    def test_doctor_disk_reports_large_cache_runs_as_advisory_when_space_is_ok(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
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

            def fake_probe(paths: dict[str, Path], config: dict[str, object] | None = None) -> dict[str, StorageStatus]:
                return {
                    name: StorageStatus(
                        path=str(path),
                        probe=str(path),
                        backing_id="test-disk",
                        backing_kind="filesystem",
                        linux_free_bytes=400 * 1024**3,
                        linux_total_bytes=500 * 1024**3,
                        host_free_bytes=None,
                        effective_free_bytes=400 * 1024**3,
                        confidence="exact",
                        mount={"available": True},
                    )
                    for name, path in paths.items()
                }

            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.doctor._path_size_bytes", side_effect=fake_size),
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
            self.assertEqual(code, 0)
            self.assertEqual(payload["warnings"], [])
            self.assertIn("workspace cache+runs exceed 30GiB", payload["advisories"])
            issue = next(item for item in payload["issues"] if item["code"] == "workspace_cache_runs_large")
            self.assertEqual(issue["severity"], "advisory")
            self.assertEqual(issue["size_bytes"], 32 * 1024**3)

    @posix_only(POSIX_PATHS)
    def test_doctor_disk_warns_about_container_private_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({"docker": {}}),
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
                    patch("kura.storage.shutil.disk_usage", return_value=Mock(total=1000 * 1024**3, used=100 * 1024**3, free=900 * 1024**3)),
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

    def test_doctor_disk_and_local_launch_share_one_free_space_floor(self) -> None:
        for min_free_gb, short in ((40, False), (60, True)):
            with self.subTest(min_free_gb=min_free_gb), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                (root / "workspace.yaml").write_text(yaml.safe_dump({"docker": {"mounts": [], "min_free_gb": min_free_gb}}), encoding="utf-8")
                (root / "cache").mkdir()
                (root / "runs").mkdir()
                previous = Path.cwd()
                os.chdir(root)
                try:
                    with (
                        _wsl_with_short_host_drive(linux_free_gib=900, host_free_gib=50),
                        patch("kura.doctor._path_size_bytes", return_value=0),
                        patch("kura.doctor._docker_storage_summary", return_value={"daemon_reachable": True, "usage": [], "kura_managed": {}}),
                        patch("kura.doctor._root_owned_files", return_value={"supported": True, "count": 0, "samples": [], "truncated": False}),
                        patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")),
                        patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout,
                    ):
                        code = cmd_doctor_disk(argparse.Namespace())
                        launch = contextlib.nullcontext() if not short else self.assertRaisesRegex(ValueError, f"requires at least {min_free_gb} GiB")
                        with launch:
                            _local_launch_disk_preflight(root, {"type": "train"}, {"docker": {"min_free_gb": min_free_gb}})
                finally:
                    os.chdir(previous)
                payload = json.loads(stdout.getvalue())
                low = [issue for issue in payload["issues"] if issue["code"] == "workspace_effective_free_low"]
                self.assertEqual(code, 1 if short else 0)
                self.assertEqual(bool(low), short)
                if short:
                    self.assertEqual(low[0]["threshold_bytes"], min_free_gb * 1024**3)
                    self.assertIn(f"workspace backing store has less than {min_free_gb}GiB effective free (C:)", payload["warnings"])

    def test_disk_checks_and_cleanup_follow_a_cache_outside_the_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as elsewhere:
            root = Path(directory).resolve()
            outside = Path(elsewhere).resolve() / "hf"
            (outside / "hub").mkdir(parents=True)
            (outside / "hub" / "weights.safetensors").write_bytes(b"x" * 10)
            config = {"schema_version": 2, "docker": {"hf_cache": str(outside), "min_free_gb": 1}}
            (root / "workspace.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.doctor._docker_storage_summary", return_value={"daemon_reachable": True, "usage": [], "kura_managed": {}}),
                    patch("kura.doctor._root_owned_files", return_value={"supported": True, "count": 0, "samples": [], "truncated": False}),
                    patch("sys.stdout", new_callable=__import__("io").StringIO) as doctor_out,
                ):
                    cmd_doctor_disk(argparse.Namespace())
                with patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
                    launch = _local_launch_disk_preflight(root, {"type": "train", "safety": {"allow_storage_risk": True}}, config)
                with patch("kura.cli._root_owned_files", return_value={"supported": True, "count": 0, "samples": [], "truncated": False}), \
                        patch("sys.stdout", new_callable=__import__("io").StringIO) as cleanup_out:
                    code = cmd_cleanup(argparse.Namespace(target="cache", keep_last=30, delete_final_artifacts=False, yes=True))
            finally:
                os.chdir(previous)
            doctor = json.loads(doctor_out.getvalue())
            cleanup = json.loads(cleanup_out.getvalue())
            self.assertEqual(doctor["sizes"]["huggingface_cache"]["path"], str(outside))
            self.assertEqual(doctor["storage"]["huggingface_cache"]["path"], str(outside))
            self.assertEqual(launch["paths"]["hf_cache"]["path"], str(outside))
            self.assertEqual([name for name, item in launch["paths"].items() if item["path"] == str(outside)], ["hf_cache"])
            self.assertEqual(code, 0)
            # A cache outside the workspace may be shared: cleanup reports it and leaves it.
            item = next(action for action in cleanup["actions"] if action.get("path") == str(outside))
            self.assertEqual(item["classification"], "outside-workspace")
            self.assertTrue((outside / "hub" / "weights.safetensors").is_file())

    def test_doctor_docker_reports_kura_managed_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump(
                    {'docker': {}, 'images': {'ai-toolkit': 'kura/ai-toolkit:test'}}
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
                with patch("kura.doctor.shutil.which", return_value="/usr/bin/docker"), patch("kura.doctor.docker_daemon_problem", return_value=None), patch("kura.doctor._docker_run", side_effect=fake_docker_run), patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    self.assertEqual(cmd_doctor_docker(argparse.Namespace()), 0)
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            managed = payload["docker_storage"]["kura_managed"]
            # Counts, and a command that removes only Kura's stopped containers.
            self.assertEqual((managed["containers"], managed["stopped_containers"], managed["volumes"]), (1, 1, 1))
            # Not `kura run prune`: with --yes it also removes old runs.
            # The workspace AGENTS.md keeps state-changing Docker commands with the user.
            self.assertEqual(managed["remove_stopped"], "ask the user to run: docker container prune --filter label=io.kura.managed=true")
            self.assertEqual(payload["huggingface_cache"]["path"], str(root.resolve() / "cache" / "huggingface"))
            self.assertNotIn("docker.mounts", payload["huggingface_cache"].get("note", ""))

    def test_doctor_docker_treats_an_unpulled_pinned_image_as_ready(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")

            def fake_docker_run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
                if command[:3] == ["docker", "image", "inspect"]:
                    return subprocess.CompletedProcess(command, 1, "", "No such image")
                if command[:3] == ["docker", "system", "df"] or command[:2] == ["docker", "ps"] or command[:3] == ["docker", "volume", "ls"]:
                    return subprocess.CompletedProcess(command, 0, "", "")
                return subprocess.CompletedProcess(command, 0, "ok\n", "")

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.shutil.which", return_value="/usr/bin/docker"), patch("kura.doctor.docker_daemon_problem", return_value=None), patch("kura.doctor._docker_run", side_effect=fake_docker_run), patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    self.assertEqual(cmd_doctor_docker(argparse.Namespace()), 0)
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["local_image"], "will be pulled")
            self.assertIsNone(payload["gpu_available"])
            self.assertIn("docker pull nomadoor/kura-ai-toolkit@sha256:", payload["diagnosis"])

    def test_doctor_workspace_points_an_old_schema_at_migrate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    self.assertEqual(cmd_doctor_workspace(argparse.Namespace()), 1)
            finally:
                os.chdir(previous)
            self.assertIn("kura workspace migrate", json.loads(stdout.getvalue())["configuration_error"])

    def test_doctor_musubi_reports_adapter_script_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump(
                    {'docker': {}, 'images': {'musubi-tuner': 'kura/musubi-tuner:test'}}
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
                    {'docker': {}, 'images': {'musubi-tuner': 'configured/missing:test'}}
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

    def test_doctor_sd_scripts_fails_when_a_required_option_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({'docker': {}, 'images': {'sd-scripts': 'kura/sd-scripts:test'}}),
                encoding="utf-8",
            )
            probe_payload = {
                "scripts": {
                    "anima_train_network.py": {
                        "exists": True,
                        "help_exit_code": 0,
                        "required_options": {"--attn_mode": False},
                    }
                },
                "imports": {"torch": "test"},
                "compatibility": {"sd_checkpoint_symlink_safe": True, "sdxl_checkpoint_symlink_safe": True},
                "torch": {"cuda_available": True},
            }

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                if command[:3] == ["/usr/bin/docker", "image", "inspect"]:
                    return subprocess.CompletedProcess(command, 0, "[]", "")
                if command[:3] == ["/usr/bin/docker", "run", "--rm"]:
                    return subprocess.CompletedProcess(command, 1, json.dumps(probe_payload), "")
                raise AssertionError(command)

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.shutil.which", return_value="/usr/bin/docker"), patch("kura.doctor.subprocess.run", side_effect=fake_run), patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    code = cmd_doctor_sd_scripts(argparse.Namespace(no_gpu=False, timeout=30.0, image=None))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertFalse(payload["checks"]["probe_exit"])
            self.assertFalse(payload["checks"]["tier1_scripts"])
            self.assertIn("probe failed", payload["diagnosis"])


class MonitorCommandTests(unittest.TestCase):
    def test_monitor_passes_limit_to_textual_app(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (root / "index.jsonl").write_text(json.dumps({"id": "example"}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text("id: example\ntype: train\n", encoding="utf-8")
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("id: example\ntype: train\nrecipe: {steps: 1}\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_step": 0}), encoding="utf-8")
            (run_dir / "resolved" / "backend-command.lock.json").write_text(
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
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("id: example\ntype: train\nrecipe: {steps: 1}\n", encoding="utf-8")
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
                (run_dir / "resolved" / "manifest.lock.yaml").write_text("id: example\ntype: train\nrecipe: {steps: 1}\n", encoding="utf-8")
                (run_dir / "status.json").write_text(json.dumps({"state": "completed", "last_step": 1, "total_steps": 1}), encoding="utf-8")
                second_id = "20260713-0000_musubi-real-smoke-ideogram4_abcd"
                second_dir = root / "runs" / second_id
                (second_dir / "resolved").mkdir(parents=True)
                (second_dir / "logs").mkdir()
                (second_dir / "realizations").mkdir()
                with (root / "index.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"id": second_id}) + "\n")
                (second_dir / "run.yaml").write_text(f"id: {second_id}\ntype: train\n", encoding="utf-8")
                (second_dir / "resolved" / "manifest.lock.yaml").write_text(f"id: {second_id}\ntype: train\nrecipe: {{steps: 1}}\n", encoding="utf-8")
                (second_dir / "status.json").write_text(json.dumps({"state": "completed", "last_step": 1, "total_steps": 1}), encoding="utf-8")
                app = KuraMonitorApp(root, interval=999)
                async with app.run_test(size=(100, 30)) as pilot:
                    await pilot.pause(0.1)
                    active = app.screen.query_one("#nav-active")
                    history_title = app.screen.query_one("#history-title")
                    history_filter = app.screen.query_one("#history-filter")
                    self.assertEqual(history_title.region.y, active.region.bottom)
                    self.assertEqual(history_title.content_region.y, active.region.bottom + 1)
                    self.assertEqual(history_filter.region.y, history_title.region.bottom)
                    for selector in ("#nav-active", "#nav-history", "#detail", "#loss", "#datasets", "#compute"):
                        self.assertEqual(app.screen.query_one(selector).styles.scrollbar_size_vertical, 1)
                    await pilot.click("#filter-render")
                    await pilot.pause(0.1)
                    self.assertEqual(app.screen.history_filter, "render")
                    await pilot.click("#filter-all")
                    await pilot.pause(0.1)
                    self.assertEqual(app.screen.history_filter, "all")
                    await pilot.click("#datasets-link")
                    await pilot.pause(0.1)
                    self.assertEqual(app.screen.tab, "datasets")
                    await pilot.click("#runs-back")
                    await pilot.pause(0.1)
                    self.assertEqual(app.screen.tab, "runs")
                    restored = next(row for row in app.screen.query(RunRow) if row.summary and row.summary.id == second_id)
                    self.assertEqual(restored.render().plain, restored.render_row().plain)
                    await pilot.press("down")
                    await pilot.pause(0.1)
                    first_selected = app.screen.selected_run_id
                    await pilot.press("down")
                    await pilot.pause(0.1)
                    self.assertIsNotNone(first_selected)
                    self.assertNotEqual(app.screen.selected_run_id, first_selected)

        asyncio.run(run_case())


class WorkspaceDiscoveryTests(unittest.TestCase):
    def test_run_status_resolves_workspace_from_subdirectory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            nested = root / "datasets" / "tiny"
            run_dir.mkdir(parents=True)
            nested.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "completed"}), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(nested)
            try:
                self.assertEqual(cmd_run_status(argparse.Namespace(run_id="example")), 0)
            finally:
                os.chdir(previous)

    def test_run_status_names_the_realization_instead_of_printing_its_launch_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            launch = {"id": "r1", "docker_command": ["--flag"] * 5000, "backend_command": {"argv": ["x"] * 5000}}
            (run_dir / "realizations" / "r1.json").write_text(json.dumps(launch), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "completed", "last_realization": "realizations/r1.json"}), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    self.assertEqual(cmd_run_status(argparse.Namespace(run_id="example")), 0)
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["last_realization"], "realizations/r1.json")
            self.assertNotIn("latest_realization", payload)
            self.assertLess(len(stdout.getvalue()), 4000)

    def test_doctor_workspace_reports_resolved_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            nested = root / "runs"
            nested.mkdir()
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(nested)
            try:
                self.assertEqual(cmd_doctor_workspace(argparse.Namespace()), 0)
            finally:
                os.chdir(previous)

    def test_doctor_workspace_reports_each_image_and_warns_only_for_a_mutable_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({"schema_version": 2, "images": {"musubi-tuner": "kura/musubi-tuner:dev"}}),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(cmd_doctor_workspace(argparse.Namespace()), 1)
                payload = json.loads(stdout.getvalue())
            finally:
                os.chdir(previous)
        self.assertEqual(payload["docker_images"]["ai-toolkit"]["origin"], "pinned")
        self.assertEqual(payload["docker_images"]["musubi-tuner"], {"image": "kura/musubi-tuner:dev", "origin": "override"})
        self.assertEqual(len(payload["warnings"]), 1)
        self.assertIn("images.musubi-tuner uses mutable tag", payload["warnings"][0])


class RunPlanTests(unittest.TestCase):
    def test_disk_warnings_do_not_flag_a_single_checkpoint_as_frequent(self) -> None:
        run = {
            "recipe": {"steps": 1},
            "backend": {"name": "musubi-tuner", "config": {"save_every_n_steps": 1}},
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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            (dataset_dir / "items.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": "plan-example",
                        "type": "train",
                                                "model": {"base": "black-forest-labs/FLUX.2-klein-base-9B", "revision": "main"},
                        "compute": {"executor": "runpod", "gpu": "NVIDIA RTX A5000"},
                        "datasets": [{"id": "tiny", "role": "target", "digest": "sha256:abc"}],
                        "recipe": {"steps": 1500, "seed": 42},
                        "sampling": {"cadence_steps": 100},
                        "backend": {"name": "musubi-tuner", "config": {
                            "network_dim": 16,
                            "network_alpha": 1024,
                            "learning_rate": "0.00005",
                            "batch_size": 2,
                            "resolution": [768],
                            "fp8_base": True,
                            "gradient_checkpointing": True,
                            "save_every_n_steps": 100,
                            "blocks_to_swap": 3,
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
            self.assertNotIn("Backend config", output)
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

    def test_run_plan_shows_live_runpod_capacity_and_choices(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "capacity-plan"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text(
                "runpod:\n  gpu_type_ids: [NVIDIA RTX A5000, NVIDIA A40]\n  cloud_types: [COMMUNITY]\n",
                encoding="utf-8",
            )
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": "capacity-plan",
                        "type": "train",
                        "model": {"base": "custom"},
                        "compute": {"executor": "runpod", "gpu": ["NVIDIA RTX A5000", "NVIDIA A40"], "capacity": {"mode": "immediate"}},
                        "datasets": [],
                        "recipe": {"steps": 10},
                        "backend": {"name": "ai-toolkit", "config": {}},
                    },
                    sort_keys=False,
                ),
                encoding="utf-8",
            )
            measurement = {
                "status": "ok",
                "checked_at": "2026-07-14T12:00:00+09:00",
                "gpu_count": 1,
                "candidates": [
                    {"gpu_type_id": "NVIDIA RTX A5000", "display_name": "RTX A5000", "memory_gb": 24, "clouds": [{"cloud_type": "COMMUNITY", "stock_status": "None", "available": False, "price_per_hour": None}]},
                    {"gpu_type_id": "NVIDIA A40", "display_name": "A40", "memory_gb": 48, "clouds": [{"cloud_type": "COMMUNITY", "stock_status": "Low", "available": True, "price_per_hour": 0.4}]},
                ],
            }
            previous = Path.cwd()
            os.chdir(root)
            try:
                with (
                    patch("kura.run_commands.plan.runpod_gpu_availability", return_value=measurement),
                    patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "missing_metadata", "size_bytes": None}),
                    patch("sys.stdout", new_callable=io.StringIO) as stdout,
                ):
                    self.assertEqual(cmd_run_plan(argparse.Namespace(run_id="capacity-plan", json=False)), 0)
            finally:
                os.chdir(previous)
            output = stdout.getvalue()
            self.assertIn("RunPod capacity", output)
            self.assertIn("RTX A5000 · 24 GB", output)
            self.assertIn("COMMUNITY: None · unavailable", output)
            self.assertIn("launch now: NVIDIA A40 / COMMUNITY", output)
            self.assertIn("set compute.capacity.mode=wait before compile", output)

    def test_runpod_capacity_plan_survives_invalid_runpod_settings(self) -> None:
        run = {
            "compute": {"executor": "runpod", "gpu": ["NVIDIA A40"]},
        }
        with patch(
            "kura.run_commands.plan.runpod_gpu_availability",
            side_effect=ValueError("runpod.cloud_types must contain COMMUNITY or SECURE"),
        ):
            payload = _runpod_capacity_payload(run, {"runpod": {}})

        assert payload is not None
        self.assertEqual(payload["measurement"]["status"], "unavailable")
        self.assertEqual(payload["measurement"]["candidates"], [])
        self.assertIn("runpod.cloud_types", payload["measurement"]["reason"])
        self.assertEqual(payload["immediate_candidates"], [])

    def test_runpod_capacity_plan_measures_hosts_for_the_compiled_image(self) -> None:
        run = {"backend": {"name": "sd-scripts"}, "compute": {"executor": "runpod", "gpu": ["NVIDIA A40"]}}
        override = {"runpod": {}, "images": {"sd-scripts": "example/sd@sha256:" + "5" * 64}}
        unavailable = {"status": "unavailable", "reason": "test", "candidates": []}
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "resolved").mkdir()
            (run_dir / "resolved" / "env.lock").write_text(yaml.safe_dump({"selected_image": PINNED_IMAGES["sd-scripts"], "image_origin": "pinned"}), encoding="utf-8")
            with patch("kura.run_commands.plan.runpod_gpu_availability", return_value=unavailable) as measure:
                _runpod_capacity_payload(run, override, run_dir)
                _runpod_capacity_payload(run, override)
        self.assertEqual(measure.call_args_list[0].kwargs["min_cuda_version"], "12.8")
        self.assertEqual(measure.call_args_list[1].kwargs["min_cuda_version"], NEWEST_KNOWN_CUDA)
        templated = {"runpod": {"template_id": "tpl"}}
        with patch("kura.run_commands.plan.runpod_gpu_availability", return_value=unavailable) as measure:
            _runpod_capacity_payload(run, templated)
            with patch("kura.run_commands.plan.get_backend", return_value=SimpleNamespace(image_name="sd-scripts", runpod_template_compatible=True)):
                _runpod_capacity_payload(run, templated)
        # Launch drops the template for adapters that do not accept one, and keeps the image's filter.
        self.assertEqual(measure.call_args_list[0].kwargs["min_cuda_version"], "12.8")
        self.assertEqual(measure.call_args_list[1].kwargs["min_cuda_version"], NEWEST_KNOWN_CUDA)

    def test_run_plan_prints_musubi_download_estimates_and_cache_hits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "download-plan"
            cached = root / "cache" / "models" / "musubi" / "example--model" / "vae" / "vae.safetensors"
            cached.parent.mkdir(parents=True)
            cached.write_bytes(b"x" * 1024)
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": "download-plan",
                        "type": "train",
                                                "model": {"base": "custom"},
                        "backend": {"name": "musubi-tuner", "config": {
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
        self.assertEqual(resources["training"]["batch_size"], "(not set)")
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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text("id: compiled-example\ntype: train\nbackend: {name: ai-toolkit, config: {config: {train: {lr: 1e-4}}}}\n", encoding="utf-8")
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("id: compiled-example\ntype: train\nbackend: {name: musubi-tuner, config: {learning_rate: 0.00005}}\ndatasets: [{id: tiny}]\n", encoding="utf-8")
            (run_dir / "resolved" / "dataset-observations.lock.yaml").write_text(
                yaml.safe_dump({"schema_version": 1, "datasets": [{"dataset": "tiny", "observations": {"sample_count": 2, "captions_missing": 0, "condition_counts": {"source": 2}, "aspect_ratio_mismatches": {}}, "structural_findings": []}]}),
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                # The host's real free space must not decide this test.
                with patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout, \
                        patch("kura.run_commands.plan._local_launch_disk_preflight", return_value={"required_gib": 100, "paths": {"workspace": {"path": "ws", "effective_free_bytes": 190 * 1024**3, "required_bytes": 100 * 1024**3}}}):
                    self.assertEqual(cmd_run_plan(argparse.Namespace(run_id="compiled-example", json=True)), 0)
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertTrue(payload["compiled"])
            self.assertEqual(payload["source"], "runs/compiled-example/resolved/manifest.lock.yaml")
            self.assertEqual(payload["intent_source"], "runs/compiled-example/run.yaml")
            self.assertEqual(payload["resolved_manifest"], "runs/compiled-example/resolved/manifest.lock.yaml")
            self.assertEqual(payload["backend"]["name"], "musubi-tuner")
            self.assertEqual(payload["backend"]["config"], {"learning_rate": 0.00005})
            self.assertEqual(payload["datasets"][0]["observations"]["samples"], 2)
            self.assertEqual(payload["datasets"][0]["observations"]["conditions"], {"source": 2})
            self.assertIn("preflight", payload)
            disk_records = [item for item in payload["preflight"] if item["check"] == "disk"]
            self.assertTrue(disk_records)
            self.assertIn("passes", disk_records[0]["fact"])
            self.assertNotIn("disk_warnings", payload)

    def test_run_plan_prints_preflight_section(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "preflight-example"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
                    patch("kura.run_commands.plan._local_launch_disk_preflight", return_value={"required_gib": 100, "paths": {"workspace": {"path": "ws", "effective_free_bytes": 190 * 1024**3, "required_bytes": 100 * 1024**3}}}),
                ):
                    self.assertEqual(cmd_run_plan(argparse.Namespace(run_id="preflight-example", json=False)), 0)
            finally:
                os.chdir(previous)
        output = stdout.getvalue()
        self.assertIn("Preflight", output)
        self.assertIn("[info] model-acquisition", output)
        self.assertIn("the trainer downloads example itself before the first step", output)
        self.assertIn("unless the Hugging Face cache local runs mount already holds it", output)
        self.assertNotIn("estimated model downloads write 0 B", output)
        self.assertIn("[info] disk", output)
        self.assertNotIn("Disk warnings", output)

    def test_run_plan_preflight_bytes_preserve_small_units(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "small-download-plan"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": "small-download-plan",
                        "type": "train",
                                                "model": {"base": "custom"},
                        "safety": {"large_model_download_gb": 1},
                        "backend": {"name": "musubi-tuner", "config": {
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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
        payload = _estimate_backend_download_bytes(
            {
                "type": "train",
                "backend": {"name": "musubi-tuner"},
                "backend_overrides": True,
            }
        )
        self.assertEqual(payload["bytes"], 0)
        self.assertIn("invalid backend model download spec", payload["unknown"])

    def test_runpod_plan_counts_remote_downloads_even_when_local_cache_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            (root / "datasets" / "tiny").mkdir(parents=True)
            run_dir = root / "runs" / "remote"
            run_dir.mkdir(parents=True)
            cache_file = root / "cache" / "models" / "musubi" / "repo--model" / "dit" / "weights.safetensors"
            cache_file.parent.mkdir(parents=True)
            cache_file.write_bytes(b"x" * 100)
            run = {
                "id": "remote",
                "type": "train",
                                "model": {"base": "repo/model"},
                "datasets": [{"id": "tiny"}],
                "recipe": {"steps": 1},
                "compute": {"executor": "runpod"},
                "backend": {"name": "musubi-tuner", "config": {"architecture": "flux2", "model_bundle": "none", "model_downloads": {"dit": {"repo": "repo/model", "filename": "weights.safetensors"}}}},
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

    def _plan_cached_musubi_model(self, *, link_target: str, hf_cache: Path | None = None) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config: dict = {"schema_version": 2}
            if hf_cache is not None:
                config["docker"] = {"hf_cache": str(hf_cache)}
            (root / "workspace.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
            (root / "datasets" / "tiny").mkdir(parents=True)
            run_dir = root / "runs" / "local"
            run_dir.mkdir(parents=True)
            target = (hf_cache or root / "cache" / "huggingface") / "hub" / "models--repo--model" / "snapshots" / "abc" / "weights.safetensors"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"x" * 123)
            cache_file = root / "cache" / "models" / "musubi" / "repo--model" / "dit" / "weights.safetensors"
            cache_file.parent.mkdir(parents=True)
            cache_file.symlink_to(link_target)
            run = {
                "id": "local",
                "type": "train",
                "model": {"base": "repo/model"},
                "datasets": [{"id": "tiny"}],
                "recipe": {"steps": 1},
                "compute": {"executor": "docker"},
                "backend": {"name": "musubi-tuner", "config": {"architecture": "flux2", "model_bundle": "none", "model_downloads": {"dit": {"repo": "repo/model", "filename": "weights.safetensors"}}}},
            }
            (run_dir / "run.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "ok", "size_bytes": 200}):
                    return plan_run("local")["model_downloads"]
            finally:
                os.chdir(previous)

    @posix_only(POSIX_PATHS)
    def test_local_plan_follows_a_link_written_through_the_legacy_cache_target(self) -> None:
        downloads = self._plan_cached_musubi_model(link_target="/root/.cache/huggingface/hub/models--repo--model/snapshots/abc/weights.safetensors")
        self.assertEqual((downloads["bytes"], downloads["cached_bytes"]), (0, 123), downloads)

    @posix_only(POSIX_PATHS)
    def test_local_plan_counts_a_model_cached_outside_the_workspace(self) -> None:
        relative = "../../../../huggingface/hub/models--repo--model/snapshots/abc/weights.safetensors"
        with tempfile.TemporaryDirectory() as elsewhere:
            downloads = self._plan_cached_musubi_model(link_target=relative, hf_cache=Path(elsewhere).resolve() / "hf")
        self.assertEqual((downloads["bytes"], downloads["cached_bytes"]), (0, 123), downloads)
        self.assertTrue(downloads["items"][0]["cached"])

    def test_local_plan_treats_unmapped_absolute_symlink_as_not_cached(self) -> None:
        downloads = self._plan_cached_musubi_model(link_target="/opt/elsewhere/weights.safetensors")
        self.assertEqual((downloads["bytes"], downloads["cached_bytes"]), (200, 0), downloads)
        self.assertFalse(downloads["items"][0]["cached"])

    def test_runpod_disk_preflight_counts_downloads_and_checkpoints(self) -> None:
        run = {
            "type": "train",
                        "recipe": {"steps": 20},
            "backend": {"name": "musubi-tuner", "config": {"save_every_n_steps": 10}},
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
                        "recipe": {"steps": 3000},
            "backend": {"name": "musubi-tuner", "config": {"save_every_n_steps": 100}},
        }
        with self.assertRaisesRegex(ValueError, "may create about 30 checkpoints"):
            _checkpoint_safety_preflight(run)
        run["backend"]["config"]["prune_checkpoints_before_step"] = 1000
        _checkpoint_safety_preflight(run)

    def test_checkpoint_safety_preflight_counts_common_recipe_steps(self) -> None:
        run = {
            "type": "train",
                        "recipe": {"steps": 3000},
            "backend": {"name": "musubi-tuner", "config": {"save_every_n_steps": 100}},
        }
        with self.assertRaisesRegex(ValueError, "may create about 30 checkpoints"):
            _checkpoint_safety_preflight(run)

    def test_checkpoint_safety_preflight_accepts_musubi_keep_last_policy(self) -> None:
        run = {
            "type": "train",
                        "recipe": {"steps": 3000},
            "backend": {"name": "musubi-tuner", "config": {
                    "save_every_n_steps": 100,
                    "save_last_n_steps": 300,
                }
            },
        }
        _checkpoint_safety_preflight(run)
        run["backend"]["config"].pop("save_last_n_steps")
        run["backend"]["config"]["extra_args"] = ["--save_last_n_epochs=2"]
        _checkpoint_safety_preflight(run)

    def test_checkpoint_safety_preflight_can_be_explicitly_overridden(self) -> None:
        run = {
            "type": "train",
                        "recipe": {"steps": 3000},
            "backend": {"name": "musubi-tuner", "config": {"save_every_n_steps": 100}},
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
    def test_runpod_render_launch_forwards_explicit_yes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "render-1"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("type: render\n", encoding="utf-8")
            (run_dir / "status.json").write_text('{"state": "compiled"}', encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.launch.launch_render_runpod", return_value=0) as launch, \
                        patch("kura.runner.ensure_runner", return_value=False), patch("kura.runner.follow", return_value=0), \
                        patch("sys.stdout", new_callable=io.StringIO) as stdout, patch("sys.stderr", new_callable=io.StringIO):
                    code = cmd_render_launch(argparse.Namespace(run_id="render-1", executor="runpod", dry_run=False, yes=True))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            self.assertIn("completed  exit 0", stdout.getvalue())
            # The command checks and confirms billing; the runner launches from the request.
            self.assertTrue(launch.call_args.kwargs["check_only"])
            self.assertTrue(launch.call_args.kwargs["yes"])
            self.assertEqual(launch.call_args.kwargs["max_lease_sec"], 12 * 3600)
            request = json.loads(next((run_dir / "requests").glob("*.launch.json")).read_text(encoding="utf-8"))
            self.assertEqual((request["executor"], request["options"]["max_lease_sec"]), ("render-runpod", 12 * 3600))
            self.assertTrue(request["billing_confirmed_at"])

    def test_render_launch_notifies_on_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "render-1"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("type: render\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.launch.launch_render", return_value=0) as launch, patch("kura.run_commands.launch._notify") as notify, patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    code = launch_run("render-1", executor="local", dry_run=False, notify_channels="ntfy")
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            self.assertIn("completed  exit 0", stdout.getvalue())
            self.assertIn("rendered", stdout.getvalue())
            launch.assert_called_once()
            notify.assert_called_once()
            self.assertIn("completed", notify.call_args.kwargs["subject"])

    def test_render_dry_run_failure_does_not_notify(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
                "images:\n"
                "  comfyui: remote/comfy\n"
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
                f"comfyui:\n  lora_dir: {lora_dir}\n  lora_stage_subdir: Kura_tmp\n  lora_stage_cleanup: remove_after_render\n  model_registry: {{checkpoints: {{base.safetensors: {{repo: example/repo}}}}}}\n  runpod: {{container_disk_gb: 200}}\n  local_note: should-not-freeze\n  custom: {{nested: private}}\n",
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
                    "evaluation": {
                        "category": "custom_family_test",
                        "future_field": {"preserved": True},
                    },
                    "workflow_patches": {"prompt": {"node": "6", "field": "inputs.text"}, "negative_prompt": {"node": "7", "field": "inputs.text"}, "seed": {"node": "3", "field": "inputs.seed"}, "lora": {"node": "12", "field": "inputs.lora_name"}},
                    "render": {"output_dir": "samples/images", "timeout_sec": 5, "default_seed": None},
                }),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            compile_render(root, run_dir)
            manifest = yaml.safe_load((run_dir / "resolved" / "manifest.lock.yaml").read_text(encoding="utf-8"))
            self.assertEqual(set(manifest["comfyui"]), {"lora_dir", "lora_stage_subdir", "lora_stage_cleanup"})
            self.assertEqual(
                manifest["evaluation"],
                {"category": "custom_family_test", "future_field": {"preserved": True}},
            )
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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            image_record = json.loads((run_dir / "samples" / "images.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(
                image_record["checkpoint_application"],
                {
                    "kind": "lora_insert",
                    "class_type": "LoraLoader",
                    "strength_model": 0.8,
                    "strength_clip": 0.8,
                },
            )

    def test_insert_lora_loader_skips_empty_lora_name(self) -> None:
        workflow = {"1": {"inputs": {"model": ["2", 0]}}, "2": {"inputs": {}}}
        insertion = {"class_type": "LoraLoaderModelOnly", "model_node": "2", "model_output": 0}
        self.assertEqual(insert_lora_loader(workflow, insertion, ""), workflow)
        self.assertEqual(
            checkpoint_application({"lora_insert": insertion}, workflow, lora_name=""),
            {"kind": "none"},
        )

    def test_empty_checkpoint_dry_run_and_image_record_report_no_lora_application(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "render-1"
            resolved = run_dir / "resolved"
            resolved.mkdir(parents=True)
            workflow = {
                "3": {"class_type": "KSampler", "inputs": {"seed": 0, "model": ["4", 0]}},
                "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "base.safetensors"}},
                "6": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                "7": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            }
            insertion = {
                "class_type": "LoraLoaderModelOnly",
                "model_node": "4",
                "model_output": 0,
                "strength_model": 0.8,
            }
            manifest = {
                "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                "executor": {"name": "local"},
                "inputs": {
                    "checkpoint": {"path": "", "hash": None},
                    "workflow": {"path": "workflows/wf.json", "digest": "sha256:workflow"},
                    "promptset": {"path": "promptsets/prompts.jsonl", "digest": "sha256:prompts"},
                },
                "workflow_patches": {
                    "prompt": {"node": "6", "field": "inputs.text"},
                    "negative_prompt": {"node": "7", "field": "inputs.text"},
                    "seed": {"node": "3", "field": "inputs.seed"},
                },
                "lora_insert": insertion,
                "render": {"output_dir": "samples/images", "timeout_sec": 5},
            }
            (resolved / "manifest.lock.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
            (resolved / "workflow_used.json").write_text(json.dumps(workflow), encoding="utf-8")
            (resolved / "promptset_used.jsonl").write_text(
                json.dumps({"id": "p1", "prompt": "hello", "negative_prompt": "", "seeds": [123]}) + "\n",
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")

            with contextlib.redirect_stdout(io.StringIO()) as stdout:
                self.assertEqual(launch_render(root, run_dir, dry_run=True), 0)
            self.assertEqual(json.loads(stdout.getvalue())["checkpoint_application"], {"kind": "none"})

            class FakeClient:
                def __init__(self, endpoint: str, timeout: int) -> None:
                    pass

                def queue(self, queued: dict[str, Any]) -> str:
                    self.assert_no_lora_loader(queued)
                    return "prompt-1"

                @staticmethod
                def assert_no_lora_loader(queued: dict[str, Any]) -> None:
                    self_types = {node.get("class_type") for node in queued.values() if isinstance(node, dict)}
                    if "LoraLoaderModelOnly" in self_types:
                        raise AssertionError("empty checkpoint inserted a LoRA loader")

                def wait(self, prompt_id: str) -> list[dict[str, Any]]:
                    return [{"filename": "image.png", "subfolder": "", "type": "output"}]

                def download(self, image: dict[str, Any]) -> bytes:
                    return b"png"

            with patch("kura.render.ComfyUIClient", FakeClient):
                self.assertEqual(launch_render(root, run_dir), 0)
            record = json.loads((run_dir / "samples" / "images.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(record["checkpoint_application"], {"kind": "none"})

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
                    "render": {"default_seed": None, "workflow_fixed": ["prompt", "negative_prompt", "seed"]},
                }),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "draft"}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model_only, model_clip, full, LoraLoaderModelOnly, LoraLoader"):
                compile_render(root, run_dir)

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
            (root / "promptsets" / "prompts.jsonl").write_text(json.dumps({"id": "p1"}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "type": "render",
                    "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/wf.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "runpod"},
                    "workflow_patches": {},
                    "render": {"default_seed": None, "workflow_fixed": ["prompt", "negative_prompt", "seed"]},
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
            (root / "promptsets" / "prompts.jsonl").write_text(json.dumps({"id": "p1"}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "type": "render",
                    "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/wf.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "runpod"},
                    "workflow_patches": {},
                    "render": {"default_seed": None, "workflow_fixed": ["prompt", "negative_prompt", "seed"]},
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
            (root / "promptsets" / "prompts.jsonl").write_text(json.dumps({"id": "p1"}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "type": "render",
                    "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/samples/toy/toy-text2image-api.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "runpod"},
                    "workflow_patches": {},
                    "render": {"default_seed": None, "workflow_fixed": ["prompt", "negative_prompt", "seed"]},
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
            (root / "promptsets" / "prompts.jsonl").write_text(json.dumps({"id": "p1"}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "type": "render",
                    "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/samples/toy/toy_text2image_api.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "runpod"},
                    "workflow_patches": {},
                    "render": {"default_seed": None, "workflow_fixed": ["prompt", "negative_prompt", "seed"]},
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
            (root / "promptsets" / "prompts.jsonl").write_text(json.dumps({"id": "p1"}) + "\n", encoding="utf-8")
            (run_dir / "run.yaml").write_text(
                yaml.safe_dump({
                    "type": "render",
                    "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/samples/toy/toy-text2image-api.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                    "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                    "executor": {"name": "runpod"},
                    "workflow_patches": {},
                    "render": {"default_seed": None, "workflow_fixed": ["prompt", "negative_prompt", "seed"]},
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
                    "images:\n"
                    "  comfyui: remote/default-comfy\n"
                    "runpod:\n"
                    "  storage_mode: upload\n"
                    "  gpu_type_ids: [NVIDIA RTX A5000]\n"
                    "comfyui:\n"
                    "  runpod:\n"
                    "    ports: [22/tcp]\n",
                    encoding="utf-8",
                )
                (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
                frozen_image = run_dir / "resolved" / "images" / "control_image" / "p1.png"
                frozen_image.parent.mkdir(parents=True)
                frozen_image.write_bytes(b"control-image")
                frozen_name = "resolved/images/control_image/p1.png"
                frozen_digest = "sha256:" + hashlib.sha256(frozen_image.read_bytes()).hexdigest()
                (run_dir / "resolved" / "cases.jsonl").write_text(
                    json.dumps({"id": "p1", "index": 1, "values": {"control_image": frozen_name}}) + "\n",
                    encoding="utf-8",
                )
                (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                    yaml.safe_dump({
                        "type": "render",
                        "inputs": {"checkpoint": {"path": ""}, "workflow": {"path": "workflows/wf.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                        "generator": {"name": "comfyui", "endpoint": "http://127.0.0.1:8188"},
                        "executor": {"name": "runpod"},
                        "workflow_patches": {"control_image": {"node": "2", "field": "inputs.image", "type": "image"}},
                        "render": {"default_seed": 1},
                        "promptset_images": [{"patch": "control_image", "prompt_id": "p1", "source": "/authored/control.png", "resolved": frozen_name, "digest": frozen_digest}],
                        "comfyui_model_registry": {"checkpoints": {"toy.safetensors": {"repo": "owner/toy", "filename": "toy.safetensors"}}},
                        "comfyui_models": [{"name": "toy.safetensors", "repo": "owner/toy", "filename": "toy.safetensors", "target_dir": "checkpoints"}],
                    }),
                    encoding="utf-8",
                )
                buffer = io.StringIO()
                availability = {
                    "status": "ok",
                    "checked_at": "2026-07-16T12:00:00+09:00",
                    "gpu_count": 1,
                    "candidates": [
                        {
                            "gpu_type_id": "NVIDIA RTX A5000",
                            "display_name": "RTX A5000",
                            "memory_gb": 24,
                            "clouds": [
                                {
                                    "cloud_type": "SECURE",
                                    "stock_status": "Low",
                                    "available": True,
                                    "price_per_hour": 0.29,
                                    "available_gpu_counts": [1],
                                }
                            ],
                        }
                    ],
                }
                with contextlib.redirect_stdout(buffer), patch(
                    "kura.run_commands.render_runpod.runpod_gpu_availability",
                    return_value=availability,
                ) as price_probe:
                    code = launch_run("render-1", executor="runpod", dry_run=True)
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            plan = json.loads(buffer.getvalue())
            self.assertEqual(plan["image"], "remote/default-comfy")
            self.assertEqual(plan["executor"], "runpod")
            self.assertEqual(plan["models"][0]["repo"], "owner/toy")
            self.assertEqual(plan["input_images"][0]["frozen"], frozen_name)
            self.assertTrue(plan["input_images"][0]["name"].startswith("Kura_tmp/"))
            self.assertEqual(plan["input_images"][0]["bytes"], len(b"control-image"))
            self.assertEqual(plan["input_image_bytes"], len(b"control-image"))
            price_probe.assert_called_once()
            self.assertEqual(plan["billing"]["gpu_candidates"][0]["display_name"], "RTX A5000")
            self.assertEqual(plan["billing"]["gpu_candidates"][0]["clouds"][0]["price_per_hour"], 0.29)
            self.assertEqual(plan["billing"]["price_checked_at"], "2026-07-16T12:00:00+09:00")
            self.assertEqual(plan["billing"]["maximum_lease"], "12h")
            self.assertEqual(plan["billing"]["maximum_lease_sec"], 12 * 3600)

    def test_runpod_render_rejects_tampered_frozen_image_before_billing_probe(self) -> None:
        from kura.run_commands.render_runpod import _render_runpod_images

        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "render-1"
            frozen_image = run_dir / "resolved" / "images" / "control_image" / "p1.png"
            frozen_image.parent.mkdir(parents=True)
            frozen_image.write_bytes(b"tampered")
            frozen_name = "resolved/images/control_image/p1.png"
            (run_dir / "resolved" / "cases.jsonl").write_text(
                json.dumps({"id": "p1", "index": 1, "values": {"control_image": frozen_name}}) + "\n",
                encoding="utf-8",
            )
            frozen = {
                "workflow_patches": {"control_image": {"node": "2", "field": "inputs.image", "type": "image"}},
                "promptset_images": [{
                    "patch": "control_image",
                    "prompt_id": "p1",
                    "source": "/authored/control.png",
                    "resolved": frozen_name,
                    "digest": "sha256:" + hashlib.sha256(b"original").hexdigest(),
                }],
            }

            with self.assertRaisesRegex(ValueError, "does not match its manifest digest"):
                _render_runpod_images(run_dir, frozen)

    def test_runpod_render_rejects_frozen_image_symlink_escape_before_pod_creation(self) -> None:
        from kura.run_commands.render_runpod import _render_runpod_images

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "render-1"
            frozen_image = run_dir / "resolved" / "images" / "control_image" / "p1.png"
            frozen_image.parent.mkdir(parents=True)
            outside = root / "outside.png"
            outside.write_bytes(b"outside")
            frozen_image.symlink_to(outside)
            frozen_name = "resolved/images/control_image/p1.png"
            (run_dir / "resolved" / "cases.jsonl").write_text(
                json.dumps({"id": "p1", "index": 1, "values": {"control_image": frozen_name}}) + "\n",
                encoding="utf-8",
            )
            frozen = {
                "workflow_patches": {"control_image": {"node": "2", "field": "inputs.image", "type": "image"}},
                "promptset_images": [{
                    "patch": "control_image",
                    "prompt_id": "p1",
                    "source": "/authored/control.png",
                    "resolved": frozen_name,
                    "digest": "sha256:" + hashlib.sha256(outside.read_bytes()).hexdigest(),
                }],
            }

            with self.assertRaisesRegex(ValueError, "missing or not a regular file"):
                _render_runpod_images(run_dir, frozen)

    def test_runpod_render_uploads_frozen_images_before_launching_cases(self) -> None:
        from kura.run_commands.render_runpod import launch_render_runpod

        class FakeTunnel:
            def terminate(self) -> None:
                pass

            def wait(self, timeout: int) -> int:
                return 0

            def kill(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            previous = Path.cwd()
            try:
                os.chdir(root)
                run_dir = root / "runs" / "render-1"
                resolved = run_dir / "resolved"
                resolved.mkdir(parents=True)
                (root / "workspace.yaml").write_text(
                    "images:\n"
                    "  comfyui: remote/comfy\n"
                    "runpod:\n"
                    "  storage_mode: upload\n"
                    "  gpu_type_ids: [NVIDIA RTX A5000]\n"
                    "comfyui:\n"
                    "  runpod:\n"
                    "    ports: [22/tcp]\n",
                    encoding="utf-8",
                )
                (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
                frozen_image = resolved / "images" / "control_image" / "p1.png"
                frozen_image.parent.mkdir(parents=True)
                frozen_image.write_bytes(b"control-image")
                frozen_name = "resolved/images/control_image/p1.png"
                image_digest = "sha256:" + hashlib.sha256(frozen_image.read_bytes()).hexdigest()
                (resolved / "cases.jsonl").write_text(
                    json.dumps({"id": "p1", "index": 1, "values": {"control_image": frozen_name}}) + "\n",
                    encoding="utf-8",
                )
                (resolved / "workflow_used.json").write_text("{}", encoding="utf-8")
                (resolved / "comfyui_model_registry.json").write_text("{}", encoding="utf-8")
                (resolved / "manifest.lock.yaml").write_text(
                    yaml.safe_dump({
                        "type": "render",
                        "inputs": {"workflow": {"path": "workflows/wf.json"}, "promptset": {"path": "promptsets/prompts.jsonl"}},
                        "generator": {"name": "comfyui", "endpoint": ""},
                        "executor": {"name": "runpod"},
                        "workflow_patches": {"control_image": {"node": "2", "field": "inputs.image", "type": "image"}},
                        "render": {"timeout_sec": 5},
                        "promptset_images": [{"patch": "control_image", "prompt_id": "p1", "source": "/authored/control.png", "resolved": frozen_name, "digest": image_digest}],
                        "comfyui_model_registry": {},
                        "comfyui_models": [],
                    }),
                    encoding="utf-8",
                )
                details = {"pod_id": "pod-1", "ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}
                completed = subprocess.CompletedProcess(["ssh"], 0, "", "")
                with patch("kura.run_commands.render_runpod.launch_runpod_session"), patch(
                    "kura.run_commands.render_runpod._runpod_ssh_details", return_value=details,
                ), patch("kura.run_commands.render_runpod._start_runpod_session_lease_guard"), patch(
                    "kura.run_commands.render_runpod.subprocess.run", return_value=completed,
                ) as remote_prepare, patch(
                    "kura.run_commands.render_runpod._scp_to_runpod",
                ) as scp, patch("kura.run_commands.render_runpod._start_runpod_comfyui"), patch(
                    "kura.run_commands.render_runpod._free_local_port", return_value=18888,
                ), patch("kura.run_commands.render_runpod.subprocess.Popen", return_value=FakeTunnel()), patch(
                    "kura.run_commands.render_runpod._wait_http_ready",
                ), patch("kura.run_commands.render_runpod.launch_render", return_value=0) as render, patch(
                    "kura.run_commands.render_runpod._sync_runpod_remote_stdout",
                ), patch("kura.run_commands.render_runpod.stop_runpod"), patch("kura.run_commands.render_runpod.record_pod_lease_deadline"), patch("kura.run_commands.render_runpod._notify"):
                    self.assertEqual(launch_render_runpod("render-1", dry_run=False, yes=True), 0)
            finally:
                os.chdir(previous)

            prepare_script = remote_prepare.call_args_list[0].args[0][-1]
            self.assertIn("/opt/ComfyUI/input/Kura_tmp", prepare_script)
            image_uploads = [
                call for call in scp.call_args_list
                if call.args[1] == frozen_image
            ]
            self.assertEqual(len(image_uploads), 1)
            remote_image = image_uploads[0].args[2]
            self.assertTrue(remote_image.startswith("/opt/ComfyUI/input/Kura_tmp/"))
            mapping = render.call_args.kwargs["image_name_overrides"]
            self.assertEqual(mapping, {frozen_name: remote_image.removeprefix("/opt/ComfyUI/input/")})

    def test_runpod_render_launch_requires_runpod_compiled_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            previous = Path.cwd()
            try:
                os.chdir(root)
                (root / "runs" / "render-1" / "resolved").mkdir(parents=True)
                run_dir = root / "runs" / "render-1"
                (root / "workspace.yaml").write_text(
                    "images:\n"
                    "  comfyui: remote/comfy\n"
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

            _materialize_stage(plan)
            _cleanup_stage(plan)

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
                _materialize_stage(plan)

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
            # The cursor is the sync's own bookkeeping and stays out of status.
            self.assertNotIn("remote_log_bytes", status)
            self.assertEqual(json.loads((run_dir / "logs" / "stdout.remote-cursor.json").read_text(encoding="utf-8"))["remote_bytes"], 77)
            self.assertIn("avr_loss=0.123", (run_dir / "logs" / "stdout.log").read_text(encoding="utf-8"))

    def test_concurrent_remote_stdout_sync_does_not_duplicate_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "remote-run"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            first_remote_call = threading.Event()
            observed_offsets: list[int] = []

            def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
                script = command[-1]
                offset = int(script.split("offset=", 1)[1].splitlines()[0])
                observed_offsets.append(offset)
                if len(observed_offsets) == 1:
                    first_remote_call.set()
                    time.sleep(0.05)
                payload = b"one line\n" if offset == 0 else b""
                return subprocess.CompletedProcess(command, 0, payload + b"\n__KURA_LOG_SIZE__:9\n", b"")

            results: list[bool] = []
            details = {"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}
            with patch("kura.cli.subprocess.run", side_effect=fake_run):
                first = threading.Thread(target=lambda: results.append(_sync_runpod_remote_stdout(run_dir, details, workspace="/workspace", run_id="remote-run")))
                second = threading.Thread(target=lambda: results.append(_sync_runpod_remote_stdout(run_dir, details, workspace="/workspace", run_id="remote-run")))
                first.start()
                self.assertTrue(first_remote_call.wait(timeout=1))
                second.start()
                # Windows file locks wait in steps of about a second, so the second sync needs time.
                first.join(timeout=10)
                second.join(timeout=10)

            self.assertEqual(results, [True, True])
            self.assertEqual(observed_offsets, [0, 9])
            self.assertEqual((run_dir / "logs" / "stdout.log").read_text(encoding="utf-8"), "one line\n")

    def test_download_lock_rejects_a_second_controller(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with _run_operation_lock(run_dir, "download", blocking=False):
                    stderr = io.StringIO()
                    with patch("sys.stderr", stderr):
                        code = cmd_run_download(argparse.Namespace(run_id="example", force=True))
            finally:
                os.chdir(previous)

            self.assertEqual(code, 1)
            self.assertIn("another download operation is already active", stderr.getvalue())

    def test_remote_exit_observation_records_recovery_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "status.json").write_text(
                json.dumps({"state": "running", "last_realization": "realizations/r1.json"}),
                encoding="utf-8",
            )

            _record_remote_exit_observation(
                run_dir,
                {"event": "remote_exit", "exit_code": 0, "timestamp": "2026-01-01T00:00:00+00:00"},
            )

            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "running")
            self.assertEqual(status["remote_state"], "completed")
            self.assertEqual(status["remote_exit_code"], 0)
            # Waiting for collection is not needing a person.
            self.assertNotIn("recovery_required", status)
            observation = run_dir / status["last_remote_exit_observation"]
            self.assertEqual(json.loads(observation.read_text(encoding="utf-8"))["event"], "remote_exit_observed")
            events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[-1]["event"], "remote_exit_observed")

    def test_remote_exit_observation_survives_convenience_log_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "logs" / "events.jsonl").mkdir(parents=True)
            (run_dir / "status.json").write_text(
                json.dumps({"state": "running", "last_realization": "realizations/r1.json"}),
                encoding="utf-8",
            )

            stderr = io.StringIO()
            with patch("sys.stderr", stderr):
                _record_remote_exit_observation(
                    run_dir,
                    {"event": "remote_exit", "exit_code": 0, "timestamp": "2026-01-01T00:00:00+00:00"},
                )

            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["remote_state"], "completed")
            # Waiting for collection is not needing a person.
            self.assertNotIn("recovery_required", status)
            self.assertIn("could not append convenience event log", stderr.getvalue())

    def test_image_preflight_names_the_image_and_warns_only_for_a_mutable_override(self) -> None:
        run = {"backend": {"name": "ai-toolkit"}}
        pinned = _image_preflight_report(run, {})
        self.assertEqual([record["severity"] for record in pinned], ["info"])
        self.assertIn("pinned by Kura", pinned[0]["fact"])
        mutable = _image_preflight_report(run, {"images": {"ai-toolkit": "ostris/aitoolkit:latest"}})
        self.assertEqual([record["severity"] for record in mutable], ["info", "warning"])
        self.assertIn("mutable tag", mutable[1]["fact"])
        digest = _image_preflight_report(run, {"images": {"ai-toolkit": "example/ai@sha256:" + "0" * 64}})
        self.assertEqual([record["severity"] for record in digest], ["info"])


class RunPodPullSelectionTests(unittest.TestCase):
    @staticmethod
    def _fake_safetensors() -> bytes:
        header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}, separators=(",", ":")).encode()
        return len(header).to_bytes(8, "little") + header + b"\x00\x00\x00\x00"

    def test_duration_parser_accepts_common_suffixes(self) -> None:
        self.assertEqual(_parse_duration_seconds("30m"), 1800)
        self.assertEqual(_parse_duration_seconds("2h"), 7200)
        self.assertEqual(_parse_duration_seconds("45"), 45)
        self.assertEqual(_parse_duration_seconds(None), 0)

    def test_terminal_manifest_rejects_unsafe_and_duplicate_paths(self) -> None:
        valid = {"size": 1, "mtime_ns": 1, "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "unsafe path"):
            _validated_snapshot_manifest([{"path": "../outputs/model.safetensors", **valid}])
        with self.assertRaisesRegex(ValueError, "duplicate path"):
            _validated_snapshot_manifest([
                {"path": "outputs/model.safetensors", **valid},
                {"path": "outputs/model.safetensors", **valid},
            ])

    def test_same_size_checkpoint_with_wrong_hash_is_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            output = run_dir / "outputs" / "model-step00000100.safetensors"
            output.parent.mkdir(parents=True)
            payload = self._fake_safetensors()
            output.write_bytes(payload)
            (run_dir / "status.json").write_text(
                json.dumps({
                    "mirrored_outputs": [{
                        "name": output.name,
                        "path": f"outputs/{output.name}",
                        "size": len(payload),
                        "remote_path": f"/workspace/runs/example/outputs/{output.name}",
                        "remote_mtime_ns": 1,
                    }]
                }),
                encoding="utf-8",
            )
            item = {
                "path": "outputs/model-step00000100.safetensors",
                "size": len(payload),
                "mtime_ns": 1,
                "sha256": hashlib.sha256(payload[:-1] + b"\x01").hexdigest(),
            }

            self.assertIsNone(_local_reusable_snapshot_source(run_dir, item, remote_root="/workspace/runs/example"))

    def test_unpublished_output_and_raw_state_file_are_not_reused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            output = run_dir / "outputs" / "model-step00000100.safetensors"
            raw_state = run_dir / "outputs" / "example-step00000100-state" / "model.safetensors"
            raw_state.parent.mkdir(parents=True)
            payload = self._fake_safetensors()
            output.write_bytes(payload)
            raw_state.write_bytes(payload)
            (run_dir / "status.json").write_text("{}", encoding="utf-8")
            common = {
                "size": len(payload),
                "mtime_ns": 1,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

            self.assertIsNone(_local_reusable_snapshot_source(
                run_dir,
                {"path": f"outputs/{output.name}", **common},
                remote_root="/workspace/runs/example",
            ))
            self.assertIsNone(_local_reusable_snapshot_source(
                run_dir,
                {"path": "outputs/example-step00000100-state/model.safetensors", **common},
                remote_root="/workspace/runs/example",
            ))

    def test_corrupt_delta_archive_is_reported_as_download_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "delta.tar.gz"
            archive.write_bytes(b"not a tar archive")
            with self.assertRaisesRegex(ValueError, "invalid delta snapshot archive"):
                _extract_snapshot_delta_archive(
                    archive,
                    Path(directory) / "destination",
                    run_id="example",
                    expected_paths={"example/outputs/model.safetensors"},
                )

    def test_snapshot_copy_fallback_checks_space_for_reused_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.bin"
            target = root / "snapshot" / "target.bin"
            source.write_bytes(b"data")
            target.parent.mkdir()

            with patch("kura.run_commands.runpod_ssh.os.link", side_effect=OSError("cross-device link")), \
                 patch("kura.run_commands.runpod_ssh.ensure_free_bytes") as ensure_free:
                _link_or_copy_snapshot_file(
                    source,
                    target,
                    free_space_root=root,
                    required_free_bytes=123,
                    config={"storage": {"host_drive": "D"}},
                )

            ensure_free.assert_called_once_with(
                root,
                127,
                context="RunPod delta download reusable copy fallback",
                config={"storage": {"host_drive": "D"}},
            )
            self.assertEqual(target.read_bytes(), b"data")

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

    def test_remote_output_version_requires_stable_path_size_and_mtime(self) -> None:
        before = {"path": "/workspace/model.safetensors", "size": 10, "mtime_ns": 20}
        self.assertTrue(_same_remote_output_version(before, dict(before)))
        self.assertFalse(_same_remote_output_version(before, {**before, "size": 11}))
        self.assertFalse(_same_remote_output_version(before, {**before, "mtime_ns": 21}))
        self.assertFalse(_same_remote_output_version(before, None))

    def test_checkpoint_sync_failure_is_recorded_without_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            with patch("kura.run_commands.runpod_ssh._runpod_remote_outputs", side_effect=ValueError("network interrupted")):
                synced = _try_sync_runpod_checkpoints(run_dir, {"ip": "host", "port": 22, "key": "key"}, workspace="/workspace", run_id="example")

            self.assertFalse(synced)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "running")
            self.assertIn("network interrupted", status["checkpoint_sync_error"])
            self.assertNotIn("training_state_sync_error", status)

    def test_training_state_sync_failure_does_not_report_checkpoint_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump({
                "id": "example",
                "backend": {"name": "musubi-tuner", "config": {}},
                "recipe": {"steps": 10, "seed": 1},
                "recovery": {"training_state": {"enabled": True, "keep_generations": 2}},
            }), encoding="utf-8")
            with patch("kura.run_commands.runpod_ssh._runpod_remote_outputs", return_value=[]), patch(
                "kura.run_commands.runpod_ssh._runpod_remote_training_states", side_effect=ValueError("state listing interrupted")
            ):
                synced = _try_sync_runpod_checkpoints(
                    run_dir,
                    {"ip": "host", "port": 22, "key": "key"},
                    workspace="/workspace",
                    run_id="example",
                )

            self.assertFalse(synced)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertIn("state listing interrupted", status["training_state_sync_error"])
            self.assertNotIn("checkpoint_sync_error", status)

    def test_training_state_status_record_failure_is_not_labeled_as_checkpoint_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump({
                "id": "example",
                "backend": {"name": "musubi-tuner", "config": {}},
                "recipe": {"steps": 10, "seed": 1},
                "recovery": {"training_state": {"enabled": True, "keep_generations": 2}},
            }), encoding="utf-8")
            with patch("kura.run_commands.runpod_ssh._runpod_remote_outputs", return_value=[]), patch(
                "kura.run_commands.runpod_ssh._runpod_remote_training_states", return_value=[]
            ), patch(
                "kura.run_commands.runpod_ssh._record_pulled_training_states", side_effect=OSError("state status interrupted")
            ):
                synced = _try_sync_runpod_checkpoints(
                    run_dir,
                    {"ip": "host", "port": 22, "key": "key"},
                    workspace="/workspace",
                    run_id="example",
                )

            self.assertFalse(synced)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertIn("state status interrupted", status["training_state_sync_error"])
            self.assertNotIn("checkpoint_sync_error", status)

    def test_checkpoint_sync_disk_shortage_is_nonfatal_and_visible(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            item = {"name": "model-step00000250.safetensors", "path": "/workspace/model.safetensors", "step": 250, "size": 10, "mtime_ns": 10}
            with patch("kura.run_commands.runpod_ssh._runpod_remote_outputs", return_value=[item]), \
                 patch("kura.run_commands.runpod_ssh._workspace_config", return_value={"safety": {}}), \
                 patch("kura.run_commands.runpod_ssh.ensure_free_bytes", side_effect=ValueError("insufficient free disk")):
                synced = _try_sync_runpod_checkpoints(run_dir, {"ip": "host", "port": 22, "key": "key"}, workspace="/workspace", run_id="example")

            self.assertFalse(synced)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "running")
            self.assertIn("insufficient free disk", status["checkpoint_sync_error"])

    def test_empty_successful_sync_clears_previous_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "checkpoint_sync_error": "old"}), encoding="utf-8")
            _record_pulled_outputs(run_dir, [])
            self.assertNotIn("checkpoint_sync_error", json.loads((run_dir / "status.json").read_text(encoding="utf-8")))

    def test_automatic_sync_skips_busy_manual_pull_without_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            with _run_operation_lock(run_dir, "checkpoint-pull"):
                self.assertTrue(_try_sync_runpod_checkpoints(run_dir, {}, workspace="/workspace", run_id="example"))
            self.assertNotIn("checkpoint_sync_error", json.loads((run_dir / "status.json").read_text(encoding="utf-8")))

    def test_operation_lock_does_not_relabel_protected_operation_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            with self.assertRaises(FileLockBusy) as raised:
                with _run_operation_lock(run_dir, "checkpoint-pull"):
                    raise FileLockBusy("inner operation lock failure")

            self.assertIs(type(raised.exception), FileLockBusy)
            self.assertEqual(str(raised.exception), "inner operation lock failure")

    def test_truncated_safetensors_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.safetensors"
            path.write_bytes((100).to_bytes(8, "little") + b"{}")
            with self.assertRaisesRegex(ValueError, "header size"):
                validate_safetensors_file(path)

    def test_safetensors_requires_dtype_and_shape(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.safetensors"
            header = json.dumps({"weight": {"data_offsets": [0, 4]}}).encode()
            path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * 4)
            with self.assertRaisesRegex(ValueError, "invalid safetensors tensor entry"):
                validate_safetensors_file(path)

    def test_safetensors_rejects_overlapping_or_gapped_data(self) -> None:
        # Each tensor has the right length for its dtype, so only the layout is wrong.
        for offsets, data_length in ((([0, 4], [2, 6]), 6), (([0, 4], [5, 9]), 9)):
            with self.subTest(offsets=offsets), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "broken.safetensors"
                header = json.dumps({
                    "first": {"dtype": "F32", "shape": [1], "data_offsets": offsets[0]},
                    "second": {"dtype": "F32", "shape": [1], "data_offsets": offsets[1]},
                }).encode()
                path.write_bytes(len(header).to_bytes(8, "little") + header + b"\0" * data_length)
                with self.assertRaisesRegex(ValueError, "not contiguous"):
                    validate_safetensors_file(path)

    def test_status_mutations_preserve_fields_across_threads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            threads = [
                threading.Thread(target=_mutate_run_status, args=(run_dir, lambda status: status.__setitem__("remote_log_bytes", 10))),
                threading.Thread(target=_mutate_run_status, args=(run_dir, lambda status: status.__setitem__("mirrored_outputs", [{"name": "step.safetensors"}]))),
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["remote_log_bytes"], 10)
            self.assertEqual(status["mirrored_outputs"], [{"name": "step.safetensors"}])

    def test_pull_verifies_then_atomically_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_dir = workspace / "runs" / "example"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            (workspace / "workspace.yaml").write_text("safety: {}\n", encoding="utf-8")
            payload = self._fake_safetensors()
            item = {"name": "model-step00000250.safetensors", "path": "/workspace/model.safetensors", "step": 250, "size": len(payload), "mtime_ns": 10}

            def fake_transfer(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                Path(command[-1]).write_bytes(payload)
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("kura.run_commands.runpod_ssh._workspace_config", return_value={"safety": {}}), \
                 patch("kura.run_commands.runpod_ssh.ensure_free_bytes"), \
                 patch("kura.run_commands.runpod_ssh._run_bounded", side_effect=fake_transfer), \
                 patch("kura.run_commands.runpod_ssh._runpod_remote_outputs", return_value=[dict(item)]):
                pulled = _pull_remote_output_items(run_dir, {"ip": "host", "port": 22, "key": "key"}, workspace="/workspace", items=[item])

            published = run_dir / pulled[0]["path"]
            self.assertEqual(published.read_bytes(), payload)
            self.assertFalse((published.parent / f".{published.name}.partial").exists())

            # The caller performs a final status merge after the per-file
            # publication. That merge must not duplicate the activity event.
            _record_pulled_outputs(run_dir, pulled, emit_event=False)
            events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([event["event"] for event in events], ["run_outputs_pulled"])

    def test_pull_refuses_when_the_wsl_backing_drive_is_short(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            item = {"name": "model-step00000250.safetensors", "path": "/workspace/model.safetensors", "step": 250, "size": 1024**3, "mtime_ns": 10}
            with _wsl_with_short_host_drive(linux_free_gib=900, host_free_gib=5), \
                 patch("kura.run_commands.runpod_ssh._workspace_config", return_value={"safety": {}}), \
                 patch("kura.run_commands.runpod_ssh._run_bounded") as transfer:
                with self.assertRaisesRegex(ValueError, "RunPod output pull needs about .* only 5 GiB is available on C:"):
                    _pull_remote_output_items(run_dir, {"ip": "host", "port": 22, "key": "key"}, workspace="/workspace", items=[item])
            transfer.assert_not_called()

    def test_pull_skips_only_valid_local_copy_with_matching_remote_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            destination = run_dir / "outputs"
            destination.mkdir(parents=True)
            payload = self._fake_safetensors()
            item = {"name": "model-step00000250.safetensors", "path": "/workspace/model.safetensors", "step": 250, "size": len(payload), "mtime_ns": 10}
            (destination / item["name"]).write_bytes(payload)
            (run_dir / "status.json").write_text(json.dumps({"mirrored_outputs": [{"name": item["name"], "remote_path": item["path"], "remote_mtime_ns": item["mtime_ns"]}]}), encoding="utf-8")
            with patch("kura.run_commands.runpod_ssh._workspace_config", return_value={"safety": {}}), \
                 patch("kura.run_commands.runpod_ssh._run_bounded") as transfer:
                pulled = _pull_remote_output_items(run_dir, {"ip": "host", "port": 22, "key": "key"}, workspace="/workspace", items=[item])
            transfer.assert_not_called()
            self.assertTrue(pulled[0]["skipped"])

    def test_pull_replaces_same_size_copy_when_remote_mtime_changed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            destination = run_dir / "outputs"
            destination.mkdir(parents=True)
            payload = self._fake_safetensors()
            item = {"name": "model-step00000250.safetensors", "path": "/workspace/model.safetensors", "step": 250, "size": len(payload), "mtime_ns": 11}
            (destination / item["name"]).write_bytes(payload)
            (run_dir / "status.json").write_text(json.dumps({"mirrored_outputs": [{"name": item["name"], "remote_path": item["path"], "remote_mtime_ns": 10}]}), encoding="utf-8")

            def fake_transfer(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                Path(command[-1]).write_bytes(payload)
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("kura.run_commands.runpod_ssh._workspace_config", return_value={"safety": {}}), \
                 patch("kura.run_commands.runpod_ssh.ensure_free_bytes"), \
                 patch("kura.run_commands.runpod_ssh._run_bounded", side_effect=fake_transfer) as transfer, \
                 patch("kura.run_commands.runpod_ssh._runpod_remote_outputs", return_value=[dict(item)]):
                pulled = _pull_remote_output_items(run_dir, {"ip": "host", "port": 22, "key": "key"}, workspace="/workspace", items=[item])
            transfer.assert_called_once()
            self.assertFalse(pulled[0]["skipped"])

    def test_pull_records_published_item_before_later_batch_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            payload = self._fake_safetensors()
            first = {"name": "model-step00000250.safetensors", "path": "/workspace/first.safetensors", "step": 250, "size": len(payload), "mtime_ns": 10}
            second = {"name": "model-step00000500.safetensors", "path": "/workspace/second.safetensors", "step": 500, "size": len(payload), "mtime_ns": 20}

            def transfer(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                if "first.safetensors" in command[-2]:
                    Path(command[-1]).write_bytes(payload)
                    return subprocess.CompletedProcess(command, 0, "", "")
                return subprocess.CompletedProcess(command, 1, "", "failed")

            with patch("kura.run_commands.runpod_ssh._workspace_config", return_value={"safety": {}}), \
                 patch("kura.run_commands.runpod_ssh.ensure_free_bytes"), \
                 patch("kura.run_commands.runpod_ssh._run_bounded", side_effect=transfer), \
                 patch("kura.run_commands.runpod_ssh._runpod_remote_outputs", return_value=[dict(first), dict(second)]):
                with self.assertRaisesRegex(ValueError, "00000500"):
                    _pull_remote_output_items(run_dir, {"ip": "host", "port": 22, "key": "key"}, workspace="/workspace", items=[first, second])

            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual([item["name"] for item in status["mirrored_outputs"]], [first["name"]])
            self.assertTrue((run_dir / "outputs" / first["name"]).is_file())

    def test_changed_remote_file_is_not_published(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            payload = self._fake_safetensors()
            item = {"name": "model-step00000250.safetensors", "path": "/workspace/model.safetensors", "step": 250, "size": len(payload), "mtime_ns": 10}

            def fake_transfer(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                Path(command[-1]).write_bytes(payload)
                return subprocess.CompletedProcess(command, 0, "", "")

            with patch("kura.run_commands.runpod_ssh._workspace_config", return_value={"safety": {}}), \
                 patch("kura.run_commands.runpod_ssh.ensure_free_bytes"), \
                 patch("kura.run_commands.runpod_ssh._run_bounded", side_effect=fake_transfer), \
                 patch("kura.run_commands.runpod_ssh._runpod_remote_outputs", return_value=[{**item, "mtime_ns": 11}]):
                with self.assertRaisesRegex(ValueError, "changed while"):
                    _pull_remote_output_items(run_dir, {"ip": "host", "port": 22, "key": "key"}, workspace="/workspace", items=[item])

            destination = run_dir / "outputs"
            self.assertFalse((destination / item["name"]).exists())
            self.assertFalse((destination / f".{item['name']}.partial").exists())

    def test_pulled_checkpoint_history_accumulates_across_syncs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "running"}), encoding="utf-8")
            _record_pulled_outputs(run_dir, [{"name": "model-step00000250.safetensors", "path": "outputs/model-step00000250.safetensors", "step": 250, "size": 10, "skipped": False}])
            _record_pulled_outputs(run_dir, [{"name": "model-step00000500.safetensors", "path": "outputs/model-step00000500.safetensors", "step": 500, "size": 10, "skipped": False}])

            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual([item["step"] for item in status["mirrored_outputs"]], [250, 500])


class AiToolkitBackendTests(unittest.TestCase):
    def _run(self) -> dict[str, object]:
        return {
            "id": "ai-toolkit-example",
            "type": "train",
            "backend": {"name": "ai-toolkit", "adapter_version": 1, "config": {"model_arch": "flux2_klein_4b", "network_dim": 4, "network_alpha": 4, "learning_rate": 1.0e-4, "batch_size": 1, "gradient_checkpointing": False, "optimizer_type": "adamw8bit", "quantize": False, "quantize_te": False, "low_vram": False}},
            "model": {"base": "black-forest-labs/FLUX.2-klein-base-4B"},
            "datasets": [{"id": "tiny", "digest": "sha256:abc"}],
            "recipe": {"steps": 1, "seed": 42},
        }

    def _write_frozen_projection(self, run: dict[str, object], destination: Path) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        report = freeze_fixture(run, destination.parent)
        return report["datasets"][0]["native"]["folder_path"]

    @posix_only(DATASET_IO)
    def test_default_compile_writes_runnable_yaml_and_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ai-toolkit"
            folder_path = self._write_frozen_projection(self._run(), destination)
            compile_ai_toolkit(self._run(), destination)
            config = yaml.safe_load((destination.with_suffix(".yaml")).read_text(encoding="utf-8"))
        command = command_ai_toolkit(self._run())

        self.assertEqual(config["config"]["name"], "ai-toolkit-example")
        process = config["config"]["process"][0]
        self.assertEqual(process["model"]["name_or_path"], "black-forest-labs/FLUX.2-klein-base-4B")
        self.assertEqual(process["datasets"][0]["folder_path"], folder_path)
        self.assertEqual(process["network"]["linear"], 4)
        self.assertEqual(process["train"]["steps"], 1)
        self.assertFalse(process["train"]["gradient_checkpointing"])
        self.assertFalse(process["model"]["quantize"])
        self.assertFalse(process["model"]["quantize_te"])
        self.assertFalse(process["model"]["low_vram"])
        self.assertEqual(command["cwd"], "/opt/ai-toolkit")
        self.assertEqual(command["argv"][:2], ["python", "-c"])
        self.assertIn("hook_before_train_loop", command["argv"][2])
        self.assertIn("ai-toolkit.yaml", command["argv"][3])
        self.assertEqual(command["env"], {
            "SEED": "42",
            "MODELS_PATH": "/workspace/cache/ai-toolkit/models",
            "KURA_AI_TOOLKIT_VIDEO_SUFFIXES": frozen_suffixes(AI_TOOLKIT_VIDEO_SUFFIXES),
        })

    def test_provider_only_runpod_command_uses_runpod_working_directory(self) -> None:
        run = self._run()
        run["compute"] = {"provider": "runpod"}

        command = command_ai_toolkit(run)

        self.assertEqual(command["cwd"], "/app/ai-toolkit")

    def test_compile_has_no_directory_inference_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "requires a frozen manifest projection"):
                compile_ai_toolkit(self._run(), Path(directory) / "ai-toolkit")

    def test_compile_rejects_legacy_minimal_projection_lock(self) -> None:
        run = self._run()
        dataset_id = str(run["datasets"][0]["id"])
        view = f"runs/{run['id']}/cache/dataset-view/ai-toolkit/{dataset_id}"
        legacy_lock = {
            "backend": "ai-toolkit",
            "datasets": [{
                "id": dataset_id,
                "native": {
                    "folder_path": f"/workspace/{view}",
                    "caption_ext": ".txt",
                    "cache_latents_to_disk": True,
                },
                "views": [{"consumers": [{
                    "kind": "recursive-directory",
                    "native_pointer": "/folder_path",
                    "path": view,
                }]}],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ai-toolkit"
            (destination.parent / "dataset-projection.lock.json").write_text(
                json.dumps(legacy_lock), encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not written by the verified dataset handoff"):
                compile_ai_toolkit(run, destination)

    @posix_only(DATASET_IO)
    def test_resume_compiles_absolute_target_and_hard_fail_runner(self) -> None:
        run = self._run()
        run["recipe"]["steps"] = 100
        run["parent_run"] = "source"
        run["continuation"] = {
            "mode": "resume",
            "source": {"artifact_id": "state-1", "manifest_sha256": "a" * 64, "observed_step": 100, "recipe_sha256": "b" * 64},
            "additional_steps": 50,
            "target_step": 150,
            "restoration_contract": {"level": "partial_resume", "restored": ["model", "optimizer", "global_step", "epoch"], "not_restored": ["scheduler", "rng", "dataloader_position"]},
        }
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ai-toolkit"
            self._write_frozen_projection(run, destination)
            command = compile_ai_toolkit(run, destination)
            process = yaml.safe_load(destination.with_suffix(".yaml").read_text(encoding="utf-8"))["config"]["process"][0]
        self.assertEqual(process["train"]["steps"], 150)
        script = command["argv"][2]
        self.assertIn("training state verified", script)
        self.assertIn("hook_before_train_loop", script)
        self.assertIn("/workspace/runs/ai-toolkit-example/resolved/training-state-source.lock.json", script)
        self.assertIn("state-1", command["argv"][2] + " ".join(command["argv"][3:]))

    @posix_only(DATASET_IO)
    def test_compile_rejects_non_mapping_native_config_override(self) -> None:
        projection_run = self._run()
        run = deepcopy(projection_run)
        run["backend"] = {"name": "ai-toolkit", "config": {"native_config": ["not", "a", "mapping"]}}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ai-toolkit"
            self._write_frozen_projection(projection_run, destination)
            with self.assertRaisesRegex(ValueError, "backend.config.native_config"):
                compile_ai_toolkit(run, destination)

    @posix_only(DATASET_IO)
    def test_compile_rejects_native_steps_that_duplicate_recipe(self) -> None:
        projection_run = self._run()
        run = deepcopy(projection_run)
        run["backend"]["config"]["native_config"] = {"train": {"steps": 2}}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "ai-toolkit"
            self._write_frozen_projection(projection_run, destination)
            with self.assertRaisesRegex(ValueError, "duplicates common recipe"):
                compile_ai_toolkit(run, destination)

    def test_removed_backend_overrides_are_rejected(self) -> None:
        run = self._run()
        run["backend_overrides"] = True
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "backend_overrides is not supported"):
                compile_ai_toolkit(run, Path(directory) / "ai-toolkit")


class MusubiBackendTests(unittest.TestCase):
    def _write_frozen_projection(self, run: dict[str, Any], destination: Path) -> None:
        architecture = str(run.get("backend", {}).get("config", {}).get("architecture") or "")
        target = "sample.mp4" if architecture in {"minimax_h3", "minimaxh3"} else "sample.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        freeze_fixture(run, destination.parent, target=target)

    def _run(self) -> dict[str, object]:
        return {
            "id": "musubi-example",
            "type": "train",
                        "model": {"base": "black-forest-labs/FLUX.2-klein-base-4B"},
            "datasets": [{"id": "tiny", "digest": "sha256:abc"}],
            "recipe": {"steps": 30, "seed": 42},
            "backend": {"name": "musubi-tuner", "config": {
                    "architecture": "flux2",
                    "model_version": "klein-base-4b",
                    "network_dim": 4,
                    "network_alpha": 4,
                    "learning_rate": 1.0e-4,
                    "batch_size": 1,
                    "resolution": [512, 512],
                    "model_paths": {
                        "dit": "/models/flux2-klein-base-4b.safetensors",
                        "vae": "/models/flux2-vae.safetensors",
                        "text_encoder": "/models/qwen_3_4b.safetensors",
                    },
                }
            },
        }

    def test_musubi_state_capture_and_constant_scheduler_resume_use_process_local_steps(self) -> None:
        run = self._run()
        run["parent_run"] = "source"
        run["continuation"] = {
            "mode": "resume",
            "source": {"artifact_id": "state-1", "manifest_sha256": "a" * 64, "observed_step": 30, "recipe_sha256": "b" * 64},
            "additional_steps": 10,
            "target_step": 40,
            "restoration_contract": {"level": "best_effort_resume", "restored": ["model", "optimizer", "scheduler", "rng"], "not_restored": ["global_step", "epoch", "data_position"]},
        }
        script = command_musubi_tuner(run)["argv"][2]
        self.assertIn("--save_state", script)
        self.assertIn("--save_state_on_train_end", script)
        self.assertIn("--save_every_n_steps 10", script)
        self.assertIn("--save_last_n_steps_state 10", script)
        self.assertIn("--resume /workspace/artifacts/training-state/state-1/payload", script)
        self.assertIn("--max_train_steps 10", script)
        self.assertNotIn("--max_train_steps 40", script)
        self.assertIn("training state verified", script)
        self.assertIn("/workspace/runs/musubi-example/resolved/training-state-source.lock.json", script)

    def test_musubi_state_capture_rejects_epoch_save_escape_hatches(self) -> None:
        run = self._run()
        run["backend"]["config"]["extra_args"] = ["--save_every_n_epochs", "1"]
        with self.assertRaisesRegex(ValueError, "epoch.*training-state"):
            command_musubi_tuner(run)

    def test_musubi_resume_rejects_finite_scheduler(self) -> None:
        run = self._run()
        run["backend"]["config"]["lr_scheduler"] = "cosine"
        run["parent_run"] = "source"
        run["continuation"] = {
            "mode": "resume",
            "source": {"artifact_id": "state-1", "manifest_sha256": "a" * 64, "observed_step": 30, "recipe_sha256": "b" * 64},
            "additional_steps": 10,
            "target_step": 40,
            "restoration_contract": {"level": "best_effort_resume", "restored": [], "not_restored": []},
        }
        with self.assertRaisesRegex(ValueError, "constant scheduler"):
            command_musubi_tuner(run)

    def test_musubi_resume_rejects_finite_scheduler_from_extra_args(self) -> None:
        for extra_args in (["--lr_scheduler", "cosine"], ["--lr_scheduler=cosine"]):
            with self.subTest(extra_args=extra_args):
                run = self._run()
                run["backend"]["config"]["extra_args"] = extra_args
                run["parent_run"] = "source"
                run["continuation"] = {
                    "mode": "resume",
                    "source": {"artifact_id": "state-1", "manifest_sha256": "a" * 64, "observed_step": 30, "recipe_sha256": "b" * 64},
                    "additional_steps": 10,
                    "target_step": 40,
                    "restoration_contract": {"level": "best_effort_resume", "restored": [], "not_restored": []},
                }
                with self.assertRaisesRegex(ValueError, "adapter-owned flag"):
                    command_musubi_tuner(run)

    def test_musubi_rejects_duplicate_value_flags_in_extra_args(self) -> None:
        run = self._run()
        run["backend"]["config"]["extra_args"] = [
            "--blocks_to_swap", "2", "--blocks_to_swap=3",
        ]

        with self.assertRaisesRegex(ValueError, "adapter-owned flag.*--blocks_to_swap"):
            command_musubi_tuner(run)

    def test_musubi_extra_args_cannot_override_adapter_owned_flags_or_abbreviate_them(self) -> None:
        for extra_args in (
            ["--dataset_config", "/workspace/datasets/unsafe.toml"],
            ["--dataset_conf=/workspace/datasets/unsafe.toml"],
            ["--output_dir", "/workspace/datasets/unsafe-output"],
            ["--res", "/workspace/artifacts/training-state/unsafe/payload"],
        ):
            with self.subTest(extra_args=extra_args):
                run = self._run()
                run["backend"]["config"]["extra_args"] = extra_args
                with self.assertRaisesRegex(ValueError, "adapter-owned flag"):
                    command_musubi_tuner(run)

    def test_musubi_extra_args_are_refused_only_where_the_architecture_emits_the_flag(self) -> None:
        # flux2 emits --timestep_sampling and --weighting_scheme itself.
        for extra_args in (["--timestep_sampling", "sigmoid"], ["--timestep_samp=sigmoid"], ["--weighting_scheme=none"]):
            with self.subTest(extra_args=extra_args):
                run = self._run()
                run["backend"]["config"]["extra_args"] = extra_args
                with self.assertRaisesRegex(ValueError, "adapter-owned flag"):
                    command_musubi_tuner(run)
        # It does not emit --discrete_flow_shift, so the native flag passes through.
        run = self._run()
        run["backend"]["config"]["extra_args"] = ["--discrete_flow_shift", "2.5"]
        command = command_musubi_tuner(run)
        self.assertIn("--discrete_flow_shift 2.5", " ".join(command["argv"]))

    def test_musubi_resume_rejects_invalid_state_save_cadence_cleanly(self) -> None:
        for cadence in ("10", True, 0, -1):
            with self.subTest(cadence=cadence):
                run = self._run()
                run["backend"]["config"]["save_every_n_steps"] = cadence
                run["parent_run"] = "source"
                run["continuation"] = {
                    "mode": "resume",
                    "source": {"artifact_id": "state-1", "manifest_sha256": "a" * 64, "observed_step": 30, "recipe_sha256": "b" * 64},
                    "additional_steps": 10,
                    "target_step": 40,
                    "restoration_contract": {"level": "best_effort_resume", "restored": [], "not_restored": []},
                }
                with self.assertRaisesRegex(ValueError, "save_every_n_steps must be a positive integer"):
                    command_musubi_tuner(run)

    @posix_only(DATASET_IO)
    def test_compile_musubi_writes_dataset_toml_and_command_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "musubi"
            self._write_frozen_projection(self._run(), destination)
            command = compile_musubi_tuner(self._run(), destination)
            dataset_toml = (destination / "dataset.toml").read_text(encoding="utf-8")
            bundle = yaml.safe_load((destination / "model-bundle.lock.yaml").read_text(encoding="utf-8"))
        self.assertIn(
            'image_jsonl_file = "/workspace/runs/musubi-example/cache/dataset-view/musubi/tiny/native/items.jsonl"',
            dataset_toml,
        )
        self.assertIn(
            'cache_directory = "/workspace/runs/musubi-example/cache/dataset-view/musubi/tiny/cache"',
            dataset_toml,
        )
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

    def test_compile_musubi_explicit_command_does_not_generate_dataset_toml(self) -> None:
        run = self._run()
        run["backend"]["config"] = {
            "command": {
                "cwd": "/opt/musubi-tuner",
                "argv": ["python", "custom_train.py"],
                "env": {},
            },
        }
        run.pop("recipe")
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "musubi"

            command = compile_musubi_tuner(run, destination)

            self.assertFalse((destination / "dataset.toml").exists())
            self.assertFalse((destination / "model-bundle.lock.yaml").exists())
            self.assertEqual(command["argv"], ["python", "custom_train.py"])

    def test_command_musubi_rejects_native_steps_that_duplicate_recipe(self) -> None:
        run = self._run()
        run["backend"]["config"]["max_train_steps"] = 2
        from kura.backends import validate_backend_config
        with self.assertRaisesRegex(ValueError, "unsupported key.*max_train_steps"):
            validate_backend_config(run)

    def test_command_musubi_requires_architecture(self) -> None:
        run = self._run()
        run["backend"]["config"].pop("architecture")
        with self.assertRaisesRegex(ValueError, "architecture is required"):
            command_musubi_tuner(run)

    def test_command_musubi_requires_seed(self) -> None:
        run = self._run()
        run["recipe"]["seed"] = None
        with self.assertRaisesRegex(ValueError, "recipe.seed must be an integer"):
            command_musubi_tuner(run)

    def test_command_musubi_rejects_recipe_duplicates_in_extra_args(self) -> None:
        run = self._run()
        run["backend"]["config"]["extra_args"] = ["--max_train_steps=2"]
        with self.assertRaisesRegex(ValueError, "adapter-owned flag"):
            command_musubi_tuner(run)

    def test_command_musubi_only_adds_memory_saving_flags_when_explicit(self) -> None:
        run = self._run()
        run["backend"]["config"].update({
            "gradient_checkpointing": True,
            "fp8_base": True,
            "fp8_scaled": True,
            "blocks_to_swap": 4,
        })

        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("--gradient_checkpointing", script)
        self.assertIn("--fp8_base", script)
        self.assertIn("--fp8_scaled", script)
        self.assertIn("--blocks_to_swap 4", script)

    def test_command_musubi_save_precision_defaults_to_bf16_but_can_be_overridden(self) -> None:
        run = self._run()
        script = command_musubi_tuner(run)["argv"][2]
        self.assertIn("--save_precision bf16", script)

        run["backend"]["config"]["save_precision"] = "fp16"
        script = command_musubi_tuner(run)["argv"][2]
        self.assertIn("--save_precision fp16", script)

    def test_command_musubi_rejects_invalid_save_precision(self) -> None:
        run = self._run()
        run["backend"]["config"]["save_precision"] = "int8"

        with self.assertRaisesRegex(ValueError, "save_precision"):
            command_musubi_tuner(run)

    def test_command_musubi_rejects_h2d_block_swap_without_gradient_checkpointing(self) -> None:
        run = self._run()
        run["backend"]["config"].update({
            "blocks_to_swap": 4,
            "block_swap_h2d_only": True,
        })

        with self.assertRaisesRegex(ValueError, "H2D-only block swap requires explicit gradient_checkpointing"):
            command_musubi_tuner(run)

    def test_command_musubi_projects_typed_block_swap_controls(self) -> None:
        run = self._run()
        run["backend"]["config"].update({
            "blocks_to_swap": 4,
            "block_swap_h2d_only": True,
            "block_swap_ring_size": 1,
            "use_pinned_memory_for_block_swap": True,
            "gradient_checkpointing": True,
        })

        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("--blocks_to_swap 4", script)
        self.assertIn("--block_swap_h2d_only", script)
        self.assertIn("--block_swap_ring_size 1", script)
        self.assertIn("--use_pinned_memory_for_block_swap", script)
        memory = display_musubi_tuner(run)["memory"]
        self.assertEqual(memory["blocks_to_swap"], 4)
        self.assertIs(memory["block_swap_h2d_only"], True)
        self.assertEqual(memory["block_swap_ring_size"], 1)
        self.assertIs(memory["use_pinned_memory_for_block_swap"], True)

    def test_command_musubi_validates_typed_block_swap_controls(self) -> None:
        cases = (
            (
                {"blocks_to_swap": 4, "block_swap_h2d_only": True},
                "H2D-only block swap requires explicit gradient_checkpointing",
            ),
            (
                {"block_swap_h2d_only": True, "gradient_checkpointing": True},
                "H2D-only block swap requires blocks_to_swap",
            ),
            (
                {"blocks_to_swap": 4, "block_swap_ring_size": 2},
                "block_swap_ring_size requires block_swap_h2d_only=true",
            ),
            (
                {
                    "blocks_to_swap": 4,
                    "block_swap_h2d_only": True,
                    "block_swap_ring_size": 0,
                    "gradient_checkpointing": True,
                },
                "block_swap_ring_size must be a positive integer",
            ),
        )
        for config, message in cases:
            with self.subTest(config=config):
                run = self._run()
                run["backend"]["config"].update(config)
                with self.assertRaisesRegex(ValueError, message):
                    command_musubi_tuner(run)

    def test_command_musubi_rejects_a40_flux2_9b_large_micro_batch(self) -> None:
        run = self._run()
        run["model"] = {"base": "black-forest-labs/FLUX.2-klein-base-9B"}
        run["compute"] = {"executor": "docker", "gpu": "NVIDIA A40"}
        run["backend"]["config"].update({"model_version": "klein-base-9b", "batch_size": 4, "resolution": [512, 512]})

        with self.assertRaisesRegex(ValueError, "batch_size=4 has been observed to OOM"):
            command_musubi_tuner(run)

    def test_command_musubi_allows_a40_flux2_9b_accumulated_effective_batch(self) -> None:
        run = self._run()
        run["model"] = {"base": "black-forest-labs/FLUX.2-klein-base-9B"}
        run["compute"] = {"executor": "docker", "gpu": "NVIDIA A40"}
        run["backend"]["config"].update({
                "model_version": "klein-base-9b",
                "gradient_checkpointing": True,
                "gradient_accumulation_steps": 4,
            })

        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("--gradient_checkpointing", script)
        self.assertIn("--gradient_accumulation_steps 4", script)

    def test_command_musubi_rejects_a40_flux2_9b_1024_without_checkpointing(self) -> None:
        run = self._run()
        run["model"] = {"base": "black-forest-labs/FLUX.2-klein-base-9B"}
        run["compute"] = {"executor": "docker", "gpu": "NVIDIA A40"}
        run["backend"]["config"].update({"model_version": "klein-base-9b", "network_dim": 32, "batch_size": 1, "resolution": [1024, 1024]})

        with self.assertRaisesRegex(ValueError, "observed to OOM even with batch_size=1"):
            command_musubi_tuner(run)

    def test_command_musubi_rejects_secret_explicit_env(self) -> None:
        run = self._run()
        run["recipe"] = {}
        run["backend"]["config"] = {"command": {"cwd": "/opt/musubi-tuner", "argv": ["python", "train.py"], "env": {"HF_TOKEN": "secret"}}}

        with self.assertRaisesRegex(ValueError, "env must not contain secrets"):
            command_musubi_tuner(run)

    def test_command_musubi_allows_non_secret_generated_env(self) -> None:
        run = self._run()
        run["backend"]["config"]["env"] = {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

        command = command_musubi_tuner(run)

        self.assertEqual(command["env"], {
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "KURA_MUSUBI_IMAGE_SUFFIXES": frozen_suffixes(MUSUBI_IMAGE_SUFFIXES),
            "KURA_MUSUBI_VIDEO_SUFFIXES": frozen_suffixes(MUSUBI_VIDEO_SUFFIXES),
            "KURA_MUSUBI_AUDIO_SUFFIXES": frozen_suffixes(MUSUBI_AUDIO_SUFFIXES),
        })

    def test_command_musubi_rejects_secret_generated_env(self) -> None:
        run = self._run()
        run["backend"]["config"]["env"] = {"HF_TOKEN": "secret"}

        with self.assertRaisesRegex(ValueError, "env must not contain secrets"):
            command_musubi_tuner(run)

    def test_command_musubi_rejects_invalid_extra_args_shape(self) -> None:
        run = self._run()
        run["backend"]["config"]["extra_args"] = "--fp8_base"

        with self.assertRaisesRegex(ValueError, "extra_args must be a list of strings"):
            command_musubi_tuner(run)

    def test_command_musubi_requires_explicit_model_paths(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "wan"}}
        with self.assertRaisesRegex(ValueError, "model_paths, model_downloads, or a known model.base bundle"):
            command_musubi_tuner(run)

    def test_command_musubi_unknown_architecture_names_kura_adapter_layer(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
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
                run["backend"] = {"name": "musubi-tuner", "config": {"architecture": architecture, "model_paths": model_paths}}
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
                run["backend"] = {"name": "musubi-tuner", "config": {"architecture": architecture, "model_paths": model_paths}}
                script = command_musubi_tuner(run)["argv"][2]
                for text in expected:
                    self.assertIn(text, script)
                self.assertIn("--max_train_steps 30", script)
                self.assertIn("--save_precision bf16", script)

    def test_command_musubi_minimax_h3_generates_guidance_loss_pipeline(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
            "architecture": "minimax_h3",
            "task": "t2va",
            "model_paths": {
                "dit": "/models/minimax-h3.safetensors",
                "video_vae": "/models/minimax-h3-video-vae.safetensors",
                "audio_vae": "/models/minimax-h3-audio-vae.safetensors",
                "text_encoder": "/models/minimax-h3-text-encoder.safetensors",
            },
            "h3_guidance_loss_scale": 4.0,
            "h3_guidance_loss_sigma_min": 0.15,
            "blocks_to_swap": 48,
            "text_encoder_blocks_to_swap": 50,
            "video_only": True,
            "gradient_checkpointing": True,
        }}

        spec = command_musubi_tuner(run)
        script = spec["argv"][2]

        self.assertIn("minimax_h3_cache_latents.py", script)
        self.assertIn("--video_vae /models/minimax-h3-video-vae.safetensors", script)
        self.assertIn("--audio_vae /models/minimax-h3-audio-vae.safetensors", script)
        self.assertIn("minimax_h3_cache_text_encoder_outputs.py", script)
        cache_path = "/workspace/runs/musubi-example/cache/musubi/minimax-h3-uncond.safetensors"
        self.assertIn(f"--uncond_output {cache_path}", script)
        self.assertIn("minimax_h3_train_network.py", script)
        self.assertGreaterEqual(script.count("--task t2va"), 3)
        self.assertIn("--h3_guidance_loss_scale 4.0", script)
        self.assertIn("--h3_guidance_loss_sigma_min 0.15", script)
        self.assertIn(f"--h3_guidance_loss_uncond_cache {cache_path}", script)
        self.assertIn("--blocks_to_swap 48", script)
        self.assertIn("--text_encoder_blocks_to_swap 50", script)
        self.assertIn("--video_only", script)
        self.assertIn("--gradient_checkpointing", script)
        self.assertNotIn("KURA_MUSUBI_TARGET_FPS", spec["env"])
        self.assertEqual(spec["env"]["KURA_MUSUBI_CACHE"], "/workspace/runs/musubi-example/cache/musubi")
        self.assertEqual(spec["write_roots"], [{
            "role": "backend-cache",
            "path": "/workspace/runs/musubi-example/cache/musubi",
            "env": "KURA_MUSUBI_CACHE",
        }])

    def test_command_musubi_minimax_h3_guidance_requires_precache(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
            "architecture": "minimax_h3",
            "task": "t2va",
            "model_paths": {
                "dit": "/models/minimax-h3.safetensors",
                "video_vae": "/models/minimax-h3-video-vae.safetensors",
                "audio_vae": "/models/minimax-h3-audio-vae.safetensors",
                "text_encoder": "/models/minimax-h3-text-encoder.safetensors",
            },
            "h3_loss_method": "guidance",
            "precache": False,
        }}

        with self.assertRaisesRegex(ValueError, "guidance requires precache=true"):
            command_musubi_tuner(run)

    def test_command_musubi_minimax_h3_projects_fl2va_and_ref2va_tasks_to_every_stage(self) -> None:
        for task in ("fl2va", "ref2va"):
            with self.subTest(task=task):
                run = self._run()
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "minimax_h3",
                    "task": task,
                    "model_bundle": "none",
                    "model_paths": {
                        "dit": f"/models/minimax-h3-{task}.safetensors",
                        "video_vae": "/models/minimax-h3-video-vae.safetensors",
                        "audio_vae": "/models/minimax-h3-audio-vae.safetensors",
                        "text_encoder": "/models/minimax-h3-text-encoder.safetensors",
                    },
                    "h3_loss_method": "guidance",
                    "h3_guidance_loss_scale": 4.0,
                    "h3_guidance_loss_sigma_min": 0.15,
                }}

                script = command_musubi_tuner(run)["argv"][2]

                self.assertEqual(script.count(f"--task {task}"), 3)
                self.assertIn("--h3_guidance_loss_uncond_cache", script)

    def test_command_musubi_minimax_h3_ref2va_auto_bundle_uses_ref2va_transformer(self) -> None:
        run = self._run()
        run["model"] = {"base": "Comfy-Org/MiniMax-H3"}
        run["backend"] = {"name": "musubi-tuner", "config": {
            "architecture": "minimax_h3",
            "task": "ref2va",
            "h3_loss_method": "guidance",
        }}

        requirements = requirements_musubi(run, declared=True)

        dit = next(item for item in requirements if item["role"] == "dit")
        self.assertEqual(
            dit["identity"]["filename"],
            "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        )

    def test_command_musubi_minimax_h3_projects_one_frame_to_every_stage(self) -> None:
        for task in ("t2va", "fl2va", "ref2va"):
            with self.subTest(task=task):
                run = self._run()
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "minimax_h3",
                    "task": task,
                    "model_bundle": "none",
                    "model_paths": {
                        "dit": f"/models/minimax-h3-{task}.safetensors",
                        "video_vae": "/models/minimax-h3-video-vae.safetensors",
                        "audio_vae": "/models/minimax-h3-audio-vae.safetensors",
                        "text_encoder": "/models/minimax-h3-text-encoder.safetensors",
                    },
                    "h3_loss_method": "guidance",
                    "one_frame": True,
                    "video_only": True,
                }}

                script = command_musubi_tuner(run)["argv"][2]

                self.assertEqual(script.count("--one_frame"), 3)
                self.assertIn("--video_only", script)

    def test_command_musubi_minimax_h3_one_frame_requires_explicit_video_only(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
            "architecture": "minimax_h3",
            "task": "t2va",
            "model_paths": {
                "dit": "/models/minimax-h3.safetensors",
                "video_vae": "/models/minimax-h3-video-vae.safetensors",
                "audio_vae": "/models/minimax-h3-audio-vae.safetensors",
                "text_encoder": "/models/minimax-h3-text-encoder.safetensors",
            },
            "h3_loss_method": "guidance",
            "one_frame": True,
        }}

        with self.assertRaisesRegex(ValueError, "one_frame requires video_only=true"):
            command_musubi_tuner(run)

    def test_command_musubi_minimax_h3_training_adapter_is_recorded_without_guidance_loss(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
            "architecture": "minimax_h3",
            "task": "t2va",
            "model_bundle": "none",
            "model_paths": {
                "dit": "/models/minimax-h3.safetensors",
                "video_vae": "/models/minimax-h3-video-vae.safetensors",
                "audio_vae": "/models/minimax-h3-audio-vae.safetensors",
                "text_encoder": "/models/minimax-h3-text-encoder.safetensors",
                "base_weights": "/models/h3-training-adapter.safetensors",
            },
            "h3_loss_method": "training_adapter",
        }}

        script = command_musubi_tuner(run)["argv"][2]
        requirements = requirements_musubi(run, declared=True)

        self.assertIn("--base_weights /models/h3-training-adapter.safetensors", script)
        self.assertNotIn("--h3_guidance_loss_scale", script)
        self.assertNotIn("--uncond_output", script)
        base_weights = next(item for item in requirements if item["role"] == "base_weights")
        self.assertEqual(base_weights["runtime_reference"], "/models/h3-training-adapter.safetensors")

    def test_command_musubi_minimax_h3_training_adapter_rejects_prequantized_bundle(self) -> None:
        run = self._run()
        run["model"] = {"base": "Comfy-Org/MiniMax-H3"}
        run["backend"] = {"name": "musubi-tuner", "config": {
            "architecture": "minimax_h3",
            "model_paths": {"base_weights": "/models/h3-training-adapter.safetensors"},
            "h3_loss_method": "training_adapter",
        }}

        with self.assertRaisesRegex(ValueError, "pre-quantized ConvRot INT8.*BF16"):
            command_musubi_tuner(run)

    def test_command_musubi_minimax_h3_teacher_matching_projects_asymmetric_cache_tasks(self) -> None:
        cases = (
            ("first,last", False, "fl2va"),
            ("ref", False, "t2va"),
            ("subject_ref", True, "ref2va"),
        )
        for conditions, one_frame, latent_task in cases:
            with self.subTest(conditions=conditions):
                run = self._run()
                run["backend"] = {"name": "musubi-tuner", "config": {
                    "architecture": "minimax_h3",
                    "task": "t2va",
                    "model_bundle": "none",
                    "model_paths": {
                        "dit": "/models/minimax-h3.safetensors",
                        "video_vae": "/models/minimax-h3-video-vae.safetensors",
                        "audio_vae": "/models/minimax-h3-audio-vae.safetensors",
                        "text_encoder": "/models/minimax-h3-text-encoder.safetensors",
                    },
                    "h3_loss_method": "teacher_matching",
                    "h3_teacher_conditions": conditions,
                    "h3_teacher_condition_sigma_min": 0.15 if conditions == "subject_ref" else 0.0,
                    "h3_teacher_condition_sigma_max": 1.0 if conditions == "subject_ref" else 0.75,
                    "h3_teacher_loss_dc_weight": 0.3,
                    "h3_teacher_loss_mag_weight": 0.5 if conditions == "subject_ref" else 1.0,
                    "h3_timestep_focus_prob": 0.5 if conditions != "subject_ref" else 0.0,
                    "one_frame": one_frame,
                    "video_only": one_frame,
                }}

                script = command_musubi_tuner(run)["argv"][2]

                self.assertRegex(script, rf"minimax_h3_cache_latents\.py[^;]+--task {latent_task}")
                self.assertRegex(script, r"minimax_h3_cache_text_encoder_outputs\.py[^;]+--task t2va")
                self.assertIn(f"--teacher_conditions {conditions}", script)
                self.assertRegex(script, r"minimax_h3_train_network\.py[^;]+--task t2va")
                self.assertIn("--h3_teacher_matching", script)
                self.assertIn(f"--h3_teacher_conditions {conditions}", script)
                self.assertNotIn("--h3_guidance_loss_uncond_cache", script)

    def test_command_musubi_minimax_h3_rejects_invalid_loss_contracts(self) -> None:
        base = self._run()
        base["backend"] = {"name": "musubi-tuner", "config": {
            "architecture": "minimax_h3",
            "task": "t2va",
            "model_paths": {
                "dit": "/models/minimax-h3.safetensors",
                "video_vae": "/models/minimax-h3-video-vae.safetensors",
                "audio_vae": "/models/minimax-h3-audio-vae.safetensors",
                "text_encoder": "/models/minimax-h3-text-encoder.safetensors",
            },
        }}
        cases = (
            ({"h3_loss_method": "training_adapter"}, "base_weights"),
            ({"h3_loss_method": "teacher_matching", "task": "ref2va", "h3_teacher_conditions": "ref"}, "requires task t2va"),
            ({"h3_loss_method": "teacher_matching", "one_frame": True, "video_only": True, "h3_teacher_conditions": "ref"}, "one_frame.*subject_ref"),
            ({"h3_loss_method": "plain"}, "h3_loss_method"),
        )
        for changes, message in cases:
            with self.subTest(changes=changes):
                run = json.loads(json.dumps(base))
                run["backend"]["config"].update(changes)
                with self.assertRaisesRegex(ValueError, message):
                    command_musubi_tuner(run)

    def test_command_musubi_framepack_uses_current_fp8_base_flag(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
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
        run["backend"] = {"name": "musubi-tuner", "config": {
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

        spec = command_musubi_tuner(run)
        script = spec["argv"][2]

        self.assertIn("--dit_high_noise /models/wan22-high.safetensors", script)
        self.assertIn("--timestep_boundary 0.875", script)
        self.assertNotIn("KURA_MUSUBI_TARGET_FPS", spec["env"])

    @posix_only(DATASET_IO)
    def test_command_musubi_flux2_dev_uses_dev_contract(self) -> None:
        run = self._run()
        run["model"]["base"] = "black-forest-labs/FLUX.2-dev"
        run["backend"] = {"name": "musubi-tuner", "config": {
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
            self._write_frozen_projection(run, destination)
            command = compile_musubi_tuner(run, destination)
            bundle = yaml.safe_load((destination / "model-bundle.lock.yaml").read_text(encoding="utf-8"))
            script = command["argv"][2]

        expected = {item["role"]: item["expected_format"] for item in bundle["models"]}
        self.assertEqual(expected["vae"], "flux2_ae_or_vae")
        self.assertEqual(expected["text_encoder"], "safetensors")
        self.assertIn("--model_version dev", script)
        self.assertIn("--fp8_base", script)
        self.assertIn("--fp8_scaled", script)
        self.assertIn("--vae_dtype bfloat16", script)

    def test_command_musubi_flux2_dev_rejects_qwen_only_fp8_text_encoder(self) -> None:
        run = self._run()
        run["backend"]["config"].update({"model_version": "dev", "fp8_text_encoder": True})

        with self.assertRaisesRegex(ValueError, "does not support fp8_text_encoder"):
            command_musubi_tuner(run)

    def test_command_musubi_wan_one_frame_updates_cache_and_train(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
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
        run["backend"] = {"name": "musubi-tuner", "config": {
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

    def test_command_musubi_wan_flf2v_precache_uses_i2v_contract(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "wan",
                "task": "flf2v-14B",
                "model_paths": {
                    "dit": "/models/wan-flf2v.safetensors",
                    "vae": "/models/wan-vae.safetensors",
                    "t5": "/models/umt5.pth",
                    "clip": "/models/clip.pth",
                },
            }
        }

        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("wan_cache_latents.py", script)
        self.assertEqual(script.count("--i2v"), 1)
        self.assertIn("--clip /models/clip.pth", script)

    def test_command_musubi_wan_22_i2v_precache_uses_i2v_without_clip(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "wan",
                "task": "i2v-A14B",
                "model_paths": {
                    "dit": "/models/wan22-i2v-low.safetensors",
                    "vae": "/models/wan-vae.safetensors",
                    "t5": "/models/umt5.pth",
                },
            }
        }

        script = command_musubi_tuner(run)["argv"][2]

        self.assertEqual(script.count("--i2v"), 1)
        self.assertNotIn("--clip", script)

    def test_command_musubi_wan_rejects_unknown_task_before_launch(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "wan",
                "task": "future-task",
                "model_paths": {
                    "dit": "/models/wan.safetensors",
                    "vae": "/models/wan-vae.safetensors",
                    "t5": "/models/umt5.pth",
                },
            }
        }

        with self.assertRaisesRegex(ValueError, "unsupported Musubi Wan native selector"):
            command_musubi_tuner(run)

    def test_command_musubi_framepack_one_frame_updates_cache_and_train(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
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

    def test_command_musubi_framepack_freezes_latent_window_size_in_cache_and_train(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
            "architecture": "framepack",
            "model_paths": {
                "dit": "/models/framepack.safetensors",
                "vae": "/models/fpack-vae.safetensors",
                "text_encoder1": "/models/te1.safetensors",
                "text_encoder2": "/models/te2.safetensors",
                "image_encoder": "/models/siglip.safetensors",
            },
            "precache": True,
        }}

        script = command_musubi_tuner(run)["argv"][2]

        self.assertEqual(script.count("--latent_window_size 9"), 1)

    def test_command_musubi_qwen_model_versions_reach_all_three_stages(self) -> None:
        for model_version, expected in (
            ("original", "original"),
            ("edit", "edit"),
            ("edit-2509", "edit-2509"),
            ("EDIT_2511", "edit-2511"),
            ("layered", "layered"),
        ):
            with self.subTest(model_version=model_version):
                run = self._run()
                run["backend"] = {"name": "musubi-tuner", "config": {
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

                self.assertEqual(script.count(f"--model_version {expected}"), 3)

    def test_command_musubi_qwen_layered_can_remove_the_base_from_training_targets(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
            "architecture": "qwen_image",
            "model_version": "layered",
            "remove_first_image_from_target": True,
            "model_paths": {
                "dit": "/models/qwen-layered-dit.safetensors",
                "vae": "/models/qwen-layered-vae.safetensors",
                "text_encoder": "/models/qwen-vl.safetensors",
            },
        }}

        script = command_musubi_tuner(run)["argv"][2]

        self.assertEqual(script.count("--remove_first_image_from_target"), 1)

    def test_command_musubi_hunyuan_15_i2v_updates_cache_and_train(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
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
        run["backend"] = {"name": "musubi-tuner", "config": {
                "architecture": "hidream_o1",
                "task": "i2i",
                "model_type": "dev",
                "model_paths": {"dit": "/models/hidream-dev.safetensors"},
                "extra_args": ["--network_args", "conv_dim=4", "conv_alpha=1"],
            }
        }

        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("--task i2i", script)
        self.assertIn("--model_type dev", script)
        self.assertIn("--network_args conv_dim=4 conv_alpha=1", script)

    def test_command_musubi_kandinsky_i2v_preserves_task(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
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
        run["backend"] = {"name": "musubi-tuner", "config": {
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
        run["backend"] = {"name": "musubi-tuner", "config": {
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
        self.assertNotIn("HF_HUB_DISABLE_XET", script)
        self.assertIn('cache_dir = os.environ.get("HF_HUB_CACHE")', script)
        self.assertIn("HF_HUB_CACHE is required before downloading models", script)
        self.assertNotIn('or "/root/.cache/huggingface"', script)
        self.assertNotIn("local_dir", script)
        self.assertNotIn("/workspace/cache/hf-models/musubi", script)
        self.assertIn("KURA_HF_DOWNLOAD_NO_PROGRESS_SEC", script)
        self.assertIn("repo_cache_dirs(cache_dir, item)", script)
        self.assertNotIn("remove_incomplete_files", script)
        self.assertIn("preserving resumable state and retrying", script)
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

    @posix_only(POSIX_PATHS)
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

    @posix_only(POSIX_PATHS)
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

    @posix_only(POSIX_PATHS)
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
        run["backend"] = {"name": "musubi-tuner", "config": {
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
        with self.assertRaisesRegex(ValueError, "model_downloads.dit contains unsupported key.*local_dir"):
            command_musubi_tuner(run)

    def test_command_musubi_resolves_known_flux2_klein_bundle(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux2", "model_version": "klein-base-4b"}}
        command = command_musubi_tuner(run)
        script = command["argv"][2]
        self.assertIn("Comfy-Org/vae-text-encorder-for-flux-klein-4b", script)
        self.assertIn("split_files/diffusion_models/flux-2-klein-base-4b.safetensors", script)
        self.assertIn("split_files/vae/flux2-vae.safetensors", script)
        self.assertIn("split_files/text_encoders/qwen_3_4b.safetensors", script)
        self.assertIn("--vae /workspace/cache/models/musubi/Comfy-Org--vae-text-encorder-for-flux-klein-4b/vae/split_files/vae/flux2-vae.safetensors", script)

    @posix_only(DATASET_IO)
    def test_command_musubi_resolves_known_flux2_klein_base_9b_bundle(self) -> None:
        run = self._run()
        run["model"] = {"base": "black-forest-labs/FLUX.2-klein-base-9B"}
        run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux2", "model_version": "klein-base-9b"}}
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
            self._write_frozen_projection(run, destination)
            compile_musubi_tuner(run, destination)
            bundle = yaml.safe_load((destination / "model-bundle.lock.yaml").read_text(encoding="utf-8"))
        expected = {item["role"]: item["expected_format"] for item in bundle["models"]}
        self.assertEqual(expected["vae"], "flux2_ae_or_vae")
        self.assertEqual(expected["text_encoder"], "qwen3_8b_text_encoder")

    @posix_only(DATASET_IO)
    def test_command_musubi_resolves_known_krea2_bundle(self) -> None:
        run = self._run()
        run["model"] = {"base": "krea/Krea-2-Raw"}
        run["backend"] = {"name": "musubi-tuner", "config": {
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
            self._write_frozen_projection(run, destination)
            compile_musubi_tuner(run, destination)
            bundle = yaml.safe_load((destination / "model-bundle.lock.yaml").read_text(encoding="utf-8"))
        self.assertEqual(bundle["architecture"], "krea2")
        expected = {item["role"]: item["expected_format"] for item in bundle["models"]}
        self.assertEqual(expected["dit"], "safetensors")
        self.assertEqual(expected["vae"], "safetensors")
        self.assertEqual(expected["text_encoder"], "safetensors")

    @posix_only(DATASET_IO)
    def test_command_musubi_resolves_known_minimax_h3_bundle(self) -> None:
        run = self._run()
        run["model"] = {"base": "Comfy-Org/MiniMax-H3"}
        run["backend"] = {"name": "musubi-tuner", "config": {
            "architecture": "minimax_h3",
            "model_bundle": "minimax-h3-pruned-int8",
            "video_only": True,
            "dataset_options": {"tiny": {"target_frames": [5]}},
        }}

        command = command_musubi_tuner(run)
        script = command["argv"][2]

        self.assertIn("Comfy-Org/MiniMax-H3", script)
        self.assertIn("diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors", script)
        self.assertIn("vae/minimax_h3_video_vae_fp16.safetensors", script)
        self.assertIn("vae/minimax_h3_audio_vae_fp32.safetensors", script)
        self.assertIn("text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", script)
        self.assertEqual(script.count('"link_mode": "hardlink"'), 4)
        self.assertIn("src/musubi_tuner/minimax_h3_train_network.py", script)

        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "musubi"
            self._write_frozen_projection(run, destination)
            compile_musubi_tuner(run, destination)
            bundle = yaml.safe_load((destination / "model-bundle.lock.yaml").read_text(encoding="utf-8"))
        self.assertEqual(bundle["architecture"], "minimax_h3")
        expected = {item["role"]: item["expected_format"] for item in bundle["models"]}
        self.assertEqual(expected, {
            "audio_vae": "safetensors",
            "dit": "safetensors",
            "text_encoder": "safetensors",
            "video_vae": "safetensors",
        })

    def test_musubi_adapter_script_registry_includes_minimax_h3(self) -> None:
        self.assertEqual(
            MUSUBI_ADAPTER_SCRIPTS["minimax_h3"],
            (
                "minimax_h3_train_network.py",
                "minimax_h3_cache_latents.py",
                "minimax_h3_cache_text_encoder_outputs.py",
            ),
        )

    def test_command_musubi_krea2_can_include_turbo_for_samples(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
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

    def test_command_musubi_krea2_projects_convrot_and_checkpoint_cpu_offload(self) -> None:
        run = self._run()
        run["backend"] = {"name": "musubi-tuner", "config": {
            "architecture": "krea2",
            "model_paths": {
                "dit": "/models/krea2.safetensors",
                "vae": "/models/qwen-image-vae.safetensors",
                "text_encoder": "/models/qwen3-vl.safetensors",
            },
            "convrot_int8": True,
            "convrot_int8_bwd": "bf16",
            "gradient_checkpointing": True,
            "gradient_checkpointing_cpu_offload": True,
        }}

        script = command_musubi_tuner(run)["argv"][2]

        self.assertIn("--convrot_int8", script)
        self.assertIn("--convrot_int8_bwd bf16", script)
        self.assertIn("--gradient_checkpointing_cpu_offload", script)

    def test_command_musubi_krea2_rejects_incompatible_memory_modes(self) -> None:
        base = self._run()
        base["backend"] = {"name": "musubi-tuner", "config": {
            "architecture": "krea2",
            "model_paths": {
                "dit": "/models/krea2.safetensors",
                "vae": "/models/qwen-image-vae.safetensors",
                "text_encoder": "/models/qwen3-vl.safetensors",
            },
        }}
        cases = (
            ({"convrot_int8": True, "fp8_base": True}, "cannot be combined with fp8"),
            ({"convrot_int8_bwd": "int8"}, "requires convrot_int8=true"),
            ({"convrot_int8": True, "include_turbo_dit": True}, "cannot be combined with include_turbo_dit"),
            ({"gradient_checkpointing_cpu_offload": True}, "requires gradient_checkpointing=true"),
        )
        for changes, expected in cases:
            with self.subTest(changes=changes):
                run = json.loads(json.dumps(base))
                run["backend"]["config"].update(changes)
                with self.assertRaisesRegex(ValueError, expected):
                    command_musubi_tuner(run)

    def test_command_musubi_infers_flux2_model_version_from_model_base(self) -> None:
        run = self._run()
        run["model"] = {"base": "black-forest-labs/FLUX.2-klein-base-9B"}
        run["backend"] = {"name": "musubi-tuner", "config": {"architecture": "flux2"}}
        command = command_musubi_tuner(run)

        self.assertIn("--model_version klein-base-9b", command["argv"][2])
        self.assertNotIn("klein-base-4b", command["argv"][2])

    def test_command_musubi_refuses_unknown_flux2_model_version_default(self) -> None:
        run = self._run()
        run["model"] = {"base": "custom/flux2-checkpoint"}
        run["backend"]["config"].pop("model_version", None)

        with self.assertRaisesRegex(ValueError, "refusing to default to 4B"):
            command_musubi_tuner(run)

    def test_command_musubi_can_prune_early_step_checkpoints(self) -> None:
        run = self._run()
        run["backend"]["config"]["save_every_n_steps"] = 100
        run["backend"]["config"]["prune_checkpoints_before_step"] = 1000
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
                        "backend": {"name": "musubi-tuner", "config": {
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
            estimate = _estimate_backend_download_bytes(run)

        self.assertEqual(estimate["unknown"], [])
        self.assertEqual(estimate["probe_failures"][0]["status"], "not_found")
        records = _model_download_preflight_report(run, estimate, executor="runpod")
        self.assertIn(("model-metadata-connectivity", "error"), {(item["check"], item["severity"]) for item in records})

    @posix_only(POSIX_PATHS)
    def test_command_is_detached_labeled_and_writes_to_mounted_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            run_dir.mkdir(parents=True)
            command, runtime_env, _ = docker_command(root, run_dir, {"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, "example:image", [], "r1")
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

    @posix_only(POSIX_PATHS)
    def test_docker_mount_sources_are_resolved_from_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            run_dir = root / "runs" / "example"
            run_dir.mkdir(parents=True)
            spec = {"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}
            command, runtime_env, _ = docker_command(root, run_dir, spec, "example:image", local_docker_mounts(root, {}), "r1")
            with tempfile.TemporaryDirectory() as elsewhere:
                outside = Path(elsewhere).resolve() / "hf"
                moved, _, _ = docker_command(root, run_dir, spec, "example:image", local_docker_mounts(root, {"docker": {"hf_cache": str(outside)}}), "r1")
        # The default cache is reached through the workspace mount; a moved one gets its own mount.
        self.assertFalse([item for item in command if item.endswith(":/workspace/cache/huggingface")])
        self.assertIn(f"{outside}:/workspace/cache/huggingface", moved)
        self.assertEqual(json.loads(runtime_env["KURA_WORKSPACE_PATH_MAPS"]), [{"container": "/workspace", "workspace": "/workspace"}])

    def test_docker_preflight_creates_writable_mount_sources(self) -> None:
        class Usage:
            total = 500 * 1024**3
            used = 100 * 1024**3
            free = 400 * 1024**3

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mounts = local_docker_mounts(root, {"docker": {"hf_cache": "./models/hf"}})
            # The free-space floor is tested separately; this test must not
            # depend on how full the machine running it is.
            with (
                patch("kura.executors.docker.docker_daemon_problem", return_value=None),
                patch("kura.executors.docker.shutil.disk_usage", return_value=Usage()),
            ):
                docker_preflight(root, mounts)
            self.assertTrue((root / "models" / "hf").is_dir())

    def test_docker_preflight_records_free_space_without_deciding_the_floor(self) -> None:
        # The floor is decided once, by _local_launch_disk_preflight, which plan and launch both run.
        class Usage:
            total = 100 * 1024**3
            used = 80 * 1024**3
            free = 20 * 1024**3

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("kura.executors.docker.docker_daemon_problem", return_value=None),
                patch("kura.executors.docker.shutil.disk_usage", return_value=Usage()),
            ):
                payload = docker_preflight(root, [])
        self.assertEqual(payload["disk"]["workspace"]["free_bytes"], 20 * 1024**3)

    def test_plan_reports_local_disk_on_a_machine_without_docker(self) -> None:
        from kura.run_commands.plan import _local_disk_preflight_report

        run = {"id": "r", "type": "train", "backend": {"name": "ai-toolkit"}, "compute": {"executor": "docker"}}
        with tempfile.TemporaryDirectory() as directory:
            with patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(200)), \
                    patch("kura.run_commands.plan.subprocess.run", side_effect=FileNotFoundError("docker")):
                records = _local_disk_preflight_report(run, Path(directory), {}, {})
        self.assertEqual([item["severity"] for item in records], ["info"])

    def test_plan_reports_the_local_disk_verdict_launch_will_reach(self) -> None:
        from kura.run_commands.plan import _local_disk_preflight_report, collect_run_preflight

        run = {"id": "r", "type": "train", "backend": {"name": "ai-toolkit"}, "compute": {"executor": "docker"}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(60)), \
                    patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
                short = _local_disk_preflight_report(run, root, {}, {})
                enough = _local_disk_preflight_report(run, root, {"docker": {"min_free_gb": 50}}, {})
                # The launch's own preflight (dry runs included) does not run the disk check twice.
                shared = [item for item in collect_run_preflight(run, root, config={}, executor="docker") if item["check"] == "disk"]
        self.assertEqual(shared, [])
        self.assertEqual([item["severity"] for item in short], ["error"])
        self.assertIn("requires at least 100 GiB", short[0]["fact"])
        self.assertEqual([item["severity"] for item in enough], ["info"])
        self.assertIn("passes", enough[0]["fact"])

    def test_local_launch_disk_preflight_uses_configured_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(60)), patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
                with self.assertRaisesRegex(ValueError, "requires at least 100 GiB"):
                    _local_launch_disk_preflight(root, {"type": "train"}, {})
                payload = _local_launch_disk_preflight(root, {"type": "train"}, {"docker": {"min_free_gb": 50}})
        self.assertEqual(payload["required_gib"], 50)

    def test_local_launch_disk_preflight_counts_estimated_hf_downloads(self) -> None:
        run = {
            "type": "train",
                        "model": {"base": "custom"},
            "backend": {"name": "musubi-tuner", "config": {
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
                patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(60)),
                patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")),
                patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "ok", "size_bytes": 20 * 1024**3}),
            ):
                with self.assertRaisesRegex(ValueError, "requires at least 70 GiB"):
                    _local_launch_disk_preflight(root, run, {"docker": {"min_free_gb": 50}})
                payload = _local_launch_disk_preflight(root, run, {"docker": {"min_free_gb": 40}})
        self.assertEqual(payload["estimates"]["musubi_downloads"]["bytes"], 20 * 1024**3)
        self.assertEqual(payload["paths"]["hf_cache"]["estimated_write_bytes"], 20 * 1024**3)

    def test_local_launch_disk_preflight_rejects_large_unapproved_model_downloads(self) -> None:
        run = {
            "type": "train",
                        "model": {"base": "custom"},
            "backend": {"name": "musubi-tuner", "config": {
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
                    _local_launch_disk_preflight(root, run, {"docker": {"min_free_gb": 50}})
                run["safety"] = {"allow_large_model_downloads": True}
                payload = _local_launch_disk_preflight(root, run, {"docker": {"min_free_gb": 50}})
        self.assertEqual(payload["estimates"]["musubi_downloads"]["bytes"], 30 * 1024**3)

    def test_local_launch_disk_preflight_counts_allowed_checkpoint_budget(self) -> None:
        run = {
            "type": "train",
                        "recipe": {"steps": 3000},
            "backend": {"name": "musubi-tuner", "config": {"architecture": "wan", "save_every_n_steps": 100}},
            "safety": {"allow_many_checkpoints": True, "checkpoint_estimate_gb": 2},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(100)),
                patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")),
            ):
                with self.assertRaisesRegex(ValueError, "requires at least 110 GiB"):
                    _local_launch_disk_preflight(root, run, {"docker": {"min_free_gb": 50}})
        self.assertEqual(_checkpoint_safety_preflight(run), None)

    def test_local_launch_disk_preflight_sums_estimates_on_shared_backing(self) -> None:
        run = {
            "type": "train",
                        "recipe": {"steps": 2000},
            "backend": {"name": "musubi-tuner", "config": {
                    "architecture": "flux2",
                    "save_every_n_steps": 100,
                    "model_downloads": {
                        "dit": {"repo": "example/model", "filename": "dit.safetensors"},
                    },
                }
            },
            "safety": {"allow_many_checkpoints": True, "checkpoint_estimate_gb": 2, "allow_large_model_downloads": True},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(100)),
                patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")),
                patch("kura.run_commands.plan._hf_file_size_probe", return_value={"status": "ok", "size_bytes": 40 * 1024**3}),
            ):
                with self.assertRaisesRegex(ValueError, "requires at least 130 GiB"):
                    _local_launch_disk_preflight(root, run, {"docker": {"min_free_gb": 50}})
                payload = _local_launch_disk_preflight(root, run, {"docker": {"min_free_gb": 10}})
        self.assertEqual(payload["paths"]["workspace"]["estimated_write_bytes"], 40 * 1024**3)
        self.assertEqual(payload["paths"]["hf_cache"]["estimated_write_bytes"], 40 * 1024**3)
        self.assertEqual(payload["paths"]["workspace"]["backing_estimated_write_bytes"], 80 * 1024**3)
        self.assertEqual(payload["paths"]["hf_cache"]["backing_estimated_write_bytes"], 80 * 1024**3)

    def test_local_launch_disk_preflight_honors_run_disk_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(140)), patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
                with self.assertRaisesRegex(ValueError, "requires at least 150 GiB"):
                    _local_launch_disk_preflight(root, {"safety": {"max_run_disk_gb": 150}}, {"docker": {"min_free_gb": 50}})

    def test_local_launch_disk_preflight_rejects_unknown_wsl_backing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(900, confidence="unknown", backing_kind="wsl2_vhdx")), patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
                with self.assertRaisesRegex(ValueError, "unknown physical backing free space"):
                    _local_launch_disk_preflight(root, {"type": "train"}, {})
                payload = _local_launch_disk_preflight(root, {"safety": {"allow_storage_risk": True}}, {})
        self.assertEqual(payload["paths"]["workspace"]["confidence"], "unknown")

    def test_free_space_gate_measures_the_wsl_backing_drive(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)
            with _wsl_with_short_host_drive(linux_free_gib=900, host_free_gib=5):
                with self.assertRaisesRegex(ValueError, "test download needs about 10 GiB free .* only 5 GiB is available on C:"):
                    ensure_free_bytes(target, 10 * 1024**3, context="test download", config={})
                ensure_free_bytes(target, 4 * 1024**3, context="test download", config={})

    def test_large_docker_build_cache_does_not_stop_a_local_launch(self) -> None:
        build_cache = json.dumps({"Type": "Build Cache", "Size": "120GB"})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("kura.run_commands.plan.probe_storages", side_effect=self._storage_probe(500)), \
                    patch("kura.run_commands.plan.subprocess.run", return_value=subprocess.CompletedProcess([], 0, build_cache + "\n")):
                payload = _local_launch_disk_preflight(root, {"type": "train"}, {"docker": {"build_cache_limit_gb": 30}})
        self.assertEqual(payload["docker_storage"], [{"Type": "Build Cache", "Size": "120GB"}])

    def test_docker_command_keeps_hf_token_value_out_of_argv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            run_dir.mkdir(parents=True)
            with patch.dict(os.environ, {"HF_TOKEN": "hf-secret"}, clear=False):
                command, runtime_env, _ = docker_command(root, run_dir, {"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, "example:image", [], "r1")
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

    def test_reconcile_docker_preserves_confirmed_terminal_outcome(self) -> None:
        for terminal_state, exit_code in (("completed", 0), ("failed", 7), ("stopped", None), ("interrupted", None), ("unknown", None), ("launch_failed", None)):
            with self.subTest(state=terminal_state), tempfile.TemporaryDirectory() as directory:
                run_dir = self._run_dir(Path(directory))
                (run_dir / "status.json").write_text(
                    json.dumps({"state": terminal_state, "exit_code": exit_code, "ended": "confirmed-end", "last_realization": "realizations/r1.json", "container_id": "container-1"}),
                    encoding="utf-8",
                )
                result = subprocess.CompletedProcess([], 0, '{"Running": true, "ExitCode": 0}')
                with patch("kura.executors.docker.subprocess.run", return_value=result):
                    status = reconcile_docker(run_dir)

                self.assertEqual(status["state"], terminal_state)
                self.assertEqual(status["exit_code"], exit_code)
                self.assertEqual(status["ended"], "confirmed-end")

    def test_reconcile_docker_records_container_times_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            state = {"Running": False, "ExitCode": 0, "StartedAt": "2026-10-01T03:45:40.123456789Z", "FinishedAt": "2026-10-01T03:49:20.987654321Z"}
            result = subprocess.CompletedProcess([], 0, json.dumps(state))
            with patch("kura.executors.docker.subprocess.run", return_value=result):
                for _ in range(2):
                    try:
                        reconcile_docker(run_dir)
                    except ValueError:
                        pass
            phases = launch_phases(run_dir, "r1")
        self.assertEqual([item["phase"] for item in phases], ["container_started", "container_exited"])
        self.assertEqual(format_launch_phases(phases), "model download + training 3m 40s")

    def test_reconcile_docker_merges_observation_into_latest_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            inspect_started = threading.Event()
            release_inspect = threading.Event()
            reconciled: list[dict[str, Any]] = []

            def inspect(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
                inspect_started.set()
                self.assertTrue(release_inspect.wait(2))
                return subprocess.CompletedProcess([], 0, '{"Running": true, "ExitCode": 0}')

            with patch("kura.executors.docker.subprocess.run", side_effect=inspect):
                thread = threading.Thread(target=lambda: reconciled.append(reconcile_docker(run_dir)))
                thread.start()
                self.assertTrue(inspect_started.wait(2))
                _mutate_run_status(run_dir, lambda status: status.update({"sampling_progress": 3}))
                release_inspect.set()
                thread.join(2)

            self.assertFalse(thread.is_alive())
            self.assertEqual(reconciled[0]["sampling_progress"], 3)

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
            self.assertEqual(status["seconds_per_iter"], 2.49)

    @posix_only(POSIX_PATHS)
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
            self.assertEqual(status["seconds_per_iter"], 2.10)
            self.assertEqual(status["outputs"], ["outputs/example.safetensors"])

    def test_reconcile_missing_container_is_unknown_not_interrupted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            result = __import__("subprocess").CompletedProcess([], 1, "", "Error: No such container")
            with patch("kura.executors.docker.subprocess.run", return_value=result):
                status = reconcile_docker(run_dir)
            self.assertEqual(status["state"], "unknown")
            self.assertIsNone(status["exit_code"])
            self.assertIsNone(status["ended"])

    def test_training_launch_rejects_executor_different_from_compiled_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                yaml.safe_dump(
                    {
                        "id": "example",
                        "type": "train",
                        "compute": {"executor": "docker"},
                        "backend": {"name": "ai-toolkit", "config": {}},
                    }
                ),
                encoding="utf-8",
            )
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch("kura.run_commands.launch.launch_runpod") as launch, patch(
                    "sys.stderr", new_callable=__import__("io").StringIO
                ) as stderr:
                    self.assertEqual(launch_run("example", executor="runpod", dry_run=True), 1)
            finally:
                os.chdir(previous)
            launch.assert_not_called()
            self.assertIn("compiled for executor.name=docker", stderr.getvalue())
            self.assertIn("recompile", stderr.getvalue())

    def test_resume_launch_rejects_runtime_image_override(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "derived"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                yaml.safe_dump({
                    "id": "derived", "type": "train", "compute": {"executor": "runpod"},
                    "backend": {"name": "ai-toolkit", "config": {}},
                    "continuation": {"mode": "resume"},
                }),
                encoding="utf-8",
            )
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch("kura.run_commands.launch.launch_runpod") as launch, patch(
                    "sys.stderr", new_callable=io.StringIO
                ) as stderr:
                    self.assertEqual(launch_run("derived", executor="runpod", dry_run=True, image="other@sha256:bad"), 1)
            finally:
                os.chdir(previous)
            launch.assert_not_called()
            self.assertIn("runtime image is frozen at compile time", stderr.getvalue())

    def test_runpod_launch_uses_the_image_frozen_by_compile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            manifest = {
                "id": "example", "type": "train", "compute": {"executor": "runpod"},
                "backend": {"name": "ai-toolkit", "config": {}},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
            (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({
                "backend": "ai-toolkit", "adapter_source": {"kind": "test", "value": "test"},
                "cwd": "/workspace", "argv": ["true"], "env": {},
            }), encoding="utf-8")
            (run_dir / "resolved" / "env.lock").write_text(yaml.safe_dump({
                "selected_image": "frozen/image@sha256:1234",
            }), encoding="utf-8")
            (root / "workspace.yaml").write_text(yaml.safe_dump({'docker': {}, 'images': {'ai-toolkit': 'changed-after-compile'}}), encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch("kura.run_commands.launch.observe_run", return_value={"state": "compiled"}), patch(
                    "kura.run_commands.launch.collect_run_preflight", return_value=[]
                ), patch("kura.run_commands.launch.launch_runpod") as launch:
                    self.assertEqual(launch_run("example", executor="runpod", dry_run=True), 0)
            finally:
                os.chdir(previous)
            self.assertEqual(launch.call_args.kwargs["image"], "frozen/image@sha256:1234")

    def test_runpod_resume_launch_rejects_missing_compile_time_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "derived"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            manifest = {
                "id": "derived",
                "type": "train",
                "compute": {"executor": "runpod"},
                "backend": {"name": "ai-toolkit", "config": {}},
                "continuation": {"mode": "resume"},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
            (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({
                "backend": "ai-toolkit", "adapter_source": {"kind": "test", "value": "test"},
                "cwd": "/workspace", "argv": ["true"], "env": {},
            }), encoding="utf-8")
            (root / "workspace.yaml").write_text(yaml.safe_dump({'docker': {}, 'images': {'ai-toolkit': 'current/default:latest'}}), encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch("kura.run_commands.launch.observe_run", return_value={"state": "compiled"}), patch(
                    "kura.run_commands.launch.collect_run_preflight", return_value=[]
                ), patch("kura.run_commands.launch.launch_runpod") as launch, patch(
                    "sys.stderr", new_callable=io.StringIO
                ) as stderr:
                    self.assertEqual(launch_run("derived", executor="runpod", dry_run=True), 1)
            finally:
                os.chdir(previous)

            launch.assert_not_called()
            self.assertIn("no compile-time frozen image", stderr.getvalue())

    def test_local_resume_launch_uses_the_content_id_observed_at_compile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "derived"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "compiled"}), encoding="utf-8")
            manifest = {
                "id": "derived", "type": "train", "compute": {"executor": "docker"},
                "backend": {"name": "ai-toolkit", "config": {}},
                "continuation": {"mode": "resume"},
            }
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
            (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({
                "backend": "ai-toolkit", "adapter_source": {"kind": "test", "value": "test"},
                "cwd": "/workspace", "argv": ["true"], "env": {},
            }), encoding="utf-8")
            (run_dir / "resolved" / "env.lock").write_text(yaml.safe_dump({
                "selected_image": "mutable-local:dev",
                "selected_image_identity": {
                    "reference": "mutable-local:dev",
                    "pinning": {"strength": "content-hash", "value": "sha256:compiled-image"},
                },
            }), encoding="utf-8")
            (root / "workspace.yaml").write_text(yaml.safe_dump({'docker': {'mounts': []}, 'images': {'ai-toolkit': 'mutable-local:dev'}}), encoding="utf-8")
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch("kura.run_commands.launch.observe_run", return_value={"state": "compiled"}), patch(
                    "kura.run_commands.launch.collect_run_preflight", return_value=[]
                ), patch("kura.run_commands.launch.launch_docker") as launch:
                    self.assertEqual(launch_run("derived", executor="docker", dry_run=True), 0)
            finally:
                os.chdir(previous)
            self.assertEqual(launch.call_args.kwargs["image"], "sha256:compiled-image")

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
                                                "backend": {"name": "ai-toolkit", "config": {"command": {"cwd": "/workspace", "argv": ["python", "-c", "print(1)"], "env": {}}}},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({"backend": "ai-toolkit", "adapter_source": {"kind": "test", "value": "test"}, "cwd": "/workspace", "argv": ["python", "-c", "print(1)"], "env": {}}), encoding="utf-8")
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({'docker': {'min_free_gb': 10, 'mounts': []}, 'images': {'ai-toolkit': 'configured-local'}}),
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
                                                "backend": {"name": "ai-toolkit", "config": {"command": {"cwd": "/workspace", "argv": ["python", "-c", "print(1)"], "env": {}}}},
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({"backend": "ai-toolkit", "adapter_source": {"kind": "test", "value": "test"}, "cwd": "/workspace", "argv": ["python", "-c", "print(1)"], "env": {}}), encoding="utf-8")
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({'docker': {'workspace_target': '/ws', 'mounts': []}, 'images': {'ai-toolkit': 'local'}}),
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
            self.assertIn("docker.workspace_target", stderr.getvalue())
            self.assertIn("kura workspace migrate", stderr.getvalue())


class LaunchPhaseTests(unittest.TestCase):
    def test_launch_phase_records_append_and_summarize(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            for phase, at in [
                ("pod_create_requested", "2026-10-01T12:43:24+09:00"),
                ("pod_created", "2026-10-01T12:43:25+09:00"),
                ("ssh_ready", "2026-10-01T12:47:58+09:00"),
                ("upload_started", "2026-10-01T12:47:58+09:00"),
                ("upload_finished", "2026-10-01T12:48:10+09:00"),
                ("remote_job_started", "2026-10-01T12:48:20+09:00"),
                ("remote_exit_observed", "2026-10-01T12:50:00+09:00"),
                ("download_started", "2026-10-01T12:50:00+09:00"),
                ("download_finished", "2026-10-01T12:50:40+09:00"),
            ]:
                record_launch_phase(run_dir, "r1", phase, at=at)
            summary = format_launch_phases(launch_phases(run_dir, "r1"))
        self.assertEqual(summary, "startup 4m 34s · upload 22s · model download + training 1m 40s · download 40s")

    def test_launch_phase_write_failure_only_warns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            with patch("kura.executors.common.append_line_durably", side_effect=OSError("disk full")), \
                 contextlib.redirect_stderr(io.StringIO()) as stderr:
                record_launch_phase(run_dir, "r1", "pod_created")
        self.assertIn("launch timing", stderr.getvalue())

    def test_ssh_commands_reuse_one_master_connection(self) -> None:
        details = {"ip": "203.0.113.5", "port": 22115, "key": "/tmp/key"}
        with tempfile.TemporaryDirectory() as directory:
            control_dir = Path(directory)
            socket_path = control_dir / "203.0.113.5_22115"
            socket_path.write_text("stale", encoding="utf-8")
            calls: list[tuple[list[str], dict[str, object]]] = []

            def fake_run(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                calls.append((argv, kwargs))
                if "-O" in argv:
                    return subprocess.CompletedProcess(argv, 255, "", "")
                self.assertFalse(socket_path.exists(), "a stale socket must be removed before the master binds")
                return subprocess.CompletedProcess(argv, 0, "", "")

            with patch("kura.run_commands.runpod_ssh._ssh_control_dir", return_value=control_dir), \
                 patch("kura.run_commands.runpod_ssh.subprocess.run", side_effect=fake_run):
                command = _ssh_base(details)
                _start_ssh_master(details)
        self.assertIn(f"ControlPath={socket_path}", command)
        self.assertIn("ControlMaster=no", command)
        check, master = calls
        self.assertIn("check", check[0])
        self.assertIn("ControlMaster=yes", master[0])
        self.assertNotIn("ControlMaster=no", master[0])
        self.assertIn("-N", master[0])
        self.assertIn("-f", master[0])
        self.assertTrue(any(option.startswith("ControlPersist=") for option in master[0]))
        # A backgrounded master must not hold the caller's pipes open.
        self.assertEqual(master[1].get("stdout"), subprocess.DEVNULL)
        self.assertEqual(master[1].get("stderr"), subprocess.DEVNULL)

    def test_ssh_master_is_skipped_when_already_running(self) -> None:
        details = {"ip": "203.0.113.5", "port": 22115, "key": "/tmp/key"}
        with tempfile.TemporaryDirectory() as directory, \
             patch("kura.run_commands.runpod_ssh._ssh_control_dir", return_value=Path(directory)), \
             patch("kura.run_commands.runpod_ssh.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
            _start_ssh_master(details)
        self.assertEqual(run.call_count, 1)

    def test_ssh_master_failure_never_stops_the_caller(self) -> None:
        details = {"ip": "203.0.113.5", "port": 22115, "key": "/tmp/key"}
        with tempfile.TemporaryDirectory() as directory, \
             patch("kura.run_commands.runpod_ssh._ssh_control_dir", return_value=Path(directory)), \
             patch("kura.run_commands.runpod_ssh.subprocess.run", side_effect=subprocess.TimeoutExpired(["ssh"], 10)):
            _start_ssh_master(details)

    @posix_only(POSIX_PATHS)
    def test_ssh_reuse_is_disabled_for_an_unsafe_socket_directory(self) -> None:
        details = {"ip": "203.0.113.5", "port": 22115, "key": "/tmp/key"}
        with tempfile.TemporaryDirectory() as directory:
            home = Path(directory)
            shared = home / ".ssh" / "kura-mux"
            shared.mkdir(parents=True)
            shared.chmod(0o777)
            with patch("kura.run_commands.runpod_ssh._ssh_control_dir", _REAL_SSH_CONTROL_DIR), \
                 patch("kura.run_commands.runpod_ssh.Path.home", return_value=home):
                options = _ssh_base(details)
                shared.chmod(0o700)
                private = _ssh_base(details)
                with patch("kura.run_commands.runpod_ssh.os.name", "nt"):
                    windows = _ssh_base(details)
        self.assertFalse(any(option.startswith("ControlPath=") for option in options))
        self.assertTrue(any(option.startswith("ControlPath=") for option in private))
        self.assertFalse(any(option.startswith("ControlPath=") for option in windows))

    def test_startup_splits_at_the_reported_container_start(self) -> None:
        phases = [
            {"phase": "pod_create_requested", "at": "2026-10-01T12:43:24+09:00"},
            {"phase": "ssh_ready", "at": "2026-10-01T12:47:58+09:00", "container_started_at": "2026-10-01T03:47:30Z"},
        ]
        self.assertEqual(format_launch_phases(phases), "startup 4m 34s (allocate+pull 4m 06s, boot 28s)")

    def test_completion_summary_prints_launch_timing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            run_dir.mkdir(parents=True)
            record_launch_phase(run_dir, "r1", "container_start_requested", at="2026-10-01T12:00:00+09:00")
            record_launch_phase(run_dir, "r1", "container_started", at="2026-10-01T03:00:15+00:00")
            record_launch_phase(run_dir, "r1", "container_exited", at="2026-10-01T03:01:00+00:00")
            text = format_run_completion(root, run_dir, {"state": "completed", "exit_code": 0, "last_realization": "realizations/r1.json"})
        self.assertIn("time       startup 15s · model download + training 45s", text)

    def test_docker_timestamps_normalize_to_parseable_values(self) -> None:
        from datetime import datetime, timezone

        self.assertEqual(_docker_timestamp("2026-10-01T03:45:40.123456789Z"), datetime(2026, 10, 1, 3, 45, 40, 123456, tzinfo=timezone.utc).astimezone().isoformat())
        self.assertEqual(_docker_timestamp("2026-10-01T03:45:40Z"), datetime(2026, 10, 1, 3, 45, 40, tzinfo=timezone.utc).astimezone().isoformat())
        self.assertIsNone(_docker_timestamp("0001-01-01T00:00:00Z"))
        self.assertIsNone(_docker_timestamp("not a time"))

    def test_unchanged_training_states_are_not_recorded_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "logs").mkdir(parents=True)
            (run_dir / "status.json").write_text("{}", encoding="utf-8")
            manifest = {"id": "state-1", "manifest_sha256": "a" * 64, "observed_step": 250, "restoration_contract": {"level": "partial_resume"}}
            _record_pulled_training_states(run_dir, [manifest])
            _record_pulled_training_states(run_dir, [manifest])
            events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual([event["event"] for event in events], ["run_training_states_pulled"])


@posix_only(POSIX_PATHS)
class RunPodUnattendedCompletionTests(unittest.TestCase):
    """The Pod-side guards that bound billing while no controller is attached."""

    def _sandbox(self, root: Path, *, refuse: tuple[str, ...] = ()) -> tuple[dict[str, str], list[dict[str, Any]], Any]:
        import http.server

        calls: list[dict[str, Any]] = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - http.server API
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                name = "podTerminate" if "podTerminate" in body["query"] else "podStop"
                calls.append({"mutation": name, "pod": body["variables"]["podId"], "auth": self.headers.get("Authorization"), "agent": self.headers.get("User-Agent")})
                payload = {"errors": [{"message": "not allowed"}]} if name in refuse else {"data": {name: None}}
                encoded = json.dumps(payload).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, *_args: object) -> None:
                return

        server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        environ = root / "environ"
        environ.write_bytes(b"PATH=/usr/bin\0RUNPOD_API_KEY=pod-scoped-key\0RUNPOD_POD_ID=pod-7\0")
        env = {
            "PATH": os.environ["PATH"],
            "KURA_POD_ENV_FILE": str(environ),
            "KURA_LOG_PATH": str(root / "stdout.log"),
            "KURA_RUNPOD_GRAPHQL_URL": f"http://127.0.0.1:{server.server_port}/graphql",
        }
        return env, calls, server

    def test_self_delete_terminates_with_the_pod_scoped_key_from_the_init_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, calls, _server = self._sandbox(root)
            script = POD_SELF_DELETE_FUNCTION + '\nkura_pod_self_delete "$KURA_LOG_PATH"\n'
            result = subprocess.run(["sh", "-c", script], env=env, text=True, capture_output=True, check=False, timeout=30)
            log = (root / "stdout.log").read_text(encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr + log)
        expected_auth = " ".join(("Bearer", "pod-scoped-key"))
        self.assertEqual(calls, [{"mutation": "podTerminate", "pod": "pod-7", "auth": expected_auth, "agent": "Kura-pod-guard"}])
        self.assertNotIn("pod-scoped-key", log)

    def test_self_delete_stops_the_pod_when_termination_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, calls, _server = self._sandbox(root, refuse=("podTerminate",))
            script = POD_SELF_DELETE_FUNCTION + '\nkura_pod_self_delete "$KURA_LOG_PATH"\n'
            result = subprocess.run(["sh", "-c", script], env=env, text=True, capture_output=True, check=False, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call["mutation"] for call in calls], ["podTerminate", "podStop"])

    def _run_timer(self, root: Path, *, collected: bool) -> list[str]:
        env, calls, _server = self._sandbox(root)
        mark = root / "example.collected"
        if collected:
            mark.touch()
        script = "\n".join([
            POD_SELF_DELETE_FUNCTION,
            "export KURA_JOB_STARTED_EPOCH=$(date +%s)",
            _unattended_completion_shell(wait_sec=1, collected_mark=str(mark)),
            "wait",
        ])
        subprocess.run(["sh", "-c", script], env=env, text=True, capture_output=True, check=False, timeout=30)
        return [call["mutation"] for call in calls]

    def test_uncollected_outputs_delete_the_pod_after_the_wait(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calls = self._run_timer(Path(directory), collected=False)
        self.assertEqual(calls, ["podTerminate"])

    def test_collected_outputs_leave_the_pod_to_the_controller(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            calls = self._run_timer(Path(directory), collected=True)
        self.assertEqual(calls, [])

    def test_a_download_in_progress_keeps_the_pod_until_it_is_collected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env, calls, _server = self._sandbox(root)
            collected, collecting = root / "example.collected", root / "example.collecting"
            collecting.touch()
            block = _unattended_completion_shell(wait_sec=1, collected_mark=str(collected), collecting_mark=str(collecting))
            # The timer re-checks every 60s; shorten that so the test observes both phases.
            block = block.replace("sleep 60", "sleep 1")
            script = "\n".join([POD_SELF_DELETE_FUNCTION, "export KURA_JOB_STARTED_EPOCH=$(date +%s)", block])
            process = subprocess.Popen(["sh", "-c", script + "\nwait\n"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(3)
            self.assertEqual(calls, [], "the timer must wait while a download is in progress")
            collected.touch()
            process.wait(timeout=30)
        self.assertEqual(calls, [])

    def test_automatic_wait_is_the_longer_of_two_hours_and_the_training_time(self) -> None:
        for elapsed, expected in ((60, 7200), (10_000, 10_000)):
            with self.subTest(elapsed=elapsed), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                env, _calls, _server = self._sandbox(root)
                block = _unattended_completion_shell(wait_sec=None, collected_mark=str(root / "never"))
                # Only the announced wait matters here, not the background timer.
                block = block.split("\n(\n")[0]
                script = f"export KURA_JOB_STARTED_EPOCH=$(( $(date +%s) - {elapsed} ))\n{block}\n"
                subprocess.run(["sh", "-c", script], env=env, text=True, capture_output=True, check=False, timeout=30)
                log = (root / "stdout.log").read_text(encoding="utf-8")
                announced = int(log.split(" in ")[1].split("s ")[0])
            self.assertTrue(expected <= announced <= expected + 2, log)

    def test_job_script_starts_the_timer_after_the_exit_record_unless_disabled(self) -> None:
        common = dict(workspace="/workspace", run_id="example", realization_id="r1", remote_secret_path="/tmp/kura-secrets/example.env", archive_name="bundle.tar.gz", remote_archive="/workspace/bundle.tar.gz", cwd="/opt/tool", command="true")
        automatic = _runpod_remote_job_script(**common)
        disabled = _runpod_remote_job_script(**common, unattended_wait_sec=0)
        for script in (automatic, disabled):
            result = subprocess.run(["sh", "-n"], input=script, text=True, capture_output=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("kura_pod_self_delete", automatic)
        self.assertLess(automatic.index("remote-exit-"), automatic.index("unattended completion"))
        self.assertIn("/tmp/kura-jobs/example.collected", automatic)
        self.assertNotIn("sleep \"$kura_wait\"", disabled)

    def test_max_lease_guard_deletes_the_pod_with_the_same_self_delete(self) -> None:
        guard = _runpod_lease_guard_shell(max_lease_sec=3600, pod_id="pod-7", log_path="/workspace/runs/example/logs/stdout.log")
        self.assertIn("kura_pod_self_delete", guard)
        self.assertIn("kura_lease_initial=$(( $(date +%s) + 3600 ))", guard)
        result = subprocess.run(["bash", "-n"], input=guard, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_a_failed_download_clears_the_collecting_mark(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {}\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1"}), encoding="utf-8")
            details = {"pod_id": "pod-1", "ip": "203.0.113.5", "port": 22, "key": "/tmp/key"}
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.runpod_ssh.shutil.which", return_value="/usr/bin/runpodctl"), \
                     patch("kura.run_commands.runpod_ssh._runpod_ssh_details", return_value=details), \
                     patch("kura.run_commands.runpod_ssh._start_ssh_master"), \
                     patch("kura.run_commands.runpod_ssh._touch_runpod_mark", return_value=True) as touch, \
                     patch("kura.run_commands.runpod_ssh._runpod_remote_snapshot_manifest", side_effect=ValueError("remote manifest unreadable")), \
                     patch("kura.run_commands.runpod_ssh._clear_runpod_mark", return_value=True) as clear, \
                     contextlib.redirect_stderr(io.StringIO()):
                    code = cmd_run_download(argparse.Namespace(run_id="example", force=True))
            finally:
                os.chdir(previous)
        self.assertEqual(code, 1)
        self.assertEqual(touch.call_args.args[1], "/tmp/kura-jobs/example.collecting")
        self.assertEqual(clear.call_args.args[1], "/tmp/kura-jobs/example.collecting")

    def test_a_download_does_not_start_without_the_collecting_mark(self) -> None:
        details = {"pod_id": "pod-1", "ip": "203.0.113.5", "port": 22, "key": "/tmp/key"}
        with patch("kura.run_commands.runpod_ssh._run_runpod_mark_command", _REAL_RUNPOD_MARK_COMMAND), \
             patch("kura.run_commands.runpod_ssh.subprocess.run", return_value=subprocess.CompletedProcess([], 255, "", "connection refused")):
            with self.assertRaisesRegex(ValueError, "cannot mark the RunPod outputs as being collected"):
                _mark_runpod_outputs_collecting(details, "example")

    def test_collection_marks_the_pod(self) -> None:
        with patch("kura.run_commands.runpod_ssh._run_runpod_mark_command", _REAL_RUNPOD_MARK_COMMAND), \
             patch("kura.run_commands.runpod_ssh.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "", "")) as run:
            _mark_runpod_outputs_collected({"ip": "203.0.113.5", "port": 22, "key": "/tmp/key"}, "example")
        self.assertIn("touch /tmp/kura-jobs/example.collected", run.call_args.args[0][-1])

    def test_billing_confirmation_names_the_unattended_wait(self) -> None:
        settings = {"gpu_type_ids": ["NVIDIA A40"], "gpu_count": 1, "cloud_types": ["SECURE"]}
        with patch("kura.executors.runpod.runpod_gpu_availability", return_value={"status": "unavailable"}), \
             contextlib.redirect_stderr(io.StringIO()) as stderr:
            _confirm_runpod_launch({}, settings, yes=True, max_lease_sec=12 * 3600, unattended_wait="longer of 2h and the training time")
        self.assertIn("Unattended wait: longer of 2h and the training time", stderr.getvalue())


class RunDiscardTests(unittest.TestCase):
    def _make_run(self, root: Path, run_id: str, *, state: str) -> Path:
        run_dir = root / "runs" / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "run.yaml").write_text(f"id: {run_id}\n", encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": state}), encoding="utf-8")
        (run_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")
        return run_dir

    @posix_only(POSIX_PATHS)
    def test_run_discard_defaults_to_dry_run(self) -> None:
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            root = Path(directory).resolve()
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({'docker': {}, 'images': {'ai-toolkit': 'kura/ai-toolkit:test'}}),
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
            root = Path(directory).resolve()
            (root / "workspace.yaml").write_text(
                yaml.safe_dump({'docker': {}, 'images': {'ai-toolkit': 'kura/ai-toolkit:test'}}),
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
    @staticmethod
    def _compiled_runpod_launch_workspace(root: Path, status: dict[str, object]) -> Path:
        run_dir = root / "runs" / "example"
        (run_dir / "resolved").mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(
            yaml.safe_dump(
                {
                    "id": "example",
                    "type": "train",
                    "compute": {"executor": "runpod", "gpu": "NVIDIA A40"},
                    "backend": {"name": "ai-toolkit", "config": {"command": {"cwd": "/workspace", "argv": ["python", "-c", "print(1)"], "env": {}}}},
                }
            ),
            encoding="utf-8",
        )
        (run_dir / "resolved" / "backend-command.lock.json").write_text(
            json.dumps({"backend": "ai-toolkit", "adapter_source": {"kind": "test", "value": "test"}, "cwd": "/app/ai-toolkit", "argv": ["python", "-c", "print(1)"], "env": {}}),
            encoding="utf-8",
        )
        (root / "workspace.yaml").write_text(
            yaml.safe_dump(
                {'runpod': {'storage_mode': 'upload', 'gpu_type_ids': ['NVIDIA A40'], 'cloud_type': 'COMMUNITY'}, 'docker': {}, 'images': {'ai-toolkit': 'local'}}
            ),
            encoding="utf-8",
        )
        return run_dir

    def test_execute_run_uses_compiled_docker_executor_and_waits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("compute: {executor: docker}\n", encoding="utf-8")
            with (
                patch("kura.run_commands.launch._run_path", return_value=run_dir),
                patch("kura.run_commands.launch._launch_docker_through_runner", return_value=0) as through_runner,
            ):
                self.assertEqual(execute_run("example"), 0)

        # Local training goes through the job runner, and the command follows it.
        through_runner.assert_called_once_with("example", image=None, follow=True, notify_channels=None)

    def test_execute_run_uses_compiled_runpod_executor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("compute: {executor: runpod}\n", encoding="utf-8")
            with (
                patch("kura.run_commands.launch._run_path", return_value=run_dir),
                patch("kura.run_commands.launch._launch_runpod_through_runner", return_value=0) as through_runner,
            ):
                self.assertEqual(execute_run("example", max_lease="3h", yes=True), 0)

        # RunPod training goes through the job runner with the options the launch needs.
        self.assertEqual(through_runner.call_args.args, ("example",))
        self.assertTrue(through_runner.call_args.kwargs["yes"])
        options = through_runner.call_args.kwargs["options"]
        # A run that names no capacity policy waits for a GPU (RunPod GPUs are often taken).
        self.assertNotIn("hold_for", options)
        self.assertEqual((options["max_lease"], options["wait_for_capacity"], options["capacity_poll_interval"]), ("3h", "24h", "30s"))

    def test_execute_run_uses_frozen_capacity_wait_policy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                "compute:\n  executor: runpod\n  capacity: {mode: wait, timeout: 8h, poll_interval: 45s}\n",
                encoding="utf-8",
            )
            with (
                patch("kura.run_commands.launch._run_path", return_value=run_dir),
                patch("kura.run_commands.launch._launch_runpod_through_runner", return_value=0) as through_runner,
            ):
                self.assertEqual(execute_run("example"), 0)

        options = through_runner.call_args.kwargs["options"]
        self.assertEqual((options["wait_for_capacity"], options["capacity_poll_interval"]), ("8h", "45s"))

    def test_runpod_gpu_availability_reports_each_cloud(self) -> None:
        payload = {
            "data": {
                "g0": [
                    {
                        "id": "NVIDIA RTX A5000",
                        "displayName": "RTX A5000",
                        "memoryInGb": 24,
                        "community": {"stockStatus": "None", "uninterruptablePrice": None, "availableGpuCounts": []},
                        "secure": {"stockStatus": "Low", "uninterruptablePrice": 0.3, "availableGpuCounts": [1]},
                    }
                ]
            }
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        config = {"gpu_type_ids": ["NVIDIA RTX A5000"], "gpu_count": 1, "cloud_types": ["COMMUNITY", "SECURE"]}
        with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), patch("kura.executors.runpod.urlopen", return_value=Response()) as urlopen:
            result = runpod_gpu_availability(config, ["NVIDIA RTX A5000"], min_cuda_version="12.8")
        query = json.loads(urlopen.call_args.args[0].data)["query"]
        self.assertEqual(query.count('minCudaVersion: "12.8"'), 2)

        self.assertEqual(result["status"], "ok")
        clouds = result["candidates"][0]["clouds"]
        self.assertFalse(clouds[0]["available"])
        self.assertTrue(clouds[1]["available"])
        request = urlopen.call_args.args[0]
        self.assertNotIn("api-secret", request.data.decode("utf-8"))
        self.assertNotIn("api-secret", request.full_url)
        self.assertEqual(request.get_header("Authorization"), "Bearer api-secret")
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(request.unredirected_hdrs["Authorization"], "Bearer api-secret")
        self.assertRegex(request.get_header("User-agent"), r"^Kura/\d")

    def test_runpod_gpu_availability_without_key_is_nonfatal(self) -> None:
        with patch.dict(os.environ, {}, clear=True), patch("kura.executors.runpod.urlopen") as urlopen:
            result = runpod_gpu_availability(self._config(), ["NVIDIA A40"])

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("RUNPOD_API_KEY", result["reason"])
        urlopen.assert_not_called()

    def test_runpod_control_plane_creates_gpu_pod_with_graphql(self) -> None:
        response_payload = {
            "data": {
                "podFindAndDeployOnDemand": {
                    "id": "pod-1",
                    "desiredStatus": "RUNNING",
                    "costPerHr": 3.49,
                    "memoryInGb": 48,
                    "vcpuCount": 8,
                    "machine": {
                        "id": "machine-1",
                        "dataCenterId": "DC-1",
                        "gpuDisplayName": "A40",
                    },
                }
            }
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(response_payload).encode("utf-8")

        payload = {
            "name": "kura-test",
            "gpuCount": 1,
            "gpuTypeIds": ["NVIDIA H100 80GB HBM3"],
            "cloudType": "SECURE",
            "containerDiskInGb": 150,
            "volumeInGb": 0,
            "imageName": "registry/image@sha256:abc",
            "ports": ["22/tcp"],
            "env": {"KURA_RUN_ID": "example"},
            "dockerStartCmd": ["sh", "-lc", "sleep infinity"],
        }
        with patch("kura.executors.runpod.urlopen", return_value=Response()) as urlopen:
            result = _runpod_request("POST", "/pods", "api-secret", payload)

        self.assertEqual(result["id"], "pod-1")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.runpod.io/graphql")
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(request.unredirected_hdrs["Authorization"], "Bearer api-secret")
        body = json.loads(request.data)
        self.assertIn("podFindAndDeployOnDemand", body["query"])
        self.assertIn("machine { id dataCenterId", body["query"])
        gql_input = body["variables"]["input"]
        self.assertEqual(gql_input["gpuTypeId"], "NVIDIA H100 80GB HBM3")
        self.assertEqual(gql_input["ports"], "22/tcp")
        self.assertEqual(gql_input["env"], [{"key": "KURA_RUN_ID", "value": "example"}])
        self.assertEqual(gql_input["dockerArgs"], "sh -lc 'sleep infinity'")
        self.assertTrue(gql_input["startSsh"])

    def test_runpod_control_plane_rejects_multi_location_create_attempts(self) -> None:
        payload = {
            "gpuTypeIds": ["NVIDIA A40"],
            "dataCenterIds": ["DC-1", "DC-2"],
        }

        with self.assertRaisesRegex(ValueError, "at most one dataCenterIds"):
            _runpod_request("POST", "/pods", "api-secret", payload)

        payload = {
            "gpuTypeIds": ["NVIDIA A40"],
            "countryCodes": ["US", "CA"],
        }
        with self.assertRaisesRegex(ValueError, "at most one countryCodes"):
            _runpod_request("POST", "/pods", "api-secret", payload)

    def test_runpod_control_plane_gets_and_terminates_pod_with_graphql(self) -> None:
        responses = [
            {"data": {"pod": {"id": "pod-1", "desiredStatus": "RUNNING"}}},
            {"data": {"podTerminate": True}},
        ]

        class Response:
            def __init__(self, payload: dict[str, object]):
                self.payload = payload

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return json.dumps(self.payload).encode("utf-8")

        with patch("kura.executors.runpod.urlopen", side_effect=[Response(item) for item in responses]) as urlopen:
            pod = _runpod_request("GET", "/pods/pod-1", "api-secret")
            deleted = _runpod_request("DELETE", "/pods/pod-1", "api-secret")

        self.assertEqual(pod["desiredStatus"], "RUNNING")
        self.assertEqual(deleted, {})
        queries = [json.loads(call.args[0].data)["query"] for call in urlopen.call_args_list]
        self.assertIn("pod(input:", queries[0])
        self.assertIn("machine { id dataCenterId", queries[0])
        self.assertIn("podTerminate", queries[1])

    @staticmethod
    def _config() -> dict[str, object]:
        return {"storage_mode": "upload", "gpu_type_ids": ["NVIDIA A40"]}

    @staticmethod
    def _availability(*, available: bool, gpu: str = "NVIDIA A40", cloud: str = "COMMUNITY") -> dict[str, object]:
        return {
            "status": "ok",
            "checked_at": "2026-07-14T12:00:00+09:00",
            "gpu_count": 1,
            "candidates": [
                {
                    "gpu_type_id": gpu,
                    "display_name": gpu,
                    "memory_gb": 48,
                    "clouds": [{"cloud_type": cloud, "stock_status": "Low" if available else "None", "available": available, "price_per_hour": 0.4 if available else None, "available_gpu_counts": [1] if available else []}],
                }
            ],
        }

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
        stage_runpod(workspace=root, run_dir=run_dir, dataset_id="tiny", config={"runpod": self._config()})

    def test_launch_runpod_non_tty_requires_yes_before_any_runpod_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            with (
                patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False),
                patch("sys.stdin", io.StringIO()),
                patch("kura.executors.runpod.runpod_gpu_availability") as availability,
                patch("kura.executors.runpod._runpod_request") as request,
            ):
                with self.assertRaisesRegex(ValueError, r"--yes.*user has explicitly instructed"):
                    launch_runpod(
                        run_dir=run_dir,
                        spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                        image="registry/image:tag",
                        config=self._container_disk_config(),
                        max_lease_sec=12 * 3600,
                    )
            availability.assert_not_called()
            request.assert_not_called()

    def test_launch_runpod_session_asks_for_hosts_that_run_the_image(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            stdout = io.StringIO()
            with patch("sys.stdout", stdout), patch("kura.executors.runpod._runpod_request") as request:
                launch_runpod_session(run_dir=run_dir, image=PINNED_IMAGES["comfyui"], config=self._config(), purpose="comfyui-render", dry_run=True)
            request.assert_not_called()
            self.assertEqual(json.loads(stdout.getvalue())["runpod_create_request"]["minCudaVersion"], "13.0")

    def test_launch_runpod_session_non_tty_requires_yes_before_any_runpod_api_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            with (
                patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False),
                patch("sys.stdin", io.StringIO()),
                patch("kura.executors.runpod.runpod_gpu_availability") as availability,
                patch("kura.executors.runpod._runpod_request") as request,
            ):
                with self.assertRaisesRegex(ValueError, r"--yes.*user has explicitly instructed"):
                    launch_runpod_session(
                        run_dir=run_dir,
                        image="registry/comfy:tag",
                        config=self._config(),
                        purpose="comfyui-render",
                    )
            availability.assert_not_called()
            request.assert_not_called()

    def test_launch_runpod_interactive_confirmation_shows_cost_and_proceeds(self) -> None:
        class TTYInput(io.StringIO):
            def isatty(self) -> bool:
                return True

        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            stderr = io.StringIO()
            with (
                patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False),
                patch("sys.stdin", TTYInput("y\n")),
                patch("sys.stderr", stderr),
                patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)),
                patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request,
            ):
                realization_id = launch_runpod(
                    run_dir=run_dir,
                    spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                    image="registry/image:tag",
                    config=self._container_disk_config(),
                    wait_for_capacity_sec=6 * 3600,
                    max_lease_sec=12 * 3600,
                )
            self.assertIsNotNone(realization_id)
            request.assert_called_once()
            confirmation = stderr.getvalue()
            self.assertIn("GPU: NVIDIA A40 x1", confirmation)
            self.assertIn("Hourly price: COMMUNITY $0.400/hr", confirmation)
            self.assertIn("Maximum lease: 12h", confirmation)
            self.assertIn("Capacity wait: up to 6h; hourly prices may change while waiting", confirmation)
            self.assertIn("[y/N]", confirmation)

    def test_launch_runpod_interactive_default_cancels_before_pod_creation(self) -> None:
        class TTYInput(io.StringIO):
            def isatty(self) -> bool:
                return True

        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            with (
                patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False),
                patch("sys.stdin", TTYInput("\n")),
                patch("sys.stderr", io.StringIO()),
                patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)),
                patch("kura.executors.runpod._runpod_request") as request,
            ):
                with self.assertRaisesRegex(ValueError, "cancelled; no Pod was created"):
                    launch_runpod_session(
                        run_dir=run_dir,
                        image="registry/comfy:tag",
                        config=self._config(),
                        purpose="comfyui-render",
                    )
            request.assert_not_called()

    def test_launch_runpod_yes_proceeds_non_interactively(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            stderr = io.StringIO()
            with (
                patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False),
                patch("sys.stdin", io.StringIO()),
                patch("sys.stderr", stderr),
                patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)),
                patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request,
            ):
                realization_id = launch_runpod_session(
                    run_dir=run_dir,
                    image="registry/comfy:tag",
                    config=self._config(),
                    purpose="comfyui-render",
                    yes=True,
                )
            self.assertIsNotNone(realization_id)
            request.assert_called_once()
            confirmation = stderr.getvalue()
            self.assertIn("GPU: NVIDIA A40 x1", confirmation)
            self.assertIn("Hourly price: COMMUNITY $0.400/hr", confirmation)
            self.assertIn("Maximum lease: 12h", confirmation)
            self.assertNotIn("[y/N]", confirmation)

    def test_launch_runpod_records_pod_without_secret_value(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret", "HF_TOKEN": "hf-secret"}, clear=False):
                with patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)), \
                     patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request:
                    realization_id = launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=self._config(), yes=True)
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

    def test_launch_runpod_records_pod_creation_phases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)), \
                     patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}):
                    realization_id = launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=self._config(), yes=True)
            phases = launch_phases(run_dir, realization_id)
            self.assertEqual([item["phase"] for item in phases], ["pod_create_requested", "pod_created"])
            self.assertEqual(phases[1]["pod_id"], "pod-1")
            self.assertLessEqual(phases[0]["at"], phases[1]["at"])

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
                with patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)), \
                     patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request:
                    launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/app/ai-toolkit", "argv": ["python", "run.py"], "env": {}}, image="ostris/aitoolkit:latest", config=config, yes=True)
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
                with patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)), \
                     patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request:
                    realization_id = launch_runpod_session(run_dir=run_dir, image="registry/comfy:tag", config=self._config(), purpose="comfyui-render", yes=True)
            self.assertIsNotNone(realization_id)
            payload = request.call_args.args[3]
            self.assertEqual(payload["env"]["KURA_MAX_LEASE_SEC"], "43200")
            self.assertEqual(payload["env"]["HF_HOME"], "/workspace/cache/huggingface")
            self.assertEqual(payload["env"]["HF_HUB_CACHE"], "/workspace/cache/huggingface/hub")
            self.assertIn("kura_pod_self_delete", payload["dockerStartCmd"][2])
            self.assertIn("podTerminate", payload["dockerStartCmd"][2])
            self.assertNotIn("runpodctl pod delete", payload["dockerStartCmd"][2])
            syntax = subprocess.run(["sh", "-n"], input=payload["dockerStartCmd"][2], text=True, capture_output=True, check=False)
            self.assertEqual(syntax.returncode, 0, syntax.stderr)
            self.assertIn("RUNPOD_POD_ID", payload["dockerStartCmd"][2])
            self._assert_arms_lease_first(payload["dockerStartCmd"][2], max_lease_sec=12 * 3600, log_path="/workspace/runs/example/logs/stdout.log")

    def _assert_arms_lease_first(self, script: str, *, max_lease_sec: int, log_path: str) -> None:
        """Every start command Kura writes arms the same maximum lease before anything else."""
        from kura.executors.runpod import POD_SELF_DELETE_FUNCTION, _runpod_lease_guard_shell

        guard = _runpod_lease_guard_shell(max_lease_sec=max_lease_sec, pod_id="", log_path=log_path)
        self.assertTrue(script.startswith(POD_SELF_DELETE_FUNCTION + "\n" + guard + "\n"), script[:200])
        syntax = subprocess.run(["sh", "-n"], input=script, text=True, capture_output=True, check=False)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_launch_runpod_training_pod_arms_the_max_lease_at_start(self) -> None:
        log_path = "/workspace/runs/example/logs/stdout.log"
        for name in ("staging", "container_disk"):
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = self._run_dir(root)
                if name == "staging":
                    self._stage_upload(root, run_dir)
                config = self._config() if name == "staging" else self._container_disk_config()
                with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                    with patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)), \
                         patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request:
                        launch_runpod(run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                                      image="registry/image:tag", config=config, yes=True, max_lease_sec=3 * 3600)
                script = request.call_args.args[3]["dockerStartCmd"][2]
                # A Pod whose controller never reaches it is still bounded.
                self._assert_arms_lease_first(script, max_lease_sec=3 * 3600, log_path=log_path)

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
                with patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)), \
                     patch("kura.executors.runpod._runpod_request", return_value={
                         "id": "pod-1",
                         "desiredStatus": "RUNNING",
                         "memoryInGb": 48,
                         "vcpuCount": 8,
                         "machine": {
                             "id": "machine-1",
                             "dataCenterId": "US-GA-1",
                             "gpuDisplayName": "A40",
                         },
                     }) as request:
                    launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=config, yes=True)
            payload = request.call_args.args[3]
            # An image Kura does not know asks for the newest CUDA Kura has seen.
            self.assertEqual(payload["minCudaVersion"], NEWEST_KNOWN_CUDA)
            self.assertEqual(_runpod_graphql_create_input({**payload, "gpuTypeIds": ["NVIDIA A40"]})["minCudaVersion"], NEWEST_KNOWN_CUDA)
            self.assertEqual(payload["dataCenterIds"], ["US-GA-1"])
            self.assertEqual(payload["countryCodes"], ["US"])
            self.assertNotIn("dataCenterPriority", payload)
            self.assertNotIn("gpuTypePriority", payload)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
            self.assertEqual(realization["request"]["dataCenterCandidates"], ["US-GA-1"])
            self.assertEqual(realization["request"]["dataCenterPriority"], "availability")
            self.assertEqual(realization["request"]["gpuTypePriority"], "availability")
            self.assertEqual(realization["request"]["countryCandidates"], ["US"])
            self.assertEqual(realization["pod"]["machine"], {
                "id": "machine-1",
                "data_center_id": "US-GA-1",
                "gpu_display_name": "A40",
                "memory_gb": 48,
                "vcpu_count": 8,
            })

    def test_launch_runpod_falls_back_across_all_configured_locations(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {
                "storage_mode": "upload",
                "gpu_type_ids": ["NVIDIA A40"],
                "cloud_types": ["COMMUNITY"],
                "gpu_type_priority": "custom",
                "data_center_ids": ["DC-1", "DC-2"],
                "data_center_priority": "custom",
                "country_codes": ["US", "CA"],
            }
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)), \
                     patch(
                         "kura.executors.runpod._runpod_request",
                         side_effect=[
                             ValueError("no GPU capacity"),
                             ValueError("no GPU capacity"),
                             ValueError("no GPU capacity"),
                             {"id": "pod-1", "desiredStatus": "RUNNING"},
                         ],
                     ) as request:
                    launch_runpod(max_lease_sec=3600, 
                        run_dir=run_dir,
                        spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                        image="registry/image:tag",
                        config=config,
                        yes=True,
                    )

            placements = [
                (call.args[3]["dataCenterIds"], call.args[3]["countryCodes"])
                for call in request.call_args_list
            ]
            self.assertEqual(
                placements,
                [(["DC-1"], ["US"]), (["DC-1"], ["CA"]), (["DC-2"], ["US"]), (["DC-2"], ["CA"])],
            )

    def test_runpod_rejects_unrepresentable_availability_priority_lists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            run_dir.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "gpu_type_priority=availability.*multiple GPU"):
                launch_runpod_session(
                    run_dir=run_dir,
                    image="registry/image:tag",
                    purpose="review-test",
                    config={
                        "gpu_type_ids": ["NVIDIA A40", "NVIDIA RTX A5000"],
                        "gpu_type_priority": "availability",
                    },
                    dry_run=True,
                )
            with self.assertRaisesRegex(ValueError, "data_center_priority=availability.*multiple data centers"):
                launch_runpod_session(
                    run_dir=run_dir,
                    image="registry/image:tag",
                    purpose="review-test",
                    config={
                        "gpu_type_ids": ["NVIDIA A40"],
                        "data_center_ids": ["DC-1", "DC-2"],
                        "data_center_priority": "availability",
                    },
                    dry_run=True,
                )

    def test_launch_runpod_falls_back_across_cloud_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)), \
                     patch("kura.executors.runpod._runpod_request", side_effect=[ValueError("no community capacity"), {"id": "pod-1", "desiredStatus": "RUNNING"}]) as request:
                    launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=self._config(), yes=True)
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
                with patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)), \
                     patch("kura.executors.runpod._runpod_request", side_effect=[ValueError("no A5000 capacity"), {"id": "pod-1", "desiredStatus": "RUNNING"}]) as request:
                    launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=config, yes=True)
            first = request.call_args_list[0].args[3]
            second = request.call_args_list[1].args[3]
            self.assertEqual(first["gpuTypeIds"], ["NVIDIA RTX A5000"])
            self.assertEqual(second["gpuTypeIds"], ["NVIDIA A40"])

    def test_launch_runpod_waits_for_capacity_then_launches(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {**self._config(), "cloud_types": ["COMMUNITY"]}
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with (
                    patch("kura.executors.runpod.runpod_gpu_availability", side_effect=[self._availability(available=False), self._availability(available=True)]) as probe,
                    patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request,
                    patch("kura.executors.runpod.time.sleep") as sleep,
                ):
                    realization_id = launch_runpod(max_lease_sec=3600, 
                        run_dir=run_dir,
                        spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                        image="registry/image:tag",
                        config=config,
                        wait_for_capacity_sec=60,
                        capacity_poll_interval_sec=5,
                        yes=True,
                    )
            self.assertIsNotNone(realization_id)
            self.assertEqual(probe.call_count, 2)
            self.assertEqual(request.call_count, 1)
            sleep.assert_called_once_with(5)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "running")
            self.assertNotIn("capacity_wait", status)
            realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
            self.assertEqual(realization["request"]["capacityWait"]["failedRounds"], 1)
            events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn("runpod_capacity_wait_started", [item["event"] for item in events])
            self.assertIn("runpod_capacity_acquired", [item["event"] for item in events])

    def test_launch_runpod_does_not_wait_for_non_capacity_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {**self._config(), "cloud_types": ["COMMUNITY"]}
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with (
                    patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)),
                    patch("kura.executors.runpod._runpod_request", side_effect=ValueError("invalid template")),
                    patch("kura.executors.runpod.time.sleep") as sleep,
                ):
                    with self.assertRaisesRegex(ValueError, "invalid template"):
                        launch_runpod(max_lease_sec=3600, 
                            run_dir=run_dir,
                            spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                            image="registry/image:tag",
                            config=config,
                            wait_for_capacity_sec=60,
                            capacity_poll_interval_sec=5,
                            yes=True,
                        )
            sleep.assert_not_called()
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "launch_failed")

    def test_launch_runpod_wait_backs_off_after_a_rate_limited_create(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {**self._config(), "cloud_types": ["COMMUNITY"]}
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with (
                    patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)) as probe,
                    patch("kura.executors.runpod._runpod_request", side_effect=[RunPodAPIError("RunPod GraphQL failed (429): slow down", status_code=429), {"id": "pod-1", "desiredStatus": "RUNNING"}]) as request,
                    patch("kura.executors.runpod._runpod_pods_named") as listed,
                    patch("kura.executors.runpod.time.sleep") as sleep,
                ):
                    realization_id = launch_runpod(max_lease_sec=3600, 
                        run_dir=run_dir,
                        spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                        image="registry/image:tag",
                        config=config,
                        wait_for_capacity_sec=60,
                        capacity_poll_interval_sec=5,
                        yes=True,
                    )
            self.assertIsNotNone(realization_id)
            self.assertEqual(probe.call_count, 2)
            self.assertEqual(request.call_count, 2)
            listed.assert_not_called()
            sleep.assert_called_once_with(10)

    def test_an_unconfirmed_create_is_never_retried_and_is_handed_over(self) -> None:
        from kura.executors.runpod import unresolved_create_intents

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {**self._config(), "cloud_types": ["COMMUNITY"]}
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with (
                    patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)),
                    patch("kura.executors.runpod._runpod_request", side_effect=RunPodAPIError("RunPod GraphQL failed (503): unavailable", status_code=503)) as request,
                    patch("kura.executors.runpod._runpod_pods_named", return_value=[]) as listed,
                    patch("kura.executors.runpod.time.sleep"),
                ):
                    with self.assertRaisesRegex(ValueError, "kura run reconcile"):
                        launch_runpod(max_lease_sec=3600, 
                            run_dir=run_dir,
                            spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                            image="registry/image:tag",
                            config=config,
                            wait_for_capacity_sec=60,
                            capacity_poll_interval_sec=5,
                            yes=True,
                        )
            self.assertEqual(request.call_count, 1)
            self.assertEqual(listed.call_count, 2)
            self.assertEqual(len(unresolved_create_intents(run_dir)), 1)
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "interrupted")

    def test_an_unconfirmed_create_that_did_succeed_is_adopted_without_a_second_create(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {**self._config(), "cloud_types": ["COMMUNITY"]}
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with (
                    patch("kura.executors.runpod._runpod_request", side_effect=ValueError("RunPod API is unreachable: timed out")) as request,
                    patch("kura.executors.runpod._runpod_pods_named", side_effect=lambda _key, name: [{"id": "pod-late", "name": name, "desiredStatus": "RUNNING"}]),
                    patch("kura.executors.runpod.time.sleep"),
                ):
                    realization_id = launch_runpod(max_lease_sec=3600, 
                        run_dir=run_dir,
                        spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                        image="registry/image:tag",
                        config=config,
                        yes=True,
                    )
            self.assertEqual(request.call_count, 1)
            realization = json.loads((run_dir / "realizations" / f"{realization_id}.json").read_text(encoding="utf-8"))
            self.assertEqual(realization["pod"]["id"], "pod-late")
            self.assertEqual(realization["create_intent"], f"{realization_id}.create-intent.json")
            self.assertTrue((run_dir / "realizations" / realization["create_intent"]).is_file())

    def test_cancelling_a_capacity_wait_after_refused_creates_settles_the_intent(self) -> None:
        from kura.executors.runpod import unresolved_create_intents

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {**self._config(), "cloud_types": ["COMMUNITY"]}
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with (
                    patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)),
                    patch("kura.executors.runpod._runpod_request", side_effect=RunPodAPIError("RunPod GraphQL failed: There are no longer any instances available with the requested specifications", status_code=400)),
                    patch("kura.executors.runpod.time.sleep", side_effect=KeyboardInterrupt),
                ):
                    with self.assertRaisesRegex(ValueError, "no Pod was created"):
                        launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=config, wait_for_capacity_sec=60, capacity_poll_interval_sec=5, yes=True)
            self.assertEqual(unresolved_create_intents(run_dir), [])
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "launch_failed")

    def test_the_create_intent_is_written_before_the_create_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            seen: list[bool] = []

            def create(*_args, **_kwargs):
                seen.append(any((run_dir / "realizations").glob("*.create-intent.json")))
                return {"id": "pod-1", "desiredStatus": "RUNNING"}

            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), patch("kura.executors.runpod._runpod_request", side_effect=create):
                launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=self._config(), yes=True)
            self.assertEqual(seen, [True])

    def test_launch_runpod_wait_does_not_create_while_probe_is_rate_limited(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {**self._config(), "cloud_types": ["COMMUNITY"]}
            rate_limited = {"status": "unavailable", "error_kind": "rate_limit", "reason": "RunPod GraphQL failed (429)", "candidates": []}
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with (
                    patch("kura.executors.runpod.runpod_gpu_availability", side_effect=[rate_limited, self._availability(available=True)]) as probe,
                    patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request,
                    patch("kura.executors.runpod.time.sleep") as sleep,
                ):
                    realization_id = launch_runpod(max_lease_sec=3600, 
                        run_dir=run_dir,
                        spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                        image="registry/image:tag",
                        config=config,
                        wait_for_capacity_sec=60,
                        capacity_poll_interval_sec=5,
                        yes=True,
                    )
            self.assertIsNotNone(realization_id)
            self.assertEqual(probe.call_count, 2)
            request.assert_called_once()
            sleep.assert_called_once_with(10)

    def test_launch_runpod_capacity_wait_timeout_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {**self._config(), "cloud_types": ["COMMUNITY"]}
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with (
                    patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=False)),
                    patch("kura.executors.runpod._runpod_request") as request,
                    patch("kura.executors.runpod.time.monotonic", side_effect=[0, 0, 31]),
                    patch("kura.executors.runpod.time.sleep") as sleep,
                ):
                    with self.assertRaisesRegex(ValueError, "stock snapshot reports no matching GPU capacity"):
                        launch_runpod(max_lease_sec=3600, 
                            run_dir=run_dir,
                            spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                            image="registry/image:tag",
                            config=config,
                            wait_for_capacity_sec=30,
                            capacity_poll_interval_sec=5,
                            yes=True,
                        )
            request.assert_not_called()
            sleep.assert_called_once_with(5)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "launch_failed")
            self.assertNotIn("capacity_wait", status)
            realization = json.loads((run_dir / status["last_realization"]).read_text(encoding="utf-8"))
            self.assertEqual(realization["request"]["launch_attempts"][-1]["classification"], "capacity")
            wait = [json.loads(line) for line in (run_dir / "realizations" / f"{realization['id']}.capacity-wait.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([line["kind"] for line in wait], ["capacity_wait_round", "capacity_wait_end"])
            self.assertEqual((wait[0]["attempts"], wait[-1]["outcome"]), (1, "gave_up"))

    def test_launch_runpod_capacity_wait_stops_on_probe_auth_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {**self._config(), "cloud_types": ["COMMUNITY"]}
            auth_error = {
                "status": "unavailable",
                "error_kind": "auth",
                "reason": "RunPod GraphQL failed (401)",
                "candidates": [],
            }
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with (
                    patch("kura.executors.runpod.runpod_gpu_availability", return_value=auth_error) as probe,
                    patch("kura.executors.runpod._runpod_request") as request,
                    patch("kura.executors.runpod.time.sleep") as sleep,
                ):
                    with self.assertRaisesRegex(ValueError, "GraphQL failed.*401"):
                        launch_runpod(max_lease_sec=3600, 
                            run_dir=run_dir,
                            spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                            image="registry/image:tag",
                            config=config,
                            wait_for_capacity_sec=60,
                            capacity_poll_interval_sec=5,
                            yes=True,
                        )
            probe.assert_called_once()
            request.assert_not_called()
            sleep.assert_not_called()
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "launch_failed")
            self.assertNotIn("capacity_wait", status)

    def test_capacity_error_classifier_rejects_disk_capacity_errors(self) -> None:
        self.assertFalse(_is_runpod_capacity_error(ValueError("container disk capacity exceeded")))
        self.assertTrue(_is_runpod_capacity_error(ValueError("no GPU capacity is currently available")))

    def test_launch_runpod_capacity_wait_can_be_cancelled_before_pod_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {**self._config(), "cloud_types": ["COMMUNITY"]}
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with (
                    patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=False)),
                    patch("kura.executors.runpod._runpod_request") as request,
                    patch("kura.executors.runpod.time.sleep", side_effect=KeyboardInterrupt),
                ):
                    with self.assertRaisesRegex(ValueError, "no Pod was created"):
                        launch_runpod(max_lease_sec=3600, 
                            run_dir=run_dir,
                            spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                            image="registry/image:tag",
                            config=config,
                            wait_for_capacity_sec=60,
                            capacity_poll_interval_sec=5,
                            yes=True,
                        )
            request.assert_not_called()
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "interrupted")
            self.assertNotIn("pod_id", status)
            self.assertNotIn("capacity_wait", status)
            [wait_path] = list((run_dir / "realizations").glob("*.capacity-wait.jsonl"))
            wait = [json.loads(line) for line in wait_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([(line["kind"], line.get("outcome")) for line in wait], [("capacity_wait_round", None), ("capacity_wait_end", "cancelled")])
            self.assertTrue(wait_path.read_text(encoding="utf-8").endswith("\n"))

    def test_an_unconfirmed_create_during_a_capacity_wait_still_ends_the_wait(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {**self._config(), "cloud_types": ["COMMUNITY"]}
            availability = [self._availability(available=False), self._availability(available=True)]
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with (
                    patch("kura.executors.runpod.runpod_gpu_availability", side_effect=availability),
                    patch("kura.executors.runpod._runpod_request", side_effect=ValueError("RunPod API is unreachable: timed out")),
                    patch("kura.executors.runpod._discover_pods", return_value=[]),
                    patch("kura.executors.runpod.time.sleep"),
                ):
                    with self.assertRaisesRegex(ValueError, "did not confirm"):
                        launch_runpod(max_lease_sec=3600, 
                            run_dir=run_dir,
                            spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                            image="registry/image:tag",
                            config=config,
                            wait_for_capacity_sec=600,
                            capacity_poll_interval_sec=5,
                            yes=True,
                        )
            [wait_path] = list((run_dir / "realizations").glob("*.capacity-wait.jsonl"))
            wait = [json.loads(line) for line in wait_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual((wait[-1]["kind"], wait[-1]["outcome"]), ("capacity_wait_end", "abandoned"))
            self.assertEqual(sum(line["kind"] == "capacity_wait_end" for line in wait), 1)
            self.assertEqual(len(list((run_dir / "realizations").glob("*.create-unconfirmed.json"))), 1)

    def test_launch_runpod_capacity_wait_records_create_phase_interruption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._run_dir(root)
            self._stage_upload(root, run_dir)
            config = {**self._config(), "cloud_types": ["COMMUNITY"]}
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with (
                    patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)),
                    patch("kura.executors.runpod._runpod_request", side_effect=KeyboardInterrupt),
                ):
                    with self.assertRaisesRegex(ValueError, "creation is unconfirmed"):
                        launch_runpod(max_lease_sec=3600, 
                            run_dir=run_dir,
                            spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}},
                            image="registry/image:tag",
                            config=config,
                            wait_for_capacity_sec=60,
                            capacity_poll_interval_sec=5,
                            yes=True,
                        )
            events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[-1]["event"], "runpod_capacity_wait_cancelled")
            self.assertEqual(events[-1]["phase"], "create")
            # The interruption is recorded beside the intent, which stays for discovery.
            [unconfirmed] = list((run_dir / "realizations").glob("*.create-unconfirmed.json"))
            self.assertEqual(json.loads(unconfirmed.read_text(encoding="utf-8"))["kind"], "create_unconfirmed")
            self.assertEqual(len(unresolved_create_intents(run_dir)), 1)

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
                                                "compute": {"executor": "runpod", "gpu": "NVIDIA A40"},
                        "backend": {"name": "ai-toolkit", "config": {"command": {"cwd": "/workspace", "argv": ["python", "-c", "print(1)"], "env": {}}}},
                    }
                ),
                encoding="utf-8",
            )
            (root / "runs" / "example" / "resolved" / "backend-command.lock.json").write_text(json.dumps({"backend": "ai-toolkit", "adapter_source": {"kind": "test", "value": "test"}, "cwd": "/app/ai-toolkit", "argv": ["python", "-c", "print(1)"], "env": {}}), encoding="utf-8")
            (root / "workspace.yaml").write_text(
                yaml.safe_dump(
                    {'runpod': {'storage_mode': 'upload', 'gpu_type_ids': ['NVIDIA RTX A5000', 'NVIDIA A40'], 'cloud_type': 'COMMUNITY', 'template_id': 'mutable-template'}, 'docker': {}, 'images': {'ai-toolkit': 'local'}}
                ),
                encoding="utf-8",
            )
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch("kura.run_commands.launch.launch_runpod") as launch:
                    self.assertEqual(launch_run("example", executor="runpod", dry_run=False, image=None, yes=True), 0)
            finally:
                os.chdir(previous)
            self.assertEqual(launch.call_args.kwargs["config"]["gpu_type_ids"], ["NVIDIA A40"])
            self.assertTrue(launch.call_args.kwargs["yes"])
            self.assertEqual(launch.call_args.kwargs["config"]["gpu_type_priority"], "custom")
            self.assertNotIn("template_id", launch.call_args.kwargs["config"])
            self.assertEqual(launch.call_args.kwargs["config"]["ports"], ["8675/http", "22/tcp"])

    def test_run_launch_rejects_a_second_capacity_wait_controller(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = self._compiled_runpod_launch_workspace(root, {"state": "compiled"})
            previous = Path.cwd()
            try:
                os.chdir(root)
                with file_lock(run_dir / ".locks" / "runpod-launch.lock", blocking=False):
                    with patch("kura.run_commands.launch.launch_runpod") as launch, patch("sys.stderr", new_callable=io.StringIO) as stderr:
                        code = launch_run("example", executor="runpod", dry_run=False, image=None)
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            launch.assert_not_called()
            self.assertIn("another operation already owns runpod-launch.lock", stderr.getvalue())

    def test_run_launch_recovers_a_stale_capacity_wait(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._compiled_runpod_launch_workspace(
                root,
                {"state": "queued", "capacity_wait": {"last_attempt_at": "2026-07-14T00:00:00+09:00", "poll_interval_sec": 30}},
            )
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch("kura.run_commands.launch.launch_runpod") as launch:
                    code = launch_run("example", executor="runpod", dry_run=False, image=None)
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            launch.assert_called_once()

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
                launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/app/ai-toolkit", "argv": ["python", "run.py"], "env": {}}, image="ostris/aitoolkit:latest", config=config, dry_run=True)

    def test_launch_runpod_object_staging_is_disabled_until_secret_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            env = {"RUNPOD_API_KEY": "api-secret", "R2_ACCESS_KEY_ID": "r2-access", "R2_SECRET_ACCESS_KEY": "r2-secret"}
            with patch.dict(os.environ, env, clear=False):
                with self.assertRaisesRegex(ValueError, "disabled until object-store credentials"):
                    launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=self._object_config())

    def test_launch_runpod_records_failed_attempt_without_stale_pod(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "status.json").write_text(json.dumps({"state": "interrupted", "pod_id": "stale-pod", "last_observation": "realizations/old.observed.json"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret", "HF_TOKEN": "hf-secret"}, clear=False):
                with patch("kura.executors.runpod.runpod_gpu_availability", return_value=self._availability(available=True)), \
                     patch("kura.executors.runpod._runpod_request", side_effect=ValueError("RunPod API POST /pods failed (500): echoed api-secret hf-secret")):
                    with self.assertRaisesRegex(ValueError, r"\\*\\*\\*"):
                        launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, image="registry/image:tag", config=self._container_disk_config(), yes=True)
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
                launch_runpod(max_lease_sec=3600, run_dir=run_dir, spec={"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {"HF_TOKEN": "should-not-enter-pod-env"}}, image="registry/image:tag", config=self._container_disk_config())

    def test_runpod_ssh_secret_injection_keeps_token_out_of_argv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "resolved").mkdir()
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("backend:\n  name: ai-toolkit\n", encoding="utf-8")
            (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({
                "backend": "ai-toolkit", "cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {"SEED": "1"},
                "adapter_source": {"kind": "test", "value": "test"},
            }), encoding="utf-8")
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
                    # The intent is on disk before the job starts, and names the pid file the Pod keeps.
                    intent = json.loads((run_dir / "realizations" / "r1.remote-job-intent.json").read_text(encoding="utf-8"))
                    self.assertIn(intent["pid_path"], command_text)
                    self.assertFalse((run_dir / "realizations" / "r1.remote-job.json").exists())
                    return subprocess.CompletedProcess(args[0], 0, "1234\n", "")
                if "remote-exit-*.json" in command_text:
                    return subprocess.CompletedProcess(args[0], 0, json.dumps({"event": "remote_exit", "exit_code": 0}), "")
                if "__KURA_LOG_SIZE__" in command_text:
                    return subprocess.CompletedProcess(args[0], 0, b"\n__KURA_LOG_SIZE__:0\n", b"")
                return subprocess.CompletedProcess(args[0], 0, "", "")

            with patch.dict(os.environ, {"HF_TOKEN": "hf-secret"}, clear=False):
                with patch("kura.run_commands.runpod_ssh._runpod_ssh_details", return_value={"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}), \
                     patch("kura.run_commands.runpod_ssh._try_sync_runpod_checkpoints", return_value=True) as checkpoint_sync:
                    with patch("kura.cli.subprocess.run", side_effect=fake_run):
                        self.assertEqual(_runpod_run_over_ssh(run_dir, ssh_timeout_sec=1, job_timeout_sec=1), 0)
            job = json.loads((run_dir / "realizations" / "r1.remote-job.json").read_text(encoding="utf-8"))
            self.assertEqual((job["kind"], job["pid"], job["intent"]), ("remote_job", "1234", "r1.remote-job-intent.json"))

            checkpoint_sync.assert_called_with(
                run_dir,
                {"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"},
                workspace="/workspace",
                run_id="example",
            )

            argv_text = "\n".join(" ".join(map(str, call[0][0])) if isinstance(call[0][0], list) else str(call[0][0]) for call in calls)
            self.assertNotIn("hf-secret", argv_text)
            self.assertIn("kura_lease_initial=$(( $(date +%s) + 43200 ))", argv_text)
            self.assertIn("RUNPOD_POD_ID=pod-1", argv_text)
            self.assertIn("kura_pod_self_delete", argv_text)
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

    def test_runpod_ssh_run_records_transport_phases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "resolved").mkdir()
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("backend:\n  name: ai-toolkit\n", encoding="utf-8")
            (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({
                "backend": "ai-toolkit", "cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {},
                "adapter_source": {"kind": "test", "value": "test"},
            }), encoding="utf-8")
            (run_dir / "transfer").mkdir()
            (run_dir / "transfer" / "bundle.tar.gz").write_bytes(b"bundle")
            (run_dir / "realizations" / "stage.json").write_text(json.dumps({"storage_mode": "upload", "archive": "transfer/bundle.tar.gz", "archive_name": "bundle.tar.gz"}), encoding="utf-8")
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "request": {"env": {"KURA_WORKSPACE": "/workspace"}}, "container_cwd": "/opt/tool", "backend_command": ["python", "train.py"]}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"pod_id": "pod-1", "last_stage": "realizations/stage.json", "last_realization": "realizations/r1.json"}), encoding="utf-8")

            def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                command_text = " ".join(map(str, args[0])) if isinstance(args[0], list) else str(args[0])
                if "nohup sh" in command_text:
                    return subprocess.CompletedProcess(args[0], 0, "1234\n", "")
                if "remote-exit-*.json" in command_text:
                    return subprocess.CompletedProcess(args[0], 0, json.dumps({"event": "remote_exit", "exit_code": 0}), "")
                if "__KURA_LOG_SIZE__" in command_text:
                    return subprocess.CompletedProcess(args[0], 0, b"\n__KURA_LOG_SIZE__:0\n", b"")
                return subprocess.CompletedProcess(args[0], 0, "", "")

            details = {"ip": "127.0.0.1", "port": 22, "key": "/tmp/key", "container_started_at": "2026-10-01T03:47:30Z"}
            with patch("kura.run_commands.runpod_ssh._runpod_ssh_details", return_value=details), \
                 patch("kura.run_commands.runpod_ssh._try_sync_runpod_checkpoints", return_value=True), \
                 patch("kura.cli.subprocess.run", side_effect=fake_run):
                self.assertEqual(_runpod_run_over_ssh(run_dir, ssh_timeout_sec=1, job_timeout_sec=1), 0)

            phases = launch_phases(run_dir, "r1")
            self.assertEqual(
                [item["phase"] for item in phases],
                ["ssh_ready", "upload_started", "upload_finished", "remote_job_started", "remote_exit_observed"],
            )
            self.assertEqual(phases[0]["container_started_at"], "2026-10-01T03:47:30Z")

    def test_runpod_ssh_run_notices_remote_exit_within_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "resolved").mkdir()
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("backend:\n  name: ai-toolkit\n", encoding="utf-8")
            (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({
                "backend": "ai-toolkit", "cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {},
                "adapter_source": {"kind": "test", "value": "test"},
            }), encoding="utf-8")
            (run_dir / "transfer").mkdir()
            (run_dir / "transfer" / "bundle.tar.gz").write_bytes(b"bundle")
            (run_dir / "realizations" / "stage.json").write_text(json.dumps({"storage_mode": "upload", "archive": "transfer/bundle.tar.gz", "archive_name": "bundle.tar.gz"}), encoding="utf-8")
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "request": {"env": {"KURA_WORKSPACE": "/workspace"}}, "container_cwd": "/opt/tool", "backend_command": ["python", "train.py"]}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"pod_id": "pod-1", "last_stage": "realizations/stage.json", "last_realization": "realizations/r1.json"}), encoding="utf-8")
            clock = {"now": 1000.0}
            exit_at = 1006.0
            syncs: list[float] = []

            def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                command_text = " ".join(map(str, args[0])) if isinstance(args[0], list) else str(args[0])
                return subprocess.CompletedProcess(args[0], 0, "1234\n" if "nohup sh" in command_text else "", "")

            def read_exit(*_args: object, **_kwargs: object) -> dict[str, Any] | None:
                return {"event": "remote_exit", "exit_code": 0} if clock["now"] >= exit_at else None

            def sleep(seconds: float) -> None:
                clock["now"] += seconds

            with patch("kura.run_commands.runpod_ssh._runpod_ssh_details", return_value={"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}), \
                 patch("kura.run_commands.runpod_ssh._try_sync_runpod_checkpoints", return_value=True), \
                 patch("kura.run_commands.runpod_ssh._sync_runpod_remote_stdout", side_effect=lambda *a, **k: syncs.append(clock["now"]) or True), \
                 patch("kura.run_commands.runpod_ssh._read_runpod_remote_exit", side_effect=read_exit), \
                 patch("kura.run_commands.runpod_ssh.time.monotonic", side_effect=lambda: clock["now"]), \
                 patch("kura.run_commands.runpod_ssh.time.sleep", side_effect=sleep), \
                 patch("kura.cli.subprocess.run", side_effect=fake_run):
                self.assertEqual(_runpod_run_over_ssh(run_dir, ssh_timeout_sec=1, job_timeout_sec=0), 0)

        # The exit is noticed within one short check interval, not the 20s log sync.
        self.assertLessEqual(clock["now"] - exit_at, 5)
        # The heavier log sync keeps its own cadence: once at start and once after exit.
        self.assertEqual(len(syncs), 2)

    def test_runpod_remote_job_diagnostics_script_has_valid_shell_syntax(self) -> None:
        script = _runpod_remote_job_script(
            workspace="/workspace",
            run_id="example",
            realization_id="r1",
            remote_secret_path="/tmp/kura-secrets/example.env",
            archive_name="bundle.tar.gz",
            remote_archive="/workspace/bundle.tar.gz",
            cwd="/opt/tool",
            command="true",
        )
        result = subprocess.run(["sh", "-n"], input=script, text=True, capture_output=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runpod_ssh_always_arms_the_pod_side_max_lease_guard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "resolved").mkdir()
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("backend:\n  name: ai-toolkit\n", encoding="utf-8")
            (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({
                "backend": "ai-toolkit", "cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {"SEED": "1"},
                "adapter_source": {"kind": "test", "value": "test"},
            }), encoding="utf-8")
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
                    self.assertEqual(_runpod_run_over_ssh(run_dir, ssh_timeout_sec=1, job_timeout_sec=1, max_lease_sec=3600), 0)

            argv_text = "\n".join(" ".join(map(str, call[0][0])) if isinstance(call[0][0], list) else str(call[0][0]) for call in calls)
            self.assertIn("kura_lease_initial=$(( $(date +%s) + 3600 ))", argv_text)

    def test_runpod_training_records_the_deadline_the_pod_holds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "logs").mkdir()
            (run_dir / "resolved").mkdir()
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("backend:\n  name: ai-toolkit\n", encoding="utf-8")
            (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({
                "backend": "ai-toolkit", "cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {"SEED": "1"},
                "adapter_source": {"kind": "test", "value": "test"},
            }), encoding="utf-8")
            (run_dir / "transfer").mkdir()
            (run_dir / "transfer" / "bundle.tar.gz").write_bytes(b"bundle")
            (run_dir / "realizations" / "stage.json").write_text(json.dumps({"storage_mode": "upload", "archive": "transfer/bundle.tar.gz", "archive_name": "bundle.tar.gz"}), encoding="utf-8")
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"executor": "runpod", "request": {"env": {"KURA_WORKSPACE": "/workspace"}}, "container_cwd": "/opt/tool", "backend_command": ["python", "train.py"]}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"pod_id": "pod-1", "last_stage": "realizations/stage.json", "last_realization": "realizations/r1.json"}), encoding="utf-8")
            # The Pod set this when it started, hours before the controller reached it.
            held = 1_900_000_000

            def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
                command_text = " ".join(map(str, args[0])) if isinstance(args[0], list) else str(args[0])
                if command_text.endswith("cat /tmp/kura-lease-deadline"):
                    return subprocess.CompletedProcess(args[0], 0, f"{held}\n", "")
                if "nohup sh" in command_text:
                    return subprocess.CompletedProcess(args[0], 0, "1234\n", "")
                if "remote-exit-*.json" in command_text:
                    return subprocess.CompletedProcess(args[0], 0, json.dumps({"event": "remote_exit", "exit_code": 0}), "")
                if "__KURA_LOG_SIZE__" in command_text:
                    return subprocess.CompletedProcess(args[0], 0, b"\n__KURA_LOG_SIZE__:0\n", b"")
                return subprocess.CompletedProcess(args[0], 0, "", "")

            with patch("kura.run_commands.runpod_ssh._runpod_ssh_details", return_value={"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}):
                with patch("kura.cli.subprocess.run", side_effect=fake_run):
                    self.assertEqual(_runpod_run_over_ssh(run_dir, ssh_timeout_sec=1, job_timeout_sec=1, max_lease_sec=3600), 0)
            [lease] = [json.loads(path.read_text(encoding="utf-8")) for path in (run_dir / "realizations").glob("r1.lease-*.json")]
            self.assertEqual((lease["deadline_epoch"], lease["reason"]), (held, "armed"))

    def test_training_and_render_record_the_lease_with_one_function(self) -> None:
        from kura.run_commands import render_runpod

        self.assertIs(render_runpod.record_pod_lease_deadline, runpod_ssh_module.record_pod_lease_deadline)
        self.assertFalse(hasattr(render_runpod, "_record_session_lease"))
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"last_realization": "realizations/r1.json"}), encoding="utf-8")
            details = {"pod_id": "pod-1", "ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}
            with patch("kura.run_commands.runpod_ssh.subprocess.run", return_value=subprocess.CompletedProcess(["ssh"], 0, "", "")):
                runpod_ssh_module.record_pod_lease_deadline(run_dir, details)
            # A Pod without a deadline file (lease disabled) records nothing.
            self.assertEqual(list((run_dir / "realizations").glob("r1.lease-*.json")), [])
            with patch("kura.run_commands.runpod_ssh.subprocess.run", return_value=subprocess.CompletedProcess(["ssh"], 0, "1900000000\n", "")) as run:
                runpod_ssh_module.record_pod_lease_deadline(run_dir, details)
            self.assertTrue(run.call_args.args[0][-1].endswith("cat /tmp/kura-lease-deadline"))
            [lease] = [json.loads(path.read_text(encoding="utf-8")) for path in (run_dir / "realizations").glob("r1.lease-*.json")]
            self.assertEqual((lease["deadline_epoch"], lease["reason"]), (1_900_000_000, "armed"))

    def test_runpod_render_session_starts_lease_guard_over_ssh(self) -> None:
        details = {"pod_id": "pod-1", "ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}
        with patch("kura.run_commands.runpod_ssh.subprocess.run", return_value=subprocess.CompletedProcess(["ssh"], 0, "", "")) as run:
            _start_runpod_session_lease_guard(details, workspace="/workspace", run_id="render-1", max_lease_sec=60)
        command = run.call_args.args[0]
        command_text = "\n".join(map(str, command))
        self.assertIn("kura_lease_initial=$(( $(date +%s) + 60 ))", command_text)
        self.assertIn("kura_pod_self_delete", command_text)
        self.assertNotIn("runpodctl pod delete", command_text)
        self.assertIn("pod-1", command_text)
        self.assertIn("/workspace/runs/render-1/logs/stdout.log", command_text)
        self.assertEqual(run.call_args.kwargs["timeout"], 60)

    def test_runpod_render_session_lease_guard_timeout_surfaces_as_value_error(self) -> None:
        details = {"pod_id": "pod-1", "ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}
        with patch("kura.run_commands.runpod_ssh.subprocess.run", side_effect=subprocess.TimeoutExpired(["ssh"], 60)):
            with self.assertRaisesRegex(ValueError, "remote lease guard setup timed out"):
                _start_runpod_session_lease_guard(details, workspace="/workspace", run_id="render-1", max_lease_sec=60)

    def test_runpod_comfyui_start_prepares_models_before_the_server(self) -> None:
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
            )
        script = run.call_args.args[0][-1]
        self.assertIn("trap cleanup EXIT", script)
        self.assertIn('rm -f "$secret_file"', script)
        # The Pod's own deadline-file guard bounds it; no second, fixed timer.
        self.assertNotIn("runpodctl", script)
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
                with patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "EXITED"}) as request:
                    status = reconcile_runpod(run_dir, self._config())
            self.assertEqual(status["state"], "unknown")
            self.assertIsNone(status["exit_code"])
            request.assert_called_once_with("GET", "/pods/pod-1", "api-secret", timeout=30.0)

    def test_reconcile_runpod_records_a_pod_that_no_longer_exists(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/r1.json"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", side_effect=RunPodAPIError("RunPod Pod not found", status_code=404)):
                    status = reconcile_runpod(run_dir, self._config())
            observation = json.loads((run_dir / status["last_observation"]).read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "interrupted")
        self.assertIn("pod_missing_at", status)
        self.assertTrue(observation["pod_missing"])

    def test_automatic_reconcile_never_marks_a_pod_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/r1.json"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", side_effect=RunPodAPIError("RunPod Pod not found", status_code=404)):
                    with self.assertRaises(RunPodAPIError):
                        reconcile_runpod(run_dir, self._config(), source="automatic")
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "running")

    def test_reconcile_runpod_still_raises_other_api_errors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/r1.json"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", side_effect=RunPodAPIError("RunPod GraphQL failed (500)", status_code=500)):
                    with self.assertRaises(RunPodAPIError):
                        reconcile_runpod(run_dir, self._config())

    def test_reconcile_runpod_preserves_confirmed_terminal_outcome(self) -> None:
        for terminal_state, exit_code in (("completed", 0), ("failed", 7), ("stopped", None), ("interrupted", None), ("unknown", None), ("launch_failed", None)):
            with self.subTest(state=terminal_state), tempfile.TemporaryDirectory() as directory:
                run_dir = self._run_dir(Path(directory))
                realization_ref = "realizations/r1.json"
                (run_dir / realization_ref).write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}}), encoding="utf-8")
                (run_dir / "status.json").write_text(
                    json.dumps({"state": terminal_state, "exit_code": exit_code, "ended": "confirmed-end", "last_realization": realization_ref, "pod_id": "pod-1"}),
                    encoding="utf-8",
                )
                with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                    with patch("kura.executors.runpod._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}):
                        status = reconcile_runpod(run_dir, self._config())

                self.assertEqual(status["state"], terminal_state)
                self.assertEqual(status["exit_code"], exit_code)
                self.assertEqual(status["ended"], "confirmed-end")
                observation = json.loads((run_dir / status["last_observation"]).read_text(encoding="utf-8"))
                self.assertEqual(observation["state"], "running")
                events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(events[-1]["event"], "run_reconciled")
                self.assertEqual(events[-1]["state"], "running")

    def test_reconcile_runpod_merges_observation_into_latest_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            realization_ref = "realizations/r1.json"
            (run_dir / realization_ref).write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": realization_ref, "pod_id": "pod-1"}), encoding="utf-8")
            request_started = threading.Event()
            release_request = threading.Event()
            result: list[dict[str, Any]] = []

            def observe(*_args: object, **_kwargs: object) -> dict[str, object]:
                request_started.set()
                self.assertTrue(release_request.wait(2))
                return {"id": "pod-1", "desiredStatus": "RUNNING"}

            def reconcile() -> None:
                result.append(reconcile_runpod(run_dir, self._config()))

            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", side_effect=observe):
                    thread = threading.Thread(target=reconcile)
                    thread.start()
                    self.assertTrue(request_started.wait(2))
                    _mutate_run_status(
                        run_dir,
                        lambda status: status.update({"remote_log_bytes": 77, "mirrored_outputs": [{"name": "step.safetensors"}]}),
                    )
                    release_request.set()
                    thread.join(2)

            self.assertFalse(thread.is_alive())
            self.assertEqual(result[0]["remote_log_bytes"], 77)
            self.assertEqual(result[0]["mirrored_outputs"], [{"name": "step.safetensors"}])

    def test_reconcile_runpod_does_not_attach_stale_realization_observation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            realization_ref = "realizations/r1.json"
            (run_dir / realization_ref).write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": realization_ref, "pod_id": "pod-1"}), encoding="utf-8")
            request_started = threading.Event()
            release_request = threading.Event()
            result: list[dict[str, Any]] = []

            def observe(*_args: object, **_kwargs: object) -> dict[str, object]:
                request_started.set()
                self.assertTrue(release_request.wait(2))
                return {"id": "pod-1", "desiredStatus": "TERMINATED"}

            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", side_effect=observe):
                    thread = threading.Thread(target=lambda: result.append(reconcile_runpod(run_dir, self._config())))
                    thread.start()
                    self.assertTrue(request_started.wait(2))
                    _mutate_run_status(
                        run_dir,
                        lambda status: status.update({"state": "running", "last_realization": "realizations/r2.json", "pod_id": "pod-2"}),
                    )
                    release_request.set()
                    thread.join(2)

            self.assertFalse(thread.is_alive())
            self.assertEqual(result[0]["last_realization"], "realizations/r2.json")
            self.assertEqual(result[0]["pod_id"], "pod-2")
            self.assertNotIn("last_observation", result[0])
            self.assertFalse(list((run_dir / "realizations").glob("r1.observed-*.json")))

    def test_cli_reconcile_runpod_syncs_remote_log_without_api_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
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
            (root / "workspace.yaml").write_text("runpod: {gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
            (root / "runs" / "example").mkdir(parents=True)
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.launch.stage_run", return_value=0), \
                     patch("kura.run_commands.launch.launch_run", return_value=0) as launch, \
                     patch("kura.run_commands.launch._runpod_run_over_ssh", return_value=0), \
                     patch("kura.run_commands.launch.download_with_retries", return_value=1), \
                     patch("kura.run_commands.launch.stop_runpod", return_value={}) as stop:
                    code = _run_remote_in_process(argparse.Namespace(run_id="example", upload_timeout=1, job_timeout=1, download_attempts=1, download_interval=1, max_lease="3h", yes=True))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            self.assertTrue(launch.call_args.kwargs["yes"])
            self.assertEqual(launch.call_args.kwargs["max_lease"], 3 * 3600)
            stop.assert_not_called()

    def test_run_remote_notifies_loudly_on_controller_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
            (root / "runs" / "example").mkdir(parents=True)
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.launch.stage_run", return_value=0), \
                     patch("kura.run_commands.launch.launch_run", return_value=0), \
                     patch("kura.run_commands.launch._runpod_run_over_ssh", side_effect=subprocess.TimeoutExpired(["runpod-remote-job", "example"], 1)), \
                     patch("kura.run_commands.launch._notify") as notify, \
                     patch("kura.run_commands.launch.stop_runpod", return_value={}) as stop:
                    code = _run_remote_in_process(argparse.Namespace(run_id="example", upload_timeout=1, job_timeout=1, download_attempts=1, download_interval=1, notify="ntfy", hold_for="30m", notify_repeat_interval="10m"))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            stop.assert_not_called()
            notify.assert_called_once()
            self.assertIn("controller failed", notify.call_args.kwargs["subject"])
            self.assertIn("may still be running and billing", notify.call_args.kwargs["body"])
            self.assertIn("deletes itself (with its outputs) after the unattended wait", notify.call_args.kwargs["body"])
            self.assertIn("kura run stop example", notify.call_args.kwargs["body"])

    def test_stop_run_without_realization_reports_clean_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            run_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            (root / "workspace.yaml").write_text("runpod: {gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
            (root / "runs" / "example").mkdir(parents=True)
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.launch.stage_run", return_value=0), \
                     patch("kura.run_commands.launch.launch_run", return_value=0), \
                     patch("kura.run_commands.launch._runpod_run_over_ssh", return_value=0), \
                     patch("kura.run_commands.launch.download_with_retries", return_value=0), \
                     patch("kura.run_commands.launch.stop_runpod", return_value={}) as stop:
                    code = _run_remote_in_process(argparse.Namespace(run_id="example", upload_timeout=1, job_timeout=1, download_attempts=1, download_interval=1, hold_for="0"))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            stop.assert_called_once()

    def test_run_remote_records_download_phases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/r1.json"}), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.launch.stage_run", return_value=0), \
                     patch("kura.run_commands.launch.launch_run", return_value=0), \
                     patch("kura.run_commands.launch._runpod_run_over_ssh", return_value=0), \
                     patch("kura.run_commands.launch.download_with_retries", return_value=0), \
                     patch("kura.run_commands.launch.stop_runpod", return_value={}):
                    code = _run_remote_in_process(argparse.Namespace(run_id="example", upload_timeout=1, job_timeout=1, download_attempts=1, download_interval=1, hold_for="0"))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            self.assertEqual([item["phase"] for item in launch_phases(run_dir, "r1")], ["download_started", "download_finished"])

    def test_run_download_reuses_verified_local_checkpoint_in_terminal_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "outputs").mkdir()
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                "backend: {name: sd-scripts}\nrecipe: {steps: 100}\n",
                encoding="utf-8",
            )
            (run_dir / "resolved" / "backend-command.lock.json").write_text("{}", encoding="utf-8")
            header = json.dumps(
                {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
                separators=(",", ":"),
            ).encode()
            checkpoint_bytes = len(header).to_bytes(8, "little") + header + b"\x00\x00\x00\x00"
            checkpoint = run_dir / "outputs" / "model-step00000100.safetensors"
            checkpoint.write_bytes(checkpoint_bytes)
            remote_exit_bytes = json.dumps(
                {"timestamp": "2026-01-01T00:00:00+00:00", "exit_code": 0}
            ).encode()
            manifest = [
                {
                    "path": "outputs/model-step00000100.safetensors",
                    "size": len(checkpoint_bytes),
                    "mtime_ns": 10,
                    "sha256": hashlib.sha256(checkpoint_bytes).hexdigest(),
                },
                {
                    "path": "realizations/remote-exit-20260101.json",
                    "size": len(remote_exit_bytes),
                    "mtime_ns": 20,
                    "sha256": hashlib.sha256(remote_exit_bytes).hexdigest(),
                },
            ]
            (run_dir / "status.json").write_text(
                json.dumps({
                    "state": "running",
                    "pod_id": "pod-1",
                    "mirrored_outputs": [{
                        "name": checkpoint.name,
                        "path": f"outputs/{checkpoint.name}",
                        "size": len(checkpoint_bytes),
                        "remote_path": f"/workspace/runs/example/outputs/{checkpoint.name}",
                        "remote_mtime_ns": 10,
                    }],
                }),
                encoding="utf-8",
            )
            transferred: list[str] = []

            def transfer_delta(*_: object, files: list[dict[str, object]], destination: Path, **__: object) -> None:
                transferred.extend(str(item["path"]) for item in files)
                target = destination / "example" / "realizations" / "remote-exit-20260101.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(remote_exit_bytes)

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.runpod_ssh.shutil.which", return_value="/usr/bin/runpodctl"), \
                     patch("kura.run_commands.runpod_ssh._runpod_ssh_details", return_value={"ip": "host", "port": 22, "key": "key"}), \
                     patch("kura.run_commands.runpod_ssh._runpod_remote_snapshot_manifest", side_effect=[manifest, manifest]), \
                     patch("kura.run_commands.runpod_ssh._transfer_runpod_snapshot_delta", side_effect=transfer_delta), \
                     patch("kura.run_commands.runpod_ssh.ensure_free_bytes"):
                    code = cmd_run_download(argparse.Namespace(run_id="example", force=True))
            finally:
                os.chdir(previous)

            self.assertEqual(code, 0)
            self.assertEqual(transferred, ["realizations/remote-exit-20260101.json"])
            snapshot_checkpoint = run_dir / "downloads" / "example" / "outputs" / checkpoint.name
            self.assertTrue(snapshot_checkpoint.samefile(checkpoint))
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "completed")
            self.assertEqual(
                status["terminal_download"],
                {
                    "files": 2,
                    "bytes": len(checkpoint_bytes) + len(remote_exit_bytes),
                    "reused_files": 1,
                    "reused_bytes": len(checkpoint_bytes),
                    "transferred_files": 1,
                    "transferred_bytes": len(remote_exit_bytes),
                },
            )

    def test_run_download_preserves_previous_snapshot_when_remote_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                "backend: {name: sd-scripts}\nrecipe: {steps: 100}\n",
                encoding="utf-8",
            )
            previous_snapshot = run_dir / "downloads" / "example"
            previous_snapshot.mkdir(parents=True)
            (previous_snapshot / "incomplete-marker.txt").write_text("keep", encoding="utf-8")
            (run_dir / "status.json").write_text(
                json.dumps({"state": "running", "pod_id": "pod-1"}), encoding="utf-8"
            )
            first_bytes = json.dumps(
                {"timestamp": "2026-01-01T00:00:00+00:00", "exit_code": 0}
            ).encode()
            changed_bytes = json.dumps(
                {"timestamp": "2026-01-01T00:00:01+00:00", "exit_code": 0}
            ).encode()
            first = [{
                "path": "realizations/remote-exit-20260101.json",
                "size": len(first_bytes),
                "mtime_ns": 10,
                "sha256": hashlib.sha256(first_bytes).hexdigest(),
            }]
            changed = [{
                "path": "realizations/remote-exit-20260101.json",
                "size": len(changed_bytes),
                "mtime_ns": 20,
                "sha256": hashlib.sha256(changed_bytes).hexdigest(),
            }]

            def transfer_delta(*_: object, destination: Path, **__: object) -> None:
                target = destination / "example" / "realizations" / "remote-exit-20260101.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(first_bytes)

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.runpod_ssh.shutil.which", return_value="/usr/bin/runpodctl"), \
                     patch("kura.run_commands.runpod_ssh._runpod_ssh_details", return_value={"ip": "host", "port": 22, "key": "key"}), \
                     patch("kura.run_commands.runpod_ssh._runpod_remote_snapshot_manifest", side_effect=[first, changed]), \
                     patch("kura.run_commands.runpod_ssh._transfer_runpod_snapshot_delta", side_effect=transfer_delta), \
                     patch("kura.run_commands.runpod_ssh.ensure_free_bytes"):
                    code = cmd_run_download(argparse.Namespace(run_id="example", force=True))
            finally:
                os.chdir(previous)

            self.assertEqual(code, 1)
            self.assertEqual((previous_snapshot / "incomplete-marker.txt").read_text(encoding="utf-8"), "keep")
            self.assertEqual(
                json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"],
                "running",
            )

    def test_run_download_rejects_manifest_mismatch_and_preserves_previous_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                "backend: {name: sd-scripts}\nrecipe: {steps: 100}\n",
                encoding="utf-8",
            )
            previous_snapshot = run_dir / "downloads" / "example"
            previous_snapshot.mkdir(parents=True)
            (previous_snapshot / "incomplete-marker.txt").write_text("keep", encoding="utf-8")
            (run_dir / "status.json").write_text(
                json.dumps({"state": "running", "pod_id": "pod-1"}), encoding="utf-8"
            )
            expected = b"expected"
            corrupted = b"corrupt!"
            manifest = [{
                "path": "logs/stdout.log",
                "size": len(expected),
                "mtime_ns": 10,
                "sha256": hashlib.sha256(expected).hexdigest(),
            }]

            def transfer_delta(*_: object, destination: Path, **__: object) -> None:
                target = destination / "example" / "logs" / "stdout.log"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(corrupted)

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.runpod_ssh.shutil.which", return_value="/usr/bin/runpodctl"), \
                     patch("kura.run_commands.runpod_ssh._runpod_ssh_details", return_value={"ip": "host", "port": 22, "key": "key"}), \
                     patch("kura.run_commands.runpod_ssh._runpod_remote_snapshot_manifest", side_effect=[manifest, manifest]), \
                     patch("kura.run_commands.runpod_ssh._transfer_runpod_snapshot_delta", side_effect=transfer_delta), \
                     patch("kura.run_commands.runpod_ssh.ensure_free_bytes"):
                    code = cmd_run_download(argparse.Namespace(run_id="example", force=True))
            finally:
                os.chdir(previous)

            self.assertEqual(code, 1)
            self.assertEqual((previous_snapshot / "incomplete-marker.txt").read_text(encoding="utf-8"), "keep")
            self.assertEqual(
                json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"],
                "running",
            )
            self.assertEqual(
                sorted(path.name for path in (run_dir / "downloads").iterdir()),
                ["example"],
            )

    def test_run_download_restores_previous_snapshot_when_completion_record_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            (run_dir / "resolved").mkdir(parents=True)
            (run_dir / "resolved" / "manifest.lock.yaml").write_text(
                "backend: {name: sd-scripts}\nrecipe: {steps: 100}\n",
                encoding="utf-8",
            )
            previous_snapshot = run_dir / "downloads" / "example"
            previous_snapshot.mkdir(parents=True)
            (previous_snapshot / "incomplete-marker.txt").write_text("keep", encoding="utf-8")
            (run_dir / "status.json").write_text(
                json.dumps({"state": "running", "pod_id": "pod-1", "recovery_artifacts": ["old"]}),
                encoding="utf-8",
            )
            log_bytes = b"finished without an exit record"
            manifest = [{
                "path": "logs/stdout.log",
                "size": len(log_bytes),
                "mtime_ns": 10,
                "sha256": hashlib.sha256(log_bytes).hexdigest(),
            }]

            def transfer_delta(*_: object, destination: Path, **__: object) -> None:
                target = destination / "example" / "logs" / "stdout.log"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(log_bytes)

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.runpod_ssh.shutil.which", return_value="/usr/bin/runpodctl"), \
                     patch("kura.run_commands.runpod_ssh._runpod_ssh_details", return_value={"ip": "host", "port": 22, "key": "key"}), \
                     patch("kura.run_commands.runpod_ssh._runpod_remote_snapshot_manifest", side_effect=[manifest, manifest]), \
                     patch("kura.run_commands.runpod_ssh._transfer_runpod_snapshot_delta", side_effect=transfer_delta), \
                     patch("kura.run_commands.runpod_ssh.ensure_free_bytes"):
                    code = cmd_run_download(argparse.Namespace(run_id="example", force=True))
            finally:
                os.chdir(previous)

            self.assertEqual(code, 1)
            self.assertEqual((previous_snapshot / "incomplete-marker.txt").read_text(encoding="utf-8"), "keep")
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "running")
            self.assertEqual(status["recovery_artifacts"], ["old"])

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

    @posix_only(POSIX_PATHS)
    def test_run_download_materializes_outputs_at_run_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            output_dir = run_dir / "downloads" / "example" / "outputs"
            realization_dir = run_dir / "downloads" / "example" / "realizations"
            output_dir.mkdir(parents=True)
            realization_dir.mkdir(parents=True)
            (run_dir / "outputs").mkdir()
            (run_dir / "outputs" / "intermediate-step00000250.safetensors").write_text("intermediate", encoding="utf-8")
            (output_dir / "artifact.safetensors").write_text("artifact", encoding="utf-8")
            state_dir = output_dir / "example-step00000010-state"
            state_dir.mkdir()
            (state_dir / "optimizer.bin").write_text("optimizer", encoding="utf-8")
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
            self.assertEqual((run_dir / "outputs" / "intermediate-step00000250.safetensors").read_text(encoding="utf-8"), "intermediate")
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["outputs"], ["outputs/artifact.safetensors"])
            self.assertEqual(status["downloaded_run"], "downloads/example")
            self.assertFalse((run_dir / "outputs" / state_dir.name).exists())

    @posix_only(POSIX_PATHS)
    def test_run_download_records_recovery_without_publishing_it_as_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            downloaded = run_dir / "downloads" / "example"
            (downloaded / "outputs").mkdir(parents=True)
            (downloaded / "recovery" / "sd-scripts" / "anima-native").mkdir(parents=True)
            (downloaded / "realizations").mkdir()
            (downloaded / "outputs" / "converted.safetensors").write_text("converted", encoding="utf-8")
            native = downloaded / "recovery" / "sd-scripts" / "anima-native" / "native.safetensors"
            native.write_text("native", encoding="utf-8")
            (downloaded / "realizations" / "remote-exit-20260101.json").write_text(
                json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "exit_code": 0}),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1"}), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                code = cmd_run_download(argparse.Namespace(run_id="example", force=False))
                retry_code = cmd_run_download(argparse.Namespace(run_id="example", force=False))
            finally:
                os.chdir(previous)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(retry_code, 0)
            self.assertEqual(status["outputs"], ["outputs/converted.safetensors"])
            self.assertEqual(status["recovery_artifacts"], ["downloads/example/recovery/sd-scripts/anima-native/native.safetensors"])
            self.assertFalse((run_dir / "outputs" / "native.safetensors").exists())
            events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(sum(item.get("event") == "run_recovery_artifacts_downloaded" for item in events), 1)

    @posix_only(POSIX_PATHS)
    def test_run_download_normalizes_ai_toolkit_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            output_dir = run_dir / "downloads" / "example" / "outputs" / "example"
            realization_dir = run_dir / "downloads" / "example" / "realizations"
            resolved_dir = run_dir / "resolved"
            output_dir.mkdir(parents=True)
            realization_dir.mkdir(parents=True)
            resolved_dir.mkdir()
            (resolved_dir / "manifest.lock.yaml").write_text(
                "id: example\ntype: train\nbackend: {name: ai-toolkit}\nrecipe: {steps: 1, seed: 1}\n",
                encoding="utf-8",
            )
            (resolved_dir / "backend-command.lock.json").write_text("{}", encoding="utf-8")
            artifacts = {
                "example.safetensors": "weight",
                "config.yaml": "config",
                "optimizer.pt": "optimizer",
            }
            for name, content in artifacts.items():
                (output_dir / name).write_text(content, encoding="utf-8")
            legacy_dir = run_dir / "outputs" / "example"
            legacy_dir.mkdir(parents=True)
            for name, content in artifacts.items():
                (legacy_dir / name).write_text(content, encoding="utf-8")
            (realization_dir / "remote-exit-20260101.json").write_text(
                json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "exit_code": 0}),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps({"state": "running", "pod_id": "pod-1"}),
                encoding="utf-8",
            )

            previous = Path.cwd()
            os.chdir(root)
            try:
                code = cmd_run_download(argparse.Namespace(run_id="example", force=False))
            finally:
                os.chdir(previous)

            self.assertEqual(code, 0)
            for name, content in artifacts.items():
                self.assertEqual((run_dir / "outputs" / name).read_text(encoding="utf-8"), content)
                self.assertEqual((output_dir / name).read_text(encoding="utf-8"), content)
            self.assertFalse(legacy_dir.exists())
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(
                status["outputs"],
                ["outputs/config.yaml", "outputs/example.safetensors", "outputs/optimizer.pt"],
            )

    def test_run_download_rejects_ai_toolkit_output_collision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            output_root = run_dir / "downloads" / "example" / "outputs"
            realization_dir = run_dir / "downloads" / "example" / "realizations"
            resolved_dir = run_dir / "resolved"
            (output_root / "example").mkdir(parents=True)
            realization_dir.mkdir(parents=True)
            resolved_dir.mkdir()
            (resolved_dir / "manifest.lock.yaml").write_text(
                "id: example\ntype: train\nbackend: {name: ai-toolkit}\nrecipe: {steps: 1, seed: 1}\n",
                encoding="utf-8",
            )
            (output_root / "artifact.safetensors").write_text("outer", encoding="utf-8")
            (output_root / "example" / "artifact.safetensors").write_text("inner", encoding="utf-8")
            (realization_dir / "remote-exit-20260101.json").write_text(
                json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "exit_code": 0}),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(
                json.dumps({"state": "running", "pod_id": "pod-1"}),
                encoding="utf-8",
            )

            previous = Path.cwd()
            os.chdir(root)
            try:
                code = cmd_run_download(argparse.Namespace(run_id="example", force=False))
            finally:
                os.chdir(previous)

            # The collision is in the snapshot itself: collected, nothing published, and a person decides.
            self.assertEqual(code, 3)
            self.assertFalse((run_dir / "outputs" / "artifact.safetensors").exists())
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "recovery_required")
            self.assertIn("collide", status["publication_error"])

    def test_run_download_preserves_modified_ai_toolkit_legacy_directory(self) -> None:
        for scenario in ("extra", "modified"):
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
                run_dir = root / "runs" / "example"
                output_dir = run_dir / "downloads" / "example" / "outputs" / "example"
                realization_dir = run_dir / "downloads" / "example" / "realizations"
                resolved_dir = run_dir / "resolved"
                legacy_dir = run_dir / "outputs" / "example"
                output_dir.mkdir(parents=True)
                realization_dir.mkdir(parents=True)
                resolved_dir.mkdir()
                legacy_dir.mkdir(parents=True)
                (resolved_dir / "manifest.lock.yaml").write_text(
                    "id: example\ntype: train\nbackend: {name: ai-toolkit}\nrecipe: {steps: 1, seed: 1}\n",
                    encoding="utf-8",
                )
                (resolved_dir / "backend-command.lock.json").write_text("{}", encoding="utf-8")
                (output_dir / "example.safetensors").write_text("native", encoding="utf-8")
                legacy_content = "changed" if scenario == "modified" else "native"
                (legacy_dir / "example.safetensors").write_text(legacy_content, encoding="utf-8")
                if scenario == "extra":
                    (legacy_dir / "keep.txt").write_text("user file", encoding="utf-8")
                (realization_dir / "remote-exit-20260101.json").write_text(
                    json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "exit_code": 0}),
                    encoding="utf-8",
                )
                (run_dir / "status.json").write_text(
                    json.dumps({"state": "running", "pod_id": "pod-1"}),
                    encoding="utf-8",
                )

                previous = Path.cwd()
                os.chdir(root)
                try:
                    code = cmd_run_download(argparse.Namespace(run_id="example", force=False))
                finally:
                    os.chdir(previous)

                self.assertEqual(code, 0)
                self.assertEqual((run_dir / "outputs" / "example.safetensors").read_text(encoding="utf-8"), "native")
                self.assertEqual((legacy_dir / "example.safetensors").read_text(encoding="utf-8"), legacy_content)
                if scenario == "extra":
                    self.assertEqual((legacy_dir / "keep.txt").read_text(encoding="utf-8"), "user file")

    def test_run_download_cleans_partial_output_when_publication_fails(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            output_dir = run_dir / "downloads" / "example" / "outputs"
            realization_dir = run_dir / "downloads" / "example" / "realizations"
            output_dir.mkdir(parents=True)
            realization_dir.mkdir(parents=True)
            (output_dir / "artifact.safetensors").write_text("artifact", encoding="utf-8")
            (realization_dir / "remote-exit-20260101.json").write_text(
                json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "exit_code": 0}),
                encoding="utf-8",
            )
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1"}), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.runpod_ssh.os.replace", side_effect=OSError("publication failed")):
                    code = cmd_run_download(argparse.Namespace(run_id="example", force=False))
            finally:
                os.chdir(previous)

            self.assertEqual(code, 1)
            self.assertEqual(list((run_dir / "outputs").glob(".artifact.safetensors.partial-*")), [])

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
        def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if command == ["runpodctl", "version"]:
                return subprocess.CompletedProcess(command, 0, "runpodctl test", "")
            if command == ["runpodctl", "pod", "list"]:
                return subprocess.CompletedProcess(command, 0, "[]", "")
            if command == ["runpodctl", "network-volume", "list"]:
                return subprocess.CompletedProcess(command, 0, '[{"id":"volume-1"}]', "")
            raise AssertionError(command)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {}\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                     patch("kura.doctor.shutil.which", return_value="/usr/bin/runpodctl"), \
                     patch("kura.doctor.subprocess.run", side_effect=fake_run):
                    code = cmd_doctor_runpod(argparse.Namespace())
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)

    def test_doctor_runpod_fails_when_network_volume_check_is_unknown(self) -> None:
        def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            if command == ["runpodctl", "version"]:
                return subprocess.CompletedProcess(command, 0, "runpodctl test", "")
            if command == ["runpodctl", "pod", "list"]:
                return subprocess.CompletedProcess(command, 0, "[]", "")
            if command == ["runpodctl", "network-volume", "list"]:
                return subprocess.CompletedProcess(command, 1, "", "network volume endpoint unavailable")
            raise AssertionError(command)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {}\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                     patch("kura.doctor.shutil.which", return_value="/usr/bin/runpodctl"), \
                     patch("kura.doctor.subprocess.run", side_effect=fake_run), \
                     patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_runpod(argparse.Namespace())
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertIsNone(payload["checks"]["network_volumes_empty"])
            self.assertIn("network_volumes_error", payload["diagnostics"])

    def test_doctor_runpod_labels_process_permission_denial_without_blame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {}\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
                    if command == ["runpodctl", "version"]:
                        return subprocess.CompletedProcess(command, 0, "runpodctl test", "")
                    raise OSError("Operation not permitted")

                with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                     patch("kura.doctor.shutil.which", return_value="/usr/bin/runpodctl"), \
                     patch("kura.doctor.subprocess.run", side_effect=fake_run), \
                     patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_runpod(argparse.Namespace())
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertIn("This process could not reach", payload["diagnosis"])
            self.assertIn("may work outside", payload["diagnosis"])
            self.assertNotIn("Codex", payload["diagnosis"])

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
                    "ModelPatchLoader": {},
                    "AnimaLLLiteApply": {},
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

    def test_doctor_comfyui_reports_core_anima_model_patch_support_without_gating_other_workflows(self) -> None:
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
                with patch("kura.doctor.urllib.request.urlopen", return_value=FakeResponse()), patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_comfyui(argparse.Namespace())
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 0)
            self.assertFalse(payload["checks"]["core_model_patch_loader"])
            self.assertFalse(payload["checks"]["core_anima_lllite_apply"])

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

    def test_doctor_comfyui_probe_stage_reports_unconfigured_lora_dir_without_permission_guess(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({"LoraLoader": {"input": {"required": {"lora_name": [[], {}]}}}}).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("comfyui: {endpoint: http://127.0.0.1:8190}\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.urllib.request.urlopen", return_value=FakeResponse()), patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_comfyui(argparse.Namespace(endpoint=None, probe_stage=True))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertIn("not configured", payload["diagnosis"])
            self.assertNotIn("could not write", payload["diagnosis"])

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
            self.assertIn("not reachable", payload["diagnosis"])
            self.assertIn("Do not start a Docker ComfyUI", payload["diagnosis"])
            self.assertNotIn("not visible", payload["diagnosis"])

    def test_doctor_comfyui_reports_alternate_default_endpoint_without_retargeting(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({"KSampler": {}}).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("comfyui: {endpoint: http://127.0.0.1:8191}\n", encoding="utf-8")

            def fake_urlopen(url: str, timeout: int = 5) -> FakeResponse:
                if url == "http://127.0.0.1:8188/object_info":
                    return FakeResponse()
                raise OSError("connection refused")

            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.urllib.request.urlopen", side_effect=fake_urlopen), patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_comfyui(argparse.Namespace(endpoint=None, probe_stage=False, workflow=None))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertEqual(payload["diagnostics"]["endpoint"], "http://127.0.0.1:8191")
            self.assertEqual(payload["diagnostics"]["candidate_endpoint"], "http://127.0.0.1:8188")
            self.assertIn("do not retarget automatically", payload["warnings"][0])

    def test_doctor_comfyui_workflow_reports_missing_local_model(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({
                    "UNETLoader": {"input": {"required": {"unet_name": [["present.safetensors"], {}]}}},
                }).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workflow = root / "workflow.json"
            workflow.write_text(json.dumps({"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "missing.safetensors"}}}), encoding="utf-8")
            (root / "workspace.yaml").write_text("comfyui: {endpoint: http://127.0.0.1:8188}\n", encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.urllib.request.urlopen", return_value=FakeResponse()), patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_comfyui(argparse.Namespace(endpoint=None, probe_stage=False, workflow="workflow.json"))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertFalse(payload["checks"]["workflow_models_visible"])
            self.assertEqual(payload["diagnostics"]["workflow_missing_models"][0]["name"], "missing.safetensors")
            self.assertIn("Local render never downloads models", payload["diagnosis"])

    def test_doctor_comfyui_distinguishes_process_write_denial_from_visibility(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *_: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps({"LoraLoader": {"input": {"required": {"lora_name": [[], {}]}}}}).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage_dir = root / "models" / "loras" / "Kura_tmp"
            stage_dir.mkdir(parents=True)
            (root / "workspace.yaml").write_text(
                f"comfyui:\n  endpoint: http://127.0.0.1:8190\n  lora_dir: {stage_dir.parent}\n  lora_stage_subdir: Kura_tmp\n",
                encoding="utf-8",
            )
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.doctor.urllib.request.urlopen", return_value=FakeResponse()), \
                     patch("kura.doctor._probe_comfyui_lora_stage", side_effect=OSError("Read-only file system")), \
                     patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
                    code = cmd_doctor_comfyui(argparse.Namespace(endpoint=None, probe_stage=True))
            finally:
                os.chdir(previous)
            payload = json.loads(stdout.getvalue())
            self.assertEqual(code, 1)
            self.assertIn("this process could not write", payload["diagnosis"])
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
                    stage_runpod(workspace=root, run_dir=run_dir, dataset_id="tiny", config={"runpod": self._object_config()})

    def test_stage_runpod_object_staging_fails_before_source_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "missing"
            with self.assertRaisesRegex(ValueError, "object_staging is experimental and disabled"):
                stage_runpod(workspace=root, run_dir=run_dir, dataset_id="missing-dataset", config={"runpod": self._object_config()})

    def test_stop_runpod_terminates_disposable_pod(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", return_value={}) as request:
                    status = stop_runpod(run_dir, self._config())
            self.assertEqual(status["state"], "interrupted")
            self.assertEqual(request.call_args.args[:3], ("DELETE", "/pods/pod-1", "api-secret"))
            events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(events[-1]["event"], "runpod_pod_stopped")

    def test_stop_runpod_records_pod_stopped_phase(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "status.json").write_text(json.dumps({"state": "completed", "pod_id": "pod-1", "last_realization": "realizations/r1.json"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", return_value={}):
                    stop_runpod(run_dir, self._config())
            self.assertEqual([item["phase"] for item in launch_phases(run_dir, "r1")], ["pod_stop_requested", "pod_stopped"])
            stops = [json.loads(path.read_text(encoding="utf-8")) for path in (run_dir / "realizations").glob("r1.stop-*.json")]
            self.assertEqual(len(stops), 1)
            self.assertEqual((stops[0]["kind"], stops[0]["outcome"], stops[0]["targets"]), ("stop", "stopped", [{"pod_id": "pod-1", "result": "deleted"}]))

    def test_stop_runpod_records_a_failed_delete_before_raising(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1", "last_realization": "realizations/r1.json"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", side_effect=ValueError("RunPod API failed (500)")), self.assertRaises(ValueError):
                    stop_runpod(run_dir, self._config())
            [stop] = [json.loads(path.read_text(encoding="utf-8")) for path in (run_dir / "realizations").glob("r1.stop-*.json")]
            self.assertEqual((stop["outcome"], stop["stopped_at"], stop["targets"]), ("failed", None, [{"pod_id": "pod-1", "result": "failed"}]))

    def test_stop_runpod_explains_how_to_cancel_capacity_wait(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "status.json").write_text(
                json.dumps({"state": "queued", "capacity_wait": {"attempts": 2}}),
                encoding="utf-8",
            )
            with patch("kura.executors.runpod._runpod_request") as request:
                with self.assertRaisesRegex(ValueError, "no Pod exists.*Ctrl\\+C.*doctor runpod.*run execute example"):
                    stop_runpod(run_dir, self._config())
            request.assert_not_called()

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

    def test_hold_reconcile_then_stop_preserves_completed_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            realization_ref = "realizations/r1.json"
            (run_dir / realization_ref).write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}}), encoding="utf-8")
            (run_dir / "status.json").write_text(
                json.dumps({"state": "completed", "exit_code": 0, "ended": "confirmed-end", "last_realization": realization_ref, "pod_id": "pod-1"}),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", side_effect=[{"id": "pod-1", "desiredStatus": "RUNNING"}, {}]):
                    reconciled = reconcile_runpod(run_dir, self._config())
                    stopped = stop_runpod(run_dir, self._config())

            self.assertEqual(reconciled["state"], "completed")
            self.assertEqual(stopped["state"], "completed")
            self.assertEqual(stopped["exit_code"], 0)
            self.assertEqual(stopped["ended"], "confirmed-end")

    def test_stop_runpod_merges_stop_into_latest_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1"}), encoding="utf-8")
            request_started = threading.Event()
            release_request = threading.Event()
            result: list[dict[str, Any]] = []

            def delete(*_args: object) -> dict[str, object]:
                request_started.set()
                self.assertTrue(release_request.wait(2))
                return {}

            def stop() -> None:
                result.append(stop_runpod(run_dir, self._config()))

            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", side_effect=delete):
                    thread = threading.Thread(target=stop)
                    thread.start()
                    self.assertTrue(request_started.wait(2))
                    _mutate_run_status(
                        run_dir,
                        lambda status: status.update({"remote_log_bytes": 77, "mirrored_outputs": [{"name": "step.safetensors"}]}),
                    )
                    release_request.set()
                    thread.join(2)

            self.assertFalse(thread.is_alive())
            self.assertEqual(result[0]["state"], "interrupted")
            self.assertEqual(result[0]["remote_log_bytes"], 77)
            self.assertEqual(result[0]["mirrored_outputs"], [{"name": "step.safetensors"}])
            self.assertIn("pod_stopped_at", result[0])

    def test_stop_runpod_does_not_mutate_replacement_pod_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1"}), encoding="utf-8")
            request_started = threading.Event()
            release_request = threading.Event()
            result: list[dict[str, Any]] = []

            def delete(*_args: object) -> dict[str, object]:
                request_started.set()
                self.assertTrue(release_request.wait(2))
                return {}

            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors.runpod._runpod_request", side_effect=delete):
                    thread = threading.Thread(target=lambda: result.append(stop_runpod(run_dir, self._config())))
                    thread.start()
                    self.assertTrue(request_started.wait(2))
                    _mutate_run_status(run_dir, lambda status: status.update({"state": "running", "pod_id": "pod-2"}))
                    release_request.set()
                    thread.join(2)

            self.assertFalse(thread.is_alive())
            self.assertEqual(result[0]["state"], "running")
            self.assertEqual(result[0]["pod_id"], "pod-2")
            self.assertNotIn("pod_stopped_at", result[0])
