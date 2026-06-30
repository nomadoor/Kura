"""Small regression tests for workspace initialization."""

from __future__ import annotations

import argparse
import asyncio
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
from unittest.mock import patch

import yaml

from kura.backends import MUSUBI_ADAPTER_SCRIPTS, _safetensors_validator_code, command_ai_toolkit, command_musubi_tuner, compile_ai_toolkit, compile_musubi_tuner
from kura.cli import _docker_cleanup_image, _load_env_local, _notification_channels, _notify, _parse_duration_seconds, _runpod_run_over_ssh, _runpod_secret_env_payload, _select_remote_outputs, _sync_runpod_remote_stdout, _workspace, cmd_cleanup, cmd_doctor_comfyui, cmd_doctor_disk, cmd_doctor_docker, cmd_doctor_musubi, cmd_doctor_runpod, cmd_doctor_workspace, cmd_fix_permissions, cmd_image_build, cmd_init, cmd_monitor, cmd_run_download, cmd_run_launch, cmd_run_plan, cmd_run_prune, cmd_run_reconcile, cmd_run_remote, cmd_run_status
from kura.executors import docker_command, docker_preflight, launch_runpod, reconcile_docker, reconcile_runpod, stage_runpod, stop_runpod
from kura.init_templates import RUNPOD_OBJECT_JOB_TEMPLATE
from kura.render import _cleanup_lora_stage, _materialize_lora_stage, _safe_stage_name, compile_render, launch_render
from kura.run_commands import _checkpoint_safety_preflight, _ensure_free_bytes, _local_launch_disk_preflight, launch_run
from kura.storage import StorageStatus
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
                self.assertEqual(workspace["docker"]["mounts"][0]["target"], "/root/.cache/huggingface")
                self.assertEqual(workspace["runpod"]["gpu_type_ids"], ["NVIDIA RTX A5000", "NVIDIA A40"])
                self.assertEqual(workspace["runpod"]["gpu_type_priority"], "custom")
                self.assertEqual(workspace["comfyui"]["lora_dir"], "")
                self.assertEqual(workspace["comfyui"]["lora_stage_cleanup"], "remove_after_render")
            finally:
                os.chdir(previous)

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
            self.assertEqual(build[-1], str(root))

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
            self.assertIn("workspace filesystem has less than 100GiB free", payload["warnings"])
            self.assertIn("Docker build cache exceeds 30GiB", payload["warnings"])
            self.assertIn("cache/runs contain root-owned files; cleanup may require permission repair", payload["warnings"])

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
                    code = cmd_monitor(argparse.Namespace(interval=1.5, stale_after=12.0, limit=7))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            monitor.assert_called_once_with(root, interval=1.5, stale_after=12.0, limit=7)


class TuiPathDisplayTests(unittest.TestCase):
    def test_compact_path_keeps_tail_at_narrow_widths(self) -> None:
        path = Path("/home/nomax/working-linux/Development/Kura/runs/example/outputs")
        self.assertEqual(_compact_path(path, max_len=1), "…")
        self.assertEqual(len(_compact_path(path, max_len=12)), 12)
        self.assertTrue(_compact_path(path, max_len=12).endswith("outputs"))

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


class RunPlanTests(unittest.TestCase):
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
                with patch("sys.stdout", new_callable=__import__("io").StringIO) as stdout:
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
            self.assertIn("Disk warnings", output)
            self.assertIn("checkpoint cadence may create about 15 checkpoints", output)

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
            self.assertIn("local Docker launch requires a disk preflight", payload["disk_warnings"][0])

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
                with patch("kura.run_commands.launch_render", return_value=0) as launch, patch("kura.run_commands._notify") as notify:
                    code = cmd_run_launch(argparse.Namespace(run_id="render-1", executor="local", dry_run=False, notify="ntfy"))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            launch.assert_called_once()
            notify.assert_called_once()
            self.assertIn("completed", notify.call_args.kwargs["subject"])

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
            images = (run_dir / "samples" / "images.jsonl").read_text(encoding="utf-8")
            self.assertIn('"comfyui_lora_name":', images)

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
        self.assertEqual(command, {"cwd": "/opt/ai-toolkit", "argv": ["python", "run.py", "/workspace/runs/ai-toolkit-example/resolved/ai-toolkit.yaml"], "env": {}})

    def test_compile_rejects_non_mapping_native_config_override(self) -> None:
        run = self._run()
        run["backend_overrides"] = {"ai-toolkit": {"config": ["not", "a", "mapping"]}}
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "backend_overrides.ai-toolkit.config"):
                compile_ai_toolkit(run, Path(directory) / "ai-toolkit")


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
        self.assertEqual(expected["vae"], "flux2_vae")
        self.assertEqual(expected["text_encoder"], "qwen3_4b_text_encoder")
        self.assertEqual(bundle["output"]["lora_format"], "comfyui")

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
        self.assertIn('cache_dir = os.environ.get("HF_HOME") or "/root/.cache/huggingface"', script)
        self.assertNotIn("local_dir", script)
        self.assertNotIn("/workspace/cache/hf-models/musubi", script)
        self.assertIn("KURA_HF_DOWNLOAD_NO_PROGRESS_SEC", script)
        self.assertIn("repo_cache_dirs(cache_dir, item)", script)
        self.assertNotIn("remove_incomplete_files(cache_dir)", script)
        self.assertIn("removed {removed} incomplete", script)
        self.assertIn("black-forest-labs/FLUX.2-klein-base-4B", script)
        self.assertIn("/workspace/cache/models/musubi/black-forest-labs--FLUX.2-klein-base-4B/dit/flux2-klein-base-4b.safetensors", script)
        self.assertIn("--dit /workspace/cache/models/musubi/black-forest-labs--FLUX.2-klein-base-4B/dit/flux2-klein-base-4b.safetensors", script)
        self.assertIn("flux2_vae", script)
        self.assertIn("qwen3_4b_text_encoder", script)
        self.assertIn("lora_unet_*", script)
        self.assertLess(script.index("hf_hub_download"), script.index("src/musubi_tuner/flux_2_cache_latents.py"))
        self.assertLess(script.index("expected_format"), script.index("src/musubi_tuner/flux_2_cache_latents.py"))
        self.assertLess(script.index("src/musubi_tuner/flux_2_train_network.py"), script.rindex("lora_unet_*"))

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
        self.assertEqual(expected["vae"], "flux2_vae")
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
        self.assertEqual(runtime_env["HF_HOME"], "/root/.cache/huggingface")
        self.assertIn("HF_HOME=/root/.cache/huggingface", command)
        self.assertIn('exec "$@" >> "$KURA_LOG_PATH" 2>&1', command)

    def test_docker_mount_sources_are_resolved_from_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            run_dir.mkdir(parents=True)
            mounts = [{"source": "./cache/huggingface", "target": "/root/.cache/huggingface", "mode": "rw"}]
            command, _, _ = docker_command(root, run_dir, {"cwd": "/opt/tool", "argv": ["python", "train.py"], "env": {}}, "example:image", mounts, True, "r1")
        self.assertIn(f"{root}/cache/huggingface:/root/.cache/huggingface", command)

    def test_docker_preflight_creates_writable_mount_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            mounts = [{"source": "./cache/huggingface", "target": "/root/.cache/huggingface", "mode": "rw"}]
            with patch("kura.executors.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
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
                patch("kura.executors.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")),
                patch("kura.executors.shutil.disk_usage", return_value=Usage()),
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
                patch("kura.executors.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")),
                patch("kura.executors.shutil.disk_usage", return_value=Usage()),
            ):
                payload = docker_preflight(root, [], min_free_gb=10)
        self.assertEqual(payload["disk"]["workspace"]["free_bytes"], 20 * 1024**3)

    def test_local_launch_disk_preflight_uses_configured_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("kura.run_commands.probe_storages", side_effect=self._storage_probe(60)), patch("kura.run_commands.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
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
                patch("kura.run_commands.probe_storages", side_effect=self._storage_probe(60)),
                patch("kura.run_commands.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")),
                patch("kura.run_commands._hf_file_size_bytes", return_value=20 * 1024**3),
            ):
                with self.assertRaisesRegex(ValueError, "requires at least 70 GiB"):
                    _local_launch_disk_preflight(root, run, {"min_free_gb": 50}, mounts)
                payload = _local_launch_disk_preflight(root, run, {"min_free_gb": 40}, mounts)
        self.assertEqual(payload["estimates"]["musubi_downloads"]["bytes"], 20 * 1024**3)
        self.assertEqual(payload["paths"]["hf_cache"]["estimated_write_bytes"], 20 * 1024**3)

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
                patch("kura.run_commands.probe_storages", side_effect=self._storage_probe(100)),
                patch("kura.run_commands.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")),
            ):
                with self.assertRaisesRegex(ValueError, "requires at least 110 GiB"):
                    _local_launch_disk_preflight(root, run, {"min_free_gb": 50}, [])
        self.assertEqual(_checkpoint_safety_preflight(run), None)

    def test_local_launch_disk_preflight_honors_run_disk_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("kura.run_commands.probe_storages", side_effect=self._storage_probe(140)), patch("kura.run_commands.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
                with self.assertRaisesRegex(ValueError, "requires at least 150 GiB"):
                    _local_launch_disk_preflight(root, {"safety": {"max_run_disk_gb": 150}}, {"min_free_gb": 50}, [])

    def test_local_launch_disk_preflight_rejects_unknown_wsl_backing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("kura.run_commands.probe_storages", side_effect=self._storage_probe(900, confidence="unknown", backing_kind="wsl2_vhdx")), patch("kura.run_commands.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "")):
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
            with patch("kura.run_commands.shutil.disk_usage", return_value=Usage()):
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
            with patch("kura.executors.subprocess.run", return_value=result):
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
            with patch("kura.executors.subprocess.run", return_value=result):
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
            with patch("kura.executors.subprocess.run", return_value=result):
                status = reconcile_docker(run_dir)
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["last_step"], 5)
            self.assertEqual(status["total_steps"], 5)
            self.assertEqual(status["outputs"], ["outputs/example.safetensors"])

    def test_reconcile_missing_container_is_interrupted_not_failed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            result = __import__("subprocess").CompletedProcess([], 1, "", "Error: No such container")
            with patch("kura.executors.subprocess.run", return_value=result):
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
                with patch("kura.run_commands.launch_docker", side_effect=fake_launch), patch("kura.run_commands.subprocess.run", return_value=subprocess.CompletedProcess([], 0, "0\n", "")) as wait, patch("kura.run_commands.reconcile_docker", return_value={"state": "completed", "exit_code": 0}) as reconcile, patch("sys.stdout", new_callable=__import__("io").StringIO):
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
                with patch("kura.run_commands.launch_docker") as launch, patch("kura.run_commands.subprocess.run", return_value=subprocess.CompletedProcess([], 0, '{"Type":"Build Cache","Size":"0B"}\n', "")):
                    self.assertEqual(launch_run("example", executor="docker", dry_run=False, image="override-image:dev"), 0)
            finally:
                os.chdir(previous)
            self.assertEqual(launch.call_args.kwargs["image"], "override-image:dev")
            self.assertEqual(launch.call_args.kwargs["min_free_gb"], 10)


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
                with patch("kura.executors._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request:
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
                with patch("kura.executors._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request:
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
                with patch("kura.executors._runpod_request", return_value={"id": "pod-1", "desiredStatus": "RUNNING"}) as request:
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
                with patch("kura.executors._runpod_request", side_effect=[ValueError("no community capacity"), {"id": "pod-1", "desiredStatus": "RUNNING"}]) as request:
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
                with patch("kura.executors._runpod_request", side_effect=[ValueError("no A5000 capacity"), {"id": "pod-1", "desiredStatus": "RUNNING"}]) as request:
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
                with patch("kura.run_commands.launch_runpod") as launch:
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
                with patch("kura.executors._runpod_request", side_effect=ValueError("RunPod API POST /pods failed (500): echoed api-secret hf-secret")):
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
                with patch("kura.run_commands._runpod_ssh_details", return_value={"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}):
                    with patch("kura.cli.subprocess.run", side_effect=fake_run):
                        self.assertEqual(_runpod_run_over_ssh(run_dir, ssh_timeout_sec=1, job_timeout_sec=1), 0)

            argv_text = "\n".join(" ".join(map(str, call[0][0])) if isinstance(call[0][0], list) else str(call[0][0]) for call in calls)
            self.assertNotIn("hf-secret", argv_text)
            self.assertIn("sleep 43200", argv_text)
            self.assertIn("KURA_LEASE_LOG_PATH=/workspace/runs/example/logs/stdout.log", argv_text)
            self.assertIn("RUNPOD_POD_ID=pod-1", argv_text)
            self.assertIn("runpodctl pod stop", argv_text)
            self.assertTrue(any(call[1].get("input") and "hf-secret" in str(call[1]["input"]) for call in calls))

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

            with patch("kura.run_commands._runpod_ssh_details", return_value={"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}):
                with patch("kura.cli.subprocess.run", side_effect=fake_run):
                    self.assertEqual(_runpod_run_over_ssh(run_dir, ssh_timeout_sec=1, job_timeout_sec=1, max_lease_sec=0), 0)

            argv_text = "\n".join(" ".join(map(str, call[0][0])) if isinstance(call[0][0], list) else str(call[0][0]) for call in calls)
            self.assertNotIn("runpodctl pod stop", argv_text)

    def test_reconcile_runpod_exited_is_unknown_without_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/r1.json"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors._runpod_request", return_value={"id": "pod-1", "desiredStatus": "EXITED"}):
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
                    with patch("kura.run_commands._runpod_ssh_details", return_value={"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}):
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
                with patch("kura.run_commands.stage_run", return_value=0), \
                     patch("kura.run_commands.launch_run", return_value=0), \
                     patch("kura.run_commands._runpod_run_over_ssh", return_value=0), \
                     patch("kura.run_commands.download_with_retries", return_value=1), \
                     patch("kura.run_commands.stop_run") as stop:
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
                with patch("kura.run_commands.stage_run", return_value=0), \
                     patch("kura.run_commands.launch_run", return_value=0), \
                     patch("kura.run_commands._runpod_run_over_ssh", side_effect=subprocess.TimeoutExpired(["runpod-remote-job", "example"], 1)), \
                     patch("kura.run_commands._notify") as notify, \
                     patch("kura.run_commands.stop_run") as stop:
                    code = cmd_run_remote(argparse.Namespace(run_id="example", upload_timeout=1, job_timeout=1, download_attempts=1, download_interval=1, notify="ntfy", hold_for="30m", notify_repeat_interval="10m"))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            stop.assert_not_called()
            notify.assert_called_once()
            self.assertIn("controller failed", notify.call_args.kwargs["subject"])
            self.assertIn("may still be running and billing", notify.call_args.kwargs["body"])
            self.assertIn("kura run stop example", notify.call_args.kwargs["body"])

    def test_run_remote_stops_pod_immediately_when_hold_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("runpod: {api_key_env: RUNPOD_API_KEY, gpu_type_ids: [NVIDIA A40]}\n", encoding="utf-8")
            (root / "runs" / "example").mkdir(parents=True)
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.stage_run", return_value=0), \
                     patch("kura.run_commands.launch_run", return_value=0), \
                     patch("kura.run_commands._runpod_run_over_ssh", return_value=0), \
                     patch("kura.run_commands.download_with_retries", return_value=0), \
                     patch("kura.run_commands.stop_run") as stop:
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
                with patch("kura.run_commands.stage_run", return_value=0), \
                     patch("kura.run_commands.launch_run", return_value=0), \
                     patch("kura.run_commands._runpod_run_over_ssh", return_value=0) as remote_run, \
                     patch("kura.run_commands.download_with_retries", return_value=0), \
                     patch("kura.run_commands._sleep_with_completion_reminders") as hold, \
                     patch("kura.run_commands.stop_run") as stop:
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
                with patch("kura.run_commands.stage_run", return_value=0), \
                     patch("kura.run_commands.launch_run", return_value=0), \
                     patch("kura.run_commands._runpod_run_over_ssh", return_value=0), \
                     patch("kura.run_commands.download_with_retries", return_value=0), \
                     patch("kura.notifications.time.sleep") as sleep, \
                     patch("kura.run_commands._notify") as initial_notify, \
                     patch("kura.notifications.notify") as reminder_notify, \
                     patch("kura.run_commands.stop_run") as stop:
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
                with patch("kura.run_commands.stage_run", return_value=0), \
                     patch("kura.run_commands.launch_run", return_value=0), \
                     patch("kura.run_commands._runpod_run_over_ssh", return_value=0), \
                     patch("kura.run_commands.download_with_retries", return_value=0), \
                     patch("kura.run_commands._sleep_with_completion_reminders", side_effect=KeyboardInterrupt), \
                     patch("kura.run_commands.stop_run") as stop:
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
                with patch("kura.executors._runpod_request", return_value={}) as request:
                    status = stop_runpod(run_dir, self._config())
            self.assertEqual(status["state"], "interrupted")
            self.assertEqual(request.call_args.args[:3], ("DELETE", "/pods/pod-1", "api-secret"))

    def test_stop_runpod_preserves_completed_status_after_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run_dir(Path(directory))
            (run_dir / "status.json").write_text(json.dumps({"state": "completed", "exit_code": 0, "pod_id": "pod-1"}), encoding="utf-8")
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False):
                with patch("kura.executors._runpod_request", return_value={}):
                    status = stop_runpod(run_dir, self._config())
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["exit_code"], 0)
            self.assertIn("pod_stopped_at", status)
