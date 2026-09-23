"""End-to-end local publication facts at the executor's public reconcile seam."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from kura.backends.ai_toolkit import command_ai_toolkit
from kura.executors.docker import reconcile_docker
from kura.run_commands.runpod_ssh import cmd_run_download
from kura.run_commands.experiment import format_run_completion


def _safetensors_bytes() -> bytes:
    header = json.dumps({"lora_A.weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    return len(header).to_bytes(8, "little") + header + b"\0\0\0\0"


class LocalOutputPublicationTests(unittest.TestCase):
    def _run(self, root: Path) -> Path:
        run_dir = root / "runs" / "example"
        (run_dir / "resolved").mkdir(parents=True)
        (run_dir / "realizations").mkdir()
        run = {
            "id": "example", "type": "train", "backend": {"name": "ai-toolkit", "config": {}},
            "model": {"base": "example/model"}, "recipe": {"steps": 1, "seed": 1},
            "recovery": {"training_state": {"enabled": False}},
        }
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
        spec = command_ai_toolkit(run)
        (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps(spec), encoding="utf-8")
        (run_dir / "realizations" / "launch.json").write_text(json.dumps({"id": "launch", "container": {"id": "container-1"}}), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/launch.json"}), encoding="utf-8")
        return run_dir

    def _reconcile(self, run_dir: Path) -> dict[str, object]:
        observed = subprocess.CompletedProcess([], 0, '{"Running": false, "ExitCode": 0}', "")
        with patch("kura.executors.docker.subprocess.run", return_value=observed):
            return reconcile_docker(run_dir)

    def test_missing_required_adapter_is_not_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            status = self._reconcile(run_dir)
            self.assertEqual(status["execution_state"], "completed")
            self.assertEqual(status["state"], "recovery_required")
            self.assertEqual(status["publication_state"], "blocked")
            self.assertIn("required trained-adapter", status["publication_error"])
            attempt = json.loads((run_dir / status["last_publication_attempt"]).read_text(encoding="utf-8"))
            self.assertEqual(attempt["result"], "blocked")
            summary = format_run_completion(run_dir.parent.parent, run_dir, status)
            self.assertIn("trainer completed", summary)
            self.assertIn("required trained-adapter", summary)

    def test_missing_frozen_command_is_not_treated_as_legacy(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            (run_dir / "resolved" / "backend-command.lock.json").unlink()
            status = self._reconcile(run_dir)
            self.assertEqual(status["execution_state"], "completed")
            self.assertEqual(status["state"], "recovery_required")
            self.assertEqual(status["publication_state"], "blocked")
            self.assertIn("frozen backend command", status["publication_error"])

    def test_valid_adapter_gets_immutable_inventory_before_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            output = run_dir / "outputs" / "example.safetensors"
            output.parent.mkdir()
            output.write_bytes(_safetensors_bytes())
            status = self._reconcile(run_dir)
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["publication_state"], "completed")
            manifest = json.loads((run_dir / status["publication_manifest"]).read_text(encoding="utf-8"))
            self.assertEqual(manifest["files"], [{
                "path": "outputs/example.safetensors",
                "size": len(_safetensors_bytes()),
                "sha256": hashlib.sha256(_safetensors_bytes()).hexdigest(),
            }])

    def test_truncated_adapter_is_not_completed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            output = run_dir / "outputs" / "example.safetensors"
            output.parent.mkdir()
            output.write_bytes(b"truncated")
            status = self._reconcile(run_dir)
            self.assertEqual(status["state"], "recovery_required")
            self.assertIn("invalid safetensors", status["publication_error"])

    def test_publication_retries_without_relaunching_the_trainer(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            first = self._reconcile(run_dir)
            self.assertEqual(first["state"], "recovery_required")
            output = run_dir / "outputs" / "example.safetensors"
            output.parent.mkdir()
            output.write_bytes(_safetensors_bytes())
            second = self._reconcile(run_dir)
            self.assertEqual(second["state"], "completed")
            self.assertIn("publication_manifest", second)
            self.assertEqual(second["last_publication_attempt"], first["last_publication_attempt"])

    def test_unchanged_output_from_before_launch_cannot_satisfy_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            output = run_dir / "outputs" / "example.safetensors"
            output.parent.mkdir()
            output.write_bytes(_safetensors_bytes())
            stat = output.stat()
            realization = run_dir / "realizations" / "launch.json"
            record = json.loads(realization.read_text(encoding="utf-8"))
            record["output_baseline"] = {"outputs/example.safetensors": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}}
            realization.write_text(json.dumps(record), encoding="utf-8")
            status = self._reconcile(run_dir)
            self.assertEqual(status["state"], "recovery_required")
            self.assertIn("required trained-adapter", status["publication_error"])


class RunPodOutputPublicationTests(unittest.TestCase):
    def test_download_with_missing_frozen_command_needs_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            downloaded = run_dir / "downloads" / "example"
            (downloaded / "outputs").mkdir(parents=True)
            (downloaded / "realizations").mkdir()
            (run_dir / "resolved").mkdir()
            (run_dir / "resolved" / "manifest.lock.yaml").write_text("type: train\n", encoding="utf-8")
            (run_dir / "realizations").mkdir()
            (downloaded / "realizations" / "remote-exit-20260101.json").write_text(
                json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "exit_code": 0}), encoding="utf-8",
            )
            (run_dir / "realizations" / "launch.json").write_text(json.dumps({"id": "launch"}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({
                "state": "running", "pod_id": "pod-1", "last_realization": "realizations/launch.json",
            }), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                code = cmd_run_download(argparse.Namespace(run_id="example", force=False))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "recovery_required")
            self.assertEqual(status["execution_state"], "completed")
            self.assertEqual(status["exit_code"], 0)
            self.assertEqual(status["publication_state"], "blocked")
            self.assertTrue(status["recovery_required"])
            self.assertIn("frozen backend command", status["publication_error"])

    def test_download_completes_only_after_valid_output_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            downloaded = run_dir / "downloads" / "example"
            (downloaded / "outputs").mkdir(parents=True)
            (downloaded / "realizations").mkdir()
            (run_dir / "resolved").mkdir()
            (run_dir / "realizations").mkdir()
            (downloaded / "outputs" / "example.safetensors").write_bytes(_safetensors_bytes())
            (downloaded / "realizations" / "remote-exit-20260101.json").write_text(
                json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "exit_code": 0}), encoding="utf-8",
            )
            (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({
                "output_contract": {"required": [{"role": "trained-adapter", "suffix": ".safetensors", "minimum": 1}]},
            }), encoding="utf-8")
            (run_dir / "realizations" / "launch.json").write_text(json.dumps({"id": "launch"}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({
                "state": "running", "pod_id": "pod-1", "last_realization": "realizations/launch.json",
            }), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                code = cmd_run_download(argparse.Namespace(run_id="example", force=False))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["publication_state"], "completed")
            self.assertEqual(status["publication_manifest"], "realizations/launch.publication.json")

    def test_download_does_not_complete_when_adapter_is_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 1\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            downloaded = run_dir / "downloads" / "example"
            (downloaded / "outputs").mkdir(parents=True)
            (downloaded / "realizations").mkdir()
            (run_dir / "resolved").mkdir()
            (run_dir / "realizations").mkdir()
            (downloaded / "outputs" / "example.safetensors").write_bytes(b"truncated")
            (downloaded / "realizations" / "remote-exit-20260101.json").write_text(
                json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "exit_code": 0}), encoding="utf-8",
            )
            (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps({
                "output_contract": {"required": [{"role": "trained-adapter", "suffix": ".safetensors", "minimum": 1}]},
            }), encoding="utf-8")
            (run_dir / "realizations" / "launch.json").write_text(json.dumps({"id": "launch"}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({
                "state": "running", "pod_id": "pod-1", "last_realization": "realizations/launch.json",
            }), encoding="utf-8")
            previous = Path.cwd()
            os.chdir(root)
            try:
                code = cmd_run_download(argparse.Namespace(run_id="example", force=False))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 1)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "recovery_required")
            self.assertEqual(status["execution_state"], "completed")
            self.assertEqual(status["exit_code"], 0)
            self.assertEqual(status["publication_state"], "blocked")
            self.assertTrue(status["recovery_required"])
            self.assertIn("invalid safetensors", status["publication_error"])
            self.assertEqual(
                json.loads((run_dir / status["last_publication_attempt"]).read_text(encoding="utf-8"))["result"],
                "blocked",
            )
            (downloaded / "outputs" / "example.safetensors").write_bytes(_safetensors_bytes())
            previous = Path.cwd()
            os.chdir(root)
            try:
                retry_code = cmd_run_download(argparse.Namespace(run_id="example", force=False))
            finally:
                os.chdir(previous)
            self.assertEqual(retry_code, 0)
            recovered = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(recovered["state"], "completed")
            self.assertEqual(recovered["publication_state"], "completed")
            self.assertFalse(recovered["recovery_required"])


if __name__ == "__main__":
    unittest.main()
