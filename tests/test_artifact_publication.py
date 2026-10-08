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
from kura.executors import docker as docker_executor
from kura.cli import _run_cleanup_candidates
from kura.run_commands.runpod_ssh import cmd_run_download
from kura.run_commands.experiment import format_run_completion
from tests.platform_support import POSIX_PATHS, posix_only


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

    def test_a_run_without_an_output_contract_records_its_unverified_publication(self) -> None:
        from kura.status_projection import project_status

        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            # A legacy run: no frozen command lock, so no outputs to verify.
            (run_dir / "resolved" / "backend-command.lock.json").unlink()
            (run_dir / "resolved" / "manifest.lock.yaml").unlink()
            status = self._reconcile(run_dir)
            self.assertEqual((status["state"], status["publication_state"]), ("completed", "legacy-unverified"))
            recorded = json.loads((run_dir / "realizations" / "launch.publication-unverified.json").read_text(encoding="utf-8"))
            self.assertEqual(recorded["kind"], "publication_unverified")
            self.assertEqual(project_status(run_dir)["state"], "completed")

    def _reconcile(self, run_dir: Path) -> dict[str, object]:
        observed = subprocess.CompletedProcess([], 0, '{"Running": false, "ExitCode": 0}', "")
        with patch("kura.executors.docker.subprocess.run", return_value=observed):
            return reconcile_docker(run_dir)

    def _manifest_view(self, run_dir: Path) -> tuple[Path, Path]:
        root = run_dir.parent.parent
        dataset = root / "datasets" / "tiny"
        dataset.mkdir(parents=True)
        source = dataset / "a.png"
        source.write_bytes(b"source")
        stat = source.stat()
        view = run_dir / "cache" / "dataset-view" / "ai-toolkit" / "tiny"
        view.mkdir(parents=True)
        link = view / "a.png"
        link.symlink_to("/workspace/datasets/tiny/a.png")
        (view / "_latent_cache").mkdir()
        (view / "_latent_cache" / "a.safetensors").write_bytes(b"cache")
        lock = {
            "schema_version": 2,
            "verification": "content-hash-at-compile",
            "input_sha256": "sha256:fixture",
            "files": [{
                "source": "datasets/tiny/a.png",
                "container_source": "/workspace/datasets/tiny/a.png",
                "stat": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns},
            }],
            "authoring_files": [],
            "dataset_roots": [{"dataset": "tiny", "logical": "datasets/tiny", "physical": str(dataset.resolve())}],
            "views": [{
                "dataset": "tiny",
                "root": "runs/example/cache/dataset-view/ai-toolkit/tiny",
                "links": [{
                    "path": "runs/example/cache/dataset-view/ai-toolkit/tiny/a.png",
                    "target": "/workspace/datasets/tiny/a.png",
                    "input_id": "d0:s0:f0",
                }],
                "files": [],
            }],
        }
        (run_dir / "resolved" / "dataset-input.lock.json").write_text(json.dumps(lock), encoding="utf-8")
        return source, view

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

    def test_blocked_publication_keeps_view_until_recovery_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            _, view = self._manifest_view(run_dir)

            blocked = self._reconcile(run_dir)

            self.assertEqual(blocked["publication_state"], "blocked")
            self.assertTrue(blocked["recovery_required"])
            self.assertEqual(blocked["dataset_input_postflight"]["view_cleanup"], "deferred")
            self.assertNotIn("cleanup_record", blocked["dataset_input_postflight"])
            self.assertTrue(view.exists())

            output = run_dir / "outputs" / "example.safetensors"
            output.parent.mkdir()
            output.write_bytes(_safetensors_bytes())
            recovered = self._reconcile(run_dir)

            self.assertEqual(recovered["publication_state"], "completed")
            self.assertFalse(recovered["recovery_required"])
            self.assertEqual(recovered["dataset_input_postflight"]["view_cleanup"], "removed")
            self.assertFalse(view.parent.parent.exists())

    def test_recovery_view_becomes_a_cleanup_remnant_only_after_the_container_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            run_dir = self._run(workspace)
            _, view = self._manifest_view(run_dir)
            blocked = self._reconcile(run_dir)
            self.assertTrue(blocked["recovery_required"])
            self.assertEqual(blocked["dataset_input_postflight"]["view_cleanup"], "deferred")

            def remnants() -> list[str]:
                return [
                    item["id"]
                    for item in _run_cleanup_candidates(workspace, keep_last=30, delete_final_artifacts=False)
                    if item["classification"] == "safe-run-dataset-view-remnant"
                ]

            self.assertEqual(remnants(), [])

            # A container that still exists but cannot report an exit code is
            # also "unknown"; it is not evidence that the container is gone.
            unclear = subprocess.CompletedProcess([], 0, '{"Running": false}', "")
            with patch("kura.executors.docker.subprocess.run", return_value=unclear):
                reconcile_docker(run_dir)
            self.assertEqual(remnants(), [])

            missing = subprocess.CompletedProcess([], 1, "", "Error: No such container: container-1")
            with patch("kura.executors.docker.subprocess.run", return_value=missing):
                status = reconcile_docker(run_dir)

            observation = json.loads((run_dir / status["last_observation"]).read_text(encoding="utf-8"))
            self.assertIs(observation["container_missing"], True)
            self.assertTrue(status["recovery_required"])
            self.assertEqual(remnants(), ["example"])
            self.assertTrue(view.exists())

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

    def test_records_say_what_they_are_and_a_legacy_manifest_still_matches(self) -> None:
        from kura.artifact_publication import publish_outputs

        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            output = run_dir / "outputs" / "example.safetensors"
            output.parent.mkdir()
            output.write_bytes(_safetensors_bytes())
            status = self._reconcile(run_dir)
            manifest_path = run_dir / status["publication_manifest"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual((manifest["kind"], manifest["schema_version"]), ("publication", 1))
            on_disk = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual((on_disk["kind"], on_disk["schema_version"]), ("run_status", 1))
            # The fixture wrote the realization by hand; the observation is Kura's.
            realization = json.loads((run_dir / on_disk["last_realization"]).read_text(encoding="utf-8"))
            observation = json.loads((run_dir / on_disk["last_observation"]).read_text(encoding="utf-8"))
            self.assertEqual((observation["kind"], observation["schema_version"]), ("observation", 1))
            # A manifest written before records carried kinds is the same publication.
            legacy = {key: value for key, value in manifest.items() if key != "kind"}
            manifest_path.write_text(json.dumps(legacy), encoding="utf-8")
            contract = {"required": [{"role": "trained-adapter", "suffix": ".safetensors", "minimum": 1}]}
            self.assertEqual(publish_outputs(run_dir, realization["id"], contract)[0], status["publication_manifest"])

    def test_terminal_reconcile_records_postflight_then_removes_view(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            _, view = self._manifest_view(run_dir)
            output = run_dir / "outputs" / "example.safetensors"
            output.parent.mkdir()
            output.write_bytes(_safetensors_bytes())

            status = self._reconcile(run_dir)

            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["publication_state"], "completed")
            self.assertFalse(view.parent.parent.exists())
            postflight = status["dataset_input_postflight"]
            self.assertEqual(postflight["status"], "matched")
            self.assertEqual(postflight["view_cleanup"], "removed")
            record = json.loads((run_dir / postflight["record"]).read_text(encoding="utf-8"))
            self.assertEqual(record["source_stat_verification"], "matched")
            self.assertEqual(record["view_link_verification"], "matched")
            self.assertIsInstance(record["execution_ended_at"], str)
            self.assertIsInstance(record["observed_at"], str)
            cleanup = json.loads((run_dir / postflight["cleanup_record"]).read_text(encoding="utf-8"))
            self.assertEqual(cleanup["status"], "removed")
            events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn("dataset_input_postflight", [item.get("event") for item in events])

    def test_input_drift_warns_without_invalidating_publication_and_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            source, view = self._manifest_view(run_dir)
            source.write_bytes(b"changed during training")
            output = run_dir / "outputs" / "example.safetensors"
            output.parent.mkdir()
            output.write_bytes(_safetensors_bytes())

            first = self._reconcile(run_dir)
            second = self._reconcile(run_dir)

            self.assertEqual(first["state"], "completed")
            self.assertEqual(first["publication_state"], "completed")
            self.assertEqual(first["dataset_input_postflight"]["status"], "changed")
            self.assertIn("between compile and post-training observation", first["dataset_input_postflight"]["warning"])
            self.assertEqual(second["dataset_input_postflight"], first["dataset_input_postflight"])
            self.assertFalse(view.parent.parent.exists())

    def test_later_reconciles_do_not_rescan_events_and_never_duplicate_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            self._manifest_view(run_dir)
            output = run_dir / "outputs" / "example.safetensors"
            output.parent.mkdir()
            output.write_bytes(_safetensors_bytes())
            self._reconcile(run_dir)

            with patch("kura.executors.docker._event_exists", side_effect=AssertionError("rescanned events")):
                self._reconcile(run_dir)
                self._reconcile(run_dir)

            # A crash after the append but before the status projection falls
            # back to the scan and still appends each event exactly once.
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status.pop("dataset_input_postflight")
            status_path.write_text(json.dumps(status), encoding="utf-8")
            self._reconcile(run_dir)

            events = [
                json.loads(line)
                for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            names = [item.get("event") for item in events]
            self.assertEqual(names.count("dataset_input_postflight"), 1)
            self.assertEqual(names.count("dataset_view_cleanup"), 1)

    def test_cleanup_failure_is_retried_on_later_reconcile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            _, view = self._manifest_view(run_dir)
            output = run_dir / "outputs" / "example.safetensors"
            output.parent.mkdir()
            output.write_bytes(_safetensors_bytes())
            actual_remove = docker_executor.remove_dataset_views
            attempts = 0

            def fail_once(*args: object, **kwargs: object) -> dict[str, object]:
                nonlocal attempts
                attempts += 1
                if attempts == 1:
                    raise OSError("busy")
                return actual_remove(*args, **kwargs)

            with patch("kura.executors.docker.remove_dataset_views", side_effect=fail_once):
                first = self._reconcile(run_dir)
                second = self._reconcile(run_dir)

            self.assertEqual(first["dataset_input_postflight"]["view_cleanup"], "failed")
            self.assertEqual(second["dataset_input_postflight"]["view_cleanup"], "removed")
            self.assertEqual(attempts, 2)
            self.assertFalse(view.parent.parent.exists())

    def test_corrupt_input_lock_records_uncheckable_without_breaking_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            output = run_dir / "outputs" / "example.safetensors"
            output.parent.mkdir()
            output.write_bytes(_safetensors_bytes())
            (run_dir / "resolved" / "dataset-input.lock.json").write_text("{", encoding="utf-8")

            status = self._reconcile(run_dir)

            self.assertEqual(status["state"], "completed")
            self.assertEqual(status["publication_state"], "completed")
            self.assertEqual(status["dataset_input_postflight"]["status"], "uncheckable")
            self.assertEqual(status["dataset_input_postflight"]["view_cleanup"], "deferred")

    def test_existing_postflight_record_backfills_missing_event(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            _, _ = self._manifest_view(run_dir)
            output = run_dir / "outputs" / "example.safetensors"
            output.parent.mkdir()
            output.write_bytes(_safetensors_bytes())
            first = self._reconcile(run_dir)
            events_path = run_dir / "logs" / "events.jsonl"
            events = [
                json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("event") != "dataset_input_postflight"
            ]
            events_path.write_text("".join(json.dumps(item) + "\n" for item in events), encoding="utf-8")
            # A crash between the record and its event also precedes the status
            # projection, which is written last.
            status_path = run_dir / "status.json"
            status = json.loads(status_path.read_text(encoding="utf-8"))
            status.pop("dataset_input_postflight")
            status_path.write_text(json.dumps(status), encoding="utf-8")

            second = self._reconcile(run_dir)

            repaired = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            matching = [item for item in repaired if item.get("event") == "dataset_input_postflight"]
            self.assertEqual(len(matching), 1)
            self.assertEqual(second["dataset_input_postflight"], first["dataset_input_postflight"])

    def test_failed_trainer_cleans_view_after_publication_decision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            _, view = self._manifest_view(run_dir)
            observed = subprocess.CompletedProcess([], 0, '{"Running": false, "ExitCode": 9}', "")

            with patch("kura.executors.docker.subprocess.run", return_value=observed):
                status = reconcile_docker(run_dir)

            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["publication_state"], "not-required")
            self.assertEqual(status["dataset_input_postflight"]["view_cleanup"], "removed")
            self.assertFalse(view.parent.parent.exists())

    def test_unknown_container_keeps_view_for_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            _, view = self._manifest_view(run_dir)
            observed = subprocess.CompletedProcess([], 1, "", "Error: No such container")

            with patch("kura.executors.docker.subprocess.run", return_value=observed):
                status = reconcile_docker(run_dir)

            self.assertEqual(status["state"], "unknown")
            self.assertTrue(view.exists())
            self.assertNotIn("dataset_input_postflight", status)

    def test_reconcile_preserves_completed_publication_after_outputs_change(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            output = run_dir / "outputs" / "example.safetensors"
            output.parent.mkdir()
            output.write_bytes(_safetensors_bytes())
            first = self._reconcile(run_dir)
            self.assertEqual(first["state"], "completed")
            (run_dir / "outputs" / "later.txt").write_text("later", encoding="utf-8")

            second = self._reconcile(run_dir)

            self.assertEqual(second["state"], "completed")
            self.assertEqual(second["publication_state"], "completed")
            self.assertEqual(second["publication_manifest"], first["publication_manifest"])
            self.assertEqual(second["outputs"], first["outputs"])
            self.assertNotIn("last_publication_attempt", second)

    def test_reconcile_does_not_publish_into_a_newer_realization_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = self._run(Path(directory))
            output = run_dir / "outputs" / "example.safetensors"
            output.parent.mkdir()
            output.write_bytes(_safetensors_bytes())
            next_ref = "realizations/next.json"

            def replace_latest_run(_run_dir: Path) -> bool:
                status_path = run_dir / "status.json"
                status = json.loads(status_path.read_text(encoding="utf-8"))
                status.update({"state": "running", "last_realization": next_ref})
                for key in ("exit_code", "ended", "execution_state", "publication_state", "last_observation"):
                    status.pop(key, None)
                status_path.write_text(json.dumps(status), encoding="utf-8")
                return False

            observed = subprocess.CompletedProcess([], 0, '{"Running": false, "ExitCode": 0}', "")
            with patch("kura.executors.docker.subprocess.run", return_value=observed), patch(
                "kura.executors.docker.training_state_capture_required", side_effect=replace_latest_run
            ):
                result = reconcile_docker(run_dir)

            self.assertEqual(result["last_realization"], next_ref)
            self.assertEqual(result["state"], "running")
            self.assertNotIn("publication_state", result)
            self.assertNotIn("publication_manifest", result)
            self.assertNotIn("outputs", result)

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
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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

    @posix_only(POSIX_PATHS)
    def test_download_completes_only_after_valid_output_publication(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
            # Collected again (as after `kura run download`) with a file the publication never listed:
            # the run stays published, as a Docker run re-observed after publication does.
            (downloaded / "outputs" / "notes.txt").write_text("later", encoding="utf-8")
            os.chdir(root)
            try:
                self.assertEqual(cmd_run_download(argparse.Namespace(run_id="example", force=False)), 0)
            finally:
                os.chdir(previous)
            again = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual((again["state"], again["publication_state"], again["publication_manifest"]),
                             ("completed", "completed", "realizations/launch.publication.json"))

    @posix_only(POSIX_PATHS)
    def test_input_postflight_failure_never_blocks_download_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
                with patch(
                    "kura.executors.runpod.finalize_runpod_dataset_handoff",
                    side_effect=OSError("disk full while promoting records"),
                ):
                    code = cmd_run_download(argparse.Namespace(run_id="example", force=False))
            finally:
                os.chdir(previous)
            self.assertEqual(code, 0)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["state"], "completed")
            projected = status["dataset_input_postflight"]
            self.assertEqual(projected["status"], "uncheckable")
            self.assertIn("reproducibility is not confirmed", projected["warning"])
            # The fact lives in an append-only record that status only projects.
            record = json.loads((run_dir / projected["record"]).read_text(encoding="utf-8"))
            self.assertEqual(record["status"], "uncheckable")
            self.assertIn("disk full", record["error"])
            events = [json.loads(line) for line in (run_dir / "logs" / "events.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertIn(projected["record"], [item.get("record") for item in events if item.get("event") == "dataset_input_postflight"])

    @posix_only(POSIX_PATHS)
    def test_download_does_not_complete_when_adapter_is_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
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
