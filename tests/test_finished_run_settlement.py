"""A finished training run is settled by one rule on every executor: the same trainer exit,
outputs, and states give the same status and the same records on Docker and on RunPod. Each
executor only brings the run's files local first (Docker: they already are; RunPod: from the
snapshot it downloaded)."""

from __future__ import annotations

import io
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from tests.test_artifact_publication import _safetensors_bytes

SILENT = ""
SHORT = "steps:  50%|#####     | 5/10 [00:05<00:05, 1.0it/s, avr_loss=0.5]\n"


def _run(root: Path, executor: str, *, exit_code: int, contract: bool, keep_state: bool, log: str, files: dict[str, bytes]) -> Path:
    (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    run_dir = root / "runs" / "example"
    (run_dir / "resolved").mkdir(parents=True)
    (run_dir / "realizations").mkdir()
    run = {"id": "example", "type": "train", "backend": {"name": "musubi-tuner", "config": {}}, "model": {"base": "example/model"},
           "recipe": {"steps": 10, "seed": 1}, "recovery": {"training_state": {"enabled": keep_state, "keep_generations": 2}}}
    (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(run), encoding="utf-8")
    lock = {"cwd": "/opt/musubi-tuner", "argv": ["python"], "env": {}, "training_state_managed": keep_state,
            **({"output_contract": {"required": [{"role": "trained-adapter", "suffix": ".safetensors", "minimum": 1}]}} if contract else {})}
    (run_dir / "resolved" / "backend-command.lock.json").write_text(json.dumps(lock), encoding="utf-8")
    realization = {"id": "launch", "executor": executor, **({"container": {"id": "container-1"}} if executor == "docker" else {"pod": {"id": "pod-1"}})}
    (run_dir / "realizations" / "launch.json").write_text(json.dumps(realization), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"state": "running", "last_realization": "realizations/launch.json",
                                                     **({"pod_id": "pod-1"} if executor == "runpod" else {})}), encoding="utf-8")
    home = run_dir if executor == "docker" else run_dir / "downloads" / "example"
    (home / "logs").mkdir(parents=True)
    (home / "logs" / "stdout.log").write_text(log, encoding="utf-8")
    for relative, content in files.items():
        (home / "outputs" / relative).parent.mkdir(parents=True, exist_ok=True)
        (home / "outputs" / relative).write_bytes(content)
    if executor == "runpod":
        (home / "realizations").mkdir()
        (home / "realizations" / "remote-exit-20260101.json").write_text(json.dumps({"timestamp": "2026-01-01T00:00:00+00:00", "exit_code": exit_code}), encoding="utf-8")
    return run_dir


def _finish(root: Path, executor: str, exit_code: int) -> dict:
    run_dir = root / "runs" / "example"
    if executor == "docker":
        from kura.executors.docker import reconcile_docker

        observed = subprocess.CompletedProcess([], 0, json.dumps({"Running": False, "ExitCode": exit_code}), "")
        with patch("kura.executors.docker.subprocess.run", return_value=observed):
            reconcile_docker(run_dir)
    else:
        from kura.run_commands.runpod_ssh import _download_run_unlocked

        previous = Path.cwd()
        os.chdir(root)
        try:
            with patch("sys.stdout", io.StringIO()), patch("sys.stderr", io.StringIO()):
                _download_run_unlocked("example")
        finally:
            os.chdir(previous)
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


def _records(run_dir: Path) -> dict:
    directory = run_dir / "realizations"
    unverified = directory / "launch.publication-unverified.json"
    return {
        "unverified_outputs": json.loads(unverified.read_text(encoding="utf-8"))["outputs"] if unverified.is_file() else None,
        "attempts": len(list(directory.glob("launch.publication-attempt-*.json"))),
    }


FIELDS = ("state", "publication_state", "outputs", "last_step", "total_steps")


class OneSettlementTests(unittest.TestCase):
    def _settle_on_both(self, **run: object) -> dict[str, tuple[dict, dict]]:
        results = {}
        for executor in ("docker", "runpod"):
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                run_dir = _run(root, executor, **run)
                status = _finish(root, executor, run["exit_code"])
                results[executor] = ({key: status.get(key) for key in FIELDS}, _records(run_dir))
        return results

    def assert_same(self, results: dict[str, tuple[dict, dict]]) -> tuple[dict, dict]:
        self.assertEqual(results["docker"], results["runpod"])
        return results["docker"]

    def test_a_run_without_an_output_contract_records_the_outputs_its_status_lists(self) -> None:
        status, records = self.assert_same(self._settle_on_both(
            exit_code=0, contract=False, keep_state=False, log=SILENT, files={"example.safetensors": _safetensors_bytes()}))
        self.assertEqual((status["state"], status["publication_state"]), ("completed", "legacy-unverified"))
        self.assertEqual(status["outputs"], ["outputs/example.safetensors"])
        self.assertEqual(records["unverified_outputs"], status["outputs"])

    def test_a_failed_run_lists_no_published_outputs_on_either_executor(self) -> None:
        status, records = self.assert_same(self._settle_on_both(
            exit_code=1, contract=True, keep_state=False, log=SHORT, files={"example-step00000005.safetensors": _safetensors_bytes()}))
        self.assertEqual((status["state"], status["publication_state"], status["outputs"]), ("failed", "not-required", []))
        self.assertEqual((status["last_step"], records["attempts"]), (5, 0))

    def test_a_completed_run_held_for_its_missing_state_still_counts_its_steps(self) -> None:
        status, records = self.assert_same(self._settle_on_both(
            exit_code=0, contract=True, keep_state=True, log=SILENT, files={"example.safetensors": _safetensors_bytes()}))
        self.assertEqual((status["state"], status["publication_state"]), ("recovery_required", "blocked"))
        self.assertEqual((status["last_step"], status["total_steps"], records["attempts"]), (10, 10, 1))


class SettlementRuleTests(unittest.TestCase):
    def test_a_failed_run_that_could_not_bring_every_file_home_stays_failed(self) -> None:
        from kura.executors.common import settle_finished_training

        with tempfile.TemporaryDirectory() as directory:
            run_dir = _run(Path(directory), "runpod", exit_code=1, contract=True, keep_state=False, log=SILENT, files={})
            status = {"state": "failed"}
            settle_finished_training(run_dir, "launch", state="failed", collection_errors=("outputs collide",))(status)
            self.assertEqual((status["state"], status["publication_state"]), ("failed", "blocked"))
            self.assertIn("outputs collide", status["publication_error"])

    def test_a_run_that_keeps_no_state_writes_no_blocked_record_for_a_state_directory(self) -> None:
        # A run compiled to keep no state never has a state error, so no publication-attempt
        # record calls a completed run blocked.
        from kura.executors.common import settle_finished_training

        with tempfile.TemporaryDirectory() as directory:
            run_dir = _run(Path(directory), "docker", exit_code=0, contract=True, keep_state=False, log=SILENT,
                           files={"example.safetensors": _safetensors_bytes(), "example-step00000005-state/optimizer.bin": b"not a torch archive"})
            status = {"state": "publishing"}
            settle_finished_training(run_dir, "launch", state="completed")(status)
            self.assertEqual(status["state"], "completed")
            self.assertEqual(list((run_dir / "realizations").glob("launch.publication-attempt-*.json")), [])


if __name__ == "__main__":
    unittest.main()
