"""How a run's end is recorded is decided once for both executors: who needs a person, what a state
verification error does, which zone end times are in, and what `kura run stop` warns before it
deletes a Pod's uncollected work."""

from __future__ import annotations

import argparse
import io
import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import yaml


def _runpod_run(root: Path, *, exit_code: int, timestamp: str = "2026-01-01T00:00:00+00:00", state_dir: bool = False) -> Path:
    (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    run_dir = root / "runs" / "source"
    downloaded = run_dir / "downloads" / "source"
    (downloaded / "outputs").mkdir(parents=True)
    (downloaded / "realizations").mkdir()
    if state_dir:
        (downloaded / "outputs" / "source-step00000100-state").mkdir()
    (downloaded / "realizations" / "remote-exit-20260101.json").write_text(json.dumps({"timestamp": timestamp, "exit_code": exit_code}), encoding="utf-8")
    (run_dir / "resolved").mkdir(parents=True)
    (run_dir / "realizations").mkdir()
    manifest = {"id": "source", "type": "train", "backend": {"name": "musubi-tuner", "config": {}}, "recipe": {"steps": 100, "seed": 1},
                "recovery": {"training_state": {"enabled": False}}}
    (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")
    (run_dir / "realizations" / "launch.json").write_text(json.dumps({"id": "launch", "executor": "runpod"}), encoding="utf-8")
    (run_dir / "status.json").write_text(json.dumps({"state": "running", "pod_id": "pod-1", "last_realization": "realizations/launch.json"}), encoding="utf-8")
    return run_dir


def _download(root: Path) -> int:
    from kura.run_commands.runpod_ssh import _download_run_unlocked

    previous = Path.cwd()
    os.chdir(root)
    try:
        with patch("sys.stderr", io.StringIO()), patch("sys.stdout", io.StringIO()):
            return _download_run_unlocked("source")
    finally:
        os.chdir(previous)


class NeedsAPersonTests(unittest.TestCase):
    def test_only_the_state_says_a_run_needs_a_person(self) -> None:
        # RunPod used to set a status flag when the remote job exited, meaning "download pending".
        from kura.monitor import collect_run_summary

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "runs" / "example"
            run_dir.mkdir(parents=True)
            (run_dir / "run.yaml").write_text("id: example\ntype: train\ncompute: {executor: runpod}\n", encoding="utf-8")
            for status, expected in (({"state": "running", "recovery_required": True}, False), ({"state": "recovery_required"}, True)):
                with self.subTest(status=status):
                    (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
                    self.assertEqual(collect_run_summary(root, "example").executor_info.recovery_required, expected)

    def test_collection_no_longer_writes_the_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = _runpod_run(Path(directory), exit_code=0)
            _download(Path(directory))
            self.assertNotIn("recovery_required", json.loads((run_dir / "status.json").read_text(encoding="utf-8")))


class StateVerificationErrorTests(unittest.TestCase):
    def test_a_runpod_state_error_hands_the_run_to_a_person_at_once_as_on_docker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _runpod_run(root, exit_code=0, state_dir=True)
            with patch("kura.run_commands.runpod_ssh.training_state_capture_required", return_value=True), \
                    patch("kura.run_commands.runpod_ssh.publish_completed_training_states", side_effect=ValueError("state manifest does not verify")):
                self.assertEqual(_download(root), 3)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["state"], "recovery_required")
        self.assertIn("state manifest does not verify", status["training_state_sync_error"])
        self.assertIn("state manifest does not verify", status["publication_error"])


class FailedRunStateErrorTests(unittest.TestCase):
    def test_a_failed_runpod_run_with_a_bad_state_stays_failed_as_on_docker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _runpod_run(root, exit_code=1, state_dir=True)
            with patch("kura.run_commands.runpod_ssh.training_state_capture_required", return_value=True), \
                    patch("kura.run_commands.runpod_ssh.publish_completed_training_states", side_effect=ValueError("state manifest does not verify")):
                _download(root)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        self.assertEqual((status["state"], status["publication_state"]), ("failed", "blocked"))
        self.assertIn("state manifest does not verify", status["training_state_sync_error"])


class RelaunchTests(unittest.TestCase):
    def test_a_new_launch_forgets_the_last_launchs_exit_and_collection(self) -> None:
        # Otherwise a stop of the new Pod would skip its confirmation as if it were collected.
        from kura.executors.runpod import PER_LAUNCH_STATUS_FIELDS

        for field in ("pod_id", "downloaded_run", "remote_state", "remote_exit_code", "remote_ended"):
            self.assertIn(field, PER_LAUNCH_STATUS_FIELDS)


class EndTimeTests(unittest.TestCase):
    def test_a_runpod_end_is_written_in_the_hosts_zone(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = _runpod_run(root, exit_code=0, timestamp="2026-01-01T03:04:05Z")
            _download(root)
            status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
        expected = datetime(2026, 1, 1, 3, 4, 5, tzinfo=timezone.utc).astimezone().isoformat()
        self.assertEqual((status["ended"], status["remote_ended"]), (expected, expected))

    def test_both_executors_convert_times_with_one_function(self) -> None:
        from kura.executors import common, docker
        from kura.run_commands import runpod_ssh

        self.assertIs(docker.host_time, common.host_time)
        self.assertIs(runpod_ssh.host_time, common.host_time)
        self.assertEqual(common.host_time("2026-10-01T03:45:40.123456789Z"),
                         datetime(2026, 10, 1, 3, 45, 40, 123456, tzinfo=timezone.utc).astimezone().isoformat())


class StopConfirmationTests(unittest.TestCase):
    def _stop(self, status: dict, *, executor: str = "runpod", yes: bool = False) -> tuple[int, str, bool]:
        from kura.run_commands.plan import cmd_run_stop

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
            run_dir = root / "runs" / "example"
            (run_dir / "realizations").mkdir(parents=True)
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({"executor": executor}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps({"last_realization": "realizations/r1.json", **status}), encoding="utf-8")
            err = io.StringIO()
            previous = Path.cwd()
            os.chdir(root)
            try:
                with patch("kura.run_commands.plan._stop_through_runner", return_value=None), \
                        patch("kura.run_commands.plan.stop_runpod", return_value={}) as stop_pod, \
                        patch("kura.run_commands.plan.stop_docker", return_value={}), \
                        patch("sys.stderr", err), patch("sys.stdout", io.StringIO()):
                    code = cmd_run_stop(argparse.Namespace(run_id="example", yes=yes))
            finally:
                os.chdir(previous)
            return code, err.getvalue(), stop_pod.called

    def test_a_pod_holding_uncollected_work_is_kept_until_the_stop_is_confirmed(self) -> None:
        for status, hint in (({"state": "running", "pod_id": "pod-1"}, "kura run pull"),
                             ({"state": "running", "pod_id": "pod-1", "remote_state": "completed"}, "kura run execute")):
            with self.subTest(status=status):
                code, said, stopped = self._stop(status)
                self.assertEqual(code, 1)
                self.assertFalse(stopped)
                self.assertIn(hint, said)
                self.assertIn("--yes", said)
                code, _, stopped = self._stop(status, yes=True)
                self.assertEqual(code, 0)
                self.assertTrue(stopped)

    def test_nothing_to_lose_needs_no_confirmation(self) -> None:
        for status, executor in (({"state": "running", "pod_id": "pod-1", "downloaded_run": "downloads/x"}, "runpod"),
                                 ({"state": "running"}, "docker")):
            with self.subTest(executor=executor):
                code, said, _ = self._stop(status, executor=executor)
                self.assertEqual(code, 0, said)


if __name__ == "__main__":
    unittest.main()
