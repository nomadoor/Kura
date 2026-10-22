"""A follower that cannot reach its Pod gives up, and the runner ends the run once RunPod says the Pod is gone.

The follower only counts unreachable exit checks and raises a plain error; the
runner's next attempt asks RunPod through `reconcile_runpod` (the one rule for
"the Pod is gone") and ends the run `interrupted` without creating a Pod.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from kura.executors.runpod import RunPodAPIError
from kura.run_commands import runpod_ssh

DETAILS = {"pod_id": "pod-1", "ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}
RUNNING = {"state": "running", "host": "runpod", "pod_id": "pod-1", "last_realization": "realizations/r1.json"}
LIVE_POD = {"id": "pod-1", "desiredStatus": "RUNNING"}
UNREACHABLE = subprocess.CompletedProcess(["ssh"], 255, "", "ssh: connect to host 127.0.0.1 port 22: Connection refused")
NO_EXIT_YET = subprocess.CompletedProcess(["ssh"], 0, "", "")


def _exit(code: int) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(["ssh"], 0, json.dumps({"exit_code": code, "timestamp": "t"}), "")


@contextmanager
def _run(status: dict = RUNNING):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
        run_dir = root / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}}), encoding="utf-8")
        (run_dir / "realizations" / "r1.remote-job.json").write_text(json.dumps({"realization_id": "r1", "pid": "42"}), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
        previous = Path.cwd()
        os.chdir(root)
        try:
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "api-secret"}, clear=False), \
                    patch.object(runpod_ssh, "_sync_runpod_remote_stdout"), patch.object(runpod_ssh, "_try_sync_runpod_checkpoints"), \
                    patch.object(runpod_ssh, "_warn_if_lease_short"), patch.object(runpod_ssh, "sleep_checking_stop"), \
                    patch.object(runpod_ssh, "_ssh_base", return_value=["ssh"]):
                yield run_dir
        finally:
            os.chdir(previous)


def _clock():
    # Every loop iteration is a new exit check.
    ticks = iter(range(0, 10_000_000, 5))
    return patch.object(runpod_ssh.time, "monotonic", side_effect=lambda: float(next(ticks)))


def _ssh(answers):
    return patch.object(runpod_ssh.subprocess, "run", side_effect=list(answers))


def _follow(run_dir: Path) -> int:
    return runpod_ssh._follow_runpod_job(run_dir, DETAILS, workspace="/workspace", run_id="example", realization_id="r1", job_timeout_sec=None)


def _status(run_dir: Path) -> dict:
    return json.loads((run_dir / "status.json").read_text(encoding="utf-8"))


class ExitCheckTests(unittest.TestCase):
    def test_an_unreachable_pod_is_not_read_as_no_exit_yet(self) -> None:
        with _ssh([NO_EXIT_YET]):
            self.assertIsNone(runpod_ssh._read_runpod_remote_exit(DETAILS, workspace="/workspace", run_id="example"))
        with _ssh([UNREACHABLE]), self.assertRaisesRegex(ValueError, "Connection refused"):
            runpod_ssh._read_runpod_remote_exit(DETAILS, workspace="/workspace", run_id="example")
        with _ssh([subprocess.TimeoutExpired(["ssh"], 30)]), self.assertRaises(ValueError):
            runpod_ssh._read_runpod_remote_exit(DETAILS, workspace="/workspace", run_id="example")


    def test_a_one_off_reconcile_still_answers_no_exit_when_the_pod_cannot_be_reached(self) -> None:
        with _run() as run_dir, patch.object(runpod_ssh, "_runpod_ssh_details", return_value=DETAILS), _ssh([UNREACHABLE]):
            self.assertFalse(runpod_ssh._try_observe_runpod_remote_exit(run_dir))


class FollowerTests(unittest.TestCase):
    def test_the_follower_gives_up_after_its_unreachable_checks_in_a_row(self) -> None:
        checks = runpod_ssh.UNREACHABLE_EXIT_CHECKS
        with _run() as run_dir, _clock(), _ssh([UNREACHABLE] * checks + [AssertionError("checked after giving up")]):
            with self.assertRaisesRegex(ValueError, "could not be reached"):
                _follow(run_dir)
            self.assertEqual(_status(run_dir)["state"], "running")  # the follower records nothing itself

    def test_a_reachable_check_starts_the_count_again(self) -> None:
        checks = runpod_ssh.UNREACHABLE_EXIT_CHECKS
        answers = [UNREACHABLE] * (checks - 1) + [NO_EXIT_YET] + [UNREACHABLE] * (checks - 1) + [_exit(0)]
        with _run() as run_dir, _clock(), _ssh(answers), patch.object(runpod_ssh, "_record_remote_exit_observation"):
            self.assertEqual(_follow(run_dir), 0)


def _api(lookups):
    """RunPod answering each Pod lookup in turn; anything else (a create) fails the test."""
    answers = iter(lookups)

    def answer(method: str, path: str, *_args, **_kwargs):
        if method != "GET":
            raise AssertionError(f"unexpected RunPod request: {method} {path}")
        value = next(answers)
        if isinstance(value, Exception):
            raise value
        return value

    return patch("kura.executors.runpod._runpod_request", side_effect=answer)


class RunnerEndsTheRunTests(unittest.TestCase):
    def test_an_unreachable_pod_that_runpod_reports_gone_ends_the_run_interrupted(self) -> None:
        from kura import runner

        with _run({"state": "compiled"}) as run_dir:
            request = runner.write_launch_request(run_dir, executor="runpod", extra={
                "options": {"max_lease": "3h", "upload_timeout": 1, "job_timeout": 0, "download_attempts": 1, "download_interval": 1}, "runpod_config": {"gpu_type_ids": ["A"]}, "billing_confirmed_at": "2026-10-19T00:00:00+00:00"})
            runner.claim_request(request, 1)
            runner._first_attempt(request)
            (run_dir / "realizations" / "r1.json").write_text(json.dumps({
                "id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}, "controlled_by": {"request": request.name}}), encoding="utf-8")
            (run_dir / "status.json").write_text(json.dumps(RUNNING), encoding="utf-8")
            workspace = run_dir.parent.parent

            # The follower reattaches (RunPod still lists the Pod), then cannot reach it and gives up.
            with _clock(), _api([LIVE_POD, LIVE_POD]), _ssh([UNREACHABLE] * runpod_ssh.UNREACHABLE_EXIT_CHECKS), \
                    patch.object(runpod_ssh, "_runpod_ssh_details", return_value=DETAILS), patch.object(runpod_ssh, "_start_ssh_master"), \
                    patch("kura.run_commands.launch._notify") as controller_notify, patch("kura.run_commands.launch.stop_runpod") as stop, \
                    patch.object(runner, "_notify_text") as runner_notify:
                self.assertEqual(runner.work(workspace, "example", request.name), 1)
            stop.assert_not_called()  # nothing was collected, so the Pod is not deleted here
            # The runner retries on its own, so nobody is asked to act on a retry.
            controller_notify.assert_not_called()
            runner_notify.assert_not_called()
            self.assertEqual(_status(run_dir)["state"], "running")
            self.assertFalse(runner._sibling(request, ".failures.json").exists())  # retried uncounted

            # The runner's next attempt asks RunPod, which says the Pod no longer exists.
            with _api([RunPodAPIError("RunPod Pod not found", status_code=404)]), \
                    patch("kura.run_commands.launch._run_remote_locked") as remote, patch.object(runner, "_notify_text") as notify:
                self.assertEqual(runner.work(workspace, "example", request.name), 0)
            remote.assert_not_called()
            notify.assert_called_once()
            status = _status(run_dir)
            self.assertEqual(status["state"], "interrupted")
            self.assertTrue(status["pod_missing_at"])
            self.assertFalse(runner.run_unfinished(run_dir))


if __name__ == "__main__":
    unittest.main()
