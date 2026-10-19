"""The follower stops when the remote job died without an exit record, or when RunPod confirms the Pod is gone."""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from kura.run_commands import runpod_ssh

DETAILS = {"ip": "127.0.0.1", "port": 22, "key": "/tmp/key"}
RUNNING = {"state": "running", "host": "runpod", "pod_id": "pod-1", "last_realization": "realizations/r1.json"}


@contextmanager
def _run():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workspace.yaml").write_text("schema_version: 2\n", encoding="utf-8")
        run_dir = root / "runs" / "example"
        (run_dir / "realizations").mkdir(parents=True)
        (run_dir / "logs").mkdir()
        (run_dir / "realizations" / "r1.json").write_text(json.dumps({"id": "r1", "executor": "runpod", "pod": {"id": "pod-1"}}), encoding="utf-8")
        (run_dir / "realizations" / "r1.remote-job.json").write_text(json.dumps(
            {"realization_id": "r1", "pid": "42", "job_path": "/tmp/kura-jobs/example.sh", "pid_path": "/tmp/kura-jobs/example.r1.pid"}), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps(RUNNING), encoding="utf-8")
        previous = Path.cwd()
        os.chdir(root)
        try:
            with patch.object(runpod_ssh, "_sync_runpod_remote_stdout"), patch.object(runpod_ssh, "_try_sync_runpod_checkpoints") as checkpoints, \
                    patch.object(runpod_ssh, "_warn_if_lease_short"), patch.object(runpod_ssh, "sleep_checking_stop"):
                yield run_dir, checkpoints
        finally:
            os.chdir(previous)


def _follow(run_dir: Path) -> int:
    return runpod_ssh._follow_runpod_job(run_dir, DETAILS, workspace="/workspace", run_id="example", realization_id="r1", job_timeout_sec=None)


def _clock():
    # Every loop iteration is a new exit check.
    ticks = iter(range(0, 100000, 5))
    return patch.object(runpod_ssh.time, "monotonic", side_effect=lambda: float(next(ticks)))


class FollowerGoneTests(unittest.TestCase):
    @unittest.skipUnless(Path("/proc/self/cmdline").exists() and shutil.which("sh"), "a Pod runs the check under Linux sh with /proc")
    def test_the_generated_state_check_runs_under_sh_against_a_live_job(self) -> None:
        import subprocess

        with tempfile.TemporaryDirectory() as directory:
            job_path = Path(directory) / "example.sh"
            job_path.write_text("sleep 30\n", encoding="utf-8")
            job = subprocess.Popen(["sh", str(job_path)])
            real_run = subprocess.run
            try:
                def state(pid: int) -> str:
                    captured: dict[str, str] = {}

                    def run(command, **kwargs):
                        captured["script"] = command[-1]
                        return real_run(["sh", "-c", command[-1]], **kwargs)

                    with patch.object(runpod_ssh, "_ssh_base", return_value=["ssh"]), patch.object(runpod_ssh.subprocess, "run", side_effect=run):
                        answer = runpod_ssh._read_remote_job_state(DETAILS, workspace=directory, run_id="example",
                                                                    job={"pid": str(pid), "job_path": str(job_path)})
                    self.assertNotIn("\x00", captured["script"])
                    return answer

                self.assertEqual(state(job.pid), "running")
                self.assertEqual(state(os.getpid()), "gone")  # a live pid that is not the job's shell
            finally:
                job.kill()
                job.wait()
            self.assertEqual(state(job.pid), "gone")


    def test_the_state_check_reads_the_exit_record_then_the_jobs_own_pid(self) -> None:
        import subprocess

        job = {"pid": "42", "job_path": "/tmp/kura-jobs/example.sh"}
        for stdout, code, expected in (('{"exit_code": 3}', 0, {"exit_code": 3}), ("__KURA_JOB_RUNNING__\n", 0, "running"),
                                       ("__KURA_JOB_GONE__\n", 0, "gone"), ("", 255, None)):
            with self.subTest(stdout=stdout), patch.object(runpod_ssh.subprocess, "run", return_value=subprocess.CompletedProcess(["ssh"], code, stdout, "")) as run:
                self.assertEqual(runpod_ssh._read_remote_job_state(DETAILS, workspace="/workspace", run_id="example", job=job), expected)
            script = run.call_args.args[0][-1]
            self.assertIn('"/proc/$pid/cmdline"', script)
            self.assertIn("/tmp/kura-jobs/example.sh", script)


    def test_a_job_that_died_without_an_exit_record_ends_the_follow(self) -> None:
        with _run() as (run_dir, checkpoints), _clock(), \
                patch.object(runpod_ssh, "_read_remote_job_state", side_effect=["running", "gone", "gone"]):
            with self.assertRaises(runpod_ssh.RemoteJobGone):
                _follow(run_dir)
            checkpoints.assert_called()  # what the Pod still holds is copied before the follow ends

    def test_a_job_whose_last_checkpoints_cannot_be_copied_keeps_its_pod(self) -> None:
        with _run() as (run_dir, checkpoints), _clock(), \
                patch.object(runpod_ssh, "_read_remote_job_state", side_effect=["gone", "gone"]):
            checkpoints.return_value = False
            with self.assertRaises(ValueError) as raised:
                _follow(run_dir)
            self.assertNotIsInstance(raised.exception, runpod_ssh.RemoteJobGone)

    def test_one_gone_answer_is_checked_again(self) -> None:
        # The job may write its exit record just after the check looked for it.
        with _run() as (run_dir, _), _clock(), \
                patch.object(runpod_ssh, "_read_remote_job_state", side_effect=["gone", {"exit_code": 0, "timestamp": "t"}]), \
                patch.object(runpod_ssh, "_record_remote_exit_observation"):
            self.assertEqual(_follow(run_dir), 0)

    def test_an_unreachable_pod_that_runpod_confirms_gone_ends_the_follow(self) -> None:
        def gone(run_dir_arg, **_kwargs):
            status = {**json.loads((run_dir_arg / "status.json").read_text(encoding="utf-8")), "state": "interrupted", "pod_missing_at": "t"}
            (run_dir_arg / "status.json").write_text(json.dumps(status), encoding="utf-8")
            return status

        with _run() as (run_dir, _), _clock(), \
                patch.object(runpod_ssh, "_read_remote_job_state", return_value=None), \
                patch("kura.executors.runpod.reconcile_runpod", side_effect=gone) as reconcile:
            with self.assertRaises(runpod_ssh.PodGone):
                _follow(run_dir)
            reconcile.assert_called_once()

    def test_an_unreachable_pod_that_still_runs_is_followed_on(self) -> None:
        states = [None] * runpod_ssh.UNREACHABLE_CHECKS_BEFORE_RECONCILE + [{"exit_code": 0, "timestamp": "t"}]
        with _run() as (run_dir, _), _clock(), \
                patch.object(runpod_ssh, "_read_remote_job_state", side_effect=states), \
                patch.object(runpod_ssh, "_record_remote_exit_observation"), \
                patch("kura.executors.runpod.reconcile_runpod", side_effect=lambda run_dir_arg, **_k: json.loads((run_dir_arg / "status.json").read_text(encoding="utf-8"))):
            self.assertEqual(_follow(run_dir), 0)

    def test_the_controller_ends_the_run_unknown_and_deletes_the_pod_when_the_job_died(self) -> None:
        from tests.test_cli import _run_remote_in_process

        with _run() as (run_dir, _):
            stderr = io.StringIO()
            with patch("kura.run_commands.launch.stage_run", return_value=0), \
                    patch("kura.run_commands.launch.launch_run", return_value=0), \
                    patch("kura.run_commands.launch._runpod_run_over_ssh", side_effect=runpod_ssh.RemoteJobGone("the remote job ended without an exit record")), \
                    patch("kura.run_commands.launch._notify"), \
                    patch("kura.run_commands.launch.stop_runpod", return_value={}) as stop, \
                    patch("sys.stderr", stderr), patch("sys.stdout", io.StringIO()):
                code = _run_remote_in_process(argparse.Namespace(run_id="example", upload_timeout=1, job_timeout=1, download_attempts=1,
                                                                 download_interval=1, max_lease="3h", yes=True))
            self.assertEqual(code, 1)
            stop.assert_called_once()
            self.assertEqual(json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["state"], "unknown")


if __name__ == "__main__":
    unittest.main()
