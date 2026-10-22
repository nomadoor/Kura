from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from unittest.mock import patch

import yaml

from kura.run_commands import launch


@contextmanager
def _runpod_run(status: dict):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "workspace.yaml").write_text("schema_version: 2\nrunpod: {}\n", encoding="utf-8")
        run_dir = root / "runs" / "example"
        (run_dir / "resolved").mkdir(parents=True)
        (run_dir / "resolved" / "manifest.lock.yaml").write_text(yaml.safe_dump({"id": "example", "executor": {"name": "runpod"}, "compute": {"executor": "runpod"}}), encoding="utf-8")
        (run_dir / "status.json").write_text(json.dumps(status), encoding="utf-8")
        (run_dir / "realizations").mkdir()
        if status.get("remote_job_started_at"):
            (run_dir / "realizations" / "r1.phases.jsonl").write_text(json.dumps({"phase": "remote_job_started", "at": "t"}) + "\n", encoding="utf-8")
        previous = Path.cwd()
        os.chdir(root)
        try:
            yield run_dir
        finally:
            os.chdir(previous)


class ExecuteReattachTests(unittest.TestCase):
    def test_a_remote_job_record_counts_as_started_without_a_phase(self) -> None:
        from kura.executors.common import remote_job_started

        with _runpod_run({"state": "running", "pod_id": "pod-1", "last_realization": "realizations/r1.json"}) as run_dir:
            self.assertFalse(remote_job_started(run_dir, "r1"))
            (run_dir / "realizations" / "r1.remote-job.json").write_text(json.dumps({"kind": "remote_job", "schema_version": 1, "pid": "42"}), encoding="utf-8")
            self.assertTrue(remote_job_started(run_dir, "r1"))

    def test_the_runner_follower_reattaches_without_deciding_again(self) -> None:
        # The runner's `_continue_runpod` decided to follow; the controller it calls trusts that decision.
        with _runpod_run({"state": "running", "pod_id": "pod-1", "remote_job_started_at": "t", "last_realization": "realizations/r1.json"}):
            with patch("kura.executors.runpod.reconcile_runpod", side_effect=AssertionError("decided twice")), \
                    patch.object(launch, "follow_running_runpod_job", return_value=0) as follow, \
                    patch.object(launch, "download_with_retries", return_value=0), patch.object(launch, "stop_runpod", return_value={}), \
                    patch.object(launch, "format_run_completion", return_value="done"), patch.object(launch, "_notify"), \
                    redirect_stderr(io.StringIO()), patch("sys.stdout", io.StringIO()):
                code = launch._run_remote_locked("example", upload_timeout=1, job_timeout=0, download_attempts=1, download_interval=0,
                                                 reattach=True, controlled_by={"request": "r.launch.json"})
        self.assertEqual(code, 0)
        follow.assert_called_once()

    def test_a_compiled_run_launches_through_the_runner(self) -> None:
        with _runpod_run({"state": "compiled"}):
            with patch.object(launch, "_launch_runpod_through_runner", return_value=0) as through_runner:
                launch.execute_run("example", yes=True)
        through_runner.assert_called_once()

    def test_a_running_run_is_followed_through_the_runner_too(self) -> None:
        # Every RunPod training run goes through the runner; runs started before it existed are gone at their lease.
        with _runpod_run({"state": "running", "pod_id": "pod-1", "last_realization": "realizations/r1.json"}):
            with patch.object(launch, "_launch_runpod_through_runner", return_value=0) as through_runner:
                launch.execute_run("example", yes=True)
        through_runner.assert_called_once()


if __name__ == "__main__":
    unittest.main()
